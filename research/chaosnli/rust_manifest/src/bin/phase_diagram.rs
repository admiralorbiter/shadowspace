use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Binomial, Dirichlet, Distribution};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::ffi::OsString;
use std::fs::{File, create_dir_all};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_ALPHAS: &[f64] = &[0.1, 0.5, 1.0];
const DEFAULT_CATEGORIES: &[usize] = &[2, 3, 5, 7, 10];
const DEFAULT_VOTE_DEPTHS: &[usize] = &[3, 5, 10, 20, 30, 50, 100];
const DEFAULT_N_ITEMS: usize = 3_113;
const DEFAULT_K: usize = 10;
const DEFAULT_REPETITIONS: usize = 100;
const BASE_SEED: u64 = 20_260_802;
const ATOL: f64 = 1e-7;
const RTOL: f64 = 1e-5;

#[derive(Debug, PartialEq)]
struct Config {
    output: PathBuf,
    empirical_distances: PathBuf,
    n_items: usize,
    k: usize,
    repetitions: usize,
    vote_depths: Vec<usize>,
    threads: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            output: PathBuf::from("research/chaosnli/artifacts/phase_diagram_100reps_rust.json"),
            empirical_distances: PathBuf::from(
                "data/chaosnli/processed/distance_matrix_human_hellinger.npy",
            ),
            n_items: DEFAULT_N_ITEMS,
            k: DEFAULT_K,
            repetitions: DEFAULT_REPETITIONS,
            vote_depths: DEFAULT_VOTE_DEPTHS.to_vec(),
            threads: None,
        }
    }
}

fn parse_usize(flag: &str, value: OsString) -> Result<usize, String> {
    value
        .to_str()
        .ok_or_else(|| format!("value after {flag} must be valid UTF-8"))?
        .parse::<usize>()
        .map_err(|_| format!("value after {flag} must be a positive integer"))
        .and_then(|parsed| {
            if parsed == 0 {
                Err(format!("value after {flag} must be greater than zero"))
            } else {
                Ok(parsed)
            }
        })
}

fn next_value<I>(args: &mut I, flag: &str) -> Result<OsString, String>
where
    I: Iterator<Item = OsString>,
{
    args.next()
        .ok_or_else(|| format!("missing value after {flag}"))
}

fn parse_vote_depths(value: OsString) -> Result<Vec<usize>, String> {
    let text = value
        .to_str()
        .ok_or_else(|| "value after --vote-depths must be valid UTF-8".to_string())?;
    let depths = text
        .split(',')
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| "--vote-depths must be a comma-separated integer list".to_string())
                .and_then(|parsed| {
                    if parsed == 0 {
                        Err("--vote-depths values must be greater than zero".to_string())
                    } else {
                        Ok(parsed)
                    }
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if depths.is_empty() {
        return Err("--vote-depths cannot be empty".to_string());
    }
    Ok(depths)
}

fn parse_config<I>(args: I) -> Result<Config, String>
where
    I: IntoIterator<Item = OsString>,
{
    let mut config = Config::default();
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        let flag = argument
            .to_str()
            .ok_or_else(|| "command-line flags must be valid UTF-8".to_string())?;
        match flag {
            "--output" => config.output = PathBuf::from(next_value(&mut args, flag)?),
            "--empirical-distances" => {
                config.empirical_distances = PathBuf::from(next_value(&mut args, flag)?)
            }
            "--n-items" => config.n_items = parse_usize(flag, next_value(&mut args, flag)?)?,
            "--k" => config.k = parse_usize(flag, next_value(&mut args, flag)?)?,
            "--repetitions" => {
                config.repetitions = parse_usize(flag, next_value(&mut args, flag)?)?
            }
            "--threads" => config.threads = Some(parse_usize(flag, next_value(&mut args, flag)?)?),
            "--vote-depths" => {
                config.vote_depths = parse_vote_depths(next_value(&mut args, flag)?)?
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }
    if config.k >= config.n_items - 1 {
        return Err("k must satisfy 1 <= k < n-items - 1".to_string());
    }
    Ok(config)
}

fn print_usage() {
    println!(
        r#"Usage: cargo run --release --locked --bin phase_diagram -- [OPTIONS]

Runs only the deterministic phase-diagram recomputation. Paths are repository-relative.

Options:
  --output PATH                 Output JSON path
  --empirical-distances PATH    Square NumPy .npy Hellinger matrix
  --n-items N                   Simulated items per repetition (default: 3113)
  --k K                         Neighborhood boundary (default: 10)
  --repetitions N               Repetitions per cell (default: 100)
  --vote-depths CSV             Vote depths (default: 3,5,10,20,30,50,100)
  --threads N                   Rayon worker threads (default: logical CPUs)
  -h, --help                    Show this help"#
    );
}

#[derive(Clone, Copy)]
struct CellSpec {
    alpha: f64,
    categories: usize,
    n_votes: usize,
}

#[derive(Debug, Serialize)]
struct PhaseCell {
    alpha: f64,
    c: usize,
    n_votes: usize,
    mean_tie_pct: f64,
    sd_tie_pct: f64,
}

#[derive(Serialize)]
struct LatticeCapacity {
    n_votes: usize,
    c: usize,
    capacity: u128,
}

#[derive(Serialize)]
struct PhaseOutput {
    description: String,
    n_repetitions_per_cell: usize,
    n_items: usize,
    k: usize,
    theoretical_lattice_capacity: Vec<LatticeCapacity>,
    empirical_chaosnli_tie_pct: f64,
    phase_diagram_100reps: Vec<PhaseCell>,
    total_runtime_ms: f64,
}

fn seed_for(spec: CellSpec, repetition: usize) -> u64 {
    BASE_SEED
        + repetition as u64 * 100_000
        + (spec.alpha * 1_000.0) as u64 * 1_000
        + spec.categories as u64 * 100
        + spec.n_votes as u64
}

fn sample_profile(
    dirichlet: &Dirichlet<f64>,
    categories: usize,
    n_votes: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<u16> {
    let theta = dirichlet.sample(rng);
    let mut counts = vec![0_u16; categories];
    let mut remaining_votes = n_votes as u64;
    let mut remaining_probability = 1.0;
    for category in 0..categories - 1 {
        if remaining_votes == 0 {
            break;
        }
        let conditional = (theta[category] / remaining_probability).clamp(0.0, 1.0);
        let sampled = Binomial::new(remaining_votes, conditional)
            .expect("valid conditional multinomial probability")
            .sample(rng);
        counts[category] = sampled as u16;
        remaining_votes -= sampled;
        remaining_probability = (remaining_probability - theta[category]).max(0.0);
    }
    counts[categories - 1] = remaining_votes as u16;
    counts
}

#[inline]
fn hellinger_counts(left: &[u16], right: &[u16], n_votes: usize) -> f64 {
    let affinity: f64 = left
        .iter()
        .zip(right)
        .map(|(&a, &b)| ((a as f64) * (b as f64)).sqrt())
        .sum::<f64>()
        / n_votes as f64;
    (1.0 - affinity).max(0.0).sqrt()
}

#[inline]
fn is_close(left: f64, right: f64) -> bool {
    (left - right).abs() <= ATOL + RTOL * right.abs()
}

fn boundary_tied_from_weighted_distances(distances: &mut [(f64, usize)], k: usize) -> bool {
    distances.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));
    let mut seen = 0;
    let mut kth = None;
    let mut next = None;
    for &(distance, multiplicity) in distances.iter() {
        if multiplicity == 0 {
            continue;
        }
        let end = seen + multiplicity;
        if kth.is_none() && end >= k {
            kth = Some(distance);
        }
        if end > k {
            next = Some(distance);
            break;
        }
        seen = end;
    }
    is_close(
        kth.expect("at least k candidates"),
        next.expect("at least k+1 candidates"),
    )
}

fn boundary_tie_percentage_profiles(
    profiles: &HashMap<Vec<u16>, usize>,
    n_votes: usize,
    k: usize,
) -> f64 {
    let groups: Vec<(&Vec<u16>, usize)> = profiles
        .iter()
        .map(|(profile, &multiplicity)| (profile, multiplicity))
        .collect();
    let tied_items: usize = groups
        .iter()
        .map(|&(query, query_count)| {
            let mut distances: Vec<(f64, usize)> = groups
                .iter()
                .map(|&(candidate, candidate_count)| {
                    let multiplicity = if std::ptr::eq(query, candidate) {
                        candidate_count - 1
                    } else {
                        candidate_count
                    };
                    (hellinger_counts(query, candidate, n_votes), multiplicity)
                })
                .collect();
            if boundary_tied_from_weighted_distances(&mut distances, k) {
                query_count
            } else {
                0
            }
        })
        .sum();
    tied_items as f64 / profiles.values().sum::<usize>() as f64 * 100.0
}

fn simulate_repetition(spec: CellSpec, repetition: usize, n_items: usize, k: usize) -> f64 {
    let dirichlet = Dirichlet::new(&vec![spec.alpha; spec.categories])
        .expect("positive symmetric Dirichlet parameters");
    let mut rng = ChaCha8Rng::seed_from_u64(seed_for(spec, repetition));
    let mut profiles = HashMap::<Vec<u16>, usize>::new();
    for _ in 0..n_items {
        *profiles
            .entry(sample_profile(
                &dirichlet,
                spec.categories,
                spec.n_votes,
                &mut rng,
            ))
            .or_default() += 1;
    }
    boundary_tie_percentage_profiles(&profiles, spec.n_votes, k)
}

fn round_to(value: f64, decimals: u32) -> f64 {
    let scale = 10_f64.powi(decimals as i32);
    (value * scale).round() / scale
}

fn simulate_cell(spec: CellSpec, repetitions: usize, n_items: usize, k: usize) -> PhaseCell {
    let values: Vec<f64> = (0..repetitions)
        .map(|repetition| simulate_repetition(spec, repetition, n_items, k))
        .collect();
    let mean = values.iter().sum::<f64>() / repetitions as f64;
    let sd = if repetitions > 1 {
        let sum_squares = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>();
        (sum_squares / (repetitions - 1) as f64).sqrt()
    } else {
        0.0
    };
    PhaseCell {
        alpha: spec.alpha,
        c: spec.categories,
        n_votes: spec.n_votes,
        mean_tie_pct: round_to(mean, 1),
        sd_tie_pct: round_to(sd, 2),
    }
}

fn binomial(n: usize, k: usize) -> u128 {
    let k = k.min(n - k);
    (1..=k).fold(1_u128, |accumulator, i| {
        accumulator * (n - k + i) as u128 / i as u128
    })
}

#[derive(Deserialize)]
struct NpyShape(usize, usize);

fn parse_npy_shape(header: &str) -> Result<NpyShape, String> {
    let shape_start = header
        .find("'shape':")
        .or_else(|| header.find("\"shape\":"))
        .ok_or_else(|| "NumPy header has no shape".to_string())?;
    let tuple_start = header[shape_start..]
        .find('(')
        .map(|offset| shape_start + offset + 1)
        .ok_or_else(|| "NumPy shape is not a tuple".to_string())?;
    let tuple_end = header[tuple_start..]
        .find(')')
        .map(|offset| tuple_start + offset)
        .ok_or_else(|| "NumPy shape tuple is unterminated".to_string())?;
    let dimensions = header[tuple_start..tuple_end]
        .split(',')
        .filter_map(|part| {
            let trimmed = part.trim();
            (!trimmed.is_empty()).then_some(trimmed.parse::<usize>())
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "NumPy shape contains a non-integer dimension".to_string())?;
    if dimensions.len() != 2 || dimensions[0] != dimensions[1] {
        return Err("empirical distance matrix must be square".to_string());
    }
    Ok(NpyShape(dimensions[0], dimensions[1]))
}

fn load_npy_f32(path: &Path) -> Result<(usize, Vec<f32>), String> {
    let mut reader = BufReader::new(
        File::open(path).map_err(|error| format!("failed to open {}: {error}", path.display()))?,
    );
    let mut magic = [0_u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|error| format!("failed to read NumPy header: {error}"))?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err("empirical distances are not a NumPy .npy file".to_string());
    }
    let header_length = match magic[6] {
        1 => {
            let mut bytes = [0_u8; 2];
            reader
                .read_exact(&mut bytes)
                .map_err(|error| error.to_string())?;
            u16::from_le_bytes(bytes) as usize
        }
        2 | 3 => {
            let mut bytes = [0_u8; 4];
            reader
                .read_exact(&mut bytes)
                .map_err(|error| error.to_string())?;
            u32::from_le_bytes(bytes) as usize
        }
        version => return Err(format!("unsupported NumPy format version {version}")),
    };
    let mut header_bytes = vec![0_u8; header_length];
    reader
        .read_exact(&mut header_bytes)
        .map_err(|error| format!("failed to read NumPy metadata: {error}"))?;
    let header = std::str::from_utf8(&header_bytes)
        .map_err(|_| "NumPy header is not valid UTF-8".to_string())?;
    if !(header.contains("'<f4'") || header.contains("'|f4'") || header.contains("\"<f4\"")) {
        return Err("empirical distance matrix must use little-endian float32".to_string());
    }
    if header.contains("'fortran_order': True") {
        return Err("Fortran-order NumPy arrays are unsupported".to_string());
    }
    let NpyShape(rows, columns) = parse_npy_shape(header)?;
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|error| format!("failed to read NumPy values: {error}"))?;
    if bytes.len() != rows * columns * 4 {
        return Err("NumPy payload length does not match its shape".to_string());
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    Ok((rows, values))
}

fn boundary_tie_percentage_matrix(matrix: &[f32], n: usize, k: usize) -> f64 {
    let tied = (0..n)
        .into_par_iter()
        .filter(|&row_index| {
            let row = &matrix[row_index * n..(row_index + 1) * n];
            let mut candidates: Vec<f64> = row
                .iter()
                .enumerate()
                .filter(|&(column, _)| column != row_index)
                .map(|(_, &distance)| distance as f64)
                .collect();
            candidates.sort_unstable_by(f64::total_cmp);
            is_close(candidates[k - 1], candidates[k])
        })
        .count();
    tied as f64 / n as f64 * 100.0
}

fn create_output_file(path: &Path) -> Result<File, String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    File::create(path).map_err(|error| format!("failed to create {}: {error}", path.display()))
}

fn run(config: Config) -> Result<(), String> {
    let started = Instant::now();
    let (empirical_n, empirical_matrix) = load_npy_f32(&config.empirical_distances)?;
    if config.k >= empirical_n - 1 {
        return Err("k must satisfy 1 <= k < empirical matrix N - 1".to_string());
    }
    let empirical_tie_pct =
        boundary_tie_percentage_matrix(&empirical_matrix, empirical_n, config.k);
    drop(empirical_matrix);

    let specs: Vec<CellSpec> = DEFAULT_ALPHAS
        .iter()
        .flat_map(|&alpha| {
            DEFAULT_CATEGORIES.iter().flat_map({
                let vote_depths = &config.vote_depths;
                move |&categories| {
                    vote_depths.iter().map(move |&n_votes| CellSpec {
                        alpha,
                        categories,
                        n_votes,
                    })
                }
            })
        })
        .collect();
    eprintln!(
        "Running {} phase cells x {} repetitions with Rayon...",
        specs.len(),
        config.repetitions
    );

    let compute = || {
        specs
            .par_iter()
            .map(|&spec| simulate_cell(spec, config.repetitions, config.n_items, config.k))
            .collect::<Vec<_>>()
    };
    let cells = if let Some(threads) = config.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| format!("failed to create Rayon pool: {error}"))?
            .install(compute)
    } else {
        compute()
    };

    let capacities = config
        .vote_depths
        .iter()
        .flat_map(|&n_votes| {
            DEFAULT_CATEGORIES.iter().map(move |&c| LatticeCapacity {
                n_votes,
                c,
                capacity: binomial(n_votes + c - 1, c - 1),
            })
        })
        .collect();
    let output = PhaseOutput {
        description: format!(
            "Boundary tie prevalence at k={} for N={} categorical-vote items; {} deterministic repetitions per Dirichlet regime cell (Rust/ChaCha8).",
            config.k, config.n_items, config.repetitions
        ),
        n_repetitions_per_cell: config.repetitions,
        n_items: config.n_items,
        k: config.k,
        theoretical_lattice_capacity: capacities,
        empirical_chaosnli_tie_pct: empirical_tie_pct,
        phase_diagram_100reps: cells,
        total_runtime_ms: started.elapsed().as_secs_f64() * 1_000.0,
    };
    let file = create_output_file(&config.output)?;
    serde_json::to_writer_pretty(file, &output)
        .map_err(|error| format!("failed to write JSON: {error}"))?;
    println!("Wrote {}", config.output.display());
    Ok(())
}

fn main() {
    let raw_args: Vec<OsString> = env::args_os().skip(1).collect();
    if raw_args
        .iter()
        .any(|argument| matches!(argument.to_str(), Some("-h" | "--help")))
    {
        print_usage();
        return;
    }
    let config = parse_config(raw_args).unwrap_or_else(|error| {
        eprintln!("error: {error}");
        eprintln!("Run with --help for usage.");
        std::process::exit(2);
    });
    if let Err(error) = run(config) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_tie_requires_tie_to_cross_k() {
        let mut inside_only = vec![(0.0, 2), (0.2, 1), (0.4, 1)];
        assert!(!boundary_tied_from_weighted_distances(&mut inside_only, 2));

        let mut crossing = vec![(0.0, 1), (0.2, 2), (0.4, 1)];
        assert!(boundary_tied_from_weighted_distances(&mut crossing, 2));
    }

    #[test]
    fn deterministic_repetition_uses_stable_seed() {
        let spec = CellSpec {
            alpha: 0.5,
            categories: 3,
            n_votes: 5,
        };
        let first = simulate_repetition(spec, 7, 40, 3);
        let second = simulate_repetition(spec, 7, 40, 3);
        assert_eq!(first, second);
    }

    #[test]
    fn config_defaults_match_canonical_grid() {
        let config = parse_config(Vec::<OsString>::new()).unwrap();
        assert_eq!(config.n_items, 3_113);
        assert_eq!(config.k, 10);
        assert_eq!(config.repetitions, 100);
        assert_eq!(config.vote_depths, vec![3, 5, 10, 20, 30, 50, 100]);
    }

    #[test]
    fn config_parses_phase_only_overrides() {
        let config = parse_config(
            [
                "--output",
                "out.json",
                "--empirical-distances",
                "human.npy",
                "--n-items",
                "40",
                "--k",
                "3",
                "--repetitions",
                "2",
                "--vote-depths",
                "3,75",
                "--threads",
                "2",
            ]
            .map(OsString::from),
        )
        .unwrap();
        assert_eq!(config.output, PathBuf::from("out.json"));
        assert_eq!(config.empirical_distances, PathBuf::from("human.npy"));
        assert_eq!(config.n_items, 40);
        assert_eq!(config.k, 3);
        assert_eq!(config.repetitions, 2);
        assert_eq!(config.vote_depths, vec![3, 75]);
        assert_eq!(config.threads, Some(2));
    }
}
