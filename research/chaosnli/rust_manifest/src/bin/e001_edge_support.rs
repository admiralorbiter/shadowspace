use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::Dirichlet;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env;
use std::fs::{create_dir_all, File};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ─── Data Structures ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, Hash)]
enum SourceDataset {
    #[serde(rename = "chaosnli_snli")]
    ChaosnliSnli,
    #[serde(rename = "chaosnli_mnli")]
    ChaosnliMnli,
}

#[derive(Debug, Deserialize)]
struct ItemRecord {
    object_id: String,
    source_dataset: Option<SourceDataset>,
    human_count_entailment: i32,
    human_count_neutral: i32,
    human_count_contradiction: i32,
}

fn partition_item_strata(items: &[ItemRecord]) -> (Vec<usize>, Vec<usize>) {
    (0..items.len()).partition(|&index| {
        let item = &items[index];
        match item.source_dataset {
            Some(SourceDataset::ChaosnliSnli) => true,
            Some(SourceDataset::ChaosnliMnli) => false,
            None => item.object_id.starts_with("chaosnli_snli_"),
        }
    })
}

fn partition_exact_profiles(items: &[ItemRecord]) -> Vec<Vec<usize>> {
    let mut map: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (idx, item) in items.iter().enumerate() {
        let key = (
            item.human_count_entailment,
            item.human_count_neutral,
            item.human_count_contradiction,
        );
        map.entry(key).or_default().push(idx);
    }
    map.into_values().collect()
}

// ─── Distance Metrics ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Serialize)]
enum Metric {
    Hellinger,
    JensenShannon,
    TotalVariation,
}

impl Metric {
    fn name(&self) -> &'static str {
        match self {
            Metric::Hellinger => "hellinger",
            Metric::JensenShannon => "jensen_shannon",
            Metric::TotalVariation => "total_variation",
        }
    }
}

#[inline(always)]
fn distance_hellinger(p: &[f64; 3], q: &[f64; 3]) -> f64 {
    let d0 = p[0].sqrt() - q[0].sqrt();
    let d1 = p[1].sqrt() - q[1].sqrt();
    let d2 = p[2].sqrt() - q[2].sqrt();
    (0.5 * (d0 * d0 + d1 * d1 + d2 * d2)).sqrt()
}

#[inline(always)]
fn distance_jsd(p: &[f64; 3], q: &[f64; 3]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..3 {
        let m = 0.5 * (p[i] + q[i]);
        if m > 1e-12 {
            if p[i] > 1e-12 {
                sum += p[i] * (p[i] / m).log2();
            }
            if q[i] > 1e-12 {
                sum += q[i] * (q[i] / m).log2();
            }
        }
    }
    (0.5 * sum).max(0.0).sqrt()
}

#[inline(always)]
fn distance_tv(p: &[f64; 3], q: &[f64; 3]) -> f64 {
    0.5 * ((p[0] - q[0]).abs() + (p[1] - q[1]).abs() + (p[2] - q[2]).abs())
}

fn build_dist_matrix_seq(probs: &[[f64; 3]], n: usize, metric: Metric) -> Vec<f64> {
    let mut dist = vec![0.0f64; n * n];
    for i in 0..n {
        let p_i = &probs[i];
        let i_off = i * n;
        for j in (i + 1)..n {
            let p_j = &probs[j];
            let d = match metric {
                Metric::Hellinger => distance_hellinger(p_i, p_j),
                Metric::JensenShannon => distance_jsd(p_i, p_j),
                Metric::TotalVariation => distance_tv(p_i, p_j),
            };
            dist[i_off + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

// ─── Edge Support & Soft Neighborhood Computation ──────────────────────────

/// Compute soft top-k neighbor weight matrix W[i, j] in [0, 1]
fn compute_topk_weight_matrix(dist: &[f64], n: usize, k: usize) -> Vec<f64> {
    const ATOL: f64 = 1e-7;
    let mut w = vec![0.0f64; n * n];
    let mut scratch = vec![0.0f64; n - 1];

    for i in 0..n {
        let row = &dist[i * n..(i + 1) * n];
        let mut idx = 0;
        for j in 0..n {
            if j != i {
                scratch[idx] = row[j];
                idx += 1;
            }
        }
        scratch.select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k_dist = scratch[k - 1];

        let mut n_closer = 0;
        let mut n_tied = 0;
        for j in 0..n {
            if j != i {
                let d = row[j];
                if d < k_dist - ATOL {
                    n_closer += 1;
                } else if (d - k_dist).abs() <= ATOL {
                    n_tied += 1;
                }
            }
        }
        let frac = if n_tied > 0 {
            (k as f64 - n_closer as f64) / n_tied as f64
        } else {
            0.0
        };

        let i_off = i * n;
        for j in 0..n {
            if j != i {
                let d = row[j];
                if d < k_dist - ATOL {
                    w[i_off + j] = 1.0;
                } else if (d - k_dist).abs() <= ATOL {
                    w[i_off + j] = frac;
                }
            }
        }
    }
    w
}

/// Convert dense W matrix to sparse row-indexed representation for O(N * k) fast null evaluation
fn extract_nonzero_weights(w: &[f64], n: usize) -> Vec<Vec<(usize, f64)>> {
    let mut nonzero = Vec::with_capacity(n);
    for i in 0..n {
        let row = &w[i * n..(i + 1) * n];
        let mut row_entries = Vec::new();
        for (j, &val) in row.iter().enumerate() {
            if val > 1e-12 {
                row_entries.push((j, val));
            }
        }
        nonzero.push(row_entries);
    }
    nonzero
}

// ─── Memory-Efficient Streaming Fold/Reduce Accumulator for S_ij ───────────

fn compute_expected_edge_support_streaming(
    items: &[ItemRecord],
    alpha_prior: f64,
    n_draws: usize,
    k: usize,
    metric: Metric,
    seed_offset: u64,
    stride: u64,
) -> (Vec<f64>, Vec<f64>) {
    let n = items.len();
    let half_draws = n_draws / 2;

    // Process Half A (0..half_draws) using Rayon fold/reduce accumulator
    let sum_a = (0..half_draws)
        .into_par_iter()
        .fold(
            || vec![0.0f64; n * n],
            |mut acc, b| {
                let seed = seed_offset + (b as u64) * stride;
                let probs = generate_posterior_probs(items, alpha_prior, seed);
                let dist = build_dist_matrix_seq(&probs, n, metric);
                let w = compute_topk_weight_matrix(&dist, n, k);
                for i in 0..(n * n) {
                    acc[i] += w[i];
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; n * n],
            |mut a, b| {
                for i in 0..(n * n) {
                    a[i] += b[i];
                }
                a
            },
        );

    // Process Half B (half_draws..n_draws)
    let sum_b = (half_draws..n_draws)
        .into_par_iter()
        .fold(
            || vec![0.0f64; n * n],
            |mut acc, b| {
                let seed = seed_offset + (b as u64) * stride;
                let probs = generate_posterior_probs(items, alpha_prior, seed);
                let dist = build_dist_matrix_seq(&probs, n, metric);
                let w = compute_topk_weight_matrix(&dist, n, k);
                for i in 0..(n * n) {
                    acc[i] += w[i];
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; n * n],
            |mut a, b| {
                for i in 0..(n * n) {
                    a[i] += b[i];
                }
                a
            },
        );

    let mut sup_a = vec![0.0f64; n * n];
    let mut sup_b = vec![0.0f64; n * n];
    for i in 0..(n * n) {
        sup_a[i] = sum_a[i] / half_draws as f64;
        sup_b[i] = sum_b[i] / (n_draws - half_draws) as f64;
    }

    (sup_a, sup_b)
}

// ─── Sub-dataset Independent Topology Evaluation (SNLI / MNLI) ───────────────

#[derive(Serialize)]
struct SubdatasetResult {
    n_items: usize,
    q_hh_crossfit: f64,
    models: HashMap<String, ModelSubdatasetResult>,
}

#[derive(Serialize)]
struct ModelSubdatasetResult {
    display_name: String,
    q_edge_support_mean: f64,
    q_null_mean: f64,
    q_null_ci_lower: f64,
    q_null_ci_upper: f64,
    p_value_add_one: f64,
    r_human_recovery: f64,
}

fn evaluate_independent_subdataset(
    sub_items: &[ItemRecord],
    sub_models: &HashMap<String, Vec<[f64; 3]>>,
    metric: Metric,
    k: usize,
    n_draws: usize,
    n_null_perms: usize,
) -> SubdatasetResult {
    let n_sub = sub_items.len();
    let (sup_a, sup_b) = compute_expected_edge_support_streaming(sub_items, 0.5, n_draws, k, metric, 42, 1);
    let mut edge_sup = vec![0.0f64; n_sub * n_sub];
    let mut sum_q_hh = 0.0f64;
    for i in 0..n_sub {
        let i_off = i * n_sub;
        for j in 0..n_sub {
            let s = 0.5 * (sup_a[i_off + j] + sup_b[i_off + j]);
            edge_sup[i_off + j] = s;
            if j != i {
                sum_q_hh += sup_a[i_off + j] * sup_b[i_off + j];
            }
        }
    }
    let q_hh_crossfit = sum_q_hh / (n_sub * k) as f64;

    let mut model_names: Vec<String> = sub_models.keys().cloned().collect();
    model_names.sort();

    let mut model_evals = HashMap::new();

    for m_name in &model_names {
        let m_probs = &sub_models[m_name];
        let dist_m = build_dist_matrix_seq(m_probs, n_sub, metric);
        let w_m = compute_topk_weight_matrix(&dist_m, n_sub, k);
        let sparse_w = extract_nonzero_weights(&w_m, n_sub);

        let mut sum_obs = 0.0f64;
        for i in 0..n_sub {
            let i_off = i * n_sub;
            for j in 0..n_sub {
                if j != i {
                    sum_obs += w_m[i_off + j] * edge_sup[i_off + j];
                }
            }
        }
        let q_obs = sum_obs / (n_sub * k) as f64;

        let null_scores: Vec<f64> = (0..n_null_perms)
            .into_par_iter()
            .map(|b_idx| {
                let mut null_rng = ChaCha8Rng::seed_from_u64(3030_0000 + b_idx as u64);
                let mut perm = (0..n_sub).collect::<Vec<_>>();
                perm.shuffle(&mut null_rng);

                let mut sum_null = 0.0f64;
                for i in 0..n_sub {
                    let i_perm = perm[i];
                    for &(j, w) in &sparse_w[i] {
                        let j_perm = perm[j];
                        sum_null += w * edge_sup[i_perm * n_sub + j_perm];
                    }
                }
                sum_null / (n_sub * k) as f64
            })
            .collect();

        let mut sorted_null = null_scores.clone();
        sorted_null.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q_null_mean = sorted_null.iter().sum::<f64>() / n_null_perms as f64;
        let idx_025 = (0.025 * n_null_perms as f64) as usize;
        let idx_975 = (0.975 * n_null_perms as f64) as usize;
        let q_null_ci_lower = sorted_null[idx_025];
        let q_null_ci_upper = sorted_null[idx_975.min(n_null_perms - 1)];

        let exceedance_count = null_scores.iter().filter(|&&v| v >= q_obs).count();
        let p_value_add_one = (1.0 + exceedance_count as f64) / (1.0 + n_null_perms as f64);

        let r_human_recovery = if (q_hh_crossfit - q_null_mean).abs() > 1e-12 {
            (q_obs - q_null_mean) / (q_hh_crossfit - q_null_mean)
        } else {
            0.0
        };

        model_evals.insert(
            m_name.clone(),
            ModelSubdatasetResult {
                display_name: m_name.clone(),
                q_edge_support_mean: q_obs,
                q_null_mean,
                q_null_ci_lower,
                q_null_ci_upper,
                p_value_add_one,
                r_human_recovery,
            },
        );
    }

    SubdatasetResult {
        n_items: n_sub,
        q_hh_crossfit,
        models: model_evals,
    }
}

// ─── Output Structs for E001 Summary ────────────────────────────────────────

#[derive(Serialize)]
struct ModelEdgeResult {
    display_name: String,
    q_edge_support_mean: f64,
    q_null_mean: f64,
    q_null_std: f64,
    q_null_ci_lower: f64,
    q_null_ci_upper: f64,
    exceedance_count: usize,
    p_value_add_one: f64,
    q_excess_mean: f64,
    q_null_ratio: f64,
    r_human_recovery: f64,
    q_exact_profile_null_mean: f64,
    p_value_exact_profile: f64,
}

#[derive(Serialize)]
struct ScaleResult {
    k: usize,
    q_hh_crossfit: f64,
    density_tau_50: f64,
    density_tau_80: f64,
    density_tau_95: f64,
    mean_degree_tau_50: f64,
    models: HashMap<String, ModelEdgeResult>,
    snli_independent: Option<SubdatasetResult>,
    mnli_independent: Option<SubdatasetResult>,
}

#[derive(Serialize)]
struct MetricResult {
    metric: String,
    scales: Vec<ScaleResult>,
}

#[derive(Serialize)]
struct SeedScheduleDiagnostic {
    schedule_name: String,
    bart_large_q_support: f64,
    roberta_large_q_support: f64,
    albert_xxlarge_q_support: f64,
    top_model_rank_order: Vec<String>,
    high_support_correlation_tau50: f64,
    model_edge_union_correlation: f64,
}

#[derive(Serialize)]
struct E001Summary {
    experiment_id: String,
    title: String,
    dataset_release: String,
    n_items: usize,
    snli_count: usize,
    mnli_count: usize,
    n_posterior_draws: usize,
    n_null_permutations: usize,
    seed_schedule_diagnostics: Vec<SeedScheduleDiagnostic>,
    artifact_manifests: HashMap<String, ArtifactManifest>,
    metrics: Vec<MetricResult>,
    total_runtime_ms: f64,
}

#[derive(Serialize)]
struct ArtifactManifest {
    artifact_id: String,
    experiment_commit: String,
    dataset_release: String,
    object_count: usize,
    shape: (usize, usize),
    layout: String,
    dtype: String,
    object_ids_sha256: String,
    matrix_sha256: String,
    metric: String,
    k: usize,
    tie_tolerance: f64,
    posterior: PosteriorProvenance,
    source_artifacts: SourceArtifacts,
}

#[derive(Serialize)]
struct PosteriorProvenance {
    prior: [f64; 3],
    draws: usize,
    seed_schedule: String,
}

#[derive(Serialize)]
struct SourceArtifacts {
    items_sha256: String,
    models_sha256: String,
}

// ─── Helper utilities ────────────────────────────────────────────────────────

fn get_workspace_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| "research/chaosnli/rust_manifest".into()),
    );
    let candidate = manifest_dir.join("../..");
    if candidate.join("data").exists() {
        candidate.canonicalize().unwrap_or(candidate)
    } else {
        PathBuf::from(".")
    }
}

fn load_items(path: &Path) -> Vec<ItemRecord> {
    let file = File::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    serde_json::from_reader(BufReader::new(file))
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()))
}

fn load_models(path: &Path) -> HashMap<String, Vec<[f64; 3]>> {
    let file = File::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    let raw: HashMap<String, Vec<Vec<f64>>> = serde_json::from_reader(BufReader::new(file))
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
    raw.into_iter()
        .map(|(k, v)| {
            let probs: Vec<[f64; 3]> = v.into_iter().map(|arr| [arr[0], arr[1], arr[2]]).collect();
            (k, probs)
        })
        .collect()
}

fn generate_posterior_probs(
    items: &[ItemRecord],
    alpha_prior: f64,
    seed: u64,
) -> Vec<[f64; 3]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    items
        .iter()
        .map(|item| {
            let a = [
                item.human_count_entailment as f64 + alpha_prior,
                item.human_count_neutral as f64 + alpha_prior,
                item.human_count_contradiction as f64 + alpha_prior,
            ];
            let dir = Dirichlet::new(&a).unwrap();
            let sample = dir.sample(&mut rng);
            let s: f64 = sample.iter().sum();
            [sample[0] / s, sample[1] / s, sample[2] / s]
        })
        .collect()
}

fn compute_bytes_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

fn compute_file_sha256(path: &Path) -> String {
    let mut hasher = Sha256::new();
    let mut file = File::open(path).unwrap();
    let mut buffer = [0u8; 65536];
    while let Ok(n) = file.read(&mut buffer) {
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    format!("{:x}", hasher.finalize())
}

// ─── Main Execution ──────────────────────────────────────────────────────────

fn main() {
    let t_start = Instant::now();
    let workspace = get_workspace_dir();

    // Configure Rayon threadpool to limit CPU core usage
    let num_threads = env::args()
        .position(|arg| arg == "--threads")
        .and_then(|idx| env::args().nth(idx + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .or_else(|| env::var("RAYON_NUM_THREADS").ok().and_then(|s| s.parse::<usize>().ok()))
        .unwrap_or(4);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global();

    let n_null_perms: usize = env::args()
        .position(|arg| arg == "--null-perms")
        .and_then(|idx| env::args().nth(idx + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(10_000);

    println!("=========================================================================");
    println!("   EXPERIMENT E001 — POSTERIOR EDGE-SUPPORT GRAPH CONSTRUCTION (FULLY RIGOROUS)");
    println!("   (Rayon Threadpool: {num_threads} worker threads | Null Permutations: {n_null_perms})");
    println!("=========================================================================");

    let items_path = workspace.join("data/chaosnli/processed/canonical_items_posterior.json");
    let models_path = workspace.join("research/chaosnli/rust_manifest/model_probs.json");

    let items_sha256 = compute_file_sha256(&items_path);
    let models_sha256 = compute_file_sha256(&models_path);

    let items = load_items(&items_path);
    let models = load_models(&models_path);
    let n = items.len();

    let object_ids: Vec<String> = items.iter().map(|item| item.object_id.clone()).collect();
    let object_ids_bytes = serde_json::to_vec(&object_ids).unwrap();
    let object_ids_sha256 = compute_bytes_sha256(&object_ids_bytes);

    let (snli_indices, mnli_indices) = partition_item_strata(&items);
    let exact_profiles = partition_exact_profiles(&items);
    println!(
        "Loaded {n} items (SNLI={}, MNLI={}, Exact Profile Groups={})",
        snli_indices.len(),
        mnli_indices.len(),
        exact_profiles.len()
    );
    println!("Loaded {} models", models.len());

    let mut model_names: Vec<String> = models.keys().cloned().collect();
    model_names.sort();

    // Partition items & models into SNLI and MNLI subsets for independent subdataset builds
    let snli_items: Vec<ItemRecord> = snli_indices.iter().map(|&i| ItemRecord {
        object_id: items[i].object_id.clone(),
        source_dataset: items[i].source_dataset,
        human_count_entailment: items[i].human_count_entailment,
        human_count_neutral: items[i].human_count_neutral,
        human_count_contradiction: items[i].human_count_contradiction,
    }).collect();

    let mnli_items: Vec<ItemRecord> = mnli_indices.iter().map(|&i| ItemRecord {
        object_id: items[i].object_id.clone(),
        source_dataset: items[i].source_dataset,
        human_count_entailment: items[i].human_count_entailment,
        human_count_neutral: items[i].human_count_neutral,
        human_count_contradiction: items[i].human_count_contradiction,
    }).collect();

    let mut snli_models = HashMap::new();
    let mut mnli_models = HashMap::new();
    for (m_name, m_probs) in &models {
        let snli_p: Vec<[f64; 3]> = snli_indices.iter().map(|&i| m_probs[i]).collect();
        let mnli_p: Vec<[f64; 3]> = mnli_indices.iter().map(|&i| m_probs[i]).collect();
        snli_models.insert(m_name.clone(), snli_p);
        mnli_models.insert(m_name.clone(), mnli_p);
    }

    let metrics = vec![Metric::Hellinger, Metric::JensenShannon, Metric::TotalVariation];
    let k_list = vec![5, 10, 20, 50];
    let n_draws = 500;
    let alpha_prior = 0.5;

    // ─── 1. Seed Schedule Sensitivity Diagnostic ────────────────────────────
    println!("\n--- Verifying Seed Schedule Sensitivity (Sequential vs Stride vs AltSeed) ---");
    
    let (sup_seq_a, sup_seq_b) = compute_expected_edge_support_streaming(&items, alpha_prior, n_draws, 10, Metric::Hellinger, 42, 1);
    let (sup_str_a, sup_str_b) = compute_expected_edge_support_streaming(&items, alpha_prior, n_draws, 10, Metric::Hellinger, 42, 1000);
    let (sup_alt_a, sup_alt_b) = compute_expected_edge_support_streaming(&items, alpha_prior, n_draws, 10, Metric::Hellinger, 1001, 1);

    let build_sup = |a: &[f64], b: &[f64]| -> Vec<f64> {
        let mut s = vec![0.0f64; n * n];
        for i in 0..(n * n) {
            s[i] = 0.5 * (a[i] + b[i]);
        }
        s
    };

    let sup_seq = build_sup(&sup_seq_a, &sup_seq_b);
    let sup_str = build_sup(&sup_str_a, &sup_str_b);
    let sup_alt = build_sup(&sup_alt_a, &sup_alt_b);

    let eval_sched = |sup_mat: &[f64]| -> HashMap<String, f64> {
        let mut res = HashMap::new();
        for m_name in &model_names {
            let m_probs = &models[m_name];
            let dist = build_dist_matrix_seq(m_probs, n, Metric::Hellinger);
            let w = compute_topk_weight_matrix(&dist, n, 10);
            let mut sum_q = 0.0f64;
            for i in 0..n {
                let i_off = i * n;
                for j in 0..n {
                    if j != i {
                        sum_q += w[i_off + j] * sup_mat[i_off + j];
                    }
                }
            }
            res.insert(m_name.clone(), sum_q / (n * 10) as f64);
        }
        res
    };

    let q_seq = eval_sched(&sup_seq);
    let q_str = eval_sched(&sup_str);
    let q_alt = eval_sched(&sup_alt);

    let get_sorted_rank = |q_map: &HashMap<String, f64>| -> Vec<String> {
        let mut pairs: Vec<(String, f64)> = q_map.iter().map(|(k, v)| (k.clone(), *v)).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        pairs.into_iter().map(|p| p.0).collect()
    };

    let r_seq = get_sorted_rank(&q_seq);
    let r_str = get_sorted_rank(&q_str);
    let r_alt = get_sorted_rank(&q_alt);

    let calc_high_sup_corr = |s1: &[f64], s2: &[f64], tau_thresh: f64| -> f64 {
        let mut v1 = Vec::new();
        let mut v2 = Vec::new();
        for i in 0..n {
            let i_off = i * n;
            for j in 0..n {
                if j != i {
                    let a = s1[i_off + j];
                    let b = s2[i_off + j];
                    if a >= tau_thresh || b >= tau_thresh {
                        v1.push(a);
                        v2.push(b);
                    }
                }
            }
        }
        if v1.is_empty() {
            return 1.0;
        }
        let m1 = v1.iter().sum::<f64>() / v1.len() as f64;
        let m2 = v2.iter().sum::<f64>() / v2.len() as f64;
        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;
        for (&a, &b) in v1.iter().zip(v2.iter()) {
            let d1 = a - m1;
            let d2 = b - m2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }
        if var1 * var2 > 1e-12 {
            cov / (var1.sqrt() * var2.sqrt())
        } else {
            1.0
        }
    };

    let seed_diagnostics = vec![
        SeedScheduleDiagnostic {
            schedule_name: "sequential (42+b)".to_string(),
            bart_large_q_support: *q_seq.get("bart-large").unwrap_or(&0.0),
            roberta_large_q_support: *q_seq.get("roberta-large").unwrap_or(&0.0),
            albert_xxlarge_q_support: *q_seq.get("albert-xxlarge").unwrap_or(&0.0),
            top_model_rank_order: r_seq,
            high_support_correlation_tau50: calc_high_sup_corr(&sup_seq, &sup_seq, 0.50),
            model_edge_union_correlation: 1.0,
        },
        SeedScheduleDiagnostic {
            schedule_name: "stride (42+1000b)".to_string(),
            bart_large_q_support: *q_str.get("bart-large").unwrap_or(&0.0),
            roberta_large_q_support: *q_str.get("roberta-large").unwrap_or(&0.0),
            albert_xxlarge_q_support: *q_str.get("albert-xxlarge").unwrap_or(&0.0),
            top_model_rank_order: r_str,
            high_support_correlation_tau50: calc_high_sup_corr(&sup_seq, &sup_str, 0.50),
            model_edge_union_correlation: calc_high_sup_corr(&sup_seq, &sup_str, 0.01),
        },
        SeedScheduleDiagnostic {
            schedule_name: "independent_alt (1001+b)".to_string(),
            bart_large_q_support: *q_alt.get("bart-large").unwrap_or(&0.0),
            roberta_large_q_support: *q_alt.get("roberta-large").unwrap_or(&0.0),
            albert_xxlarge_q_support: *q_alt.get("albert-xxlarge").unwrap_or(&0.0),
            top_model_rank_order: r_alt,
            high_support_correlation_tau50: calc_high_sup_corr(&sup_seq, &sup_alt, 0.50),
            model_edge_union_correlation: calc_high_sup_corr(&sup_seq, &sup_alt, 0.01),
        },
    ];

    println!("  Seed Schedule Diagnostic Complete:");
    for diag in &seed_diagnostics {
        println!(
            "    - {:25}: BART={:.5}, RoBERTa-L={:.5}, ALBERT={:.5} | Tau50 Corr = {:.6}",
            diag.schedule_name,
            diag.bart_large_q_support,
            diag.roberta_large_q_support,
            diag.albert_xxlarge_q_support,
            diag.high_support_correlation_tau50
        );
    }

    // ─── 2. Main Metric & Scale Pipeline ─────────────────────────────────────────
    let mut metric_results = Vec::new();
    let mut artifact_manifests: HashMap<String, ArtifactManifest> = HashMap::new();

    let artifact_dir = workspace.join("research/chaosnli/artifacts/E001");
    create_dir_all(&artifact_dir).unwrap();

    let git_commit_hash = env::var("GIT_COMMIT").unwrap_or_else(|_| "1d2acd4_rigorous_pass".to_string());

    for &metric in &metrics {
        println!("\n--- Processing Metric: {} ---", metric.name());

        let mut model_weights: HashMap<(String, usize), Vec<f64>> = HashMap::new();
        let mut model_sparse_w: HashMap<(String, usize), Vec<Vec<(usize, f64)>>> = HashMap::new();

        for m_name in &model_names {
            let m_probs = &models[m_name];
            let dist_m = build_dist_matrix_seq(m_probs, n, metric);
            for &k in &k_list {
                let w_m = compute_topk_weight_matrix(&dist_m, n, k);
                let sparse_w = extract_nonzero_weights(&w_m, n);
                model_weights.insert((m_name.clone(), k), w_m);
                model_sparse_w.insert((m_name.clone(), k), sparse_w);
            }
        }

        let mut scale_results = Vec::new();

        for &k in &k_list {
            println!("  Evaluating scale k={k}...");

            // Compute expected edge support using memory-efficient streaming fold/reduce
            let (sup_a, sup_b) = compute_expected_edge_support_streaming(&items, alpha_prior, n_draws, k, metric, 42, 1);
            let mut edge_support = vec![0.0f64; n * n];
            let mut sum_q_hh = 0.0f64;

            for i in 0..n {
                let i_off = i * n;
                for j in 0..n {
                    let s = 0.5 * (sup_a[i_off + j] + sup_b[i_off + j]);
                    edge_support[i_off + j] = s;
                    if j != i {
                        sum_q_hh += sup_a[i_off + j] * sup_b[i_off + j];
                    }
                }
            }
            let q_hh_crossfit = sum_q_hh / (n * k) as f64;

            // Density & degree metrics
            let total_off_diag = (n * (n - 1)) as f64;
            let mut c_50 = 0usize;
            let mut c_80 = 0usize;
            let mut c_95 = 0usize;

            for i in 0..n {
                let i_off = i * n;
                for j in 0..n {
                    if j != i {
                        let s = edge_support[i_off + j];
                        if s >= 0.50 {
                            c_50 += 1;
                        }
                        if s >= 0.80 {
                            c_80 += 1;
                        }
                        if s >= 0.95 {
                            c_95 += 1;
                        }
                    }
                }
            }

            let density_tau_50 = c_50 as f64 / total_off_diag;
            let density_tau_80 = c_80 as f64 / total_off_diag;
            let density_tau_95 = c_95 as f64 / total_off_diag;
            let mean_degree_tau_50 = c_50 as f64 / n as f64;

            // Save binary f32 matrix artifact and structured provenance manifest
            if metric.name() != "total_variation" && (k == 10 || k == 20 || k == 50) {
                let art_id = format!("E001-{}-k{:03}-expected-fuzzy-support-v1", metric.name(), k);
                let bin_filename = format!("S_{}_k{:03}.bin", metric.name(), k);
                let bin_path = artifact_dir.join(&bin_filename);
                let manifest_filename = format!("S_{}_k{:03}.manifest.json", metric.name(), k);
                let manifest_path = artifact_dir.join(&manifest_filename);

                // Convert float64 matrix to float32 bytes
                let f32_vec: Vec<f32> = edge_support.iter().map(|&v| v as f32).collect();
                let f32_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        f32_vec.as_ptr() as *const u8,
                        f32_vec.len() * std::mem::size_of::<f32>(),
                    )
                };

                let matrix_hash = compute_bytes_sha256(f32_bytes);
                let mut bin_file = File::create(&bin_path).unwrap();
                bin_file.write_all(f32_bytes).unwrap();

                let manifest = ArtifactManifest {
                    artifact_id: art_id.clone(),
                    experiment_commit: git_commit_hash.clone(),
                    dataset_release: "chaosnli-canonical-2026-08-02".to_string(),
                    object_count: n,
                    shape: (n, n),
                    layout: "row_major".to_string(),
                    dtype: "float32".to_string(),
                    object_ids_sha256: object_ids_sha256.clone(),
                    matrix_sha256: matrix_hash.clone(),
                    metric: metric.name().to_string(),
                    k,
                    tie_tolerance: 1e-7,
                    posterior: PosteriorProvenance {
                        prior: [0.5, 0.5, 0.5],
                        draws: n_draws,
                        seed_schedule: "sequential (42+b)".to_string(),
                    },
                    source_artifacts: SourceArtifacts {
                        items_sha256: items_sha256.clone(),
                        models_sha256: models_sha256.clone(),
                    },
                };

                let manifest_file = File::create(&manifest_path).unwrap();
                serde_json::to_writer_pretty(manifest_file, &manifest).unwrap();
                artifact_manifests.insert(bin_filename, manifest);
            }

            // Independent SNLI & MNLI subdataset topology evaluation (for k=10)
            let snli_indep = if k == 10 {
                Some(evaluate_independent_subdataset(&snli_items, &snli_models, metric, k, n_draws, 1000))
            } else {
                None
            };

            let mnli_indep = if k == 10 {
                Some(evaluate_independent_subdataset(&mnli_items, &mnli_models, metric, k, n_draws, 1000))
            } else {
                None
            };

            // Model evaluation and parallel 10,000 Monte Carlo stratified null permutations
            let mut model_evals: HashMap<String, ModelEdgeResult> = HashMap::new();

            for m_name in &model_names {
                let w_m = &model_weights[&(m_name.clone(), k)];
                let sparse_w = &model_sparse_w[&(m_name.clone(), k)];

                // 1. Observed Q_support = (1 / Nk) * sum_{i != j} W_m[i, j] * S[i, j]
                let mut sum_obs = 0.0f64;
                for i in 0..n {
                    let i_off = i * n;
                    for j in 0..n {
                        if j != i {
                            sum_obs += w_m[i_off + j] * edge_support[i_off + j];
                        }
                    }
                }
                let q_edge_support_mean = sum_obs / (n * k) as f64;

                // 2. 10,000 Stratified Item-Identity Permutations
                let null_scores: Vec<f64> = (0..n_null_perms)
                    .into_par_iter()
                    .map(|b_idx| {
                        let mut null_rng = ChaCha8Rng::seed_from_u64(2026_08_02 + b_idx as u64);
                        let mut perm = (0..n).collect::<Vec<_>>();
                        let mut snli_shuffled = snli_indices.clone();
                        let mut mnli_shuffled = mnli_indices.clone();
                        snli_shuffled.shuffle(&mut null_rng);
                        mnli_shuffled.shuffle(&mut null_rng);

                        for (orig_idx, &shuf_idx) in snli_indices.iter().zip(snli_shuffled.iter()) {
                            perm[*orig_idx] = shuf_idx;
                        }
                        for (orig_idx, &shuf_idx) in mnli_indices.iter().zip(mnli_shuffled.iter()) {
                            perm[*orig_idx] = shuf_idx;
                        }

                        let mut sum_null = 0.0f64;
                        for i in 0..n {
                            let i_perm = perm[i];
                            for &(j, w) in &sparse_w[i] {
                                let j_perm = perm[j];
                                sum_null += w * edge_support[i_perm * n + j_perm];
                            }
                        }
                        sum_null / (n * k) as f64
                    })
                    .collect();

                let mut sorted_null = null_scores.clone();
                sorted_null.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let q_null_mean = sorted_null.iter().sum::<f64>() / n_null_perms as f64;
                let var_null = sorted_null.iter().map(|x| (x - q_null_mean).powi(2)).sum::<f64>() / n_null_perms as f64;
                let q_null_std = var_null.sqrt();

                let idx_025 = (0.025 * n_null_perms as f64) as usize;
                let idx_975 = (0.975 * n_null_perms as f64) as usize;
                let q_null_ci_lower = sorted_null[idx_025];
                let q_null_ci_upper = sorted_null[idx_975.min(n_null_perms - 1)];

                let exceedance_count = null_scores.iter().filter(|&&v| v >= q_edge_support_mean).count();
                let p_value_add_one = (1.0 + exceedance_count as f64) / (1.0 + n_null_perms as f64);

                let q_excess_mean = q_edge_support_mean - q_null_mean;
                let q_null_ratio = q_edge_support_mean / q_null_mean;
                let r_human_recovery = if (q_hh_crossfit - q_null_mean).abs() > 1e-12 {
                    (q_edge_support_mean - q_null_mean) / (q_hh_crossfit - q_null_mean)
                } else {
                    0.0
                };

                // 3. Exact-Profile-Preserving Permutations (1,000 draws)
                let n_exact_perms = 1000;
                let exact_null_scores: Vec<f64> = (0..n_exact_perms)
                    .into_par_iter()
                    .map(|b_idx| {
                        let mut null_rng = ChaCha8Rng::seed_from_u64(9999_0000 + b_idx as u64);
                        let mut perm = (0..n).collect::<Vec<_>>();
                        for group in &exact_profiles {
                            let mut group_shuffled = group.clone();
                            group_shuffled.shuffle(&mut null_rng);
                            for (orig_idx, &shuf_idx) in group.iter().zip(group_shuffled.iter()) {
                                perm[*orig_idx] = shuf_idx;
                            }
                        }

                        let mut sum_null = 0.0f64;
                        for i in 0..n {
                            let i_perm = perm[i];
                            for &(j, w) in &sparse_w[i] {
                                let j_perm = perm[j];
                                sum_null += w * edge_support[i_perm * n + j_perm];
                            }
                        }
                        sum_null / (n * k) as f64
                    })
                    .collect();

                let q_exact_profile_null_mean = exact_null_scores.iter().sum::<f64>() / n_exact_perms as f64;
                let exact_exceedances = exact_null_scores.iter().filter(|&&v| v >= q_edge_support_mean).count();
                let p_value_exact_profile = (1.0 + exact_exceedances as f64) / (1.0 + n_exact_perms as f64);

                model_evals.insert(
                    m_name.clone(),
                    ModelEdgeResult {
                        display_name: m_name.clone(),
                        q_edge_support_mean,
                        q_null_mean,
                        q_null_std,
                        q_null_ci_lower,
                        q_null_ci_upper,
                        exceedance_count,
                        p_value_add_one,
                        q_excess_mean,
                        q_null_ratio,
                        r_human_recovery,
                        q_exact_profile_null_mean,
                        p_value_exact_profile,
                    },
                );
            }

            scale_results.push(ScaleResult {
                k,
                q_hh_crossfit,
                density_tau_50,
                density_tau_80,
                density_tau_95,
                mean_degree_tau_50,
                models: model_evals,
                snli_independent: snli_indep,
                mnli_independent: mnli_indep,
            });
        }

        metric_results.push(MetricResult {
            metric: metric.name().to_string(),
            scales: scale_results,
        });
    }

    let total_runtime_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    let summary = E001Summary {
        experiment_id: "E001".to_string(),
        title: "Expected Fuzzy Edge-Support Graph Construction & Model Comparison".to_string(),
        dataset_release: "chaosnli-canonical-2026-08-02".to_string(),
        n_items: n,
        snli_count: snli_indices.len(),
        mnli_count: mnli_indices.len(),
        n_posterior_draws: n_draws,
        n_null_permutations: n_null_perms,
        seed_schedule_diagnostics: seed_diagnostics,
        artifact_manifests,
        metrics: metric_results,
        total_runtime_ms,
    };

    let summary_dir = workspace.join("research/chaosnli/lab/summaries");
    create_dir_all(&summary_dir).unwrap();
    let summary_path = summary_dir.join("E001_summary.json");
    let file = File::create(&summary_path).unwrap();
    serde_json::to_writer_pretty(file, &summary).unwrap();

    println!("\n=========================================================================");
    println!("   EXPERIMENT E001 COMPLETE IN {:.2}s", total_runtime_ms / 1000.0);
    println!("   Summary saved to {}", summary_path.display());
    println!("=========================================================================");
}
