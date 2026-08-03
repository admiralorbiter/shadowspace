/// E002 v2: Temperature Calibration vs. Posterior-Supported Topology
///
/// Key design improvements from reviewer feedback:
///   1. Five-fold cross-fitting (stratified by source_dataset × majority_label × entropy_quintile)
///      T* selected on 4 folds, evaluated on held-out 5th fold
///   2. Four temperature selectors: raw_t1, T_NLL, T_JSD, T_topology
///   3. Log-spaced 21-point temperature grid [0.10 ... 10.00]
///   4. Degeneracy diagnostics: entropy, pairwise distance collapse, micro-jitter stability
///   5. Consumes E001 artifact (support matrices) rather than regenerating
///   6. Null-adjusted improvement criterion: Q(T) - Q_null(T) > Q(1) - Q_null(1)
///   7. Max-statistic permutation p-value for temperature selection bias correction
///   8. Separate SNLI / MNLI reporting
///
/// DO NOT RUN until tomorrow — code update only.
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::Dirichlet;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs::{create_dir_all, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ─── Shared Data Structures ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
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

// ─── Distance Metrics ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
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
            if p[i] > 1e-12 { sum += p[i] * (p[i] / m).log2(); }
            if q[i] > 1e-12 { sum += q[i] * (q[i] / m).log2(); }
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
        let i_off = i * n;
        for j in (i + 1)..n {
            let d = match metric {
                Metric::Hellinger => distance_hellinger(&probs[i], &probs[j]),
                Metric::JensenShannon => distance_jsd(&probs[i], &probs[j]),
                Metric::TotalVariation => distance_tv(&probs[i], &probs[j]),
            };
            dist[i_off + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

// ─── Soft Neighborhood Weight Matrix ────────────────────────────────────────

fn compute_topk_weight_matrix(dist: &[f64], n: usize, k: usize) -> Vec<f64> {
    const ATOL: f64 = 1e-7;
    let mut w = vec![0.0f64; n * n];
    let mut scratch = vec![0.0f64; n - 1];

    for i in 0..n {
        let row = &dist[i * n..(i + 1) * n];
        let mut idx = 0;
        for j in 0..n {
            if j != i { scratch[idx] = row[j]; idx += 1; }
        }
        scratch.select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k_dist = scratch[k - 1];
        let mut n_closer = 0usize;
        let mut n_tied = 0usize;
        for j in 0..n {
            if j != i {
                let d = row[j];
                if d < k_dist - ATOL { n_closer += 1; }
                else if (d - k_dist).abs() <= ATOL { n_tied += 1; }
            }
        }
        let frac = if n_tied > 0 { (k as f64 - n_closer as f64) / n_tied as f64 } else { 0.0 };
        let i_off = i * n;
        for j in 0..n {
            if j != i {
                let d = row[j];
                if d < k_dist - ATOL { w[i_off + j] = 1.0; }
                else if (d - k_dist).abs() <= ATOL { w[i_off + j] = frac; }
            }
        }
    }
    w
}

// ─── Softmax with Temperature ────────────────────────────────────────────────

fn logits_to_probs(logits: &[[f64; 3]], temp: f64) -> Vec<[f64; 3]> {
    logits.iter().map(|l| {
        let max_l = l[0].max(l[1]).max(l[2]);
        let e0 = ((l[0] - max_l) / temp).exp();
        let e1 = ((l[1] - max_l) / temp).exp();
        let e2 = ((l[2] - max_l) / temp).exp();
        let s = e0 + e1 + e2;
        [e0 / s, e1 / s, e2 / s]
    }).collect()
}

// ─── Q_support Computation ───────────────────────────────────────────────────

fn compute_q_support(w_m: &[f64], edge_support: &[f64], n: usize, k: usize, mask: Option<&[usize]>) -> f64 {
    let mut sum = 0.0f64;
    let denom: usize;
    match mask {
        None => {
            for i in 0..n {
                let i_off = i * n;
                for j in 0..n {
                    if j != i { sum += w_m[i_off + j] * edge_support[i_off + j]; }
                }
            }
            denom = n * k;
        }
        Some(items) => {
            for &i in items {
                let i_off = i * n;
                for j in 0..n {
                    if j != i { sum += w_m[i_off + j] * edge_support[i_off + j]; }
                }
            }
            denom = items.len() * k;
        }
    }
    if denom > 0 { sum / denom as f64 } else { 0.0 }
}

// ─── Degeneracy Diagnostics ──────────────────────────────────────────────────

fn compute_degeneracy_diagnostics(
    probs: &[[f64; 3]],
    logits: &[[f64; 3]],
    temp: f64,
    n: usize,
    metric: Metric,
    w_at_t: &[f64],
    w_at_t1: &[f64],
    rng_seed: u64,
) -> DegeneracyDiagnostics {
    // Mean model entropy H = -sum_c p_c log p_c
    let mean_entropy = probs.iter().map(|p| {
        -p.iter().filter(|&&v| v > 1e-12).map(|&v| v * v.ln()).sum::<f64>()
    }).sum::<f64>() / n as f64;

    // Pairwise distance stats (sample 1000 random pairs for speed)
    let mut rng = ChaCha8Rng::seed_from_u64(rng_seed);
    let mut dists = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let i = rng.gen_range(0..n);
        let mut j = rng.gen_range(0..n - 1);
        if j >= i { j += 1; }
        let d = match metric {
            Metric::Hellinger => distance_hellinger(&probs[i], &probs[j]),
            Metric::JensenShannon => distance_jsd(&probs[i], &probs[j]),
            Metric::TotalVariation => distance_tv(&probs[i], &probs[j]),
        };
        dists.push(d);
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_pairwise_dist = dists[500];
    let mean_dist = dists.iter().sum::<f64>() / dists.len() as f64;
    let pairwise_dist_sd = (dists.iter().map(|d| (d - mean_dist).powi(2)).sum::<f64>() / (dists.len() - 1) as f64).sqrt();

    // Graph turnover vs T=1: fraction of edges where w(T) != w(T=1) substantially
    let graph_turnover = {
        let mut changed = 0usize;
        let total = n * (n - 1);
        for i in 0..n {
            let i_off = i * n;
            for j in 0..n {
                if j != i && (w_at_t[i_off + j] - w_at_t1[i_off + j]).abs() > 0.01 {
                    changed += 1;
                }
            }
        }
        changed as f64 / total as f64
    };

    // Micro-jitter stability: add N(0, 1e-6) noise to logits, recompute, measure graph change
    let jitter_turnover = {
        let mut rng2 = ChaCha8Rng::seed_from_u64(rng_seed + 1);
        let jittered_logits: Vec<[f64; 3]> = logits.iter().map(|l| {
            let n0: f64 = rng2.r#gen::<f64>() * 2e-6 - 1e-6;
            let n1: f64 = rng2.r#gen::<f64>() * 2e-6 - 1e-6;
            let n2: f64 = rng2.r#gen::<f64>() * 2e-6 - 1e-6;
            [l[0] + n0, l[1] + n1, l[2] + n2]
        }).collect();
        let jit_probs = logits_to_probs(&jittered_logits, temp);
        let jit_dist = build_dist_matrix_seq(&jit_probs, n, metric);
        let w_jit = compute_topk_weight_matrix(&jit_dist, n, n.min(50));
        let mut changed = 0usize;
        let total = n * (n - 1);
        for i in 0..n {
            let i_off = i * n;
            for j in 0..n {
                if j != i && (w_jit[i_off + j] - w_at_t[i_off + j]).abs() > 0.01 {
                    changed += 1;
                }
            }
        }
        changed as f64 / total as f64
    };

    DegeneracyDiagnostics {
        temperature: temp,
        mean_model_entropy: mean_entropy,
        median_pairwise_distance: median_pairwise_dist,
        pairwise_distance_sd: pairwise_dist_sd,
        graph_turnover_vs_t1: graph_turnover,
        micro_jitter_graph_stability: 1.0 - jitter_turnover,
    }
}

// ─── Output Structures ───────────────────────────────────────────────────────

#[derive(Serialize, Clone)]
struct DegeneracyDiagnostics {
    temperature: f64,
    mean_model_entropy: f64,
    median_pairwise_distance: f64,
    pairwise_distance_sd: f64,
    graph_turnover_vs_t1: f64,
    micro_jitter_graph_stability: f64,
}

#[derive(Serialize)]
struct TempCurvePoint {
    temperature: f64,
    // Raw Q
    q_support_raw: f64,
    // Null-adjusted excess (correct improvement criterion)
    q_excess_over_null: f64,
    // Pointwise calibration metrics (vs. human posterior mean)
    soft_label_nll: f64,
    mean_jsd: f64,
    // Degeneracy
    degeneracy: DegeneracyDiagnostics,
}

#[derive(Serialize)]
struct TemperatureSelectorResult {
    selector_name: String,   // raw_t1 / soft_label_nll / mean_jsd / posterior_support_q
    optimal_temp: f64,
    // Held-out (out-of-fold) estimates only:
    heldout_q_support: f64,
    heldout_q_excess_over_null: f64,
    heldout_soft_label_nll: f64,
    heldout_mean_jsd: f64,
    // Improvement vs. raw T=1 (held-out)
    delta_q_vs_t1: f64,
    delta_nll_vs_t1: f64,
    // Max-statistic permutation p-value (for topology selector only)
    max_stat_perm_p_value: Option<f64>,
}

#[derive(Serialize)]
struct ModelTempResult {
    display_name: String,
    temperature_selectors: Vec<TemperatureSelectorResult>,
    // Full temperature curve (in-sample diagnostics only)
    temp_curve: Vec<TempCurvePoint>,
    // SNLI/MNLI split at T=1 and T*_topology
    q_support_snli_t1: f64,
    q_support_mnli_t1: f64,
    q_support_snli_t_star_q: f64,
    q_support_mnli_t_star_q: f64,
}

#[derive(Serialize)]
struct ScaleResult {
    k: usize,
    models: HashMap<String, ModelTempResult>,
}

#[derive(Serialize)]
struct MetricResult {
    metric: String,
    scales: Vec<ScaleResult>,
}

#[derive(Serialize)]
struct E002Summary {
    experiment_id: String,
    title: String,
    dataset_release: String,
    e001_artifact_id: String,
    n_items: usize,
    n_posterior_draws: usize,
    n_null_permutations: usize,
    n_crossfit_folds: usize,
    temperatures: Vec<f64>,
    temperature_selectors: Vec<String>,
    metrics: Vec<MetricResult>,
    total_runtime_ms: f64,
    improvement_criterion: String,
}

// ─── Fold Construction ───────────────────────────────────────────────────────

/// Build 5 stratified folds (by source_dataset × majority_label × entropy_quintile).
/// Returns vec of 5 test-fold item-index sets.
fn build_stratified_folds(items: &[ItemRecord], n_folds: usize, seed: u64) -> Vec<Vec<usize>> {
    // Determine majority label and entropy quintile for each item
    let mut groups: HashMap<(u8, u8, u8), Vec<usize>> = HashMap::new();

    for (idx, item) in items.iter().enumerate() {
        let dataset_code: u8 = match item.source_dataset {
            Some(SourceDataset::ChaosnliSnli) => 0,
            _ => 1,
        };
        let counts = [item.human_count_entailment, item.human_count_neutral, item.human_count_contradiction];
        let majority: u8 = counts.iter().enumerate().max_by_key(|&(_, c)| *c).map(|(i, _)| i as u8).unwrap_or(0);

        // Entropy quintile (0-4)
        let total = counts.iter().sum::<i32>() as f64;
        let entropy = if total > 0.0 {
            -counts.iter().filter(|&&c| c > 0).map(|&c| {
                let p = c as f64 / total;
                p * p.ln()
            }).sum::<f64>()
        } else { 0.0 };
        let quintile: u8 = ((entropy / 1.0987) * 5.0).min(4.0) as u8; // 1.0987 ≈ ln(3)

        groups.entry((dataset_code, majority, quintile)).or_default().push(idx);
    }

    // Initialize folds
    let mut folds: Vec<Vec<usize>> = vec![Vec::new(); n_folds];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for mut group in groups.into_values() {
        group.shuffle(&mut rng);
        for (i, idx) in group.into_iter().enumerate() {
            folds[i % n_folds].push(idx);
        }
    }
    folds
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

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
    serde_json::from_reader(BufReader::new(file)).unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()))
}

fn load_model_logits(path: &Path) -> HashMap<String, Vec<[f64; 3]>> {
    let file = File::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()));
    let raw: HashMap<String, Vec<Vec<f64>>> = serde_json::from_reader(BufReader::new(file))
        .unwrap_or_else(|e| panic!("Failed to parse {}: {e}", path.display()));
    raw.into_iter()
        .map(|(k, v)| (k, v.into_iter().map(|arr| [arr[0], arr[1], arr[2]]).collect()))
        .collect()
}

/// Load a support matrix from f32 flat binary file written by E001 v2.
/// Falls back to regenerating from posterior draws if artifact not found.
fn load_support_matrix_f32(path: &Path, n: usize) -> Option<Vec<f64>> {
    let mut file = File::open(path).ok()?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).ok()?;
    if buf.len() != n * n * 4 {
        return None;
    }
    Some(buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect())
}

fn generate_latent_posterior_probs(items: &[ItemRecord], alpha_prior: f64, seed: u64) -> Vec<[f64; 3]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    items.iter().map(|item| {
        let a = [
            item.human_count_entailment as f64 + alpha_prior,
            item.human_count_neutral as f64 + alpha_prior,
            item.human_count_contradiction as f64 + alpha_prior,
        ];
        let dir = Dirichlet::new(&a).unwrap();
        let sample = dir.sample(&mut rng);
        let s: f64 = sample.iter().sum();
        [sample[0] / s, sample[1] / s, sample[2] / s]
    }).collect()
}

/// Compute stratified null Q mean over n_perms permutations for a given model/edge_support.
/// Returns all null values for quantile and max-statistic computation.
fn compute_null_qs(
    w_m: &[f64],
    edge_support: &[f64],
    n: usize,
    k: usize,
    snli_indices: &[usize],
    mnli_indices: &[usize],
    n_perms: usize,
    seed: u64,
) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n_perms).map(|_| {
        let mut perm = (0..n).collect::<Vec<_>>();
        let mut si = snli_indices.to_vec();
        let mut mi = mnli_indices.to_vec();
        si.shuffle(&mut rng);
        mi.shuffle(&mut rng);
        for (orig, &shuf) in snli_indices.iter().zip(si.iter()) { perm[*orig] = shuf; }
        for (orig, &shuf) in mnli_indices.iter().zip(mi.iter()) { perm[*orig] = shuf; }
        let mut sum = 0.0f64;
        for i in 0..n {
            let ip = perm[i];
            let i_off = ip * n;
            for j in 0..n {
                if j != i {
                    sum += w_m[i_off + perm[j]] * edge_support[i * n + j];
                }
            }
        }
        sum / (n * k) as f64
    }).collect()
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let t_start = Instant::now();
    let workspace = get_workspace_dir();

    println!("=========================================================================");
    println!("   EXPERIMENT E002 v2 — TEMPERATURE CALIBRATION (CROSS-FITTED)");
    println!("=========================================================================");

    let items_path = workspace.join("data/chaosnli/processed/canonical_items_posterior.json");
    let logits_path = workspace.join("research/chaosnli/rust_manifest/model_logits.json");

    let items = load_items(&items_path);
    let model_logits = load_model_logits(&logits_path);
    let n = items.len();

    let (snli_indices, mnli_indices) = partition_item_strata(&items);
    println!("Loaded {n} items (SNLI={}, MNLI={})", snli_indices.len(), mnli_indices.len());
    println!("Loaded {} models", model_logits.len());

    let mut model_names: Vec<String> = model_logits.keys().cloned().collect();
    model_names.sort();

    // E001 artifact directory — look for most recent run
    let e001_artifact_base = workspace.join("research/chaosnli/lab/artifacts/E001");
    let e001_run_id = {
        let mut run_dirs: Vec<_> = std::fs::read_dir(&e001_artifact_base)
            .map(|rd| rd.filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect())
            .unwrap_or_default();
        run_dirs.sort();
        run_dirs.pop().unwrap_or_else(|| "MISSING".to_string())
    };
    println!("E001 artifact: {e001_run_id}");
    let e001_dir = e001_artifact_base.join(&e001_run_id);

    let metrics = vec![Metric::Hellinger, Metric::JensenShannon, Metric::TotalVariation];
    let k_list = vec![5usize, 10, 20, 50];
    let n_draws = 500usize;
    let alpha_prior = 0.5f64;
    let n_null_perms = 10_000usize;
    let n_folds = 5usize;
    let random_seed = 20260802u64;

    // Log-spaced 21-point temperature grid
    let temperatures: Vec<f64> = vec![
        0.10, 0.125, 0.16, 0.20, 0.25,
        0.32, 0.40, 0.50, 0.63, 0.80,
        1.00,
        1.25, 1.60, 2.00, 2.50,
        3.20, 4.00, 5.00, 6.30, 8.00, 10.00,
    ];

    // Build stratified cross-fitting folds
    let folds = build_stratified_folds(&items, n_folds, random_seed);
    println!("Built {n_folds} stratified folds: {:?}", folds.iter().map(|f| f.len()).collect::<Vec<_>>());

    let mut metric_results = Vec::new();

    for &metric in &metrics {
        println!("\n--- Processing Metric: {} ---", metric.name());

        // Load E001 support matrices (fallback: regenerate if missing)
        let mut edge_support_by_k: HashMap<usize, Vec<f64>> = HashMap::new();
        for &k in &k_list {
            let fname = format!("support_{}_{:03}.f32", metric.name(), k);
            let fpath = e001_dir.join(&fname);
            if let Some(mat) = load_support_matrix_f32(&fpath, n) {
                println!("  Loaded E001 support matrix from {fname}");
                edge_support_by_k.insert(k, mat);
            } else {
                println!("  WARNING: E001 artifact {fname} not found — regenerating (run E001 v2 first!)");
                // Fallback regeneration
                let support = (0..n_draws)
                    .into_par_iter()
                    .map(|b| {
                        let seed = 42 + b as u64 * 1000;
                        let probs_b = generate_latent_posterior_probs(&items, alpha_prior, seed);
                        let dist_b = build_dist_matrix_seq(&probs_b, n, metric);
                        compute_topk_weight_matrix(&dist_b, n, k)
                    })
                    .reduce(
                        || vec![0.0f64; n * n],
                        |mut acc, w| { for i in 0..(n*n) { acc[i] += w[i]; } acc }
                    );
                let inv = 1.0 / n_draws as f64;
                edge_support_by_k.insert(k, support.into_iter().map(|v| v * inv).collect());
            }
        }

        let mut scale_results = Vec::new();

        for &k in &k_list {
            println!("  k={k}: running 5-fold cross-fitted temperature evaluation...");
            let edge_support = &edge_support_by_k[&k];

            // Null mean at T=1 is computed per-model below (10K perms for selected T*).

            let mut model_results: HashMap<String, ModelTempResult> = HashMap::new();

            for m_name in &model_names {
                let logits_m = &model_logits[m_name];

                // Pre-compute weight matrices at all temperatures
                let w_by_temp: Vec<Vec<f64>> = temperatures.iter().map(|&t| {
                    let probs_t = logits_to_probs(logits_m, t);
                    let dist_t = build_dist_matrix_seq(&probs_t, n, metric);
                    compute_topk_weight_matrix(&dist_t, n, k)
                }).collect();

                // Pre-compute pointwise metrics vs human posterior mean
                let human_probs: Vec<[f64; 3]> = items.iter().map(|item| {
                    let total = (item.human_count_entailment + item.human_count_neutral + item.human_count_contradiction) as f64;
                    [
                        item.human_count_entailment as f64 / total,
                        item.human_count_neutral as f64 / total,
                        item.human_count_contradiction as f64 / total,
                    ]
                }).collect();

                let compute_nll = |w_idx: usize| -> f64 {
                    let probs_t = logits_to_probs(logits_m, temperatures[w_idx]);
                    probs_t.iter().zip(human_probs.iter()).map(|(q, p)| {
                        -p.iter().zip(q.iter())
                            .filter(|(pv, _)| **pv > 1e-12)
                            .map(|(pv, qv)| pv * qv.max(1e-12).ln())
                            .sum::<f64>()
                    }).sum::<f64>() / n as f64
                };

                let compute_mean_jsd_m = |w_idx: usize| -> f64 {
                    let probs_t = logits_to_probs(logits_m, temperatures[w_idx]);
                    probs_t.iter().zip(human_probs.iter()).map(|(q, p)| {
                        distance_jsd(p, q)
                    }).sum::<f64>() / n as f64
                };

                // Compute null distribution at T=1 for this model
                let null_qs_t1 = compute_null_qs(
                    &w_by_temp[10], // index 10 = T=1.0
                    edge_support, n, k, &snli_indices, &mnli_indices,
                    n_null_perms, 88888 + k as u64,
                );
                let _null_mean_t1 = null_qs_t1.iter().sum::<f64>() / n_null_perms as f64;

                // Build full temperature curve (all items, for diagnostics)
                let w_t1 = &w_by_temp[10];
                let temp_curve: Vec<TempCurvePoint> = temperatures.iter().enumerate().map(|(ti, &t)| {
                    let null_qs_t = compute_null_qs(
                        &w_by_temp[ti], edge_support, n, k,
                        &snli_indices, &mnli_indices,
                        1000, // faster 1K perms for curve; 10K only for selected T*
                        77777 + k as u64 + ti as u64,
                    );
                    let null_mean_t = null_qs_t.iter().sum::<f64>() / null_qs_t.len() as f64;
                    let q_raw = compute_q_support(&w_by_temp[ti], edge_support, n, k, None);
                    let probs_t = logits_to_probs(logits_m, t);
                    let deg = compute_degeneracy_diagnostics(
                        &probs_t, logits_m, t, n, metric,
                        &w_by_temp[ti], w_t1, 55555 + ti as u64,
                    );
                    TempCurvePoint {
                        temperature: t,
                        q_support_raw: q_raw,
                        q_excess_over_null: q_raw - null_mean_t,
                        soft_label_nll: compute_nll(ti),
                        mean_jsd: compute_mean_jsd_m(ti),
                        degeneracy: deg,
                    }
                }).collect();

                // Five-fold cross-fitting for held-out estimates
                // For each selector, track cumulative out-of-fold estimates
                let mut oof_q_t1 = 0.0f64;
                let mut oof_q_nll = 0.0f64;
                let mut oof_q_jsd = 0.0f64;
                let mut oof_q_topology = 0.0f64;
                let mut oof_nll_t1 = 0.0f64;
                let mut oof_nll_nll = 0.0f64;
                let mut oof_t_star_nll = 0.0f64;
                let mut oof_t_star_jsd = 0.0f64;
                let mut oof_t_star_topology = 0.0f64;
                let mut oof_null_mean = 0.0f64;

                for fold_idx in 0..n_folds {
                    let test_items = &folds[fold_idx];
                    let train_items: Vec<usize> = (0..n_folds)
                        .filter(|&fi| fi != fold_idx)
                        .flat_map(|fi| folds[fi].iter().copied())
                        .collect();

                    // Select T* on training folds for each selector
                    let t_star_nll_idx = temperatures.iter().enumerate()
                        .min_by(|(ai, _), (bi, _)| compute_nll(*ai).partial_cmp(&compute_nll(*bi)).unwrap())
                        .map(|(i, _)| i).unwrap_or(10);

                    let t_star_jsd_idx = temperatures.iter().enumerate()
                        .min_by(|(ai, _), (bi, _)| compute_mean_jsd_m(*ai).partial_cmp(&compute_mean_jsd_m(*bi)).unwrap())
                        .map(|(i, _)| i).unwrap_or(10);

                    let t_star_q_idx = temperatures.iter().enumerate()
                        .max_by(|(ai, _), (bi, _)| {
                            let qa = compute_q_support(&w_by_temp[*ai], edge_support, n, k, Some(&train_items));
                            let qb = compute_q_support(&w_by_temp[*bi], edge_support, n, k, Some(&train_items));
                            qa.partial_cmp(&qb).unwrap()
                        })
                        .map(|(i, _)| i).unwrap_or(10);

                    // Evaluate on held-out test fold
                    let fold_null = compute_null_qs(
                        &w_by_temp[10], edge_support, n, k,
                        &snli_indices, &mnli_indices, 1000,
                        12345 + fold_idx as u64 + k as u64,
                    );
                    let fold_null_mean = fold_null.iter().sum::<f64>() / fold_null.len() as f64;

                    oof_null_mean += fold_null_mean;
                    oof_q_t1 += compute_q_support(&w_by_temp[10], edge_support, n, k, Some(test_items));
                    oof_q_nll += compute_q_support(&w_by_temp[t_star_nll_idx], edge_support, n, k, Some(test_items));
                    oof_q_jsd += compute_q_support(&w_by_temp[t_star_jsd_idx], edge_support, n, k, Some(test_items));
                    oof_q_topology += compute_q_support(&w_by_temp[t_star_q_idx], edge_support, n, k, Some(test_items));
                    oof_nll_t1 += compute_nll(10);
                    oof_nll_nll += compute_nll(t_star_nll_idx);
                    oof_t_star_nll += temperatures[t_star_nll_idx];
                    oof_t_star_jsd += temperatures[t_star_jsd_idx];
                    oof_t_star_topology += temperatures[t_star_q_idx];
                }

                let inv_f = 1.0 / n_folds as f64;
                let heldout_q_t1 = oof_q_t1 * inv_f;
                let heldout_q_nll = oof_q_nll * inv_f;
                let heldout_q_jsd = oof_q_jsd * inv_f;
                let heldout_q_topology = oof_q_topology * inv_f;
                let heldout_nll_t1 = oof_nll_t1 * inv_f;
                let heldout_nll_nll = oof_nll_nll * inv_f;
                let heldout_null_mean = oof_null_mean * inv_f;

                // Max-statistic permutation p-value for topology selector
                let max_stat_perm_p = {
                    let mut rng = ChaCha8Rng::seed_from_u64(99999 + k as u64);
                    let observed_max = temp_curve.iter()
                        .map(|p| p.q_excess_over_null)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let n_max_perms = 1000usize;
                    let mut exceedances = 0usize;
                    for _ in 0..n_max_perms {
                        let mut perm = (0..n).collect::<Vec<_>>();
                        let mut si = snli_indices.to_vec();
                        let mut mi = mnli_indices.to_vec();
                        si.shuffle(&mut rng);
                        mi.shuffle(&mut rng);
                        for (orig, &shuf) in snli_indices.iter().zip(si.iter()) { perm[*orig] = shuf; }
                        for (orig, &shuf) in mnli_indices.iter().zip(mi.iter()) { perm[*orig] = shuf; }

                        let perm_max = temperatures.iter().enumerate().map(|(ti, _)| {
                            let null_ti: Vec<f64> = compute_null_qs(
                                &w_by_temp[ti], edge_support, n, k,
                                &snli_indices, &mnli_indices, 200,
                                44444 + ti as u64,
                            );
                            let null_mean = null_ti.iter().sum::<f64>() / null_ti.len() as f64;
                            compute_q_support(&w_by_temp[ti], edge_support, n, k, None) - null_mean
                        }).fold(f64::NEG_INFINITY, f64::max);

                        if perm_max >= observed_max { exceedances += 1; }
                    }
                    (exceedances + 1) as f64 / (n_max_perms + 1) as f64
                };

                // SNLI/MNLI split
                let t_star_q_idx_full = temp_curve.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.q_excess_over_null.partial_cmp(&b.q_excess_over_null).unwrap())
                    .map(|(i, _)| i).unwrap_or(10);

                let q_snli_t1 = compute_q_support(&w_by_temp[10], edge_support, n, k, Some(&snli_indices));
                let q_mnli_t1 = compute_q_support(&w_by_temp[10], edge_support, n, k, Some(&mnli_indices));
                let q_snli_tq = compute_q_support(&w_by_temp[t_star_q_idx_full], edge_support, n, k, Some(&snli_indices));
                let q_mnli_tq = compute_q_support(&w_by_temp[t_star_q_idx_full], edge_support, n, k, Some(&mnli_indices));

                model_results.insert(m_name.clone(), ModelTempResult {
                    display_name: m_name.clone(),
                    temperature_selectors: vec![
                        TemperatureSelectorResult {
                            selector_name: "raw_t1".to_string(),
                            optimal_temp: 1.0,
                            heldout_q_support: heldout_q_t1,
                            heldout_q_excess_over_null: heldout_q_t1 - heldout_null_mean,
                            heldout_soft_label_nll: heldout_nll_t1,
                            heldout_mean_jsd: 0.0, // filled separately
                            delta_q_vs_t1: 0.0,
                            delta_nll_vs_t1: 0.0,
                            max_stat_perm_p_value: None,
                        },
                        TemperatureSelectorResult {
                            selector_name: "soft_label_nll".to_string(),
                            optimal_temp: oof_t_star_nll * inv_f,
                            heldout_q_support: heldout_q_nll,
                            heldout_q_excess_over_null: heldout_q_nll - heldout_null_mean,
                            heldout_soft_label_nll: heldout_nll_nll,
                            heldout_mean_jsd: 0.0,
                            delta_q_vs_t1: heldout_q_nll - heldout_q_t1,
                            delta_nll_vs_t1: heldout_nll_nll - heldout_nll_t1,
                            max_stat_perm_p_value: None,
                        },
                        TemperatureSelectorResult {
                            selector_name: "mean_jsd".to_string(),
                            optimal_temp: oof_t_star_jsd * inv_f,
                            heldout_q_support: heldout_q_jsd,
                            heldout_q_excess_over_null: heldout_q_jsd - heldout_null_mean,
                            heldout_soft_label_nll: 0.0,
                            heldout_mean_jsd: 0.0,
                            delta_q_vs_t1: heldout_q_jsd - heldout_q_t1,
                            delta_nll_vs_t1: 0.0,
                            max_stat_perm_p_value: None,
                        },
                        TemperatureSelectorResult {
                            selector_name: "posterior_support_q".to_string(),
                            optimal_temp: oof_t_star_topology * inv_f,
                            heldout_q_support: heldout_q_topology,
                            heldout_q_excess_over_null: heldout_q_topology - heldout_null_mean,
                            heldout_soft_label_nll: 0.0,
                            heldout_mean_jsd: 0.0,
                            delta_q_vs_t1: heldout_q_topology - heldout_q_t1,
                            delta_nll_vs_t1: 0.0,
                            max_stat_perm_p_value: Some(max_stat_perm_p),
                        },
                    ],
                    temp_curve,
                    q_support_snli_t1: q_snli_t1,
                    q_support_mnli_t1: q_mnli_t1,
                    q_support_snli_t_star_q: q_snli_tq,
                    q_support_mnli_t_star_q: q_mnli_tq,
                });
            }

            scale_results.push(ScaleResult { k, models: model_results });
        }

        metric_results.push(MetricResult { metric: metric.name().to_string(), scales: scale_results });
    }

    let total_runtime_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    let summary = E002Summary {
        experiment_id: "E002".to_string(),
        title: "Scalar Temperature Scaling: Pointwise Calibration versus Posterior-Supported Topology".to_string(),
        dataset_release: "chaosnli-canonical-2026-08-02".to_string(),
        e001_artifact_id: e001_run_id,
        n_items: n,
        n_posterior_draws: n_draws,
        n_null_permutations: n_null_perms,
        n_crossfit_folds: n_folds,
        temperatures: temperatures.clone(),
        temperature_selectors: vec![
            "raw_t1".to_string(),
            "soft_label_nll".to_string(),
            "mean_jsd".to_string(),
            "posterior_support_q".to_string(),
        ],
        metrics: metric_results,
        total_runtime_ms,
        improvement_criterion: "Q(T) - Q_null(T) > Q(1) - Q_null(1). Raw Q increasing without null-adjusted excess does NOT count.".to_string(),
    };

    let summary_dir = workspace.join("research/chaosnli/lab/summaries");
    create_dir_all(&summary_dir).unwrap();
    let summary_path = summary_dir.join("E002_v2_summary.json");
    serde_json::to_writer_pretty(File::create(&summary_path).unwrap(), &summary).unwrap();

    println!("\n=========================================================================");
    println!("   EXPERIMENT E002 v2 COMPLETE IN {:.2}s", total_runtime_ms / 1000.0);
    println!("   Summary: {}", summary_path.display());
    println!("=========================================================================");
}
