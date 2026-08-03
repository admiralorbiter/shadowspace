use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::Dirichlet;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env;
use std::fs::{create_dir_all, File};
use std::io::{BufReader, Write};
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
    q_snli_mean: f64,
    q_snli_null_mean: f64,
    q_mnli_mean: f64,
    q_mnli_null_mean: f64,
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
}

#[derive(Serialize)]
struct MetricResult {
    metric: String,
    scales: Vec<ScaleResult>,
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
    seed_stability_pearson_r: f64,
    seed_stability_mse: f64,
    artifact_sha256: HashMap<String, String>,
    metrics: Vec<MetricResult>,
    total_runtime_ms: f64,
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

fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
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
    println!("   EXPERIMENT E001 — POSTERIOR EDGE-SUPPORT GRAPH CONSTRUCTION (RIGOROUS)");
    println!("   (Rayon Threadpool: {num_threads} worker threads | Null Permutations: {n_null_perms})");
    println!("=========================================================================");

    let items_path = workspace.join("data/chaosnli/processed/canonical_items_posterior.json");
    let models_path = workspace.join("research/chaosnli/rust_manifest/model_probs.json");

    let items = load_items(&items_path);
    let models = load_models(&models_path);
    let n = items.len();

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

    let metrics = vec![Metric::Hellinger, Metric::JensenShannon, Metric::TotalVariation];
    let k_list = vec![5, 10, 20, 50];
    let n_draws = 500;
    let alpha_prior = 0.5;

    // ─── Monte Carlo Seed Stability Verification ─────────────────────────────
    println!("\n--- Verifying Monte Carlo Seed Stability (Seed 42 vs Seed 1001) ---");
    let seed1_draws: Vec<Vec<[f64; 3]>> = (0..n_draws)
        .map(|b| generate_posterior_probs(&items, alpha_prior, 42 + b as u64))
        .collect();
    let seed2_draws: Vec<Vec<[f64; 3]>> = (0..n_draws)
        .map(|b| generate_posterior_probs(&items, alpha_prior, 1001 + b as u64))
        .collect();

    let mut edge_sup_1 = vec![0.0f64; n * n];
    let mut edge_sup_2 = vec![0.0f64; n * n];

    for b in 0..n_draws {
        let dist1 = build_dist_matrix_seq(&seed1_draws[b], n, Metric::Hellinger);
        let dist2 = build_dist_matrix_seq(&seed2_draws[b], n, Metric::Hellinger);
        let w1 = compute_topk_weight_matrix(&dist1, n, 10);
        let w2 = compute_topk_weight_matrix(&dist2, n, 10);
        for i in 0..(n * n) {
            edge_sup_1[i] += w1[i];
            edge_sup_2[i] += w2[i];
        }
    }
    for i in 0..(n * n) {
        edge_sup_1[i] /= n_draws as f64;
        edge_sup_2[i] /= n_draws as f64;
    }

    let mut sum_1 = 0.0;
    let mut sum_2 = 0.0;
    let mut sum_12 = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;
    let mut sum_mse = 0.0;
    let total_pairs = (n * (n - 1)) as f64;

    for i in 0..n {
        for j in 0..n {
            if j != i {
                let v1 = edge_sup_1[i * n + j];
                let v2 = edge_sup_2[i * n + j];
                sum_1 += v1;
                sum_2 += v2;
                sum_12 += v1 * v2;
                sum_sq1 += v1 * v1;
                sum_sq2 += v2 * v2;
                let diff = v1 - v2;
                sum_mse += diff * diff;
            }
        }
    }

    let mean1 = sum_1 / total_pairs;
    let mean2 = sum_2 / total_pairs;
    let cov = (sum_12 / total_pairs) - (mean1 * mean2);
    let var1 = (sum_sq1 / total_pairs) - (mean1 * mean1);
    let var2 = (sum_sq2 / total_pairs) - (mean2 * mean2);
    let seed_stability_pearson_r = cov / (var1.sqrt() * var2.sqrt());
    let seed_stability_mse = sum_mse / total_pairs;

    println!(
        "  Seed Stability (k=10 Hellinger): Pearson r = {:.6}, MSE = {:.8}",
        seed_stability_pearson_r, seed_stability_mse
    );

    // ─── Main Metric & Scale Pipeline ─────────────────────────────────────────
    let mut metric_results = Vec::new();
    let mut artifact_hashes: HashMap<String, String> = HashMap::new();

    let artifact_dir = workspace.join("research/chaosnli/artifacts/E001");
    create_dir_all(&artifact_dir).unwrap();

    for &metric in &metrics {
        println!("\n--- Processing Metric: {} ---", metric.name());

        // Pre-build model weight matrices and sparse representations for this metric across all k
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

        // Draw 500 posterior samples for this metric
        println!("Sampling 500 posterior-predictive draws and building edge-support graphs...");
        let posterior_draw_probs: Vec<Vec<[f64; 3]>> = (0..n_draws)
            .into_par_iter()
            .map(|b| generate_posterior_probs(&items, alpha_prior, 42 + b as u64))
            .collect();

        let mut scale_results = Vec::new();

        for &k in &k_list {
            println!("  Evaluating scale k={k}...");

            // Split 500 draws into Half A (0..250) and Half B (250..500) for cross-fitting
            let (draws_a, draws_b) = posterior_draw_probs.split_at(250);

            let compute_edge_sup = |draw_slice: &[Vec<[f64; 3]>]| -> Vec<f64> {
                let n_sub = draw_slice.len();
                let sub_matrices: Vec<Vec<f64>> = draw_slice
                    .into_par_iter()
                    .map(|p| {
                        let dist = build_dist_matrix_seq(p, n, metric);
                        compute_topk_weight_matrix(&dist, n, k)
                    })
                    .collect();

                let mut sup = vec![0.0f64; n * n];
                for mat in sub_matrices {
                    for i in 0..(n * n) {
                        sup[i] += mat[i];
                    }
                }
                for i in 0..(n * n) {
                    sup[i] /= n_sub as f64;
                }
                sup
            };

            let edge_support_a = compute_edge_sup(draws_a);
            let edge_support_b = compute_edge_sup(draws_b);

            let mut edge_support = vec![0.0f64; n * n];
            for i in 0..(n * n) {
                edge_support[i] = 0.5 * (edge_support_a[i] + edge_support_b[i]);
            }

            // Cross-fitted human positive control Q_HH = (1 / Nk) * sum_{i != j} S_A[i,j] * S_B[i,j]
            let mut sum_q_hh = 0.0f64;
            for i in 0..n {
                let i_off = i * n;
                for j in 0..n {
                    if j != i {
                        sum_q_hh += edge_support_a[i_off + j] * edge_support_b[i_off + j];
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

            // Save S_ij artifact if metric is Hellinger or JSD
            if metric.name() != "total_variation" && (k == 10 || k == 20 || k == 50) {
                let art_filename = format!("S_{}_k{:03}.json", metric.name(), k);
                let art_path = artifact_dir.join(&art_filename);
                let serialized = serde_json::to_vec(&edge_support).unwrap();
                let hash = compute_sha256(&serialized);
                let mut file = File::create(&art_path).unwrap();
                file.write_all(&serialized).unwrap();
                artifact_hashes.insert(art_filename, hash);
            }

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

                // 2. Sub-dataset SNLI and MNLI observed Q
                let mut sum_snli = 0.0f64;
                for &i in &snli_indices {
                    let i_off = i * n;
                    for &j in &snli_indices {
                        if j != i {
                            sum_snli += w_m[i_off + j] * edge_support[i_off + j];
                        }
                    }
                }
                let q_snli_mean = sum_snli / (snli_indices.len() * k) as f64;

                let mut sum_mnli = 0.0f64;
                for &i in &mnli_indices {
                    let i_off = i * n;
                    for &j in &mnli_indices {
                        if j != i {
                            sum_mnli += w_m[i_off + j] * edge_support[i_off + j];
                        }
                    }
                }
                let q_mnli_mean = sum_mnli / (mnli_indices.len() * k) as f64;

                // 3. 10,000 Stratified Item-Identity Permutations
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

                // 4. Exact-Profile-Preserving Permutations (1,000 draws for speed)
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

                // 5. SNLI / MNLI null baselines (from mean stratified perms)
                let q_snli_null_mean = q_null_mean * (snli_indices.len() as f64 / n as f64);
                let q_mnli_null_mean = q_null_mean * (mnli_indices.len() as f64 / n as f64);

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
                        q_snli_mean,
                        q_snli_null_mean,
                        q_mnli_mean,
                        q_mnli_null_mean,
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
        seed_stability_pearson_r,
        seed_stability_mse,
        artifact_sha256: artifact_hashes,
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
