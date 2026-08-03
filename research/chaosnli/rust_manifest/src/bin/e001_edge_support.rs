use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::Dirichlet;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs::{create_dir_all, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::Instant;

// ─── Data Structures ────────────────────────────────────────────────────────

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

// ─── Output Structs for E001 Summary ────────────────────────────────────────

#[derive(Serialize)]
struct ModelEdgeResult {
    display_name: String,
    q_edge_support_mean: f64,
    delta_edge_mean: f64,
    q_null_mean: f64,
}

#[derive(Serialize)]
struct ScaleResult {
    k: usize,
    mean_human_edge_support: f64,
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

// ─── Main Execution ──────────────────────────────────────────────────────────

fn main() {
    let t_start = Instant::now();
    let workspace = get_workspace_dir();

    println!("=========================================================================");
    println!("   EXPERIMENT E001 — POSTERIOR EDGE-SUPPORT GRAPH CONSTRUCTION");
    println!("=========================================================================");

    let items_path = workspace.join("data/chaosnli/processed/canonical_items_posterior.json");
    let models_path = workspace.join("research/chaosnli/rust_manifest/model_probs.json");

    let items = load_items(&items_path);
    let models = load_models(&models_path);
    let n = items.len();

    let (snli_indices, mnli_indices) = partition_item_strata(&items);
    println!("Loaded {n} items (SNLI={}, MNLI={})", snli_indices.len(), mnli_indices.len());
    println!("Loaded {} models", models.len());

    let mut model_names: Vec<String> = models.keys().cloned().collect();
    model_names.sort();

    let metrics = vec![Metric::Hellinger, Metric::JensenShannon, Metric::TotalVariation];
    let k_list = vec![5, 10, 20, 50];
    let n_draws = 500;
    let alpha_prior = 0.5;

    let mut metric_results = Vec::new();

    for &metric in &metrics {
        println!("\n--- Processing Metric: {} ---", metric.name());

        // Pre-build model weight matrices for this metric across all k
        let mut model_weights: HashMap<(String, usize), Vec<f64>> = HashMap::new();
        for m_name in &model_names {
            let m_probs = &models[m_name];
            let dist_m = build_dist_matrix_seq(m_probs, n, metric);
            for &k in &k_list {
                let w_m = compute_topk_weight_matrix(&dist_m, n, k);
                model_weights.insert((m_name.clone(), k), w_m);
            }
        }

        // Generate posterior draw weight matrices in parallel and reduce sum into edge_support_by_k
        println!("Sampling {n_draws} posterior-predictive draws and computing top-k graphs...");
        let edge_support_by_k: HashMap<usize, Vec<f64>> = (0..n_draws)
            .into_par_iter()
            .map(|b| {
                let seed = 42 + b as u64 * 1000;
                let probs_b = generate_posterior_probs(&items, alpha_prior, seed);
                let dist_b = build_dist_matrix_seq(&probs_b, n, metric);
                let mut map = HashMap::new();
                for &k in &k_list {
                    let w_b = compute_topk_weight_matrix(&dist_b, n, k);
                    map.insert(k, w_b);
                }
                map
            })
            .reduce(
                || {
                    let mut empty_map = HashMap::new();
                    for &k in &k_list {
                        empty_map.insert(k, vec![0.0f64; n * n]);
                    }
                    empty_map
                },
                |mut acc, map| {
                    for &k in &k_list {
                        let acc_vec = acc.get_mut(&k).unwrap();
                        let b_vec = &map[&k];
                        for idx in 0..(n * n) {
                            acc_vec[idx] += b_vec[idx];
                        }
                    }
                    acc
                },
            );

        let mut scale_results = Vec::new();

        for &k in &k_list {
            println!("  Evaluating scale k={k}...");

            let mut edge_support = edge_support_by_k[&k].clone();
            let inv_b = 1.0 / n_draws as f64;
            for idx in 0..(n * n) {
                edge_support[idx] *= inv_b;
            }

            // Compute human self-support & edge support stats
            let total_possible_edges = (n * (n - 1)) as f64;
            let mut sum_edge_support = 0.0f64;
            let mut count_tau_50 = 0usize;
            let mut count_tau_80 = 0usize;
            let mut count_tau_95 = 0usize;

            for i in 0..n {
                let i_off = i * n;
                for j in 0..n {
                    if j != i {
                        let s = edge_support[i_off + j];
                        sum_edge_support += s;
                        if s >= 0.50 { count_tau_50 += 1; }
                        if s >= 0.80 { count_tau_80 += 1; }
                        if s >= 0.95 { count_tau_95 += 1; }
                    }
                }
            }

            let mean_human_edge_support = sum_edge_support / (n * k) as f64;
            let density_tau_50 = count_tau_50 as f64 / total_possible_edges;
            let density_tau_80 = count_tau_80 as f64 / total_possible_edges;
            let density_tau_95 = count_tau_95 as f64 / total_possible_edges;
            let mean_degree_tau_50 = count_tau_50 as f64 / n as f64;

            // Evaluate each model against edge_support S_ij(k)
            let mut model_evals = HashMap::new();

            for m_name in &model_names {
                let w_m = &model_weights[&(m_name.clone(), k)];

                // Model alignment Q_edge(m, S) = (1 / (N * k)) sum_i sum_{j != i} W^m_ij * S_ij
                let mut sum_match = 0.0f64;
                for i in 0..n {
                    let i_off = i * n;
                    for j in 0..n {
                        if j != i {
                            sum_match += w_m[i_off + j] * edge_support[i_off + j];
                        }
                    }
                }
                let q_edge_support_mean = sum_match / (n * k) as f64;
                let delta_edge_mean = mean_human_edge_support - q_edge_support_mean;

                // Stratified Item Identity Permutation Null (100 shuffles)
                let mut null_rng = ChaCha8Rng::seed_from_u64(1001 + k as u64);
                let mut q_null_sum = 0.0f64;
                let n_null_perms = 100;

                for _ in 0..n_null_perms {
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
                        let i_off = i_perm * n;
                        for j in 0..n {
                            if j != i {
                                let j_perm = perm[j];
                                sum_null += w_m[i_off + j_perm] * edge_support[i * n + j];
                            }
                        }
                    }
                    q_null_sum += sum_null / (n * k) as f64;
                }
                let q_null_mean = q_null_sum / n_null_perms as f64;

                model_evals.insert(
                    m_name.clone(),
                    ModelEdgeResult {
                        display_name: m_name.clone(),
                        q_edge_support_mean,
                        delta_edge_mean,
                        q_null_mean,
                    },
                );
            }

            scale_results.push(ScaleResult {
                k,
                mean_human_edge_support,
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
        title: "Posterior Edge-Support Graph Construction & Model Comparison".to_string(),
        dataset_release: "chaosnli-canonical-2026-08-02".to_string(),
        n_items: n,
        snli_count: snli_indices.len(),
        mnli_count: mnli_indices.len(),
        n_posterior_draws: n_draws,
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
