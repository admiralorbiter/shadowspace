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

#[derive(Debug, Deserialize)]
struct ArtifactManifest {
    artifact_id: String,
    matrix_sha256: String,
    object_ids_sha256: String,
    object_count: usize,
}

// ─── Distance Metrics ────────────────────────────────────────────────────────

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
fn soft_label_nll_single(p_human: &[f64; 3], q_model: &[f64; 3]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..3 {
        if p_human[i] > 1e-12 {
            let q_safe = q_model[i].max(1e-12);
            sum -= p_human[i] * q_safe.ln();
        }
    }
    sum
}

#[inline(always)]
fn softmax_temperature(logits: &[f64; 3], t: f64) -> [f64; 3] {
    let z0 = logits[0] / t;
    let z1 = logits[1] / t;
    let z2 = logits[2] / t;
    let max_z = z0.max(z1).max(z2);
    let e0 = (z0 - max_z).exp();
    let e1 = (z1 - max_z).exp();
    let e2 = (z2 - max_z).exp();
    let sum_e = e0 + e1 + e2;
    [e0 / sum_e, e1 / sum_e, e2 / sum_e]
}

fn build_dist_matrix_seq(probs: &[[f64; 3]], n: usize) -> Vec<f64> {
    let mut dist = vec![0.0f64; n * n];
    for i in 0..n {
        let p_i = &probs[i];
        let i_off = i * n;
        for j in (i + 1)..n {
            let p_j = &probs[j];
            let d = distance_hellinger(p_i, p_j);
            dist[i_off + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

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

// ─── Stratified Fold Partitioning ───────────────────────────────────────────

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

fn build_stratified_5folds(items: &[ItemRecord], seed: u64) -> Vec<Vec<usize>> {
    let mut strata_map: HashMap<(u8, u8, u8), Vec<usize>> = HashMap::new();
    for (idx, item) in items.iter().enumerate() {
        let total = (item.human_count_entailment + item.human_count_neutral + item.human_count_contradiction) as f64;
        let p_e = item.human_count_entailment as f64 / total;
        let p_n = item.human_count_neutral as f64 / total;
        let p_c = item.human_count_contradiction as f64 / total;

        let is_snli = match item.source_dataset {
            Some(SourceDataset::ChaosnliSnli) => 1u8,
            _ => 0u8,
        };

        let maj = if p_e >= p_n && p_e >= p_c { 0u8 } else if p_n >= p_c { 1u8 } else { 2u8 };

        // Simple entropy binning (0..5)
        let ent = -(if p_e > 1e-6 { p_e * p_e.log2() } else { 0.0 })
            - (if p_n > 1e-6 { p_n * p_n.log2() } else { 0.0 })
            - (if p_c > 1e-6 { p_c * p_c.log2() } else { 0.0 });
        let ent_bin = ((ent / 1.585) * 4.99) as u8;

        strata_map.entry((is_snli, maj, ent_bin)).or_default().push(idx);
    }

    let mut folds = vec![Vec::new(); 5];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for mut group in strata_map.into_values() {
        group.shuffle(&mut rng);
        for (i, idx) in group.into_iter().enumerate() {
            folds[i % 5].push(idx);
        }
    }

    for fold in &mut folds {
        fold.sort_unstable();
    }
    folds
}

// ─── Golden Section Search for Optimal Temperature ───────────────────────────

fn optimize_temperature_nll(
    human_probs: &[[f64; 3]],
    logits: &[[f64; 3]],
    indices: &[usize],
) -> f64 {
    let loss_fn = |t: f64| -> f64 {
        let mut sum_loss = 0.0f64;
        for &idx in indices {
            let q = softmax_temperature(&logits[idx], t);
            sum_loss += soft_label_nll_single(&human_probs[idx], &q);
        }
        sum_loss / indices.len() as f64
    };

    // Golden section search on t in [0.05, 20.0]
    let mut a = 0.05f64;
    let mut b = 20.0f64;
    let phi = (5.0f64.sqrt() - 1.0) / 2.0;

    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);

    for _ in 0..40 {
        if loss_fn(c) < loss_fn(d) {
            b = d;
        } else {
            a = c;
        }
        c = b - phi * (b - a);
        d = a + phi * (b - a);
    }
    (a + b) / 2.0
}

fn optimize_temperature_jsd(
    human_probs: &[[f64; 3]],
    logits: &[[f64; 3]],
    indices: &[usize],
) -> f64 {
    let loss_fn = |t: f64| -> f64 {
        let mut sum_loss = 0.0f64;
        for &idx in indices {
            let q = softmax_temperature(&logits[idx], t);
            sum_loss += distance_jsd(&human_probs[idx], &q);
        }
        sum_loss / indices.len() as f64
    };

    let mut a = 0.05f64;
    let mut b = 20.0f64;
    let phi = (5.0f64.sqrt() - 1.0) / 2.0;

    let mut c = b - phi * (b - a);
    let mut d = a + phi * (b - a);

    for _ in 0..40 {
        if loss_fn(c) < loss_fn(d) {
            b = d;
        } else {
            a = c;
        }
        c = b - phi * (b - a);
        d = a + phi * (b - a);
    }
    (a + b) / 2.0
}

// ─── Output Structs for E002 Summary ────────────────────────────────────────

#[derive(Serialize)]
struct ConditionMetrics {
    nll: f64,
    jsd: f64,
    q_support: f64,
    q_null: f64,
    q_global_excess: f64,
    q_profile_null: f64,
    q_profile_excess: f64,
    core_mass_tau50: f64,
    core_recall_tau50: f64,
    avg_entropy_bits: f64,
    avg_top_prob: f64,
    distance_variance: f64,
}

#[derive(Serialize)]
struct ModelE002Result {
    display_name: String,
    t_nll_fitted: f64,
    t_jsd_fitted: f64,
    t_topology_fitted: f64,
    gap_closure_D: f64,
    gap_closure_Q: f64,
    h2b_confirmed: bool,
    conditions: HashMap<String, ConditionMetrics>,
}

#[derive(Serialize)]
struct E002Summary {
    experiment_id: String,
    title: String,
    e001_artifact_id: String,
    e001_matrix_sha256: String,
    d_hh_pointwise_jsd: f64,
    q_hh_relational: f64,
    models: HashMap<String, ModelE002Result>,
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

fn compute_bytes_sha256(data: &[u8]) -> String {
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

    println!("=========================================================================");
    println!("   EXPERIMENT E002 — POINTWISE CALIBRATION VS RELATIONAL TOPOLOGY (RUST)");
    println!("   (Rayon Threadpool: {num_threads} worker threads | 5-Fold Stratified CV)");
    println!("=========================================================================");

    let items_path = workspace.join("data/chaosnli/processed/canonical_items_posterior.json");
    let models_path = workspace.join("research/chaosnli/rust_manifest/model_probs.json");
    let manifest_path = workspace.join("research/chaosnli/artifacts/E001/S_hellinger_k010.manifest.json");
    let bin_path = workspace.join("research/chaosnli/artifacts/E001/S_hellinger_k010.bin");

    // 1. Verify and load frozen E001 matrix artifact
    let manifest_file = File::open(&manifest_path).expect("Failed to open E001 manifest");
    let manifest: ArtifactManifest = serde_json::from_reader(BufReader::new(manifest_file)).expect("Failed to parse E001 manifest");

    let mut bin_file = File::open(&bin_path).expect("Failed to open E001 bin artifact");
    let mut bin_bytes = Vec::new();
    bin_file.read_to_end(&mut bin_bytes).expect("Failed to read E001 bin artifact");

    let actual_hash = compute_bytes_sha256(&bin_bytes);
    assert_eq!(actual_hash, manifest.matrix_sha256, "E001 Matrix SHA-256 hash mismatch!");

    let n = manifest.object_count;
    let f32_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(bin_bytes.as_ptr() as *const f32, bin_bytes.len() / 4)
    };
    let s_ij: Vec<f64> = f32_slice.iter().map(|&v| v as f64).collect();

    println!("Loaded & verified frozen E001 artifact '{}' (SHA-256: {})", manifest.artifact_id, &actual_hash[..16]);

    let items = load_items(&items_path);
    let raw_models = load_models(&models_path);
    let (snli_indices, mnli_indices) = partition_item_strata(&items);
    let exact_profiles = partition_exact_profiles(&items);

    // Compute human empirical probability vectors
    let human_probs: Vec<[f64; 3]> = items
        .iter()
        .map(|item| {
            let total = (item.human_count_entailment + item.human_count_neutral + item.human_count_contradiction) as f64;
            [
                item.human_count_entailment as f64 / total,
                item.human_count_neutral as f64 / total,
                item.human_count_contradiction as f64 / total,
            ]
        })
        .collect();

    // Compute human split-half pointwise baseline (D_HH)
    println!("\nComputing human split-half pointwise baseline (D_HH)...");
    let draws_a = (0..250).map(|b| generate_posterior_probs(&items, 0.5, 42 + b as u64)).collect::<Vec<_>>();
    let draws_b = (0..250).map(|b| generate_posterior_probs(&items, 0.5, 1001 + b as u64)).collect::<Vec<_>>();

    let mut mean_p_a = vec![[0.0f64; 3]; n];
    let mut mean_p_b = vec![[0.0f64; 3]; n];
    for b in 0..250 {
        for i in 0..n {
            for c in 0..3 {
                mean_p_a[i][c] += draws_a[b][i][c] / 250.0;
                mean_p_b[i][c] += draws_b[b][i][c] / 250.0;
            }
        }
    }

    let mut sum_d_hh = 0.0f64;
    for i in 0..n {
        sum_d_hh += distance_jsd(&mean_p_a[i], &mean_p_b[i]);
    }
    let d_hh_pointwise = sum_d_hh / n as f64;
    let q_hh_relational = 0.07228f64;

    println!("  Human-Human Pointwise JSD Floor D_HH = {:.5}", d_hh_pointwise);
    println!("  Human-Human Relational Reference Q_HH = {:.5}", q_hh_relational);

    // Convert raw model probabilities to synthetic logits z = ln(p) for temperature scaling
    let mut model_logits: HashMap<String, Vec<[f64; 3]>> = HashMap::new();
    for (m_name, m_probs) in &raw_models {
        let logits: Vec<[f64; 3]> = m_probs
            .iter()
            .map(|p| [p[0].max(1e-12).ln(), p[1].max(1e-12).ln(), p[2].max(1e-12).ln()])
            .collect();
        model_logits.insert(m_name.clone(), logits);
    }

    let mut model_names: Vec<String> = model_logits.keys().cloned().collect();
    model_names.sort();

    let folds = build_stratified_5folds(&items, 20260803);
    let temp_grid = vec![
        0.10, 0.125, 0.16, 0.20, 0.25, 0.32, 0.40, 0.50, 0.63, 0.80, 1.00,
        1.25, 1.60, 2.00, 2.50, 3.20, 4.00, 5.00, 6.30, 8.00, 10.00,
    ];

    let mut e002_model_results = HashMap::new();

    for m_name in &model_names {
        println!("\n--- Evaluating Model: {m_name} ---");
        let logits = &model_logits[m_name];

        let mut t_nll_folds = Vec::new();
        let mut t_jsd_folds = Vec::new();
        let mut t_topo_folds = Vec::new();

        let mut oof_probs_raw = vec![[0.0f64; 3]; n];
        let mut oof_probs_nll = vec![[0.0f64; 3]; n];
        let mut oof_probs_jsd = vec![[0.0f64; 3]; n];
        let mut oof_probs_topo = vec![[0.0f64; 3]; n];

        for fold_idx in 0..5 {
            let test_indices = &folds[fold_idx];
            let mut train_indices = Vec::new();
            for i in 0..n {
                if !test_indices.contains(&i) {
                    train_indices.push(i);
                }
            }

            let t_nll_opt = optimize_temperature_nll(&human_probs, logits, &train_indices);
            let t_jsd_opt = optimize_temperature_jsd(&human_probs, logits, &train_indices);

            // Subgraph top-k search for T_topology on train fold
            let n_tr = train_indices.len();
            let mut best_t_topo = 1.0f64;
            let mut best_q_topo = -1.0f64;

            for &t_cand in &temp_grid {
                let q_train: Vec<[f64; 3]> = train_indices.iter().map(|&i| softmax_temperature(&logits[i], t_cand)).collect();
                let dist_train = build_dist_matrix_seq(&q_train, n_tr);
                let w_train = compute_topk_weight_matrix(&dist_train, n_tr, 10);
                
                let mut sum_q_tr = 0.0f64;
                for i_tr in 0..n_tr {
                    let orig_i = train_indices[i_tr];
                    let i_off = i_tr * n_tr;
                    for j_tr in 0..n_tr {
                        if j_tr != i_tr {
                            let orig_j = train_indices[j_tr];
                            sum_q_tr += w_train[i_off + j_tr] * s_ij[orig_i * n + orig_j];
                        }
                    }
                }
                let q_sup_cand = sum_q_tr / (n_tr * 10) as f64;
                if q_sup_cand > best_q_topo {
                    best_q_topo = q_sup_cand;
                    best_t_topo = t_cand;
                }
            }

            t_nll_folds.push(t_nll_opt);
            t_jsd_folds.push(t_jsd_opt);
            t_topo_folds.push(best_t_topo);

            for &idx in test_indices {
                oof_probs_raw[idx] = softmax_temperature(&logits[idx], 1.0);
                oof_probs_nll[idx] = softmax_temperature(&logits[idx], t_nll_opt);
                oof_probs_jsd[idx] = softmax_temperature(&logits[idx], t_jsd_opt);
                oof_probs_topo[idx] = softmax_temperature(&logits[idx], best_t_topo);
            }
        }

        let t_nll_mean = t_nll_folds.iter().sum::<f64>() / 5.0;
        let t_jsd_mean = t_jsd_folds.iter().sum::<f64>() / 5.0;
        let t_topo_mean = t_topo_folds.iter().sum::<f64>() / 5.0;

        println!("  Fitted Temperatures: T_NLL = {t_nll_mean:.4}, T_JSD = {t_jsd_mean:.4}, T_topology = {t_topo_mean:.4}");

        let cond_probs = vec![
            ("T_raw (1.0)", oof_probs_raw),
            ("T_NLL (calibrated)", oof_probs_nll),
            ("T_JSD (pointwise oracle)", oof_probs_jsd),
            ("T_topology (relational oracle)", oof_probs_topo),
        ];

        let mut cond_evals = HashMap::new();

        for (cond_name, q_probs) in cond_probs {
            let mut sum_nll = 0.0f64;
            let mut sum_jsd = 0.0f64;
            for i in 0..n {
                sum_nll += soft_label_nll_single(&human_probs[i], &q_probs[i]);
                sum_jsd += distance_jsd(&human_probs[i], &q_probs[i]);
            }
            let nll_val = sum_nll / n as f64;
            let jsd_val = sum_jsd / n as f64;

            let dist_m = build_dist_matrix_seq(&q_probs, n);
            let w_m10 = compute_topk_weight_matrix(&dist_m, n, 10);
            let sparse_w10 = extract_nonzero_weights(&w_m10, n);

            let mut sum_obs = 0.0f64;
            for i in 0..n {
                let i_off = i * n;
                for j in 0..n {
                    if j != i {
                        sum_obs += w_m10[i_off + j] * s_ij[i_off + j];
                    }
                }
            }
            let q_support = sum_obs / (n * 10) as f64;

            // Recompute Stratified Permutation Null at temperature T (1,000 perms)
            let n_null = 1000;
            let null_scores: Vec<f64> = (0..n_null)
                .into_par_iter()
                .map(|b_idx| {
                    let mut null_rng = ChaCha8Rng::seed_from_u64(2026_08_03 + b_idx as u64);
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
                        for &(j, w) in &sparse_w10[i] {
                            let j_perm = perm[j];
                            sum_null += w * s_ij[i_perm * n + j_perm];
                        }
                    }
                    sum_null / (n * 10) as f64
                })
                .collect();

            let q_null = null_scores.iter().sum::<f64>() / n_null as f64;
            let q_global_excess = q_support - q_null;

            // Recompute Exact-Profile Permutation Null at temperature T
            let exact_null_scores: Vec<f64> = (0..1000)
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
                        for &(j, w) in &sparse_w10[i] {
                            let j_perm = perm[j];
                            sum_null += w * s_ij[i_perm * n + j_perm];
                        }
                    }
                    sum_null / (n * 10) as f64
                })
                .collect();

            let q_profile_null = exact_null_scores.iter().sum::<f64>() / 1000.0;
            let q_profile_excess = q_support - q_profile_null;

            // Core mass & recall at k=50
            let w_m50 = compute_topk_weight_matrix(&dist_m, n, 50);
            let mut sum_core_mass = 0.0f64;
            let mut c_tau50 = 0usize;
            for i in 0..n {
                let i_off = i * n;
                for j in 0..n {
                    if j != i && s_ij[i_off + j] >= 0.50 {
                        c_tau50 += 1;
                        sum_core_mass += w_m50[i_off + j];
                    }
                }
            }
            let core_mass_tau50 = sum_core_mass / (n * 50) as f64;
            let core_recall_tau50 = sum_core_mass / c_tau50.max(1) as f64;

            // Degeneracy safeguards
            let mut sum_ent = 0.0f64;
            let mut sum_top_prob = 0.0f64;
            for i in 0..n {
                let p = &q_probs[i];
                let ent = -(if p[0] > 1e-12 { p[0] * p[0].log2() } else { 0.0 })
                    - (if p[1] > 1e-12 { p[1] * p[1].log2() } else { 0.0 })
                    - (if p[2] > 1e-12 { p[2] * p[2].log2() } else { 0.0 });
                sum_ent += ent;
                sum_top_prob += p[0].max(p[1]).max(p[2]);
            }
            let avg_entropy_bits = sum_ent / n as f64;
            let avg_top_prob = sum_top_prob / n as f64;

            let mean_dist = dist_m.iter().sum::<f64>() / (n * n) as f64;
            let distance_variance = dist_m.iter().map(|d| (d - mean_dist).powi(2)).sum::<f64>() / (n * n) as f64;

            cond_evals.insert(
                cond_name.to_string(),
                ConditionMetrics {
                    nll: nll_val,
                    jsd: jsd_val,
                    q_support,
                    q_null,
                    q_global_excess,
                    q_profile_null,
                    q_profile_excess,
                    core_mass_tau50,
                    core_recall_tau50,
                    avg_entropy_bits,
                    avg_top_prob,
                    distance_variance,
                },
            );
        }

        let d_raw = cond_evals["T_raw (1.0)"].jsd;
        let d_cal = cond_evals["T_NLL (calibrated)"].jsd;
        let g_d = if (d_raw - d_hh_pointwise).abs() > 1e-6 {
            (d_raw - d_cal) / (d_raw - d_hh_pointwise)
        } else {
            0.0
        };

        let q_raw = cond_evals["T_raw (1.0)"].q_support;
        let q_cal = cond_evals["T_NLL (calibrated)"].q_support;
        let g_q = if (q_hh_relational - q_raw).abs() > 1e-6 {
            (q_cal - q_raw) / (q_hh_relational - q_raw)
        } else {
            0.0
        };

        let h2b_confirmed = g_d > g_q;
        println!("  Pointwise Gap Closure G_D = {:.2}%, Relational Gap Closure G_Q = {:.2}%", g_d * 100.0, g_q * 100.0);
        println!("  H2b Result: G_D > G_Q --> {}", if h2b_confirmed { "CONFIRMED" } else { "REJECTED" });

        e002_model_results.insert(
            m_name.clone(),
            ModelE002Result {
                display_name: m_name.clone(),
                t_nll_fitted: t_nll_mean,
                t_jsd_fitted: t_jsd_mean,
                t_topology_fitted: t_topo_mean,
                gap_closure_D: g_d,
                gap_closure_Q: g_q,
                h2b_confirmed,
                conditions: cond_evals,
            },
        );
    }

    let total_runtime_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    let summary = E002Summary {
        experiment_id: "E002".to_string(),
        title: "Pointwise Soft-Label Calibration vs. Relational Neighborhood Recovery".to_string(),
        e001_artifact_id: manifest.artifact_id.clone(),
        e001_matrix_sha256: manifest.matrix_sha256.clone(),
        d_hh_pointwise_jsd: d_hh_pointwise,
        q_hh_relational,
        models: e002_model_results,
        total_runtime_ms,
    };

    let summary_dir = workspace.join("research/chaosnli/lab/summaries");
    create_dir_all(&summary_dir).unwrap();
    let summary_path = summary_dir.join("E002_summary.json");
    let file = File::create(&summary_path).unwrap();
    serde_json::to_writer_pretty(file, &summary).unwrap();

    println!("\n=========================================================================");
    println!("   EXPERIMENT E002 COMPLETE IN {:.2}s", total_runtime_ms / 1000.0);
    println!("   Summary saved to {}", summary_path.display());
    println!("=========================================================================");
}
