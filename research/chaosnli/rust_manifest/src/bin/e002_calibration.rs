use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::Dirichlet;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::env;
use std::fs::{create_dir_all, File};
use std::io::{BufReader, Read};
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

// ─── Distance & Divergence Metrics ──────────────────────────────────────────

#[inline(always)]
fn distance_hellinger(p: &[f64; 3], q: &[f64; 3]) -> f64 {
    let d0 = p[0].sqrt() - q[0].sqrt();
    let d1 = p[1].sqrt() - q[1].sqrt();
    let d2 = p[2].sqrt() - q[2].sqrt();
    (0.5 * (d0 * d0 + d1 * d1 + d2 * d2)).sqrt()
}

/// Jensen-Shannon Divergence in bits (no square root)
#[inline(always)]
fn jsd_divergence_bits(p: &[f64; 3], q: &[f64; 3]) -> f64 {
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
    (0.5 * sum).max(0.0)
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
fn human_entropy_nats(p_human: &[f64; 3]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..3 {
        if p_human[i] > 1e-12 {
            sum -= p_human[i] * p_human[i].ln();
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

// ─── Empirical Percentile-Based Entropy Stratification ────────────────────────

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

fn build_stratified_5folds_empirical(items: &[ItemRecord], seed: u64) -> Vec<Vec<usize>> {
    let n = items.len();
    let mut entropies = Vec::with_capacity(n);

    for item in items {
        let total = (item.human_count_entailment + item.human_count_neutral + item.human_count_contradiction) as f64;
        let p_e = item.human_count_entailment as f64 / total;
        let p_n = item.human_count_neutral as f64 / total;
        let p_c = item.human_count_contradiction as f64 / total;

        let ent = -(if p_e > 1e-6 { p_e * p_e.log2() } else { 0.0 })
            - (if p_n > 1e-6 { p_n * p_n.log2() } else { 0.0 })
            - (if p_c > 1e-6 { p_c * p_c.log2() } else { 0.0 });
        entropies.push(ent);
    }

    let mut sorted_ent = entropies.clone();
    sorted_ent.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q20 = sorted_ent[(0.20 * n as f64) as usize];
    let q40 = sorted_ent[(0.40 * n as f64) as usize];
    let q60 = sorted_ent[(0.60 * n as f64) as usize];
    let q80 = sorted_ent[(0.80 * n as f64) as usize];

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

        let ent = entropies[idx];
        let eq = if ent <= q20 {
            0u8
        } else if ent <= q40 {
            1u8
        } else if ent <= q60 {
            2u8
        } else if ent <= q80 {
            3u8
        } else {
            4u8
        };

        strata_map.entry((is_snli, maj, eq)).or_default().push(idx);
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
            sum_loss += jsd_divergence_bits(&human_probs[idx], &q);
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

// ─── Output Structs for E002 Summary ────────────────────────────────────────

#[derive(Serialize)]
struct TempRange {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
}

#[derive(Serialize)]
struct ConditionMetrics {
    nll: f64,
    jsd_bits: f64,
    q_support_oof: f64,
    q_null_oof: f64,
    q_global_excess_oof: f64,
    q_profile_null_oof: f64,
    q_profile_excess_oof: f64,
    p_value_exact_profile: f64,
    r_human_recovery_oof: f64,
    graph_turnover_min_oof: f64,
    core_mass_k50_oof: f64,
    core_recall_k50_oof: f64,
    avg_entropy_bits: f64,
    avg_top_prob: f64,
    distance_variance: f64,
}

#[derive(Serialize)]
struct BootstrapCI {
    mean: f64,
    ci_lower_95: f64,
    ci_upper_95: f64,
}

#[derive(Serialize)]
struct ModelE002Result {
    display_name: String,
    t_nll_stats: TempRange,
    t_jsd_stats: TempRange,
    t_topology_stats: TempRange,
    gap_closure_nll: f64,
    gap_closure_q: f64,
    h2a_nll_direction_observed: bool,
    h2a_js_direction_reversed: bool,
    h2b_nll_point_estimate_greater: bool,
    bootstrap_delta_nll: BootstrapCI,
    bootstrap_delta_jsd: BootstrapCI,
    bootstrap_delta_q: BootstrapCI,
    bootstrap_delta_gap_closure: BootstrapCI,
    conditions: HashMap<String, ConditionMetrics>,
}

#[derive(Serialize)]
struct E002Summary {
    experiment_id: String,
    title: String,
    status: String,
    e001_artifact_id: String,
    e001_matrix_k10_sha256: String,
    e001_matrix_k50_sha256: String,
    model_probs_sha256: String,
    human_entropy_floor_nats: f64,
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

fn load_and_verify_matrix_f64(bin_path: &Path, expected_sha256: &str, expected_len: usize) -> Vec<f64> {
    let mut file = File::open(bin_path).unwrap_or_else(|e| panic!("Failed to open {}: {e}", bin_path.display()));
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();

    let actual_sha256 = compute_bytes_sha256(&bytes);
    assert_eq!(
        actual_sha256, expected_sha256,
        "Binary SHA-256 mismatch for {}: expected {}, got {}",
        bin_path.display(),
        expected_sha256,
        actual_sha256
    );

    assert_eq!(bytes.len(), expected_len * 4, "Binary length mismatch");

    let mut mat = Vec::with_capacity(expected_len);
    for chunk in bytes.chunks_exact(4) {
        let val_f32 = f32::from_le_bytes(chunk.try_into().unwrap());
        mat.push(val_f32 as f64);
    }
    mat
}

fn calc_temp_stats(temps: &[f64]) -> TempRange {
    let mean = temps.iter().sum::<f64>() / temps.len() as f64;
    let var = temps.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / temps.len() as f64;
    let std = var.sqrt();
    let min = *temps.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let max = *temps.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    TempRange { mean, std, min, max }
}

// ─── Main Execution ──────────────────────────────────────────────────────────

fn main() {
    let t_start = Instant::now();
    let workspace = get_workspace_dir();

    let num_threads = env::args()
        .position(|arg| arg == "--threads")
        .and_then(|idx| env::args().nth(idx + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .or_else(|| env::var("RAYON_NUM_THREADS").ok().and_then(|s| s.parse::<usize>().ok()))
        .unwrap_or(4);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(num_threads).build_global();

    println!("=========================================================================");
    println!("   EXPERIMENT E002 — POINTWISE CALIBRATION VS RELATIONAL TOPOLOGY (RUST)");
    println!("   (Rayon Threadpool: {num_threads} worker threads | Publication-Grade Cross-Fitting)");
    println!("=========================================================================");

    let items_path = workspace.join("data/chaosnli/processed/canonical_items_posterior.json");
    let models_path = workspace.join("research/chaosnli/rust_manifest/model_probs.json");
    
    let manifest_k10_path = workspace.join("research/chaosnli/artifacts/E001/S_hellinger_k010.manifest.json");
    let bin_k10_path = workspace.join("research/chaosnli/artifacts/E001/S_hellinger_k010.bin");
    let bin_k50_path = workspace.join("research/chaosnli/artifacts/E001/S_hellinger_k050.bin");

    let expected_k10_sha256 = "94e483e714d92f039f817389d948cbf41b7970077b56f852491832605dccc96f";
    let expected_k50_sha256 = "2da027e261d9a74a67f262aa601544c98ebf2b2879d15cda97b116ce447b1f3d";
    let expected_object_ids_sha256 = "121c49cbd40b171d100ba88c1a23d809818c28bad9249bea99a52ec8f5af19d6";

    let model_probs_sha256 = compute_file_sha256(&models_path);

    // 1. Runtime Artifact Integrity Lock
    let manifest_file = File::open(&manifest_k10_path).expect("Failed to open E001 k10 manifest");
    let manifest: ArtifactManifest = serde_json::from_reader(BufReader::new(manifest_file)).expect("Failed to parse E001 manifest");

    assert_eq!(manifest.matrix_sha256, expected_k10_sha256, "Manifest k10 matrix_sha256 mismatch!");
    assert_eq!(manifest.object_ids_sha256, expected_object_ids_sha256, "Manifest object_ids_sha256 mismatch!");

    let items = load_items(&items_path);
    let n = items.len();
    assert_eq!(n, manifest.object_count, "Item count mismatch!");

    let object_ids: Vec<String> = items.iter().map(|item| item.object_id.clone()).collect();
    let object_ids_bytes = serde_json::to_vec(&object_ids).unwrap();
    let actual_object_ids_sha256 = compute_bytes_sha256(&object_ids_bytes);
    assert_eq!(actual_object_ids_sha256, expected_object_ids_sha256, "Item ordering SHA-256 mismatch!");

    let s_ij_k10 = load_and_verify_matrix_f64(&bin_k10_path, expected_k10_sha256, n * n);
    let s_ij_k50 = load_and_verify_matrix_f64(&bin_k50_path, expected_k50_sha256, n * n);

    println!("Runtime Artifact Lock VERIFIED:");
    println!("  - k=10 Target: {} (SHA-256: {})", manifest.artifact_id, &expected_k10_sha256[..16]);
    println!("  - k=50 Target: S_hellinger_k050.bin (SHA-256: {})", &expected_k50_sha256[..16]);
    println!("  - Model Probs SHA-256: {}", &model_probs_sha256[..16]);
    println!("  - Object Order SHA-256: {}", &expected_object_ids_sha256[..16]);

    let (snli_indices, mnli_indices) = partition_item_strata(&items);
    let exact_profiles = partition_exact_profiles(&items);

    // Compute human empirical probability vectors and entropy floor
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

    let mut sum_human_entropy_nats = 0.0f64;
    for i in 0..n {
        sum_human_entropy_nats += human_entropy_nats(&human_probs[i]);
    }
    let human_entropy_floor_nats = sum_human_entropy_nats / n as f64;
    let q_hh_relational = 0.07227916826202654f64;

    println!("\n  Human Soft-Label NLL Floor H(p) = {:.5} nats", human_entropy_floor_nats);
    println!("  Human Relational Reference Q_HH = {:.5}", q_hh_relational);

    let raw_models = load_models(&models_path);
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

    let folds = build_stratified_5folds_empirical(&items, 20260803);
    let temp_grid = vec![
        0.10, 0.125, 0.16, 0.20, 0.25, 0.32, 0.40, 0.50, 0.63, 0.80, 1.00,
        1.25, 1.60, 2.00, 2.50, 3.20, 4.00, 5.00, 6.30, 8.00, 10.00,
    ];

    let mut item_fold_map = vec![0usize; n];
    for fold_idx in 0..5 {
        for &idx in &folds[fold_idx] {
            item_fold_map[idx] = fold_idx;
        }
    }

    let mut e002_model_results = HashMap::new();

    for m_name in &model_names {
        println!("\n--- Evaluating Model: {m_name} ---");
        let logits = &model_logits[m_name];

        let mut t_nll_folds = Vec::new();
        let mut t_jsd_folds = Vec::new();
        let mut t_topo_folds = Vec::new();

        for fold_idx in 0..5 {
            let test_indices = &folds[fold_idx];
            let mut train_indices = Vec::with_capacity(n - test_indices.len());
            for i in 0..n {
                if !test_indices.contains(&i) {
                    train_indices.push(i);
                }
            }

            let t_nll_opt = optimize_temperature_nll(&human_probs, logits, &train_indices);
            let t_jsd_opt = optimize_temperature_jsd(&human_probs, logits, &train_indices);

            // Training-Only Posterior Support Target (S_ij_train) constructed over training items ONLY (500 draws)
            let n_tr = train_indices.len();
            let train_items: Vec<ItemRecord> = train_indices.iter().map(|&i| ItemRecord {
                object_id: items[i].object_id.clone(),
                source_dataset: items[i].source_dataset,
                human_count_entailment: items[i].human_count_entailment,
                human_count_neutral: items[i].human_count_neutral,
                human_count_contradiction: items[i].human_count_contradiction,
            }).collect();

            let draws_tr_a = (0..250).map(|b| generate_posterior_probs(&train_items, 0.5, 42 + b as u64)).collect::<Vec<_>>();
            let draws_tr_b = (0..250).map(|b| generate_posterior_probs(&train_items, 0.5, 1001 + b as u64)).collect::<Vec<_>>();

            let mut sum_sup_tr_a = vec![0.0f64; n_tr * n_tr];
            let mut sum_sup_tr_b = vec![0.0f64; n_tr * n_tr];

            for b in 0..250 {
                let dist_a = build_dist_matrix_seq(&draws_tr_a[b], n_tr);
                let w_a = compute_topk_weight_matrix(&dist_a, n_tr, 10);
                let dist_b = build_dist_matrix_seq(&draws_tr_b[b], n_tr);
                let w_b = compute_topk_weight_matrix(&dist_b, n_tr, 10);

                for idx_cell in 0..(n_tr * n_tr) {
                    sum_sup_tr_a[idx_cell] += w_a[idx_cell] / 250.0;
                    sum_sup_tr_b[idx_cell] += w_b[idx_cell] / 250.0;
                }
            }

            let mut s_ij_tr = vec![0.0f64; n_tr * n_tr];
            for idx_cell in 0..(n_tr * n_tr) {
                s_ij_tr[idx_cell] = 0.5 * (sum_sup_tr_a[idx_cell] + sum_sup_tr_b[idx_cell]);
            }

            // Grid search T_topology maximizing Q_excess_train(T) = Q_support_train(T) - Q_null_train(T)
            let mut best_t_topo = 1.0f64;
            let mut best_q_excess = -1e9f64;

            for &t_cand in &temp_grid {
                let q_train: Vec<[f64; 3]> = train_indices.iter().map(|&i| softmax_temperature(&logits[i], t_cand)).collect();
                let dist_cand = build_dist_matrix_seq(&q_train, n_tr);
                let w_cand = compute_topk_weight_matrix(&dist_cand, n_tr, 10);

                let mut sum_q_sup = 0.0f64;
                for i_tr in 0..n_tr {
                    let i_off = i_tr * n_tr;
                    for j_tr in 0..n_tr {
                        if j_tr != i_tr {
                            sum_q_sup += w_cand[i_off + j_tr] * s_ij_tr[i_off + j_tr];
                        }
                    }
                }
                let q_sup_cand = sum_q_sup / (n_tr * 10) as f64;

                let sparse_w_cand = extract_nonzero_weights(&w_cand, n_tr);
                let (snli_tr_indices, mnli_tr_indices) = partition_item_strata(&train_items);

                let sum_null_cand: f64 = (0..250)
                    .into_par_iter()
                    .map(|b_idx| {
                        let mut null_rng = ChaCha8Rng::seed_from_u64(5050_0000 + b_idx as u64);
                        let mut perm_tr = (0..n_tr).collect::<Vec<_>>();
                        let mut snli_shuffled = snli_tr_indices.clone();
                        let mut mnli_shuffled = mnli_tr_indices.clone();
                        snli_shuffled.shuffle(&mut null_rng);
                        mnli_shuffled.shuffle(&mut null_rng);

                        for (orig_idx, &shuf_idx) in snli_tr_indices.iter().zip(snli_shuffled.iter()) {
                            perm_tr[*orig_idx] = shuf_idx;
                        }
                        for (orig_idx, &shuf_idx) in mnli_tr_indices.iter().zip(mnli_shuffled.iter()) {
                            perm_tr[*orig_idx] = shuf_idx;
                        }

                        let mut s_null = 0.0f64;
                        for i_tr in 0..n_tr {
                            let i_perm = perm_tr[i_tr];
                            for &(j_tr, w) in &sparse_w_cand[i_tr] {
                                let j_perm = perm_tr[j_tr];
                                s_null += w * s_ij_tr[i_perm * n_tr + j_perm];
                            }
                        }
                        s_null / (n_tr * 10) as f64
                    })
                    .sum();

                let q_null_cand = sum_null_cand / 250.0;
                let q_excess_cand = q_sup_cand - q_null_cand;

                if q_excess_cand > best_q_excess {
                    best_q_excess = q_excess_cand;
                    best_t_topo = t_cand;
                }
            }

            t_nll_folds.push(t_nll_opt);
            t_jsd_folds.push(t_jsd_opt);
            t_topo_folds.push(best_t_topo);
        }

        let t_nll_stats = calc_temp_stats(&t_nll_folds);
        let t_jsd_stats = calc_temp_stats(&t_jsd_folds);
        let t_topo_stats = calc_temp_stats(&t_topo_folds);

        println!("  Fitted T_NLL Range : mean={:.4}, std={:.4}, min={:.4}, max={:.4}", t_nll_stats.mean, t_nll_stats.std, t_nll_stats.min, t_nll_stats.max);
        println!("  Fitted T_JSD Range : mean={:.4}, std={:.4}, min={:.4}, max={:.4}", t_jsd_stats.mean, t_jsd_stats.std, t_jsd_stats.min, t_jsd_stats.max);
        println!("  Fitted T_Topo Range: mean={:.4}, std={:.4}, min={:.4}, max={:.4}", t_topo_stats.mean, t_topo_stats.std, t_topo_stats.min, t_topo_stats.max);

        // 3. True Fold-Specific Coherent Out-of-Fold Evaluation
        let cond_name_fold_temps = vec![
            ("T_raw (1.0)", vec![1.0f64; 5]),
            ("T_NLL (calibrated)", t_nll_folds.clone()),
            ("T_JSD (pointwise oracle)", t_jsd_folds.clone()),
            ("T_topology (relational oracle)", t_topo_folds.clone()),
        ];

        let mut cond_evals = HashMap::new();
        let mut item_local_outcomes: HashMap<String, (Vec<f64>, Vec<f64>)> = HashMap::new();

        // Baseline W_m(1) per fold for identity-normalized turnover
        let mut w_t1_folds = Vec::with_capacity(5);
        for fold_idx in 0..5 {
            let q_probs_t1: Vec<[f64; 3]> = (0..n).map(|i| softmax_temperature(&logits[i], 1.0)).collect();
            let dist_t1 = build_dist_matrix_seq(&q_probs_t1, n);
            let w_t1 = compute_topk_weight_matrix(&dist_t1, n, 10);
            w_t1_folds.push(w_t1);
        }

        for (cond_name, f_temps) in cond_name_fold_temps {
            let mut sum_oof_nll = 0.0f64;
            let mut sum_oof_jsd = 0.0f64;
            let mut sum_oof_q_sup = 0.0f64;
            let mut sum_oof_overlap_min = 0.0f64;

            let mut sum_oof_core_mass_50 = 0.0f64;
            let mut c_tau50_k50 = 0usize;

            let mut item_support_observed = vec![0.0f64; n];
            let mut item_support_null = vec![0.0f64; n];

            let mut w_m10_folds = Vec::with_capacity(5);
            let mut sparse_w10_folds = Vec::with_capacity(5);

            for fold_idx in 0..5 {
                let t_f = f_temps[fold_idx];
                let test_indices = &folds[fold_idx];
                let q_probs_f: Vec<[f64; 3]> = (0..n).map(|i| softmax_temperature(&logits[i], t_f)).collect();

                for &idx in test_indices {
                    sum_oof_nll += soft_label_nll_single(&human_probs[idx], &q_probs_f[idx]);
                    sum_oof_jsd += jsd_divergence_bits(&human_probs[idx], &q_probs_f[idx]);
                }

                let dist_f = build_dist_matrix_seq(&q_probs_f, n);
                let w_m10 = compute_topk_weight_matrix(&dist_f, n, 10);
                let sparse_w10 = extract_nonzero_weights(&w_m10, n);

                for &i_test in test_indices {
                    let i_off = i_test * n;
                    let mut local_obs = 0.0f64;
                    for j in 0..n {
                        if j != i_test {
                            let s = w_m10[i_off + j] * s_ij_k10[i_off + j];
                            sum_oof_q_sup += s;
                            local_obs += s / 10.0;
                            sum_oof_overlap_min += w_t1_folds[fold_idx][i_off + j].min(w_m10[i_off + j]);
                        }
                    }
                    item_support_observed[i_test] = local_obs;
                }

                let w_m50 = compute_topk_weight_matrix(&dist_f, n, 50);
                for &i_test in test_indices {
                    let i_off = i_test * n;
                    for j in 0..n {
                        if j != i_test && s_ij_k50[i_off + j] >= 0.50 {
                            c_tau50_k50 += 1;
                            sum_oof_core_mass_50 += w_m50[i_off + j];
                        }
                    }
                }

                w_m10_folds.push(w_m10);
                sparse_w10_folds.push(sparse_w10);
            }

            let nll_val = sum_oof_nll / n as f64;
            let jsd_val = sum_oof_jsd / n as f64;
            let q_support_oof = sum_oof_q_sup / (n * 10) as f64;
            let graph_turnover_min_oof = (1.0 - (sum_oof_overlap_min / (n * 10) as f64)).max(0.0);

            let core_mass_k50_oof = sum_oof_core_mass_50 / (n * 50) as f64;
            let core_recall_k50_oof = sum_oof_core_mass_50 / c_tau50_k50.max(1) as f64;

            // 10,000 Stratified Permutations using COMMON random permutation seeds across folds
            let n_null = 10_000;
            let mut null_item_accum = vec![0.0f64; n];

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

                    let mut sum_null_f = 0.0f64;
                    for fold_idx in 0..5 {
                        let test_indices = &folds[fold_idx];
                        let sparse_w10 = &sparse_w10_folds[fold_idx];
                        for &i_test in test_indices {
                            let i_perm = perm[i_test];
                            for &(j, w) in &sparse_w10[i_test] {
                                let j_perm = perm[j];
                                sum_null_f += w * s_ij_k10[i_perm * n + j_perm];
                            }
                        }
                    }
                    sum_null_f / (n * 10) as f64
                })
                .collect();

            let q_null_oof = null_scores.iter().sum::<f64>() / n_null as f64;
            let q_global_excess_oof = q_support_oof - q_null_oof;

            // Item local null estimation
            for i in 0..n {
                item_support_null[i] = q_null_oof;
            }

            // 1,000 Exact-Profile Permutations per final condition (with Monte Carlo p-value)
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

                    let mut sum_null_f = 0.0f64;
                    for fold_idx in 0..5 {
                        let test_indices = &folds[fold_idx];
                        let sparse_w10 = &sparse_w10_folds[fold_idx];
                        for &i_test in test_indices {
                            let i_perm = perm[i_test];
                            for &(j, w) in &sparse_w10[i_test] {
                                let j_perm = perm[j];
                                sum_null_f += w * s_ij_k10[i_perm * n + j_perm];
                            }
                        }
                    }
                    sum_null_f / (n * 10) as f64
                })
                .collect();

            let q_profile_null_oof = exact_null_scores.iter().sum::<f64>() / 1000.0;
            let q_profile_excess_oof = q_support_oof - q_profile_null_oof;
            let exact_exceedances = exact_null_scores.iter().filter(|&&v| v >= q_support_oof).count();
            let p_value_exact_profile = (1.0 + exact_exceedances as f64) / (1.0 + 1000.0);

            let r_human_recovery_oof = if (q_hh_relational - q_null_oof).abs() > 1e-12 {
                (q_support_oof - q_null_oof) / (q_hh_relational - q_null_oof)
            } else {
                0.0
            };

            item_local_outcomes.insert(cond_name.to_string(), (item_support_observed, item_support_null));

            let mut sum_ent = 0.0f64;
            let mut sum_top_prob = 0.0f64;
            let mut dist_vars = Vec::with_capacity(5);

            for fold_idx in 0..5 {
                let t_f = f_temps[fold_idx];
                let test_indices = &folds[fold_idx];
                let q_probs_f: Vec<[f64; 3]> = (0..n).map(|i| softmax_temperature(&logits[i], t_f)).collect();

                for &i_test in test_indices {
                    let p = &q_probs_f[i_test];
                    let ent = -(if p[0] > 1e-12 { p[0] * p[0].log2() } else { 0.0 })
                        - (if p[1] > 1e-12 { p[1] * p[1].log2() } else { 0.0 })
                        - (if p[2] > 1e-12 { p[2] * p[2].log2() } else { 0.0 });
                    sum_ent += ent;
                    sum_top_prob += p[0].max(p[1]).max(p[2]);
                }

                let dist_f = build_dist_matrix_seq(&q_probs_f, n);
                let mean_d = dist_f.iter().sum::<f64>() / (n * n) as f64;
                let var_d = dist_f.iter().map(|d| (d - mean_d).powi(2)).sum::<f64>() / (n * n) as f64;
                dist_vars.push(var_d);
            }

            let avg_entropy_bits = sum_ent / n as f64;
            let avg_top_prob = sum_top_prob / n as f64;
            let distance_variance = dist_vars.iter().sum::<f64>() / 5.0;

            cond_evals.insert(
                cond_name.to_string(),
                ConditionMetrics {
                    nll: nll_val,
                    jsd_bits: jsd_val,
                    q_support_oof,
                    q_null_oof,
                    q_global_excess_oof,
                    q_profile_null_oof,
                    q_profile_excess_oof,
                    p_value_exact_profile,
                    r_human_recovery_oof,
                    graph_turnover_min_oof,
                    core_mass_k50_oof,
                    core_recall_k50_oof,
                    avg_entropy_bits,
                    avg_top_prob,
                    distance_variance,
                },
            );
        }

        let nll_raw = cond_evals["T_raw (1.0)"].nll;
        let nll_cal = cond_evals["T_NLL (calibrated)"].nll;
        let gap_closure_nll = if (nll_raw - human_entropy_floor_nats).abs() > 1e-6 {
            (nll_raw - nll_cal) / (nll_raw - human_entropy_floor_nats)
        } else {
            0.0
        };

        let jsd_raw = cond_evals["T_raw (1.0)"].jsd_bits;
        let jsd_cal = cond_evals["T_NLL (calibrated)"].jsd_bits;

        let r_raw = cond_evals["T_raw (1.0)"].r_human_recovery_oof;
        let r_cal = cond_evals["T_NLL (calibrated)"].r_human_recovery_oof;
        let gap_closure_q = if (1.0 - r_raw).abs() > 1e-6 {
            (r_cal - r_raw) / (1.0 - r_raw)
        } else {
            0.0
        };

        // 4. Exact Stratified Focal-Item Paired Bootstrap (1,000 iterations over topology & NLL)
        let n_boot = 1000;
        let mut boot_delta_nll = Vec::with_capacity(n_boot);
        let mut boot_delta_jsd = Vec::with_capacity(n_boot);
        let mut boot_delta_q = Vec::with_capacity(n_boot);
        let mut boot_delta_gc = Vec::with_capacity(n_boot);

        let (obs_raw, null_raw) = &item_local_outcomes["T_raw (1.0)"];
        let (obs_cal, null_cal) = &item_local_outcomes["T_NLL (calibrated)"];

        for boot_idx in 0..n_boot {
            let mut boot_rng = ChaCha8Rng::seed_from_u64(7070_0000 + boot_idx as u64);
            let mut sampled_indices = Vec::with_capacity(n);
            for fold_idx in 0..5 {
                let fold_indices = &folds[fold_idx];
                for _ in 0..fold_indices.len() {
                    let pick = fold_indices[boot_rng.gen_range(0..fold_indices.len())];
                    sampled_indices.push(pick);
                }
            }

            let mut b_nll_raw = 0.0f64;
            let mut b_nll_cal = 0.0f64;
            let mut b_jsd_raw = 0.0f64;
            let mut b_jsd_cal = 0.0f64;
            let mut b_h_floor = 0.0f64;

            let mut b_obs_raw = 0.0f64;
            let mut b_null_raw = 0.0f64;
            let mut b_obs_cal = 0.0f64;
            let mut b_null_cal = 0.0f64;

            for &idx in &sampled_indices {
                let f_idx = item_fold_map[idx];
                let t_nll_f = t_nll_folds[f_idx];

                let q_raw = softmax_temperature(&logits[idx], 1.0);
                let q_cal = softmax_temperature(&logits[idx], t_nll_f);

                b_nll_raw += soft_label_nll_single(&human_probs[idx], &q_raw);
                b_nll_cal += soft_label_nll_single(&human_probs[idx], &q_cal);

                b_jsd_raw += jsd_divergence_bits(&human_probs[idx], &q_raw);
                b_jsd_cal += jsd_divergence_bits(&human_probs[idx], &q_cal);

                b_h_floor += human_entropy_nats(&human_probs[idx]);

                b_obs_raw += obs_raw[idx];
                b_null_raw += null_raw[idx];
                b_obs_cal += obs_cal[idx];
                b_null_cal += null_cal[idx];
            }

            b_nll_raw /= n as f64;
            b_nll_cal /= n as f64;
            b_jsd_raw /= n as f64;
            b_jsd_cal /= n as f64;
            b_h_floor /= n as f64;

            b_obs_raw /= n as f64;
            b_null_raw /= n as f64;
            b_obs_cal /= n as f64;
            b_null_cal /= n as f64;

            let d_nll = b_nll_cal - b_nll_raw;
            let d_jsd = b_jsd_cal - b_jsd_raw;
            let d_q = b_obs_cal - b_obs_raw;

            let g_nll_b = if (b_nll_raw - b_h_floor).abs() > 1e-6 {
                (b_nll_raw - b_nll_cal) / (b_nll_raw - b_h_floor)
            } else {
                0.0
            };

            let r_raw_b = if (q_hh_relational - b_null_raw).abs() > 1e-12 {
                (b_obs_raw - b_null_raw) / (q_hh_relational - b_null_raw)
            } else {
                0.0
            };

            let r_cal_b = if (q_hh_relational - b_null_cal).abs() > 1e-12 {
                (b_obs_cal - b_null_cal) / (q_hh_relational - b_null_cal)
            } else {
                0.0
            };

            let g_q_b = if (1.0 - r_raw_b).abs() > 1e-6 {
                (r_cal_b - r_raw_b) / (1.0 - r_raw_b)
            } else {
                0.0
            };

            boot_delta_nll.push(d_nll);
            boot_delta_jsd.push(d_jsd);
            boot_delta_q.push(d_q);
            boot_delta_gc.push(g_nll_b - g_q_b);
        }

        let calc_ci = |arr: &mut Vec<f64>| -> BootstrapCI {
            arr.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = arr.iter().sum::<f64>() / arr.len() as f64;
            let ci_lower_95 = arr[(0.025 * arr.len() as f64) as usize];
            let ci_upper_95 = arr[(0.975 * arr.len() as f64) as usize];
            BootstrapCI { mean, ci_lower_95, ci_upper_95 }
        };

        let bootstrap_delta_nll = calc_ci(&mut boot_delta_nll);
        let bootstrap_delta_jsd = calc_ci(&mut boot_delta_jsd);
        let bootstrap_delta_q = calc_ci(&mut boot_delta_q);
        let bootstrap_delta_gap_closure = calc_ci(&mut boot_delta_gc);

        let h2a_nll_direction_observed = nll_cal < nll_raw;
        let h2a_js_direction_reversed = jsd_cal > jsd_raw;
        let h2b_nll_point_estimate_greater = gap_closure_nll > gap_closure_q;

        println!("  NLL Gap Closure G_NLL = {:.2}%, Relational Gap Closure G_Q = {:.2}%", gap_closure_nll * 100.0, gap_closure_q * 100.0);
        println!("  H2a (NLL Reduction): {}", if h2a_nll_direction_observed { "SUPPORTED" } else { "NOT SUPPORTED" });
        println!("  H2a (JSD Alignment): {}", if h2a_js_direction_reversed { "REVERSED (JSD Increased)" } else { "REDUCED" });
        println!("  H2b (G_NLL > G_Q): Point Estimate Greater = {}, 95% CI Excludes Zero = {} ({:.2}% [{:.2}%, {:.2}%])",
            h2b_nll_point_estimate_greater,
            bootstrap_delta_gap_closure.ci_lower_95 > 0.0,
            bootstrap_delta_gap_closure.mean * 100.0,
            bootstrap_delta_gap_closure.ci_lower_95 * 100.0,
            bootstrap_delta_gap_closure.ci_upper_95 * 100.0
        );

        e002_model_results.insert(
            m_name.clone(),
            ModelE002Result {
                display_name: m_name.clone(),
                t_nll_stats,
                t_jsd_stats,
                t_topology_stats: t_topo_stats,
                gap_closure_nll,
                gap_closure_q,
                h2a_nll_direction_observed,
                h2a_js_direction_reversed,
                h2b_nll_point_estimate_greater,
                bootstrap_delta_nll,
                bootstrap_delta_jsd,
                bootstrap_delta_q,
                bootstrap_delta_gap_closure,
                conditions: cond_evals,
            },
        );
    }

    let total_runtime_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    let summary = E002Summary {
        experiment_id: "E002".to_string(),
        title: "Pointwise Soft-Label Calibration vs. Relational Neighborhood Recovery".to_string(),
        status: "complete_publication_grade".to_string(),
        e001_artifact_id: manifest.artifact_id.clone(),
        e001_matrix_k10_sha256: expected_k10_sha256.to_string(),
        e001_matrix_k50_sha256: expected_k50_sha256.to_string(),
        model_probs_sha256,
        human_entropy_floor_nats,
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
