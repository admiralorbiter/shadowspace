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

// ─── Output Structs for E002 Summary ────────────────────────────────────────

#[derive(Serialize)]
struct ConditionMetrics {
    nll: f64,
    jsd_bits: f64,
    q_support_heldout: f64,
    q_null_heldout: f64,
    q_global_excess: f64,
    q_profile_null: f64,
    q_profile_excess: f64,
    graph_turnover_rel_t1: f64,
    core_mass_k50: f64,
    core_recall_k50: f64,
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
    gap_closure_nll: f64,
    gap_closure_q: f64,
    h2a_nll_supported: bool,
    h2a_jsd_contradicted: bool,
    h2b_nll_confirmed: bool,
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
    println!("   (Rayon Threadpool: {num_threads} worker threads | 5-Fold Coherent Cross-Fitting)");
    println!("=========================================================================");

    let items_path = workspace.join("data/chaosnli/processed/canonical_items_posterior.json");
    let models_path = workspace.join("research/chaosnli/rust_manifest/model_probs.json");
    
    let manifest_k10_path = workspace.join("research/chaosnli/artifacts/E001/S_hellinger_k010.manifest.json");
    let bin_k10_path = workspace.join("research/chaosnli/artifacts/E001/S_hellinger_k010.bin");
    let bin_k50_path = workspace.join("research/chaosnli/artifacts/E001/S_hellinger_k050.bin");

    let expected_k10_sha256 = "94e483e714d92f039f817389d948cbf41b7970077b56f852491832605dccc96f";
    let expected_k50_sha256 = "2da027e261d9a74a67f262aa601544c98ebf2b2879d15cda97b116ce447b1f3d";
    let expected_object_ids_sha256 = "121c49cbd40b171d100ba88c1a23d809818c28bad9249bea99a52ec8f5af19d6";

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
    let q_hh_relational = 0.07228f64;

    println!("\n  Human Soft-Label NLL Floor H(p) = {:.5} nats", human_entropy_floor_nats);
    println!("  Human Relational Reference Q_HH = {:.5}", q_hh_relational);

    // Convert raw model probabilities to synthetic logits z = ln(p) for temperature scaling
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

            // 2. Training-Only Posterior Support Target (S_ij_train) constructed over training items ONLY
            let n_tr = train_indices.len();
            let train_items: Vec<ItemRecord> = train_indices.iter().map(|&i| ItemRecord {
                object_id: items[i].object_id.clone(),
                source_dataset: items[i].source_dataset,
                human_count_entailment: items[i].human_count_entailment,
                human_count_neutral: items[i].human_count_neutral,
                human_count_contradiction: items[i].human_count_contradiction,
            }).collect();

            // Compute expected edge support matrix S_train over training items (200 draws)
            let draws_tr_a = (0..100).map(|b| generate_posterior_probs(&train_items, 0.5, 42 + b as u64)).collect::<Vec<_>>();
            let draws_tr_b = (0..100).map(|b| generate_posterior_probs(&train_items, 0.5, 1001 + b as u64)).collect::<Vec<_>>();

            let mut sum_sup_tr_a = vec![0.0f64; n_tr * n_tr];
            let mut sum_sup_tr_b = vec![0.0f64; n_tr * n_tr];

            for b in 0..100 {
                let dist_a = build_dist_matrix_seq(&draws_tr_a[b], n_tr);
                let w_a = compute_topk_weight_matrix(&dist_a, n_tr, 10);
                let dist_b = build_dist_matrix_seq(&draws_tr_b[b], n_tr);
                let w_b = compute_topk_weight_matrix(&dist_b, n_tr, 10);

                for idx_cell in 0..(n_tr * n_tr) {
                    sum_sup_tr_a[idx_cell] += w_a[idx_cell] / 100.0;
                    sum_sup_tr_b[idx_cell] += w_b[idx_cell] / 100.0;
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

                // Fast 50-permutation training null at candidate T
                let sparse_w_cand = extract_nonzero_weights(&w_cand, n_tr);
                let mut sum_null_cand = 0.0f64;
                for b_idx in 0..50 {
                    let mut null_rng = ChaCha8Rng::seed_from_u64(5050_0000 + b_idx as u64);
                    let mut perm_tr = (0..n_tr).collect::<Vec<_>>();
                    perm_tr.shuffle(&mut null_rng);

                    for i_tr in 0..n_tr {
                        let i_perm = perm_tr[i_tr];
                        for &(j_tr, w) in &sparse_w_cand[i_tr] {
                            let j_perm = perm_tr[j_tr];
                            sum_null_cand += w * s_ij_tr[i_perm * n_tr + j_perm];
                        }
                    }
                }
                let q_null_cand = sum_null_cand / (50 * n_tr * 10) as f64;
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

        let t_nll_mean = t_nll_folds.iter().sum::<f64>() / 5.0;
        let t_jsd_mean = t_jsd_folds.iter().sum::<f64>() / 5.0;
        let t_topo_mean = t_topo_folds.iter().sum::<f64>() / 5.0;

        println!("  Fitted Temperatures: T_NLL = {t_nll_mean:.4}, T_JSD = {t_jsd_mean:.4}, T_topology = {t_topo_mean:.4}");

        // 3. Coherent Full-Dataset Graph Evaluation (Single Temperature applied to all N items)
        let cond_temps = vec![
            ("T_raw (1.0)", 1.0f64),
            ("T_NLL (calibrated)", t_nll_mean),
            ("T_JSD (pointwise oracle)", t_jsd_mean),
            ("T_topology (relational oracle)", t_topo_mean),
        ];

        let mut cond_evals = HashMap::new();

        // Baseline W_m(1) at T=1.0 for graph turnover calculation
        let q_probs_t1: Vec<[f64; 3]> = (0..n).map(|i| softmax_temperature(&logits[i], 1.0)).collect();
        let dist_t1 = build_dist_matrix_seq(&q_probs_t1, n);
        let w_t1 = compute_topk_weight_matrix(&dist_t1, n, 10);

        for (cond_name, t_val) in cond_temps {
            let q_probs: Vec<[f64; 3]> = (0..n).map(|i| softmax_temperature(&logits[i], t_val)).collect();

            let mut sum_nll = 0.0f64;
            let mut sum_jsd = 0.0f64;
            for i in 0..n {
                sum_nll += soft_label_nll_single(&human_probs[i], &q_probs[i]);
                sum_jsd += jsd_divergence_bits(&human_probs[i], &q_probs[i]);
            }
            let nll_val = sum_nll / n as f64;
            let jsd_val = sum_jsd / n as f64;

            let dist_m = build_dist_matrix_seq(&q_probs, n);
            let w_m10 = compute_topk_weight_matrix(&dist_m, n, 10);
            let sparse_w10 = extract_nonzero_weights(&w_m10, n);

            // Compute held-out topology score across 5 coherent fold evaluations
            let mut sum_q_heldout = 0.0f64;
            for fold_idx in 0..5 {
                let test_indices = &folds[fold_idx];
                for &i_test in test_indices {
                    let i_off = i_test * n;
                    for j in 0..n {
                        if j != i_test {
                            sum_q_heldout += w_m10[i_off + j] * s_ij_k10[i_off + j];
                        }
                    }
                }
            }
            let q_support_heldout = sum_q_heldout / (n * 10) as f64;

            // Compute Graph Turnover relative to T=1.0: Turnover(T) = 1 - (1/Nk) sum W(1) * W(T)
            let mut sum_overlap = 0.0f64;
            for idx_cell in 0..(n * n) {
                sum_overlap += w_t1[idx_cell] * w_m10[idx_cell];
            }
            let graph_turnover_rel_t1 = 1.0 - (sum_overlap / (n * 10) as f64);

            // 10,000 Stratified Permutations per final condition via Rayon
            let n_null = 10_000;
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
                            sum_null += w * s_ij_k10[i_perm * n + j_perm];
                        }
                    }
                    sum_null / (n * 10) as f64
                })
                .collect();

            let q_null_heldout = null_scores.iter().sum::<f64>() / n_null as f64;
            let q_global_excess = q_support_heldout - q_null_heldout;

            // 1,000 Exact-Profile Permutations per final condition
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
                            sum_null += w * s_ij_k10[i_perm * n + j_perm];
                        }
                    }
                    sum_null / (n * 10) as f64
                })
                .collect();

            let q_profile_null = exact_null_scores.iter().sum::<f64>() / 1000.0;
            let q_profile_excess = q_support_heldout - q_profile_null;

            // Core mass & recall computed against independent k=50 support matrix (S_hellinger_k050.bin)
            let w_m50 = compute_topk_weight_matrix(&dist_m, n, 50);
            let mut sum_core_mass_50 = 0.0f64;
            let mut c_tau50_k50 = 0usize;
            for i in 0..n {
                let i_off = i * n;
                for j in 0..n {
                    if j != i && s_ij_k50[i_off + j] >= 0.50 {
                        c_tau50_k50 += 1;
                        sum_core_mass_50 += w_m50[i_off + j];
                    }
                }
            }
            let core_mass_k50 = sum_core_mass_50 / (n * 50) as f64;
            let core_recall_k50 = sum_core_mass_50 / c_tau50_k50.max(1) as f64;

            // Degeneracy & summary metrics
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
                    jsd_bits: jsd_val,
                    q_support_heldout,
                    q_null_heldout,
                    q_global_excess,
                    q_profile_null,
                    q_profile_excess,
                    graph_turnover_rel_t1,
                    core_mass_k50,
                    core_recall_k50,
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

        let q_raw = cond_evals["T_raw (1.0)"].q_support_heldout;
        let q_cal = cond_evals["T_NLL (calibrated)"].q_support_heldout;
        let gap_closure_q = if (q_hh_relational - q_raw).abs() > 1e-6 {
            (q_cal - q_raw) / (q_hh_relational - q_raw)
        } else {
            0.0
        };

        let h2a_nll_supported = nll_cal < nll_raw;
        let h2a_jsd_contradicted = jsd_cal > jsd_raw;
        let h2b_nll_confirmed = gap_closure_nll > gap_closure_q;

        println!("  NLL Gap Closure G_NLL = {:.2}%, Relational Gap Closure G_Q = {:.2}%", gap_closure_nll * 100.0, gap_closure_q * 100.0);
        println!("  H2a (NLL Reduction): {}", if h2a_nll_supported { "SUPPORTED" } else { "NOT SUPPORTED" });
        println!("  H2a (JSD Contradiction): {}", if h2a_jsd_contradicted { "CONTRADICTED (JSD Increased)" } else { "REDUCED" });
        println!("  H2b (G_NLL > G_Q): {}", if h2b_nll_confirmed { "CONFIRMED" } else { "REJECTED" });

        e002_model_results.insert(
            m_name.clone(),
            ModelE002Result {
                display_name: m_name.clone(),
                t_nll_fitted: t_nll_mean,
                t_jsd_fitted: t_jsd_mean,
                t_topology_fitted: t_topo_mean,
                gap_closure_nll,
                gap_closure_q,
                h2a_nll_supported,
                h2a_jsd_contradicted,
                h2b_nll_confirmed,
                conditions: cond_evals,
            },
        );
    }

    let total_runtime_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    let summary = E002Summary {
        experiment_id: "E002".to_string(),
        title: "Pointwise Soft-Label Calibration vs. Relational Neighborhood Recovery".to_string(),
        status: "pilot_requires_graph_crossfit_rerun".to_string(),
        e001_artifact_id: manifest.artifact_id.clone(),
        e001_matrix_k10_sha256: expected_k10_sha256.to_string(),
        e001_matrix_k50_sha256: expected_k50_sha256.to_string(),
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
