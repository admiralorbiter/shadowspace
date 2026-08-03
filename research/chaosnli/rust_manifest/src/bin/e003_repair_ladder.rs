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

// ─── Post-Hoc Calibration Transformations ──────────────────────────────

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

#[inline(always)]
fn softmax_vector_scaling(logits: &[f64; 3], v: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    let z0 = logits[0] * v[0] + b[0];
    let z1 = logits[1] * v[1] + b[1];
    let z2 = logits[2] * v[2] + b[2];
    let max_z = z0.max(z1).max(z2);
    let e0 = (z0 - max_z).exp();
    let e1 = (z1 - max_z).exp();
    let e2 = (z2 - max_z).exp();
    let sum_e = e0 + e1 + e2;
    [e0 / sum_e, e1 / sum_e, e2 / sum_e]
}

/// Identifiable 8-Parameter Affine Matrix Scaling (z2' = 0 reference class)
#[inline(always)]
fn softmax_affine_matrix_8param(logits: &[f64; 3], a: &[[f64; 3]; 2], b: &[f64; 2]) -> [f64; 3] {
    let z0 = a[0][0] * logits[0] + a[0][1] * logits[1] + a[0][2] * logits[2] + b[0];
    let z1 = a[1][0] * logits[0] + a[1][1] * logits[1] + a[1][2] * logits[2] + b[1];
    let z2 = 0.0f64;
    let max_z = z0.max(z1).max(z2);
    let e0 = (z0 - max_z).exp();
    let e1 = (z1 - max_z).exp();
    let e2 = (z2 - max_z).exp();
    let sum_e = e0 + e1 + e2;
    [e0 / sum_e, e1 / sum_e, e2 / sum_e]
}

/// Identifiable 8-Parameter Multinomial Dirichlet Calibration (logp transform)
#[inline(always)]
fn dirichlet_calibration_8param(probs: &[f64; 3], a: &[[f64; 3]; 2], b: &[f64; 2]) -> [f64; 3] {
    let l0 = probs[0].max(1e-12).ln();
    let l1 = probs[1].max(1e-12).ln();
    let l2 = probs[2].max(1e-12).ln();
    let z0 = a[0][0] * l0 + a[0][1] * l1 + a[0][2] * l2 + b[0];
    let z1 = a[1][0] * l0 + a[1][1] * l1 + a[1][2] * l2 + b[1];
    let z2 = 0.0f64;
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

fn build_stratified_30groups_empirical(items: &[ItemRecord]) -> Vec<Vec<usize>> {
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

    strata_map.into_values().collect()
}

fn build_stratified_5folds_empirical(items: &[ItemRecord], seed: u64) -> Vec<Vec<usize>> {
    let groups = build_stratified_30groups_empirical(items);
    let mut folds = vec![Vec::new(); 5];
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for mut group in groups {
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

// ─── Optimizers for Vector, Full Affine Matrix, Dirichlet, & Ensembles ────────

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

fn optimize_vector_scaling_with_bias(
    human_probs: &[[f64; 3]],
    logits: &[[f64; 3]],
    indices: &[usize],
) -> ([f64; 3], [f64; 3]) {
    let mut best_v = [1.0f64, 1.0f64, 1.0f64];
    let mut best_b = [0.0f64, 0.0f64, 0.0f64];
    let mut min_loss = 1e9f64;

    let v_grid = [0.3, 0.6, 1.0, 1.6, 2.5, 4.0];
    let b_grid = [-0.4, -0.1, 0.0, 0.1, 0.4];

    for &v0 in &v_grid {
        for &v1 in &v_grid {
            for &v2 in &v_grid {
                for &b0 in &b_grid {
                    for &b1 in &b_grid {
                        for &b2 in &b_grid {
                            let v = [v0, v1, v2];
                            let b = [b0, b1, b2];
                            let mut sum_loss = 0.0f64;
                            for &idx in indices {
                                let q = softmax_vector_scaling(&logits[idx], &v, &b);
                                sum_loss += soft_label_nll_single(&human_probs[idx], &q);
                            }
                            if sum_loss < min_loss {
                                min_loss = sum_loss;
                                best_v = v;
                                best_b = b;
                            }
                        }
                    }
                }
            }
        }
    }
    (best_v, best_b)
}

fn optimize_full_affine_matrix_8param(
    human_probs: &[[f64; 3]],
    logits: &[[f64; 3]],
    indices: &[usize],
) -> ([[f64; 3]; 2], [f64; 2]) {
    let mut best_a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut best_b = [0.0, 0.0];
    let mut min_loss = 1e9f64;

    let d_grid = [0.4, 0.8, 1.2, 1.8, 2.8];
    let o_grid = [-0.3, 0.0, 0.3];
    let b_grid = [-0.2, 0.0, 0.2];

    for &a00 in &d_grid {
        for &a11 in &d_grid {
            for &a01 in &o_grid {
                for &a02 in &o_grid {
                    for &a10 in &o_grid {
                        for &a12 in &o_grid {
                            for &b0 in &b_grid {
                                for &b1 in &b_grid {
                                    let a = [[a00, a01, a02], [a10, a11, a12]];
                                    let b = [b0, b1];
                                    let mut sum_loss = 0.0f64;
                                    for &idx in indices {
                                        let q = softmax_affine_matrix_8param(&logits[idx], &a, &b);
                                        sum_loss += soft_label_nll_single(&human_probs[idx], &q);
                                    }
                                    if sum_loss < min_loss {
                                        min_loss = sum_loss;
                                        best_a = a;
                                        best_b = b;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (best_a, best_b)
}

fn optimize_full_dirichlet_calibration_8param(
    human_probs: &[[f64; 3]],
    logits: &[[f64; 3]],
    indices: &[usize],
) -> ([[f64; 3]; 2], [f64; 2]) {
    let raw_probs: Vec<[f64; 3]> = indices.iter().map(|&idx| softmax_temperature(&logits[idx], 1.0)).collect();
    let mut best_a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut best_b = [0.0, 0.0];
    let mut min_loss = 1e9f64;

    let d_grid = [0.3, 0.6, 1.0, 1.6, 2.4];
    let o_grid = [-0.2, 0.0, 0.2];
    let b_grid = [-0.15, 0.0, 0.15];

    for &a00 in &d_grid {
        for &a11 in &d_grid {
            for &a01 in &o_grid {
                for &a02 in &o_grid {
                    for &a10 in &o_grid {
                        for &a12 in &o_grid {
                            for &b0 in &b_grid {
                                for &b1 in &b_grid {
                                    let a = [[a00, a01, a02], [a10, a11, a12]];
                                    let b = [b0, b1];
                                    let mut sum_loss = 0.0f64;
                                    for (i, &idx) in indices.iter().enumerate() {
                                        let q = dirichlet_calibration_8param(&raw_probs[i], &a, &b);
                                        sum_loss += soft_label_nll_single(&human_probs[idx], &q);
                                    }
                                    if sum_loss < min_loss {
                                        min_loss = sum_loss;
                                        best_a = a;
                                        best_b = b;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (best_a, best_b)
}

fn optimize_convex_ensemble_nll(
    human_probs: &[[f64; 3]],
    model_probs_pool: &[Vec<[f64; 3]>],
    indices: &[usize],
) -> Vec<f64> {
    let m = model_probs_pool.len();
    let mut best_alpha = vec![1.0 / m as f64; m];
    let mut min_loss = 1e9f64;

    let grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0];
    for &a0 in &grid {
        for &a1 in &grid {
            for &a2 in &grid {
                let sum_a = a0 + a1 + a2;
                if sum_a < 1e-6 {
                    continue;
                }
                let alpha = vec![a0 / sum_a, a1 / sum_a, a2 / sum_a];
                let mut sum_loss = 0.0f64;
                for &idx in indices {
                    let mut q = [0.0f64; 3];
                    for k in 0..m {
                        q[0] += alpha[k] * model_probs_pool[k][idx][0];
                        q[1] += alpha[k] * model_probs_pool[k][idx][1];
                        q[2] += alpha[k] * model_probs_pool[k][idx][2];
                    }
                    sum_loss += soft_label_nll_single(&human_probs[idx], &q);
                }
                if sum_loss < min_loss {
                    min_loss = sum_loss;
                    best_alpha = alpha;
                }
            }
        }
    }
    best_alpha
}

fn optimize_ensemble_topology(
    items: &[ItemRecord],
    s_ij_k10: &[f64],
    model_probs_pool: &[Vec<[f64; 3]>],
    indices: &[usize],
    n: usize,
) -> Vec<f64> {
    let m = model_probs_pool.len();
    let mut best_alpha = vec![1.0 / m as f64; m];
    let mut max_excess = -1e9f64;

    let n_tr = indices.len();
    let (snli_tr_indices, mnli_tr_indices) = partition_item_strata(items);

    let grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0];
    for &a0 in &grid {
        for &a1 in &grid {
            for &a2 in &grid {
                let sum_a = a0 + a1 + a2;
                if sum_a < 1e-6 {
                    continue;
                }
                let alpha = vec![a0 / sum_a, a1 / sum_a, a2 / sum_a];
                let mut q_blend = vec![[0.0f64; 3]; n];
                for i in 0..n {
                    for k in 0..m {
                        q_blend[i][0] += alpha[k] * model_probs_pool[k][i][0];
                        q_blend[i][1] += alpha[k] * model_probs_pool[k][i][1];
                        q_blend[i][2] += alpha[k] * model_probs_pool[k][i][2];
                    }
                }

                let dist = build_dist_matrix_seq(&q_blend, n);
                let w = compute_topk_weight_matrix(&dist, n, 10);

                let mut sum_q = 0.0f64;
                for &i_tr in indices {
                    let i_off = i_tr * n;
                    for &j_tr in indices {
                        if j_tr != i_tr {
                            sum_q += w[i_off + j_tr] * s_ij_k10[i_off + j_tr];
                        }
                    }
                }
                let q_sup_cand = sum_q / (n_tr * 10) as f64;

                let sparse_w = extract_nonzero_weights(&w, n);
                let sum_null: f64 = (0..50)
                    .into_par_iter()
                    .map(|b_idx| {
                        let mut null_rng = ChaCha8Rng::seed_from_u64(6060_0000 + b_idx as u64);
                        let mut perm = (0..n).collect::<Vec<_>>();
                        let mut snli_shuffled = snli_tr_indices.clone();
                        let mut mnli_shuffled = mnli_tr_indices.clone();
                        snli_shuffled.shuffle(&mut null_rng);
                        mnli_shuffled.shuffle(&mut null_rng);

                        for (orig_idx, &shuf_idx) in snli_tr_indices.iter().zip(snli_shuffled.iter()) {
                            perm[*orig_idx] = shuf_idx;
                        }
                        for (orig_idx, &shuf_idx) in mnli_tr_indices.iter().zip(mnli_shuffled.iter()) {
                            perm[*orig_idx] = shuf_idx;
                        }

                        let mut s_null = 0.0f64;
                        for &i_tr in indices {
                            let i_perm = perm[i_tr];
                            for &(j, w_val) in &sparse_w[i_tr] {
                                if indices.contains(&j) {
                                    let j_perm = perm[j];
                                    s_null += w_val * s_ij_k10[i_perm * n + j_perm];
                                }
                            }
                        }
                        s_null / (n_tr * 10) as f64
                    })
                    .sum();

                let q_null_cand = sum_null / 50.0;
                let q_excess_cand = q_sup_cand - q_null_cand;

                if q_excess_cand > max_excess {
                    max_excess = q_excess_cand;
                    best_alpha = alpha;
                }
            }
        }
    }
    best_alpha
}

// ─── Output Structs for E003 Summary ────────────────────────────────────────

#[derive(Serialize)]
struct ConditionMetrics {
    nll: f64,
    jsd_bits: f64,
    q_support_oof: f64,
    q_null_oof: f64,
    q_global_excess_oof: f64,
    r_human_recovery_oof: f64,
    graph_turnover_min_oof: f64,
    core_mass_k50_oof: f64,
    core_recall_k50_oof: f64,
    avg_entropy_bits: f64,
}

#[derive(Serialize)]
struct BootstrapCI {
    mean: f64,
    ci_lower_95: f64,
    ci_upper_95: f64,
}

#[derive(Serialize)]
struct LadderLevelResult {
    level_name: String,
    display_name: String,
    gap_closure_nll: f64,
    gap_closure_q: f64,
    bootstrap_delta_nll: BootstrapCI,
    bootstrap_delta_jsd: BootstrapCI,
    bootstrap_delta_q: BootstrapCI,
    bootstrap_delta_gap_closure: BootstrapCI,
    metrics: ConditionMetrics,
}

#[derive(Serialize)]
struct E003Summary {
    experiment_id: String,
    title: String,
    status: String,
    e001_artifact_id: String,
    e001_matrix_k10_sha256: String,
    e001_matrix_k50_sha256: String,
    model_probs_sha256: String,
    human_entropy_floor_nats: f64,
    q_hh_relational: f64,
    ladder_results: HashMap<String, LadderLevelResult>,
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
    println!("   EXPERIMENT E003 — RELATIONAL REPAIR LADDER (RUST PARALLEL ENGINE)");
    println!("   (Rayon Threadpool: {num_threads} worker threads | Unconstrained Multi-Parameter Calibration)");
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

    let manifest_file = File::open(&manifest_k10_path).expect("Failed to open E001 k10 manifest");
    let manifest: ArtifactManifest = serde_json::from_reader(BufReader::new(manifest_file)).expect("Failed to parse E001 manifest");

    let items = load_items(&items_path);
    let n = items.len();

    let object_ids: Vec<String> = items.iter().map(|item| item.object_id.clone()).collect();
    let object_ids_bytes = serde_json::to_vec(&object_ids).unwrap();
    let actual_object_ids_sha256 = compute_bytes_sha256(&object_ids_bytes);
    assert_eq!(actual_object_ids_sha256, expected_object_ids_sha256, "Item ordering SHA-256 mismatch!");

    let s_ij_k10 = load_and_verify_matrix_f64(&bin_k10_path, expected_k10_sha256, n * n);
    let s_ij_k50 = load_and_verify_matrix_f64(&bin_k50_path, expected_k50_sha256, n * n);

    let (snli_indices, mnli_indices) = partition_item_strata(&items);

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

    let raw_models = load_models(&models_path);
    let mut model_logits: HashMap<String, Vec<[f64; 3]>> = HashMap::new();
    for (m_name, m_probs) in &raw_models {
        let logits: Vec<[f64; 3]> = m_probs
            .iter()
            .map(|p| [p[0].max(1e-12).ln(), p[1].max(1e-12).ln(), p[2].max(1e-12).ln()])
            .collect();
        model_logits.insert(m_name.clone(), logits);
    }

    let folds = build_stratified_5folds_empirical(&items, 20260803);
    let strata_30groups = build_stratified_30groups_empirical(&items);

    let mut item_fold_map = vec![0usize; n];
    for fold_idx in 0..5 {
        for &idx in &folds[fold_idx] {
            item_fold_map[idx] = fold_idx;
        }
    }

    let lead_model_name = "bart-large";
    let lead_logits = &model_logits[lead_model_name];
    let raw_probs_all: Vec<[f64; 3]> = (0..n).map(|i| softmax_temperature(&lead_logits[i], 1.0)).collect();

    let ensemble_pool = vec![
        raw_models["bart-large"].clone(),
        raw_models["roberta-large"].clone(),
        raw_models["xlnet-large"].clone(),
    ];

    let ladder_names = vec![
        ("Level 0: Raw Model Baseline", "Level 0: Raw Model Baseline"),
        ("Level 1: Scalar Temperature", "Level 1: Global Isotropic Scalar Temperature"),
        ("Level 2: Vector Scaling + Bias", "Level 2: Class-Wise Vector Scaling + Bias"),
        ("Level 3: Full Affine Matrix", "Level 3: Full 8-Parameter Affine Matrix Scaling"),
        ("Level 4: Full Dirichlet Calibration", "Level 4: Full 8-Parameter Multinomial Dirichlet Calibration"),
        ("Level 5a: Equal Weight Ensemble", "Level 5a: Equal-Weight Multi-Model Ensemble"),
        ("Level 5b: Convex NLL Ensemble", "Level 5b: Convex NLL-Optimized Simplex Ensemble"),
        ("Level 6a: Topology Ensemble", "Level 6a: Topology-Optimized Simplex Ensemble"),
    ];

    let mut ladder_results = HashMap::new();

    // Baseline O_i_raw and O_i_null_raw pre-computation using actual 10,000 stratified perms
    let dist_raw_ref = build_dist_matrix_seq(&raw_probs_all, n);
    let w_raw_ref = compute_topk_weight_matrix(&dist_raw_ref, n, 10);
    let sparse_w_raw = extract_nonzero_weights(&w_raw_ref, n);

    let mut o_i_raw = vec![0.0f64; n];
    let mut o_i_null_raw = vec![0.0f64; n];

    for i in 0..n {
        let i_off = i * n;
        let mut sum_s = 0.0f64;
        for j in 0..n {
            if j != i {
                sum_s += w_raw_ref[i_off + j] * s_ij_k10[i_off + j];
            }
        }
        o_i_raw[i] = sum_s / 10.0;
    }

    let n_null_raw = 10_000;
    let (item_null_scores_raw, _): (Vec<Vec<f64>>, Vec<f64>) = (0..n_null_raw)
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

            let mut per_item_null = vec![0.0f64; n];
            let mut sum_null_f = 0.0f64;
            for i in 0..n {
                let i_perm = perm[i];
                let mut local_n = 0.0f64;
                for &(j, w_val) in &sparse_w_raw[i] {
                    let j_perm = perm[j];
                    local_n += w_val * s_ij_k10[i_perm * n + j_perm];
                }
                per_item_null[i] = local_n / 10.0;
                sum_null_f += local_n;
            }
            (per_item_null, sum_null_f / (n * 10) as f64)
        })
        .unzip();

    for b_idx in 0..n_null_raw {
        for i in 0..n {
            o_i_null_raw[i] += item_null_scores_raw[b_idx][i] / n_null_raw as f64;
        }
    }

    let q_raw_oof_baseline = o_i_raw.iter().sum::<f64>() / n as f64;
    let q_null_oof_baseline = o_i_null_raw.iter().sum::<f64>() / n as f64;
    let r_raw_baseline = (q_raw_oof_baseline - q_null_oof_baseline) / (q_hh_relational - q_null_oof_baseline);

    for (level_key, display_name) in ladder_names {
        println!("\n--- Evaluating Repair Ladder Step: {display_name} ---");

        let mut f_probs_all = vec![vec![[0.0f64; 3]; n]; 5];

        for fold_idx in 0..5 {
            let test_indices = &folds[fold_idx];
            let mut train_indices = Vec::with_capacity(n - test_indices.len());
            for i in 0..n {
                if !test_indices.contains(&i) {
                    train_indices.push(i);
                }
            }

            let probs_f = match level_key {
                "Level 0: Raw Model Baseline" => raw_probs_all.clone(),
                "Level 1: Scalar Temperature" => {
                    let t_opt = optimize_temperature_nll(&human_probs, lead_logits, &train_indices);
                    (0..n).map(|i| softmax_temperature(&lead_logits[i], t_opt)).collect()
                }
                "Level 2: Vector Scaling + Bias" => {
                    let (v_opt, b_opt) = optimize_vector_scaling_with_bias(&human_probs, lead_logits, &train_indices);
                    (0..n).map(|i| softmax_vector_scaling(&lead_logits[i], &v_opt, &b_opt)).collect()
                }
                "Level 3: Full Affine Matrix" => {
                    let (a_opt, b_opt) = optimize_full_affine_matrix_8param(&human_probs, lead_logits, &train_indices);
                    (0..n).map(|i| softmax_affine_matrix_8param(&lead_logits[i], &a_opt, &b_opt)).collect()
                }
                "Level 4: Full Dirichlet Calibration" => {
                    let (a_opt, b_opt) = optimize_full_dirichlet_calibration_8param(&human_probs, lead_logits, &train_indices);
                    (0..n).map(|i| dirichlet_calibration_8param(&raw_probs_all[i], &a_opt, &b_opt)).collect()
                }
                "Level 5a: Equal Weight Ensemble" => {
                    (0..n).map(|i| [
                        (ensemble_pool[0][i][0] + ensemble_pool[1][i][0] + ensemble_pool[2][i][0]) / 3.0,
                        (ensemble_pool[0][i][1] + ensemble_pool[1][i][1] + ensemble_pool[2][i][1]) / 3.0,
                        (ensemble_pool[0][i][2] + ensemble_pool[1][i][2] + ensemble_pool[2][i][2]) / 3.0,
                    ]).collect()
                }
                "Level 5b: Convex NLL Ensemble" => {
                    let alpha_opt = optimize_convex_ensemble_nll(&human_probs, &ensemble_pool, &train_indices);
                    (0..n).map(|i| [
                        alpha_opt[0] * ensemble_pool[0][i][0] + alpha_opt[1] * ensemble_pool[1][i][0] + alpha_opt[2] * ensemble_pool[2][i][0],
                        alpha_opt[0] * ensemble_pool[0][i][1] + alpha_opt[1] * ensemble_pool[1][i][1] + alpha_opt[2] * ensemble_pool[2][i][2],
                        alpha_opt[0] * ensemble_pool[0][i][2] + alpha_opt[1] * ensemble_pool[1][i][2] + alpha_opt[2] * ensemble_pool[2][i][2],
                    ]).collect()
                }
                "Level 6a: Topology Ensemble" => {
                    let alpha_topo = optimize_ensemble_topology(&items, &s_ij_k10, &ensemble_pool, &train_indices, n);
                    (0..n).map(|i| [
                        alpha_topo[0] * ensemble_pool[0][i][0] + alpha_topo[1] * ensemble_pool[1][i][0] + alpha_topo[2] * ensemble_pool[2][i][0],
                        alpha_topo[0] * ensemble_pool[0][i][1] + alpha_topo[1] * ensemble_pool[1][i][1] + alpha_topo[2] * ensemble_pool[2][i][1],
                        alpha_topo[0] * ensemble_pool[0][i][2] + alpha_topo[1] * ensemble_pool[1][i][2] + alpha_topo[2] * ensemble_pool[2][i][2],
                    ]).collect()
                }
                _ => raw_probs_all.clone(),
            };

            f_probs_all[fold_idx] = probs_f;
        }

        // Out-of-Fold Evaluation
        let mut sum_oof_nll = 0.0f64;
        let mut sum_oof_jsd = 0.0f64;
        let mut sum_oof_q_sup = 0.0f64;
        let mut sum_oof_overlap_min = 0.0f64;
        let mut sum_oof_core_mass_50 = 0.0f64;
        let mut c_tau50_k50 = 0usize;

        let mut item_support_observed = vec![0.0f64; n];
        let mut sparse_w10_folds = Vec::with_capacity(5);

        for fold_idx in 0..5 {
            let test_indices = &folds[fold_idx];
            let q_probs_f = &f_probs_all[fold_idx];

            for &idx in test_indices {
                sum_oof_nll += soft_label_nll_single(&human_probs[idx], &q_probs_f[idx]);
                sum_oof_jsd += jsd_divergence_bits(&human_probs[idx], &q_probs_f[idx]);
            }

            let dist_f = build_dist_matrix_seq(q_probs_f, n);
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
                    }
                }
                item_support_observed[i_test] = local_obs;
            }

            let dist_raw = build_dist_matrix_seq(&raw_probs_all, n);
            let w_raw10 = compute_topk_weight_matrix(&dist_raw, n, 10);
            for &i_test in test_indices {
                let i_off = i_test * n;
                for j in 0..n {
                    if j != i_test {
                        sum_oof_overlap_min += w_raw10[i_off + j].min(w_m10[i_off + j]);
                    }
                }
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

            sparse_w10_folds.push(sparse_w10);
        }

        let nll_val = sum_oof_nll / n as f64;
        let jsd_val = sum_oof_jsd / n as f64;
        let q_support_oof = sum_oof_q_sup / (n * 10) as f64;
        let graph_turnover_min_oof = (1.0 - (sum_oof_overlap_min / (n * 10) as f64)).max(0.0);
        let core_mass_k50_oof = sum_oof_core_mass_50 / (n * 50) as f64;
        let core_recall_k50_oof = sum_oof_core_mass_50 / c_tau50_k50.max(1) as f64;

        // 10,000 Stratified Permutation Nulls with per-item null accumulation
        let n_null = 10_000;
        let (item_null_scores, null_scores): (Vec<Vec<f64>>, Vec<f64>) = (0..n_null)
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

                let mut per_item_null = vec![0.0f64; n];
                let mut sum_null_f = 0.0f64;
                for fold_idx in 0..5 {
                    let test_indices = &folds[fold_idx];
                    let sparse_w10 = &sparse_w10_folds[fold_idx];
                    for &i_test in test_indices {
                        let i_perm = perm[i_test];
                        let mut local_n = 0.0f64;
                        for &(j, w) in &sparse_w10[i_test] {
                            let j_perm = perm[j];
                            local_n += w * s_ij_k10[i_perm * n + j_perm];
                        }
                        per_item_null[i_test] = local_n / 10.0;
                        sum_null_f += local_n;
                    }
                }
                (per_item_null, sum_null_f / (n * 10) as f64)
            })
            .unzip();

        let q_null_oof = null_scores.iter().sum::<f64>() / n_null as f64;
        let q_global_excess_oof = q_support_oof - q_null_oof;

        let mut item_support_null = vec![0.0f64; n];
        for b_idx in 0..n_null {
            for i in 0..n {
                item_support_null[i] += item_null_scores[b_idx][i] / n_null as f64;
            }
        }

        let r_human_recovery_oof = if (q_hh_relational - q_null_oof).abs() > 1e-12 {
            (q_support_oof - q_null_oof) / (q_hh_relational - q_null_oof)
        } else {
            0.0
        };

        let nll_raw = 0.8626835793f64;
        let gap_closure_nll = if (nll_raw - human_entropy_floor_nats).abs() > 1e-6 {
            (nll_raw - nll_val) / (nll_raw - human_entropy_floor_nats)
        } else {
            0.0
        };

        let gap_closure_q = if (1.0 - r_raw_baseline).abs() > 1e-6 {
            (r_human_recovery_oof - r_raw_baseline) / (1.0 - r_raw_baseline)
        } else {
            0.0
        };

        // 1,000 Stratified Item-Level Paired Bootstrap Iterations
        let n_boot = 1000;
        let mut boot_delta_nll = Vec::with_capacity(n_boot);
        let mut boot_delta_jsd = Vec::with_capacity(n_boot);
        let mut boot_delta_q = Vec::with_capacity(n_boot);
        let mut boot_delta_gc = Vec::with_capacity(n_boot);

        for boot_idx in 0..n_boot {
            let mut boot_rng = ChaCha8Rng::seed_from_u64(7070_0000 + boot_idx as u64);
            let mut sampled_indices = Vec::with_capacity(n);
            for group in &strata_30groups {
                for _ in 0..group.len() {
                    let pick = group[boot_rng.gen_range(0..group.len())];
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
                let q_raw = raw_probs_all[idx];
                let q_cal = f_probs_all[f_idx][idx];

                b_nll_raw += soft_label_nll_single(&human_probs[idx], &q_raw);
                b_nll_cal += soft_label_nll_single(&human_probs[idx], &q_cal);

                b_jsd_raw += jsd_divergence_bits(&human_probs[idx], &q_raw);
                b_jsd_cal += jsd_divergence_bits(&human_probs[idx], &q_cal);

                b_h_floor += human_entropy_nats(&human_probs[idx]);

                b_obs_raw += o_i_raw[idx];
                b_null_raw += o_i_null_raw[idx];
                b_obs_cal += item_support_observed[idx];
                b_null_cal += item_support_null[idx];
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

        let mut sum_ent = 0.0f64;
        for i in 0..n {
            let p = &f_probs_all[item_fold_map[i]][i];
            let ent = -(if p[0] > 1e-12 { p[0] * p[0].log2() } else { 0.0 })
                - (if p[1] > 1e-12 { p[1] * p[1].log2() } else { 0.0 })
                - (if p[2] > 1e-12 { p[2] * p[2].log2() } else { 0.0 });
            sum_ent += ent;
        }
        let avg_entropy_bits = sum_ent / n as f64;

        let level_res = LadderLevelResult {
            level_name: level_key.to_string(),
            display_name: display_name.to_string(),
            gap_closure_nll,
            gap_closure_q,
            bootstrap_delta_nll,
            bootstrap_delta_jsd,
            bootstrap_delta_q,
            bootstrap_delta_gap_closure,
            metrics: ConditionMetrics {
                nll: nll_val,
                jsd_bits: jsd_val,
                q_support_oof,
                q_null_oof,
                q_global_excess_oof,
                r_human_recovery_oof,
                graph_turnover_min_oof,
                core_mass_k50_oof,
                core_recall_k50_oof,
                avg_entropy_bits,
            },
        };

        println!("  NLL = {:.4}, G_NLL = {:.2}%, Q_support = {:.5}, G_Q = {:.2}%", nll_val, gap_closure_nll * 100.0, q_support_oof, gap_closure_q * 100.0);

        ladder_results.insert(level_key.to_string(), level_res);
    }

    let total_runtime_ms = t_start.elapsed().as_secs_f64() * 1000.0;

    let summary = E003Summary {
        experiment_id: "E003".to_string(),
        title: "Relational Repair Capacity of Flexible Post-Hoc Transformations & Ensembling".to_string(),
        status: "complete_publication_grade".to_string(),
        e001_artifact_id: manifest.artifact_id.clone(),
        e001_matrix_k10_sha256: expected_k10_sha256.to_string(),
        e001_matrix_k50_sha256: expected_k50_sha256.to_string(),
        model_probs_sha256,
        human_entropy_floor_nats,
        q_hh_relational,
        ladder_results,
        total_runtime_ms,
    };

    let summary_dir = workspace.join("research/chaosnli/lab/summaries");
    create_dir_all(&summary_dir).unwrap();
    let summary_path = summary_dir.join("E003_summary.json");
    let file = File::create(&summary_path).unwrap();
    serde_json::to_writer_pretty(file, &summary).unwrap();

    println!("\n=========================================================================");
    println!("   EXPERIMENT E003 COMPLETE IN {:.2}s", total_runtime_ms / 1000.0);
    println!("   Summary saved to {}", summary_path.display());
    println!("=========================================================================");
}
