use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::Dirichlet;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

// ─── Data structures ────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ItemRecord {
    object_id: String,
    source_pair_id: String,
    source_dataset: Option<String>,
    human_count_entailment: i32,
    human_count_neutral: i32,
    human_count_contradiction: i32,
    human_p_entailment: f64,
    human_p_neutral: f64,
    human_p_contradiction: f64,
}

#[derive(Serialize)]
struct ModelPairedResult {
    q_paired_hm_mean: f64,
    delta_m_mean: f64,
    delta_m_95ci: [f64; 2],
    replicates_gt_zero: String,
    q_fixed_reference: f64,
}

#[derive(Serialize)]
struct PairedEstimandResults {
    estimand: String,
    description: String,
    n_pairs: usize,
    n_bootstrap: usize,
    k: usize,
    hh100_bootstrap_mean: f64,
    hh100_bootstrap_95ci: [f64; 2],
    models: HashMap<String, ModelPairedResult>,
    total_runtime_ms: f64,
}

// ─── Core geometry ───────────────────────────────────────────────────────────

#[inline(always)]
fn hellinger(p: &[f64], q: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for i in 0..p.len() {
        let d = p[i].sqrt() - q[i].sqrt();
        s += d * d;
    }
    (0.5 * s).sqrt()
}

/// Build full NxN Hellinger distance matrix — PARALLEL (top-level only, not inside par_iter)
fn build_dist_matrix_par(probs: &[[f64; 3]], n: usize) -> Vec<f64> {
    (0..n)
        .into_par_iter()
        .flat_map(|i| {
            let mut row = vec![0.0f64; n];
            for j in 0..n {
                row[j] = hellinger(&probs[i], &probs[j]);
            }
            row
        })
        .collect()
}

/// Build full NxN Hellinger distance matrix — SEQUENTIAL (safe inside par_iter)
fn build_dist_matrix_seq(probs: &[[f64; 3]], n: usize) -> Vec<f64> {
    let mut dist = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            dist[i * n + j] = hellinger(&probs[i], &probs[j]);
        }
    }
    dist
}

/// Compute per-item soft Q_NX overlap — SEQUENTIAL (safe inside outer par_iter)
fn soft_qnx_per_item_seq(dist_a: &[f64], dist_b: &[f64], n: usize, k: usize) -> Vec<f64> {
    const ATOL: f64 = 1e-7;
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        let row_a = &dist_a[i * n..(i + 1) * n];
        let row_b = &dist_b[i * n..(i + 1) * n];

        let mut sorted_a: Vec<f64> = row_a.iter().enumerate()
            .filter(|&(j, _)| j != i).map(|(_, &d)| d).collect();
        sorted_a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let k_dist_a = sorted_a[k - 1];

        let mut sorted_b: Vec<f64> = row_b.iter().enumerate()
            .filter(|&(j, _)| j != i).map(|(_, &d)| d).collect();
        sorted_b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let k_dist_b = sorted_b[k - 1];

        // Pre-compute tie fractions for this row
        let n_closer_a = row_a.iter().enumerate()
            .filter(|&(j, &d)| j != i && d < k_dist_a - ATOL).count();
        let n_tied_a = row_a.iter().enumerate()
            .filter(|&(j, &d)| j != i && (d - k_dist_a).abs() <= ATOL).count();
        let frac_a = if n_tied_a > 0 { (k as f64 - n_closer_a as f64) / n_tied_a as f64 } else { 0.0 };

        let n_closer_b = row_b.iter().enumerate()
            .filter(|&(j, &d)| j != i && d < k_dist_b - ATOL).count();
        let n_tied_b = row_b.iter().enumerate()
            .filter(|&(j, &d)| j != i && (d - k_dist_b).abs() <= ATOL).count();
        let frac_b = if n_tied_b > 0 { (k as f64 - n_closer_b as f64) / n_tied_b as f64 } else { 0.0 };

        let mut sum_min = 0.0f64;
        for j in 0..n {
            if j == i { continue; }
            let d_a = row_a[j];
            let d_b = row_b[j];
            let w_a = if d_a < k_dist_a - ATOL { 1.0 }
                      else if (d_a - k_dist_a).abs() <= ATOL { frac_a }
                      else { 0.0 };
            let w_b = if d_b < k_dist_b - ATOL { 1.0 }
                      else if (d_b - k_dist_b).abs() <= ATOL { frac_b }
                      else { 0.0 };
            sum_min += w_a.min(w_b);
        }
        result[i] = sum_min / k as f64;
    }
    result
}

/// Compute per-item soft Q_NX overlap — PARALLEL version (top-level only)
#[allow(dead_code)]
fn soft_qnx_per_item(dist_a: &[f64], dist_b: &[f64], n: usize, k: usize) -> Vec<f64> {
    const ATOL: f64 = 1e-7;
    (0..n)
        .into_par_iter()
        .map(|i| {
            // ── Weights for A[i] ──────────────────────────────────────────
            let row_a = &dist_a[i * n..(i + 1) * n];
            let mut sorted_a: Vec<f64> = row_a
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &d)| d)
                .collect();
            sorted_a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            let k_dist_a = sorted_a[k - 1];

            // ── Weights for B[i] ──────────────────────────────────────────
            let row_b = &dist_b[i * n..(i + 1) * n];
            let mut sorted_b: Vec<f64> = row_b
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &d)| d)
                .collect();
            sorted_b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            let k_dist_b = sorted_b[k - 1];

            // ── Compute min(w_a[i][j], w_b[i][j]) for each j ──────────────
            let mut sum_min = 0.0f64;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let d_a = dist_a[i * n + j];
                let d_b = dist_b[i * n + j];

                let w_a = if d_a < k_dist_a - ATOL {
                    1.0
                } else if (d_a - k_dist_a).abs() <= ATOL {
                    // Compute n_closer_a and n_tied_a
                    let n_closer_a = row_a
                        .iter()
                        .enumerate()
                        .filter(|&(jj, &d)| jj != i && d < k_dist_a - ATOL)
                        .count();
                    let n_tied_a = row_a
                        .iter()
                        .enumerate()
                        .filter(|&(jj, &d)| jj != i && (d - k_dist_a).abs() <= ATOL)
                        .count();
                    let r_a = k as f64 - n_closer_a as f64;
                    if n_tied_a > 0 { r_a / n_tied_a as f64 } else { 0.0 }
                } else {
                    0.0
                };

                let w_b = if d_b < k_dist_b - ATOL {
                    1.0
                } else if (d_b - k_dist_b).abs() <= ATOL {
                    let n_closer_b = row_b
                        .iter()
                        .enumerate()
                        .filter(|&(jj, &d)| jj != i && d < k_dist_b - ATOL)
                        .count();
                    let n_tied_b = row_b
                        .iter()
                        .enumerate()
                        .filter(|&(jj, &d)| jj != i && (d - k_dist_b).abs() <= ATOL)
                        .count();
                    let r_b = k as f64 - n_closer_b as f64;
                    if n_tied_b > 0 { r_b / n_tied_b as f64 } else { 0.0 }
                } else {
                    0.0
                };

                sum_min += w_a.min(w_b);
            }
            sum_min / k as f64
        })
        .collect()
}

/// Fast variant with pre-computed A weights — SEQUENTIAL (safe inside par_iter)
fn soft_qnx_with_wa_seq(
    weights_a: &[f64],
    dist_b: &[f64],
    n: usize,
    k: usize,
) -> Vec<f64> {
    const ATOL: f64 = 1e-7;
    let mut result = vec![0.0f64; n];
    for i in 0..n {
        let row_b = &dist_b[i * n..(i + 1) * n];
        let mut sorted_b: Vec<f64> = row_b.iter().enumerate()
            .filter(|&(j, _)| j != i).map(|(_, &d)| d).collect();
        sorted_b.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let k_dist_b = sorted_b[k - 1];
        let n_closer_b = row_b.iter().enumerate()
            .filter(|&(j, &d)| j != i && d < k_dist_b - ATOL).count();
        let n_tied_b = row_b.iter().enumerate()
            .filter(|&(j, &d)| j != i && (d - k_dist_b).abs() <= ATOL).count();
        let frac_b = if n_tied_b > 0 { (k as f64 - n_closer_b as f64) / n_tied_b as f64 } else { 0.0 };

        let mut sum_min = 0.0f64;
        for j in 0..n {
            if j == i { continue; }
            let w_a = weights_a[i * n + j];
            let d_b = row_b[j];
            let w_b = if d_b < k_dist_b - ATOL { 1.0 }
                      else if (d_b - k_dist_b).abs() <= ATOL { frac_b }
                      else { 0.0 };
            sum_min += w_a.min(w_b);
        }
        result[i] = sum_min / k as f64;
    }
    result
}

/// Fast variant with pre-computed A weights — PARALLEL (top-level only)
#[allow(dead_code)]
fn soft_qnx_per_item_with_wa(
    weights_a: &[f64],
    dist_b: &[f64],
    n: usize,
    k: usize,
) -> Vec<f64> {
    const ATOL: f64 = 1e-7;
    (0..n)
        .into_par_iter()
        .map(|i| {
            // B weights for row i
            let row_b = &dist_b[i * n..(i + 1) * n];
            let mut sorted_b: Vec<f64> = row_b
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &d)| d)
                .collect();
            sorted_b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            let k_dist_b = sorted_b[k - 1];

            let n_closer_b = row_b
                .iter()
                .enumerate()
                .filter(|&(j, &d)| j != i && d < k_dist_b - ATOL)
                .count();
            let n_tied_b = row_b
                .iter()
                .enumerate()
                .filter(|&(j, &d)| j != i && (d - k_dist_b).abs() <= ATOL)
                .count();
            let r_b = k as f64 - n_closer_b as f64;
            let frac_b = if n_tied_b > 0 { r_b / n_tied_b as f64 } else { 0.0 };

            let mut sum_min = 0.0f64;
            for j in 0..n {
                if j == i { continue; }
                let w_a = weights_a[i * n + j];
                let d_b = row_b[j];
                let w_b = if d_b < k_dist_b - ATOL {
                    1.0
                } else if (d_b - k_dist_b).abs() <= ATOL {
                    frac_b
                } else {
                    0.0
                };
                sum_min += w_a.min(w_b);
            }
            sum_min / k as f64
        })
        .collect()
}

/// Compute the full n×n weight matrix for a distance matrix (for model graphs,
/// computed once and reused across all 500 pairs).
fn build_weight_matrix(dist: &[f64], n: usize, k: usize) -> Vec<f64> {
    const ATOL: f64 = 1e-7;
    let mut weights = vec![0.0f64; n * n];
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &dist[i * n..(i + 1) * n];
            let mut sorted_d: Vec<f64> = row
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, &d)| d)
                .collect();
            sorted_d.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let k_dist = sorted_d[k - 1];

            let n_closer = row
                .iter()
                .enumerate()
                .filter(|&(j, &d)| j != i && d < k_dist - ATOL)
                .count();
            let n_tied = row
                .iter()
                .enumerate()
                .filter(|&(j, &d)| j != i && (d - k_dist).abs() <= ATOL)
                .count();
            let r_i = k as f64 - n_closer as f64;
            let frac = if n_tied > 0 { r_i / n_tied as f64 } else { 0.0 };

            let mut w_row = vec![0.0f64; n];
            for j in 0..n {
                if j == i { continue; }
                let d = row[j];
                w_row[j] = if d < k_dist - ATOL {
                    1.0
                } else if (d - k_dist).abs() <= ATOL {
                    frac
                } else {
                    0.0
                };
            }
            w_row
        })
        .collect();

    for (i, row) in rows.iter().enumerate() {
        weights[i * n..(i + 1) * n].copy_from_slice(row);
    }
    weights
}

// ─── Posterior-predictive pair generation ────────────────────────────────────

fn generate_posterior_pair(
    counts: &[[i32; 3]],
    n_votes: usize,
    alpha_prior: f64,
    seed: u64,
    n: usize,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut p1 = vec![[0.0f64; 3]; n];
    let mut p2 = vec![[0.0f64; 3]; n];

    for i in 0..n {
        let alpha: Vec<f64> = counts[i]
            .iter()
            .map(|&c| c as f64 + alpha_prior)
            .collect();
        let dirichlet = Dirichlet::new(&alpha).unwrap();
        let theta: Vec<f64> = dirichlet.sample(&mut rng);

        // Draw two multinomial samples
        let mut c1 = [0u64; 3];
        let mut c2 = [0u64; 3];
        for _ in 0..n_votes {
            let u1: f64 = rng.gen_range(0.0..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            let mut cum = 0.0;
            let mut ch1 = 2usize;
            let mut ch2 = 2usize;
            for cat in 0..3 {
                cum += theta[cat];
                if u1 < cum && ch1 == 2 { ch1 = cat; }
                if u2 < cum && ch2 == 2 { ch2 = cat; }
            }
            // Reset cum for second loop
            let mut cum1 = 0.0;
            let mut cum2 = 0.0;
            ch1 = 2; ch2 = 2;
            for cat in 0..3 {
                cum1 += theta[cat];
                cum2 += theta[cat];
                if u1 < cum1 && ch1 == 2 { ch1 = cat; }
                if u2 < cum2 && ch2 == 2 { ch2 = cat; }
            }
            c1[ch1] += 1;
            c2[ch2] += 1;
        }
        for cat in 0..3 {
            p1[i][cat] = c1[cat] as f64 / n_votes as f64;
            p2[i][cat] = c2[cat] as f64 / n_votes as f64;
        }
    }
    (p1, p2)
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let t0 = Instant::now();
    println!("=========================================================================");
    println!("   PAIRED ESTIMAND ENGINE — RUST/RAYON IMPLEMENTATION");
    println!("=========================================================================\n");

    const K: usize = 10;
    const N_PAIRS: usize = 500;
    const N_BOOT: usize = 1000;
    const ALPHA_PRIOR: f64 = 0.5;

    // ── Load canonical items ────────────────────────────────────────────────
    let item_path = "c:/Users/admir/Github/shadowspace/data/chaosnli/processed/canonical_items_posterior.json";
    let file = File::open(item_path).expect("Failed to open canonical items JSON");
    let reader = BufReader::new(file);
    let items: Vec<ItemRecord> = serde_json::from_reader(reader).expect("Failed to parse items");
    let n = items.len();
    println!("Loaded {} items in {:?}", n, t0.elapsed());

    let probs_human: Vec<[f64; 3]> = items.iter()
        .map(|it| [it.human_p_entailment, it.human_p_neutral, it.human_p_contradiction])
        .collect();
    let counts: Vec<[i32; 3]> = items.iter()
        .map(|it| [it.human_count_entailment, it.human_count_neutral, it.human_count_contradiction])
        .collect();

    // Stratification indices
    let snli_idx: Vec<usize> = items.iter().enumerate()
        .filter(|(_, it)| it.source_dataset.as_deref().map(|s| s.contains("snli")).unwrap_or(false)
            || (!it.object_id.contains("mnli")))
        .map(|(i, _)| i)
        .collect();
    // Simple split: first 1514 = SNLI, rest = MNLI (matching Python)
    let snli_indices: Vec<usize> = (0..1514).collect();
    let mnli_indices: Vec<usize> = (1514..n).collect();

    // ── Load model probabilities ────────────────────────────────────────────
    let model_path = "c:/Users/admir/Github/shadowspace/research/chaosnli/rust_manifest/model_probs.json";
    let mfile = File::open(model_path).expect("Failed to open model_probs.json");
    let mreader = BufReader::new(mfile);
    let model_probs_raw: HashMap<String, Vec<[f64; 3]>> =
        serde_json::from_reader(mreader).expect("Failed to parse model probs");

    let mut model_keys: Vec<String> = model_probs_raw.keys().cloned().collect();
    model_keys.sort();
    println!("Loaded {} models: {:?}", model_keys.len(), model_keys);

    // ── Build empirical human dist matrix + weight matrix ──────────────────
    let t_dist = Instant::now();
    let d_emp = build_dist_matrix_par(&probs_human, n);
    println!("Built empirical dist matrix in {:?}", t_dist.elapsed());

    // ── Build each model's weight matrix (once, reused across all 500 pairs) ─
    println!("\nBuilding model weight matrices...");
    let t_model = Instant::now();
    let mut model_weights: HashMap<String, Vec<f64>> = HashMap::new();
    let mut model_fixed_q: HashMap<String, f64> = HashMap::new();

    for m_key in &model_keys {
        let q_m = &model_probs_raw[m_key];
        let d_m = build_dist_matrix_par(q_m, n);
        let w_m = build_weight_matrix(&d_m, n, K);

        // Fixed reference Q (model vs observed)
        let w_emp_full = build_weight_matrix(&d_emp, n, K);
        let overlap: Vec<f64> = (0..n).map(|i| {
            let mut s = 0.0f64;
            for j in 0..n {
                s += w_emp_full[i * n + j].min(w_m[i * n + j]);
            }
            s / K as f64
        }).collect();
        let q_fixed = overlap.iter().sum::<f64>() / n as f64;
        println!("  {}: fixed Q = {:.5}", m_key, q_fixed);
        model_fixed_q.insert(m_key.clone(), q_fixed);
        model_weights.insert(m_key.clone(), w_m);
    }
    println!("Model weight matrices built in {:?}", t_model.elapsed());

    // ── Pre-compute 500 posterior pairs: per-item overlaps ──────────────────
    // For each pair s: hh_oi[s] = per-item Q(H1, H2) — [n floats]
    // For each model m, pair s: mh1_oi[m][s] = per-item Q(G_m, H1), mh2_oi[m][s] = per-item Q(G_m, H2)
    println!("\nPre-computing {} posterior-predictive pairs...", N_PAIRS);
    let t_pairs = Instant::now();

    // Compute all 500 pairs in parallel — inner calls MUST be sequential to avoid Rayon deadlock
    let pair_results: Vec<(Vec<f64>, HashMap<String, (Vec<f64>, Vec<f64>)>)> = (0..N_PAIRS)
        .into_par_iter()
        .map(|s| {
            let (p1, p2) = generate_posterior_pair(&counts, 100, ALPHA_PRIOR, s as u64, n);
            // Use _seq variants: we are already inside a par_iter
            let d1 = build_dist_matrix_seq(&p1, n);
            let d2 = build_dist_matrix_seq(&p2, n);

            // HH overlap per item (sequential)
            let hh_item = soft_qnx_per_item_seq(&d1, &d2, n, K);

            // Model vs H1 and H2 per item (sequential, using pre-built weight matrices)
            let mut model_items: HashMap<String, (Vec<f64>, Vec<f64>)> = HashMap::new();
            for m_key in model_keys.iter() {
                let w_m = model_weights[m_key].as_slice();
                let mh1 = soft_qnx_with_wa_seq(w_m, &d1, n, K);
                let mh2 = soft_qnx_with_wa_seq(w_m, &d2, n, K);
                model_items.insert(m_key.clone(), (mh1, mh2));
            }

            (hh_item, model_items)
        })
        .collect();

    println!("All {} pairs computed in {:?}", N_PAIRS, t_pairs.elapsed());

    // ── Bootstrap ────────────────────────────────────────────────────────────
    println!("\nRunning {} stratified bootstrap replicates...", N_BOOT);
    let t_boot = Instant::now();

    let mut q_hhs_boot: Vec<f64> = Vec::with_capacity(N_BOOT);
    let mut q_hms_paired: HashMap<String, Vec<f64>> = model_keys.iter()
        .map(|k| (k.clone(), Vec::with_capacity(N_BOOT))).collect();
    let mut delta_ms: HashMap<String, Vec<f64>> = model_keys.iter()
        .map(|k| (k.clone(), Vec::with_capacity(N_BOOT))).collect();

    let mut rng_boot = ChaCha8Rng::seed_from_u64(42);

    for b in 0..N_BOOT {
        let s = b % N_PAIRS;

        // Stratified resample
        let b_snli: Vec<usize> = (0..snli_indices.len())
            .map(|_| snli_indices[rng_boot.gen_range(0..snli_indices.len())])
            .collect();
        let b_mnli: Vec<usize> = (0..mnli_indices.len())
            .map(|_| mnli_indices[rng_boot.gen_range(0..mnli_indices.len())])
            .collect();
        let b_idx: Vec<usize> = b_snli.iter().chain(b_mnli.iter()).copied().collect();

        let hh_oi = &pair_results[s].0;
        let h_b: f64 = b_idx.iter().map(|&i| hh_oi[i]).sum::<f64>() / b_idx.len() as f64;
        q_hhs_boot.push(h_b);

        for m_key in &model_keys {
            let (mh1_oi, mh2_oi) = &pair_results[s].1[m_key];
            let m1: f64 = b_idx.iter().map(|&i| mh1_oi[i]).sum::<f64>() / b_idx.len() as f64;
            let m2: f64 = b_idx.iter().map(|&i| mh2_oi[i]).sum::<f64>() / b_idx.len() as f64;
            let m_b = 0.5 * (m1 + m2);
            q_hms_paired.get_mut(m_key).unwrap().push(m_b);
            delta_ms.get_mut(m_key).unwrap().push(h_b - m_b);
        }
    }
    println!("Bootstrap complete in {:?}", t_boot.elapsed());

    // ── Summary ──────────────────────────────────────────────────────────────
    let mut hh_sorted = q_hhs_boot.clone();
    hh_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let hh_mean = hh_sorted.iter().sum::<f64>() / N_BOOT as f64;
    let hh_p025 = hh_sorted[(N_BOOT as f64 * 0.025) as usize];
    let hh_p975 = hh_sorted[(N_BOOT as f64 * 0.975) as usize];

    println!("\nHH100 paired bootstrap mean: {:.5}", hh_mean);
    println!("HH100 paired bootstrap 95%CI: [{:.5}, {:.5}]", hh_p025, hh_p975);
    println!();
    println!("{:<22} {:>12} {:>10} {:>12} {:>12} {:>12}",
        "Model", "Paired Q_m", "Delta_m", "CI Low", "CI Hi", "Fixed Q_m");
    println!("{}", "-".repeat(82));

    let mut model_results: HashMap<String, ModelPairedResult> = HashMap::new();
    for m_key in &model_keys {
        let arr = q_hms_paired[m_key].as_slice();
        let mut d_arr = delta_ms[m_key].clone();
        d_arr.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let q_mean = arr.iter().sum::<f64>() / N_BOOT as f64;
        let d_mean = d_arr.iter().sum::<f64>() / N_BOOT as f64;
        let d_p025 = d_arr[(N_BOOT as f64 * 0.025) as usize];
        let d_p975 = d_arr[(N_BOOT as f64 * 0.975) as usize];
        let n_gt = d_arr.iter().filter(|&&x| x > 0.0).count();
        let q_fixed = model_fixed_q[m_key];

        println!("{:<22} {:>12.5} {:>10.5} {:>12.5} {:>12.5} {:>12.5}",
            m_key, q_mean, d_mean, d_p025, d_p975, q_fixed);

        model_results.insert(m_key.clone(), ModelPairedResult {
            q_paired_hm_mean: (q_mean * 100000.0).round() / 100000.0,
            delta_m_mean: (d_mean * 100000.0).round() / 100000.0,
            delta_m_95ci: [
                (d_p025 * 100000.0).round() / 100000.0,
                (d_p975 * 100000.0).round() / 100000.0,
            ],
            replicates_gt_zero: format!("{}/{}", n_gt, N_BOOT),
            q_fixed_reference: (q_fixed * 100000.0).round() / 100000.0,
        });
    }

    let total = t0.elapsed();
    println!("\n=========================================================================");
    println!("   RUST PAIRED ESTIMAND COMPLETE IN {:.2?}", total);
    println!("=========================================================================");

    let out = PairedEstimandResults {
        estimand: "paired".to_string(),
        description: "M_m,b = 0.5 * [Q(G_m, G_H1^(b)) + Q(G_m, G_H2^(b))]; fully paired design".to_string(),
        n_pairs: N_PAIRS,
        n_bootstrap: N_BOOT,
        k: K,
        hh100_bootstrap_mean: (hh_mean * 100000.0).round() / 100000.0,
        hh100_bootstrap_95ci: [
            (hh_p025 * 100000.0).round() / 100000.0,
            (hh_p975 * 100000.0).round() / 100000.0,
        ],
        models: model_results,
        total_runtime_ms: total.as_secs_f64() * 1000.0,
    };

    let f_out = File::create(
        "c:/Users/admir/Github/shadowspace/results/paired_estimand_results.yaml"
    ).unwrap();
    serde_json::to_writer_pretty(f_out, &out).unwrap();
    println!("Saved to results/paired_estimand_results.yaml");
}
