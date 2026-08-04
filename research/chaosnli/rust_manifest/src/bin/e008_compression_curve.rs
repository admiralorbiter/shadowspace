/// E008: Audited Relational Rate-Distortion Compression Engine
/// Fits 5-fold cross-validated Hellinger prototype quantization in square-root simplex z-space
/// with multi-start k-means++, dataset-stratified folds, analytic stratified nulls Q_null(K),
/// and empirical human baseline recovery C_K.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use chaosnli_engine::distance::{distance_hellinger, distance_hellinger_matrix, jsd, soft_label_nll};
use chaosnli_engine::topk::{compute_topk_weight_matrix, evaluate_q_support};

#[derive(Debug, Clone, Deserialize)]
struct ManifestItem {
    row_index: usize,
    source_dataset: String,
    human_p_entailment: f64,
    human_p_neutral: f64,
    human_p_contradiction: f64,
}

#[derive(Debug, Clone, Serialize)]
struct PrototypePoint {
    k_prototypes: usize,
    nll: f64,
    jsd_bits: f64,
    q_support_k10: f64,
    q_null_analytic_k10: f64,
    r_normalized_k10: f64,
    c_empirical_retained_k10: f64,
    min_cluster_size: usize,
    median_cluster_size: usize,
    max_cluster_size: usize,
    empty_cluster_count: usize,
    zero_distance_tie_frac_k10: f64,
}

#[derive(Debug, Clone, Serialize)]
struct E008Summary {
    experiment_id: String,
    title: String,
    subset: String,
    object_count: usize,
    q_hh_relational: f64,
    q_empirical_relational: f64,
    prototype_ladder: Vec<PrototypePoint>,
}

fn resolve_path(rel_path: &str) -> String {
    if Path::new(rel_path).exists() {
        return rel_path.to_string();
    }
    let p1 = format!("../{}", rel_path);
    if Path::new(&p1).exists() {
        return p1;
    }
    let stripped = rel_path.strip_prefix("research/chaosnli/").unwrap_or(rel_path);
    let p2 = format!("../{}", stripped);
    if Path::new(&p2).exists() {
        return p2;
    }
    let p3 = format!("../../{}", stripped);
    if Path::new(&p3).exists() {
        return p3;
    }
    panic!("Required artifact file not found: {}", rel_path);
}

/// Compute dataset-stratified analytic expected null Q_null(W) for a top-k weight matrix W
fn compute_analytic_stratified_null(
    w_matrix: &[Vec<f64>],
    s_target: &[Vec<f64>],
    is_snli: &[bool],
    k_neighbors: usize,
) -> f64 {
    let n = w_matrix.len();
    let n_snli = is_snli.iter().filter(|&&b| b).count();
    let n_mnli = n - n_snli;

    let mut s_ss = 0.0;
    let mut s_sm = 0.0;
    let mut s_ms = 0.0;
    let mut s_mm = 0.0;

    let mut w_ss = 0.0;
    let mut w_sm = 0.0;
    let mut w_ms = 0.0;
    let mut w_mm = 0.0;

    for r in 0..n {
        for c in 0..n {
            if r == c {
                continue;
            }
            let s_val = s_target[r][c];
            let w_val = w_matrix[r][c];

            if is_snli[r] && is_snli[c] {
                s_ss += s_val;
                if w_val > 1e-12 {
                    w_ss += w_val;
                }
            } else if is_snli[r] && !is_snli[c] {
                s_sm += s_val;
                if w_val > 1e-12 {
                    w_sm += w_val;
                }
            } else if !is_snli[r] && is_snli[c] {
                s_ms += s_val;
                if w_val > 1e-12 {
                    w_ms += w_val;
                }
            } else {
                s_mm += s_val;
                if w_val > 1e-12 {
                    w_mm += w_val;
                }
            }
        }
    }

    let e_w_ss = if n_snli > 1 { w_ss / (n_snli * (n_snli - 1)) as f64 } else { 0.0 };
    let e_w_sm = if n_snli > 0 && n_mnli > 0 { w_sm / (n_snli * n_mnli) as f64 } else { 0.0 };
    let e_w_ms = if n_snli > 0 && n_mnli > 0 { w_ms / (n_mnli * n_snli) as f64 } else { 0.0 };
    let e_w_mm = if n_mnli > 1 { w_mm / (n_mnli * (n_mnli - 1)) as f64 } else { 0.0 };

    (e_w_ss * s_ss + e_w_sm * s_sm + e_w_ms * s_ms + e_w_mm * s_mm) / ((n * k_neighbors) as f64)
}

/// Perform k-means++ in square-root simplex space z_i = sqrt(p_i) with 20 deterministic restarts
fn k_means_plus_plus_zspace(train_p: &[Vec<f64>], k: usize, n_restarts: usize, seed_base: u64) -> Vec<Vec<f64>> {
    let n = train_p.len();
    if n <= k {
        return train_p.to_vec();
    }

    let train_z: Vec<Vec<f64>> = train_p
        .iter()
        .map(|p| vec![p[0].max(0.0).sqrt(), p[1].max(0.0).sqrt(), p[2].max(0.0).sqrt()])
        .collect();

    let mut best_distortion = f64::INFINITY;
    let mut best_centroids_z = Vec::new();

    for restart in 0..n_restarts {
        let mut rng = ChaCha8Rng::seed_from_u64(seed_base + restart as u64);

        // k-means++ initialization
        let mut centroids_z: Vec<Vec<f64>> = Vec::with_capacity(k);
        let first_idx = rng.gen_range(0..n);
        centroids_z.push(train_z[first_idx].clone());

        for _ in 1..k {
            let mut dists_sq = vec![f64::INFINITY; n];
            let mut sum_dists = 0.0;
            for i in 0..n {
                for c in &centroids_z {
                    let dz0 = train_z[i][0] - c[0];
                    let dz1 = train_z[i][1] - c[1];
                    let dz2 = train_z[i][2] - c[2];
                    let d2 = dz0 * dz0 + dz1 * dz1 + dz2 * dz2;
                    if d2 < dists_sq[i] {
                        dists_sq[i] = d2;
                    }
                }
                sum_dists += dists_sq[i];
            }

            if sum_dists <= 1e-12 {
                let fallback = rng.gen_range(0..n);
                centroids_z.push(train_z[fallback].clone());
            } else {
                let target = rng.gen_range(0.0..1.0) * sum_dists;
                let mut cum = 0.0;
                let mut chosen = 0;
                for i in 0..n {
                    cum += dists_sq[i];
                    if cum >= target {
                        chosen = i;
                        break;
                    }
                }
                centroids_z.push(train_z[chosen].clone());
            }
        }

        // Standard Lloyd iterations in z-space
        let mut current_distortion = 0.0;
        for _iter in 0..30 {
            let mut assignments = vec![0usize; n];
            let mut total_d2 = 0.0;

            for i in 0..n {
                let mut min_d2 = f64::INFINITY;
                let mut best_c = 0;
                for c in 0..k {
                    let dz0 = train_z[i][0] - centroids_z[c][0];
                    let dz1 = train_z[i][1] - centroids_z[c][1];
                    let dz2 = train_z[i][2] - centroids_z[c][2];
                    let d2 = dz0 * dz0 + dz1 * dz1 + dz2 * dz2;
                    if d2 < min_d2 {
                        min_d2 = d2;
                        best_c = c;
                    }
                }
                assignments[i] = best_c;
                total_d2 += min_d2;
            }

            current_distortion = total_d2;

            let mut new_centroids_z = vec![vec![0.0; 3]; k];
            let mut counts = vec![0usize; k];

            for i in 0..n {
                let c = assignments[i];
                for comp in 0..3 {
                    new_centroids_z[c][comp] += train_z[i][comp];
                }
                counts[c] += 1;
            }

            let mut changed = false;
            for c in 0..k {
                if counts[c] > 0 {
                    let mut sum_sq = 0.0;
                    for comp in 0..3 {
                        new_centroids_z[c][comp] /= counts[c] as f64;
                        sum_sq += new_centroids_z[c][comp] * new_centroids_z[c][comp];
                    }
                    let norm = sum_sq.sqrt().max(1e-12);
                    for comp in 0..3 {
                        new_centroids_z[c][comp] /= norm;
                        if (new_centroids_z[c][comp] - centroids_z[c][comp]).abs() > 1e-5 {
                            changed = true;
                        }
                    }
                    centroids_z[c] = new_centroids_z[c].clone();
                }
            }

            if !changed {
                break;
            }
        }

        if current_distortion < best_distortion {
            best_distortion = current_distortion;
            best_centroids_z = centroids_z;
        }
    }

    // Map z-space centroids back to probability simplex
    best_centroids_z
        .into_iter()
        .map(|z| {
            let p0 = z[0] * z[0];
            let p1 = z[1] * z[1];
            let p2 = z[2] * z[2];
            let sum = (p0 + p1 + p2).max(1e-12);
            vec![p0 / sum, p1 / sum, p2 / sum]
        })
        .collect()
}

fn compute_zero_distance_tie_fraction(dist_matrix: &[Vec<f64>], k: usize) -> f64 {
    let n = dist_matrix.len();
    let mut tie_count = 0;
    for i in 0..n {
        let mut row_dists = dist_matrix[i].clone();
        row_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if k < n && (row_dists[k] - row_dists[k + 1]).abs() < 1e-12 && row_dists[k] < 1e-6 {
            tie_count += 1;
        }
    }
    (tie_count as f64) / (n as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let subset = if args.len() > 1 { &args[1] } else { "preflight" };

    let (rel_manifest, rel_k10_bin, rel_k10_meta, expected_n) = match subset {
        "preflight" => (
            "research/chaosnli/artifacts/E004/manifests/preflight_60.jsonl",
            "research/chaosnli/artifacts/E004/pilot_support/S_hellinger_k010_preflight.bin",
            "research/chaosnli/artifacts/E004/pilot_support/S_hellinger_k010_preflight.manifest.json",
            60,
        ),
        "pilot" => (
            "research/chaosnli/artifacts/E004/manifests/pilot_600.jsonl",
            "research/chaosnli/artifacts/E004/pilot_support/S_hellinger_k010_pilot.bin",
            "research/chaosnli/artifacts/E004/pilot_support/S_hellinger_k010_pilot.manifest.json",
            600,
        ),
        "full" => (
            "research/chaosnli/artifacts/E004/manifests/full_3113.jsonl",
            "research/chaosnli/artifacts/E004/pilot_support/S_hellinger_k010_full.bin",
            "research/chaosnli/artifacts/E004/pilot_support/S_hellinger_k010_full.manifest.json",
            3113,
        ),
        _ => panic!("Unknown subset: '{}'. Must be one of ['preflight', 'pilot', 'full'].", subset),
    };

    let manifest_path = resolve_path(rel_manifest);
    let k10_bin = resolve_path(rel_k10_bin);
    let k10_manifest = resolve_path(rel_k10_meta);

    println!("=========================================================================");
    println!("   E008: AUDITED RATE-DISTORTION COMPRESSION ENGINE ({})", subset.to_uppercase());
    println!("=========================================================================");

    let file = File::open(&manifest_path)?;
    let reader = BufReader::new(file);
    let mut items = Vec::new();
    for line in reader.lines() {
        let line_str = line?;
        let item: ManifestItem = serde_json::from_str(&line_str)?;
        items.push(item);
    }

    let n = items.len();
    assert_eq!(n, expected_n, "Manifest item count mismatch for subset {}", subset);
    let is_snli: Vec<bool> = items.iter().map(|it| it.source_dataset.to_lowercase().contains("snli")).collect();
    println!("Loaded {} items from {}", n, manifest_path);

    let meta_file = File::open(&k10_manifest)?;
    let meta_json: serde_json::Value = serde_json::from_reader(meta_file)?;
    let q_hh = meta_json["q_hh_relational"].as_f64().expect("Missing q_hh_relational in meta");

    let bin_bytes = std::fs::read(&k10_bin)?;
    assert_eq!(bin_bytes.len(), n * n * 4, "Binary matrix byte length mismatch");
    let f32_floats: Vec<f32> = bin_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    let mut s_k10 = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            s_k10[i][j] = f32_floats[i * n + j] as f64;
        }
    }

    let p_human: Vec<Vec<f64>> = items
        .iter()
        .map(|it| vec![it.human_p_entailment, it.human_p_neutral, it.human_p_contradiction])
        .collect();

    // Empirical human baseline support & analytic null
    let dist_emp = distance_hellinger_matrix(&p_human);
    let w_emp = compute_topk_weight_matrix(&dist_emp, 10);
    let q_emp = evaluate_q_support(&w_emp, &s_k10, 10);
    let q_null_emp = compute_analytic_stratified_null(&w_emp, &s_k10, &is_snli, 10);
    let q_emp_excess = q_emp - q_null_emp;

    println!("Human Empirical Relational Q_emp = {:.5} | Q_null_emp = {:.5} | Excess = {:.5}", q_emp, q_null_emp, q_emp_excess);

    // Dense prototype grid
    let k_ladder = vec![1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 48, 64, 128];
    let n_folds = 5;

    let mut prototype_points = Vec::new();

    for &k_proto in &k_ladder {
        if k_proto > n {
            continue;
        }
        let mut q_reconstructed = vec![vec![0.0; 3]; n];
        let mut fold_cluster_counts: Vec<usize> = Vec::new();

        for fold in 0..n_folds {
            let val_indices: Vec<usize> = (0..n).filter(|i| i % n_folds == fold).collect();
            let train_indices: Vec<usize> = (0..n).filter(|i| i % n_folds != fold).collect();

            let train_p: Vec<Vec<f64>> = train_indices.iter().map(|&i| p_human[i].clone()).collect();
            let centroids = k_means_plus_plus_zspace(&train_p, k_proto, 20, 100 + fold as u64);

            let mut f_counts = vec![0usize; k_proto];
            for &v_idx in &val_indices {
                let mut min_d = f64::INFINITY;
                let mut best_c = 0;
                for c in 0..k_proto {
                    let d = distance_hellinger(&p_human[v_idx], &centroids[c]);
                    if d < min_d {
                        min_d = d;
                        best_c = c;
                    }
                }
                q_reconstructed[v_idx] = centroids[best_c].clone();
                f_counts[best_c] += 1;
            }
            fold_cluster_counts.extend(f_counts);
        }

        let mut sorted_occupancies = fold_cluster_counts.clone();
        sorted_occupancies.sort();
        let min_c = sorted_occupancies[0];
        let max_c = sorted_occupancies[sorted_occupancies.len() - 1];
        let med_c = sorted_occupancies[sorted_occupancies.len() / 2];
        let empty_c = fold_cluster_counts.iter().filter(|&&cnt| cnt == 0).count();

        let mut nll_sum = 0.0;
        let mut jsd_sum = 0.0;
        for i in 0..n {
            nll_sum += soft_label_nll(&p_human[i], &q_reconstructed[i]);
            jsd_sum += jsd(&p_human[i], &q_reconstructed[i]);
        }
        let mean_nll = nll_sum / (n as f64);
        let mean_jsd = jsd_sum / (n as f64);

        let dist_rec = distance_hellinger_matrix(&q_reconstructed);
        let w_rec = compute_topk_weight_matrix(&dist_rec, 10);
        let q_supp_k10 = evaluate_q_support(&w_rec, &s_k10, 10);

        // Exact analytic stratified null for prototype graph
        let q_null_proto = compute_analytic_stratified_null(&w_rec, &s_k10, &is_snli, 10);
        let r_norm_k10 = (q_supp_k10 - q_null_proto) / (q_hh - q_null_proto).max(1e-12);
        let c_emp_retained = (q_supp_k10 - q_null_proto) / (q_emp_excess).max(1e-12);
        let tie_frac = compute_zero_distance_tie_fraction(&dist_rec, 10);

        println!(
            "  K = {:>3}: R_norm = {:>6.2}% | C_retained = {:>6.2}% | Q_null_analytic = {:.5} | NLL = {:.4} | TieFrac = {:.2}%",
            k_proto,
            r_norm_k10 * 100.0,
            c_emp_retained * 100.0,
            q_null_proto,
            mean_nll,
            tie_frac * 100.0
        );

        prototype_points.push(PrototypePoint {
            k_prototypes: k_proto,
            nll: mean_nll,
            jsd_bits: mean_jsd,
            q_support_k10: q_supp_k10,
            q_null_analytic_k10: q_null_proto,
            r_normalized_k10: r_norm_k10,
            c_empirical_retained_k10: c_emp_retained,
            min_cluster_size: min_c,
            median_cluster_size: med_c,
            max_cluster_size: max_c,
            empty_cluster_count: empty_c,
            zero_distance_tie_frac_k10: tie_frac,
        });
    }

    let summary = E008Summary {
        experiment_id: "E008".to_string(),
        title: "Relational Rate-Distortion Compression Curve".to_string(),
        subset: subset.to_string(),
        object_count: n,
        q_hh_relational: q_hh,
        q_empirical_relational: q_emp,
        prototype_ladder: prototype_points,
    };

    let rel_out_dir = format!("research/chaosnli/artifacts/E008/summaries");
    let out_dir_str = resolve_path(&rel_out_dir);
    let out_dir = Path::new(&out_dir_str);
    std::fs::create_dir_all(out_dir)?;
    let out_path = out_dir.join("E008_summary.json");
    let out_file = File::create(&out_path)?;
    serde_json::to_writer_pretty(out_file, &summary)?;

    println!("\n=========================================================================");
    println!("Saved E008 summary JSON to {}", out_path.display());
    println!("=========================================================================");

    Ok(())
}
