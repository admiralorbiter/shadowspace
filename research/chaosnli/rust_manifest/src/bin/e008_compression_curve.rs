/// E008: Relational Rate-Distortion / Human Geometry Compression Curve Engine
/// Fits 5-fold cross-validated Hellinger prototype quantization (in square-root simplex z-space)
/// across a dense grid K in {1..128} to audit stratum sampling vs intrinsic compression thresholds.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use chaosnli_engine::distance::{distance_hellinger, distance_hellinger_matrix, jsd, soft_label_nll};
use chaosnli_engine::topk::{compute_topk_weight_matrix, evaluate_q_support};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
struct ManifestItem {
    row_index: usize,
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
    r_normalized_k10: f64,
    q_support_k5: f64,
    r_normalized_k5: f64,
    q_support_k20: f64,
    r_normalized_k20: f64,
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

/// Perform k-means in square-root simplex space z_i = sqrt(p_i) where Euclidean distance is Hellinger distance
fn k_means_hellinger_zspace(train_p: &[Vec<f64>], k: usize, max_iter: usize) -> Vec<Vec<f64>> {
    let n = train_p.len();
    if n <= k {
        return train_p.to_vec();
    }

    // Map to z-space: z_i = sqrt(p_i)
    let train_z: Vec<Vec<f64>> = train_p
        .iter()
        .map(|p| vec![p[0].max(0.0).sqrt(), p[1].max(0.0).sqrt(), p[2].max(0.0).sqrt()])
        .collect();

    // Initialize centroids in z-space
    let step = n / k;
    let mut centroids_z: Vec<Vec<f64>> = (0..k).map(|i| train_z[(i * step).min(n - 1)].clone()).collect();

    for _iter in 0..max_iter {
        let mut assignments = vec![0usize; n];
        for i in 0..n {
            let mut min_d = f64::INFINITY;
            let mut best_c = 0;
            for c in 0..k {
                let dz0 = train_z[i][0] - centroids_z[c][0];
                let dz1 = train_z[i][1] - centroids_z[c][1];
                let dz2 = train_z[i][2] - centroids_z[c][2];
                let d_sq = dz0 * dz0 + dz1 * dz1 + dz2 * dz2;
                if d_sq < min_d {
                    min_d = d_sq;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }

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

    // Map z-space centroids back to probability simplex p = z^2 / sum(z^2)
    centroids_z
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
        // Check if distance at k-th neighbor equals distance at (k+1)-th neighbor when dist near 0
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
    println!("   E008: RELATIONAL RATE-DISTORTION COMPRESSION CURVE ({})", subset.to_uppercase());
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

    // Denser grid near knee [20..36]
    let k_ladder = vec![1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 40, 48, 64, 128];
    let q_null_stratified = 10.0 / (n as f64);
    let n_folds = 5;

    let mut prototype_points = Vec::new();

    for &k_proto in &k_ladder {
        if k_proto > n {
            continue;
        }
        let mut q_reconstructed = vec![vec![0.0; 3]; n];
        let mut cluster_counts = vec![0usize; k_proto];

        for fold in 0..n_folds {
            let val_indices: Vec<usize> = (0..n).filter(|i| i % n_folds == fold).collect();
            let train_indices: Vec<usize> = (0..n).filter(|i| i % n_folds != fold).collect();

            let train_p: Vec<Vec<f64>> = train_indices.iter().map(|&i| p_human[i].clone()).collect();
            let centroids = k_means_hellinger_zspace(&train_p, k_proto, 40);

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
                cluster_counts[best_c] += 1;
            }
        }

        let mut counts_sorted = cluster_counts.clone();
        counts_sorted.sort();
        let min_c = counts_sorted[0];
        let max_c = counts_sorted[k_proto - 1];
        let med_c = counts_sorted[k_proto / 2];
        let empty_c = cluster_counts.iter().filter(|&&cnt| cnt == 0).count();

        let mut nll_sum = 0.0;
        let mut jsd_sum = 0.0;
        for i in 0..n {
            nll_sum += soft_label_nll(&p_human[i], &q_reconstructed[i]);
            jsd_sum += jsd(&p_human[i], &q_reconstructed[i]);
        }
        let mean_nll = nll_sum / (n as f64);
        let mean_jsd = jsd_sum / (n as f64);

        let dist_rec = distance_hellinger_matrix(&q_reconstructed);
        let tie_frac = compute_zero_distance_tie_fraction(&dist_rec, 10);

        // Evaluate at k=10
        let w_rec_k10 = compute_topk_weight_matrix(&dist_rec, 10);
        let q_supp_k10 = evaluate_q_support(&w_rec_k10, &s_k10, 10);
        let r_norm_k10 = (q_supp_k10 - q_null_stratified) / (q_hh - q_null_stratified).max(1e-12);

        // Evaluate at k=5
        let w_rec_k5 = compute_topk_weight_matrix(&dist_rec, 5);
        let q_supp_k5 = evaluate_q_support(&w_rec_k5, &s_k10, 5);
        let r_norm_k5 = (q_supp_k5 - q_null_stratified) / (q_hh - q_null_stratified).max(1e-12);

        // Evaluate at k=20
        let w_rec_k20 = compute_topk_weight_matrix(&dist_rec, 20);
        let q_supp_k20 = evaluate_q_support(&w_rec_k20, &s_k10, 20);
        let r_norm_k20 = (q_supp_k20 - q_null_stratified) / (q_hh - q_null_stratified).max(1e-12);

        println!(
            "  K = {:>3} Prototypes (z-space CV): R_k10 = {:>6.2}% | NLL = {:.4} | Empty = {} | TieFrac = {:.2}%",
            k_proto,
            r_norm_k10 * 100.0,
            mean_nll,
            empty_c,
            tie_frac * 100.0
        );

        prototype_points.push(PrototypePoint {
            k_prototypes: k_proto,
            nll: mean_nll,
            jsd_bits: mean_jsd,
            q_support_k10: q_supp_k10,
            r_normalized_k10: r_norm_k10,
            q_support_k5: q_supp_k5,
            r_normalized_k5: r_norm_k5,
            q_support_k20: q_supp_k20,
            r_normalized_k20: r_norm_k20,
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
