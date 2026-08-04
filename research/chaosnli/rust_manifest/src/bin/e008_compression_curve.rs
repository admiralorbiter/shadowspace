/// E008: Relational Rate-Distortion / Human Geometry Compression Curve Engine
/// Fits 5-fold cross-validated Hellinger prototype quantization for K in {2..128}.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
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
    q_support: f64,
    r_normalized: f64,
}

#[derive(Debug, Clone, Serialize)]
struct E008Summary {
    experiment_id: String,
    title: String,
    subset: String,
    object_count: usize,
    q_hh_relational: f64,
    q_null_stratified: f64,
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
    rel_path.to_string()
}

fn k_means_hellinger(train_p: &[Vec<f64>], k: usize, max_iter: usize) -> Vec<Vec<f64>> {
    let n = train_p.len();
    if n <= k {
        return train_p.to_vec();
    }

    // Initialize centroids uniformly spaced across data
    let step = n / k;
    let mut centroids: Vec<Vec<f64>> = (0..k).map(|i| train_p[(i * step).min(n - 1)].clone()).collect();

    for _iter in 0..max_iter {
        let mut assignments = vec![0usize; n];
        for i in 0..n {
            let mut min_d = f64::INFINITY;
            let mut best_c = 0;
            for c in 0..k {
                let d = distance_hellinger(&train_p[i], &centroids[c]);
                if d < min_d {
                    min_d = d;
                    best_c = c;
                }
            }
            assignments[i] = best_c;
        }

        let mut new_centroids = vec![vec![0.0; 3]; k];
        let mut counts = vec![0usize; k];

        for i in 0..n {
            let c = assignments[i];
            for comp in 0..3 {
                new_centroids[c][comp] += train_p[i][comp];
            }
            counts[c] += 1;
        }

        let mut changed = false;
        for c in 0..k {
            if counts[c] > 0 {
                let mut sum = 0.0;
                for comp in 0..3 {
                    new_centroids[c][comp] /= counts[c] as f64;
                    sum += new_centroids[c][comp];
                }
                for comp in 0..3 {
                    new_centroids[c][comp] /= sum.max(1e-12);
                    if (new_centroids[c][comp] - centroids[c][comp]).abs() > 1e-5 {
                        changed = true;
                    }
                }
                centroids[c] = new_centroids[c].clone();
            }
        }

        if !changed {
            break;
        }
    }

    centroids
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let subset = if args.len() > 1 { &args[1] } else { "preflight" };

    let rel_manifest = if subset == "pilot" {
        "research/chaosnli/artifacts/E004/manifests/pilot_600.jsonl"
    } else {
        "research/chaosnli/artifacts/E004/manifests/preflight_60.jsonl"
    };
    let manifest_path = resolve_path(rel_manifest);

    let rel_k10_bin = format!("research/chaosnli/artifacts/E004/pilot_support/S_hellinger_k010_{}.bin", subset);
    let k10_bin = resolve_path(&rel_k10_bin);

    let rel_k10_meta = format!("research/chaosnli/artifacts/E004/pilot_support/S_hellinger_k010_{}.manifest.json", subset);
    let k10_manifest = resolve_path(&rel_k10_meta);

    println!("=========================================================================");
    println!("   E008: RELATIONAL RATE-DISTORTION COMPRESSION CURVE ({})", subset.to_uppercase());
    println!("=========================================================================");

    let file = File::open(&manifest_path)?;
    let reader = BufReader::new(file);
    let items: Vec<ManifestItem> = serde_json::Deserializer::from_reader(reader)
        .into_iter::<ManifestItem>()
        .filter_map(Result::ok)
        .collect();

    let n = items.len();
    println!("Loaded {} items from {}", n, manifest_path);

    let meta_file = File::open(&k10_manifest)?;
    let meta_json: serde_json::Value = serde_json::from_reader(meta_file)?;
    let q_hh = meta_json["q_hh_relational"].as_f64().unwrap_or(0.77494);

    let bin_bytes = std::fs::read(&k10_bin)?;
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

    let k_ladder = vec![2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128];
    let q_null_stratified = 0.16879;
    let n_folds = 5;

    let mut prototype_points = Vec::new();

    for &k in &k_ladder {
        if k > n {
            continue;
        }
        // 5-fold cross-validated prototype assignment
        let mut q_reconstructed = vec![vec![0.0; 3]; n];

        for fold in 0..n_folds {
            let val_indices: Vec<usize> = (0..n).filter(|i| i % n_folds == fold).collect();
            let train_indices: Vec<usize> = (0..n).filter(|i| i % n_folds != fold).collect();

            let train_p: Vec<Vec<f64>> = train_indices.iter().map(|&i| p_human[i].clone()).collect();
            let centroids = k_means_hellinger(&train_p, k, 30);

            for &v_idx in &val_indices {
                let mut min_d = f64::INFINITY;
                let mut best_c = 0;
                for c in 0..k {
                    let d = distance_hellinger(&p_human[v_idx], &centroids[c]);
                    if d < min_d {
                        min_d = d;
                        best_c = c;
                    }
                }
                q_reconstructed[v_idx] = centroids[best_c].clone();
            }
        }

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
        let q_supp = evaluate_q_support(&w_rec, &s_k10, 10);
        let r_norm = (q_supp - q_null_stratified) / (q_hh - q_null_stratified).max(1e-12);

        println!(
            "  K = {:>3} Prototypes (5-fold CV): R_norm = {:>6.2}% | NLL = {:.4} | JSD = {:.4}",
            k,
            r_norm * 100.0,
            mean_nll,
            mean_jsd
        );

        prototype_points.push(PrototypePoint {
            k_prototypes: k,
            nll: mean_nll,
            jsd_bits: mean_jsd,
            q_support: q_supp,
            r_normalized: r_norm,
        });
    }

    let summary = E008Summary {
        experiment_id: "E008".to_string(),
        title: "Relational Rate-Distortion Compression Curve".to_string(),
        subset: subset.to_string(),
        object_count: n,
        q_hh_relational: q_hh,
        q_null_stratified,
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
