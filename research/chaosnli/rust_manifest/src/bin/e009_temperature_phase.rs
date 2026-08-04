/// E009: Temperature-Topology Phase Diagram Engine
/// Evaluates a 50-point logarithmic grid T in [0.05, 100.0] for all classifiers.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use chaosnli_engine::distance::{distance_hellinger_matrix, jsd, soft_label_nll};
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
struct TempGridPoint {
    temperature: f64,
    nll: f64,
    jsd_bits: f64,
    q_support: f64,
    r_normalized: f64,
    graph_turnover_frac: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ModelPhaseDiagram {
    model_name: String,
    raw_nll: f64,
    raw_r_norm: f64,
    opt_nll_temp: f64,
    opt_nll_val: f64,
    opt_nll_r_norm: f64,
    opt_q_temp: f64,
    opt_q_r_norm: f64,
    max_r_gain: f64,
    grid_points: Vec<TempGridPoint>,
}

#[derive(Debug, Clone, Serialize)]
struct E009Summary {
    experiment_id: String,
    title: String,
    subset: String,
    object_count: usize,
    q_hh_relational: f64,
    q_null_stratified: f64,
    temperature_grid: Vec<f64>,
    models: HashMap<String, ModelPhaseDiagram>,
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

fn temperature_scale_probs(p_raw: &[Vec<f64>], temp: f64) -> Vec<Vec<f64>> {
    let n = p_raw.len();
    let mut p_scaled = vec![vec![0.0; 3]; n];
    for i in 0..n {
        let mut logit = [0.0; 3];
        for c in 0..3 {
            logit[c] = (p_raw[i][c].max(1e-12)).ln() / temp;
        }
        let max_l = logit[0].max(logit[1]).max(logit[2]);
        let mut exp_sum = 0.0;
        let mut exps = [0.0; 3];
        for c in 0..3 {
            exps[c] = (logit[c] - max_l).exp();
            exp_sum += exps[c];
        }
        for c in 0..3 {
            p_scaled[i][c] = exps[c] / exp_sum;
        }
    }
    p_scaled
}

fn compute_graph_turnover(w_ref: &[Vec<f64>], w_t: &[Vec<f64>]) -> f64 {
    let n = w_ref.len();
    let mut diff = 0.0;
    let mut total = 0.0;
    for i in 0..n {
        for j in 0..n {
            diff += (w_ref[i][j] - w_t[i][j]).abs();
            total += w_ref[i][j];
        }
    }
    if total > 0.0 {
        diff / (2.0 * total)
    } else {
        0.0
    }
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

    let rel_probs_json = "research/chaosnli/rust_manifest/model_probs.json";
    let probs_json_path = resolve_path(rel_probs_json);

    println!("=========================================================================");
    println!("   E009: TEMPERATURE-TOPOLOGY PHASE DIAGRAM ({})", subset.to_uppercase());
    println!("=========================================================================");

    let file = File::open(&manifest_path)?;
    let reader = BufReader::new(file);
    let items: Vec<ManifestItem> = serde_json::Deserializer::from_reader(reader)
        .into_iter::<ManifestItem>()
        .filter_map(Result::ok)
        .collect();

    let n = items.len();
    let row_indices: Vec<usize> = items.iter().map(|it| it.row_index).collect();
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

    let probs_file = File::open(&probs_json_path)?;
    let full_model_probs: HashMap<String, Vec<Vec<f64>>> = serde_json::from_reader(probs_file)?;

    let canonical_models = vec![
        "bart-large",
        "roberta-large",
        "xlnet-large",
        "albert-xxlarge",
        "bert-large",
        "roberta-base",
        "xlnet-base",
        "distilbert",
        "bert-base",
    ];

    // Generate temperature grid with 1.0 explicitly included
    let mut temp_grid: Vec<f64> = (0..50)
        .map(|i| {
            let frac = i as f64 / 49.0;
            (0.05f64.ln() + frac * (100.0f64.ln() - 0.05f64.ln())).exp()
        })
        .collect();
    temp_grid.push(1.0);
    temp_grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    temp_grid.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    let k_neighbors = 10usize;
    let q_null_stratified = (k_neighbors as f64) / (n as f64);
    let mut model_phase_results = HashMap::new();

    for m_name in &canonical_models {
        println!("\n--- Computing Temperature Phase Diagram for: {} ---", m_name);
        let full_p = &full_model_probs[*m_name];
        let mut sliced_p = Vec::with_capacity(n);
        for &r_idx in &row_indices {
            if r_idx < full_p.len() {
                sliced_p.push(full_p[r_idx].clone());
            } else {
                sliced_p.push(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
            }
        }

        let dist_ref = distance_hellinger_matrix(&sliced_p);
        let w_ref = compute_topk_weight_matrix(&dist_ref, k_neighbors);

        let mut grid_points = Vec::new();
        let mut min_nll = f64::INFINITY;
        let mut opt_nll_t = 1.0;
        let mut opt_nll_r = 0.0;

        let mut max_q = f64::NEG_INFINITY;
        let mut opt_q_t = 1.0;
        let mut opt_q_r = 0.0;

        let mut raw_nll = 0.0;
        let mut raw_r_norm = 0.0;

        for &t in &temp_grid {
            let p_t = temperature_scale_probs(&sliced_p, t);

            let mut nll_sum = 0.0;
            let mut jsd_sum = 0.0;
            for i in 0..n {
                nll_sum += soft_label_nll(&p_human[i], &p_t[i]);
                jsd_sum += jsd(&p_human[i], &p_t[i]);
            }
            let mean_nll = nll_sum / (n as f64);
            let mean_jsd = jsd_sum / (n as f64);

            let dist_t = distance_hellinger_matrix(&p_t);
            let w_t = compute_topk_weight_matrix(&dist_t, k_neighbors);
            let q_supp = evaluate_q_support(&w_t, &s_k10, k_neighbors);
            let r_norm = (q_supp - q_null_stratified) / (q_hh - q_null_stratified).max(1e-12);
            let turnover = compute_graph_turnover(&w_ref, &w_t);

            if (t - 1.0).abs() < 1e-4 {
                raw_nll = mean_nll;
                raw_r_norm = r_norm;
            }

            if mean_nll < min_nll {
                min_nll = mean_nll;
                opt_nll_t = t;
                opt_nll_r = r_norm;
            }

            if q_supp > max_q {
                max_q = q_supp;
                opt_q_t = t;
                opt_q_r = r_norm;
            }

            grid_points.push(TempGridPoint {
                temperature: t,
                nll: mean_nll,
                jsd_bits: mean_jsd,
                q_support: q_supp,
                r_normalized: r_norm,
                graph_turnover_frac: turnover,
            });
        }

        let max_r_gain = opt_q_r - raw_r_norm;

        println!(
            "  Raw (T=1.0) : NLL = {:.4} | R_norm = {:>6.2}%",
            raw_nll, raw_r_norm * 100.0
        );
        println!(
            "  NLL Opt     : T = {:.3} | NLL = {:.4} | R_norm = {:>6.2}%",
            opt_nll_t, min_nll, opt_nll_r * 100.0
        );
        println!(
            "  Q Opt       : T = {:.3} | R_norm = {:>6.2}% | Gain = {:>+6.2}%",
            opt_q_t, opt_q_r * 100.0, max_r_gain * 100.0
        );

        model_phase_results.insert(
            m_name.to_string(),
            ModelPhaseDiagram {
                model_name: m_name.to_string(),
                raw_nll,
                raw_r_norm,
                opt_nll_temp: opt_nll_t,
                opt_nll_val: min_nll,
                opt_nll_r_norm: opt_nll_r,
                opt_q_temp: opt_q_t,
                opt_q_r_norm: opt_q_r,
                max_r_gain,
                grid_points,
            },
        );
    }

    let summary = E009Summary {
        experiment_id: "E009".to_string(),
        title: "Temperature-Topology Phase Diagram".to_string(),
        subset: subset.to_string(),
        object_count: n,
        q_hh_relational: q_hh,
        q_null_stratified,
        temperature_grid: temp_grid,
        models: model_phase_results,
    };

    let rel_out_dir = format!("research/chaosnli/artifacts/E009/summaries");
    let out_dir_str = resolve_path(&rel_out_dir);
    let out_dir = Path::new(&out_dir_str);
    std::fs::create_dir_all(out_dir)?;
    let out_path = out_dir.join("E009_summary.json");
    let out_file = File::create(&out_path)?;
    serde_json::to_writer_pretty(out_file, &summary)?;

    println!("\n=========================================================================");
    println!("Saved E009 summary JSON to {}", out_path.display());
    println!("=========================================================================");

    Ok(())
}
