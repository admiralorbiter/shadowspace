/// E007: Complete Ensemble Census & Exact Shapley Value Attribution Engine
/// Evaluates all 511 non-empty subsets of 9 canonical NLI classifiers.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use chaosnli_engine::shapley::{compute_ensemble_census_and_shapley, ShapleyResult, SubsetResult};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
struct ManifestItem {
    row_index: usize,
    human_p_entailment: f64,
    human_p_neutral: f64,
    human_p_contradiction: f64,
}

#[derive(Debug, Clone, Serialize)]
struct E007Summary {
    experiment_id: String,
    title: String,
    subset: String,
    object_count: usize,
    model_count: usize,
    total_ensemble_subsets: usize,
    q_hh_relational: f64,
    q_null_stratified: f64,
    all_models: Vec<String>,
    shapley_attributions: Vec<ShapleyResult>,
    pareto_frontier_subsets: Vec<SubsetResult>,
    best_subset_by_size: HashMap<usize, SubsetResult>,
    subsets: Vec<SubsetResult>,
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
    println!("   E007: COMPLETE ENSEMBLE CENSUS & SHAPLEY ATTRIBUTION ({})", subset.to_uppercase());
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
        "bart-large".to_string(),
        "roberta-large".to_string(),
        "xlnet-large".to_string(),
        "albert-xxlarge".to_string(),
        "bert-large".to_string(),
        "roberta-base".to_string(),
        "xlnet-base".to_string(),
        "distilbert".to_string(),
        "bert-base".to_string(),
    ];

    let mut sliced_model_probs: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    for m_name in &canonical_models {
        if let Some(full_p) = full_model_probs.get(m_name) {
            let mut sliced = Vec::with_capacity(n);
            for &r_idx in &row_indices {
                if r_idx < full_p.len() {
                    sliced.push(full_p[r_idx].clone());
                } else {
                    sliced.push(vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
                }
            }
            sliced_model_probs.insert(m_name.clone(), sliced);
        } else {
            eprintln!("Warning: Model {} missing from model_probs.json", m_name);
        }
    }

    let q_null_stratified = 0.16879;

    println!("Computing 511 ensemble subset evaluations and exact Shapley attributions...");
    let (subsets, shapley) = compute_ensemble_census_and_shapley(
        &canonical_models,
        &sliced_model_probs,
        &p_human,
        &s_k10,
        q_hh,
        q_null_stratified,
        10,
    );

    let mut best_by_size: HashMap<usize, SubsetResult> = HashMap::new();
    for sub in &subsets {
        let size = sub.subset_size;
        match best_by_size.get(&size) {
            None => {
                best_by_size.insert(size, sub.clone());
            }
            Some(existing) => {
                if sub.r_normalized > existing.r_normalized {
                    best_by_size.insert(size, sub.clone());
                }
            }
        }
    }

    println!("\n--- BEST ENSEMBLE SUBSET BY SIZE ---");
    for size in 1..=canonical_models.len() {
        if let Some(b) = best_by_size.get(&size) {
            println!(
                "  Size {:>2}: R_norm = {:>6.2}% | NLL = {:.4} | JSD = {:.4} | Models: [{}]",
                size,
                b.r_normalized * 100.0,
                b.nll,
                b.jsd_bits,
                b.model_names.join(", ")
            );
        }
    }

    println!("\n--- EXACT SHAPLEY ATTRIBUTION VALUES (R_normalized Contribution) ---");
    let mut sorted_shapley = shapley.clone();
    sorted_shapley.sort_by(|a, b| b.shapley_r_normalized.partial_cmp(&a.shapley_r_normalized).unwrap());
    for s in &sorted_shapley {
        println!(
            "  {:>15}: phi_R = {:>+6.2}% | phi_NLL = {:>+.4} nats | phi_Q = {:>+.5}",
            s.model_name,
            s.shapley_r_normalized * 100.0,
            s.shapley_nll_reduction,
            s.shapley_q_support
        );
    }

    let summary = E007Summary {
        experiment_id: "E007".to_string(),
        title: "Complete Ensemble Census & Shapley Attribution".to_string(),
        subset: subset.to_string(),
        object_count: n,
        model_count: canonical_models.len(),
        total_ensemble_subsets: subsets.len(),
        q_hh_relational: q_hh,
        q_null_stratified,
        all_models: canonical_models,
        shapley_attributions: shapley,
        pareto_frontier_subsets: best_by_size.values().cloned().collect(),
        best_subset_by_size: best_by_size,
        subsets,
    };

    let rel_out_dir = format!("research/chaosnli/artifacts/E007/summaries");
    let out_dir_str = resolve_path(&rel_out_dir);
    let out_dir = Path::new(&out_dir_str);
    std::fs::create_dir_all(out_dir)?;
    let out_path = out_dir.join("E007_summary.json");
    let out_file = File::create(&out_path)?;
    serde_json::to_writer_pretty(out_file, &summary)?;

    println!("\n=========================================================================");
    println!("Saved E007 summary JSON to {}", out_path.display());
    println!("=========================================================================");

    Ok(())
}
