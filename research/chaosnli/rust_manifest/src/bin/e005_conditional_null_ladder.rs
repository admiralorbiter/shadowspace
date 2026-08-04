/// E005: Conditional Null Ladder Experiment
/// Evaluates the 6-level strictly nested hierarchical null ladder N0..N5 across all 9 classifiers + ensembles + human target.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use chaosnli_engine::distance::{distance_hellinger_matrix, jsd, soft_label_nll};
use chaosnli_engine::nulls::{compute_conditional_null, NullResult};
use chaosnli_engine::topk::compute_topk_weight_matrix;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
struct ManifestItem {
    row_index: usize,
    source_dataset: String,
    human_count_entailment: u32,
    human_count_neutral: u32,
    human_count_contradiction: u32,
    human_p_entailment: f64,
    human_p_neutral: f64,
    human_p_contradiction: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ConditionLadderResult {
    condition_name: String,
    nll: f64,
    jsd_bits: f64,
    q_observed: f64,
    null_ladder: Vec<NullResult>,
}

#[derive(Debug, Clone, Serialize)]
struct E005Summary {
    experiment_id: String,
    title: String,
    subset: String,
    object_count: usize,
    q_hh_relational: f64,
    conditions: HashMap<String, ConditionLadderResult>,
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

fn compute_entropy_quintile(entropy: f64, all_entropies: &[f64]) -> usize {
    let mut sorted = all_entropies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let q1 = sorted[(n as f64 * 0.2) as usize];
    let q2 = sorted[(n as f64 * 0.4) as usize];
    let q3 = sorted[(n as f64 * 0.6) as usize];
    let q4 = sorted[(n as f64 * 0.8) as usize];

    if entropy <= q1 {
        0
    } else if entropy <= q2 {
        1
    } else if entropy <= q3 {
        2
    } else if entropy <= q4 {
        3
    } else {
        4
    }
}

fn compute_margin_bin(margin: f64) -> usize {
    if margin <= 0.2 {
        0
    } else if margin <= 0.4 {
        1
    } else if margin <= 0.6 {
        2
    } else if margin <= 0.8 {
        3
    } else {
        4
    }
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
    let probs_json_path = resolve_path("research/chaosnli/rust_manifest/model_probs.json");

    println!("=========================================================================");
    println!("   E005: STRICTLY NESTED CONDITIONAL NULL LADDER (Subset: {})", subset.to_uppercase());
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
    let row_indices: Vec<usize> = items.iter().map(|it| it.row_index).collect();
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

    let mut p_human = vec![vec![0.0; 3]; n];
    let mut entropies = vec![0.0; n];

    for (i, it) in items.iter().enumerate() {
        let p = vec![it.human_p_entailment, it.human_p_neutral, it.human_p_contradiction];
        let mut ent = 0.0;
        for &v in &p {
            if v > 1e-12 {
                ent -= v * (v.max(1e-12)).log2();
            }
        }
        p_human[i] = p;
        entropies[i] = ent;
    }

    // Precompute Strictly Nested Group Keys N0..N5
    let keys_n0 = vec!["global".to_string(); n];
    let mut keys_n1 = vec!["".to_string(); n];
    let mut keys_n2 = vec!["".to_string(); n];
    let mut keys_n3 = vec!["".to_string(); n];
    let mut keys_n4 = vec!["".to_string(); n];
    let mut keys_n5 = vec!["".to_string(); n];

    for i in 0..n {
        let it = &items[i];
        let d = &it.source_dataset;
        let p = &p_human[i];

        let maj = if p[0] >= p[1] && p[0] >= p[2] {
            0
        } else if p[1] >= p[0] && p[1] >= p[2] {
            1
        } else {
            2
        };

        let eq = compute_entropy_quintile(entropies[i], &entropies);

        let mut indexed: Vec<(usize, f64)> = p.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top1 = indexed[0].0;
        let top2 = indexed[1].0;
        let margin = indexed[0].1 - indexed[1].1;
        let margin_bin = compute_margin_bin(margin);

        keys_n1[i] = d.clone();
        keys_n2[i] = format!("{}_{}", keys_n1[i], maj);
        keys_n3[i] = format!("{}_{}", keys_n2[i], eq);
        keys_n4[i] = format!("{}_{}_{}_{}", keys_n3[i], top1, top2, margin_bin);
        // N5 MUST strictly embed N4
        keys_n5[i] = format!("{}|{}_{}_{}", keys_n4[i], it.human_count_entailment, it.human_count_neutral, it.human_count_contradiction);
    }

    // Programmatic hierarchy nesting validation assertion
    let mut parent_map: HashMap<String, String> = HashMap::new();
    for i in 0..n {
        if let Some(parent) = parent_map.get(&keys_n5[i]) {
            assert_eq!(parent, &keys_n4[i], "Strict nesting assertion failed at N5 -> N4");
        } else {
            parent_map.insert(keys_n5[i].clone(), keys_n4[i].clone());
        }
    }

    let null_levels = vec![
        ("N0", "Global Identity Permutation", keys_n0),
        ("N1", "Dataset Stratified (SNLI/MNLI)", keys_n1),
        ("N2", "N1 + Majority Label", keys_n2),
        ("N3", "N2 + Entropy Quintile", keys_n3),
        ("N4", "N3 + Top-2 Label Pair + Margin Bin", keys_n4),
        ("N5", "N4 + Exact 100-Vote Profile", keys_n5),
    ];

    // Load available model probability matrices
    let mut conditions: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    conditions.insert("00_human_empirical".to_string(), p_human.clone());

    let mean_p0 = p_human.iter().map(|v| v[0]).sum::<f64>() / (n as f64);
    let mean_p1 = p_human.iter().map(|v| v[1]).sum::<f64>() / (n as f64);
    let mean_p2 = p_human.iter().map(|v| v[2]).sum::<f64>() / (n as f64);
    conditions.insert("01_global_class_prior".to_string(), vec![vec![mean_p0, mean_p1, mean_p2]; n]);

    // Load 9 canonical models from model_probs.json
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

    for m_name in &canonical_models {
        let full_p = full_model_probs.get(*m_name).unwrap_or_else(|| panic!("Missing model prediction key: {}", m_name));
        let mut sliced = Vec::with_capacity(n);
        for &r_idx in &row_indices {
            assert!(r_idx < full_p.len(), "Row index {} out of range for model {}", r_idx, m_name);
            sliced.push(full_p[r_idx].clone());
        }
        conditions.insert(format!("model_{}", m_name), sliced);
    }

    // Add E003 equal 3-model ensemble (BART + RoBERTa + XLNet)
    if let (Some(p_b), Some(p_r), Some(p_x)) = (
        conditions.get("model_bart-large"),
        conditions.get("model_roberta-large"),
        conditions.get("model_xlnet-large"),
    ) {
        let mut ens3 = vec![vec![0.0; 3]; n];
        for i in 0..n {
            for c in 0..3 {
                ens3[i][c] = (p_b[i][c] + p_r[i][c] + p_x[i][c]) / 3.0;
            }
        }
        conditions.insert("ensemble_e003_anchor_3model".to_string(), ens3);
    }

    let mut condition_results = HashMap::new();

    for (cond_name, q_mod) in &conditions {
        println!("\n--- Evaluating Null Ladder for Condition: {} ---", cond_name);

        let mut nll_sum = 0.0;
        let mut jsd_sum = 0.0;
        for i in 0..n {
            nll_sum += soft_label_nll(&p_human[i], &q_mod[i]);
            jsd_sum += jsd(&p_human[i], &q_mod[i]);
        }
        let mean_nll = nll_sum / (n as f64);
        let mean_jsd = jsd_sum / (n as f64);

        let dist_mod = distance_hellinger_matrix(q_mod);
        let w_mod = compute_topk_weight_matrix(&dist_mod, 10);

        let mut ladder_results = Vec::new();

        for (lvl_id, lvl_name, keys) in &null_levels {
            let res = compute_conditional_null(lvl_id, lvl_name, &w_mod, &s_k10, keys, 10000, 42, 10);
            let inf_str = if res.is_informative { "true" } else { "NON-INFORMATIVE (0 movable)" };
            println!(
                "  [{}] {:<42} | Groups: {:>3} | Informative: {:<25} | Q_null: {:.5} | Excess: {:+.5} | p: {:.4}",
                res.level_id, res.level_name, res.n_groups, inf_str, res.null_mean, res.q_excess, res.p_value_monte_carlo
            );
            ladder_results.push(res);
        }

        let q_obs = ladder_results[0].q_observed;

        condition_results.insert(
            cond_name.clone(),
            ConditionLadderResult {
                condition_name: cond_name.clone(),
                nll: mean_nll,
                jsd_bits: mean_jsd,
                q_observed: q_obs,
                null_ladder: ladder_results,
            },
        );
    }

    let summary = E005Summary {
        experiment_id: "E005".to_string(),
        title: "Strictly Nested Conditional Null Ladder".to_string(),
        subset: subset.to_string(),
        object_count: n,
        q_hh_relational: q_hh,
        conditions: condition_results,
    };

    let rel_out_dir = format!("research/chaosnli/artifacts/E005/summaries");
    let out_dir_str = resolve_path(&rel_out_dir);
    let out_dir = Path::new(&out_dir_str);
    std::fs::create_dir_all(out_dir)?;
    let out_path = out_dir.join("E005_summary.json");
    let out_file = File::create(&out_path)?;
    serde_json::to_writer_pretty(out_file, &summary)?;

    println!("\n=========================================================================");
    println!("Saved E005 summary JSON to {}", out_path.display());
    println!("=========================================================================");

    Ok(())
}
