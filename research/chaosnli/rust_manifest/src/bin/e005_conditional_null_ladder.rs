/// E005: Conditional Null Ladder Experiment
/// Evaluates the 6-level hierarchical null ladder N0..N5 across all model conditions.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use chaosnli_engine::distance::{distance_hellinger_matrix, jsd, soft_label_nll};
use chaosnli_engine::nulls::{compute_conditional_null, NullResult};
use chaosnli_engine::topk::compute_topk_weight_matrix;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize)]
struct ManifestItem {
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
    rel_path.to_string()
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
    println!("   E005: CONDITIONAL NULL LADDER (Subset: {})", subset.to_uppercase());
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
        keys_n2[i] = format!("{}_{}", d, maj);
        keys_n3[i] = format!("{}_{}_{}", d, maj, eq);
        keys_n4[i] = format!("{}_{}_{}_{}", d, top1, top2, margin_bin);
        keys_n5[i] = format!("{}_{}_{}", it.human_count_entailment, it.human_count_neutral, it.human_count_contradiction);
    }

    let null_levels = vec![
        ("N0", "Global Identity Permutation", keys_n0),
        ("N1", "Dataset Stratified (SNLI/MNLI)", keys_n1),
        ("N2", "Dataset x Majority Label", keys_n2),
        ("N3", "Dataset x Majority Label x Entropy Quintile", keys_n3),
        ("N4", "Dataset x Top-2 Label Pair x Margin Bin", keys_n4),
        ("N5", "Exact 100-Vote Profile", keys_n5),
    ];

    let mut conditions: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    conditions.insert("00_human_empirical".to_string(), p_human.clone());

    let mean_p0 = p_human.iter().map(|v| v[0]).sum::<f64>() / (n as f64);
    let mean_p1 = p_human.iter().map(|v| v[1]).sum::<f64>() / (n as f64);
    let mean_p2 = p_human.iter().map(|v| v[2]).sum::<f64>() / (n as f64);
    conditions.insert("01_global_class_prior".to_string(), vec![vec![mean_p0, mean_p1, mean_p2]; n]);

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
            println!(
                "  [{}] {:<42} | Groups: {:>3} | Informative: {:<5} | Q_null: {:.5} | Excess: {:+.5} | p: {:.4}",
                res.level_id, res.level_name, res.n_groups, res.is_informative, res.null_mean, res.q_excess, res.p_value_monte_carlo
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
        title: "Conditional Null Ladder".to_string(),
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
