/// E007: Ultra-Fast Genuine 5-Fold Cross-Fitted Coalition Selection Engine (Rust Rayon)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

#[derive(Debug, Clone, Deserialize)]
struct ManifestItem {
    row_index: usize,
    source_dataset: String,
    human_p_entailment: f64,
    human_p_neutral: f64,
    human_p_contradiction: f64,
}

#[derive(Debug, Clone, Serialize)]
struct FoldRecord {
    fold: usize,
    coalition_size: usize,
    selected_mask: usize,
    selected_models: Vec<String>,
    train_q_support: f64,
    held_out_r_normalized: f64,
    held_out_nll: f64,
    n_train: usize,
    n_held_out: usize,
}

#[derive(Debug, Clone, Serialize)]
struct SizeSummary {
    coalition_size: usize,
    selected_models: Vec<String>,
    held_out_r_normalized_mean: f64,
    held_out_nll_mean: f64,
    top_mask_selection_frequency: f64,
}

#[derive(Debug, Clone, Serialize)]
struct CrossfitOutput {
    n_folds: usize,
    method: String,
    held_out_summary_by_size: Vec<SizeSummary>,
    fold_details: Vec<FoldRecord>,
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
    rel_path.to_string()
}

fn distance_hellinger(p: &[f64; 3], q: &[f64; 3]) -> f64 {
    let bc = (p[0] * q[0]).sqrt() + (p[1] * q[1]).sqrt() + (p[2] * q[2]).sqrt();
    let bc_clamped = bc.clamp(0.0, 1.0);
    (1.0 - bc_clamped).max(0.0).sqrt()
}

fn compute_q_support_fast(dist: &[f64], s_mat: &[f64], n: usize, k: usize) -> f64 {
    let mut sum_q = 0.0f64;
    let atol = 1e-7;

    for i in 0..n {
        let i_off = i * n;
        let mut row_dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, dist[i_off + j]))
            .collect();

        row_dists.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        let k_dist = row_dists[k - 1].1;

        let mut n_closer = 0;
        let mut n_tied = 0;
        for &(_, d) in &row_dists {
            if d < k_dist - atol {
                n_closer += 1;
            } else if (d - k_dist).abs() <= atol {
                n_tied += 1;
            }
        }

        let frac = if n_tied > 0 {
            (k - n_closer) as f64 / n_tied as f64
        } else {
            0.0
        };

        for &(j, d) in &row_dists {
            if d < k_dist - atol {
                sum_q += s_mat[i_off + j];
            } else if (d - k_dist).abs() <= atol {
                sum_q += frac * s_mat[i_off + j];
            }
        }
    }

    sum_q / (n * k) as f64
}

fn main() {
    println!("=========================================================================");
    println!("   E007: GENUINE 5-FOLD CROSS-FITTED COALITION SELECTION ENGINE (RUST)");
    println!("=========================================================================");

    let manifest_path = resolve_path("research/chaosnli/artifacts/E004/manifests/full_3113.jsonl");
    let target_bin_path = resolve_path("research/chaosnli/artifacts/E001/S_hellinger_k010.bin");
    let model_probs_path = resolve_path("research/chaosnli/rust_manifest/model_probs.json");

    let file = File::open(&manifest_path).expect("Failed to open manifest file");
    let reader = BufReader::new(file);

    let mut items: Vec<ManifestItem> = Vec::new();
    for line in reader.lines() {
        let l = line.expect("Failed to read line");
        if !l.trim().is_empty() {
            let item: ManifestItem = serde_json::from_str(&l).expect("Failed to parse JSONL");
            items.push(item);
        }
    }
    items.sort_by_key(|it| it.row_index);
    let n_items = items.len();

    let bin_bytes = std::fs::read(&target_bin_path).expect("Failed to read binary target matrix");
    let s_target: Vec<f64> = bin_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()) as f64)
        .collect();

    let model_file = File::open(&model_probs_path).expect("Failed to open model_probs.json");
    let model_map: HashMap<String, Vec<[f64; 3]>> = serde_json::from_reader(BufReader::new(model_file)).expect("Failed to parse model_probs.json");

    let canonical_models = vec![
        "bart-large", "roberta-large", "xlnet-large", "albert-xxlarge",
        "bert-large", "roberta-base", "xlnet-base", "distilbert", "bert-base"
    ];

    let m_num = canonical_models.len();
    let model_mats: Vec<Vec<[f64; 3]>> = canonical_models.iter().map(|m| model_map.get(*m).unwrap().clone()).collect();
    let total_subsets = (1 << m_num) - 1;

    // Build 5 stratified fold indices
    let n_folds = 5;
    let mut item_fold_ids = vec![0usize; n_items];
    let mut strata_map: HashMap<String, Vec<usize>> = HashMap::new();

    for (i, it) in items.iter().enumerate() {
        let p = [it.human_p_entailment, it.human_p_neutral, it.human_p_contradiction];
        let maj = if p[0] >= p[1] && p[0] >= p[2] { 0 } else if p[1] >= p[2] { 1 } else { 2 };
        let d_str = if it.source_dataset.contains("snli") { "snli" } else { "mnli" };
        let key = format!("{}_{}", d_str, maj);
        strata_map.entry(key).or_default().push(i);
    }

    for (_, idxs) in strata_map.iter() {
        for (rank, &idx) in idxs.iter().enumerate() {
            item_fold_ids[idx] = rank % n_folds;
        }
    }

    let mut fold_records: Vec<FoldRecord> = Vec::new();
    let mut selected_masks_counts: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for sz in 1..=m_num {
        selected_masks_counts.insert(sz, HashMap::new());
    }

    for fold in 0..n_folds {
        println!("Executing Fold {}/{}...", fold + 1, n_folds);
        let train_indices: Vec<usize> = (0..n_items).filter(|&i| item_fold_ids[i] != fold).collect();
        let val_indices: Vec<usize> = (0..n_items).filter(|&i| item_fold_ids[i] == fold).collect();
        let n_train = train_indices.len();
        let n_val = val_indices.len();

        let mut s_tr = vec![0.0f64; n_train * n_train];
        for r in 0..n_train {
            let r_idx = train_indices[r];
            for c in 0..n_train {
                let c_idx = train_indices[c];
                s_tr[r * n_train + c] = s_target[r_idx * n_items + c_idx];
            }
        }

        let mut s_val = vec![0.0f64; n_val * n_val];
        for r in 0..n_val {
            let r_idx = val_indices[r];
            for c in 0..n_val {
                let c_idx = val_indices[c];
                s_val[r * n_val + c] = s_target[r_idx * n_items + c_idx];
            }
        }

        let best_train: Vec<(usize, f64, usize, Vec<usize>)> = (1..=total_subsets)
            .into_par_iter()
            .map(|mask| {
                let active: Vec<usize> = (0..m_num).filter(|&i| (mask & (1 << i)) != 0).collect();
                let sz = active.len();

                let mut q_tr = vec![[0.0f64; 3]; n_train];
                for &m_idx in &active {
                    for r in 0..n_train {
                        let p = model_mats[m_idx][train_indices[r]];
                        q_tr[r][0] += p[0];
                        q_tr[r][1] += p[1];
                        q_tr[r][2] += p[2];
                    }
                }
                for r in 0..n_train {
                    q_tr[r][0] /= sz as f64;
                    q_tr[r][1] /= sz as f64;
                    q_tr[r][2] /= sz as f64;
                }

                let mut dist_tr = vec![0.0f64; n_train * n_train];
                for i in 0..n_train {
                    for j in 0..n_train {
                        if j != i {
                            dist_tr[i * n_train + j] = distance_hellinger(&q_tr[i], &q_tr[j]);
                        }
                    }
                }

                let q_supp_tr = compute_q_support_fast(&dist_tr, &s_tr, n_train, 10);
                (sz, q_supp_tr, mask, active)
            })
            .collect();

        let mut best_by_sz: HashMap<usize, (f64, usize, Vec<usize>)> = HashMap::new();
        for (sz, q_supp, mask, active) in best_train {
            if !best_by_sz.contains_key(&sz) || q_supp > best_by_sz[&sz].0 {
                best_by_sz.insert(sz, (q_supp, mask, active));
            }
        }

        for sz in 1..=m_num {
            let (q_supp_tr, win_mask, win_indices) = best_by_sz[&sz].clone();
            let win_models: Vec<String> = win_indices.iter().map(|&i| canonical_models[i].to_string()).collect();

            *selected_masks_counts.get_mut(&sz).unwrap().entry(win_mask).or_default() += 1;

            let mut q_val = vec![[0.0f64; 3]; n_val];
            let mut val_nll = 0.0f64;
            for r in 0..n_val {
                let item_r = &items[val_indices[r]];
                let p_h = [item_r.human_p_entailment, item_r.human_p_neutral, item_r.human_p_contradiction];
                for &m_idx in &win_indices {
                    let p = model_mats[m_idx][val_indices[r]];
                    q_val[r][0] += p[0];
                    q_val[r][1] += p[1];
                    q_val[r][2] += p[2];
                }
                q_val[r][0] /= sz as f64;
                q_val[r][1] /= sz as f64;
                q_val[r][2] /= sz as f64;

                let nll_r = -(p_h[0] * q_val[r][0].max(1e-12).ln() + p_h[1] * q_val[r][1].max(1e-12).ln() + p_h[2] * q_val[r][2].max(1e-12).ln());
                val_nll += nll_r;
            }
            val_nll /= n_val as f64;

            let mut dist_val = vec![0.0f64; n_val * n_val];
            for i in 0..n_val {
                for j in 0..n_val {
                    if j != i {
                        dist_val[i * n_val + j] = distance_hellinger(&q_val[i], &q_val[j]);
                    }
                }
            }

            let q_supp_val = compute_q_support_fast(&dist_val, &s_val, n_val, 10);
            let sum_s_val: f64 = s_val.iter().sum();
            let q_null_val = sum_s_val * (10.0 / (n_val - 1) as f64) / (n_val * 10) as f64;
            let r_norm_val = (q_supp_val - q_null_val) / (0.038987226 - q_null_val);

            fold_records.push(FoldRecord {
                fold,
                coalition_size: sz,
                selected_mask: win_mask,
                selected_models: win_models,
                train_q_support: q_supp_tr,
                held_out_r_normalized: r_norm_val,
                held_out_nll: val_nll,
                n_train,
                n_held_out: n_val,
            });
        }
    }

    let mut summary_by_size = Vec::new();
    for sz in 1..=m_num {
        let f_sz: Vec<&FoldRecord> = fold_records.iter().filter(|r| r.coalition_size == sz).collect();
        let mean_r = f_sz.iter().map(|r| r.held_out_r_normalized).sum::<f64>() / n_folds as f64;
        let mean_nll = f_sz.iter().map(|r| r.held_out_nll).sum::<f64>() / n_folds as f64;

        let counts = &selected_masks_counts[&sz];
        let top_mask = counts.iter().max_by_key(|x| x.1).unwrap().0;
        let top_freq = *counts.get(top_mask).unwrap() as f64 / n_folds as f64;
        let top_models: Vec<String> = (0..m_num)
            .filter(|i| (top_mask & (1 << i)) != 0)
            .map(|i| canonical_models[i].to_string())
            .collect();

        summary_by_size.push(SizeSummary {
            coalition_size: sz,
            selected_models: top_models,
            held_out_r_normalized_mean: mean_r,
            held_out_nll_mean: mean_nll,
            top_mask_selection_frequency: top_freq,
        });
    }

    let output = CrossfitOutput {
        n_folds,
        method: "genuine_5fold_stratified_train_selection_heldout_evaluation".to_string(),
        held_out_summary_by_size: summary_by_size.clone(),
        fold_details: fold_records,
    };

    let out_file = resolve_path("research/chaosnli/results/E007_held_out_selection.json");
    let writer = File::create(&out_file).expect("Failed to create output file");
    serde_json::to_writer_pretty(writer, &output).expect("Failed to write output JSON");

    println!("\nHeld-Out Cross-Fitted Coalition Performance by Size:");
    for s in &summary_by_size {
        println!("  Size {}: Held-Out R_norm = {:>6.2}% | Held-Out NLL = {:.4} | Selection Freq = {:.0}% | Models: {:?}",
            s.coalition_size, s.held_out_r_normalized_mean * 100.0, s.held_out_nll_mean, s.top_mask_selection_frequency * 100.0, s.selected_models);
    }
    println!("\nSaved GENUINE E007 cross-fitted selection results to {}", out_file);
}
