/// Complete Ensemble Census & Exact Shapley Value Attribution Engine (E007)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::engine::distance::{distance_hellinger_matrix, jsd, soft_label_nll};
use crate::engine::topk::{compute_topk_weight_matrix, evaluate_q_support};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubsetResult {
    pub subset_mask: usize,
    pub subset_size: usize,
    pub model_names: Vec<String>,
    pub nll: f64,
    pub jsd_bits: f64,
    pub q_support: f64,
    pub r_normalized: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapleyResult {
    pub model_name: String,
    pub shapley_r_normalized: f64,
    pub shapley_nll_reduction: f64,
    pub shapley_q_support: f64,
}

fn factorial(n: usize) -> f64 {
    match n {
        0 | 1 => 1.0,
        2 => 2.0,
        3 => 6.0,
        4 => 24.0,
        5 => 120.0,
        6 => 720.0,
        7 => 5040.0,
        8 => 40320.0,
        9 => 362880.0,
        _ => (1..=n).map(|x| x as f64).product(),
    }
}

pub fn compute_ensemble_census_and_shapley(
    model_names: &[String],
    model_probs: &HashMap<String, Vec<Vec<f64>>>,
    p_human: &[Vec<f64>],
    s_target: &[Vec<f64>],
    q_hh: f64,
    q_null: f64,
    k: usize,
) -> (Vec<SubsetResult>, Vec<ShapleyResult>) {
    let m_num = model_names.len();
    let n = p_human.len();
    let total_subsets = 1 << m_num; // 2^M

    let mut subset_results: Vec<Option<SubsetResult>> = vec![None; total_subsets];

    // Compute metrics for all 2^M - 1 non-empty subsets
    let indices: Vec<usize> = (1..total_subsets).collect();

    let computed: Vec<(usize, SubsetResult)> = indices
        .into_iter()
        .map(|mask| {
            let active_models: Vec<String> = (0..m_num)
                .filter(|&i| (mask & (1 << i)) != 0)
                .map(|i| model_names[i].clone())
                .collect();
            let size = active_models.len();

            // Compute ensemble equal average probability matrix
            let mut q_ens = vec![vec![0.0; 3]; n];
            for item_idx in 0..n {
                for m_name in &active_models {
                    let p = &model_probs[m_name][item_idx];
                    for c in 0..3 {
                        q_ens[item_idx][c] += p[c];
                    }
                }
                for c in 0..3 {
                    q_ens[item_idx][c] /= size as f64;
                }
            }

            // NLL and JSD
            let mut total_nll = 0.0;
            let mut total_jsd = 0.0;
            for item_idx in 0..n {
                total_nll += soft_label_nll(&p_human[item_idx], &q_ens[item_idx]);
                total_jsd += jsd(&p_human[item_idx], &q_ens[item_idx]);
            }
            let mean_nll = total_nll / (n as f64);
            let mean_jsd = total_jsd / (n as f64);

            // Relational support
            let dist_ens = distance_hellinger_matrix(&q_ens);
            let w_ens = compute_topk_weight_matrix(&dist_ens, k);
            let q_supp = evaluate_q_support(&w_ens, s_target, k);

            // R normalized
            let r_norm = (q_supp - q_null) / (q_hh - q_null).max(1e-12);

            let res = SubsetResult {
                subset_mask: mask,
                subset_size: size,
                model_names: active_models,
                nll: mean_nll,
                jsd_bits: mean_jsd,
                q_support: q_supp,
                r_normalized: r_norm,
            };

            (mask, res)
        })
        .collect();

    for (mask, res) in computed {
        subset_results[mask] = Some(res);
    }

    // Compute exact Shapley values for each model
    let m_fact = factorial(m_num);
    let mut shapley_results = Vec::new();

    for (i, m_name) in model_names.iter().enumerate() {
        let bit = 1 << i;
        let mut phi_r = 0.0;
        let mut phi_nll = 0.0;
        let mut phi_q = 0.0;

        for mask in 0..total_subsets {
            if (mask & bit) == 0 {
                // Subset A does NOT contain model i
                let a_size = (0..m_num).filter(|&j| (mask & (1 << j)) != 0).count();
                let weight = (factorial(a_size) * factorial(m_num - a_size - 1)) / m_fact;

                let mask_with_i = mask | bit;

                let r_a = if mask == 0 { 0.0 } else { subset_results[mask].as_ref().unwrap().r_normalized };
                let r_a_i = subset_results[mask_with_i].as_ref().unwrap().r_normalized;

                let nll_a = if mask == 0 { 10.0 } else { subset_results[mask].as_ref().unwrap().nll };
                let nll_a_i = subset_results[mask_with_i].as_ref().unwrap().nll;

                let q_a = if mask == 0 { q_null } else { subset_results[mask].as_ref().unwrap().q_support };
                let q_a_i = subset_results[mask_with_i].as_ref().unwrap().q_support;

                phi_r += weight * (r_a_i - r_a);
                phi_nll += weight * (nll_a - nll_a_i); // positive for reduction
                phi_q += weight * (q_a_i - q_a);
            }
        }

        shapley_results.push(ShapleyResult {
            model_name: m_name.clone(),
            shapley_r_normalized: phi_r,
            shapley_nll_reduction: phi_nll,
            shapley_q_support: phi_q,
        });
    }

    let flat_subsets: Vec<SubsetResult> = subset_results.into_iter().flatten().collect();

    (flat_subsets, shapley_results)
}
