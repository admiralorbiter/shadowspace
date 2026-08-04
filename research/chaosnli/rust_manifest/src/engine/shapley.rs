/// Complete Ensemble Census & Exact Shapley Value Attribution Engine (E007)
/// Computes exact dataset-stratified analytic expected null Q_null(A) per coalition.

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
    pub q_null_analytic: f64,
    pub q_excess: f64,
    pub r_normalized: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapleyResult {
    pub model_name: String,
    pub shapley_r_normalized: f64,
    pub shapley_nll_reduction: f64,
    pub shapley_q_support: f64,
    pub shapley_q_excess: f64,
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
    source_datasets: &[String],
    q_hh: f64,
    q_prior_nll: f64,
    k: usize,
) -> (Vec<SubsetResult>, Vec<ShapleyResult>) {
    let m_num = model_names.len();
    let n = p_human.len();
    let total_subsets = 1 << m_num; // 2^M

    // Precompute dataset stratum membership (0 for SNLI, 1 for MNLI/other)
    let is_snli: Vec<bool> = source_datasets.iter().map(|d| d.to_lowercase().contains("snli")).collect();
    let n_snli = is_snli.iter().filter(|&&b| b).count();
    let n_mnli = n - n_snli;

    // Precompute block mass sums of target matrix S_target
    let mut s_snli_snli = 0.0;
    let mut s_snli_mnli = 0.0;
    let mut s_mnli_snli = 0.0;
    let mut s_mnli_mnli = 0.0;

    for r in 0..n {
        for c in 0..n {
            if r == c {
                continue;
            }
            let s_val = s_target[r][c];
            if is_snli[r] && is_snli[c] {
                s_snli_snli += s_val;
            } else if is_snli[r] && !is_snli[c] {
                s_snli_mnli += s_val;
            } else if !is_snli[r] && is_snli[c] {
                s_mnli_snli += s_val;
            } else {
                s_mnli_mnli += s_val;
            }
        }
    }

    let mut subset_results: Vec<Option<SubsetResult>> = vec![None; total_subsets];
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

            // Relational support matrix W_ens
            let dist_ens = distance_hellinger_matrix(&q_ens);
            let w_ens = compute_topk_weight_matrix(&dist_ens, k);
            let q_supp = evaluate_q_support(&w_ens, s_target, k);

            // Compute exact coalition-specific analytic stratified expected null
            let mut w_snli_snli = 0.0;
            let mut w_snli_mnli = 0.0;
            let mut w_mnli_snli = 0.0;
            let mut w_mnli_mnli = 0.0;

            for r in 0..n {
                for c in 0..n {
                    if r == c {
                        continue;
                    }
                    let w_val = w_ens[r][c];
                    if w_val > 1e-12 {
                        if is_snli[r] && is_snli[c] {
                            w_snli_snli += w_val;
                        } else if is_snli[r] && !is_snli[c] {
                            w_snli_mnli += w_val;
                        } else if !is_snli[r] && is_snli[c] {
                            w_mnli_snli += w_val;
                        } else {
                            w_mnli_mnli += w_val;
                        }
                    }
                }
            }

            let e_w_ss = if n_snli > 1 { w_snli_snli / (n_snli * (n_snli - 1)) as f64 } else { 0.0 };
            let e_w_sm = if n_snli > 0 && n_mnli > 0 { w_snli_mnli / (n_snli * n_mnli) as f64 } else { 0.0 };
            let e_w_ms = if n_snli > 0 && n_mnli > 0 { w_mnli_snli / (n_mnli * n_snli) as f64 } else { 0.0 };
            let e_w_mm = if n_mnli > 1 { w_mnli_mnli / (n_mnli * (n_mnli - 1)) as f64 } else { 0.0 };

            let q_null_analytic = (e_w_ss * s_snli_snli + e_w_sm * s_snli_mnli + e_w_ms * s_mnli_snli + e_w_mm * s_mnli_mnli) / ((n * k) as f64);
            let q_exc = q_supp - q_null_analytic;
            let r_norm = (q_supp - q_null_analytic) / (q_hh - q_null_analytic).max(1e-12);

            let res = SubsetResult {
                subset_mask: mask,
                subset_size: size,
                model_names: active_models,
                nll: mean_nll,
                jsd_bits: mean_jsd,
                q_support: q_supp,
                q_null_analytic,
                q_excess: q_exc,
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
        let mut phi_exc = 0.0;

        for mask in 0..total_subsets {
            if (mask & bit) == 0 {
                // Subset A does NOT contain model i
                let a_size = (0..m_num).filter(|&j| (mask & (1 << j)) != 0).count();
                let weight = (factorial(a_size) * factorial(m_num - a_size - 1)) / m_fact;

                let v_a_r = if mask == 0 { 0.0 } else { subset_results[mask].as_ref().unwrap().r_normalized };
                let v_ai_r = subset_results[mask | bit].as_ref().unwrap().r_normalized;

                // NLL reduction characteristic v_NLL(A) = NLL(q_prior) - NLL(q_A) with v_NLL(empty) = 0
                let v_a_nll = if mask == 0 { 0.0 } else { q_prior_nll - subset_results[mask].as_ref().unwrap().nll };
                let v_ai_nll = q_prior_nll - subset_results[mask | bit].as_ref().unwrap().nll;

                let v_a_q = if mask == 0 { 0.0 } else { subset_results[mask].as_ref().unwrap().q_support };
                let v_ai_q = subset_results[mask | bit].as_ref().unwrap().q_support;

                let v_a_exc = if mask == 0 { 0.0 } else { subset_results[mask].as_ref().unwrap().q_excess };
                let v_ai_exc = subset_results[mask | bit].as_ref().unwrap().q_excess;

                phi_r += weight * (v_ai_r - v_a_r);
                phi_nll += weight * (v_ai_nll - v_a_nll);
                phi_q += weight * (v_ai_q - v_a_q);
                phi_exc += weight * (v_ai_exc - v_a_exc);
            }
        }

        shapley_results.push(ShapleyResult {
            model_name: m_name.clone(),
            shapley_r_normalized: phi_r,
            shapley_nll_reduction: phi_nll,
            shapley_q_support: phi_q,
            shapley_q_excess: phi_exc,
        });
    }

    // Verify exact Shapley Efficiency assertions
    let grand_mask = total_subsets - 1;
    let grand_res = subset_results[grand_mask].as_ref().unwrap();

    let sum_phi_r: f64 = shapley_results.iter().map(|s| s.shapley_r_normalized).sum();
    let sum_phi_nll: f64 = shapley_results.iter().map(|s| s.shapley_nll_reduction).sum();
    let sum_phi_exc: f64 = shapley_results.iter().map(|s| s.shapley_q_excess).sum();

    let grand_nll_reduction = q_prior_nll - grand_res.nll;

    assert!(
        (sum_phi_r - grand_res.r_normalized).abs() < 1e-4,
        "Shapley R efficiency check failed: sum={:.5}, grand={:.5}",
        sum_phi_r,
        grand_res.r_normalized
    );
    assert!(
        (sum_phi_nll - grand_nll_reduction).abs() < 1e-4,
        "Shapley NLL efficiency check failed: sum={:.5}, grand={:.5}",
        sum_phi_nll,
        grand_nll_reduction
    );
    assert!(
        (sum_phi_exc - grand_res.q_excess).abs() < 1e-4,
        "Shapley Q_excess efficiency check failed: sum={:.5}, grand={:.5}",
        sum_phi_exc,
        grand_res.q_excess
    );

    let all_subsets: Vec<SubsetResult> = subset_results.into_iter().filter_map(|x| x).collect();
    (all_subsets, shapley_results)
}
