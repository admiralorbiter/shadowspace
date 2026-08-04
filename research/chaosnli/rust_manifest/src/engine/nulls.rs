/// Hierarchical Conditional Null Permutation Engine (E005)
/// Optimized with O(N*k) sparse evaluation for fast execution on full N=3113 datasets.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::engine::topk::evaluate_q_support;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NullResult {
    pub level_id: String,
    pub level_name: String,
    pub n_groups: usize,
    pub n_singleton_groups: usize,
    pub n_non_singleton_groups: usize,
    pub effective_movable_items: usize,
    pub max_group_size: usize,
    pub is_informative: bool,
    pub q_observed: f64,
    pub null_mean: f64,
    pub null_ci_95: (f64, f64),
    pub q_excess: f64,
    pub p_value_monte_carlo: f64,
}

#[derive(Debug, Clone)]
struct SparseEntry {
    row: usize,
    col: usize,
    weight: f64,
}

pub fn compute_conditional_null(
    level_id: &str,
    level_name: &str,
    w_model: &[Vec<f64>],
    s_target: &[Vec<f64>],
    group_keys: &[String],
    n_permutations: usize,
    seed: u64,
    k: usize,
) -> NullResult {
    let n = w_model.len();
    let q_observed = evaluate_q_support(w_model, s_target, k);

    let mut group_map: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, key) in group_keys.iter().enumerate() {
        group_map.entry(key.clone()).or_default().push(idx);
    }

    let n_groups = group_map.len();
    let mut n_singleton_groups = 0;
    let mut n_non_singleton_groups = 0;
    let mut effective_movable_items = 0;
    let mut max_group_size = 0;

    let mut non_singleton_groups: Vec<Vec<usize>> = Vec::new();
    for (_key, indices) in group_map {
        let size = indices.len();
        if size > max_group_size {
            max_group_size = size;
        }
        if size == 1 {
            n_singleton_groups += 1;
        } else {
            n_non_singleton_groups += 1;
            effective_movable_items += size;
            non_singleton_groups.push(indices);
        }
    }

    let is_informative = effective_movable_items > 0;

    if !is_informative {
        return NullResult {
            level_id: level_id.to_string(),
            level_name: level_name.to_string(),
            n_groups,
            n_singleton_groups,
            n_non_singleton_groups: 0,
            effective_movable_items: 0,
            max_group_size,
            is_informative: false,
            q_observed,
            null_mean: q_observed,
            null_ci_95: (q_observed, q_observed),
            q_excess: 0.0,
            p_value_monte_carlo: 1.0,
        };
    }

    // Extract non-zero entries of W_model for O(N*k) sparse evaluation
    let mut sparse_entries: Vec<SparseEntry> = Vec::new();
    for r in 0..n {
        for c in 0..n {
            let w = w_model[r][c];
            if w > 1e-12 {
                sparse_entries.push(SparseEntry { row: r, col: c, weight: w });
            }
        }
    }

    let norm_factor = (n as f64) * (k as f64);

    // Parallelized Monte Carlo permutation engine (O(Nk) inner loop)
    let seeds: Vec<u64> = (0..n_permutations).map(|i| seed + i as u64).collect();

    let null_scores: Vec<f64> = seeds
        .into_par_iter()
        .map(|perm_seed| {
            let mut rng = ChaCha8Rng::seed_from_u64(perm_seed);
            let mut perm_idx: Vec<usize> = (0..n).collect();

            for grp in &non_singleton_groups {
                let mut shuffled = grp.clone();
                shuffled.shuffle(&mut rng);
                for (orig_pos, &new_val) in grp.iter().zip(shuffled.iter()) {
                    perm_idx[*orig_pos] = new_val;
                }
            }

            // Compute inverse permutation tau where tau[perm_idx[i]] = i
            let mut inv_perm = vec![0usize; n];
            for i in 0..n {
                inv_perm[perm_idx[i]] = i;
            }

            // Evaluate Q_support in O(N*k) without matrix allocation
            let mut sum_score = 0.0;
            for entry in &sparse_entries {
                let r_inv = inv_perm[entry.row];
                let c_inv = inv_perm[entry.col];
                sum_score += entry.weight * s_target[r_inv][c_inv];
            }

            sum_score / norm_factor
        })
        .collect();

    let mut sorted_scores = null_scores.clone();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let null_mean: f64 = null_scores.iter().sum::<f64>() / (n_permutations as f64);
    let ci_low_idx = ((n_permutations as f64) * 0.025).round() as usize;
    let ci_high_idx = ((n_permutations as f64) * 0.975).round() as usize;
    let null_ci_95 = (
        sorted_scores[ci_low_idx.min(n_permutations - 1)],
        sorted_scores[ci_high_idx.min(n_permutations - 1)],
    );

    let q_excess = q_observed - null_mean;
    let ge_count = null_scores.iter().filter(|&&s| s >= q_observed).count();
    let p_value_monte_carlo = (ge_count as f64 + 1.0) / (n_permutations as f64 + 1.0);

    NullResult {
        level_id: level_id.to_string(),
        level_name: level_name.to_string(),
        n_groups,
        n_singleton_groups,
        n_non_singleton_groups,
        effective_movable_items,
        max_group_size,
        is_informative: true,
        q_observed,
        null_mean,
        null_ci_95,
        q_excess,
        p_value_monte_carlo,
    }
}
