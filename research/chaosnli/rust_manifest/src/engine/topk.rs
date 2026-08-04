/// Soft Tie-Aware Top-K Neighbor Weight Matrix Computation

pub fn compute_topk_weight_matrix(dist: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let n = dist.len();
    let atol = 1e-7;
    let mut w = vec![vec![0.0; n]; n];

    for i in 0..n {
        let mut row_dists: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, dist[i][j]))
            .collect();
        row_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        if row_dists.is_empty() {
            continue;
        }

        let k_eff = k.min(row_dists.len());
        let k_dist = row_dists[k_eff - 1].1;

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
            (k as f64 - n_closer as f64) / (n_tied as f64)
        } else {
            0.0
        };

        for &(j, d) in &row_dists {
            if d < k_dist - atol {
                w[i][j] = 1.0;
            } else if (d - k_dist).abs() <= atol {
                w[i][j] = frac;
            }
        }
    }

    w
}

pub fn evaluate_q_support(w_model: &[Vec<f64>], s_target: &[Vec<f64>], k: usize) -> f64 {
    let n = w_model.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            sum += w_model[i][j] * s_target[i][j];
        }
    }
    sum / (n as f64 * k as f64)
}
