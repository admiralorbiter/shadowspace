/// Distance and Divergence Functions for ChaosNLI Distributions

pub fn distance_hellinger(p: &[f64], q: &[f64]) -> f64 {
    let mut bc = 0.0;
    for i in 0..p.len() {
        let pi = p[i].max(1e-12).min(1.0);
        let qi = q[i].max(1e-12).min(1.0);
        bc += (pi * qi).sqrt();
    }
    bc = bc.max(0.0).min(1.0);
    (1.0 - bc).max(0.0).sqrt()
}

pub fn distance_hellinger_matrix(probs: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = probs.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let d = distance_hellinger(&probs[i], &probs[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

pub fn jsd(p: &[f64], q: &[f64]) -> f64 {
    let mut m = vec![0.0; p.len()];
    for i in 0..p.len() {
        m[i] = 0.5 * (p[i] + q[i]);
    }
    let mut kl_pm = 0.0;
    let mut kl_qm = 0.0;
    for i in 0..p.len() {
        let pi = p[i].max(1e-12).min(1.0);
        let qi = q[i].max(1e-12).min(1.0);
        let mi = m[i].max(1e-12).min(1.0);
        if p[i] > 1e-12 {
            kl_pm += pi * (pi / mi).log2();
        }
        if q[i] > 1e-12 {
            kl_qm += qi * (qi / mi).log2();
        }
    }
    (0.5 * kl_pm + 0.5 * kl_qm).max(0.0)
}

pub fn soft_label_nll(p_human: &[f64], q_model: &[f64]) -> f64 {
    let mut nll = 0.0;
    for i in 0..p_human.len() {
        let qi = q_model[i].max(1e-12).min(1.0);
        nll -= p_human[i] * qi.ln();
    }
    nll
}

pub fn brier_score(p: &[f64], q: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 0..p.len() {
        let diff = p[i] - q[i];
        s += diff * diff;
    }
    s
}
