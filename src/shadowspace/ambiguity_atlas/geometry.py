"""Mathematical mirror distributions, entropy identity, and exact geometric metrics."""

import numpy as np
from typing import Tuple, Union

ArrayLike = Union[float, np.ndarray]


def binary_entropy(x: ArrayLike) -> ArrayLike:
    """Compute binary entropy h2(x) in bits.
    
    h2(x) = -x * log2(x) - (1-x) * log2(1-x), with 0 * log2(0) = 0.
    """
    x = np.asarray(x, dtype=np.float64)
    # Clip x to [0, 1] safely
    x_clamped = np.clip(x, 1e-15, 1.0 - 1e-15)
    
    term1 = np.where(x <= 0.0, 0.0, -x * np.log2(x_clamped))
    term2 = np.where(x >= 1.0, 0.0, -(1.0 - x) * np.log2(1.0 - x_clamped))
    res = term1 + term2
    
    # Handle scalar vs array return
    if res.ndim == 0:
        return float(res)
    return res


def mirror_distribution(m: ArrayLike, delta: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mirror probability distributions p+ and p-.
    
    p+(m, delta) = (m, (1-m)(1+delta)/2, (1-m)(1-delta)/2)
    p-(m, delta) = (m, (1-m)(1-delta)/2, (1-m)(1+delta)/2)
    
    Args:
        m: Majority probability, m in [1/3, 1.0]
        delta: Minority orientation parameter, delta in [-1, 1]
        
    Returns:
        p_plus, p_minus as ndarrays of shape (..., 3)
    """
    m_arr = np.asarray(m, dtype=np.float64)
    d_arr = np.asarray(delta, dtype=np.float64)
    
    p1 = m_arr
    p2_plus = (1.0 - m_arr) * (1.0 + d_arr) / 2.0
    p3_plus = (1.0 - m_arr) * (1.0 - d_arr) / 2.0
    
    p2_minus = p3_plus
    p3_minus = p2_plus
    
    p_plus = np.stack([p1, p2_plus, p3_plus], axis=-1)
    p_minus = np.stack([p1, p2_minus, p3_minus], axis=-1)
    
    return p_plus, p_minus


def summary_entropy(m: ArrayLike, delta: ArrayLike) -> ArrayLike:
    """Compute Shannon entropy in bits for mirror distributions p+(m, delta) and p-(m, delta).
    
    H(m, delta) = h2(m) + (1-m) * h2((1+delta)/2)
    """
    m_arr = np.asarray(m, dtype=np.float64)
    d_arr = np.asarray(delta, dtype=np.float64)
    
    h_m = binary_entropy(m_arr)
    h_minority = binary_entropy((1.0 + d_arr) / 2.0)
    
    res = h_m + (1.0 - m_arr) * h_minority
    if res.ndim == 0:
        return float(res)
    return res


def hellinger_mirror_distance(m: ArrayLike, delta: ArrayLike) -> ArrayLike:
    """Analytical Hellinger distance between p+ and p-.
    
    BC = m + (1-m) * sqrt(1 - delta^2)
    d_H = sqrt(1 - BC)
    """
    m_arr = np.asarray(m, dtype=np.float64)
    d_arr = np.asarray(delta, dtype=np.float64)
    
    d_sq = np.clip(d_arr ** 2, 0.0, 1.0)
    bc = m_arr + (1.0 - m_arr) * np.sqrt(1.0 - d_sq)
    d_h = np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))
    
    if d_h.ndim == 0:
        return float(d_h)
    return d_h


def fisher_rao_mirror_distance(m: ArrayLike, delta: ArrayLike) -> ArrayLike:
    """Analytical Fisher-Rao distance between p+ and p-.
    
    d_FR = 2 * arccos(BC)
    """
    m_arr = np.asarray(m, dtype=np.float64)
    d_arr = np.asarray(delta, dtype=np.float64)
    
    d_sq = np.clip(d_arr ** 2, 0.0, 1.0)
    bc = np.clip(m_arr + (1.0 - m_arr) * np.sqrt(1.0 - d_sq), -1.0, 1.0)
    d_fr = 2.0 * np.arccos(bc)
    
    if d_fr.ndim == 0:
        return float(d_fr)
    return d_fr


def js_mirror_distance(m: ArrayLike, delta: ArrayLike) -> ArrayLike:
    """Analytical Jensen-Shannon distance between p+ and p-.
    
    JS = (1-m) * [1 - h2((1+delta)/2)]
    d_JS = sqrt(JS)
    """
    m_arr = np.asarray(m, dtype=np.float64)
    d_arr = np.asarray(delta, dtype=np.float64)
    
    h_minority = binary_entropy((1.0 + d_arr) / 2.0)
    js_div = np.clip((1.0 - m_arr) * (1.0 - h_minority), 0.0, None)
    d_js = np.sqrt(js_div)
    
    if d_js.ndim == 0:
        return float(d_js)
    return d_js


def aitchison_mirror_distance(m: ArrayLike, delta: ArrayLike) -> ArrayLike:
    """Analytical Aitchison simplex distance between interior mirror distributions.
    
    d_A = sqrt(2) * |log((1-delta)/(1+delta))|
    """
    d_arr = np.asarray(delta, dtype=np.float64)
    d_abs = np.abs(d_arr)
    d_clamped = np.clip(d_abs, 0.0, 1.0 - 1e-12)
    
    ratio = (1.0 - d_clamped) / (1.0 + d_clamped)
    d_a = np.sqrt(2.0) * np.abs(np.log(ratio))
    
    if d_a.ndim == 0:
        return float(d_a)
    return d_a


# Direct numerical vector distance functions
def hellinger_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute direct Hellinger distance between 3D probability distributions p and q.
    
    d_H = 1/sqrt(2) * || sqrt(p) - sqrt(q) ||_2 = sqrt(1 - sum(sqrt(p*q)))
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    
    bc = np.sum(np.sqrt(np.clip(p_arr * q_arr, 0.0, None)), axis=-1)
    return np.sqrt(np.clip(1.0 - bc, 0.0, None))


def fisher_rao_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute direct Fisher-Rao geodesic distance between distributions p and q.
    
    d_FR = 2 * arccos(sum(sqrt(p*q)))
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    
    bc = np.clip(np.sum(np.sqrt(np.clip(p_arr * q_arr, 0.0, None)), axis=-1), -1.0, 1.0)
    return 2.0 * np.arccos(bc)


def js_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute direct Jensen-Shannon distance in bits between distributions p and q.
    
    d_JS = sqrt( JS(p, q) )
    """
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    m_arr = 0.5 * (p_arr + q_arr)
    
    def _kl(a, b):
        mask = (a > 0) & (b > 0)
        res = np.zeros_like(a)
        res[mask] = a[mask] * np.log2(a[mask] / b[mask])
        return np.sum(res, axis=-1)
    
    js_div = 0.5 * _kl(p_arr, m_arr) + 0.5 * _kl(q_arr, m_arr)
    return np.sqrt(np.clip(js_div, 0.0, None))


def aitchison_distance(p: np.ndarray, q: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Compute direct Aitchison distance using Dirichlet smoothing for boundary zeroes."""
    p_arr = np.asarray(p, dtype=np.float64)
    q_arr = np.asarray(q, dtype=np.float64)
    
    # Smooth if zero elements exist
    if np.any(p_arr <= 0) or np.any(q_arr <= 0):
        p_arr = p_arr + alpha / 100.0  # soft smoothing
        p_arr = p_arr / np.sum(p_arr, axis=-1, keepdims=True)
        q_arr = q_arr + alpha / 100.0
        q_arr = q_arr / np.sum(q_arr, axis=-1, keepdims=True)
        
    def _clr(x):
        log_x = np.log(x)
        mean_log_x = np.mean(log_x, axis=-1, keepdims=True)
        return log_x - mean_log_x

    clr_p = _clr(p_arr)
    clr_q = _clr(q_arr)
    return np.linalg.norm(clr_p - clr_q, axis=-1)
