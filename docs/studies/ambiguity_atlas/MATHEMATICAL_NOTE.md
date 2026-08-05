# Mathematical Foundations of the Ambiguity Doppelgänger Collision

## 1. Parameterization of Mirror Distributions

Let $p \in \Delta^2 = \{ (p_1, p_2, p_3) \in \mathbb{R}^3 : p_i \ge 0, \sum p_i = 1 \}$ be a 3-class probability distribution.

Assume without loss of generality that the first component is the designated majority class, with probability $m = \max(p_1, p_2, p_3) \ge 1/3$.
The total remaining probability assigned to minority classes is $1 - m$.

We parameterize the minority distribution split using a signed orientation parameter $\delta \in [-1, 1]$:
$$p^+(m, \delta) = \left( m, \, \frac{(1-m)(1+\delta)}{2}, \, \frac{(1-m)(1-\delta)}{2} \right)$$

and its mirror distribution:
$$p^-(m, \delta) = \left( m, \, \frac{(1-m)(1-\delta)}{2}, \, \frac{(1-m)(1+\delta)}{2} \right)$$

### Valid Parameter Domain
To maintain class 1 as the majority class ($\max(p^+) = m$):
- If $m \ge 1/2$, any $\delta \in [-1, 1]$ satisfies $\frac{(1-m)(1+|\delta|)}{2} \le 1-m \le m$.
- If $1/3 \le m < 1/2$, we require $\frac{(1-m)(1+|\delta|)}{2} \le m$, which yields $|\delta| \le \frac{3m - 1}{1 - m}$.

---

## 2. Information Invariance under Minority Swapping

### 2.1 Sorted Probability Vector Invariance
The sorted probability vectors of $p^+(m, \delta)$ and $p^-(m, \delta)$ are identical:
$$\text{sort}(p^+(m, \delta)) = \text{sort}(p^-(m, \delta)) = \left( m, \, \frac{(1-m)(1+|\delta|)}{2}, \, \frac{(1-m)(1-|\delta|)}{2} \right)$$

### 2.2 Shannon Entropy Invariance
The Shannon entropy in bits ($H(p) = -\sum p_i \log_2 p_i$) of $p^+(m, \delta)$ is given by:
$$H(m, \delta) = h_2(m) + (1-m) h_2\left( \frac{1+\delta}{2} \right)$$
where $h_2(x) = -x \log_2 x - (1-x) \log_2 (1-x)$ is the binary entropy function.

Since binary entropy is symmetric about $x = 1/2$, $h_2\left(\frac{1+\delta}{2}\right) = h_2\left(\frac{1-\delta}{2}\right)$. Thus:
$$H(m, \delta) = H(m, -\delta)$$

### 2.3 Minority-Swap Collision Theorem
> **Theorem (Minority-Swap Collision)**: In a 3-class probability simplex, fixing the majority class, majority probability $m$, and Shannon entropy $H(p)$ determines the minority probabilities only up to permutation. Specifically, the summary map:
> $$\mathcal{S}: p \mapsto (\text{majority\_label}, \max(p), H(p))$$
> is generally two-to-one on interior non-symmetric distributions ($\delta \neq 0$), collapsing opposite minority disagreement directions into identical scalar summary coordinates.

---

## 3. Exact Distance Formulas Between Mirror Distributions

Let $BC(p^+, p^-)$ be the Bhattacharyya coefficient between mirror distributions:
$$BC = \sqrt{m \cdot m} + \sqrt{\frac{(1-m)(1+\delta)}{2} \frac{(1-m)(1-\delta)}{2}} + \sqrt{\frac{(1-m)(1-\delta)}{2} \frac{(1-m)(1+\delta)}{2}}$$
$$BC = m + (1-m) \sqrt{1 - \delta^2}$$

### 3.1 Hellinger Distance
$$d_H(p^+, p^-) = \sqrt{1 - BC} = \sqrt{(1-m)\left(1 - \sqrt{1-\delta^2}\right)}$$

### 3.2 Fisher–Rao Geodesic Distance
$$d_{FR}(p^+, p^-) = 2 \arccos(BC) = 2 \arccos\left( m + (1-m)\sqrt{1-\delta^2} \right)$$

### 3.3 Jensen–Shannon Divergence & Distance
The midpoint distribution is $M = \frac{p^+ + p^-}{2} = \left( m, \, \frac{1-m}{2}, \, \frac{1-m}{2} \right)$.
The Jensen–Shannon divergence in bits is:
$$JS(p^+, p^-) = H(M) - \frac{H(p^+) + H(p^-)}{2} = (1-m) \left[ 1 - h_2\left( \frac{1+\delta}{2} \right) \right]$$
The Jensen–Shannon distance is $d_{JS}(p^+, p^-) = \sqrt{JS(p^+, p^-)}$.

### 3.4 Aitchison Simplex Distance
For interior distributions ($p_i > 0$), using centered log-ratio (clr) coordinates:
$$d_A(p^+, p^-) = \sqrt{2} \left| \log \frac{1-\delta}{1+\delta} \right|$$
At simplex boundaries where $p_i = 0$, Dirichlet count smoothing ($\alpha = 0.5$) is applied prior to Aitchison distance calculation.
