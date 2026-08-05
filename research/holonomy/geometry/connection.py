"""Homogeneous Affine Parallel Transport Connection (3x3 Representation).

Estimates and composes affine parallel transport operators in 3x3 homogeneous coordinates:
T = [[A, b], [0, 1]]
Provides both OLS and Total Least Squares (TLS) estimators to correct errors-in-variables attenuation bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


class EstimatorIdentifiabilityError(ValueError):
    """Raised when edge transport design matrix is rank-deficient, ill-conditioned, or unidentifiable."""

    def __init__(
        self,
        msg: str,
        *,
        generator_name: str | None = None,
        reason: str | None = None,
        design_rank: int | None = None,
        required_rank: int | None = None,
        required_observations: int | None = None,
        condition_number: float | None = None,
        condition_threshold: float | None = None,
        v22_condition_number: float | None = None,
        v22_determinant: float | None = None,
        v22_determinant_threshold: float | None = None,
        matrix_norm: float | None = None,
        bias_norm: float | None = None,
        singular_values: List[float] | None = None,
    ) -> None:
        super().__init__(msg)
        self.generator_name = generator_name
        self.reason = reason
        self.design_rank = design_rank
        self.required_rank = required_rank
        self.required_observations = required_observations
        self.condition_number = condition_number
        self.condition_threshold = condition_threshold
        self.v22_condition_number = v22_condition_number
        self.v22_determinant = v22_determinant
        self.v22_determinant_threshold = v22_determinant_threshold
        self.matrix_norm = matrix_norm
        self.bias_norm = bias_norm
        self.singular_values = singular_values or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generator_name": self.generator_name,
            "reason": self.reason,
            "design_rank": self.design_rank,
            "required_rank": self.required_rank,
            "required_observations": self.required_observations,
            "condition_number": self.condition_number,
            "condition_threshold": self.condition_threshold,
            "v22_condition_number": self.v22_condition_number,
            "v22_determinant": self.v22_determinant,
            "v22_determinant_threshold": self.v22_determinant_threshold,
            "matrix_norm": self.matrix_norm,
            "bias_norm": self.bias_norm,
            "singular_values": self.singular_values,
        }


@dataclass
class ParallelTransportMap:
    """Homogeneous parallel transport operator T_g in Aff(2) represented as 3x3 matrix."""

    generator_name: str
    source_id: str
    target_id: str
    matrix_2d: NDArray[np.float64]  # (2, 2) Linear matrix A
    bias_2d: NDArray[np.float64]    # (2,) Translation vector b
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def homogeneous_matrix(self) -> NDArray[np.float64]:
        """Returns 3x3 homogeneous matrix representation [[A, b], [0, 1]]."""
        H = np.eye(3, dtype=np.float64)
        H[:2, :2] = self.matrix_2d
        H[:2, 2] = self.bias_2d
        return H

    def transform(self, vector: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply transport T(v) = A v + b."""
        return np.dot(self.matrix_2d, vector) + self.bias_2d


def whiten_coordinates(coords: NDArray[np.float64], mean: NDArray[np.float64], cov_sqrt_inv: NDArray[np.float64]) -> NDArray[np.float64]:
    """Whitens coordinates z_tilde = (z - mu) @ cov_sqrt_inv."""
    return np.dot(coords - mean, cov_sqrt_inv)


def compute_derived_inverse_map(t_map: ParallelTransportMap, inverse_generator_name: str) -> ParallelTransportMap:
    """Computes exact mathematical affine inverse T_g^{-1} where A^{-1} is linear inverse and b^{-1} = -A^{-1} b."""
    A_inv = np.linalg.inv(t_map.matrix_2d)
    b_inv = -np.dot(A_inv, t_map.bias_2d)
    return ParallelTransportMap(
        generator_name=inverse_generator_name,
        source_id=t_map.target_id,
        target_id=t_map.source_id,
        matrix_2d=A_inv,
        bias_2d=b_inv,
        metadata={**t_map.metadata, "is_derived_inverse": True},
    )


def compute_forward_affine_commutator(t_a: ParallelTransportMap, t_b: ParallelTransportMap) -> Any:
    """Derives exact affine commutator H_gamma = T_b^{-1} T_a^{-1} T_b T_a from forward maps."""
    from research.holonomy.geometry.parallel_transport import PathTransport
    t_a_inv = compute_derived_inverse_map(t_a, f"{t_a.generator_name}_inv")
    t_b_inv = compute_derived_inverse_map(t_b, f"{t_b.generator_name}_inv")
    return PathTransport([t_a, t_b, t_a_inv, t_b_inv])


def compute_holonomy_norm_statistics(path_transport: Any) -> Dict[str, float]:
    """Computes the 3 separate holonomy norm statistics S_A, S_b, S_H."""
    import scipy.linalg
    A_gamma = path_transport.compute_composite_matrix()
    H_hom = path_transport.compute_homogeneous_matrix()
    b_gamma = H_hom[:2, 2]

    # S_A = ||log(A_gamma)||_F
    try:
        log_A = scipy.linalg.logm(A_gamma)
        linear_norm_S_A = float(np.linalg.norm(log_A, "fro"))
    except Exception:
        linear_norm_S_A = float(np.linalg.norm(A_gamma - np.eye(2), "fro"))

    # S_b = ||b_gamma||_2
    translation_norm_S_b = float(np.linalg.norm(b_gamma))

    # S_H = ||H_gamma - I_3||_F
    homogeneous_norm_S_H = float(np.linalg.norm(H_hom - np.eye(3), "fro"))

    return {
        "linear_norm_S_A": linear_norm_S_A,
        "translation_norm_S_b": translation_norm_S_b,
        "homogeneous_norm_S_H": homogeneous_norm_S_H,
    }


def fit_pooled_forward_transports(
    estimator: ConnectionEstimator,
    a_src_list: List[NDArray[np.float64]],
    a_tgt_list: List[NDArray[np.float64]],
    b_src_list: List[NDArray[np.float64]],
    b_tgt_list: List[NDArray[np.float64]],
) -> Tuple[ParallelTransportMap, ParallelTransportMap]:
    """Fits pooled forward generators T_a and T_b combining both square contexts.

    D_a = {(z0, z1), (z3, z2)} for T_a
    D_b = {(z0, z3), (z1, z2)} for T_b
    """
    pooled_a_src = np.vstack(a_src_list)
    pooled_a_tgt = np.vstack(a_tgt_list)
    pooled_b_src = np.vstack(b_src_list)
    pooled_b_tgt = np.vstack(b_tgt_list)

    t_a = estimator.estimate_linear_transport("rename_a_pooled", "x_src", "x_tgt", pooled_a_src, pooled_a_tgt)
    t_b = estimator.estimate_linear_transport("rename_b_pooled", "x_src", "x_tgt", pooled_b_src, pooled_b_tgt)
    return t_a, t_b


def evaluate_edge_predictive_skill(
    t_map: ParallelTransportMap,
    source_coords: NDArray[np.float64],
    target_coords: NDArray[np.float64],
) -> Dict[str, float]:
    """Evaluates predictive skill of affine transport map against Identity and Mean-Shift baselines."""
    preds = np.dot(source_coords, t_map.matrix_2d.T) + t_map.bias_2d
    errors_affine = preds - target_coords
    rmse_affine = float(np.sqrt(np.mean(errors_affine ** 2)))
    mae_affine = float(np.mean(np.abs(errors_affine)))

    # Identity baseline: z_hat = z_src
    errors_id = source_coords - target_coords
    rmse_identity = float(np.sqrt(np.mean(errors_id ** 2)))

    # Mean-Shift baseline: z_hat = z_src + (mean_tgt - mean_src)
    mean_shift = target_coords.mean(axis=0) - source_coords.mean(axis=0)
    errors_ms = (source_coords + mean_shift) - target_coords
    rmse_mean_shift = float(np.sqrt(np.mean(errors_ms ** 2)))

    # R2 coefficient of determination
    ss_tot = np.sum((target_coords - target_coords.mean(axis=0)) ** 2)
    ss_res = np.sum(errors_affine ** 2)
    r2 = float(1.0 - (ss_res / np.maximum(ss_tot, 1e-12)))

    # Relative skill vs Identity
    relative_skill = float(1.0 - (rmse_affine / np.maximum(rmse_identity, 1e-12)))

    return {
        "rmse_affine": rmse_affine,
        "mae_affine": mae_affine,
        "r2_affine": r2,
        "rmse_identity": rmse_identity,
        "rmse_mean_shift": rmse_mean_shift,
        "relative_skill_vs_identity": relative_skill,
    }


def fit_constrained_commuting_transports(
    a_src: NDArray[np.float64],
    a_tgt: NDArray[np.float64],
    b_src: NDArray[np.float64],
    b_tgt: NDArray[np.float64],
) -> Tuple[ParallelTransportMap, ParallelTransportMap]:
    """Fits affine maps T_a^c, T_b^c subject to exact commutation constraints:

    1. Linear commutation: A_a A_b - A_b A_a = 0 (4 constraints)
    2. Translation commutation: (A_a - I) b_b - (A_b - I) b_a = 0 (2 constraints)

    Uses scipy.optimize.minimize with SLSQP.
    """
    import scipy.optimize

    # Initial guess from unrestricted OLS
    estimator = ConnectionEstimator()
    t_a_raw = estimator.estimate_linear_transport("rename_a_init", "src", "tgt", a_src, a_tgt)
    t_b_raw = estimator.estimate_linear_transport("rename_b_init", "src", "tgt", b_src, b_tgt)

    x0 = np.concatenate([
        t_a_raw.matrix_2d.ravel(),
        t_a_raw.bias_2d,
        t_b_raw.matrix_2d.ravel(),
        t_b_raw.bias_2d,
    ])  # 12 parameters

    def objective(x: np.ndarray) -> float:
        A_a = x[:4].reshape(2, 2)
        b_a = x[4:6]
        A_b = x[6:10].reshape(2, 2)
        b_b = x[10:12]

        pred_a = np.dot(a_src, A_a.T) + b_a
        pred_b = np.dot(b_src, A_b.T) + b_b

        sse_a = np.sum((pred_a - a_tgt) ** 2)
        sse_b = np.sum((pred_b - b_tgt) ** 2)
        return float(sse_a + sse_b)

    def constraint_eq(x: np.ndarray) -> np.ndarray:
        A_a = x[:4].reshape(2, 2)
        b_a = x[4:6]
        A_b = x[6:10].reshape(2, 2)
        b_b = x[10:12]

        # 1. Linear commutation: A_a A_b - A_b A_a = 0 (4 elements)
        lin_comm = np.dot(A_a, A_b) - np.dot(A_b, A_a)

        # 2. Translation commutation: (A_a - I) b_b - (A_b - I) b_a = 0 (2 elements)
        trans_comm = np.dot(A_a - np.eye(2), b_b) - np.dot(A_b - np.eye(2), b_a)

        return np.concatenate([lin_comm.ravel(), trans_comm])

    constraints = {"type": "eq", "fun": constraint_eq}
    res = scipy.optimize.minimize(objective, x0, method="SLSQP", constraints=constraints, tol=1e-12, options={"maxiter": 2000, "ftol": 1e-15})

    x_opt = res.x
    A_a_opt = x_opt[:4].reshape(2, 2)
    b_a_opt = x_opt[4:6]
    A_b_opt = x_opt[6:10].reshape(2, 2)
    b_b_opt = x_opt[10:12]


    # Verify constraint satisfaction
    c_err = constraint_eq(x_opt)
    max_c_err = float(np.max(np.abs(c_err)))

    t_a_c = ParallelTransportMap(
        "rename_a_c", "src", "tgt", A_a_opt, b_a_opt,
        metadata={"is_constrained_null": True, "optimizer_success": bool(res.success), "max_constraint_error": max_c_err}
    )
    t_b_c = ParallelTransportMap(
        "rename_b_c", "src", "tgt", A_b_opt, b_b_opt,
        metadata={"is_constrained_null": True, "optimizer_success": bool(res.success), "max_constraint_error": max_c_err}
    )
    return t_a_c, t_b_c



def compute_rename_context_interaction_test(
    orbit_coords_dict: Dict[str, Dict[str, NDArray[np.float64]]],
    n_permutations: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Tests rename-context interaction vector d_i = (z2 - z3) - (z1 - z0).

    Uses orbit-clustered sign-flip permutation to test H0: mean(d) = 0.
    """
    interaction_vectors = []
    for o_dict in orbit_coords_dict.values():
        z0 = o_dict["x0"]
        z1 = o_dict["x1"]
        z2 = o_dict["x2"]
        z3 = o_dict["x3"]
        d_i = (z2 - z3) - (z1 - z0)
        interaction_vectors.append(d_i)

    D = np.array(interaction_vectors)  # (N, 2)
    obs_mean_norm = float(np.linalg.norm(D.mean(axis=0)))

    rng = np.random.default_rng(seed)
    perm_norms = []
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=(len(D), 1))
        perm_mean = (D * signs).mean(axis=0)
        perm_norms.append(float(np.linalg.norm(perm_mean)))

    p_val = float((1.0 + np.sum(np.array(perm_norms) >= obs_mean_norm)) / (n_permutations + 1.0))

    return {
        "interaction_mean_norm": obs_mean_norm,
        "interaction_std_norm": float(np.std([np.linalg.norm(v) for v in D])),
        "interaction_p_value": p_val,
        "num_orbits_evaluated": len(D),
    }





class ConnectionEstimator:

    """Estimates affine parallel transport maps T_{g,x} via OLS or Total Least Squares (TLS)."""

    def __init__(
        self,
        ridge_alpha: float = 1e-6,
        max_condition_number: float = 1e6,
        max_v22_condition_number: float = 1e6,
        min_abs_v22_determinant: float = 1e-8,
        max_transform_norm: float = 1e3,
        relative_rank_tolerance: float = 1e-7,
        absolute_singular_value_floor: float = 1e-15,
    ) -> None:
        self.ridge_alpha = ridge_alpha
        self.max_condition_number = max_condition_number
        self.max_v22_condition_number = max_v22_condition_number
        self.min_abs_v22_determinant = min_abs_v22_determinant
        self.max_transform_norm = max_transform_norm
        self.relative_rank_tolerance = relative_rank_tolerance
        self.absolute_singular_value_floor = absolute_singular_value_floor

    def _validate_raw_inputs(
        self, generator_name: str, source_coords: NDArray[np.float64], target_coords: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Validates input coordinates for shape, finiteness, observation count, and variance."""
        Z_src = np.atleast_2d(source_coords)
        Z_tgt = np.atleast_2d(target_coords)

        if Z_src.ndim != 2 or Z_tgt.ndim != 2:
            raise EstimatorIdentifiabilityError(
                f"Edge '{generator_name}' inputs must be 2D arrays (got src={Z_src.shape}, tgt={Z_tgt.shape})",
                generator_name=generator_name,
                reason="invalid_shape",
            )

        if Z_src.shape != Z_tgt.shape:
            raise EstimatorIdentifiabilityError(
                f"Edge '{generator_name}' observation shape mismatch: src={Z_src.shape} vs tgt={Z_tgt.shape}",
                generator_name=generator_name,
                reason="shape_mismatch",
            )

        if not (np.all(np.isfinite(Z_src)) and np.all(np.isfinite(Z_tgt))):
            raise EstimatorIdentifiabilityError(
                f"Edge '{generator_name}' input contains non-finite values (NaN or Inf)",
                generator_name=generator_name,
                reason="non_finite_values",
            )

        N, d = Z_src.shape
        if N < d + 1:
            raise EstimatorIdentifiabilityError(
                f"Edge '{generator_name}' insufficient observations: N={N} < required={d+1}",
                generator_name=generator_name,
                reason="insufficient_observations",
                required_observations=d + 1,
            )

        return Z_src, Z_tgt

    def validate_design_matrix(self, generator_name: str, Z_src_c: NDArray[np.float64]) -> Tuple[int, float, List[float]]:
        """Validates that centered source observations span the full fiber dimension."""
        d_x = Z_src_c.shape[1]
        _, s, _ = np.linalg.svd(Z_src_c, full_matrices=False)
        s_list = [float(v) for v in s]
        s_max = float(s[0]) if len(s) > 0 else 0.0

        rank = int(np.sum(s > self.relative_rank_tolerance * s_max)) if s_max > 0 else 0
        cond = float(s_max / s[-1]) if (len(s) > 0 and s[-1] > self.absolute_singular_value_floor) else float("inf")

        if rank < d_x:
            raise EstimatorIdentifiabilityError(
                f"Edge '{generator_name}' source design matrix is rank-deficient (rank={rank} < {d_x}).",
                generator_name=generator_name,
                reason="rank_deficient",
                design_rank=rank,
                required_rank=d_x,
                condition_number=cond,
                condition_threshold=self.max_condition_number,
                singular_values=s_list,
            )
        if cond > self.max_condition_number:
            raise EstimatorIdentifiabilityError(
                f"Edge '{generator_name}' design matrix is ill-conditioned (cond={cond:.2e} > {self.max_condition_number:.2e}).",
                generator_name=generator_name,
                reason="ill_conditioned_design",
                design_rank=rank,
                required_rank=d_x,
                condition_number=cond,
                condition_threshold=self.max_condition_number,
                singular_values=s_list,
            )
        return rank, cond, s_list

    def diagnose_transport_design(
        self, generator_name: str, source_coords: NDArray[np.float64], target_coords: NDArray[np.float64]
    ) -> Dict[str, Any]:
        """Runs pre-fit diagnostic checks on an edge without altering estimator state."""
        try:
            Z_src, Z_tgt = self._validate_raw_inputs(generator_name, source_coords, target_coords)
            mean_src = Z_src.mean(axis=0)
            mean_tgt = Z_tgt.mean(axis=0)
            Z_src_c = Z_src - mean_src
            Z_tgt_c = Z_tgt - mean_tgt
            d_x = Z_src_c.shape[1]
            rank, cond, s_list = self.validate_design_matrix(generator_name, Z_src_c)

            Aug = np.column_stack([Z_src_c, Z_tgt_c])
            _, _, Vt = np.linalg.svd(Aug, full_matrices=True)
            V = Vt.T
            V22 = V[d_x:, d_x:]
            cond_v22 = float(np.linalg.cond(V22))
            det_v22 = float(np.linalg.det(V22))

            if cond_v22 > self.max_v22_condition_number or abs(det_v22) < self.min_abs_v22_determinant:
                return {
                    "generator_name": generator_name,
                    "status": "not_estimable",
                    "reason": "ill_conditioned_v22",
                    "design_rank": rank,
                    "required_rank": d_x,
                    "condition_number": cond,
                    "v22_condition_number": cond_v22,
                    "v22_determinant": det_v22,
                    "v22_determinant_threshold": self.min_abs_v22_determinant,
                    "condition_threshold": self.max_v22_condition_number,
                    "singular_values": s_list,
                    "relative_rank_tolerance": self.relative_rank_tolerance,
                    "absolute_singular_value_floor": self.absolute_singular_value_floor,
                }

            return {
                "generator_name": generator_name,
                "status": "estimable",
                "design_rank": rank,
                "required_rank": d_x,
                "condition_number": cond,
                "v22_condition_number": cond_v22,
                "v22_determinant": det_v22,
                "v22_determinant_threshold": self.min_abs_v22_determinant,
                "singular_values": s_list,
                "relative_rank_tolerance": self.relative_rank_tolerance,
                "absolute_singular_value_floor": self.absolute_singular_value_floor,
            }
        except EstimatorIdentifiabilityError as err:
            res_dict = err.to_dict()
            res_dict["status"] = "not_estimable"
            res_dict["relative_rank_tolerance"] = self.relative_rank_tolerance
            res_dict["absolute_singular_value_floor"] = self.absolute_singular_value_floor
            return res_dict

    def estimate_linear_transport(

        self,
        generator_name: str,
        source_id: str,
        target_id: str,
        source_coords: NDArray[np.float64],
        target_coords: NDArray[np.float64],
        strict_identifiability: bool = True,
    ) -> ParallelTransportMap:
        """Estimates affine map T_g via OLS: z_target approx A z_source + b."""
        Z_src, Z_tgt = self._validate_raw_inputs(generator_name, source_coords, target_coords)

        mean_src = Z_src.mean(axis=0)
        mean_tgt = Z_tgt.mean(axis=0)

        Z_src_c = Z_src - mean_src
        Z_tgt_c = Z_tgt - mean_tgt

        d = Z_src_c.shape[1]
        rank, cond, s_list = d, 1.0, []
        if strict_identifiability:
            rank, cond, s_list = self.validate_design_matrix(generator_name, Z_src_c)

        cov = np.dot(Z_src_c.T, Z_src_c) + self.ridge_alpha * np.eye(d)
        cross = np.dot(Z_src_c.T, Z_tgt_c)

        T_mat_T = np.linalg.solve(cov, cross)
        T_mat = T_mat_T.T

        bias = mean_tgt - np.dot(T_mat, mean_src)

        matrix_norm = float(np.linalg.norm(T_mat, "fro"))
        bias_norm = float(np.linalg.norm(bias))

        if strict_identifiability and (matrix_norm > self.max_transform_norm or bias_norm > self.max_transform_norm):
            raise EstimatorIdentifiabilityError(
                f"Edge '{generator_name}' transport estimate norm exploded (||A||_F={matrix_norm:.2e}, ||b||={bias_norm:.2e}).",
                generator_name=generator_name,
                reason="exploding_norm",
                design_rank=rank,
                required_rank=d,
                condition_number=cond,
                matrix_norm=matrix_norm,
                bias_norm=bias_norm,
                singular_values=s_list,
            )

        return ParallelTransportMap(
            generator_name=generator_name,
            source_id=source_id,
            target_id=target_id,
            matrix_2d=T_mat,
            bias_2d=bias,
            metadata={
                "design_rank": rank,
                "condition_number": cond,
                "matrix_norm": matrix_norm,
                "bias_norm": bias_norm,
                "singular_values": s_list,
            },
        )

    def estimate_total_least_squares_transport(
        self,
        generator_name: str,
        source_id: str,
        target_id: str,
        source_coords: NDArray[np.float64],
        target_coords: NDArray[np.float64],
        strict_identifiability: bool = True,
    ) -> ParallelTransportMap:
        """Estimates affine map T_g via Total Least Squares (TLS) to correct errors-in-variables attenuation bias."""
        Z_src, Z_tgt = self._validate_raw_inputs(generator_name, source_coords, target_coords)

        mean_src = Z_src.mean(axis=0)
        mean_tgt = Z_tgt.mean(axis=0)

        Z_src_c = Z_src - mean_src
        Z_tgt_c = Z_tgt - mean_tgt

        d_x = Z_src_c.shape[1]
        rank, cond, s_list = d_x, 1.0, []
        if strict_identifiability:
            rank, cond, s_list = self.validate_design_matrix(generator_name, Z_src_c)

        # Stack augmented matrix [X_c | Y_c]
        Aug = np.column_stack([Z_src_c, Z_tgt_c])

        # Perform SVD of augmented matrix
        _, _, Vt = np.linalg.svd(Aug, full_matrices=True)
        V = Vt.T

        # Partition V into [[V11, V12], [V21, V22]]
        V12 = V[:d_x, d_x:]
        V22 = V[d_x:, d_x:]

        cond_v22 = float(np.linalg.cond(V22))
        det_v22 = float(np.linalg.det(V22))

        if strict_identifiability and (cond_v22 > self.max_v22_condition_number or abs(det_v22) < self.min_abs_v22_determinant):
            raise EstimatorIdentifiabilityError(
                f"Edge '{generator_name}' TLS V22 submatrix is ill-conditioned (cond(V22)={cond_v22:.2e}, det={det_v22:.2e}).",
                generator_name=generator_name,
                reason="ill_conditioned_v22",
                design_rank=rank,
                required_rank=d_x,
                condition_number=cond,
                v22_condition_number=cond_v22,
                v22_determinant=det_v22,
                v22_determinant_threshold=self.min_abs_v22_determinant,
                condition_threshold=self.max_v22_condition_number,
                singular_values=s_list,
            )

        if cond_v22 > 1e10 or abs(det_v22) < 1e-10:
            T_mat_T = -np.dot(V12, np.linalg.pinv(V22))
        else:
            T_mat_T = -np.linalg.solve(V22.T, V12.T).T

        T_mat = T_mat_T.T
        bias = mean_tgt - np.dot(T_mat, mean_src)

        matrix_norm = float(np.linalg.norm(T_mat, "fro"))
        bias_norm = float(np.linalg.norm(bias))

        if strict_identifiability and (matrix_norm > self.max_transform_norm or bias_norm > self.max_transform_norm):
            raise EstimatorIdentifiabilityError(
                f"Edge '{generator_name}' TLS transport estimate norm exploded (||A||_F={matrix_norm:.2e}, ||b||={bias_norm:.2e}).",
                generator_name=generator_name,
                reason="exploding_norm",
                design_rank=rank,
                required_rank=d_x,
                condition_number=cond,
                v22_condition_number=cond_v22,
                v22_determinant=det_v22,
                matrix_norm=matrix_norm,
                bias_norm=bias_norm,
                singular_values=s_list,
            )

        return ParallelTransportMap(
            generator_name=generator_name,
            source_id=source_id,
            target_id=target_id,
            matrix_2d=T_mat,
            bias_2d=bias,
            metadata={
                "design_rank": rank,
                "condition_number": cond,
                "v22_condition_number": cond_v22,
                "v22_determinant": det_v22,
                "matrix_norm": matrix_norm,
                "bias_norm": bias_norm,
                "singular_values": s_list,
            },
        )
