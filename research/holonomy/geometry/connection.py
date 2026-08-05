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


class ConnectionEstimator:
    """Estimates affine parallel transport maps T_{g,x} via OLS or Total Least Squares (TLS)."""

    def __init__(
        self,
        ridge_alpha: float = 1e-6,
        max_condition_number: float = 1e6,
        max_transform_norm: float = 1e3,
        relative_rank_tolerance: float = 1e-7,
        absolute_singular_value_floor: float = 1e-15,
    ) -> None:
        self.ridge_alpha = ridge_alpha
        self.max_condition_number = max_condition_number
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

            return {
                "generator_name": generator_name,
                "status": "estimable",
                "design_rank": rank,
                "required_rank": d_x,
                "condition_number": cond,
                "v22_condition_number": cond_v22,
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
        if strict_identifiability and (cond_v22 > self.max_condition_number or abs(float(np.linalg.det(V22))) < 1e-8):
            raise EstimatorIdentifiabilityError(
                f"Edge '{generator_name}' TLS V22 submatrix is ill-conditioned (cond(V22)={cond_v22:.2e}).",
                generator_name=generator_name,
                reason="ill_conditioned_v22",
                design_rank=rank,
                required_rank=d_x,
                condition_number=cond,
                v22_condition_number=cond_v22,
                condition_threshold=self.max_condition_number,
                singular_values=s_list,
            )

        if cond_v22 > 1e10 or abs(float(np.linalg.det(V22))) < 1e-10:
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
                "matrix_norm": matrix_norm,
                "bias_norm": bias_norm,
                "singular_values": s_list,
            },
        )


