"""Model Adapter & Label Alignment for Phase E2 Classifier Audits.

Maps raw HuggingFace NLI logits/softmax predictions to standard [Entailment, Neutral, Contradiction] order
by inspecting model.config.id2label string mappings. Supports multidimensional arrays (3,), (N, 3), (B, N, 3).
Includes HuggingFaceNLIAdapter for live pre-trained model inference (e.g. roberta-large-mnli, deberta-large-mnli).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Tuple
import numpy as np
from numpy.typing import NDArray



def get_helmert_basis() -> NDArray[np.float64]:
    """Returns (3, 2) Helmert basis matrix V for 3-simplex ILR coordinates."""
    v1 = np.array([1.0 / np.sqrt(2), -1.0 / np.sqrt(2), 0.0], dtype=np.float64)
    v2 = np.array([1.0 / np.sqrt(6), 1.0 / np.sqrt(6), -2.0 / np.sqrt(6)], dtype=np.float64)
    return np.column_stack([v1, v2])  # (3, 2)


@dataclass(frozen=True)
class LiveNLIConfig:
    """Configuration for pinned live HuggingFace NLI model inference."""

    model_id: str = "FacebookAI/roberta-large-mnli"
    revision: str | None = "2a8f12d27941090092df78e4ba6f0928eb5eac98"
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = 16
    max_length: int = 128
    truncation: bool = True
    local_files_only: bool = False
    use_mock_fallback: bool = False



@dataclass(frozen=True)
class NLIInferenceBatch:
    """Structured result of batched live NLI inference."""

    raw_logits: NDArray[np.float64]        # (N, 3) raw model logits in model native order
    aligned_logits: NDArray[np.float64]    # (N, 3) aligned logits in [E, N, C] order
    probabilities: NDArray[np.float64]     # (N, 3) probabilities in [E, N, C] order
    ilr_coordinates: NDArray[np.float64]   # (N, 2) ILR coordinates calculated directly from aligned logits
    token_counts: NDArray[np.int64]        # (N,) token counts per example
    truncated: NDArray[np.bool_]           # (N,) boolean truncation flags


class NLIModelAdapter:
    """Safely aligns NLI model outputs to standard [E, N, C] probability and logit order."""

    def __init__(self, id2label: Dict[int, str] | None = None) -> None:
        # Default RoBERTa / DeBERTa MNLI mapping: 0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT
        if id2label is None:
            self.id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        else:
            self.id2label = id2label

        self._build_permutation()

    def _build_permutation(self) -> None:
        """Builds index permutation array to reorder probabilities into [E, N, C]."""
        lbl_values = [str(v).upper() for v in self.id2label.values()]
        if any("NON_ENTAIL" in v for v in lbl_values):
            raise ValueError(f"Ambiguous label 'NON_ENTAILMENT' detected in id2label: {self.id2label}")
        if len(set(lbl_values)) != len(lbl_values):
            raise ValueError(f"Duplicate label mappings detected in id2label: {self.id2label}")

        e_idx, n_idx, c_idx = -1, -1, -1
        for idx, lbl in self.id2label.items():
            l_str = str(lbl).upper()
            if "ENTAIL" in l_str or l_str == "E":
                e_idx = int(idx)
            elif "NEUTRAL" in l_str or l_str == "N":
                n_idx = int(idx)
            elif "CONTRADICT" in l_str or l_str == "C":
                c_idx = int(idx)

        if e_idx == -1 or n_idx == -1 or c_idx == -1:
            raise ValueError(f"Could not map all 3 NLI labels from id2label: {self.id2label}")

        self.perm_indices = np.array([e_idx, n_idx, c_idx], dtype=np.int64)

    def align_probabilities(self, raw_probs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reorders raw model probability array to standard [Entailment, Neutral, Contradiction].

        Supports 1D (3,), 2D (N, 3), and 3D (B, N, 3) arrays.
        """
        raw_arr = np.array(raw_probs, dtype=np.float64)
        if raw_arr.shape[-1] != 3:
            raise ValueError(f"Last dimension must be 3 classes, got shape {raw_arr.shape}")

        aligned = raw_arr[..., self.perm_indices]
        sums = aligned.sum(axis=-1, keepdims=True)
        return aligned / np.maximum(sums, 1e-12)

    def align_logits(self, raw_logits: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reorders raw model logits array to standard [Entailment, Neutral, Contradiction]."""
        raw_arr = np.array(raw_logits, dtype=np.float64)
        if raw_arr.shape[-1] != 3:
            raise ValueError(f"Last dimension must be 3 classes, got shape {raw_arr.shape}")
        return raw_arr[..., self.perm_indices]

    def compute_direct_ilr_coordinates(self, aligned_logits: NDArray[np.float64]) -> NDArray[np.float64]:
        """Computes ILR coordinates directly from aligned logits z = ell(x) @ V.

        Bypasses softmax underflow and machine precision loss.
        """
        V = get_helmert_basis()  # (3, 2)
        return np.dot(aligned_logits, V)

    def predict_mock_orbit_vertices(
        self, premise: str, hypothesis: str
    ) -> NDArray[np.float64]:
        """Deterministic SHA-256 process-invariant mock prediction for testing pipeline."""
        content = (premise + "||" + hypothesis).encode("utf-8")
        h_bytes = hashlib.sha256(content).digest()
        val = int.from_bytes(h_bytes[:4], byteorder="big") % 100

        p_raw = np.array([0.5 + 0.001 * (val % 10), 0.3, 0.2], dtype=np.float64)
        return p_raw / p_raw.sum()


class HuggingFaceNLIAdapter:
    """Pre-trained HuggingFace NLI model adapter with pinned revision and batched logit-direct ILR coordinates."""

    def __init__(self, model_name: str = "FacebookAI/roberta-large-mnli", use_mock_fallback: bool = True, config: LiveNLIConfig | None = None) -> None:
        self.model_name = model_name
        self.config = config or LiveNLIConfig(model_id=model_name, use_mock_fallback=use_mock_fallback)
        self.use_mock_fallback = self.config.use_mock_fallback
        self.tokenizer = None
        self.model = None
        self.adapter = None
        self.is_loaded = False
        self.load_error: str | None = None
        self.resolved_model_revision: str | None = None
        self.resolved_tokenizer_revision: str | None = None

    def load(self) -> bool:
        """Attempts to load tokenizer and sequence classification model from HuggingFace with pinned revision."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            kwargs = {}
            if self.config.revision:
                kwargs["revision"] = self.config.revision
            if self.config.local_files_only:
                kwargs["local_files_only"] = True

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, **kwargs)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_id, **kwargs)

            device = torch.device(self.config.device if torch.cuda.is_available() or self.config.device == "cpu" else "cpu")
            self.model.to(device)
            self.model.eval()

            self.resolved_model_revision = getattr(self.model.config, "_commit_hash", self.config.revision)
            self.resolved_tokenizer_revision = getattr(self.tokenizer, "_commit_hash", self.config.revision)

            id2label = getattr(self.model.config, "id2label", None)
            if id2label:
                id2label = {int(k): str(v) for k, v in id2label.items()}
            self.adapter = NLIModelAdapter(id2label)
            self.is_loaded = True
            self.load_error = None
            return True
        except Exception as err:
            self.load_error = str(err)
            self.adapter = NLIModelAdapter()
            self.is_loaded = False
            if not self.use_mock_fallback:
                raise RuntimeError(
                    f"Failed to load requested live model '{self.config.model_id}' and use_mock_fallback=False: {err}"
                ) from err
            return False

    def get_provenance_metadata(self) -> Dict[str, Any]:
        """Returns runtime model environment and loading provenance."""
        transformers_ver = None
        torch_ver = None
        try:
            import transformers
            transformers_ver = getattr(transformers, "__version__", None)
        except ImportError:
            pass
        try:
            import torch
            torch_ver = getattr(torch, "__version__", None)
        except ImportError:
            pass

        device_str = None
        dtype_str = None
        if self.is_loaded and self.model:
            try:
                param = next(self.model.parameters())
                device_str = str(param.device)
                dtype_str = str(param.dtype)
            except Exception:
                pass

        return {
            "model_requested": self.config.model_id,
            "model_resolved": self.config.model_id if self.is_loaded else None,
            "resolved_model_revision": self.resolved_model_revision,
            "resolved_tokenizer_revision": self.resolved_tokenizer_revision,
            "adapter_mode": "huggingface_live" if self.is_loaded else ("mock_fallback" if self.use_mock_fallback else "failed"),
            "is_loaded": self.is_loaded,
            "use_mock_fallback": self.use_mock_fallback,
            "transformers_version": transformers_ver,
            "torch_version": torch_ver,
            "device": device_str,
            "dtype": dtype_str,
            "load_error": self.load_error,
        }

    def predict(self, premise: str, hypothesis: str) -> NDArray[np.float64]:
        """Runs single-example inference and returns aligned [E, N, C] probability vector."""
        if not self.is_loaded:
            if not self.load() and self.use_mock_fallback:
                return self.adapter.predict_mock_orbit_vertices(premise, hypothesis)

        import torch

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=self.config.max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy()[0]
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        return self.adapter.align_probabilities(probs)

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> NLIInferenceBatch:
        """Runs batched inference and returns NLIInferenceBatch with direct logit ILR coordinates."""
        if not self.is_loaded:
            if not self.load() and self.use_mock_fallback:
                # Return mock batch
                raw_probs = np.array([self.adapter.predict_mock_orbit_vertices(p, h) for p, h in pairs])
                raw_logits = np.log(raw_probs + 1e-12)
                aligned_logits = self.adapter.align_logits(raw_logits)
                ilr_coords = self.adapter.compute_direct_ilr_coordinates(aligned_logits)
                token_counts = np.array([len(p.split()) + len(h.split()) for p, h in pairs], dtype=np.int64)
                truncated = np.zeros(len(pairs), dtype=bool)
                return NLIInferenceBatch(
                    raw_logits=raw_logits,
                    aligned_logits=aligned_logits,
                    probabilities=raw_probs,
                    ilr_coordinates=ilr_coords,
                    token_counts=token_counts,
                    truncated=truncated,
                )

        import torch

        device = next(self.model.parameters()).device
        bs = self.config.batch_size

        all_raw_logits = []
        all_probs = []
        all_token_counts = []
        all_truncated = []

        for i in range(0, len(pairs), bs):
            chunk = pairs[i : i + bs]
            premises = [p for p, h in chunk]
            hypotheses = [h for p, h in chunk]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=self.config.truncation,
                max_length=self.config.max_length,
            )

            input_ids = inputs["input_ids"]
            counts = (input_ids != self.tokenizer.pad_token_id).sum(dim=1).cpu().numpy()
            trunc = (counts >= self.config.max_length)

            inputs_dev = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs_dev)
                logits = outputs.logits.cpu().numpy()
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            all_raw_logits.append(logits)
            all_probs.append(probs)
            all_token_counts.append(counts)
            all_truncated.append(trunc)

        raw_logits = np.vstack(all_raw_logits)
        probs = np.vstack(all_probs)
        token_counts = np.concatenate(all_token_counts)
        truncated = np.concatenate(all_truncated)

        aligned_logits = self.adapter.align_logits(raw_logits)
        aligned_probs = self.adapter.align_probabilities(probs)
        ilr_coords = self.adapter.compute_direct_ilr_coordinates(aligned_logits)

        return NLIInferenceBatch(
            raw_logits=raw_logits,
            aligned_logits=aligned_logits,
            probabilities=aligned_probs,
            ilr_coordinates=ilr_coords,
            token_counts=token_counts,
            truncated=truncated,
        )



