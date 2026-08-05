"""Ollama Live Model Adapter for Educational Counterfactual Audit (Phase EDU-2a).

Communicates with local Ollama service to execute pinned live generative LLM inference.
Captures model tag, immutable digest, quantization, hardware parameters, and latency.
Refuses execution if mock fallback is requested in live canary mode.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from research.education_audit.schemas import AuditCase, CounterfactualVariant, GenerationRecord


class OllamaEducationAdapter:
    """Live Ollama Model Adapter for Phase EDU-2a Canary."""

    def __init__(
        self,
        model_name: str = "gemma:12b",
        host: str = "http://localhost:11434",
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_predict: int = 400,
        use_mock_fallback: bool = False,

    ) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.top_p = top_p
        self.num_predict = num_predict
        self.use_mock_fallback = use_mock_fallback

        self.model_digest: Optional[str] = None
        self.ollama_version: Optional[str] = None
        self.quantization: Optional[str] = None
        self.is_loaded: bool = False
        self.load_error: Optional[str] = None

    def ping_and_inspect(self) -> bool:
        """Pings Ollama service and inspects model metadata."""
        try:
            # Check version
            ver_req = urllib.request.Request(f"{self.host}/api/version")
            with urllib.request.urlopen(ver_req, timeout=5) as resp:
                ver_data = json.loads(resp.read().decode("utf-8"))
                self.ollama_version = ver_data.get("version", "unknown")

            # Check model show metadata
            show_req = urllib.request.Request(
                f"{self.host}/api/show",
                data=json.dumps({"name": self.model_name}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(show_req, timeout=5) as resp:
                show_data = json.loads(resp.read().decode("utf-8"))
                details = show_data.get("details", {})
                self.quantization = details.get("quantization_level", "unknown")

            # Check tags to get digest
            tags_req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(tags_req, timeout=5) as resp:
                tags_data = json.loads(resp.read().decode("utf-8"))
                for m in tags_data.get("models", []):
                    m_name = m.get("name", "")
                    if m_name == self.model_name or m_name.startswith(self.model_name) or (self.model_name in ["gemma:12b", "gemma3:12b"] and "gemma3:12b" in m_name):
                        self.model_name = m_name
                        self.model_digest = m.get("digest")
                        break

            self.is_loaded = (self.model_digest is not None)
            return self.is_loaded

        except Exception as err:
            self.load_error = str(err)
            self.is_loaded = False
            if not self.use_mock_fallback:
                print(f"Ollama Live Adapter Inspection Failed: {err}")
            return False

    def generate(
        self,
        case: AuditCase,
        variant: CounterfactualVariant,
        prompt_id: str,
        prompt_template: str,
        repeat_index: int = 0,
        seed: int = 101,
    ) -> GenerationRecord:
        """Executes live Ollama generation or mock fallback if allowed."""
        if not self.is_loaded:
            self.ping_and_inspect()

        if not self.is_loaded:
            if not self.use_mock_fallback:
                raise RuntimeError(
                    f"Ollama Live Execution Refusal: Could not connect to Ollama at {self.host} "
                    f"or inspect model '{self.model_name}'. Load error: {self.load_error}"
                )
            # Fallback mock for testing environment
            from research.education_audit.adapters.mock import SeededStochasticMockAdapter
            mock = SeededStochasticMockAdapter(model_id=f"mock-fallback-{self.model_name}")
            return mock.generate(case, variant, prompt_id, prompt_template, repeat_index=repeat_index)

        prompt_content = prompt_template.format(
            rendered_input=variant.rendered_input,
            target_opportunity=case.target_opportunity,
        )
        p_hash = hashlib.sha256((prompt_id + "||" + prompt_content).encode("utf-8")).hexdigest()

        payload = {
            "model": self.model_name,
            "prompt": prompt_content,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.num_predict,
                "seed": seed,
            },
        }

        start_time = time.perf_counter()
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode("utf-8"))


        elapsed_ms = float((time.perf_counter() - start_time) * 1000.0)
        output_text = data.get("response", "").strip()
        done_reason = data.get("done_reason", "stop")
        out_hash = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        gen_id = f"ollama_gen_{variant.variant_id}_{prompt_id}_s{seed}_r{repeat_index}"

        return GenerationRecord(
            generation_id=gen_id,
            case_id=case.case_id,
            variant_id=variant.variant_id,
            condition=variant.condition,
            prompt_id=prompt_id,
            prompt_hash=p_hash,
            model_id=self.model_name,
            model_revision=self.model_digest or "unknown_digest",
            parameters={
                "model_tag": self.model_name,
                "model_digest": self.model_digest,
                "quantization": self.quantization,
                "ollama_version": self.ollama_version,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.num_predict,
                "requested_seed": seed,
                "latency_ms": elapsed_ms,
                "done_reason": done_reason,
            },
            repeat_index=repeat_index,
            output_text=output_text,
            output_hash=out_hash,
        )
