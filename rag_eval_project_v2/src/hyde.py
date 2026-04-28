from __future__ import annotations

from typing import Any

import numpy as np

from src.generator import OllamaGenerator, load_generation_config
from src.utils.embeddings import encode_for_task, load_embedder_from_config


class HyDEExpander:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.strict_mode = bool(config.get("runtime", {}).get("strict_mode", False))
        self.enabled = bool(config.get("hyde", {}).get("enabled", True))
        self.apply_to_intents = set(config.get("hyde", {}).get("apply_to_intents", []))
        self.model_name = config.get("hyde", {}).get("model", "qwen2.5:14b")
        self._local_qwen: OllamaGenerator | None = None
        self.embedding_model: Any = None

    def _ensure_embedder(self) -> Any:
        # Defer construction until used so disabled HyDE doesn't load an embedder (A10).
        if self.embedding_model is None:
            self.embedding_model = load_embedder_from_config(self.config)
        return self.embedding_model

    def _ensure_generator(self) -> OllamaGenerator:
        if self._local_qwen is None:
            ollama_options = dict(self.config.get("runtime", {}).get("ollama_options", {}))
            gen_cfg = load_generation_config(self.config)
            self._local_qwen = OllamaGenerator(
                self.model_name,
                strict=self.strict_mode,
                timeout=gen_cfg["timeout"],
                max_retries=gen_cfg["max_retries"],
                retry_backoff_sec=gen_cfg["retry_backoff_sec"],
                keep_alive=str(self.config.get("runtime", {}).get("ollama_keep_alive", "0s")),
                options=ollama_options,
            )
        return self._local_qwen

    @property
    def local_qwen(self) -> OllamaGenerator:
        # Backwards-compatible accessor for tests that monkeypatch this attribute.
        return self._ensure_generator()

    @local_qwen.setter
    def local_qwen(self, value: OllamaGenerator) -> None:
        self._local_qwen = value

    async def expand(self, question: str, intent: dict[str, Any]) -> tuple[str, bool]:
        """Return (search_text, used_hyde). used_hyde=True means search_text is a hypothetical answer."""
        if not self.enabled:
            return question, False
        intent_type = intent.get("type")
        if intent_type not in self.apply_to_intents:
            return question, False
        prompt = (
            "You are a knowledgeable assistant about Georgia's EV automotive supply chain.\n"
            "Write a 3-5 sentence factual answer that WOULD correctly answer this question.\n"
            "Use accurate domain terms, tier labels, role names, and company types.\n"
            f"Question: {question}\n"
            "Hypothetical Answer:"
        )
        try:
            qwen = self._ensure_generator()
            out = await qwen.generate(prompt=prompt, system="Be factual, concise, and domain-specific.")
            text = out.strip()
            if not text:
                return question, False
            return text, True
        except Exception:
            if self.strict_mode:
                raise
            return question, False

    def get_search_vector(self, text: str, is_expansion: bool = False) -> list[float]:
        # If text is a HyDE-generated hypothetical answer, embed it as a document so it
        # lives in document-space (matters for nomic instruction prefixes — A1).
        task = "document" if is_expansion else "query"
        embedder = self._ensure_embedder()
        encoded = encode_for_task(
            embedder,
            [text],
            task=task,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(encoded[0], dtype=float).tolist()
