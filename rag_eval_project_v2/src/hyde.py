from __future__ import annotations

from typing import Any

import numpy as np

from src.generator import OllamaGenerator
from src.utils.embeddings import encode_for_task, load_embedder_from_config


class HyDEExpander:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.strict_mode = bool(config.get("runtime", {}).get("strict_mode", False))
        self.enabled = bool(config.get("hyde", {}).get("enabled", True))
        self.apply_to_intents = set(config.get("hyde", {}).get("apply_to_intents", []))
        self.model_name = config.get("hyde", {}).get("model", "qwen2.5:14b")
        ollama_options = dict(config.get("runtime", {}).get("ollama_options", {}))
        self.local_qwen = OllamaGenerator(
            self.model_name,
            keep_alive=str(config.get("runtime", {}).get("ollama_keep_alive", "0s")),
            options=ollama_options,
        )
        self.embedding_model = load_embedder_from_config(config)

    async def expand(self, question: str, intent: dict[str, Any]) -> str:
        if not self.enabled:
            return question
        intent_type = intent.get("type")
        if intent_type not in self.apply_to_intents:
            return question
        prompt = (
            "You are a knowledgeable assistant about Georgia's EV automotive supply chain.\n"
            "Write a 3-5 sentence factual answer that WOULD correctly answer this question.\n"
            "Use accurate domain terms, tier labels, role names, and company types.\n"
            f"Question: {question}\n"
            "Hypothetical Answer:"
        )
        try:
            out = await self.local_qwen.generate(prompt=prompt, system="Be factual, concise, and domain-specific.")
            return out.strip() or question
        except Exception:
            if self.strict_mode:
                raise
            return question

    def get_search_vector(self, text: str) -> list[float]:
        encoded = encode_for_task(
            self.embedding_model,
            [text],
            task="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(encoded[0], dtype=float).tolist()
