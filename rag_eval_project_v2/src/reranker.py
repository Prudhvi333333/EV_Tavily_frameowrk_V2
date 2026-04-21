from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np


class CrossEncoderReranker:
    _MODEL_CACHE: dict[tuple[str, bool], Any] = {}

    def __init__(self, config: dict[str, Any]) -> None:
        reranker_cfg = config.get("reranker", {})
        self.model_name = str(reranker_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L6-v2"))
        self.max_candidates = max(int(reranker_cfg.get("max_candidates", 18)), 1)
        self.blend_weight = float(reranker_cfg.get("blend_weight", 0.65))
        self.batch_size = max(int(reranker_cfg.get("batch_size", 16)), 1)
        self.local_files_only = bool(
            reranker_cfg.get(
                "local_files_only",
                config.get("embeddings", {}).get("local_files_only", False),
            )
        )
        self._model = None

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        cache_key = (self.model_name, self.local_files_only)
        cached = self._MODEL_CACHE.get(cache_key)
        if cached is not None:
            self._model = cached
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError(
                "Cross-encoder reranker is enabled but dependency is missing. Install: sentence-transformers"
            ) from exc

        try:
            self._model = CrossEncoder(
                self.model_name,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load cross-encoder model '{self.model_name}'. "
                "If this model is not cached, set reranker.local_files_only=false."
            ) from exc
        self._MODEL_CACHE[cache_key] = self._model
        return self._model

    def rerank(self, question_text: str, docs: list[Any], top_k: int) -> list[Any]:
        if not docs:
            return []
        if not question_text.strip():
            raise RuntimeError("Cross-encoder reranker requires a non-empty query text.")

        model = self._ensure_model()
        pool = docs[: min(len(docs), self.max_candidates)]
        pairs = [[question_text, str(d.text)] for d in pool]
        request_k = min(max(int(top_k), 1), len(pool))

        try:
            scores = np.asarray(
                model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                ),
                dtype=float,
            )
        except TypeError:
            # Compatibility with older sentence-transformers versions.
            scores = np.asarray(model.predict(pairs), dtype=float)
        except Exception as exc:
            raise RuntimeError("Cross-encoder rerank call failed.") from exc

        if scores.size != len(pool):
            raise RuntimeError(
                "Cross-encoder output length does not match candidate document pool."
            )

        order = np.argsort(-scores)
        ranked_pairs = [(pool[int(idx)], float(scores[int(idx)])) for idx in order[:request_k]]
        raw_scores = np.array([score for _, score in ranked_pairs], dtype=float)
        min_s = float(raw_scores.min()) if raw_scores.size else 0.0
        max_s = float(raw_scores.max()) if raw_scores.size else 0.0
        denom = (max_s - min_s) if max_s > min_s else 1.0

        out: list[Any] = []
        blend = max(0.0, min(1.0, self.blend_weight))
        for doc, raw_score in ranked_pairs[:request_k]:
            normalized_rerank = (float(raw_score) - min_s) / denom
            blended_score = (1.0 - blend) * float(doc.score) + blend * normalized_rerank
            out.append(
                replace(
                    doc,
                    score=round(blended_score, 4),
                    rerank_score=round(float(raw_score), 4),
                )
            )
        return out
