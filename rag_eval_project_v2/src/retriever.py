from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.indexer import HybridIndex
from src.utils.embeddings import encode_for_task


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class RetrievedDoc:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float
    semantic_score: float
    bm25_score: float
    rerank_score: float | None = None


class HybridRetriever:
    def __init__(self, index: HybridIndex, config: dict[str, Any]) -> None:
        self.index = index
        self.config = config
        self.strict_mode = bool(config.get("runtime", {}).get("strict_mode", False))
        self.id_to_doc = {d["id"]: d for d in self.index.documents}
        retrieval_cfg = config.get("retrieval", {})
        self.top_k = int(retrieval_cfg["top_k"])
        self.use_adaptive_top_k = bool(retrieval_cfg.get("adaptive_top_k", False))
        self.semantic_weight = float(retrieval_cfg["semantic_weight"])
        self.bm25_weight = float(retrieval_cfg["bm25_weight"])
        self.max_context_chars = int(retrieval_cfg.get("max_context_chars", 4500))
        self.max_doc_chars = int(retrieval_cfg.get("max_doc_chars", 700))
        metadata_scoring = dict(retrieval_cfg.get("metadata_scoring", {}))
        self.location_match_boost = float(metadata_scoring.get("location_match_boost", 0.08))
        self.location_mismatch_penalty = float(metadata_scoring.get("location_mismatch_penalty", 0.0))
        self.tier_match_boost = float(metadata_scoring.get("tier_match_boost", 0.12))
        self.tier_mismatch_penalty = float(metadata_scoring.get("tier_mismatch_penalty", -0.08))
        self.role_match_boost = float(metadata_scoring.get("role_match_boost", 0.10))
        self.oem_match_boost = float(metadata_scoring.get("oem_match_boost", 0.05))
        self.max_metadata_boost = float(metadata_scoring.get("max_boost", 0.30))
        self.max_metadata_penalty = float(metadata_scoring.get("max_penalty", -0.20))
        self.backend = str(retrieval_cfg.get("backend", "hybrid")).strip().lower()
        if self.backend not in {"hybrid", "llamaindex"}:
            raise ValueError("retrieval.backend must be one of: hybrid, llamaindex")

        self.semantic_backend = None
        if self.backend == "llamaindex":
            from src.llamaindex_backend import LlamaIndexSemanticBackend

            self.semantic_backend = LlamaIndexSemanticBackend(
                collection=self.index.collection,
                strict_mode=self.strict_mode,
            )

        reranker_cfg = config.get("reranker", {})
        self.reranker = None
        if bool(reranker_cfg.get("enabled", False)):
            provider = str(reranker_cfg.get("provider", "cross_encoder")).strip().lower()
            if provider not in {"cross_encoder", "cross-encoder"}:
                raise ValueError("reranker.provider currently supports only 'cross_encoder'.")
            from src.reranker import CrossEncoderReranker

            self.reranker = CrossEncoderReranker(config)

    def detect_query_intent(self, question: str) -> dict[str, Any]:
        q = question.lower()
        intent = "direct"
        if any(x in q for x in ["indirect", "through", "linked", "relationship", "multi-hop"]):
            intent = "indirect"
        elif any(x in q for x in ["compare", "difference", "versus", "vs", "better"]):
            intent = "comparison"
        elif any(x in q for x in ["how many", "count", "number of", "list all", "show all"]):
            intent = "multi_hop"
        detected_tiers: list[str] = []
        if "tier 1" in q:
            detected_tiers.append("tier 1")
        if "tier 2" in q:
            detected_tiers.append("tier 2")
        if "tier 1/2" in q or "tier1/2" in q:
            detected_tiers.extend(["tier 1", "tier 2"])

        role_terms = self._extract_role_terms(q)
        location_value = "georgia" if "georgia" in q else ""
        list_query = any(x in q for x in ["list", "show", "give", "all suppliers", "all companies"])
        return {
            "type": intent,
            "indirect": intent == "indirect",
            "comparison": intent == "comparison",
            "multi_hop": intent == "multi_hop",
            "list_query": list_query,
            "tier_filter": bool(detected_tiers),
            "detected_tiers": sorted(set(detected_tiers)),
            "role_terms": role_terms,
            "role_filter": bool(role_terms),
            "location_filter": bool(location_value),
            "location_value": location_value,
            "oem_filter": "oem" in q,
        }

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedDoc]:
        top_k = top_k or self.top_k
        encoded = encode_for_task(
            self.index.embedder,
            [question],
            task="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vector = np.asarray(encoded[0], dtype=float).tolist()
        return self.retrieve_with_vector(
            vector,
            question_text=question,
            top_k=top_k,
            intent=self.detect_query_intent(question),
        )

    def retrieve_with_vector(
        self,
        vector: list[float],
        question_text: str = "",
        top_k: int | None = None,
        intent: dict[str, Any] | None = None,
    ) -> list[RetrievedDoc]:
        top_k = top_k or self.top_k
        intent = intent or self.detect_query_intent(question_text)
        top_k = self._adaptive_top_k(top_k, intent)
        k = max(top_k * 2, 12)
        semantic_scores: dict[str, float] = {}
        for hit in self._semantic_query_hits(vector, min(k, len(self.index.documents))):
            doc_text = str(hit.get("text", ""))
            metadata = dict(hit.get("metadata", {}))
            doc_id = str(hit.get("doc_id") or self._find_doc_id(doc_text, metadata))
            semantic_scores[doc_id] = float(hit.get("similarity", 0.0))

        bm25_scores: dict[str, float] = {}
        if question_text.strip():
            raw_scores = self.index.bm25.get_scores(_tokenize(question_text))
            max_raw = float(np.max(raw_scores)) if len(raw_scores) else 1.0
            for doc, score in zip(self.index.documents, raw_scores):
                bm25_scores[doc["id"]] = float(score / max(max_raw, 1e-9))

        scored: list[RetrievedDoc] = []
        for doc in self.index.documents:
            doc_id = doc["id"]
            sem = semantic_scores.get(doc_id, 0.0)
            bm = bm25_scores.get(doc_id, 0.0)
            base_score = self.semantic_weight * sem + self.bm25_weight * bm
            boost = self._metadata_boost(doc.get("metadata", {}), intent)
            final_score = base_score + boost
            scored.append(
                RetrievedDoc(
                    id=doc_id,
                    text=doc["text"],
                    metadata=doc["metadata"],
                    score=final_score,
                    semantic_score=sem,
                    bm25_score=bm,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        if self.reranker is not None:
            max_candidates = int(self.config.get("reranker", {}).get("max_candidates", max(top_k, 16)))
            pool = scored[: min(len(scored), max(max_candidates, top_k))]
            return self.reranker.rerank(question_text=question_text, docs=pool, top_k=top_k)
        return scored[:top_k]

    def _adaptive_top_k(self, base_top_k: int, intent: dict[str, Any]) -> int:
        if not self.use_adaptive_top_k:
            return max(3, base_top_k)
        boost = 0
        if intent.get("multi_hop") or intent.get("list_query"):
            boost += 4
        if intent.get("comparison"):
            boost += 2
        return max(4, base_top_k + boost)

    def _metadata_boost(self, metadata: dict[str, Any], intent: dict[str, Any]) -> float:
        boost = 0.0
        category = str(metadata.get("category", "")).lower()
        role = str(metadata.get("role", "")).lower()
        location = str(metadata.get("location", "")).lower()
        company = str(metadata.get("company", "")).lower()

        if intent.get("location_filter"):
            if intent.get("location_value") in location:
                boost += self.location_match_boost
            else:
                boost += self.location_mismatch_penalty
        if intent.get("tier_filter"):
            tiers = intent.get("detected_tiers", [])
            if any(t in category for t in tiers):
                boost += self.tier_match_boost
            else:
                boost += self.tier_mismatch_penalty
        if intent.get("role_filter"):
            role_terms = intent.get("role_terms", [])
            if any(term in role for term in role_terms):
                boost += self.role_match_boost
        if intent.get("oem_filter"):
            if "oem" in category or "oem" in role or "oem" in company:
                boost += self.oem_match_boost
        return max(self.max_metadata_penalty, min(boost, self.max_metadata_boost))

    def _extract_role_terms(self, q: str) -> list[str]:
        vocab = [
            "battery",
            "cell",
            "pack",
            "electrolyte",
            "thermal",
            "charging",
            "powertrain",
            "materials",
            "electronics",
            "supply chain",
            "infrastructure",
        ]
        found = [t for t in vocab if t in q]
        return sorted(set(found))

    def build_context(self, docs: list[RetrievedDoc]) -> str:
        lines: list[str] = []
        total_chars = 0
        for i, doc in enumerate(docs, start=1):
            if doc.rerank_score is None:
                header = f"[DOC {i}] score={doc.score:.3f}"
            else:
                header = (
                    f"[DOC {i}] score={doc.score:.3f} "
                    f"(semantic={doc.semantic_score:.3f}, bm25={doc.bm25_score:.3f}, rerank={doc.rerank_score:.3f})"
                )
            doc_text = str(doc.text).strip()
            if len(doc_text) > self.max_doc_chars:
                doc_text = doc_text[: self.max_doc_chars].rsplit(" ", 1)[0].rstrip() + " ..."
            block = f"{header}\n{doc_text}\n"
            if total_chars + len(block) > self.max_context_chars:
                remaining = self.max_context_chars - total_chars
                if remaining <= 120:
                    break
                block = block[:remaining].rstrip()
                lines.append(block)
                total_chars += len(block)
                break
            lines.append(block)
            total_chars += len(block)
        return "\n".join(lines).strip()

    def _semantic_query_hits(self, vector: list[float], n_results: int) -> list[dict[str, Any]]:
        if self.semantic_backend is not None:
            return self.semantic_backend.query(vector, n_results=n_results)

        query_res = self.index.collection.query(
            query_embeddings=[vector],
            n_results=n_results,
            include=["distances", "metadatas", "documents"],
        )
        sem_docs = query_res.get("documents", [[]])[0]
        sem_metas = query_res.get("metadatas", [[]])[0]
        sem_distances = query_res.get("distances", [[]])[0]
        out: list[dict[str, Any]] = []
        for doc_text, meta, dist in zip(sem_docs, sem_metas, sem_distances):
            similarity = 1.0 / (1.0 + float(dist))
            out.append(
                {
                    "doc_id": self._find_doc_id(str(doc_text), dict(meta or {})),
                    "text": str(doc_text),
                    "metadata": dict(meta or {}),
                    "similarity": similarity,
                }
            )
        return out

    def _find_doc_id(self, doc_text: str, metadata: dict[str, Any]) -> str:
        for d in self.index.documents:
            if d["text"] == doc_text and d["metadata"] == metadata:
                return d["id"]
        company = (metadata or {}).get("company", "")
        for d in self.index.documents:
            if d["metadata"].get("company") == company and company:
                return d["id"]
        return self.index.documents[0]["id"]
