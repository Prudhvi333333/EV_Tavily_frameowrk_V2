from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.indexer import HybridIndex


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


class HybridRetriever:
    def __init__(self, index: HybridIndex, config: dict[str, Any]) -> None:
        self.index = index
        self.config = config
        self.id_to_doc = {d["id"]: d for d in self.index.documents}
        self.top_k = int(config["retrieval"]["top_k"])
        self.semantic_weight = float(config["retrieval"]["semantic_weight"])
        self.bm25_weight = float(config["retrieval"]["bm25_weight"])

    def detect_query_intent(self, question: str) -> dict[str, Any]:
        q = question.lower()
        intent = "direct"
        if any(x in q for x in ["indirect", "through", "linked", "relationship", "multi-hop"]):
            intent = "indirect"
        elif any(x in q for x in ["compare", "difference", "versus", "vs", "better"]):
            intent = "comparison"
        elif any(x in q for x in ["how many", "count", "number of", "list all", "show all"]):
            intent = "multi_hop"
        return {
            "type": intent,
            "indirect": intent == "indirect",
            "comparison": intent == "comparison",
            "multi_hop": intent == "multi_hop",
        }

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedDoc]:
        top_k = top_k or self.top_k
        encoded = self.index.embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True)
        vector = np.asarray(encoded[0], dtype=float).tolist()
        return self.retrieve_with_vector(vector, question_text=question, top_k=top_k)

    def retrieve_with_vector(
        self,
        vector: list[float],
        question_text: str = "",
        top_k: int | None = None,
    ) -> list[RetrievedDoc]:
        top_k = top_k or self.top_k
        k = max(top_k * 2, 10)
        query_res = self.index.collection.query(
            query_embeddings=[vector],
            n_results=min(k, len(self.index.documents)),
            include=["distances", "metadatas", "documents"],
        )
        sem_docs = query_res["documents"][0]
        sem_metas = query_res["metadatas"][0]
        sem_distances = query_res["distances"][0]

        semantic_scores: dict[str, float] = {}
        for doc_text, meta, dist in zip(sem_docs, sem_metas, sem_distances):
            doc_id = self._find_doc_id(doc_text, meta)
            similarity = 1.0 / (1.0 + float(dist))
            semantic_scores[doc_id] = similarity

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
            final_score = self.semantic_weight * sem + self.bm25_weight * bm
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
        return scored[:top_k]

    def build_context(self, docs: list[RetrievedDoc]) -> str:
        lines = []
        for i, doc in enumerate(docs, start=1):
            lines.append(f"[DOC {i}] score={doc.score:.3f}")
            lines.append(doc.text)
            lines.append("")
        return "\n".join(lines).strip()

    def _find_doc_id(self, doc_text: str, metadata: dict[str, Any]) -> str:
        for d in self.index.documents:
            if d["text"] == doc_text and d["metadata"] == metadata:
                return d["id"]
        company = (metadata or {}).get("company", "")
        for d in self.index.documents:
            if d["metadata"].get("company") == company and company:
                return d["id"]
        return self.index.documents[0]["id"]
