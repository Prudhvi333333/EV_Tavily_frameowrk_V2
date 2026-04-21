from __future__ import annotations

import pytest

from src.reranker import CrossEncoderReranker
from src.retriever import RetrievedDoc


class _FakeCrossEncoder:
    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        assert len(pairs) == 3
        return [0.1, 0.9, 0.4]


def _doc(doc_id: str, score: float) -> RetrievedDoc:
    return RetrievedDoc(
        id=doc_id,
        text=f"text for {doc_id}",
        metadata={"company": doc_id},
        score=score,
        semantic_score=score,
        bm25_score=score / 2,
    )


def test_cross_encoder_reranker_ranks_and_blends() -> None:
    cfg = {
        "reranker": {
            "enabled": True,
            "provider": "cross_encoder",
            "model": "cross-encoder/ms-marco-MiniLM-L6-v2",
            "max_candidates": 5,
            "blend_weight": 0.65,
            "batch_size": 4,
            "local_files_only": False,
        },
        "embeddings": {"local_files_only": False},
    }
    reranker = CrossEncoderReranker(cfg)
    reranker._model = _FakeCrossEncoder()
    docs = [_doc("d1", 0.30), _doc("d2", 0.50), _doc("d3", 0.40)]

    ranked = reranker.rerank("test query", docs, top_k=2)
    assert [d.id for d in ranked] == ["d2", "d3"]
    assert ranked[0].rerank_score == 0.9
    assert ranked[1].rerank_score == 0.4
    assert ranked[0].score >= ranked[1].score


def test_cross_encoder_reranker_requires_query_text() -> None:
    cfg = {"reranker": {"enabled": True, "provider": "cross_encoder"}}
    reranker = CrossEncoderReranker(cfg)
    with pytest.raises(RuntimeError):
        reranker.rerank("", [_doc("d1", 0.2)], top_k=1)
