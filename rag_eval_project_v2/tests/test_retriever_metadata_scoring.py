from __future__ import annotations

import numpy as np

from src.indexer import HybridIndex
from src.retriever import HybridRetriever


class _DummyCollection:
    def query(self, **kwargs):  # noqa: D401
        _ = kwargs
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}


class _DummyBM25:
    def get_scores(self, tokens):  # noqa: D401
        _ = tokens
        return np.array([1.0], dtype=float)


def _dummy_index() -> HybridIndex:
    docs = [
        {
            "id": "doc_1",
            "text": "Company: A",
            "metadata": {
                "category": "Tier 1/2",
                "location": "Georgia",
                "role": "Battery Pack",
                "company": "A",
            },
        }
    ]
    return HybridIndex(
        collection=_DummyCollection(),
        documents=docs,
        embedder=object(),
        bm25=_DummyBM25(),
        tokenized_corpus=[["company", "a"]],
    )


def test_metadata_scoring_applies_match_and_mismatch_penalties() -> None:
    cfg = {
        "runtime": {"strict_mode": True},
        "retrieval": {
            "top_k": 8,
            "adaptive_top_k": True,
            "semantic_weight": 0.7,
            "bm25_weight": 0.3,
            "backend": "hybrid",
            "metadata_scoring": {
                "location_match_boost": 0.08,
                "location_mismatch_penalty": -0.06,
                "tier_match_boost": 0.18,
                "tier_mismatch_penalty": -0.15,
                "role_match_boost": 0.10,
                "oem_match_boost": 0.05,
                "max_boost": 0.30,
                "max_penalty": -0.25,
            },
        },
        "reranker": {"enabled": False},
    }
    retriever = HybridRetriever(_dummy_index(), cfg)
    intent = {
        "location_filter": True,
        "location_value": "georgia",
        "tier_filter": True,
        "detected_tiers": ["tier 1", "tier 2"],
        "role_filter": False,
        "role_terms": [],
        "oem_filter": False,
    }

    matched = retriever._metadata_boost(
        metadata={"category": "Tier 1/2", "location": "West Point, Georgia"},
        intent=intent,
    )
    mismatched = retriever._metadata_boost(
        metadata={"category": "OEM Footprint", "location": "Austin, Texas"},
        intent=intent,
    )

    assert matched > 0
    assert mismatched < 0

