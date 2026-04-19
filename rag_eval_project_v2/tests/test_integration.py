from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from src.evaluator import METRIC_SETS
from src.generator import PipelineMode, build_prompt
from src.hyde import HyDEExpander
from src.score_validator import ScoreValidator
from src.splitter import load_split
from src.utils.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config(PROJECT_ROOT / "config" / "config.yaml")


def test_split_is_stratified(config: dict) -> None:
    train, test = load_split(config)
    assert len(train) == 35
    assert len(test) == 15
    train_cats = set(train["Use Case Category"])
    test_cats = set(test["Use Case Category"])
    assert train_cats == test_cats


def test_hyde_changes_vector(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    expander = HyDEExpander(config)
    q = "Which companies supply thermal management?"
    intent = {"type": "indirect", "indirect": True}

    class FakeEmbedder:
        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            if isinstance(texts, str):
                texts = [texts]
            vectors = []
            for text in texts:
                vec = np.zeros(16, dtype=float)
                for i, ch in enumerate(text.lower()):
                    idx = (ord(ch) + i) % 16
                    vec[idx] += 1.0
                norm = np.linalg.norm(vec) or 1.0
                vectors.append(vec / norm if normalize_embeddings else vec)
            arr = np.array(vectors)
            return arr if convert_to_numpy else arr.tolist()

    expander.embedding_model = FakeEmbedder()

    async def fake_expand(question: str, intent_dict: dict) -> str:
        return f"Expanded query with additional terms: {question}"

    monkeypatch.setattr(expander, "expand", fake_expand)
    expanded = asyncio.run(expander.expand(q, intent))
    v_raw = np.array(expander.get_search_vector(q))
    v_hyde = np.array(expander.get_search_vector(expanded))
    cosine = float(np.dot(v_raw, v_hyde) / (np.linalg.norm(v_raw) * np.linalg.norm(v_hyde)))
    assert cosine < 0.9999


def test_pipeline_prompts_differ() -> None:
    outputs = []
    for mode in PipelineMode:
        system, user = build_prompt("test q", mode, "context", "web")
        assert len(system) > 50
        assert len(user) > 20
        outputs.append(system + "\n" + user)
    assert len(set(outputs)) == len(PipelineMode)


def test_metric_sets_correct() -> None:
    assert "context_precision" not in METRIC_SETS["no_rag"]
    assert "faithfulness" in METRIC_SETS["rag"]
    assert "source_attribution" in METRIC_SETS["rag_pretrained"]


def test_validator_flags_bad_score(config: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    validator = ScoreValidator(config)

    async def fake_generate(*args, **kwargs) -> str:
        return "FLAG:score too high for clearly incorrect answer"

    monkeypatch.setattr(validator.local_qwen, "generate", fake_generate)
    result = asyncio.run(
        validator.validate(
            "answer_correctness",
            1.0,
            "List Tier 1 companies",
            "F&P Georgia; Hitachi Astemo",
            "I don't know",
        )
    )
    assert result["valid"] is False
