from __future__ import annotations

import asyncio

from src.web_crawler import DocumentValidator, parse_judge_response, score_domain_keywords


class _FakeCentroidValidator:
    def __init__(self, score_value: float) -> None:
        self._score = score_value

    def score(self, text: str) -> float:
        return self._score


class _FakeQwen:
    def __init__(self, response_text: str) -> None:
        self._response = response_text

    async def generate(self, *args, **kwargs) -> str:
        return self._response


def test_score_domain_keywords_respects_tier_logic() -> None:
    config = {
        "tier1": ["electric vehicle", "battery cell", "tier 1"],
        "tier2": ["automotive", "georgia", "facility"],
    }
    off_domain = "General manufacturing facility update in georgia."
    on_domain = "Georgia electric vehicle battery cell production at tier 1 automotive facility."

    off_score = score_domain_keywords(off_domain, config)
    on_score = score_domain_keywords(on_domain, config)

    assert off_score == 0.0
    assert on_score > 0.4


def test_parse_judge_response_strict_json() -> None:
    response = '{"score": 8, "reason": "Clearly focused on Georgia EV supply chain."}'
    score, reason = parse_judge_response(response, strict_mode=True)
    assert score == 0.8
    assert "Georgia EV supply chain" in reason


def test_document_validator_accepts_low_and_rejects() -> None:
    cfg = {
        "runtime": {"strict_mode": True},
        "web_validator": {
            "threshold": 0.60,
            "low_confidence_floor": 0.55,
            "signal_weights": [0.40, 0.35, 0.25],
            "domain_keywords": {
                "tier1": ["electric vehicle", "battery cell", "supply chain", "tier 1"],
                "tier2": ["automotive", "georgia", "facility"],
            },
        },
    }

    validator = DocumentValidator(
        config=cfg,
        kb_centroid_validator=_FakeCentroidValidator(0.70),
        qwen_generator=_FakeQwen('{"score": 7, "reason": "Mostly relevant."}'),
        logger=None,
    )

    accepted = asyncio.run(
        validator.validate(
            text="Georgia electric vehicle battery cell supply chain facility update.",
            url="https://example.com/relevant",
            question="Which Georgia EV battery companies are involved?",
        )
    )
    assert accepted.accepted is True
    assert accepted.final_score >= 0.60

    rejected = asyncio.run(
        validator.validate(
            text="Random sports article with no domain signal.",
            url="https://example.com/offtopic",
            question="Which Georgia EV battery companies are involved?",
        )
    )
    assert rejected.accepted is False
    assert rejected.final_score < 0.55


def test_document_validator_rejects_when_llm_signal_is_too_low() -> None:
    cfg = {
        "runtime": {"strict_mode": True},
        "web_validator": {
            "threshold": 0.60,
            "low_confidence_floor": 0.55,
            "llm_min_score": 0.20,
            "signal_weights": [0.40, 0.35, 0.25],
            "domain_keywords": {
                "tier1": ["electric vehicle", "battery cell", "supply chain", "tier 1"],
                "tier2": ["automotive", "georgia", "facility"],
            },
        },
    }
    validator = DocumentValidator(
        config=cfg,
        kb_centroid_validator=_FakeCentroidValidator(0.95),
        qwen_generator=_FakeQwen('{"score": 0, "reason": "Not relevant."}'),
        logger=None,
    )
    result = asyncio.run(
        validator.validate(
            text="Georgia electric vehicle battery cell supply chain facility update.",
            url="https://example.com/false-positive-risk",
            question="Which Georgia EV battery companies are involved?",
        )
    )
    assert result.accepted is False
    assert "llm_relevance" in result.decision
