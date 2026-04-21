from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluator import RAGASEvaluator
from src.generator import OllamaGenerator
from src.utils.config_loader import load_config
from src.utils.ollama import resolve_ollama_base_url


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_ollama_host_env_is_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11435")
    assert resolve_ollama_base_url() == "http://127.0.0.1:11435"


def test_generator_uses_runtime_ollama_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11435")
    generator = OllamaGenerator("qwen2.5:14b", strict=True)
    assert generator.base_url == "http://127.0.0.1:11435"


def test_evaluator_json_parser_disallows_regex_salvage() -> None:
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    evaluator = RAGASEvaluator(cfg)
    parsed = evaluator._parse_json_object("prefix {\"score\": 1} suffix")  # intentional malformed payload
    assert parsed is None

