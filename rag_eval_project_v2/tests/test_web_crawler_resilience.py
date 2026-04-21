from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.web_crawler import WebCrawler


class _FakeCollection:
    def get(self, include=None):  # noqa: D401
        _ = include
        return {"embeddings": [[0.1, 0.2, 0.3], [0.2, 0.1, 0.25]]}


class _FakeEmbedder:
    def encode_with_task(self, texts, task="generic", normalize_embeddings=True, convert_to_numpy=False):
        _ = (task, normalize_embeddings, convert_to_numpy)
        items = [texts] if isinstance(texts, str) else list(texts)
        out = [[0.1, 0.2, 0.3] for _ in items]
        if convert_to_numpy:
            import numpy as np

            return np.asarray(out, dtype=float)
        return out


def test_crawl_handles_tavily_failure_without_crashing(tmp_path: Path, monkeypatch) -> None:
    cfg = {
        "runtime": {"strict_mode": True},
        "paths": {
            "web_cache_dir": str(tmp_path / "web_cache"),
            "logs_dir": str(tmp_path / "logs"),
        },
        "crawler": {
            "cache_enabled": False,
            "max_urls": 3,
            "top_results": 2,
            "tavily_max_retries": 1,
            "fail_on_search_error": True,
            "tavily_api_key_env": "TAVILY_API_KEY",
            "firecrawl_api_key_env": "FIRECRAWL_API_KEY",
        },
        "hyde": {"model": "qwen2.5:14b"},
        "web_validator": {
            "rerank_enabled": False,
            "judge": {"provider": "ollama", "model": "kimi-k2.5:cloud"},
            "signal_weights": [0.4, 0.35, 0.25],
            "domain_keywords": {"tier1": ["ev"], "tier2": ["battery"]},
        },
    }

    crawler = WebCrawler(cfg, kb_collection=_FakeCollection(), embedding_model=_FakeEmbedder())

    async def _fake_query(_: str) -> str:
        return "ev suppliers georgia"

    async def _fail_search(_: str):
        raise RuntimeError("synthetic_tavily_failure")

    monkeypatch.setattr(crawler, "generate_search_query", _fake_query)
    monkeypatch.setattr(crawler, "_search_tavily", _fail_search)

    payload = asyncio.run(crawler.crawl("Which suppliers provide EV battery components in Georgia?"))
    assert payload.get("docs") == []
    assert payload.get("records") == []
    assert "search_error" in payload


def test_firecrawl_retry_recovers_after_initial_timeout(tmp_path: Path, monkeypatch) -> None:
    cfg = {
        "runtime": {"strict_mode": True},
        "paths": {
            "web_cache_dir": str(tmp_path / "web_cache"),
            "logs_dir": str(tmp_path / "logs"),
        },
        "crawler": {
            "cache_enabled": False,
            "timeout_per_url": 1,
            "firecrawl_max_retries": 2,
            "firecrawl_retry_backoff_sec": 0.01,
            "tavily_api_key_env": "TAVILY_API_KEY",
            "firecrawl_api_key_env": "FIRECRAWL_API_KEY",
        },
        "hyde": {"model": "qwen2.5:14b"},
        "web_validator": {
            "rerank_enabled": False,
            "judge": {"provider": "ollama", "model": "kimi-k2.5:cloud"},
            "signal_weights": [0.4, 0.35, 0.25],
            "domain_keywords": {"tier1": ["ev"], "tier2": ["battery"]},
        },
    }
    crawler = WebCrawler(cfg, kb_collection=_FakeCollection(), embedding_model=_FakeEmbedder())
    calls = {"count": 0}

    async def _flaky(_url: str) -> str:
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError()
        return "ok_text"

    monkeypatch.setattr(crawler, "_scrape_with_firecrawl", _flaky)
    out = asyncio.run(crawler._scrape_with_firecrawl_with_retry("https://example.com"))
    assert out == "ok_text"
    assert calls["count"] == 2


def test_firecrawl_retry_raises_after_retries_exhausted(tmp_path: Path, monkeypatch) -> None:
    cfg = {
        "runtime": {"strict_mode": True},
        "paths": {
            "web_cache_dir": str(tmp_path / "web_cache"),
            "logs_dir": str(tmp_path / "logs"),
        },
        "crawler": {
            "cache_enabled": False,
            "timeout_per_url": 1,
            "firecrawl_max_retries": 2,
            "firecrawl_retry_backoff_sec": 0.01,
            "tavily_api_key_env": "TAVILY_API_KEY",
            "firecrawl_api_key_env": "FIRECRAWL_API_KEY",
        },
        "hyde": {"model": "qwen2.5:14b"},
        "web_validator": {
            "rerank_enabled": False,
            "judge": {"provider": "ollama", "model": "kimi-k2.5:cloud"},
            "signal_weights": [0.4, 0.35, 0.25],
            "domain_keywords": {"tier1": ["ev"], "tier2": ["battery"]},
        },
    }
    crawler = WebCrawler(cfg, kb_collection=_FakeCollection(), embedding_model=_FakeEmbedder())

    async def _always_timeout(_url: str) -> str:
        raise TimeoutError()

    monkeypatch.setattr(crawler, "_scrape_with_firecrawl", _always_timeout)
    with pytest.raises(TimeoutError):
        asyncio.run(crawler._scrape_with_firecrawl_with_retry("https://example.com"))


def test_tavily_answer_fallback_is_used_and_cached(tmp_path: Path, monkeypatch) -> None:
    cfg = {
        "runtime": {"strict_mode": True},
        "paths": {
            "web_cache_dir": str(tmp_path / "web_cache"),
            "logs_dir": str(tmp_path / "logs"),
        },
        "crawler": {
            "cache_enabled": True,
            "max_urls": 3,
            "top_results": 2,
            "tavily_max_retries": 1,
            "tavily_answer_fallback": {
                "enabled": True,
                "max_retries": 1,
                "retry_backoff_sec": 0.01,
                "search_depth": "advanced",
                "max_results": 3,
            },
            "tavily_api_key_env": "TAVILY_API_KEY",
            "firecrawl_api_key_env": "FIRECRAWL_API_KEY",
        },
        "hyde": {"model": "qwen2.5:14b"},
        "web_validator": {
            "rerank_enabled": False,
            "judge": {"provider": "ollama", "model": "kimi-k2.5:cloud"},
            "signal_weights": [0.4, 0.35, 0.25],
            "domain_keywords": {"tier1": ["ev"], "tier2": ["battery"]},
        },
    }
    crawler = WebCrawler(cfg, kb_collection=_FakeCollection(), embedding_model=_FakeEmbedder())

    async def _fake_query(_: str) -> str:
        return "ev battery suppliers georgia"

    async def _no_results(_: str) -> list[dict[str, str]]:
        return []

    class _FakeTavily:
        def __init__(self) -> None:
            self.calls = 0

        def qna_search(self, **kwargs) -> str:  # noqa: D401
            _ = kwargs
            self.calls += 1
            return "Fallback answer from Tavily."

    fake_tavily = _FakeTavily()
    crawler._tavily_client = fake_tavily
    monkeypatch.setattr(crawler, "generate_search_query", _fake_query)
    monkeypatch.setattr(crawler, "_search_tavily", _no_results)

    q = "Which suppliers provide EV battery components in Georgia?"
    first = asyncio.run(crawler.crawl(q))
    assert first.get("fallback_used") is True
    assert first.get("fallback_source") == "tavily_qna_search"
    assert "Fallback answer from Tavily." in str(first.get("fallback_answer", ""))
    assert len(first.get("docs", [])) == 1
    assert "WEB_FALLBACK" in str(first["docs"][0].get("context_block", ""))
    assert fake_tavily.calls == 1

    second = asyncio.run(crawler.crawl(q))
    assert second.get("fallback_used") is True
    assert fake_tavily.calls == 1
