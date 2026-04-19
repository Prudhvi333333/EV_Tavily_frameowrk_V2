from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

from src.generator import OllamaGenerator
from src.utils.config_loader import resolve_path
from src.utils.logger import get_logger

try:
    from duckduckgo_search import DDGS
except Exception:  # pragma: no cover - optional dependency
    DDGS = None

try:
    from tavily import TavilyClient
except Exception:  # pragma: no cover - optional dependency
    TavilyClient = None

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional dependency
    CrossEncoder = None


class WebCrawler:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger("web_crawler", config)
        crawler_cfg = config.get("crawler", {})
        self.timeout_per_url = float(crawler_cfg.get("timeout_per_url", 8))
        self.total_timeout = float(crawler_cfg.get("total_timeout", 30))
        self.max_urls = int(crawler_cfg.get("max_urls", 5))
        self.top_results = int(crawler_cfg.get("top_results", 2))
        self.cache_enabled = bool(crawler_cfg.get("cache_enabled", True))
        self.cache_dir = resolve_path(config, config["paths"]["web_cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.local_qwen = OllamaGenerator(config.get("hyde", {}).get("model", "qwen2.5:14b"))
        self.cross_encoder_model = config.get("embeddings", {}).get(
            "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._cross_encoder = None

    async def generate_search_query(self, question: str) -> str:
        prompt = (
            f"Generate ONE high-quality web search query for this question:\n{question}\n"
            "Return only the query text."
        )
        response = await self.local_qwen.generate(prompt=prompt, system="Return one search query.")
        query = response.strip().splitlines()[0] if response.strip() else question
        return query.strip('" ')

    async def fetch_url(self, url: str) -> str | None:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Eval/2.0)"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout_per_url, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
            text = " ".join(paragraphs)
            text = re.sub(r"\s+", " ", text).strip()
            return text[:2000] if text else None
        except Exception:
            return None

    async def extract_relevant(self, raw_text: str, question: str) -> str:
        prompt = (
            "From the text below, extract 2-4 sentences that are directly relevant to the question.\n"
            "Return only those sentences.\n\n"
            f"Question: {question}\n"
            f"Text: {raw_text[:1500]}"
        )
        extracted = await self.local_qwen.generate(prompt=prompt, system="Extract relevant evidence only.")
        cleaned = extracted.strip()
        if cleaned:
            return cleaned
        return self._keyword_extract(raw_text, question)

    async def crawl(self, question: str) -> list[dict[str, str]]:
        cache_path = self._cache_path(question)
        if self.cache_enabled and cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        try:
            async with asyncio.timeout(self.total_timeout):
                query = await self.generate_search_query(question)
                urls = await self._search_web(query)
                if not urls:
                    return []

                fetched = await asyncio.gather(*(self.fetch_url(url) for url in urls))
                candidates: list[dict[str, str]] = []
                for url, raw in zip(urls, fetched):
                    if not raw:
                        continue
                    relevant = await self.extract_relevant(raw, question)
                    if relevant.strip():
                        candidates.append({"url": url, "text": relevant.strip()})

                ranked = await self._rerank(question, candidates)
                result = ranked[: self.top_results]
                if self.cache_enabled:
                    with cache_path.open("w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                return result
        except TimeoutError:
            self.logger.warning("Crawler timeout for question: %s", question)
            return []
        except Exception as e:
            self.logger.warning("Crawler error: %s", e)
            return []

    async def _search_web(self, query: str) -> list[str]:
        urls = await self._search_ddg(query)
        if urls:
            return urls[: self.max_urls]
        urls = await self._search_tavily(query)
        return urls[: self.max_urls]

    async def _search_ddg(self, query: str) -> list[str]:
        if DDGS is None:
            return []

        def _run() -> list[str]:
            out: list[str] = []
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=self.max_urls):
                    href = item.get("href") or item.get("url")
                    if href:
                        out.append(href)
            return out

        try:
            return await asyncio.to_thread(_run)
        except Exception:
            return []

    async def _search_tavily(self, query: str) -> list[str]:
        if TavilyClient is None:
            return []
        api_key = os.getenv("TAVILY_API_KEY", "").strip()
        if not api_key:
            return []

        def _run() -> list[str]:
            client = TavilyClient(api_key=api_key)
            result = client.search(query=query, max_results=self.max_urls)
            urls: list[str] = []
            for item in result.get("results", []):
                url = item.get("url")
                if url:
                    urls.append(url)
            return urls

        try:
            return await asyncio.to_thread(_run)
        except Exception:
            return []

    async def _rerank(self, question: str, docs: list[dict[str, str]]) -> list[dict[str, str]]:
        if not docs:
            return []
        if CrossEncoder is None:
            return self._rerank_lexical(question, docs)
        try:
            if self._cross_encoder is None:
                self._cross_encoder = CrossEncoder(self.cross_encoder_model, local_files_only=True)
            pairs = [[question, d["text"]] for d in docs]
            scores = self._cross_encoder.predict(pairs)
            with_scores = list(zip(docs, scores))
            with_scores.sort(key=lambda x: float(x[1]), reverse=True)
            return [d for d, _ in with_scores]
        except Exception:
            return self._rerank_lexical(question, docs)

    def _rerank_lexical(self, question: str, docs: list[dict[str, str]]) -> list[dict[str, str]]:
        q_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        scored: list[tuple[dict[str, str], float]] = []
        for d in docs:
            d_terms = set(re.findall(r"[a-z0-9]+", d["text"].lower()))
            overlap = len(q_terms & d_terms) / max(len(q_terms), 1)
            scored.append((d, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored]

    def _keyword_extract(self, raw_text: str, question: str) -> str:
        keywords = [k for k in re.findall(r"[a-zA-Z0-9]+", question.lower()) if len(k) > 3]
        sentences = re.split(r"(?<=[.!?])\s+", raw_text)
        picked: list[str] = []
        for s in sentences:
            sl = s.lower()
            if any(k in sl for k in keywords):
                picked.append(s.strip())
            if len(picked) >= 4:
                break
        return " ".join(picked[:4]).strip()[:800]

    def _cache_path(self, question: str) -> Path:
        digest = hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"
