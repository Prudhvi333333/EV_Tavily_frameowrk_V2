from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from src.generator import OllamaGenerator, OpenRouterGenerator
from src.utils.config_loader import resolve_path
from src.utils.logger import get_logger

try:
    from tavily import TavilyClient
except Exception:  # pragma: no cover - optional dependency
    TavilyClient = None

try:
    from firecrawl import FirecrawlApp
except Exception:  # pragma: no cover - optional dependency
    FirecrawlApp = None

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover - optional dependency
    CrossEncoder = None


def _normalize_text(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _domain_of_url(url: str) -> str:
    host = urlparse(url).netloc.casefold().strip()
    if host.startswith("www."):
        return host[4:]
    return host


def _prepare_keywords(items: list[Any]) -> list[str]:
    normalized: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        # Support either one phrase per list item or comma-separated phrases.
        parts = [part.strip() for part in item.split(",")]
        for part in parts:
            phrase = _normalize_text(part)
            if phrase:
                normalized.add(phrase)
    return sorted(normalized)


def score_domain_keywords(text: str, keyword_config: dict[str, Any]) -> float:
    text_norm = _normalize_text(text)
    tier1 = _prepare_keywords(keyword_config.get("tier1", []))
    tier2 = _prepare_keywords(keyword_config.get("tier2", []))

    tier1_hits = sum(1 for kw in tier1 if kw in text_norm)
    tier2_hits = sum(1 for kw in tier2 if kw in text_norm)

    score = min(tier1_hits * 0.15, 1.0)
    if tier1_hits >= 1:
        score = min(score + tier2_hits * 0.05, 1.0)
    return round(float(score), 4)


@dataclass
class ValidationResult:
    accepted: bool
    low_confidence: bool
    final_score: float
    s1_keyword: float
    s2_semantic: float
    s3_llm: float
    s3_partial_relevance: float
    s3_reason: str
    url: str
    source_domain: str
    decision: str
    text_preview: str


def parse_judge_response(response: str, strict_mode: bool = True) -> tuple[float, float, str]:
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines: list[str] = []
        for line in cleaned.splitlines():
            if line.strip().startswith("```"):
                continue
            lines.append(line)
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
        score_raw = float(parsed.get("score", 0))
        partial_raw = parsed.get("partial_relevance", None)
        reason = str(parsed.get("reason", "")).strip() or "no_reason"
        score = round(min(max(score_raw / 10.0, 0.0), 1.0), 4)
        if partial_raw is None:
            partial = score
        else:
            partial = round(min(max(float(partial_raw), 0.0), 1.0), 4)
        return score, partial, reason
    except Exception as exc:
        if strict_mode:
            raise RuntimeError(f"Failed to parse LLM judge JSON: {response!r}") from exc
        return 0.5, 0.5, "parse_error"


class KBCentroidValidator:
    def __init__(self, collection: Any, embedding_model: Any) -> None:
        self.collection = collection
        self.embedding_model = embedding_model
        payload = self.collection.get(include=["embeddings"])
        all_embeddings = payload.get("embeddings", []) if isinstance(payload, dict) else []
        if all_embeddings is None:
            raise RuntimeError("Chroma collection returned no embeddings; cannot compute KB centroid.")
        if hasattr(all_embeddings, "__len__") and len(all_embeddings) == 0:
            raise RuntimeError("Chroma collection has no embeddings; cannot compute KB centroid.")
        arr = np.asarray(all_embeddings, dtype=float)
        centroid = np.mean(arr, axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0:
            raise RuntimeError("KB centroid norm is zero; invalid embedding state.")
        self.centroid = centroid / norm

    def score(self, text: str) -> float:
        vec = self.embedding_model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        vec = np.asarray(vec, dtype=float)
        vec_norm = float(np.linalg.norm(vec))
        if vec_norm == 0:
            return 0.0
        vec = vec / vec_norm
        similarity = float(np.dot(vec, self.centroid))
        similarity = max(0.0, min(1.0, similarity))
        return round(similarity, 4)


class ProofLogger:
    def __init__(self, config: dict[str, Any], logger: Any) -> None:
        self.logger = logger
        self.path = resolve_path(config, config["paths"]["logs_dir"]) / "web_validation_proof.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        question: str,
        search_query: str,
        result: ValidationResult,
        question_id: str = "",
        pipeline: str = "",
    ) -> None:
        try:
            if result.accepted:
                if result.low_confidence:
                    injected_as = (
                        f"[WEB | confidence: {result.final_score:.3f} | LOW_CONFIDENCE | "
                        f"source: {result.source_domain}]"
                    )
                else:
                    injected_as = f"[WEB | confidence: {result.final_score:.3f} | source: {result.source_domain}]"
            else:
                injected_as = None

            record = {
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "question": question,
                "question_id": question_id,
                "pipeline": pipeline,
                "search_query": search_query,
                "url": result.url,
                "source_domain": result.source_domain,
                "decision": result.decision,
                "final_score": result.final_score,
                "signals": {
                    "s1_keyword": result.s1_keyword,
                    "s2_semantic": result.s2_semantic,
                    "s3_llm": result.s3_llm,
                    "s3_partial_relevance": result.s3_partial_relevance,
                },
                "s3_reason": result.s3_reason,
                "injected_as": injected_as,
                "text_preview": result.text_preview[:240],
            }
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover - I/O resilience
            self.logger.warning("Proof logging failed for %s: %s", result.url, exc)


class DocumentValidator:
    def __init__(
        self,
        config: dict[str, Any],
        kb_centroid_validator: KBCentroidValidator,
        qwen_generator: Any,
        logger: Any,
    ) -> None:
        self.config = config
        self.logger = logger
        self.strict_mode = bool(config.get("runtime", {}).get("strict_mode", False))
        web_cfg = config.get("web_validator", {})
        self.threshold = float(web_cfg.get("threshold", 0.60))
        self.low_confidence_floor = float(web_cfg.get("low_confidence_floor", 0.55))
        self.llm_min_score = float(web_cfg.get("llm_min_score", 0.20))
        self.partial_relevance_floor = float(web_cfg.get("partial_relevance_floor", 0.20))
        self.partial_semantic_override_min = float(web_cfg.get("partial_semantic_override_min", 0.72))
        self.partial_keyword_override_min = float(web_cfg.get("partial_keyword_override_min", 0.45))
        weights = web_cfg.get("signal_weights", [0.40, 0.35, 0.25])
        if not isinstance(weights, list) or len(weights) != 3:
            raise ValueError("web_validator.signal_weights must be a list of 3 values.")
        self.weights = [float(w) for w in weights]
        self.keyword_config = web_cfg.get("domain_keywords", {})
        self.centroid_validator = kb_centroid_validator
        self.qwen = qwen_generator

    async def validate(self, text: str, url: str, question: str) -> ValidationResult:
        source_domain = _domain_of_url(url)
        preview = text[:240]

        if not text.strip():
            return ValidationResult(
                accepted=False,
                low_confidence=False,
                final_score=0.0,
                s1_keyword=0.0,
                s2_semantic=0.0,
                s3_llm=0.0,
                s3_partial_relevance=0.0,
                s3_reason="empty_extracted_text",
                url=url,
                source_domain=source_domain,
                decision="REJECTED: score=0.000",
                text_preview=preview,
            )

        s1 = score_domain_keywords(text, self.keyword_config)
        s2 = self.centroid_validator.score(text)
        s3_raw, s3_partial, s3_reason = await self._llm_relevance_judge(text, question)
        # Blend direct relevance score with partial-coverage estimate so 20% useful docs are not crushed to zero.
        s3 = round(max(s3_raw, 0.8 * s3_partial), 4)
        w1, w2, w3 = self.weights
        final = round(s1 * w1 + s2 * w2 + s3 * w3, 4)

        if s3 < self.llm_min_score:
            if (
                s3_partial >= self.partial_relevance_floor
                and s2 >= self.partial_semantic_override_min
                and s1 >= self.partial_keyword_override_min
            ):
                accepted = True
                low_confidence = True
                decision = (
                    f"ACCEPTED_PARTIAL_RELEVANCE: llm={s3:.3f}, partial={s3_partial:.3f}, "
                    f"s2={s2:.3f}, s1={s1:.3f}"
                )
            else:
                accepted = False
                low_confidence = False
                decision = f"REJECTED: llm_relevance={s3:.3f}"
        elif final >= self.threshold:
            accepted = True
            low_confidence = False
            decision = "ACCEPTED"
        elif final >= self.low_confidence_floor:
            accepted = True
            low_confidence = True
            decision = f"ACCEPTED_LOW_CONFIDENCE: score={final:.3f}"
        else:
            accepted = False
            low_confidence = False
            decision = f"REJECTED: score={final:.3f}"

        return ValidationResult(
            accepted=accepted,
            low_confidence=low_confidence,
            final_score=final,
            s1_keyword=s1,
            s2_semantic=s2,
            s3_llm=s3,
            s3_partial_relevance=s3_partial,
            s3_reason=s3_reason,
            url=url,
            source_domain=source_domain,
            decision=decision,
            text_preview=preview,
        )

    async def _llm_relevance_judge(self, text: str, question: str) -> tuple[float, float, str]:
        prompt = (
            "You are validating whether a crawled document is useful for a retrieval-augmented answer.\n"
            "Score relevance to the QUESTION, not to generic EV text.\n\n"
            "Rubric:\n"
            "0 = completely unrelated\n"
            "2 = only weak/tangential relevance (~10-20%)\n"
            "4 = partially useful (~20-40%)\n"
            "6 = moderately useful (~40-60%)\n"
            "8 = strongly useful (~60-80%)\n"
            "10 = directly answers key parts (>80%)\n"
            "Important: use score 0 only when truly unrelated.\n\n"
            f"Question:\n{question}\n\n"
            f"Document excerpt:\n{text[:1400]}\n\n"
            'Respond with ONLY JSON:\n{"score": <0-10>, "partial_relevance": <0.0-1.0>, "reason": "<one short sentence>"}'
        )
        response = await self.qwen.generate(prompt=prompt, system="Return strict JSON only.", temperature=0.0)
        return parse_judge_response(response, strict_mode=self.strict_mode)


class WebCrawler:
    def __init__(
        self,
        config: dict[str, Any],
        kb_collection: Any,
        embedding_model: Any,
    ) -> None:
        self.config = config
        self.logger = get_logger("web_crawler", config)
        self.strict_mode = bool(config.get("runtime", {}).get("strict_mode", False))

        crawler_cfg = config.get("crawler", {})
        self.timeout_per_url = float(crawler_cfg.get("timeout_per_url", 20))
        self.total_timeout = float(crawler_cfg.get("total_timeout", 60))
        self.max_urls = int(crawler_cfg.get("max_urls", 5))
        self.top_results = int(crawler_cfg.get("top_results", 2))
        self.cache_enabled = bool(crawler_cfg.get("cache_enabled", True))
        self.cache_dir = resolve_path(config, config["paths"]["web_cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tavily_key_env = str(crawler_cfg.get("tavily_api_key_env", "TAVILY_API_KEY"))
        self.firecrawl_key_env = str(crawler_cfg.get("firecrawl_api_key_env", "FIRECRAWL_API_KEY"))
        self.tavily_api_key = os.getenv(self.tavily_key_env, "").strip()
        self.firecrawl_api_key = os.getenv(self.firecrawl_key_env, "").strip()

        qwen_model = config.get("hyde", {}).get("model", "qwen2.5:14b")
        self.local_qwen = OllamaGenerator(qwen_model, strict=self.strict_mode)
        web_judge_cfg = config.get("web_validator", {}).get("judge", {})
        web_judge_provider = str(web_judge_cfg.get("provider", "ollama")).lower()
        web_judge_model = str(web_judge_cfg.get("model", qwen_model))
        if web_judge_provider in {"openrouter", "kimi_cloud", "kimi"}:
            self.web_relevance_judge = OpenRouterGenerator(
                web_judge_model,
                api_key_env=str(web_judge_cfg.get("api_key_env", "OPENROUTER_API_KEY")),
                base_url=str(web_judge_cfg.get("base_url", "https://openrouter.ai/api/v1")),
                strict=self.strict_mode,
            )
        else:
            self.web_relevance_judge = OllamaGenerator(web_judge_model, strict=self.strict_mode)
        self.centroid_validator = KBCentroidValidator(kb_collection, embedding_model)
        self.validator = DocumentValidator(config, self.centroid_validator, self.web_relevance_judge, self.logger)
        self.proof_logger = ProofLogger(config, self.logger)

        self.cross_encoder_model = config.get("embeddings", {}).get(
            "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._cross_encoder = None
        self._tavily_client = TavilyClient(api_key=self.tavily_api_key) if TavilyClient and self.tavily_api_key else None
        self._firecrawl_app = FirecrawlApp(api_key=self.firecrawl_api_key) if FirecrawlApp and self.firecrawl_api_key else None

    async def generate_search_query(self, question: str) -> str:
        prompt = (
            "Generate ONE web search query for this research question.\n"
            "Return only query text.\n"
            f"Question: {question}"
        )
        response = await self.local_qwen.generate(prompt=prompt, system="Return one query only.", temperature=0.0)
        query = response.strip().splitlines()[0].strip().strip('"') if response.strip() else ""
        if not query:
            if self.strict_mode:
                raise RuntimeError("Search query generation returned empty output.")
            return question
        return query

    async def extract_relevant(self, raw_text: str, question: str) -> str:
        prompt = (
            "Extract factual snippets that help answer the question.\n"
            "If only part of the text is relevant, keep that partial section.\n"
            "Keep original facts. Do not add new claims. Return plain text only.\n\n"
            f"Question: {question}\n"
            f"Text: {raw_text[:2500]}"
        )
        response = await self.local_qwen.generate(prompt=prompt, system="Extract relevant evidence only.")
        extracted = response.strip()
        if not extracted:
            extracted = raw_text[:1200].strip()
        return extracted

    async def crawl(self, question: str, question_id: str = "", pipeline: str = "") -> dict[str, Any]:
        cache_path = self._cache_path(question)
        if self.cache_enabled and cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                self.logger.warning("Web cache read failed for %s; rebuilding.", question[:80])

        result_payload = {"docs": [], "records": [], "search_query": "", "timed_out": False}
        accepted_docs: list[dict[str, Any]] = []
        validation_records: list[dict[str, Any]] = []
        try:
            async with asyncio.timeout(self.total_timeout):
                search_query = await self.generate_search_query(question)
                result_payload["search_query"] = search_query
                search_results = await self._search_tavily(search_query)
                if not search_results:
                    self.logger.warning("No Tavily search results for question: %s", question[:100])
                    return result_payload

                for item in search_results:
                    url = str(item.get("url", "")).strip()
                    if not url:
                        continue

                    raw_text = ""
                    extracted = ""
                    try:
                        raw_text = await asyncio.wait_for(
                            self._scrape_with_firecrawl(url),
                            timeout=self.timeout_per_url,
                        )
                    except TimeoutError:
                        self.logger.warning("Firecrawl timeout for URL: %s", url)
                        vr = ValidationResult(
                            accepted=False,
                            low_confidence=False,
                            final_score=0.0,
                            s1_keyword=0.0,
                            s2_semantic=0.0,
                            s3_llm=0.0,
                            s3_partial_relevance=0.0,
                            s3_reason="firecrawl_timeout",
                            url=url,
                            source_domain=_domain_of_url(url),
                            decision="REJECTED: score=0.000",
                            text_preview="",
                        )
                    except Exception as exc:
                        self.logger.warning("Firecrawl failed for %s: %s", url, exc)
                        vr = ValidationResult(
                            accepted=False,
                            low_confidence=False,
                            final_score=0.0,
                            s1_keyword=0.0,
                            s2_semantic=0.0,
                            s3_llm=0.0,
                            s3_partial_relevance=0.0,
                            s3_reason=f"firecrawl_error:{type(exc).__name__}",
                            url=url,
                            source_domain=_domain_of_url(url),
                            decision="REJECTED: score=0.000",
                            text_preview="",
                        )
                    else:
                        if not raw_text.strip():
                            vr = ValidationResult(
                                accepted=False,
                                low_confidence=False,
                                final_score=0.0,
                                s1_keyword=0.0,
                                s2_semantic=0.0,
                                s3_llm=0.0,
                                s3_partial_relevance=0.0,
                                s3_reason="firecrawl_empty",
                                url=url,
                                source_domain=_domain_of_url(url),
                                decision="REJECTED: score=0.000",
                                text_preview="",
                            )
                        else:
                            try:
                                extracted = await self.extract_relevant(raw_text, question)
                                validation_text = raw_text[:2800].strip() or extracted
                                vr = await self.validator.validate(validation_text, url, question)
                            except Exception as exc:
                                self.logger.warning("Validation failed for %s: %s", url, exc)
                                vr = ValidationResult(
                                    accepted=False,
                                    low_confidence=False,
                                    final_score=0.0,
                                    s1_keyword=0.0,
                                    s2_semantic=0.0,
                                    s3_llm=0.0,
                                    s3_partial_relevance=0.0,
                                    s3_reason=f"validation_error:{type(exc).__name__}",
                                    url=url,
                                    source_domain=_domain_of_url(url),
                                    decision="REJECTED: score=0.000",
                                    text_preview=(raw_text[:240] if raw_text else ""),
                                )

                    self.proof_logger.log(
                        question=question,
                        search_query=search_query,
                        result=vr,
                        question_id=question_id,
                        pipeline=pipeline,
                    )
                    rec = self._validation_record(vr)
                    validation_records.append(rec)
                    result_payload["records"] = list(validation_records)

                    if vr.accepted:
                        context_block = self._web_context_block(vr, vr.text_preview if not raw_text.strip() else extracted)
                        accepted_docs.append(
                            {
                                "url": vr.url,
                                "source_domain": vr.source_domain,
                                "text": extracted if raw_text.strip() else "",
                                "confidence": vr.final_score,
                                "low_confidence": vr.low_confidence,
                                "context_block": context_block,
                                "signals": {
                                    "s1_keyword": vr.s1_keyword,
                                    "s2_semantic": vr.s2_semantic,
                                    "s3_llm": vr.s3_llm,
                                    "s3_partial_relevance": vr.s3_partial_relevance,
                                },
                                "decision": vr.decision,
                            }
                        )

                result_payload["records"] = validation_records

                if not accepted_docs:
                    self.logger.warning("All web documents rejected for question: %s", question[:100])
                    if self.cache_enabled:
                        with cache_path.open("w", encoding="utf-8") as f:
                            json.dump(result_payload, f, ensure_ascii=False, indent=2)
                    return result_payload

                reranked = await self._rerank(question, accepted_docs)
                result_payload["docs"] = reranked[: self.top_results]

                if self.cache_enabled:
                    with cache_path.open("w", encoding="utf-8") as f:
                        json.dump(result_payload, f, ensure_ascii=False, indent=2)
                return result_payload
        except TimeoutError:
            self.logger.warning("Crawler timeout for question: %s", question[:100])
            result_payload["timed_out"] = True
            result_payload["records"] = list(validation_records)
            if accepted_docs:
                accepted_docs.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
                result_payload["docs"] = accepted_docs[: self.top_results]
            return result_payload

    async def _search_tavily(self, query: str) -> list[dict[str, str]]:
        if self._tavily_client is None:
            message = (
                f"Tavily is not configured. Install tavily-python and set {self.tavily_key_env}."
            )
            if self.strict_mode:
                raise RuntimeError(message)
            self.logger.warning(message)
            return []

        def _run() -> list[dict[str, str]]:
            response = self._tavily_client.search(
                query=query,
                max_results=self.max_urls,
                search_depth="advanced",
            )
            out: list[dict[str, str]] = []
            for item in response.get("results", []):
                url = str(item.get("url", "")).strip()
                if not url:
                    continue
                out.append(
                    {
                        "url": url,
                        "title": str(item.get("title", "")),
                        "content": str(item.get("content", "")),
                    }
                )
            return out

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:
            if self.strict_mode:
                raise RuntimeError("Tavily search failed.") from exc
            self.logger.warning("Tavily search failed: %s", exc)
            return []

    async def _scrape_with_firecrawl(self, url: str) -> str:
        if self._firecrawl_app is None:
            message = (
                f"Firecrawl is not configured. Install firecrawl and set {self.firecrawl_key_env}."
            )
            if self.strict_mode:
                raise RuntimeError(message)
            self.logger.warning(message)
            return ""

        def _run() -> str:
            response = self._firecrawl_app.scrape(url, formats=["markdown"])
            if isinstance(response, dict):
                direct = str(response.get("markdown", "") or response.get("content", "")).strip()
                if direct:
                    return direct
                data = response.get("data")
                if isinstance(data, dict):
                    nested = str(data.get("markdown", "") or data.get("content", "")).strip()
                    if nested:
                        return nested
            # object-style response compatibility
            if hasattr(response, "markdown"):
                value = str(getattr(response, "markdown", "")).strip()
                if value:
                    return value
            if hasattr(response, "data"):
                data = getattr(response, "data")
                if isinstance(data, dict):
                    nested = str(data.get("markdown", "") or data.get("content", "")).strip()
                    if nested:
                        return nested
            return ""

        try:
            content = await asyncio.to_thread(_run)
            return content[:12000]
        except Exception as exc:
            if self.strict_mode:
                raise RuntimeError(f"Firecrawl scrape failed for {url}.") from exc
            self.logger.warning("Firecrawl scrape failed for %s: %s", url, exc)
            return ""

    async def _rerank(self, question: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not docs:
            return []
        if CrossEncoder is None:
            raise RuntimeError("CrossEncoder is required for web reranking but is not installed.")
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(self.cross_encoder_model, local_files_only=True)
        pairs = [[question, str(d.get("text", ""))] for d in docs]
        scores = self._cross_encoder.predict(pairs)
        zipped = list(zip(docs, scores))
        zipped.sort(key=lambda x: float(x[1]), reverse=True)
        out: list[dict[str, Any]] = []
        for doc, score in zipped:
            row = dict(doc)
            row["rerank_score"] = float(score)
            out.append(row)
        return out

    def _validation_record(self, result: ValidationResult) -> dict[str, Any]:
        return {
            "url": result.url,
            "source_domain": result.source_domain,
            "decision": result.decision,
            "accepted": result.accepted,
            "low_confidence": result.low_confidence,
            "final_score": result.final_score,
            "signals": {
                "s1_keyword": result.s1_keyword,
                "s2_semantic": result.s2_semantic,
                "s3_llm": result.s3_llm,
                "s3_partial_relevance": result.s3_partial_relevance,
            },
            "s3_reason": result.s3_reason,
            "text_preview": result.text_preview,
        }

    def _web_context_block(self, result: ValidationResult, text: str) -> str:
        if result.low_confidence:
            header = (
                f"[WEB | confidence: {result.final_score:.3f} | LOW_CONFIDENCE | "
                f"source: {result.source_domain}]"
            )
        else:
            header = f"[WEB | confidence: {result.final_score:.3f} | source: {result.source_domain}]"
        return f"{header}\n{text.strip()}"

    def _cache_path(self, question: str) -> Path:
        digest = hashlib.sha256(question.strip().casefold().encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"
