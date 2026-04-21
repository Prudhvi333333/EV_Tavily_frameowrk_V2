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
from src.utils.embeddings import encode_for_task
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

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


def _normalize_text(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _domain_of_url(url: str) -> str:
    host = urlparse(url).netloc.casefold().strip()
    if host.startswith("www."):
        return host[4:]
    return host


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


class RegistryMetadataFilter:
    def __init__(self, config: dict[str, Any], logger: Any, strict_mode: bool) -> None:
        md_cfg = config.get("crawler", {}).get("metadata_filtering", {})
        self.enabled = bool(md_cfg.get("enabled", False))
        self.strict_mode = strict_mode
        self.logger = logger
        registry_raw = str(md_cfg.get("registry_path", "rag_data_management_registry.xlsx"))
        if self.enabled:
            if Path(registry_raw).is_absolute() or "_meta" in config:
                self.registry_path = resolve_path(config, registry_raw)
            else:
                self.registry_path = Path(registry_raw).resolve()
        else:
            self.registry_path = Path(registry_raw)
        self.min_metadata_score = float(md_cfg.get("min_metadata_score", 0.0))
        self.min_credibility_score = float(md_cfg.get("min_credibility_score", 0.0))
        self.block_rejected_domains = bool(md_cfg.get("block_rejected_domains", True))
        self.allow_decisions = {
            str(x).strip().casefold()
            for x in md_cfg.get("allow_decisions", ["keep", "selected", "approved", "accept"])
            if str(x).strip()
        }
        self.block_decisions = {
            str(x).strip().casefold()
            for x in md_cfg.get("block_decisions", ["discard", "rejected", "reject", "drop", "blocked"])
            if str(x).strip()
        }
        self._loaded = False
        self.allowed_domains: set[str] = set()
        self.blocked_domains: dict[str, str] = {}
        self.domain_scores: dict[str, dict[str, float | None]] = {}

    def _domain_from_row(self, row: dict[str, Any]) -> str:
        domain_keys = [
            "Source_Domain",
            "source_domain",
            "Domain",
            "domain",
        ]
        for key in domain_keys:
            value = str(row.get(key, "")).strip()
            if value:
                cleaned = value.casefold()
                return cleaned[4:] if cleaned.startswith("www.") else cleaned

        url_keys = ["Document_URL", "Source_URL", "url", "URL"]
        for key in url_keys:
            value = str(row.get(key, "")).strip()
            if value:
                return _domain_of_url(value)
        return ""

    def _rows(self, path: str, sheet_name: str) -> list[dict[str, Any]]:
        if pd is None:
            raise RuntimeError("pandas is required for metadata registry filtering.")
        try:
            frame = pd.read_excel(path, sheet_name=sheet_name)
        except ValueError:
            return []
        if frame.empty:
            return []
        return frame.to_dict(orient="records")

    def _load(self) -> None:
        if self._loaded or not self.enabled:
            return
        self._loaded = True

        if not self.registry_path.exists():
            raise RuntimeError(
                f"Metadata registry file is missing: {self.registry_path}"
            )

        registry = str(self.registry_path)
        review_rows = self._rows(registry, "Review_Ready")
        rejected_rows = self._rows(registry, "Rejected_Documents")
        failed_rows = self._rows(registry, "Failed_Acquisitions")

        for row in review_rows:
            domain = self._domain_from_row(row)
            if not domain:
                continue
            decision = str(row.get("Final_Decision", "")).strip().casefold()
            metadata_score = _float_or_none(row.get("Metadata_Score"))
            credibility_score = _float_or_none(row.get("Credibility_Score"))

            if domain not in self.domain_scores:
                self.domain_scores[domain] = {"metadata": metadata_score, "credibility": credibility_score}
            else:
                current = self.domain_scores[domain]
                if metadata_score is not None:
                    prev = current.get("metadata")
                    current["metadata"] = metadata_score if prev is None else max(float(prev), metadata_score)
                if credibility_score is not None:
                    prev = current.get("credibility")
                    current["credibility"] = credibility_score if prev is None else max(float(prev), credibility_score)

            if decision in self.allow_decisions:
                self.allowed_domains.add(domain)
            elif decision in self.block_decisions and self.block_rejected_domains:
                self.blocked_domains.setdefault(domain, "registry_review_blocked")

        if self.block_rejected_domains:
            for row in rejected_rows:
                domain = self._domain_from_row(row)
                if not domain:
                    continue
                category = str(row.get("Rejection_Category", "")).strip().casefold() or "registry_rejected_document"
                self.blocked_domains.setdefault(domain, category)

            for row in failed_rows:
                domain = self._domain_from_row(row)
                if not domain:
                    continue
                status = str(row.get("Acquisition_Status", "")).strip().casefold()
                if any(flag in status for flag in ["failed", "blocked", "forbidden"]):
                    self.blocked_domains.setdefault(domain, "registry_failed_acquisition")

        self.logger.info(
            "Metadata registry loaded | path=%s | allow_domains=%s | block_domains=%s",
            self.registry_path,
            len(self.allowed_domains),
            len(self.blocked_domains),
        )

    def filter_search_results(self, results: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
        if not self.enabled:
            return results, []

        self._load()

        accepted: list[dict[str, str]] = []
        rejected: list[dict[str, Any]] = []

        for item in results:
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            domain = _domain_of_url(url)
            if not domain:
                accepted.append(item)
                continue

            if domain in self.allowed_domains:
                accepted.append(item)
                continue

            if domain in self.blocked_domains:
                rejected.append(
                    {
                        "url": url,
                        "source_domain": domain,
                        "reason": f"registry_domain_block:{self.blocked_domains[domain]}",
                        "metadata_score": None,
                        "credibility_score": None,
                    }
                )
                continue

            scores = self.domain_scores.get(domain, {})
            metadata_score = _float_or_none(scores.get("metadata"))
            credibility_score = _float_or_none(scores.get("credibility"))

            if metadata_score is not None and metadata_score < self.min_metadata_score:
                rejected.append(
                    {
                        "url": url,
                        "source_domain": domain,
                        "reason": f"registry_metadata_score_below_threshold:{metadata_score:.2f}",
                        "metadata_score": metadata_score,
                        "credibility_score": credibility_score,
                    }
                )
                continue

            if credibility_score is not None and credibility_score < self.min_credibility_score:
                rejected.append(
                    {
                        "url": url,
                        "source_domain": domain,
                        "reason": f"registry_credibility_score_below_threshold:{credibility_score:.2f}",
                        "metadata_score": metadata_score,
                        "credibility_score": credibility_score,
                    }
                )
                continue

            accepted.append(item)

        self.logger.info(
            "Metadata filter applied | candidates=%s | accepted=%s | rejected=%s",
            len(results),
            len(accepted),
            len(rejected),
        )
        return accepted, rejected


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
        try:
            parsed = json.loads(cleaned)
        except Exception:
            # Some judge models may prepend short text before JSON.
            start = cleaned.find("{")
            if start < 0:
                raise
            decoder = json.JSONDecoder()
            parsed, _ = decoder.raw_decode(cleaned[start:])
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
        vec = encode_for_task(
            self.embedding_model,
            [text],
            task="document",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
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
        self.tavily_timeout = float(crawler_cfg.get("tavily_timeout", 75))
        self.tavily_max_retries = max(int(crawler_cfg.get("tavily_max_retries", 3)), 1)
        self.tavily_retry_backoff_sec = max(float(crawler_cfg.get("tavily_retry_backoff_sec", 2.0)), 0.1)
        self.firecrawl_max_retries = max(int(crawler_cfg.get("firecrawl_max_retries", 2)), 1)
        self.firecrawl_retry_backoff_sec = max(float(crawler_cfg.get("firecrawl_retry_backoff_sec", 1.5)), 0.1)
        self.fail_on_search_error = bool(crawler_cfg.get("fail_on_search_error", False))
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
        keep_alive = str(config.get("runtime", {}).get("ollama_keep_alive", "0s"))
        ollama_options = dict(config.get("runtime", {}).get("ollama_options", {}))
        self.local_qwen = OllamaGenerator(
            qwen_model,
            strict=self.strict_mode,
            keep_alive=keep_alive,
            options=ollama_options,
        )
        web_judge_cfg = config.get("web_validator", {}).get("judge", {})
        self.web_rerank_enabled = bool(config.get("web_validator", {}).get("rerank_enabled", True))
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
            self.web_relevance_judge = OllamaGenerator(
                web_judge_model,
                strict=self.strict_mode,
                keep_alive=keep_alive,
                options=ollama_options,
            )
        self.centroid_validator = KBCentroidValidator(kb_collection, embedding_model)
        self.validator = DocumentValidator(config, self.centroid_validator, self.web_relevance_judge, self.logger)
        self.proof_logger = ProofLogger(config, self.logger)
        self.registry_filter = RegistryMetadataFilter(config, self.logger, strict_mode=self.strict_mode)

        self.cross_encoder_model = config.get("embeddings", {}).get(
            "cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L6-v2"
        )
        self.cross_encoder_local_files_only = bool(
            config.get("web_validator", {}).get("cross_encoder_local_files_only", False)
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
                try:
                    search_results = await self._search_tavily(search_query)
                except Exception as exc:
                    self.logger.warning("Tavily search failed for question '%s': %s", question[:100], exc)
                    result_payload["search_error"] = f"{type(exc).__name__}:{exc}"
                    if self.cache_enabled:
                        with cache_path.open("w", encoding="utf-8") as f:
                            json.dump(result_payload, f, ensure_ascii=False, indent=2)
                    return result_payload
                if not search_results:
                    self.logger.warning("No Tavily search results for question: %s", question[:100])
                    return result_payload

                try:
                    filtered_results, registry_rejections = self.registry_filter.filter_search_results(search_results)
                except Exception as exc:
                    raise RuntimeError("Metadata registry filtering failed.") from exc

                for reject in registry_rejections:
                    reason = str(reject.get("reason", "registry_rejected"))
                    source_domain = str(reject.get("source_domain", ""))
                    url = str(reject.get("url", ""))
                    metadata_score = reject.get("metadata_score")
                    credibility_score = reject.get("credibility_score")
                    score_hint = []
                    if metadata_score is not None:
                        score_hint.append(f"metadata={float(metadata_score):.2f}")
                    if credibility_score is not None:
                        score_hint.append(f"credibility={float(credibility_score):.2f}")
                    detail = f"{reason}" if not score_hint else f"{reason} ({', '.join(score_hint)})"
                    vr = ValidationResult(
                        accepted=False,
                        low_confidence=False,
                        final_score=0.0,
                        s1_keyword=0.0,
                        s2_semantic=0.0,
                        s3_llm=0.0,
                        s3_partial_relevance=0.0,
                        s3_reason=detail,
                        url=url,
                        source_domain=source_domain,
                        decision=f"REJECTED_REGISTRY: {detail}",
                        text_preview="",
                    )
                    self.proof_logger.log(
                        question=question,
                        search_query=search_query,
                        result=vr,
                        question_id=question_id,
                        pipeline=pipeline,
                    )
                    validation_records.append(self._validation_record(vr))

                search_results = filtered_results
                result_payload["records"] = list(validation_records)
                if not search_results:
                    self.logger.warning("All Tavily candidates rejected by metadata policy for question: %s", question[:100])
                    if self.cache_enabled:
                        with cache_path.open("w", encoding="utf-8") as f:
                            json.dump(result_payload, f, ensure_ascii=False, indent=2)
                    return result_payload

                for item in search_results:
                    url = str(item.get("url", "")).strip()
                    if not url:
                        continue

                    raw_text = ""
                    extracted = ""
                    try:
                        raw_text = await self._scrape_with_firecrawl_with_retry(url)
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
                timeout=self.tavily_timeout,
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

        last_exc: Exception | None = None
        for attempt in range(1, self.tavily_max_retries + 1):
            try:
                return await asyncio.to_thread(_run)
            except Exception as exc:
                last_exc = exc
                if attempt < self.tavily_max_retries:
                    wait_s = self.tavily_retry_backoff_sec * attempt
                    self.logger.warning(
                        "Tavily search attempt %s/%s failed for query '%s': %s. Retrying in %.1fs",
                        attempt,
                        self.tavily_max_retries,
                        query[:90],
                        exc,
                        wait_s,
                    )
                    await asyncio.sleep(wait_s)
                    continue

        self.logger.warning(
            "Tavily search failed after %s attempts for query '%s': %s",
            self.tavily_max_retries,
            query[:90],
            last_exc,
        )
        if self.fail_on_search_error:
            raise RuntimeError("Tavily search failed.") from last_exc
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

    async def _scrape_with_firecrawl_with_retry(self, url: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(1, self.firecrawl_max_retries + 1):
            try:
                return await asyncio.wait_for(
                    self._scrape_with_firecrawl(url),
                    timeout=self.timeout_per_url,
                )
            except TimeoutError as exc:
                last_exc = exc
                if attempt < self.firecrawl_max_retries:
                    wait_s = self.firecrawl_retry_backoff_sec * attempt
                    self.logger.warning(
                        "Firecrawl timeout attempt %s/%s for URL: %s. Retrying in %.1fs",
                        attempt,
                        self.firecrawl_max_retries,
                        url,
                        wait_s,
                    )
                    await asyncio.sleep(wait_s)
                    continue
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < self.firecrawl_max_retries:
                    wait_s = self.firecrawl_retry_backoff_sec * attempt
                    self.logger.warning(
                        "Firecrawl error attempt %s/%s for URL: %s | %s. Retrying in %.1fs",
                        attempt,
                        self.firecrawl_max_retries,
                        url,
                        exc,
                        wait_s,
                    )
                    await asyncio.sleep(wait_s)
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        return ""

    async def _rerank(self, question: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not docs:
            return []
        if not self.web_rerank_enabled:
            out = sorted(docs, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
            for row in out:
                row["rerank_score"] = None
            return out
        if CrossEncoder is None:
            raise RuntimeError("CrossEncoder is required for web reranking but is not installed.")
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(
                self.cross_encoder_model,
                local_files_only=self.cross_encoder_local_files_only,
            )
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
