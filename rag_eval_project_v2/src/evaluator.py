from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
import numpy as np

from src.utils.ollama import resolve_ollama_base_url

METRIC_SETS: dict[str, list[str]] = {
    "rag": [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ],
    "no_rag": [
        "answer_relevancy",
        "answer_correctness",
    ],
    "rag_pretrained": [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "source_attribution",
    ],
    "rag_pretrained_web": [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "source_attribution",
        "web_grounding",
    ],
}


class RAGASEvaluator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        eval_cfg = config.get("evaluation", {})
        judge_cfg = eval_cfg.get("judge", {})
        provider = str(judge_cfg.get("provider", "openrouter")).lower()
        self.judge_provider = "openrouter" if provider in {"openrouter", "kimi_cloud", "kimi"} else provider
        self.judge_model = judge_cfg.get("model", "moonshotai/kimi-k2")
        self.ollama_base_url = resolve_ollama_base_url(judge_cfg.get("ollama_base_url"))
        self.judge_base_url = judge_cfg.get("base_url", "https://openrouter.ai/api/v1").rstrip("/")
        self.judge_api_key = os.getenv(judge_cfg.get("api_key_env", "OPENROUTER_API_KEY"), "").strip()
        self.strict_mode = bool(config.get("runtime", {}).get("strict_mode", False))
        self.ollama_keep_alive = str(config.get("runtime", {}).get("ollama_keep_alive", "0s"))
        self.ollama_options = dict(config.get("runtime", {}).get("ollama_options", {}))
        self.allow_heuristic_fallback = bool(eval_cfg.get("allow_heuristic_fallback", True))
        self._last_judge_debug = ""
        self._judge_checked = False
        self._judge_available = False
        self.weight_sets = {
            "rag": eval_cfg.get("weights_rag", {}),
            "no_rag": eval_cfg.get("weights_norag", {}),
            "rag_pretrained": eval_cfg.get("weights_rag_pretrained", {}),
            "rag_pretrained_web": eval_cfg.get("weights_rag_pretrained_web", {}),
        }

    async def evaluate_all(self, rows: list[dict[str, Any]], pipeline: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(await self.evaluate_row(row, pipeline))
        return out

    async def evaluate_row(self, row: dict[str, Any], pipeline: str) -> dict[str, Any]:
        metrics = METRIC_SETS[pipeline]
        judge_scores = await self._judge_scores(row, pipeline, metrics)
        final_score = self.compute_final_score(judge_scores, pipeline)

        row["metric_scores"] = judge_scores
        row["final_score"] = final_score
        row["pipeline_mode"] = pipeline
        for metric in metrics:
            row[metric] = judge_scores.get(metric, np.nan)
        return row

    def compute_final_score(self, scores: dict[str, float], pipeline: str) -> float:
        weights = self.weight_sets.get(pipeline, {})
        if not weights:
            return float(np.mean(list(scores.values()))) if scores else 0.0
        total = 0.0
        for metric, weight in weights.items():
            if metric in scores:
                total += float(scores[metric]) * float(weight)
        return round(total, 4)

    async def _judge_scores(
        self,
        row: dict[str, Any],
        pipeline: str,
        metrics: list[str],
    ) -> dict[str, float]:
        if await self._is_judge_available():
            if self.judge_provider == "ollama":
                judged = await self._judge_with_ollama(row, pipeline, metrics)
            else:
                judged = await self._judge_with_openrouter(row, pipeline, metrics)
            if judged is not None:
                return judged
            debug_hint = f" Last judge output/error: {self._last_judge_debug}" if self._last_judge_debug else ""
        else:
            debug_hint = f" Last judge output/error: {self._last_judge_debug}" if self._last_judge_debug else ""
        if self.strict_mode and not self.allow_heuristic_fallback:
            raise RuntimeError(
                "Judge is unavailable or returned invalid output, and heuristic fallback is disabled."
                + debug_hint
            )
        return self._heuristic_scores(row, pipeline, metrics)

    async def _is_judge_available(self) -> bool:
        if self._judge_checked:
            return self._judge_available
        self._judge_checked = True
        try:
            if self.judge_provider == "ollama":
                timeout = httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0)
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.get(f"{self.ollama_base_url}/api/tags")
                    resp.raise_for_status()
                    names = {m.get("name", "") for m in resp.json().get("models", [])}
                    expected = {self.judge_model, f"{self.judge_model}:latest"}
                    self._judge_available = bool(names & expected)
                if not self._judge_available and self.strict_mode and not self.allow_heuristic_fallback:
                    raise RuntimeError(
                        f"Ollama judge model '{self.judge_model}' is missing. Run: ollama pull {self.judge_model}"
                    )
                return self._judge_available

            if not self.judge_api_key:
                self._judge_available = False
                if self.strict_mode and not self.allow_heuristic_fallback:
                    raise RuntimeError("OPENROUTER_API_KEY is not set for strict judge mode.")
                return False

            url = f"{self.judge_base_url}/models"
            headers = {"Authorization": f"Bearer {self.judge_api_key}"}
            timeout = httpx.Timeout(connect=3.0, read=3.0, write=3.0, pool=3.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code >= 400:
                    self._judge_available = False
                    return False
                model_ids = {m.get("id", "") for m in resp.json().get("data", [])}
                self._judge_available = self.judge_model in model_ids or not model_ids
                if not self._judge_available and self.strict_mode and not self.allow_heuristic_fallback:
                    raise RuntimeError(f"Judge model '{self.judge_model}' is not available on OpenRouter.")
        except Exception as e:
            self._judge_available = False
            if self.strict_mode and not self.allow_heuristic_fallback:
                raise RuntimeError("Judge endpoint is unreachable in strict mode.") from e
        return self._judge_available

    async def _judge_with_ollama(
        self,
        row: dict[str, Any],
        pipeline: str,
        metrics: list[str],
    ) -> dict[str, float] | None:
        prompt = (
            "You are an evaluation judge for RAG responses.\n"
            "Return ONLY valid JSON with metric names as keys and 0-1 floats as values.\n"
            "No explanation text.\n\n"
            f"Pipeline: {pipeline}\n"
            f"Metrics: {metrics}\n"
            f"Question: {row.get('question', '')}\n"
            f"Golden Answer: {row.get('golden', '')}\n"
            f"Generated Answer: {row.get('answer', '')}\n"
            f"KB Context: {row.get('kb_context', '')[:1800]}\n"
            f"Web Context: {row.get('web_context', '')[:1800]}\n"
        )
        payload = {
            "model": self.judge_model,
            "system": "Return only JSON.",
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {**self.ollama_options, "temperature": 0},
        }
        if self.ollama_keep_alive:
            payload["keep_alive"] = self.ollama_keep_alive
        try:
            timeout = httpx.Timeout(connect=5.0, read=120.0, write=20.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{self.ollama_base_url}/api/generate", json=payload)
                resp.raise_for_status()
            content = str(resp.json().get("response", "")).strip()
            self._last_judge_debug = content[:300]
            obj = self._parse_json_object(content)
            if not obj:
                return None
            cleaned: dict[str, float] = {}
            for metric in metrics:
                val = float(obj.get(metric, 0.0))
                cleaned[metric] = max(0.0, min(1.0, val))
            return cleaned
        except Exception as exc:
            self._last_judge_debug = f"ollama_error:{type(exc).__name__}:{exc}"
            return None

    async def _judge_with_openrouter(
        self,
        row: dict[str, Any],
        pipeline: str,
        metrics: list[str],
    ) -> dict[str, float] | None:
        prompt = (
            "You are an evaluation judge for RAG responses.\n"
            "Return ONLY a compact JSON object where keys are metric names and values are numbers 0.0-1.0.\n"
            "Do not include explanation text.\n\n"
            f"Pipeline: {pipeline}\n"
            f"Metrics: {metrics}\n"
            f"Question: {row.get('question', '')}\n"
            f"Golden Answer: {row.get('golden', '')}\n"
            f"Generated Answer: {row.get('answer', '')}\n"
            f"KB Context: {row.get('kb_context', '')[:1800]}\n"
            f"Web Context: {row.get('web_context', '')[:1800]}\n"
        )
        url = f"{self.judge_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.judge_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.judge_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
            content = (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            self._last_judge_debug = content[:300]
            obj = self._parse_json_object(content)
            if not obj:
                return None
            cleaned: dict[str, float] = {}
            for metric in metrics:
                val = float(obj.get(metric, 0.0))
                cleaned[metric] = max(0.0, min(1.0, val))
            return cleaned
        except Exception as exc:
            self._last_judge_debug = f"openrouter_error:{type(exc).__name__}:{exc}"
            return None

    def _parse_json_object(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        text = text.strip()
        if text.startswith("```"):
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except Exception:
            return None

    def _heuristic_scores(
        self,
        row: dict[str, Any],
        pipeline: str,
        metrics: list[str],
    ) -> dict[str, float]:
        q = str(row.get("question", ""))
        g = str(row.get("golden", ""))
        a = str(row.get("answer", ""))
        kb = str(row.get("kb_context", ""))
        web = str(row.get("web_context", ""))

        scores: dict[str, float] = {}
        for metric in metrics:
            if metric == "answer_relevancy":
                scores[metric] = self._overlap(q, a, mode="recall")
            elif metric == "answer_correctness":
                scores[metric] = self._overlap(g, a, mode="f1")
            elif metric == "faithfulness":
                scores[metric] = self._overlap(a, kb, mode="precision")
            elif metric == "context_precision":
                scores[metric] = self._overlap(a, kb, mode="precision")
            elif metric == "context_recall":
                scores[metric] = self._overlap(g, kb, mode="recall")
            elif metric == "source_attribution":
                scores[metric] = self._source_tag_score(a, pipeline)
            elif metric == "web_grounding":
                web_overlap = self._overlap(a, web, mode="precision")
                has_web_tag = 1.0 if "[WEB]" in a else 0.0
                scores[metric] = round(0.5 * web_overlap + 0.5 * has_web_tag, 4)
            else:
                scores[metric] = 0.0
        return scores

    def _source_tag_score(self, answer: str, pipeline: str) -> float:
        lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        if not lines:
            return 0.0
        if pipeline == "rag_pretrained":
            tags = ("[KB]", "[PRETRAINED]", "kb", "context", "pretrained", "general knowledge")
        else:
            tags = ("[KB]", "[PRETRAINED]", "[WEB]", "kb", "context", "web", "pretrained", "general knowledge")
        tags_norm = tuple(t.casefold() for t in tags)
        tagged = sum(1 for ln in lines if any(t in ln.casefold() for t in tags_norm))
        return round(tagged / len(lines), 4)

    def _overlap(self, left: str, right: str, mode: str = "f1") -> float:
        lt = self._tokens(left)
        rt = self._tokens(right)
        if not lt or not rt:
            return 0.0
        inter = len(lt & rt)
        prec = inter / max(len(rt), 1)
        rec = inter / max(len(lt), 1)
        if mode == "precision":
            return round(prec, 4)
        if mode == "recall":
            return round(rec, 4)
        if prec + rec == 0:
            return 0.0
        return round((2 * prec * rec) / (prec + rec), 4)

    def _tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))
