from __future__ import annotations

import re
from typing import Any

import numpy as np

from src.generator import OllamaGenerator, OpenRouterGenerator


class ScoreValidator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        validator_cfg = config.get("evaluation", {}).get("validator", {})
        self.provider = str(validator_cfg.get("provider", "ollama")).lower()
        self.model = validator_cfg.get("model", "qwen2.5:14b")
        self.base_url = str(validator_cfg.get("base_url", "https://openrouter.ai/api/v1")).rstrip("/")
        self.api_key_env = str(validator_cfg.get("api_key_env", "OPENROUTER_API_KEY"))
        self.flag_threshold = float(validator_cfg.get("flag_threshold", 0.3))
        keep_alive = str(config.get("runtime", {}).get("ollama_keep_alive", "0s"))
        ollama_options = dict(config.get("runtime", {}).get("ollama_options", {}))
        if self.provider in {"openrouter", "kimi_cloud", "kimi"}:
            self.local_qwen = OpenRouterGenerator(
                self.model,
                api_key_env=self.api_key_env,
                base_url=self.base_url,
                strict=bool(config.get("runtime", {}).get("strict_mode", False)),
            )
        else:
            self.local_qwen = OllamaGenerator(
                self.model,
                strict=bool(config.get("runtime", {}).get("strict_mode", False)),
                keep_alive=keep_alive,
                options=ollama_options,
            )
        eval_cfg = config.get("evaluation", {})
        self.weight_sets = {
            "rag": eval_cfg.get("weights_rag", {}),
            "no_rag": eval_cfg.get("weights_norag", {}),
            "rag_pretrained": eval_cfg.get("weights_rag_pretrained", {}),
            "rag_pretrained_web": eval_cfg.get("weights_rag_pretrained_web", {}),
        }

    async def validate(
        self,
        metric: str,
        score: float,
        question: str,
        golden: str,
        answer: str,
    ) -> dict[str, Any]:
        prompt = (
            "You are a scoring validator.\n"
            f"A judge gave this answer a {metric} score of {score:.2f}.\n"
            f"Question: {question}\n"
            f"Golden Answer: {golden[:300]}\n"
            f"Generated Answer: {answer[:300]}\n"
            "Is this score reasonable? Reply with ONLY:\n"
            "VALID\n"
            "or\n"
            "FLAG:<brief reason>\n"
            f"Use FLAG only if the score seems wrong by more than {self.flag_threshold}."
        )
        response = await self.local_qwen.generate(prompt=prompt, system="Reply in required format only.", temperature=0.0)
        response = (response or "").strip()
        if response.upper().startswith("FLAG:"):
            reason = response[5:].strip() or "validator_flagged"
            rescored = self._heuristic_rescore(metric, question, golden, answer)
            adjusted = round((float(score) + float(rescored)) / 2.0, 4)
            return {
                "valid": False,
                "reason": reason,
                "original_score": float(score),
                "rescored": float(rescored),
                "adjusted_score": adjusted,
            }
        return {
            "valid": True,
            "reason": "ok",
            "original_score": float(score),
            "rescored": None,
            "adjusted_score": float(score),
        }

    async def validate_row(self, eval_row: dict[str, Any], pipeline: str) -> dict[str, Any]:
        metric_scores = dict(eval_row.get("metric_scores", {}))
        flags: list[str] = []
        reasons: list[str] = []
        adjustments: dict[str, dict[str, float]] = {}

        for metric, score in metric_scores.items():
            result = await self.validate(
                metric=metric,
                score=float(score),
                question=str(eval_row.get("question", "")),
                golden=str(eval_row.get("golden", "")),
                answer=str(eval_row.get("answer", "")),
            )
            if not result["valid"]:
                flags.append(metric)
                reasons.append(f"{metric}: {result['reason']}")
                metric_scores[metric] = result["adjusted_score"]
                adjustments[metric] = {
                    "original": result["original_score"],
                    "rescored": result["rescored"],
                    "adjusted": result["adjusted_score"],
                }

        eval_row["metric_scores"] = metric_scores
        eval_row["validation_flags"] = "FLAGGED" if flags else ""
        eval_row["validation_reason"] = "; ".join(reasons)
        eval_row["flagged_metrics"] = ",".join(flags)
        eval_row["metric_adjustments"] = adjustments
        eval_row["final_score"] = self._compute_final_score(metric_scores, pipeline)
        for metric, value in metric_scores.items():
            eval_row[metric] = value
        return eval_row

    async def validate_all(self, rows: list[dict[str, Any]], pipeline: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(await self.validate_row(row, pipeline))
        return out

    def _compute_final_score(self, scores: dict[str, float], pipeline: str) -> float:
        weights = self.weight_sets.get(pipeline, {})
        if not weights:
            return float(np.mean(list(scores.values()))) if scores else 0.0
        return round(
            sum(float(scores.get(metric, 0.0)) * float(weight) for metric, weight in weights.items()),
            4,
        )

    def _heuristic_rescore(self, metric: str, question: str, golden: str, answer: str) -> float:
        if metric == "answer_relevancy":
            return self._overlap(question, answer, "recall")
        if metric == "answer_correctness":
            return self._overlap(golden, answer, "f1")
        if metric in {"faithfulness", "context_precision"}:
            return self._overlap(answer, golden, "precision")
        if metric == "context_recall":
            return self._overlap(golden, answer, "recall")
        if metric == "source_attribution":
            lines = [x.strip() for x in answer.splitlines() if x.strip()]
            if not lines:
                return 0.0
            tags = tuple(
                t.casefold()
                for t in ("[KB]", "[PRETRAINED]", "[WEB]", "kb", "context", "web", "pretrained", "general knowledge")
            )
            tagged = sum(1 for x in lines if any(t in x.casefold() for t in tags))
            return round(tagged / len(lines), 4)
        return 0.5

    def _overlap(self, left: str, right: str, mode: str) -> float:
        lt = set(re.findall(r"[a-z0-9]+", left.lower()))
        rt = set(re.findall(r"[a-z0-9]+", right.lower()))
        if not lt or not rt:
            return 0.0
        inter = len(lt & rt)
        p = inter / max(len(rt), 1)
        r = inter / max(len(lt), 1)
        if mode == "precision":
            return round(p, 4)
        if mode == "recall":
            return round(r, 4)
        if p + r == 0:
            return 0.0
        return round((2 * p * r) / (p + r), 4)
