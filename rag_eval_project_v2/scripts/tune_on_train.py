from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import re
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hyde import HyDEExpander
from src.indexer import build_or_load_index
from src.kb_loader import load_kb
from src.retriever import HybridRetriever, RetrievedDoc
from src.splitter import load_split
from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _retrieval_precision_recall(golden: str, context: str) -> tuple[float, float]:
    g = _token_set(golden)
    c = _token_set(context)
    if not g or not c:
        return 0.0, 0.0
    inter = len(g & c)
    precision = inter / len(c)
    recall = inter / len(g)
    return precision, recall


def _tier_alignment(intent: dict[str, Any], docs: list[RetrievedDoc]) -> float:
    if not docs:
        return 0.0
    tiers = [str(t).lower() for t in intent.get("detected_tiers", [])]
    if not tiers:
        return 1.0
    matched = 0
    for doc in docs:
        category = str((doc.metadata or {}).get("category", "")).lower()
        if any(t in category for t in tiers):
            matched += 1
    return matched / len(docs)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted retrieval fix pass tuner (train split only).")
    parser.add_argument(
        "--profile",
        choices=["quick", "full"],
        default="quick",
        help="Grid size profile. quick is recommended for regular use.",
    )
    parser.add_argument(
        "--max-train-questions",
        type=int,
        default=0,
        help="Optional cap for train questions during tuning (0 = all).",
    )
    parser.add_argument(
        "--recall-weight",
        type=float,
        default=0.70,
        help="Objective weight for context recall (0-1).",
    )
    parser.add_argument(
        "--tier-alignment-weight",
        type=float,
        default=0.15,
        help="Extra objective weight for Tier alignment (>=0).",
    )
    parser.add_argument(
        "--write-best-config",
        action="store_true",
        help="Write best retrieval/reranker params to config/best_config.yaml.",
    )
    parser.add_argument(
        "--no-write-best-config",
        action="store_true",
        help="Do not write best_config.yaml (analysis only).",
    )
    return parser.parse_args()


def _search_space(profile: str) -> dict[str, list[Any]]:
    if profile == "full":
        return {
            "top_k": [8, 10, 12],
            "hyde_enabled": [False, True],
            "weight_pair": [(0.65, 0.35), (0.70, 0.30), (0.75, 0.25)],
            "adaptive_top_k": [True, False],
            "tier_match_boost": [0.14, 0.20],
            "tier_mismatch_penalty": [-0.10, -0.16],
            "location_mismatch_penalty": [-0.02],
        }

    return {
        "top_k": [8, 10],
        "hyde_enabled": [False, True],
        "weight_pair": [(0.70, 0.30), (0.75, 0.25)],
        "adaptive_top_k": [True, False],
        "tier_match_boost": [0.14, 0.20],
        "tier_mismatch_penalty": [-0.10, -0.16],
        "location_mismatch_penalty": [-0.02],
    }


def _should_write_best_config(args: argparse.Namespace) -> bool:
    if args.no_write_best_config:
        return False
    if args.write_best_config:
        return True
    return True


def _objective(
    mean_precision: float,
    mean_recall: float,
    mean_tier_alignment: float,
    recall_weight: float,
    tier_alignment_weight: float,
) -> float:
    rw = max(0.0, min(1.0, float(recall_weight)))
    pw = 1.0 - rw
    tw = max(0.0, float(tier_alignment_weight))
    raw = rw * mean_recall + pw * mean_precision + tw * mean_tier_alignment
    denom = 1.0 + tw
    return raw / denom


def _preflight_reranker_model(config: dict[str, Any]) -> None:
    rer_cfg = dict(config.get("reranker", {}))
    model_name = str(rer_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L6-v2")).strip()
    local_files_only = bool(
        rer_cfg.get(
            "local_files_only",
            config.get("embeddings", {}).get("local_files_only", False),
        )
    )
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise RuntimeError("sentence-transformers is required for retrieval tuning with reranker.") from exc

    try:
        CrossEncoder(model_name, local_files_only=local_files_only)
    except Exception as exc:
        mode_hint = "cached locally" if local_files_only else "downloadable from HuggingFace"
        raise RuntimeError(
            f"Reranker preflight failed for '{model_name}'. Ensure this model is {mode_hint}. "
            "Warmup command: .\\.venv\\Scripts\\python -c \"from sentence_transformers import CrossEncoder; "
            f"CrossEncoder('{model_name}', local_files_only=False); print('CE_OK')\""
        ) from exc


async def run() -> None:
    args = _parse_args()
    config = load_config(PROJECT_ROOT / "config/config.yaml")
    logger = get_logger("tune_on_train", config)
    _preflight_reranker_model(config)
    train_df, _ = load_split(config)
    if args.max_train_questions > 0:
        train_df = train_df.head(args.max_train_questions).copy()
        logger.info("Using first %s train questions for tuning.", len(train_df))

    kb_docs = load_kb(config)
    index = build_or_load_index(config, kb_docs)
    space = _search_space(args.profile)
    keys = list(space.keys())
    all_combos = list(itertools.product(*[space[k] for k in keys]))
    logger.info("Tuning profile=%s with %s candidate configs.", args.profile, len(all_combos))

    results: list[dict[str, Any]] = []
    for idx, combo in enumerate(all_combos, start=1):
        candidate = dict(zip(keys, combo))
        semantic_weight, bm25_weight = candidate["weight_pair"]

        cfg = deepcopy(config)
        cfg["retrieval"] = dict(cfg.get("retrieval", {}))
        cfg["retrieval"]["top_k"] = int(candidate["top_k"])
        cfg["retrieval"]["semantic_weight"] = float(semantic_weight)
        cfg["retrieval"]["bm25_weight"] = float(bm25_weight)
        cfg["retrieval"]["adaptive_top_k"] = bool(candidate["adaptive_top_k"])
        md = dict(cfg["retrieval"].get("metadata_scoring", {}))
        md["tier_match_boost"] = float(candidate["tier_match_boost"])
        md["tier_mismatch_penalty"] = float(candidate["tier_mismatch_penalty"])
        md["location_mismatch_penalty"] = float(candidate["location_mismatch_penalty"])
        cfg["retrieval"]["metadata_scoring"] = md

        cfg["hyde"] = dict(cfg.get("hyde", {}))
        cfg["hyde"]["enabled"] = bool(candidate["hyde_enabled"])

        cfg["reranker"] = dict(cfg.get("reranker", {}))
        cfg["reranker"]["enabled"] = True

        retriever = HybridRetriever(index, cfg)
        hyde = HyDEExpander(cfg)

        precisions: list[float] = []
        recalls: list[float] = []
        tier_scores: list[float] = []

        for _, row in train_df.iterrows():
            question = str(row["Question"])
            golden = str(row["Human validated answers"])
            intent = retriever.detect_query_intent(question)
            expanded = await hyde.expand(question, intent)
            vector = hyde.get_search_vector(expanded)
            docs = retriever.retrieve_with_vector(
                vector,
                question_text=question,
                top_k=int(candidate["top_k"]),
                intent=intent,
            )
            context = retriever.build_context(docs)
            precision, recall = _retrieval_precision_recall(golden, context)
            precisions.append(precision)
            recalls.append(recall)
            tier_scores.append(_tier_alignment(intent, docs))

        mean_precision = float(np.mean(precisions)) if precisions else 0.0
        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        mean_tier_alignment = float(np.mean(tier_scores)) if tier_scores else 0.0
        score = _objective(
            mean_precision=mean_precision,
            mean_recall=mean_recall,
            mean_tier_alignment=mean_tier_alignment,
            recall_weight=args.recall_weight,
            tier_alignment_weight=args.tier_alignment_weight,
        )
        rec = {
            "top_k": int(candidate["top_k"]),
            "hyde_enabled": bool(candidate["hyde_enabled"]),
            "semantic_weight": round(float(semantic_weight), 4),
            "bm25_weight": round(float(bm25_weight), 4),
            "adaptive_top_k": bool(candidate["adaptive_top_k"]),
            "tier_match_boost": round(float(candidate["tier_match_boost"]), 4),
            "tier_mismatch_penalty": round(float(candidate["tier_mismatch_penalty"]), 4),
            "location_mismatch_penalty": round(float(candidate["location_mismatch_penalty"]), 4),
            "context_precision": round(mean_precision, 4),
            "context_recall": round(mean_recall, 4),
            "tier_alignment": round(mean_tier_alignment, 4),
            "objective_score": round(float(score), 4),
        }
        results.append(rec)
        logger.info(
            "[%s/%s] objective=%.4f recall=%.4f precision=%.4f tier=%.4f | top_k=%s hyde=%s sem=%.2f bm25=%.2f blend=%.2f max_cand=%s",
            idx,
            len(all_combos),
            score,
            mean_recall,
            mean_precision,
            mean_tier_alignment,
            rec["top_k"],
            rec["hyde_enabled"],
            rec["semantic_weight"],
            rec["bm25_weight"],
            cfg["reranker"].get("blend_weight", 0.65),
            cfg["reranker"].get("max_candidates", 18),
        )

    results.sort(key=lambda x: x["objective_score"], reverse=True)
    best = results[0]
    logger.info("Best retrieval config on train objective: %s", best)

    progress_dir = resolve_path(config, config["paths"]["progress_dir"])
    tuning_path = progress_dir / "tuning_results.json"
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "profile": args.profile,
        "train_questions_used": int(len(train_df)),
        "objective": {
            "recall_weight": float(args.recall_weight),
            "precision_weight": round(1.0 - float(args.recall_weight), 4),
            "tier_alignment_weight": float(args.tier_alignment_weight),
        },
        "best": best,
        "top10": results[:10],
        "all": results,
    }
    with tuning_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote tuning results to %s", tuning_path)

    if _should_write_best_config(args):
        best_cfg = {
            "retrieval": {
                "top_k": int(best["top_k"]),
                "semantic_weight": float(best["semantic_weight"]),
                "bm25_weight": float(best["bm25_weight"]),
                "adaptive_top_k": bool(best["adaptive_top_k"]),
                "metadata_scoring": {
                    "tier_match_boost": float(best["tier_match_boost"]),
                    "tier_mismatch_penalty": float(best["tier_mismatch_penalty"]),
                    "location_mismatch_penalty": float(best["location_mismatch_penalty"]),
                },
            },
            "hyde": {"enabled": bool(best["hyde_enabled"])},
            "reranker": {
                "enabled": True,
                "blend_weight": float(config.get("reranker", {}).get("blend_weight", 0.65)),
                "max_candidates": int(config.get("reranker", {}).get("max_candidates", 18)),
            },
        }
        best_cfg_path = PROJECT_ROOT / "config" / "best_config.yaml"
        with best_cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(best_cfg, f, sort_keys=False)
        logger.info("Wrote best config to %s", best_cfg_path)
    else:
        logger.info("Skipped writing config/best_config.yaml (--no-write-best-config).")


if __name__ == "__main__":
    asyncio.run(run())
