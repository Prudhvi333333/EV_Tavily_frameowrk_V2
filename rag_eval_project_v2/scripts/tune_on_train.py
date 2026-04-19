from __future__ import annotations

import asyncio
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hyde import HyDEExpander
from src.indexer import build_or_load_index
from src.kb_loader import load_kb
from src.retriever import HybridRetriever
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


async def run() -> None:
    config = load_config(PROJECT_ROOT / "config/config.yaml")
    logger = get_logger("tune_on_train", config)
    train_df, _ = load_split(config)
    kb_docs = load_kb(config)
    index = build_or_load_index(config, kb_docs)

    top_k_values = [5, 8, 12]
    hyde_values = [True, False]
    chunk_values = ["row_only", "row_plus_window"]
    prompting_values = ["standard", "chain_of_thought", "few_shot"]

    results: list[dict] = []
    for top_k, hyde_enabled, chunk_strategy, prompting in itertools.product(
        top_k_values, hyde_values, chunk_values, prompting_values
    ):
        cfg = dict(config)
        cfg["retrieval"] = dict(config["retrieval"])
        cfg["retrieval"]["top_k"] = top_k
        cfg["retrieval"]["chunk_strategy"] = chunk_strategy
        cfg["retrieval"]["prompt_variant"] = prompting
        cfg["hyde"] = dict(config["hyde"])
        cfg["hyde"]["enabled"] = hyde_enabled

        retriever = HybridRetriever(index, cfg)
        hyde = HyDEExpander(cfg)
        precisions = []
        recalls = []

        for _, row in train_df.iterrows():
            q = str(row["Question"])
            golden = str(row["Human validated answers"])
            intent = retriever.detect_query_intent(q)
            expanded = await hyde.expand(q, intent)
            vector = hyde.get_search_vector(expanded)
            docs = retriever.retrieve_with_vector(vector, question_text=q, top_k=top_k)
            context = retriever.build_context(docs)
            p, r = _retrieval_precision_recall(golden, context)
            precisions.append(p)
            recalls.append(r)

        mean_precision = float(np.mean(precisions)) if precisions else 0.0
        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        mean_score = (mean_precision + mean_recall) / 2.0
        results.append(
            {
                "top_k": top_k,
                "hyde_enabled": hyde_enabled,
                "chunk_strategy": chunk_strategy,
                "prompting": prompting,
                "context_precision": round(mean_precision, 4),
                "context_recall": round(mean_recall, 4),
                "score": round(mean_score, 4),
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    logger.info(
        "Best config: top_k=%s, hyde=%s, chunks=%s, prompting=%s, score=%s",
        best["top_k"],
        best["hyde_enabled"],
        best["chunk_strategy"],
        best["prompting"],
        best["score"],
    )

    progress_dir = resolve_path(config, config["paths"]["progress_dir"])
    tuning_path = progress_dir / "tuning_results.json"
    with tuning_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    best_cfg = {
        "retrieval": {
            "top_k": best["top_k"],
            "chunk_strategy": best["chunk_strategy"],
            "prompt_variant": best["prompting"],
        },
        "hyde": {"enabled": bool(best["hyde_enabled"])},
    }
    best_cfg_path = PROJECT_ROOT / "config" / "best_config.yaml"
    with best_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)

    logger.info("Wrote tuning results to %s", tuning_path)
    logger.info("Wrote best config to %s", best_cfg_path)


if __name__ == "__main__":
    asyncio.run(run())

