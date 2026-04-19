from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.evaluator import RAGASEvaluator
from src.few_shot_builder import FewShotBuilder
from src.generator import ModelGenerator, PipelineMode
from src.hyde import HyDEExpander
from src.indexer import build_or_load_index
from src.kb_loader import load_kb
from src.reporter import build_comparison_report, build_report
from src.retriever import HybridRetriever
from src.score_validator import ScoreValidator
from src.splitter import load_split
from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger
from src.web_crawler import WebCrawler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG Evaluation Framework v2 runner")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model keys to run")
    parser.add_argument("--pipelines", nargs="*", default=None, help="Subset of pipelines to run")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit of test questions")
    parser.add_argument(
        "--eval-split",
        choices=["test", "train", "both"],
        default="test",
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--use-kimi-cloud-judge",
        action="store_true",
        help="Use OpenRouter Kimi Cloud as evaluator + web validator judge.",
    )
    parser.add_argument("--judge-provider", default=None, help="Override evaluation judge provider.")
    parser.add_argument("--judge-model", default=None, help="Override evaluation judge model.")
    parser.add_argument("--web-judge-provider", default=None, help="Override web validator judge provider.")
    parser.add_argument("--web-judge-model", default=None, help="Override web validator judge model.")
    return parser.parse_args()


def _pipeline_list(config: dict[str, Any], requested: list[str] | None) -> list[PipelineMode]:
    available = [PipelineMode(x) for x in config["pipelines"]]
    if not requested:
        return available
    requested_set = {x.strip().lower() for x in requested}
    return [p for p in available if p.value in requested_set]


def _model_list(config: dict[str, Any], requested: list[str] | None) -> list[str]:
    available = list(config["models"].keys())
    if not requested:
        defaults = config.get("run", {}).get("default_models")
        if defaults:
            selected = [m for m in defaults if m in available]
            if selected:
                return selected
        return available
    req = [m for m in requested if m in available]
    return req or available


async def run_single_pipeline(
    model_key: str,
    pipeline_mode: PipelineMode,
    config: dict[str, Any],
    retriever: HybridRetriever,
    crawler: WebCrawler,
    questions_df: pd.DataFrame,
    few_shot_builder: FewShotBuilder,
) -> list[dict[str, Any]]:
    generator = ModelGenerator(model_key, config, few_shot_builder=few_shot_builder)
    hyde = HyDEExpander(config)
    results: list[dict[str, Any]] = []

    for _, row in questions_df.iterrows():
        q = str(row["Question"])
        golden = str(row["Human validated answers"])
        q_id = row.get("Num", "")
        category = str(row.get("Use Case Category", ""))
        intent = retriever.detect_query_intent(q)

        kb_context = ""
        if pipeline_mode in {
            PipelineMode.RAG,
            PipelineMode.RAG_PRETRAINED,
            PipelineMode.RAG_PRETRAINED_WEB,
        }:
            search_text = await hyde.expand(q, intent)
            vector = hyde.get_search_vector(search_text)
            docs = retriever.retrieve_with_vector(vector, question_text=q, top_k=config["retrieval"]["top_k"])
            kb_context = retriever.build_context(docs)
        else:
            docs = []

        web_context = ""
        web_status = "NOT_USED"
        web_docs: list[dict[str, Any]] = []
        web_validation_records: list[dict[str, Any]] = []
        web_search_query = ""
        web_accepted = 0
        web_low_confidence = 0
        web_rejected = 0
        web_timed_out = False
        if pipeline_mode == PipelineMode.RAG_PRETRAINED_WEB:
            try:
                q_id_label = f"Q_{int(float(q_id)):03d}"
            except Exception:
                q_id_label = str(q_id)
            crawl_payload = await crawler.crawl(
                question=q,
                question_id=q_id_label,
                pipeline=f"{model_key}_{pipeline_mode.value}",
            )
            web_docs = list(crawl_payload.get("docs", []))
            web_validation_records = list(crawl_payload.get("records", []))
            web_search_query = str(crawl_payload.get("search_query", ""))
            web_timed_out = bool(crawl_payload.get("timed_out", False))
            web_accepted = sum(1 for r in web_validation_records if r.get("accepted"))
            web_low_confidence = sum(1 for r in web_validation_records if r.get("low_confidence"))
            web_rejected = sum(1 for r in web_validation_records if not r.get("accepted"))
            if web_docs:
                web_context = "\n\n".join([str(d.get("context_block", "")).strip() for d in web_docs if str(d.get("context_block", "")).strip()])
                web_status = "PARTIAL_TIMEOUT_OK" if web_timed_out else "OK"
            elif web_validation_records:
                web_status = "PARTIAL_TIMEOUT_NO_ACCEPTED" if web_timed_out else "REJECTED_ALL"
            else:
                web_status = "WEB_TIMEOUT" if web_timed_out else "WEB_UNAVAILABLE"

        answer = await generator.generate_with_mode(
            question=q,
            pipeline_mode=pipeline_mode,
            kb_context=kb_context,
            web_context=web_context,
        )
        results.append(
            {
                "q_id": q_id,
                "category": category,
                "question": q,
                "golden": golden,
                "answer": answer,
                "kb_context": kb_context,
                "web_context": web_context,
                "web_status": web_status,
                "web_search_query": web_search_query,
                "web_timed_out": web_timed_out,
                "web_docs_selected": web_docs,
                "web_validation_records": web_validation_records,
                "web_accepted_count": web_accepted,
                "web_low_confidence_count": web_low_confidence,
                "web_rejected_count": web_rejected,
                "retrieved_docs": [d.id for d in docs],
                "model_key": model_key,
                "pipeline_mode": pipeline_mode.value,
            }
        )
    return results


async def main() -> None:
    run_started_utc = datetime.now(timezone.utc)
    args = _parse_args()
    load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=False)
    config = load_config(args.config)
    logger = get_logger("main", config)

    if args.use_kimi_cloud_judge:
        config.setdefault("evaluation", {}).setdefault("judge", {})
        config["evaluation"]["judge"]["provider"] = "openrouter"
        config["evaluation"]["judge"]["model"] = "kimi-k2.5:cloud"
        config.setdefault("web_validator", {}).setdefault("judge", {})
        config["web_validator"]["judge"]["provider"] = "openrouter"
        config["web_validator"]["judge"]["model"] = "kimi-k2.5:cloud"
    if args.judge_provider:
        config.setdefault("evaluation", {}).setdefault("judge", {})
        config["evaluation"]["judge"]["provider"] = args.judge_provider
    if args.judge_model:
        config.setdefault("evaluation", {}).setdefault("judge", {})
        config["evaluation"]["judge"]["model"] = args.judge_model
    if args.web_judge_provider:
        config.setdefault("web_validator", {}).setdefault("judge", {})
        config["web_validator"]["judge"]["provider"] = args.web_judge_provider
    if args.web_judge_model:
        config.setdefault("web_validator", {}).setdefault("judge", {})
        config["web_validator"]["judge"]["model"] = args.web_judge_model
    logger.info(
        "Judge config | evaluation: %s/%s | web_validator: %s/%s",
        config.get("evaluation", {}).get("judge", {}).get("provider", "ollama"),
        config.get("evaluation", {}).get("judge", {}).get("model", "qwen2.5:14b"),
        config.get("web_validator", {}).get("judge", {}).get("provider", "ollama"),
        config.get("web_validator", {}).get("judge", {}).get("model", config.get("hyde", {}).get("model", "qwen2.5:14b")),
    )

    train_df, test_df = load_split(config)
    split_frames: dict[str, pd.DataFrame] = {"train": train_df.copy(), "test": test_df.copy()}
    if args.eval_split == "both":
        selected_splits = ["train", "test"]
    else:
        selected_splits = [args.eval_split]
    if args.limit:
        for split_name in selected_splits:
            split_frames[split_name] = split_frames[split_name].head(args.limit).copy()
        logger.info("Applying --limit=%s for quick run.", args.limit)

    kb_docs = load_kb(config)
    index = build_or_load_index(config, kb_docs)
    retriever = HybridRetriever(index, config)
    crawler = WebCrawler(config, kb_collection=index.collection, embedding_model=index.embedder)
    evaluator = RAGASEvaluator(config)
    validator = ScoreValidator(config)
    few_shot_builder = FewShotBuilder(train_df, config)

    selected_models = _model_list(config, args.models)
    selected_pipelines = _pipeline_list(config, args.pipelines)

    all_results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    progress: list[dict[str, Any]] = []

    for split_name in selected_splits:
        questions_df = split_frames[split_name]
        for model_key in selected_models:
            for pipeline_mode in selected_pipelines:
                logger.info("Running split=%s model=%s pipeline=%s", split_name, model_key, pipeline_mode.value)
                generated = await run_single_pipeline(
                    model_key=model_key,
                    pipeline_mode=pipeline_mode,
                    config=config,
                    retriever=retriever,
                    crawler=crawler,
                    questions_df=questions_df,
                    few_shot_builder=few_shot_builder,
                )
                evaluated = await evaluator.evaluate_all(generated, pipeline_mode.value)
                validated = await validator.validate_all(evaluated, pipeline_mode.value)
                report_model_name = model_key if split_name == "test" else f"{model_key}_{split_name}"
                report_path = build_report(validated, report_model_name, pipeline_mode.value, config)
                all_results[(report_model_name, pipeline_mode.value)] = validated

                run_score = round(sum(r.get("final_score", 0.0) for r in validated) / max(len(validated), 1), 4)
                progress.append(
                    {
                        "split": split_name,
                        "model": model_key,
                        "pipeline": pipeline_mode.value,
                        "questions": len(validated),
                        "mean_final_score": run_score,
                        "report": report_path,
                    }
                )
                logger.info(
                    "Completed split=%s model=%s pipeline=%s mean_final_score=%.4f report=%s",
                    split_name,
                    model_key,
                    pipeline_mode.value,
                    run_score,
                    report_path,
                )

    comparison_path = build_comparison_report(all_results, config)
    logger.info("Comparison report: %s", comparison_path)
    reviewer_ui_cmd = r".\.venv\Scripts\streamlit run ui\reviewer_app.py"

    progress_dir = resolve_path(config, config["paths"]["progress_dir"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_path = Path(progress_dir) / f"run_progress_{ts}.json"
    with progress_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "runs": progress,
                "comparison_report": str(comparison_path),
                "reviewer_ui_command": reviewer_ui_cmd,
                "run_started_utc": run_started_utc.isoformat().replace("+00:00", "Z"),
            },
            f,
            indent=2,
        )
    logger.info("Progress log written: %s", progress_path)


if __name__ == "__main__":
    asyncio.run(main())
