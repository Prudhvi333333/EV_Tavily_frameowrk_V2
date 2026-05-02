from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from src.answer_verifier import AnswerVerifier
from src.evaluator import RAGASEvaluator
from src.few_shot_builder import FewShotBuilder
from src.generator import ModelGenerator, OllamaGenerator, PipelineMode, load_generation_config
from src.hyde import HyDEExpander
from src.indexer import build_or_load_index
from src.kb_loader import load_kb
from src.reporter import build_comparison_report, build_report
from src.retriever import HybridRetriever
from src.score_validator import ScoreValidator
from src.splitter import load_split
from src.structured_query_router import StructuredQueryRouter
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
    parser.add_argument(
        "--retrieval-backend",
        choices=["hybrid", "llamaindex"],
        default=None,
        help="Override retrieval backend.",
    )
    parser.add_argument(
        "--enable-reranker",
        action="store_true",
        help="Enable cross-encoder reranker for KB retrieval.",
    )
    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Disable reranker even if config enables it.",
    )
    parser.add_argument(
        "--enable-web-reranker",
        action="store_true",
        help="Enable web document reranking.",
    )
    parser.add_argument(
        "--disable-web-reranker",
        action="store_true",
        help="Disable web document reranking.",
    )
    parser.add_argument(
        "--extra-test-questions",
        default=None,
        help="Optional path (xlsx/csv) of extra test questions to append at runtime.",
    )
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


def _probe_cross_encoder_model(model_name: str, local_files_only: bool, label: str) -> None:
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise RuntimeError("sentence-transformers is required when rerankers are enabled.") from exc

    try:
        CrossEncoder(model_name, local_files_only=local_files_only)
    except Exception as exc:
        mode_hint = "cached locally" if local_files_only else "downloadable from HuggingFace"
        raise RuntimeError(
            f"Cross-encoder probe failed for {label} model '{model_name}'. "
            f"Ensure the model is {mode_hint}."
        ) from exc


async def _probe_ollama_model(
    config: dict[str, Any],
    model_name: str,
    label: str,
    base_url: str | None = None,
    api_key_env: str = "OLLAMA_API_KEY",
) -> None:
    runtime_cfg = config.get("runtime", {})
    gen_cfg = load_generation_config(config)
    probe = OllamaGenerator(
        model_name,
        base_url=base_url,
        api_key_env=api_key_env,
        strict=True,
        timeout=gen_cfg["timeout"],
        max_retries=gen_cfg["max_retries"],
        retry_backoff_sec=gen_cfg["retry_backoff_sec"],
        keep_alive=str(runtime_cfg.get("ollama_keep_alive", "0s")),
        options=dict(runtime_cfg.get("ollama_options", {})),
    )
    out = await probe.generate(
        prompt="Reply only: MODEL_OK",
        system="Return the token only.",
        temperature=0.0,
    )
    text = str(out or "").strip()
    if not text:
        raise RuntimeError(f"Ollama probe returned unexpected output for {label} model '{model_name}'.")


async def _runtime_preflight(config: dict[str, Any], selected_models: list[str], logger: Any) -> None:
    # Fail fast before indexing/evaluation when strict mode is enabled.
    if not bool(config.get("runtime", {}).get("strict_mode", False)):
        return

    to_probe: list[tuple[str, str, str | None, str]] = []
    for model_key in selected_models:
        if model_key == "gemini":
            continue
        model_name = str(config.get("models", {}).get(model_key, "")).strip()
        if model_name:
            to_probe.append((model_name, f"generator:{model_key}", None, "OLLAMA_API_KEY"))

    eval_judge = config.get("evaluation", {}).get("judge", {})
    if str(eval_judge.get("provider", "ollama")).lower() == "ollama":
        judge_model = str(eval_judge.get("model", "")).strip()
        if judge_model:
            to_probe.append((
                judge_model,
                "evaluation_judge",
                str(eval_judge.get("ollama_base_url", "")).strip() or None,
                str(eval_judge.get("api_key_env", "OLLAMA_API_KEY")),
            ))

    web_judge = config.get("web_validator", {}).get("judge", {})
    if str(web_judge.get("provider", "ollama")).lower() == "ollama":
        judge_model = str(web_judge.get("model", "")).strip()
        if judge_model:
            to_probe.append((
                judge_model,
                "web_judge",
                str(web_judge.get("ollama_base_url", "")).strip() or None,
                str(web_judge.get("api_key_env", "OLLAMA_API_KEY")),
            ))

    seen: set[tuple[str, str | None]] = set()
    for model_name, label, base_url, api_key_env in to_probe:
        seen_key = (model_name, base_url)
        if seen_key in seen:
            continue
        seen.add(seen_key)
        logger.info("Runtime preflight: probing %s model '%s'", label, model_name)
        await _probe_ollama_model(config, model_name, label, base_url=base_url, api_key_env=api_key_env)

    reranker_cfg = config.get("reranker", {})
    if bool(reranker_cfg.get("enabled", False)):
        rerank_model = str(reranker_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L6-v2")).strip()
        rerank_local_only = bool(reranker_cfg.get("local_files_only", True))
        logger.info("Runtime preflight: probing kb reranker model '%s'", rerank_model)
        _probe_cross_encoder_model(rerank_model, local_files_only=rerank_local_only, label="kb_reranker")

    web_cfg = config.get("web_validator", {})
    if bool(web_cfg.get("rerank_enabled", True)):
        web_ce_model = str(config.get("embeddings", {}).get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L6-v2")).strip()
        web_local_only = bool(web_cfg.get("cross_encoder_local_files_only", True))
        logger.info("Runtime preflight: probing web reranker model '%s'", web_ce_model)
        _probe_cross_encoder_model(web_ce_model, local_files_only=web_local_only, label="web_reranker")


def _skipped_row(
    q_id: Any,
    category: str,
    question: str,
    golden: str,
    pipeline_value: str,
    model_key: str,
    exc: BaseException,
) -> dict[str, Any]:
    """Build a placeholder result for a question that failed mid-flight.

    Downstream evaluator/validator iterate per row; an empty answer scores 0 across
    all heuristic metrics and the judge will see no answer to score.
    """
    return {
        "q_id": q_id,
        "category": category,
        "question": question,
        "golden": golden,
        "answer": "",
        "kb_context": "",
        "web_context": "",
        "web_status": "SKIPPED",
        "web_search_query": "",
        "web_timed_out": False,
        "web_fallback_used": False,
        "web_fallback_source": "",
        "web_docs_selected": [],
        "web_validation_records": [],
        "web_accepted_count": 0,
        "web_low_confidence_count": 0,
        "web_rejected_count": 0,
        "retrieved_docs": [],
        "router_used": False,
        "router_kind": "",
        "router_n_rows": 0,
        "verifier_modified": False,
        "model_key": model_key,
        "pipeline_mode": pipeline_value,
        "error": f"{type(exc).__name__}: {exc}",
        "judge_method": "skipped",
    }


async def run_single_pipeline(
    model_key: str,
    pipeline_mode: PipelineMode,
    config: dict[str, Any],
    retriever: HybridRetriever,
    crawler: WebCrawler | None,
    questions_df: pd.DataFrame,
    few_shot_builder: FewShotBuilder,
    router: StructuredQueryRouter | None = None,
    verifier: AnswerVerifier | None = None,
    router_log_path: Path | None = None,
) -> list[dict[str, Any]]:
    generator = ModelGenerator(model_key, config, few_shot_builder=few_shot_builder)
    hyde = HyDEExpander(config)
    pipeline_logger = get_logger("pipeline", config)
    on_error = load_generation_config(config)["on_error"]
    results: list[dict[str, Any]] = []
    # Router only makes sense for KB-grounded pipelines; NO_RAG never sees KB context.
    use_router_for_pipeline = pipeline_mode in {
        PipelineMode.RAG,
        PipelineMode.RAG_PRETRAINED,
        PipelineMode.RAG_PRETRAINED_WEB,
    } and router is not None and router.enabled

    for _, row in questions_df.iterrows():
        q = str(row["Question"])
        golden = str(row["Human validated answers"])
        q_id = row.get("Num", "")
        category = str(row.get("Use Case Category", ""))
        try:
            intent = retriever.detect_query_intent(q)
        except Exception as exc:
            if on_error == "fail":
                raise
            pipeline_logger.error(
                "Skipping q_id=%s model=%s pipeline=%s during intent detection: %s",
                q_id, model_key, pipeline_mode.value, exc,
            )
            results.append(_skipped_row(q_id, category, q, golden, pipeline_mode.value, model_key, exc))
            continue

        kb_context = ""
        web_context = ""
        web_status = "NOT_USED"
        web_docs: list[dict[str, Any]] = []
        web_validation_records: list[dict[str, Any]] = []
        web_search_query = ""
        web_accepted = 0
        web_low_confidence = 0
        web_rejected = 0
        web_timed_out = False
        web_fallback_used = False
        web_fallback_source = ""
        docs: list[Any] = []
        router_used = False
        router_kind = ""
        router_filter_spec: dict[str, Any] = {}
        router_n_rows = 0
        try:
            if pipeline_mode in {
                PipelineMode.RAG,
                PipelineMode.RAG_PRETRAINED,
                PipelineMode.RAG_PRETRAINED_WEB,
            }:
                route = None
                if use_router_for_pipeline:
                    try:
                        route = await router.route(q, intent)
                    except Exception as exc:
                        pipeline_logger.warning(
                            "Router failed for q_id=%s; falling back to vector retrieval: %s",
                            q_id, exc,
                        )

                if route is not None and route.used_router:
                    router_used = True
                    router_kind = route.intent_kind
                    router_filter_spec = dict(route.filter_spec)
                    router_n_rows = len(route.matched_rows)
                    docs = list(route.pseudo_docs)
                    kb_context = route.kb_context
                else:
                    search_text, used_hyde = await hyde.expand(q, intent)
                    vector = hyde.get_search_vector(search_text, is_expansion=used_hyde)
                    docs = retriever.retrieve_with_vector(
                        vector,
                        question_text=q,
                        top_k=config["retrieval"]["top_k"],
                        intent=intent,
                    )
                    kb_context = retriever.build_context(docs)

                if router_log_path is not None:
                    _append_router_log(
                        router_log_path,
                        {
                            "q_id": q_id,
                            "model": model_key,
                            "pipeline": pipeline_mode.value,
                            "question": q,
                            "intent": intent.get("type", ""),
                            "used_router": router_used,
                            "router_kind": router_kind,
                            "filter_spec": router_filter_spec,
                            "router_n_rows": router_n_rows,
                            "fallback_reason": (route.fallback_reason if route is not None else ""),
                        },
                    )

            if pipeline_mode == PipelineMode.RAG_PRETRAINED_WEB:
                if crawler is None:
                    raise RuntimeError(
                        "RAG_PRETRAINED_WEB pipeline selected but WebCrawler was not initialized."
                    )
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
                web_fallback_used = bool(crawl_payload.get("fallback_used", False))
                web_fallback_source = str(crawl_payload.get("fallback_source", ""))
                web_accepted = sum(1 for r in web_validation_records if r.get("accepted"))
                web_low_confidence = sum(1 for r in web_validation_records if r.get("low_confidence"))
                web_rejected = sum(1 for r in web_validation_records if not r.get("accepted"))
                if web_docs:
                    web_context = "\n\n".join([str(d.get("context_block", "")).strip() for d in web_docs if str(d.get("context_block", "")).strip()])
                    max_web_chars = int(config.get("crawler", {}).get("max_web_context_chars", 3200))
                    if len(web_context) > max_web_chars:
                        web_context = web_context[:max_web_chars].rsplit(" ", 1)[0].rstrip() + " ..."
                    if web_fallback_used:
                        web_status = "TAVILY_ANSWER_FALLBACK"
                    else:
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
            verifier_modified = False
            if verifier is not None and verifier.enabled and answer:
                # Verify against KB context for the grounded pipelines (NO_RAG has no
                # context to check against, so verify() will return the draft unchanged).
                verify_context = kb_context
                if pipeline_mode == PipelineMode.RAG_PRETRAINED_WEB and web_context:
                    verify_context = (
                        f"{kb_context}\n\n[WEB_CONTEXT]\n{web_context}"
                        if kb_context else f"[WEB_CONTEXT]\n{web_context}"
                    )
                verified, verifier_modified = await verifier.verify(
                    question=q, context=verify_context, draft=answer,
                )
                if verifier_modified:
                    answer = verified
        except Exception as exc:
            if on_error == "fail":
                raise
            pipeline_logger.error(
                "Skipping q_id=%s model=%s pipeline=%s due to %s: %s",
                q_id, model_key, pipeline_mode.value, type(exc).__name__, exc,
            )
            row_record = _skipped_row(q_id, category, q, golden, pipeline_mode.value, model_key, exc)
            row_record["kb_context"] = kb_context
            row_record["web_context"] = web_context
            row_record["web_status"] = web_status
            row_record["retrieved_docs"] = [d.id for d in docs] if docs else []
            results.append(row_record)
            continue

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
                "web_fallback_used": web_fallback_used,
                "web_fallback_source": web_fallback_source,
                "web_docs_selected": web_docs,
                "web_validation_records": web_validation_records,
                "web_accepted_count": web_accepted,
                "web_low_confidence_count": web_low_confidence,
                "web_rejected_count": web_rejected,
                "retrieved_docs": [d.id for d in docs],
                "router_used": router_used,
                "router_kind": router_kind,
                "router_n_rows": router_n_rows,
                "verifier_modified": verifier_modified,
                "model_key": model_key,
                "pipeline_mode": pipeline_mode.value,
            }
        )
    return results


def _append_router_log(path: Path, record: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception:
        # Logging must never crash the run.
        pass


async def main() -> None:
    run_started_utc = datetime.now(timezone.utc)
    args = _parse_args()
    load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"), override=True)
    config = load_config(args.config)
    logger = get_logger("main", config)

    if args.use_kimi_cloud_judge:
        config.setdefault("evaluation", {}).setdefault("judge", {})
        config["evaluation"]["judge"]["provider"] = "ollama"
        config["evaluation"]["judge"]["model"] = "kimi-k2.5:cloud"
        config.setdefault("web_validator", {}).setdefault("judge", {})
        config["web_validator"]["judge"]["provider"] = "ollama"
        config["web_validator"]["judge"]["model"] = "kimi-k2.5:cloud"
    if args.judge_provider:
        config.setdefault("evaluation", {}).setdefault("judge", {})
        config["evaluation"]["judge"]["provider"] = args.judge_provider
        config.setdefault("evaluation", {}).setdefault("validator", {})
        config["evaluation"]["validator"]["provider"] = args.judge_provider
    if args.judge_model:
        config.setdefault("evaluation", {}).setdefault("judge", {})
        config["evaluation"]["judge"]["model"] = args.judge_model
        config.setdefault("evaluation", {}).setdefault("validator", {})
        config["evaluation"]["validator"]["model"] = args.judge_model
    if args.web_judge_provider:
        config.setdefault("web_validator", {}).setdefault("judge", {})
        config["web_validator"]["judge"]["provider"] = args.web_judge_provider
    if args.web_judge_model:
        config.setdefault("web_validator", {}).setdefault("judge", {})
        config["web_validator"]["judge"]["model"] = args.web_judge_model
    if args.retrieval_backend:
        config.setdefault("retrieval", {})
        config["retrieval"]["backend"] = args.retrieval_backend
    if args.extra_test_questions:
        config.setdefault("paths", {})
        config["paths"]["extra_test_questions"] = str(args.extra_test_questions)
    if args.enable_reranker:
        config.setdefault("reranker", {})
        config["reranker"]["enabled"] = True
    if args.disable_reranker:
        config.setdefault("reranker", {})
        config["reranker"]["enabled"] = False
    if args.enable_web_reranker:
        config.setdefault("web_validator", {})
        config["web_validator"]["rerank_enabled"] = True
    if args.disable_web_reranker:
        config.setdefault("web_validator", {})
        config["web_validator"]["rerank_enabled"] = False
    logger.info(
        "Judge config | evaluation: %s/%s | web_validator: %s/%s",
        config.get("evaluation", {}).get("judge", {}).get("provider", "ollama"),
        config.get("evaluation", {}).get("judge", {}).get("model", "qwen2.5:14b"),
        config.get("web_validator", {}).get("judge", {}).get("provider", "ollama"),
        config.get("web_validator", {}).get("judge", {}).get("model", config.get("hyde", {}).get("model", "qwen2.5:14b")),
    )
    logger.info(
        "Retrieval config | backend=%s | reranker.enabled=%s",
        config.get("retrieval", {}).get("backend", "hybrid"),
        config.get("reranker", {}).get("enabled", False),
    )
    gen_cfg_summary = load_generation_config(config)
    logger.info(
        "Generation config | timeout=%.0fs | retries=%d | backoff=%.1fs | on_error=%s",
        gen_cfg_summary["timeout"],
        gen_cfg_summary["max_retries"],
        gen_cfg_summary["retry_backoff_sec"],
        gen_cfg_summary["on_error"],
    )

    selected_models = _model_list(config, args.models)
    selected_pipelines = _pipeline_list(config, args.pipelines)
    await _runtime_preflight(config, selected_models, logger)

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
    # Only build the WebCrawler if a pipeline that actually uses it is selected — it
    # holds Tavily/Firecrawl/Ollama clients that are wasted on no_rag/rag/rag_pretrained (B1).
    needs_crawler = any(p == PipelineMode.RAG_PRETRAINED_WEB for p in selected_pipelines)
    crawler: WebCrawler | None = (
        WebCrawler(config, kb_collection=index.collection, embedding_model=index.embedder)
        if needs_crawler
        else None
    )
    evaluator = RAGASEvaluator(config)
    validator = ScoreValidator(config)
    few_shot_builder = FewShotBuilder(train_df, config)
    router = StructuredQueryRouter(config) if bool(config.get("structured_router", {}).get("enabled", True)) else None
    verifier = AnswerVerifier(config) if bool(config.get("verifier", {}).get("enabled", True)) else None

    logs_dir = resolve_path(config, config["paths"]["logs_dir"])
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    router_log_path = Path(logs_dir) / f"router_{run_started_utc.strftime('%Y%m%d_%H%M%S')}.jsonl"
    logger.info(
        "Router enabled=%s | Verifier enabled=%s | Router log: %s",
        router is not None and router.enabled,
        verifier is not None and verifier.enabled,
        router_log_path,
    )

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
                    router=router,
                    verifier=verifier,
                    router_log_path=router_log_path,
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

                # Refresh the comparison workbook after every cell so a mid-run crash
                # still leaves a usable FINAL_COMPARISON.xlsx reflecting partial work.
                try:
                    interim_comparison_path = build_comparison_report(all_results, config)
                    logger.info("Interim comparison report refreshed: %s", interim_comparison_path)
                except Exception as exc:  # never let report writing kill the run
                    logger.warning("Failed to refresh comparison report mid-run: %s", exc)

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
