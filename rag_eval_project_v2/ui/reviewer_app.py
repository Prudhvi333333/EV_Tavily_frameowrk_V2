from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROGRESS_DIR = OUTPUTS_DIR / "progress"
LOG_PATH = OUTPUTS_DIR / "logs" / "web_validation_proof.jsonl"


def _parse_iso_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _list_progress_files() -> list[Path]:
    if not PROGRESS_DIR.exists():
        return []
    files = list(PROGRESS_DIR.glob("run_progress_*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _cli_progress_override() -> Path | None:
    args = sys.argv[1:]
    if "--progress-file" not in args:
        return None
    idx = args.index("--progress-file")
    if idx + 1 >= len(args):
        return None
    p = Path(args[idx + 1]).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p if p.exists() else None


def _load_progress(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_read_sheet(report_path: Path, sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(report_path, sheet_name=sheet_name)
    except Exception:
        return pd.DataFrame()


def _load_proof_rows(
    run_started_utc: datetime | None,
    pipeline_filters: set[str],
    question_filters: set[str],
) -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        rec_ts = _parse_iso_utc(rec.get("timestamp", ""))
        if run_started_utc and rec_ts and rec_ts < run_started_utc:
            continue
        pipeline = str(rec.get("pipeline", ""))
        qid = str(rec.get("question_id", ""))
        if pipeline_filters and pipeline not in pipeline_filters:
            continue
        if question_filters and qid and qid not in question_filters:
            continue
        rows.append(
            {
                "timestamp": rec.get("timestamp", ""),
                "question_id": qid,
                "pipeline": pipeline,
                "source_domain": rec.get("source_domain", ""),
                "decision": rec.get("decision", ""),
                "final_score": rec.get("final_score", None),
                "s1_keyword": rec.get("signals", {}).get("s1_keyword", None),
                "s2_semantic": rec.get("signals", {}).get("s2_semantic", None),
                "s3_llm": rec.get("signals", {}).get("s3_llm", None),
                "s3_partial_relevance": rec.get("signals", {}).get("s3_partial_relevance", None),
                "reason": rec.get("s3_reason", ""),
                "url": rec.get("url", ""),
                "search_query": rec.get("search_query", ""),
                "text_preview": rec.get("text_preview", ""),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def _build_run_data(progress_data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows = progress_data.get("runs", [])
    runs_df = pd.DataFrame(run_rows)
    if runs_df.empty:
        return runs_df, pd.DataFrame(), pd.DataFrame()
    if "split" not in runs_df.columns:
        runs_df["split"] = "test"

    result_frames: list[pd.DataFrame] = []
    web_frames: list[pd.DataFrame] = []
    for row in run_rows:
        report = Path(str(row.get("report", "")))
        if not report.exists():
            continue
        split = str(row.get("split", "test"))
        model = str(row.get("model", ""))
        pipeline = str(row.get("pipeline", ""))

        results_df = _safe_read_sheet(report, "Results")
        if not results_df.empty:
            results_df["Run_Split"] = split
            results_df["Run_Model"] = model
            results_df["Run_Pipeline"] = pipeline
            result_frames.append(results_df)

        web_df = _safe_read_sheet(report, "Web_Validation")
        if not web_df.empty and "No web validation records available for this run." not in web_df.to_string(index=False):
            web_df["Run_Split"] = split
            web_df["Run_Model"] = model
            web_df["Run_Pipeline"] = pipeline
            web_frames.append(web_df)

    return (
        runs_df,
        pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame(),
        pd.concat(web_frames, ignore_index=True) if web_frames else pd.DataFrame(),
    )


def main() -> None:
    st.set_page_config(page_title="RAG Reviewer UI", page_icon="RAG", layout="wide")
    st.title("RAG Evaluation Reviewer UI")
    st.caption("Interactive reviewer app for run summaries, question outcomes, web validation, and proof logs.")

    progress_files = _list_progress_files()
    if not progress_files:
        st.warning("No progress files found in outputs/progress. Run main.py first.")
        return

    override_path = _cli_progress_override()
    if override_path is not None and override_path not in progress_files:
        progress_files = [override_path] + progress_files

    file_labels = [p.name for p in progress_files]
    default_idx = 0
    if override_path is not None:
        for i, p in enumerate(progress_files):
            if p == override_path:
                default_idx = i
                break
    selected_name = st.sidebar.selectbox("Progress File", file_labels, index=default_idx)
    selected_progress = progress_files[file_labels.index(selected_name)]
    progress_data = _load_progress(selected_progress)
    run_started_utc = _parse_iso_utc(progress_data.get("run_started_utc", ""))

    runs_df, results_df, web_df = _build_run_data(progress_data)
    if runs_df.empty:
        st.warning("Selected progress file has no run entries.")
        return

    split_options = sorted(set(runs_df.get("split", pd.Series(["test"])).astype(str)))
    model_options = sorted(set(runs_df.get("model", pd.Series(dtype=str)).astype(str)))
    pipeline_options = sorted(set(runs_df.get("pipeline", pd.Series(dtype=str)).astype(str)))

    selected_splits = st.sidebar.multiselect("Split", split_options, default=split_options)
    selected_models = st.sidebar.multiselect("Model", model_options, default=model_options)
    selected_pipelines = st.sidebar.multiselect("Pipeline", pipeline_options, default=pipeline_options)

    run_filter = runs_df["split"].astype(str).isin(selected_splits) & runs_df["model"].astype(str).isin(selected_models) & runs_df["pipeline"].astype(str).isin(selected_pipelines)
    filtered_runs = runs_df[run_filter].reset_index(drop=True)

    left, mid, right, right2 = st.columns(4)
    left.metric("Run Rows", len(filtered_runs))
    mid.metric("Questions", int(filtered_runs["questions"].sum()) if "questions" in filtered_runs else 0)
    right.metric("Mean Final Score", round(float(filtered_runs["mean_final_score"].mean()), 4) if "mean_final_score" in filtered_runs and not filtered_runs.empty else 0.0)
    right2.metric("Comparison Report", "Available" if progress_data.get("comparison_report") else "N/A")

    st.subheader("Run Summary")
    st.dataframe(filtered_runs, use_container_width=True, hide_index=True)

    filtered_results = pd.DataFrame()
    if not results_df.empty:
        mask = (
            results_df["Run_Split"].astype(str).isin(selected_splits)
            & results_df["Run_Model"].astype(str).isin(selected_models)
            & results_df["Run_Pipeline"].astype(str).isin(selected_pipelines)
        )
        filtered_results = results_df[mask].reset_index(drop=True)

    if not filtered_results.empty:
        st.subheader("Question Outcomes")
        metric_col = "Final_Score" if "Final_Score" in filtered_results.columns else None
        flagged_col = "Validation_Flags" if "Validation_Flags" in filtered_results.columns else None
        cols = st.columns(3)
        cols[0].metric("Rows", len(filtered_results))
        cols[1].metric(
            "Flagged",
            int((filtered_results[flagged_col].astype(str) == "FLAGGED").sum()) if flagged_col else 0,
        )
        cols[2].metric(
            "Avg Final Score",
            round(float(filtered_results[metric_col].mean()), 4) if metric_col else 0.0,
        )
        st.dataframe(filtered_results, use_container_width=True, hide_index=True)

    filtered_web = pd.DataFrame()
    if not web_df.empty:
        mask = (
            web_df["Run_Split"].astype(str).isin(selected_splits)
            & web_df["Run_Model"].astype(str).isin(selected_models)
            & web_df["Run_Pipeline"].astype(str).isin(selected_pipelines)
        )
        filtered_web = web_df[mask].reset_index(drop=True)

    if not filtered_web.empty:
        st.subheader("Web Validation")
        decision_counts = filtered_web["Decision"].astype(str).value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("Web Rows", len(filtered_web))
        c2.metric("Accepted", int(filtered_web["Accepted"].fillna(False).astype(bool).sum()) if "Accepted" in filtered_web.columns else 0)
        c3.metric("Rejected", int((~filtered_web["Accepted"].fillna(False).astype(bool)).sum()) if "Accepted" in filtered_web.columns else 0)
        st.bar_chart(decision_counts)
        st.dataframe(filtered_web, use_container_width=True, hide_index=True)

    question_ids: set[str] = set()
    if not filtered_results.empty and "Q_ID" in filtered_results.columns:
        question_ids = {str(x).strip() for x in filtered_results["Q_ID"].tolist() if str(x).strip()}
    pipeline_ids = {
        f"{str(row.get('model', '')).strip()}_{str(row.get('pipeline', '')).strip()}"
        for _, row in filtered_runs.iterrows()
    }
    proof_df = _load_proof_rows(run_started_utc=run_started_utc, pipeline_filters=pipeline_ids, question_filters=question_ids)

    st.subheader("Proof Log")
    if proof_df.empty:
        st.info("No proof records matched the selected run filters.")
    else:
        st.dataframe(proof_df, use_container_width=True, hide_index=True)

    st.caption(f"Selected progress file: {selected_progress}")
    if progress_data.get("comparison_report"):
        st.caption(f"Comparison report: {progress_data.get('comparison_report')}")


if __name__ == "__main__":
    main()
