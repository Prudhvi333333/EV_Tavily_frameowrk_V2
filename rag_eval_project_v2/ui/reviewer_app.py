from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROGRESS_DIR = OUTPUTS_DIR / "progress"
LOG_PATH = OUTPUTS_DIR / "logs" / "web_validation_proof.jsonl"


def _apply_custom_styles() -> None:
    st.markdown(
        """
        <style>
          :root {
            --bg-1: #f2f7f5;
            --bg-2: #eaf2ff;
            --card: #ffffff;
            --line: #d9e4e1;
            --ink: #12312b;
            --muted: #5c6d69;
            --brand: #0a7d6e;
            --brand-2: #1c64f2;
          }
          .stApp {
            background:
              radial-gradient(circle at 10% -10%, #d8f0ea 0, transparent 35%),
              radial-gradient(circle at 100% 0%, #dce8ff 0, transparent 32%),
              linear-gradient(180deg, var(--bg-1), var(--bg-2));
          }
          .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2.2rem;
          }
          .dashboard-hero {
            border: 1px solid #bfe0d9;
            border-radius: 16px;
            padding: 16px 20px;
            background: linear-gradient(125deg, #0c7f70, #1656cf);
            color: #f4fffd;
            box-shadow: 0 10px 28px rgba(18, 38, 70, 0.18);
          }
          .dashboard-hero h2 {
            margin: 0;
            font-size: 1.35rem;
            letter-spacing: 0.2px;
          }
          .dashboard-hero p {
            margin: 4px 0 0;
            color: #dbf6f0;
            font-size: 0.92rem;
          }
          .kpi-card {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--card);
            padding: 10px 12px;
            box-shadow: 0 4px 12px rgba(17, 38, 47, 0.06);
          }
          .kpi-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.35px;
          }
          .kpi-value {
            color: var(--ink);
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 3px;
          }
          .section-note {
            color: #425e58;
            font-size: 0.9rem;
            margin-top: -6px;
          }
          div[data-testid="stDataFrame"] {
            border: 1px solid #d5e3df;
            border-radius: 10px;
            background: #fff;
          }
          .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
          }
          .stTabs [data-baseweb="tab"] {
            background: #eef6f3;
            border-radius: 10px;
            border: 1px solid #d5e5df;
            color: #16433b;
            font-weight: 600;
          }
          .stTabs [aria-selected="true"] {
            background: #0d7b6d !important;
            color: #ffffff !important;
            border-color: #0d7b6d !important;
          }
          .stTextArea textarea {
            background: #fcfefd !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _safe_cell(row: pd.Series, col: str | None) -> str:
    if not col:
        return ""
    return str(row.get(col, "")).strip()


def _metric_columns(df: pd.DataFrame) -> list[str]:
    ordered = [
        "Faithfulness",
        "Answer_Relevancy",
        "Context_Precision",
        "Context_Recall",
        "Answer_Correctness",
        "Source_Attribution",
        "Web_Grounding",
        "Pretrained_Leak_Rate",
        "Final_Score",
    ]
    return [c for c in ordered if c in df.columns]


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
    return pd.DataFrame(rows).sort_values("timestamp", ascending=False).reset_index(drop=True)


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


def _render_kpi_card(label: str, value: str) -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
    </div>
    """


def main() -> None:
    st.set_page_config(page_title="RAG Reviewer UI", page_icon="RI", layout="wide")
    _apply_custom_styles()

    st.markdown(
        """
        <div class="dashboard-hero">
          <h2>RAG Evaluation Reviewer Console</h2>
          <p>Business-ready view of run health, quality metrics, web evidence validation, and audit proof trails.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    run_filter = (
        runs_df["split"].astype(str).isin(selected_splits)
        & runs_df["model"].astype(str).isin(selected_models)
        & runs_df["pipeline"].astype(str).isin(selected_pipelines)
    )
    filtered_runs = runs_df[run_filter].reset_index(drop=True)

    filtered_results = pd.DataFrame()
    if not results_df.empty:
        mask = (
            results_df["Run_Split"].astype(str).isin(selected_splits)
            & results_df["Run_Model"].astype(str).isin(selected_models)
            & results_df["Run_Pipeline"].astype(str).isin(selected_pipelines)
        )
        filtered_results = results_df[mask].reset_index(drop=True)

    filtered_web = pd.DataFrame()
    if not web_df.empty:
        mask = (
            web_df["Run_Split"].astype(str).isin(selected_splits)
            & web_df["Run_Model"].astype(str).isin(selected_models)
            & web_df["Run_Pipeline"].astype(str).isin(selected_pipelines)
        )
        filtered_web = web_df[mask].reset_index(drop=True)

    question_ids: set[str] = set()
    if not filtered_results.empty and "Q_ID" in filtered_results.columns:
        question_ids = {str(x).strip() for x in filtered_results["Q_ID"].tolist() if str(x).strip()}
    pipeline_ids = {
        f"{str(row.get('model', '')).strip()}_{str(row.get('pipeline', '')).strip()}"
        for _, row in filtered_runs.iterrows()
    }
    proof_df = _load_proof_rows(
        run_started_utc=run_started_utc,
        pipeline_filters=pipeline_ids,
        question_filters=question_ids,
    )

    total_questions = int(filtered_runs["questions"].sum()) if "questions" in filtered_runs.columns else 0
    mean_score = (
        round(float(filtered_runs["mean_final_score"].mean()), 4)
        if "mean_final_score" in filtered_runs.columns and not filtered_runs.empty
        else 0.0
    )
    accepted = int(filtered_web["Accepted"].fillna(False).astype(bool).sum()) if "Accepted" in filtered_web.columns else 0
    rejected = int((~filtered_web["Accepted"].fillna(False).astype(bool)).sum()) if "Accepted" in filtered_web.columns else 0
    acceptance_rate = round((accepted / max(accepted + rejected, 1)) * 100, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_render_kpi_card("Run Rows", str(len(filtered_runs))), unsafe_allow_html=True)
    c2.markdown(_render_kpi_card("Questions", str(total_questions)), unsafe_allow_html=True)
    c3.markdown(_render_kpi_card("Mean Final Score", f"{mean_score:.4f}"), unsafe_allow_html=True)
    c4.markdown(_render_kpi_card("Web Acceptance", f"{acceptance_rate:.1f}%"), unsafe_allow_html=True)

    tabs = st.tabs(["Overview", "Question Explorer", "Web Evidence", "Audit Trail"])

    with tabs[0]:
        st.markdown('<p class="section-note">Executive summary of model/pipeline performance for selected filters.</p>', unsafe_allow_html=True)

        if not filtered_runs.empty and "mean_final_score" in filtered_runs.columns:
            perf = (
                filtered_runs.groupby(["model", "pipeline"], as_index=False)["mean_final_score"]
                .mean()
                .sort_values("mean_final_score", ascending=False)
            )
            st.markdown("**Pipeline Performance**")
            st.bar_chart(perf.set_index(["model", "pipeline"])["mean_final_score"], height=300)

            heat = perf.pivot_table(index="model", columns="pipeline", values="mean_final_score")
            st.markdown("**Model x Pipeline Score Grid**")
            st.dataframe(
                heat.style.format("{:.4f}").background_gradient(cmap="YlGnBu"),
                use_container_width=True,
            )

        st.markdown("**Run Summary**")
        st.dataframe(filtered_runs, use_container_width=True, hide_index=True)

        if progress_data.get("comparison_report"):
            st.caption(f"Comparison report: {progress_data.get('comparison_report')}")

    with tabs[1]:
        st.markdown('<p class="section-note">Drill into one specific question to inspect context, answer quality, and validation reasons.</p>', unsafe_allow_html=True)

        if filtered_results.empty:
            st.info("No question-level rows found for current filters.")
        else:
            flagged_col = _first_existing_column(filtered_results, ["Validation_Flags", "validation_flags"])
            avg_final = (
                round(float(filtered_results["Final_Score"].mean()), 4)
                if "Final_Score" in filtered_results.columns
                else 0.0
            )
            qm1, qm2, qm3 = st.columns(3)
            qm1.metric("Rows", len(filtered_results))
            qm2.metric(
                "Flagged",
                int((filtered_results[flagged_col].astype(str) == "FLAGGED").sum()) if flagged_col else 0,
            )
            qm3.metric("Avg Final Score", avg_final)
            st.dataframe(filtered_results, use_container_width=True, hide_index=True, height=280)

            qid_col = _first_existing_column(filtered_results, ["Q_ID", "Question_ID", "q_id"])
            question_col = _first_existing_column(filtered_results, ["Question", "question"])
            golden_col = _first_existing_column(filtered_results, ["Golden", "Golden_Answer", "golden"])
            answer_col = _first_existing_column(filtered_results, ["Answer", "Generated_Answer", "answer"])
            context_col = _first_existing_column(filtered_results, ["Context", "KB_Context", "kb_context"])
            web_context_col = _first_existing_column(filtered_results, ["Web_Context", "web_context"])
            web_status_col = _first_existing_column(filtered_results, ["Web_Status", "web_status"])
            validation_flags_col = _first_existing_column(filtered_results, ["Validation_Flags", "validation_flags"])
            validation_reason_col = _first_existing_column(filtered_results, ["Validation_Reason", "validation_reason"])
            search_query_col = _first_existing_column(filtered_results, ["Web_Search_Query", "web_search_query"])
            timed_out_col = _first_existing_column(filtered_results, ["Web_Timed_Out", "web_timed_out"])
            run_model_col = _first_existing_column(filtered_results, ["Run_Model", "Model", "model", "model_key"])
            run_pipeline_col = _first_existing_column(filtered_results, ["Run_Pipeline", "Pipeline", "pipeline", "pipeline_mode"])

            result_rows = filtered_results.reset_index(drop=True)

            def _deep_dive_label(idx: int) -> str:
                row = result_rows.iloc[idx]
                qid = _safe_cell(row, qid_col) or str(idx + 1)
                model = _safe_cell(row, run_model_col) or "model?"
                pipeline = _safe_cell(row, run_pipeline_col) or "pipeline?"
                q_text = _safe_cell(row, question_col).replace("\n", " ")
                if len(q_text) > 88:
                    q_text = q_text[:88].rstrip() + " ..."
                return f"Q{qid} | {model}/{pipeline} | {q_text}"

            selected_idx = st.selectbox(
                "Select question row",
                options=list(range(len(result_rows))),
                format_func=_deep_dive_label,
                index=0,
            )
            selected = result_rows.iloc[int(selected_idx)]

            qid_value = _safe_cell(selected, qid_col)
            model_value = _safe_cell(selected, run_model_col)
            pipeline_value = _safe_cell(selected, run_pipeline_col)
            pipeline_key = f"{model_value}_{pipeline_value}".strip("_")

            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Q_ID", qid_value or "N/A")
            d2.metric("Model", model_value or "N/A")
            d3.metric("Pipeline", pipeline_value or "N/A")
            d4.metric("Final", _safe_cell(selected, "Final_Score") or "N/A")

            with st.expander("Prompt Inputs", expanded=True):
                st.text_area("Question", _safe_cell(selected, question_col), height=80, disabled=True)
                if context_col:
                    st.text_area("KB Context", _safe_cell(selected, context_col), height=200, disabled=True)
                if web_context_col:
                    st.text_area("Web Context", _safe_cell(selected, web_context_col), height=160, disabled=True)
                if search_query_col:
                    st.text_input("Web Search Query", _safe_cell(selected, search_query_col), disabled=True)

            with st.expander("Generated Output", expanded=True):
                st.text_area("Golden Answer", _safe_cell(selected, golden_col), height=120, disabled=True)
                st.text_area("Generated Answer", _safe_cell(selected, answer_col), height=180, disabled=True)

            with st.expander("Evaluation & Validation", expanded=True):
                metric_cols = _metric_columns(result_rows)
                if metric_cols:
                    metric_rows = pd.DataFrame(
                        [{"Metric": c, "Score": selected.get(c, np.nan)} for c in metric_cols]
                    )
                    st.dataframe(metric_rows, use_container_width=True, hide_index=True)
                if validation_flags_col:
                    st.text_input("Validation Flags", _safe_cell(selected, validation_flags_col), disabled=True)
                if validation_reason_col:
                    st.text_area("Validation Reason", _safe_cell(selected, validation_reason_col), height=110, disabled=True)
                if web_status_col:
                    st.text_input("Web Status", _safe_cell(selected, web_status_col), disabled=True)
                if timed_out_col:
                    st.text_input("Web Timed Out", _safe_cell(selected, timed_out_col), disabled=True)

            if not filtered_web.empty:
                qid_web_col = _first_existing_column(filtered_web, ["Q_ID", "q_id", "Question_ID"])
                model_web_col = _first_existing_column(filtered_web, ["Run_Model", "Model", "model"])
                pipeline_web_col = _first_existing_column(filtered_web, ["Run_Pipeline", "Pipeline", "pipeline"])
                web_mask = pd.Series([True] * len(filtered_web), index=filtered_web.index)
                if qid_web_col and qid_value:
                    web_mask = web_mask & (filtered_web[qid_web_col].astype(str).str.strip() == qid_value)
                if model_web_col and model_value:
                    web_mask = web_mask & (filtered_web[model_web_col].astype(str).str.strip() == model_value)
                if pipeline_web_col and pipeline_value:
                    web_mask = web_mask & (filtered_web[pipeline_web_col].astype(str).str.strip() == pipeline_value)
                deep_web = filtered_web[web_mask].reset_index(drop=True)
                st.markdown("**Web validation rows for selected question**")
                if deep_web.empty:
                    st.info("No web validation rows for this selected question.")
                else:
                    st.dataframe(deep_web, use_container_width=True, hide_index=True)

            if not proof_df.empty:
                proof_mask = pd.Series([True] * len(proof_df), index=proof_df.index)
                if qid_value and "question_id" in proof_df.columns:
                    proof_mask = proof_mask & (proof_df["question_id"].astype(str).str.strip() == qid_value)
                if pipeline_key and "pipeline" in proof_df.columns:
                    proof_mask = proof_mask & (proof_df["pipeline"].astype(str).str.strip() == pipeline_key)
                deep_proof = proof_df[proof_mask].reset_index(drop=True)
                st.markdown("**Proof log rows for selected question**")
                if deep_proof.empty:
                    st.info("No proof rows matched this question/pipeline.")
                else:
                    st.dataframe(deep_proof, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown('<p class="section-note">Web document validation quality and relevance signal distribution.</p>', unsafe_allow_html=True)
        if filtered_web.empty:
            st.info("No web validation records for current filters.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Web Rows", len(filtered_web))
            c2.metric("Accepted", accepted)
            c3.metric("Rejected", rejected)

            if "Decision" in filtered_web.columns:
                st.markdown("**Decision Distribution**")
                st.bar_chart(filtered_web["Decision"].astype(str).value_counts(), height=250)
            if "Source_Domain" in filtered_web.columns:
                st.markdown("**Top Source Domains**")
                st.bar_chart(filtered_web["Source_Domain"].astype(str).value_counts().head(12), height=250)

            st.markdown("**Web Validation Table**")
            st.dataframe(filtered_web, use_container_width=True, hide_index=True, height=380)

    with tabs[3]:
        st.markdown('<p class="section-note">Run-level evidence trail for reviewer audits and reproducibility checks.</p>', unsafe_allow_html=True)
        if proof_df.empty:
            st.info("No proof records matched current run filters.")
        else:
            p1, p2, p3 = st.columns(3)
            p1.metric("Proof Rows", len(proof_df))
            p2.metric("Unique Questions", proof_df["question_id"].astype(str).nunique() if "question_id" in proof_df.columns else 0)
            p3.metric("Unique Pipelines", proof_df["pipeline"].astype(str).nunique() if "pipeline" in proof_df.columns else 0)
            st.dataframe(proof_df, use_container_width=True, hide_index=True, height=420)

    st.caption(f"Progress file: {selected_progress}")


if __name__ == "__main__":
    main()

