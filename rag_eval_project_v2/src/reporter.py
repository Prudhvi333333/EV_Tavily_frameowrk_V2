from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

from src.utils.config_loader import resolve_path


METRIC_COLUMN_MAP = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer_Relevancy",
    "context_precision": "Context_Precision",
    "context_recall": "Context_Recall",
    "answer_correctness": "Answer_Correctness",
    "source_attribution": "Source_Attribution",
    "web_grounding": "Web_Grounding",
}


def build_report(
    results: list[dict[str, Any]],
    model_name: str,
    pipeline_mode: str,
    config: dict[str, Any],
    train_results: list[dict[str, Any]] | None = None,
) -> str:
    reports_dir = resolve_path(config, config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(reports_dir) / f"{model_name}_{pipeline_mode}_report.xlsx"

    rows = [_row_to_result_record(r, pipeline_mode) for r in results]
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame([{"Note": "No results available."}])

    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Results", index=False)
        _build_summary_sheet(df).to_excel(writer, sheet_name="Summary", index=False)
        _build_train_test_sheet(df, train_results).to_excel(writer, sheet_name="Train_vs_Test", index=False)
        _build_validation_audit_sheet(results).to_excel(writer, sheet_name="Validation_Audit", index=False)

    _apply_conditional_colors(report_path, sheet_name="Results")
    return str(report_path)


def build_comparison_report(all_results: dict[tuple[str, str], list[dict[str, Any]]], config: dict[str, Any]) -> str:
    reports_dir = resolve_path(config, config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(reports_dir) / "FINAL_COMPARISON.xlsx"

    agg_rows: list[dict[str, Any]] = []
    for (model, pipeline), rows in all_results.items():
        metric_matrix = pd.DataFrame([r.get("metric_scores", {}) for r in rows])
        means = metric_matrix.mean(numeric_only=True).to_dict() if not metric_matrix.empty else {}
        agg_rows.append(
            {
                "Model": model,
                "Pipeline": pipeline,
                "Final_Score_Mean": float(np.mean([r.get("final_score", 0.0) for r in rows])) if rows else 0.0,
                **{METRIC_COLUMN_MAP.get(k, k): float(v) for k, v in means.items()},
            }
        )
    agg_df = pd.DataFrame(agg_rows).sort_values(["Model", "Pipeline"])

    rag_vs_no = _build_rag_vs_norag(agg_df)
    rankings = agg_df.sort_values("Final_Score_Mean", ascending=False).reset_index(drop=True)
    pretrained_delta = _build_delta_sheet(agg_df, base_pipeline="rag", compare_pipeline="rag_pretrained", label="Pretrained")
    web_delta = _build_delta_sheet(agg_df, base_pipeline="rag_pretrained", compare_pipeline="rag_pretrained_web", label="Web")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        agg_df.to_excel(writer, sheet_name="All_12_Pipelines", index=False)
        rag_vs_no.to_excel(writer, sheet_name="RAG_vs_NoRAG", index=False)
        rankings.to_excel(writer, sheet_name="Rankings", index=False)
        pretrained_delta.to_excel(writer, sheet_name="Pretrained_Contribution", index=False)
        web_delta.to_excel(writer, sheet_name="Web_Contribution", index=False)

    return str(out_path)


def _row_to_result_record(row: dict[str, Any], pipeline_mode: str) -> dict[str, Any]:
    out = {
        "Q_ID": row.get("q_id", ""),
        "Category": row.get("category", ""),
        "Question": row.get("question", ""),
        "Golden": row.get("golden", ""),
        "Answer": row.get("answer", ""),
    }
    if pipeline_mode != "no_rag":
        out["Context"] = row.get("kb_context", "")
    if pipeline_mode == "rag_pretrained_web":
        out["Web_Context"] = row.get("web_context", "")

    metric_scores = row.get("metric_scores", {})
    for metric, value in metric_scores.items():
        out[METRIC_COLUMN_MAP.get(metric, metric)] = value

    if pipeline_mode in {"rag_pretrained", "rag_pretrained_web"}:
        answer_text = str(row.get("answer", ""))
        pretrained_tags = answer_text.count("[PRETRAINED]")
        kb_tags = max(answer_text.count("[KB]"), 1)
        out["Pretrained_Leak_Rate"] = round(pretrained_tags / kb_tags, 4)

    out["Final_Score"] = row.get("final_score", np.nan)
    out["Validation_Flags"] = row.get("validation_flags", "")
    out["Validation_Reason"] = row.get("validation_reason", "")
    return out


def _build_summary_sheet(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c in set(METRIC_COLUMN_MAP.values()) | {"Final_Score"}]
    if not metric_cols:
        return pd.DataFrame([{"Metric": "N/A", "Mean": np.nan, "Std": np.nan}])
    rows = []
    for col in metric_cols:
        rows.append({"Metric": col, "Mean": float(df[col].mean()), "Std": float(df[col].std(ddof=0))})
    return pd.DataFrame(rows)


def _build_train_test_sheet(test_df: pd.DataFrame, train_results: list[dict[str, Any]] | None) -> pd.DataFrame:
    if not train_results:
        return pd.DataFrame(
            [{"Split": "test", "Final_Score_Mean": float(test_df["Final_Score"].mean()) if "Final_Score" in test_df else np.nan}]
        )
    train_scores = [float(r.get("final_score", 0.0)) for r in train_results]
    test_scores = test_df["Final_Score"].dropna().tolist() if "Final_Score" in test_df else []
    return pd.DataFrame(
        [
            {"Split": "train", "Final_Score_Mean": float(np.mean(train_scores)) if train_scores else np.nan},
            {"Split": "test", "Final_Score_Mean": float(np.mean(test_scores)) if test_scores else np.nan},
        ]
    )


def _build_validation_audit_sheet(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for r in results:
        if not r.get("validation_flags"):
            continue
        rows.append(
            {
                "Q_ID": r.get("q_id", ""),
                "Question": r.get("question", ""),
                "Flagged_Metrics": r.get("flagged_metrics", ""),
                "Reason": r.get("validation_reason", ""),
                "Adjustments": str(r.get("metric_adjustments", {})),
            }
        )
    if not rows:
        rows = [{"Q_ID": "", "Question": "No flagged rows", "Flagged_Metrics": "", "Reason": "", "Adjustments": ""}]
    return pd.DataFrame(rows)


def _apply_conditional_colors(report_path: str | Path, sheet_name: str) -> None:
    wb = load_workbook(report_path)
    ws = wb[sheet_name]
    headers = [c.value for c in ws[1]]
    metric_columns = {"Faithfulness", "Answer_Relevancy", "Context_Precision", "Context_Recall", "Answer_Correctness", "Source_Attribution", "Web_Grounding", "Final_Score"}
    green = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    yellow = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    red = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")

    for col_idx, header in enumerate(headers, start=1):
        if header not in metric_columns:
            continue
        for row_idx in range(2, ws.max_row + 1):
            val = ws.cell(row=row_idx, column=col_idx).value
            if isinstance(val, (int, float)):
                if val >= 0.7:
                    ws.cell(row=row_idx, column=col_idx).fill = green
                elif val >= 0.5:
                    ws.cell(row=row_idx, column=col_idx).fill = yellow
                else:
                    ws.cell(row=row_idx, column=col_idx).fill = red

    wb.save(report_path)


def _build_rag_vs_norag(agg_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(agg_df["Model"].unique()):
        mdf = agg_df[agg_df["Model"] == model]
        rag = mdf[mdf["Pipeline"] == "rag"]
        no_rag = mdf[mdf["Pipeline"] == "no_rag"]
        rows.append(
            {
                "Model": model,
                "RAG_Answer_Correctness": float(rag["Answer_Correctness"].iloc[0]) if not rag.empty and "Answer_Correctness" in rag else np.nan,
                "NoRAG_Answer_Correctness": float(no_rag["Answer_Correctness"].iloc[0]) if not no_rag.empty and "Answer_Correctness" in no_rag else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_delta_sheet(agg_df: pd.DataFrame, base_pipeline: str, compare_pipeline: str, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in sorted(agg_df["Model"].unique()):
        mdf = agg_df[agg_df["Model"] == model]
        base = mdf[mdf["Pipeline"] == base_pipeline]
        comp = mdf[mdf["Pipeline"] == compare_pipeline]
        if base.empty or comp.empty:
            delta = np.nan
        else:
            delta = float(comp["Final_Score_Mean"].iloc[0] - base["Final_Score_Mean"].iloc[0])
        rows.append({"Model": model, f"{label}_Delta": delta})
    return pd.DataFrame(rows)

