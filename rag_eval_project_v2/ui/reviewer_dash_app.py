from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html
from dash import dash_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROGRESS_DIR = OUTPUTS_DIR / "progress"
LOG_PATH = OUTPUTS_DIR / "logs" / "web_validation_proof.jsonl"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def _parse_iso_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _safe_read_sheet(report_path: Path, sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(report_path, sheet_name=sheet_name)
    except Exception:
        return pd.DataFrame()


def _list_progress_files() -> list[Path]:
    if not PROGRESS_DIR.exists():
        return []
    files = list(PROGRESS_DIR.glob("run_progress_*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


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
                "Timestamp": rec.get("timestamp", ""),
                "Question_ID": qid,
                "Pipeline": pipeline,
                "Source_Domain": rec.get("source_domain", ""),
                "Decision": rec.get("decision", ""),
                "Final_Score": rec.get("final_score", None),
                "S1_Keyword": rec.get("signals", {}).get("s1_keyword", None),
                "S2_Semantic": rec.get("signals", {}).get("s2_semantic", None),
                "S3_LLM": rec.get("signals", {}).get("s3_llm", None),
                "S3_Partial_Relevance": rec.get("signals", {}).get("s3_partial_relevance", None),
                "Reason": rec.get("s3_reason", ""),
                "URL": rec.get("url", ""),
                "Search_Query": rec.get("search_query", ""),
                "Text_Preview": rec.get("text_preview", ""),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Timestamp", ascending=False).reset_index(drop=True)


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


def _serialize_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    cleaned = df.replace({np.nan: None})
    return cleaned.to_dict(orient="records")


def _make_kpi(label: str, value: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(label, className="kpi-label"),
                html.Div(value, className="kpi-value"),
            ]
        ),
        className="kpi-card",
    )


def _tone_for_metric(
    value: float | None,
    *,
    good_at_least: float,
    warn_at_least: float,
) -> str:
    if value is None or np.isnan(value):
        return "insight-neutral"
    if value >= good_at_least:
        return "insight-good"
    if value >= warn_at_least:
        return "insight-warn"
    return "insight-bad"


def _insight_chip(label: str, value: str, tone_class: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, className="insight-label"),
            html.Div(value, className="insight-value"),
        ],
        className=f"insight-chip {tone_class}",
    )


def _empty_fig(title: str) -> Any:
    fig = px.bar(title=title)
    fig.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "No data for current filters",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 14, "color": "#66757f"},
            }
        ],
    )
    return fig


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _table(
    columns: list[str],
    table_id: str,
    page_size: int = 10,
    style_data_conditional: list[dict[str, Any]] | None = None,
) -> dash_table.DataTable:
    base_style = [
        {
            "if": {"row_index": "odd"},
            "backgroundColor": "#fbfdff",
        }
    ]
    if style_data_conditional:
        base_style.extend(style_data_conditional)
    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": c, "id": c} for c in columns],
        data=[],
        page_size=page_size,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#f0f6fb",
            "fontWeight": "700",
            "border": "1px solid #d7e2ea",
            "color": "#1f3442",
            "fontFamily": "Manrope, Segoe UI, Arial, sans-serif",
        },
        style_cell={
            "textAlign": "left",
            "padding": "8px",
            "border": "1px solid #e4ebf0",
            "whiteSpace": "normal",
            "height": "auto",
            "fontSize": "13px",
            "fontFamily": "Manrope, Segoe UI, Arial, sans-serif",
            "maxWidth": "520px",
        },
        style_data_conditional=base_style,
    )


def _build_layout(progress_files: list[Path], initial_progress: str) -> dbc.Container:
    progress_options = [{"label": p.name, "value": str(p)} for p in progress_files]

    return dbc.Container(
        fluid=True,
        className="main-shell",
        children=[
            dcc.Store(id="bundle-store"),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.H2("RAG Reviewer Dashboard", className="hero-title"),
                            html.P(
                                "Executive analytics view for model quality, web evidence reliability, and audit traceability.",
                                className="hero-subtitle",
                            ),
                        ],
                        className="hero-wrap",
                    ),
                    width=12,
                ),
                className="mt-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Progress File", className="filter-label"),
                                    dcc.Dropdown(
                                        id="progress-file",
                                        options=progress_options,
                                        value=initial_progress,
                                        clearable=False,
                                    ),
                                ]
                            ),
                            className="filter-card",
                        ),
                        md=4,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Split", className="filter-label"),
                                    dcc.Dropdown(id="split-filter", multi=True),
                                ]
                            ),
                            className="filter-card",
                        ),
                        md=2,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Model", className="filter-label"),
                                    dcc.Dropdown(id="model-filter", multi=True),
                                ]
                            ),
                            className="filter-card",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div("Pipeline", className="filter-label"),
                                    dcc.Dropdown(id="pipeline-filter", multi=True),
                                ]
                            ),
                            className="filter-card",
                        ),
                        md=3,
                    ),
                ],
                className="g-3 mt-1",
            ),
            dbc.Row(
                [
                    dbc.Col(html.Div(id="kpi-1"), md=2),
                    dbc.Col(html.Div(id="kpi-2"), md=2),
                    dbc.Col(html.Div(id="kpi-3"), md=2),
                    dbc.Col(html.Div(id="kpi-4"), md=2),
                    dbc.Col(html.Div(id="kpi-5"), md=2),
                    dbc.Col(html.Div(id="kpi-6"), md=2),
                ],
                className="g-3 mt-1",
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(id="insight-banner", className="insight-banner"),
                    width=12,
                ),
                className="mt-2",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Tabs(
                        id="tabs",
                        active_tab="tab-overview",
                        children=[
                            dbc.Tab(
                                label="Overview",
                                tab_id="tab-overview",
                                children=[
                                    dbc.Row(
                                        [
                                            dbc.Col(dcc.Graph(id="perf-chart"), md=8),
                                            dbc.Col(dcc.Graph(id="heatmap-chart"), md=4),
                                        ],
                                        className="g-3 mt-2",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(dcc.Graph(id="score-dist-chart"), md=8),
                                            dbc.Col(dcc.Graph(id="flag-chart"), md=4),
                                        ],
                                        className="g-3 mt-1",
                                    ),
                                    dbc.Row(
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.H5("Run Summary", className="card-title"),
                                                        _table(
                                                            [
                                                                "split",
                                                                "model",
                                                                "pipeline",
                                                                "questions",
                                                                "mean_final_score",
                                                                "report",
                                                            ],
                                                            "run-table",
                                                            page_size=8,
                                                            style_data_conditional=[
                                                                {
                                                                    "if": {
                                                                        "column_id": "mean_final_score",
                                                                        "filter_query": "{mean_final_score} >= 0.6",
                                                                    },
                                                                    "backgroundColor": "#e6f7ef",
                                                                    "color": "#0e5f42",
                                                                    "fontWeight": "700",
                                                                },
                                                                {
                                                                    "if": {
                                                                        "column_id": "mean_final_score",
                                                                        "filter_query": "{mean_final_score} < 0.35",
                                                                    },
                                                                    "backgroundColor": "#fff1f0",
                                                                    "color": "#a13b37",
                                                                    "fontWeight": "700",
                                                                },
                                                            ],
                                                        ),
                                                    ]
                                                ),
                                                className="content-card",
                                            ),
                                            width=12,
                                        ),
                                        className="mt-1",
                                    ),
                                ],
                            ),
                            dbc.Tab(
                                label="Question Explorer",
                                tab_id="tab-question",
                                children=[
                                    dbc.Row(
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.H5("Question Outcomes", className="card-title"),
                                                        _table(
                                                            [
                                                                "Q_ID",
                                                                "Run_Model",
                                                                "Run_Pipeline",
                                                                "Question",
                                                                "Final_Score",
                                                                "Validation_Flags",
                                                                "Validation_Reason",
                                                            ],
                                                            "results-table",
                                                            page_size=10,
                                                            style_data_conditional=[
                                                                {
                                                                    "if": {
                                                                        "column_id": "Final_Score",
                                                                        "filter_query": "{Final_Score} >= 0.6",
                                                                    },
                                                                    "backgroundColor": "#e6f7ef",
                                                                    "color": "#0e5f42",
                                                                    "fontWeight": "700",
                                                                },
                                                                {
                                                                    "if": {
                                                                        "column_id": "Final_Score",
                                                                        "filter_query": "{Final_Score} < 0.35",
                                                                    },
                                                                    "backgroundColor": "#fff1f0",
                                                                    "color": "#a13b37",
                                                                    "fontWeight": "700",
                                                                },
                                                                {
                                                                    "if": {
                                                                        "column_id": "Validation_Flags",
                                                                        "filter_query": '{Validation_Flags} = "FLAGGED"',
                                                                    },
                                                                    "backgroundColor": "#fff5df",
                                                                    "color": "#8a5b00",
                                                                    "fontWeight": "700",
                                                                },
                                                            ],
                                                        ),
                                                    ]
                                                ),
                                                className="content-card",
                                            )
                                        ),
                                        className="mt-2",
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            html.Div("Deep Dive Row", className="filter-label"),
                                                            dcc.Dropdown(id="deep-row"),
                                                            html.H6("Evaluation Metrics", className="mt-3"),
                                                            _table(["Metric", "Score"], "metric-table", page_size=10),
                                                        ]
                                                    ),
                                                    className="content-card",
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            html.H6("Prompt Inputs", className="mb-2"),
                                                            html.Div(id="deep-question", className="text-block"),
                                                            html.Div(id="deep-kb", className="text-block mt-2"),
                                                            html.Div(id="deep-web", className="text-block mt-2"),
                                                            html.Div(id="deep-query", className="text-inline mt-2"),
                                                        ]
                                                    ),
                                                    className="content-card",
                                                ),
                                                md=4,
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        [
                                                            html.H6("Output & Validation", className="mb-2"),
                                                            html.Div(id="deep-golden", className="text-block"),
                                                            html.Div(id="deep-answer", className="text-block mt-2"),
                                                            html.Div(id="deep-flags", className="text-inline mt-2"),
                                                            html.Div(id="deep-reason", className="text-block mt-2"),
                                                        ]
                                                    ),
                                                    className="content-card",
                                                ),
                                                md=4,
                                            ),
                                        ],
                                        className="g-3 mt-1",
                                    ),
                                ],
                            ),
                            dbc.Tab(
                                label="Web Evidence",
                                tab_id="tab-web",
                                children=[
                                    dbc.Row(
                                        [
                                            dbc.Col(dcc.Graph(id="web-decision-chart"), md=6),
                                            dbc.Col(dcc.Graph(id="web-domain-chart"), md=6),
                                        ],
                                        className="g-3 mt-2",
                                    ),
                                    dbc.Row(
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.H5("Web Validation Records", className="card-title"),
                                                        _table(
                                                            [
                                                                "Q_ID",
                                                                "Run_Model",
                                                                "Run_Pipeline",
                                                                "Source_Domain",
                                                                "Decision",
                                                                "Final_Score",
                                                                "S1_Keyword",
                                                                "S2_Semantic",
                                                                "S3_LLM",
                                                                "S3_Partial_Relevance",
                                                                "URL",
                                                            ],
                                                            "web-table",
                                                            page_size=10,
                                                            style_data_conditional=[
                                                                {
                                                                    "if": {
                                                                        "column_id": "Decision",
                                                                        "filter_query": '{Decision} contains "ACCEPTED"',
                                                                    },
                                                                    "backgroundColor": "#e9f8f1",
                                                                    "color": "#0f6145",
                                                                },
                                                                {
                                                                    "if": {
                                                                        "column_id": "Decision",
                                                                        "filter_query": '{Decision} contains "REJECTED"',
                                                                    },
                                                                    "backgroundColor": "#fff1f0",
                                                                    "color": "#a13b37",
                                                                },
                                                            ],
                                                        ),
                                                    ]
                                                ),
                                                className="content-card",
                                            )
                                        ),
                                        className="mt-1",
                                    ),
                                ],
                            ),
                            dbc.Tab(
                                label="Audit Trail",
                                tab_id="tab-proof",
                                children=[
                                    dbc.Row(
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody(
                                                    [
                                                        html.H5("Proof Log", className="card-title"),
                                                        _table(
                                                            [
                                                                "Timestamp",
                                                                "Question_ID",
                                                                "Pipeline",
                                                                "Source_Domain",
                                                                "Decision",
                                                                "Final_Score",
                                                                "Reason",
                                                                "URL",
                                                            ],
                                                            "proof-table",
                                                            page_size=12,
                                                        ),
                                                    ]
                                                ),
                                                className="content-card",
                                            )
                                        ),
                                        className="mt-2",
                                    )
                                ],
                            ),
                        ],
                    ),
                    width=12,
                ),
                className="mt-2",
            ),
        ],
    )


def _to_options(values: list[str]) -> list[dict[str, str]]:
    return [{"label": v, "value": v} for v in values]


def _normalize_filters(
    selected: list[str] | None,
    options: list[str],
) -> list[str]:
    if not options:
        return []
    if not selected:
        return options
    keep = [x for x in selected if x in options]
    return keep or options


def _prepare_bundle(progress_file: str | None) -> dict[str, Any]:
    if not progress_file:
        return {"runs": [], "results": [], "web": [], "proof": [], "meta": {}}
    p = Path(progress_file)
    if not p.exists():
        return {"runs": [], "results": [], "web": [], "proof": [], "meta": {}}

    progress_data = json.loads(p.read_text(encoding="utf-8"))
    run_started_utc = _parse_iso_utc(progress_data.get("run_started_utc", ""))
    runs_df, results_df, web_df = _build_run_data(progress_data)

    question_ids: set[str] = set()
    if not results_df.empty and "Q_ID" in results_df.columns:
        question_ids = {str(x).strip() for x in results_df["Q_ID"].tolist() if str(x).strip()}
    pipeline_ids = {
        f"{str(row.get('model', '')).strip()}_{str(row.get('pipeline', '')).strip()}"
        for row in progress_data.get("runs", [])
    }
    proof_df = _load_proof_rows(
        run_started_utc=run_started_utc,
        pipeline_filters=pipeline_ids,
        question_filters=question_ids,
    )

    return {
        "runs": _serialize_df(runs_df),
        "results": _serialize_df(results_df),
        "web": _serialize_df(web_df),
        "proof": _serialize_df(proof_df),
        "meta": {
            "comparison_report": progress_data.get("comparison_report", ""),
            "run_started_utc": progress_data.get("run_started_utc", ""),
            "progress_file": str(p),
        },
    }


def _filter_data(
    bundle: dict[str, Any],
    splits: list[str],
    models: list[str],
    pipelines: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.DataFrame(bundle.get("runs", []))
    results = pd.DataFrame(bundle.get("results", []))
    web = pd.DataFrame(bundle.get("web", []))
    proof = pd.DataFrame(bundle.get("proof", []))

    if runs.empty:
        return runs, results, web, proof

    if "split" not in runs.columns:
        runs["split"] = "test"

    run_mask = (
        runs["split"].astype(str).isin(splits)
        & runs["model"].astype(str).isin(models)
        & runs["pipeline"].astype(str).isin(pipelines)
    )
    fruns = runs[run_mask].copy()

    if not results.empty:
        rmask = (
            results["Run_Split"].astype(str).isin(splits)
            & results["Run_Model"].astype(str).isin(models)
            & results["Run_Pipeline"].astype(str).isin(pipelines)
        )
        fresults = results[rmask].copy()
    else:
        fresults = results

    if not web.empty:
        wmask = (
            web["Run_Split"].astype(str).isin(splits)
            & web["Run_Model"].astype(str).isin(models)
            & web["Run_Pipeline"].astype(str).isin(pipelines)
        )
        fweb = web[wmask].copy()
    else:
        fweb = web

    if not proof.empty:
        valid_pipeline_keys = {
            f"{str(m).strip()}_{str(p).strip()}" for m in models for p in pipelines
        }
        pmask = proof["Pipeline"].astype(str).isin(valid_pipeline_keys)
        fproof = proof[pmask].copy()
    else:
        fproof = proof

    return fruns.reset_index(drop=True), fresults.reset_index(drop=True), fweb.reset_index(drop=True), fproof.reset_index(drop=True)


def _deep_row_options(results_df: pd.DataFrame) -> list[dict[str, str]]:
    if results_df.empty:
        return []
    qcol = _first_existing_column(results_df, ["Question", "question"])
    idcol = _first_existing_column(results_df, ["Q_ID", "q_id"])
    out: list[dict[str, str]] = []
    for idx, row in results_df.reset_index(drop=True).iterrows():
        qid = str(row.get(idcol, "")).strip() if idcol else str(idx + 1)
        model = str(row.get("Run_Model", "")).strip()
        pipe = str(row.get("Run_Pipeline", "")).strip()
        qtxt = str(row.get(qcol, "") if qcol else "").replace("\n", " ").strip()
        if len(qtxt) > 90:
            qtxt = qtxt[:90].rstrip() + " ..."
        label = f"Q{qid} | {model}/{pipe} | {qtxt}"
        out.append({"label": label, "value": str(idx)})
    return out


def create_app(progress_override: str | None = None) -> Dash:
    progress_files = _list_progress_files()
    initial = progress_override or (str(progress_files[0]) if progress_files else "")

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        assets_folder=str(ASSETS_DIR),
        suppress_callback_exceptions=True,
        title="RAG Reviewer Dashboard",
    )
    app.layout = _build_layout(progress_files, initial)

    @app.callback(
        Output("bundle-store", "data"),
        Output("split-filter", "options"),
        Output("split-filter", "value"),
        Output("model-filter", "options"),
        Output("model-filter", "value"),
        Output("pipeline-filter", "options"),
        Output("pipeline-filter", "value"),
        Input("progress-file", "value"),
        State("split-filter", "value"),
        State("model-filter", "value"),
        State("pipeline-filter", "value"),
    )
    def refresh_bundle(
        progress_file: str | None,
        selected_splits: list[str] | None,
        selected_models: list[str] | None,
        selected_pipelines: list[str] | None,
    ):
        bundle = _prepare_bundle(progress_file)
        runs = pd.DataFrame(bundle.get("runs", []))
        if runs.empty:
            return bundle, [], [], [], [], [], []

        split_vals = sorted(set(runs.get("split", pd.Series(["test"])).astype(str).tolist()))
        model_vals = sorted(set(runs.get("model", pd.Series(dtype=str)).astype(str).tolist()))
        pipeline_vals = sorted(set(runs.get("pipeline", pd.Series(dtype=str)).astype(str).tolist()))

        final_splits = _normalize_filters(selected_splits, split_vals)
        final_models = _normalize_filters(selected_models, model_vals)
        final_pipelines = _normalize_filters(selected_pipelines, pipeline_vals)

        return (
            bundle,
            _to_options(split_vals),
            final_splits,
            _to_options(model_vals),
            final_models,
            _to_options(pipeline_vals),
            final_pipelines,
        )

    @app.callback(
        Output("kpi-1", "children"),
        Output("kpi-2", "children"),
        Output("kpi-3", "children"),
        Output("kpi-4", "children"),
        Output("kpi-5", "children"),
        Output("kpi-6", "children"),
        Output("insight-banner", "children"),
        Output("perf-chart", "figure"),
        Output("heatmap-chart", "figure"),
        Output("score-dist-chart", "figure"),
        Output("flag-chart", "figure"),
        Output("run-table", "data"),
        Output("results-table", "data"),
        Output("web-decision-chart", "figure"),
        Output("web-domain-chart", "figure"),
        Output("web-table", "data"),
        Output("proof-table", "data"),
        Output("deep-row", "options"),
        Output("deep-row", "value"),
        Input("bundle-store", "data"),
        Input("split-filter", "value"),
        Input("model-filter", "value"),
        Input("pipeline-filter", "value"),
    )
    def render_dashboard(
        bundle: dict[str, Any] | None,
        splits: list[str] | None,
        models: list[str] | None,
        pipelines: list[str] | None,
    ):
        bundle = bundle or {"runs": [], "results": [], "web": [], "proof": []}
        runs = pd.DataFrame(bundle.get("runs", []))
        if runs.empty:
            empty = _empty_fig("No Data")
            return (
                _make_kpi("Run Rows", "0"),
                _make_kpi("Questions", "0"),
                _make_kpi("Mean Score", "N/A"),
                _make_kpi("Web Acceptance", "N/A"),
                _make_kpi("Flag Rate", "N/A"),
                _make_kpi("Top Pipeline", "N/A"),
                html.Div("Load a progress file to see executive insights.", className="insight-empty"),
                empty,
                empty,
                empty,
                empty,
                [],
                [],
                empty,
                empty,
                [],
                [],
                [],
                None,
            )

        split_vals = sorted(set(runs.get("split", pd.Series(["test"])).astype(str).tolist()))
        model_vals = sorted(set(runs.get("model", pd.Series(dtype=str)).astype(str).tolist()))
        pipeline_vals = sorted(set(runs.get("pipeline", pd.Series(dtype=str)).astype(str).tolist()))
        fsplits = _normalize_filters(splits, split_vals)
        fmodels = _normalize_filters(models, model_vals)
        fpipes = _normalize_filters(pipelines, pipeline_vals)

        fruns, fresults, fweb, fproof = _filter_data(bundle, fsplits, fmodels, fpipes)

        total_questions = int(fruns["questions"].sum()) if "questions" in fruns.columns and not fruns.empty else 0
        mean_score = (
            float(fruns["mean_final_score"].mean())
            if "mean_final_score" in fruns.columns and not fruns.empty
            else np.nan
        )

        acceptance_ratio: float | None = None
        if not fweb.empty and "Accepted" in fweb.columns:
            accepted = int(fweb["Accepted"].fillna(False).astype(bool).sum())
            rejected = int((~fweb["Accepted"].fillna(False).astype(bool)).sum())
            acceptance_ratio = accepted / max(accepted + rejected, 1)
            acceptance = f"{acceptance_ratio * 100:.1f}%"
        else:
            acceptance = "N/A"

        flag_ratio: float | None = None
        if not fresults.empty and "Validation_Flags" in fresults.columns:
            flagged = int((fresults["Validation_Flags"].astype(str) == "FLAGGED").sum())
            flag_ratio = flagged / max(len(fresults), 1)
            flag_rate = f"{flag_ratio * 100:.1f}%"
        else:
            flag_rate = "N/A"

        context_recall_mean: float | None = None
        answer_correctness_mean: float | None = None
        if not fresults.empty:
            if "Context_Recall" in fresults.columns:
                context_recall_mean = float(
                    pd.to_numeric(fresults["Context_Recall"], errors="coerce").dropna().mean()
                )
            if "Answer_Correctness" in fresults.columns:
                answer_correctness_mean = float(
                    pd.to_numeric(fresults["Answer_Correctness"], errors="coerce").dropna().mean()
                )

        if not fruns.empty and "mean_final_score" in fruns.columns:
            perf = (
                fruns.groupby(["model", "pipeline"], as_index=False)["mean_final_score"]
                .mean()
                .sort_values("mean_final_score", ascending=False)
            )
            if not perf.empty:
                best = perf.iloc[0]
                top_pipeline = f"{best['model']}/{best['pipeline']} ({best['mean_final_score']:.3f})"
            else:
                top_pipeline = "N/A"
            perf_fig = px.bar(
                perf,
                x="pipeline",
                y="mean_final_score",
                color="model",
                barmode="group",
                title="Pipeline Performance",
                labels={"mean_final_score": "Mean Final Score", "pipeline": "Pipeline"},
            )
            perf_fig.update_layout(
                template="plotly_white",
                height=340,
                margin=dict(l=10, r=10, t=45, b=10),
                legend_title_text="Model",
            )
            heat = perf.pivot(index="model", columns="pipeline", values="mean_final_score").fillna(0.0)
            heat_fig = px.imshow(
                heat,
                text_auto=".3f",
                aspect="auto",
                color_continuous_scale="YlGnBu",
                title="Model vs Pipeline Score Grid",
            )
            heat_fig.update_layout(template="plotly_white", height=340, margin=dict(l=10, r=10, t=45, b=10))
        else:
            top_pipeline = "N/A"
            perf_fig = _empty_fig("Pipeline Performance")
            heat_fig = _empty_fig("Model vs Pipeline Score Grid")

        if not fresults.empty and "Final_Score" in fresults.columns:
            score_dist = fresults.copy()
            if "Run_Pipeline" not in score_dist.columns:
                score_dist["Run_Pipeline"] = "pipeline"
            if "Run_Model" not in score_dist.columns:
                score_dist["Run_Model"] = "model"
            dist_fig = px.box(
                score_dist,
                x="Run_Pipeline",
                y="Final_Score",
                color="Run_Model",
                points="all",
                title="Question Score Distribution",
            )
            dist_fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=45, b=10))
        else:
            dist_fig = _empty_fig("Question Score Distribution")

        if not fresults.empty and "Validation_Flags" in fresults.columns:
            vf = fresults["Validation_Flags"].astype(str)
            flagged_count = int((vf == "FLAGGED").sum())
            clean_count = max(len(fresults) - flagged_count, 0)
            flag_df = pd.DataFrame(
                {"Status": ["Flagged", "Clean"], "Count": [flagged_count, clean_count]}
            )
            flag_fig = px.pie(
                flag_df,
                names="Status",
                values="Count",
                title="Validation Status",
                hole=0.45,
                color="Status",
                color_discrete_map={"Flagged": "#f59e0b", "Clean": "#10b981"},
            )
            flag_fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=45, b=10))
        else:
            flag_fig = _empty_fig("Validation Status")

        if not fweb.empty and "Decision" in fweb.columns:
            dec = fweb["Decision"].astype(str).value_counts().reset_index()
            dec.columns = ["Decision", "Count"]
            web_dec_fig = px.bar(dec, x="Decision", y="Count", title="Web Decision Distribution", color="Decision")
            web_dec_fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=45, b=10))
        else:
            web_dec_fig = _empty_fig("Web Decision Distribution")

        if not fweb.empty and "Source_Domain" in fweb.columns:
            dom = fweb["Source_Domain"].astype(str).value_counts().head(12).reset_index()
            dom.columns = ["Source_Domain", "Count"]
            web_dom_fig = px.bar(dom, x="Source_Domain", y="Count", title="Top Web Source Domains")
            web_dom_fig.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=45, b=10))
        else:
            web_dom_fig = _empty_fig("Top Web Source Domains")

        insight_bar = html.Div(
            [
                _insight_chip(
                    "Overall Quality",
                    f"{mean_score:.3f}" if not np.isnan(mean_score) else "N/A",
                    _tone_for_metric(
                        value=None if np.isnan(mean_score) else float(mean_score),
                        good_at_least=0.60,
                        warn_at_least=0.40,
                    ),
                ),
                _insight_chip(
                    "Retrieval Coverage",
                    f"{context_recall_mean:.3f}" if context_recall_mean is not None else "N/A",
                    _tone_for_metric(
                        value=context_recall_mean,
                        good_at_least=0.55,
                        warn_at_least=0.35,
                    ),
                ),
                _insight_chip(
                    "Answer Correctness",
                    f"{answer_correctness_mean:.3f}" if answer_correctness_mean is not None else "N/A",
                    _tone_for_metric(
                        value=answer_correctness_mean,
                        good_at_least=0.60,
                        warn_at_least=0.40,
                    ),
                ),
                _insight_chip(
                    "Web Acceptance",
                    acceptance,
                    _tone_for_metric(
                        value=acceptance_ratio,
                        good_at_least=0.55,
                        warn_at_least=0.35,
                    ),
                ),
                _insight_chip(
                    "Validation Risk",
                    flag_rate,
                    _tone_for_metric(
                        value=None if flag_ratio is None else 1.0 - flag_ratio,
                        good_at_least=0.70,
                        warn_at_least=0.45,
                    ),
                ),
                _insight_chip(
                    "Top Pipeline",
                    top_pipeline,
                    "insight-neutral",
                ),
            ],
            className="insight-grid",
        )

        deep_opts = _deep_row_options(fresults)
        deep_default = deep_opts[0]["value"] if deep_opts else None

        run_cols = ["split", "model", "pipeline", "questions", "mean_final_score", "report"]
        run_table = fruns[[c for c in run_cols if c in fruns.columns]].to_dict("records") if not fruns.empty else []

        result_cols = ["Q_ID", "Run_Model", "Run_Pipeline", "Question", "Final_Score", "Validation_Flags", "Validation_Reason"]
        result_table = fresults[[c for c in result_cols if c in fresults.columns]].to_dict("records") if not fresults.empty else []

        web_cols = ["Q_ID", "Run_Model", "Run_Pipeline", "Source_Domain", "Decision", "Final_Score", "S1_Keyword", "S2_Semantic", "S3_LLM", "S3_Partial_Relevance", "URL"]
        web_table = fweb[[c for c in web_cols if c in fweb.columns]].to_dict("records") if not fweb.empty else []

        proof_cols = ["Timestamp", "Question_ID", "Pipeline", "Source_Domain", "Decision", "Final_Score", "Reason", "URL"]
        proof_table = fproof[[c for c in proof_cols if c in fproof.columns]].to_dict("records") if not fproof.empty else []

        return (
            _make_kpi("Run Rows", str(len(fruns))),
            _make_kpi("Questions", str(total_questions)),
            _make_kpi("Mean Score", f"{mean_score:.4f}" if not np.isnan(mean_score) else "N/A"),
            _make_kpi("Web Acceptance", acceptance),
            _make_kpi("Flag Rate", flag_rate),
            _make_kpi("Top Pipeline", top_pipeline),
            insight_bar,
            perf_fig,
            heat_fig,
            dist_fig,
            flag_fig,
            run_table,
            result_table,
            web_dec_fig,
            web_dom_fig,
            web_table,
            proof_table,
            deep_opts,
            deep_default,
        )

    @app.callback(
        Output("metric-table", "data"),
        Output("deep-question", "children"),
        Output("deep-kb", "children"),
        Output("deep-web", "children"),
        Output("deep-query", "children"),
        Output("deep-golden", "children"),
        Output("deep-answer", "children"),
        Output("deep-flags", "children"),
        Output("deep-reason", "children"),
        Input("bundle-store", "data"),
        Input("split-filter", "value"),
        Input("model-filter", "value"),
        Input("pipeline-filter", "value"),
        Input("deep-row", "value"),
    )
    def render_deep_dive(
        bundle: dict[str, Any] | None,
        splits: list[str] | None,
        models: list[str] | None,
        pipelines: list[str] | None,
        deep_row: str | None,
    ):
        bundle = bundle or {"runs": [], "results": [], "web": [], "proof": []}
        runs = pd.DataFrame(bundle.get("runs", []))
        if runs.empty:
            return [], "No data", "", "", "", "", "", "", ""

        split_vals = sorted(set(runs.get("split", pd.Series(["test"])).astype(str).tolist()))
        model_vals = sorted(set(runs.get("model", pd.Series(dtype=str)).astype(str).tolist()))
        pipeline_vals = sorted(set(runs.get("pipeline", pd.Series(dtype=str)).astype(str).tolist()))
        fsplits = _normalize_filters(splits, split_vals)
        fmodels = _normalize_filters(models, model_vals)
        fpipes = _normalize_filters(pipelines, pipeline_vals)
        _, fresults, _, _ = _filter_data(bundle, fsplits, fmodels, fpipes)

        if fresults.empty:
            return [], "No rows for selected filters", "", "", "", "", "", "", ""

        idx = int(deep_row) if deep_row is not None and str(deep_row).isdigit() else 0
        if idx >= len(fresults):
            idx = 0
        row = fresults.reset_index(drop=True).iloc[idx]

        metric_candidates = [
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
        metrics = [{"Metric": m, "Score": row.get(m, None)} for m in metric_candidates if m in fresults.columns]

        q = str(row.get("Question", ""))
        kb = str(row.get("Context", ""))
        web = str(row.get("Web_Context", ""))
        sq = str(row.get("Web_Search_Query", ""))
        g = str(row.get("Golden", ""))
        a = str(row.get("Answer", ""))
        vf = str(row.get("Validation_Flags", ""))
        vr = str(row.get("Validation_Reason", ""))

        return (
            metrics,
            f"Question: {q}" if q else "Question: N/A",
            f"KB Context: {kb[:2200]}" if kb else "KB Context: N/A",
            f"Web Context: {web[:1800]}" if web else "Web Context: N/A",
            f"Web Search Query: {sq}" if sq else "Web Search Query: N/A",
            f"Golden: {g[:1400]}" if g else "Golden: N/A",
            f"Generated Answer: {a[:1800]}" if a else "Generated Answer: N/A",
            f"Validation Flags: {vf}" if vf else "Validation Flags: N/A",
            f"Validation Reason: {vr}" if vr else "Validation Reason: N/A",
        )

    return app


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reviewer dashboard (Dash)")
    parser.add_argument("--progress-file", default="", help="Optional progress file path override")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", default=8050, type=int, help="Port to bind")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    progress_file = args.progress_file.strip() or None
    if progress_file:
        p = Path(progress_file)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        progress_file = str(p)
    app = create_app(progress_override=progress_file)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
