from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.config_loader import resolve_path


KB_TEXT_FIELDS = [
    "Company",
    "Category",
    "Industry Group",
    "Updated Location",
    "Address",
    "Primary Facility Type",
    "EV Supply Chain Role",
    "Primary OEMs",
    "Supplier or Affiliation Type",
    "Employment",
    "Product / Service",
    "EV / Battery Relevant",
    "Classification Method",
]


def _clean(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _row_to_text(row: pd.Series) -> str:
    pairs: list[str] = []
    for field in KB_TEXT_FIELDS:
        val = _clean(row.get(field, ""))
        if val:
            pairs.append(f"{field}: {val}")
    return "\n".join(pairs)


def load_kb(config: dict[str, Any]) -> list[dict[str, Any]]:
    kb_path = resolve_path(config, config["paths"]["kb_input"])
    df = pd.read_excel(kb_path)
    documents: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        text = _row_to_text(row)
        documents.append(
            {
                "id": f"kb_{idx + 1}",
                "text": text,
                "metadata": {
                    "company": _clean(row.get("Company", "")),
                    "category": _clean(row.get("Category", "")),
                    "location": _clean(row.get("Updated Location", "")),
                    "role": _clean(row.get("EV Supply Chain Role", "")),
                },
            }
        )
    return documents

