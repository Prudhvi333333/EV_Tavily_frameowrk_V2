from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.web_crawler import RegistryMetadataFilter


def _write_registry(path: Path, review_rows: list[dict], rejected_rows: list[dict]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(review_rows).to_excel(writer, sheet_name="Review_Ready", index=False)
        pd.DataFrame(rejected_rows).to_excel(writer, sheet_name="Rejected_Documents", index=False)


def test_registry_filter_blocks_rejected_domain(tmp_path: Path) -> None:
    registry = tmp_path / "registry.xlsx"
    _write_registry(
        registry,
        review_rows=[],
        rejected_rows=[
            {
                "Source_Domain": "blocked.example.com",
                "Rejection_Category": "insufficient_grounded_evidence",
            }
        ],
    )

    cfg = {
        "crawler": {
            "metadata_filtering": {
                "enabled": True,
                "registry_path": str(registry),
                "block_rejected_domains": True,
            }
        }
    }
    f = RegistryMetadataFilter(cfg, logging.getLogger("test"), strict_mode=True)
    accepted, rejected = f.filter_search_results(
        [
            {"url": "https://blocked.example.com/a"},
            {"url": "https://allowed.example.org/b"},
        ]
    )

    assert [x["url"] for x in accepted] == ["https://allowed.example.org/b"]
    assert len(rejected) == 1
    assert rejected[0]["source_domain"] == "blocked.example.com"
    assert str(rejected[0]["reason"]).startswith("registry_domain_block:")


def test_registry_filter_allow_decision_overrides_rejected_sheet(tmp_path: Path) -> None:
    registry = tmp_path / "registry.xlsx"
    _write_registry(
        registry,
        review_rows=[
            {
                "Source_Domain": "trusted.example.com",
                "Final_Decision": "keep",
                "Metadata_Score": 80,
                "Credibility_Score": 77,
            }
        ],
        rejected_rows=[
            {
                "Source_Domain": "trusted.example.com",
                "Rejection_Category": "legacy_reject",
            }
        ],
    )

    cfg = {
        "crawler": {
            "metadata_filtering": {
                "enabled": True,
                "registry_path": str(registry),
                "block_rejected_domains": True,
                "allow_decisions": ["keep"],
                "block_decisions": ["discard", "reject", "rejected"],
            }
        }
    }
    f = RegistryMetadataFilter(cfg, logging.getLogger("test"), strict_mode=True)
    accepted, rejected = f.filter_search_results([{"url": "https://trusted.example.com/page"}])

    assert [x["url"] for x in accepted] == ["https://trusted.example.com/page"]
    assert rejected == []

