from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.web_crawler import RegistryMetadataFilter


def _write_registry_reference(path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {
                    "Source_Domain": "should-not-be-used.example.com",
                    "Final_Decision": "discard",
                    "Metadata_Score": 0.0,
                }
            ]
        ).to_excel(writer, sheet_name="Review_Ready", index=False)
        pd.DataFrame(
            [{"Source_Domain": "should-not-be-used.example.com", "Rejection_Category": "legacy"}]
        ).to_excel(writer, sheet_name="Rejected_Documents", index=False)
        pd.DataFrame(
            [{"Source_Domain": "should-not-be-used.example.com", "Acquisition_Status": "failed"}]
        ).to_excel(writer, sheet_name="Failed_Acquisitions", index=False)


def test_registry_workbook_is_schema_reference_only(tmp_path: Path) -> None:
    registry = tmp_path / "registry.xlsx"
    _write_registry_reference(registry)
    cfg = {
        "crawler": {
            "metadata_filtering": {
                "enabled": True,
                "registry_path": str(registry),
                "validate_registry_schema": True,
                "min_tavily_metadata_score": 0.0,
                "min_query_overlap": 0.0,
                "allowed_domains": [],
                "blocked_domains": [],
            }
        },
        "web_validator": {
            "domain_keywords": {"tier1": ["ev battery"], "tier2": ["georgia"]},
        },
    }
    f = RegistryMetadataFilter(cfg, logging.getLogger("test"), strict_mode=True)
    accepted, rejected = f.filter_search_results(
        [{"url": "https://should-not-be-used.example.com/a", "title": "EV battery update", "content": "Georgia plant"}],
        query_text="ev battery suppliers in georgia",
    )

    assert [x["url"] for x in accepted] == ["https://should-not-be-used.example.com/a"]
    assert rejected == []


def test_metadata_policy_uses_live_tavily_metadata_and_config_rules(tmp_path: Path) -> None:
    registry = tmp_path / "registry.xlsx"
    _write_registry_reference(registry)
    cfg = {
        "crawler": {
            "metadata_filtering": {
                "enabled": True,
                "registry_path": str(registry),
                "validate_registry_schema": True,
                "min_tavily_metadata_score": 0.15,
                "min_query_overlap": 0.05,
                "max_results_per_domain": 2,
                "allowed_domains": ["trusted.example.com"],
                "blocked_domains": ["blocked.example.com"],
            }
        },
        "web_validator": {
            "domain_keywords": {"tier1": ["ev battery"], "tier2": ["supply chain", "georgia"]},
        },
    }
    f = RegistryMetadataFilter(cfg, logging.getLogger("test"), strict_mode=True)

    accepted, rejected = f.filter_search_results(
        [
            {"url": "https://blocked.example.com/a", "title": "EV battery", "content": "Georgia"},
            {"url": "https://noise.example.org/b", "title": "Sports", "content": "Cricket scores and weather"},
            {"url": "https://trusted.example.com/c", "title": "General market commentary", "content": "Noisy text"},
            {"url": "https://signal.example.org/d", "title": "EV battery supply chain in Georgia", "content": "Supplier update"},
        ],
        query_text="ev battery supply chain georgia suppliers",
    )

    accepted_urls = [x["url"] for x in accepted]
    assert "https://trusted.example.com/c" in accepted_urls
    assert "https://signal.example.org/d" in accepted_urls
    assert "https://blocked.example.com/a" not in accepted_urls
    assert "https://noise.example.org/b" not in accepted_urls
    reasons = " ".join(str(r.get("reason", "")) for r in rejected)
    assert "policy_blocked_domain" in reasons
    assert "policy_score_below_threshold" in reasons or "policy_query_overlap_below_threshold" in reasons

