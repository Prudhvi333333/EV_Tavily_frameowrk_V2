"""One-shot script: generate a 2-3 sentence narrative summary per KB row.

For each row in data/kb/kb_master.xlsx, this script asks the configured generator
model (default: gemma4:31b-chat) to produce a short narrative covering facility
role, OEM relationships, tier classification, employment band, and EV relevance.
The summary is appended as a new column ``LLM_Description`` in
``data/kb/kb_master_enriched.xlsx`` so the original file stays untouched.

Why: vector retrieval over flat Excel rows underperforms on relationship and role
queries. A short LLM-written narrative gives the embedder + BM25 better signal on
who is connected to whom and what the facility actually does.

The script caches per-row results by a stable hash of the row text in
``data/kb/.enrich_cache.json`` so re-runs are idempotent and only fill in missing
or changed rows.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generator import OllamaGenerator, load_generation_config
from src.kb_loader import KB_TEXT_FIELDS
from src.utils.config_loader import load_config, resolve_path
from src.utils.logger import get_logger


_SYSTEM = (
    "You write concise factual summaries for an EV automotive supply chain knowledge "
    "base. Summaries must stay grounded in the provided fields; do not invent facts."
)

_PROMPT_TEMPLATE = (
    "Write a 2-3 sentence narrative description for the following supply chain entry. "
    "Cover, when present: the company's role in the EV supply chain, its OEM "
    "relationships, tier classification, employment band, and EV relevance. Be "
    "specific (use names, not generic terms) but do not add facts that aren't in the "
    "fields below.\n\n"
    "Fields:\n{fields}\n\n"
    "Description:"
)


def _row_text(row: pd.Series) -> str:
    pairs: list[str] = []
    for field in KB_TEXT_FIELDS:
        val = row.get(field, "")
        if pd.isna(val):
            continue
        s = str(val).strip()
        if s:
            pairs.append(f"{field}: {s}")
    return "\n".join(pairs)


def _row_hash(text: str, model_name: str) -> str:
    payload = f"{model_name}\n---\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


async def _generate_one(
    gen: OllamaGenerator,
    fields_block: str,
) -> str:
    prompt = _PROMPT_TEMPLATE.format(fields=fields_block)
    out = await gen.generate(prompt=prompt, system=_SYSTEM, temperature=0.0)
    return str(out or "").strip()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich kb_master.xlsx with LLM-written narratives.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--model-key",
        default="gemma",
        help="Key from config.models to use as the generator (default: gemma).",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Override input KB path (default: paths.kb_input from config).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override output enriched KB path (default: kb_master_enriched.xlsx alongside input).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke testing.",
    )
    parser.add_argument(
        "--cache",
        default=None,
        help="Override cache path (default: data/kb/.enrich_cache.json alongside input).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    logger = get_logger("enrich_kb", config)

    kb_path = Path(args.input) if args.input else resolve_path(config, config["paths"]["kb_input"])
    out_path = (
        Path(args.output)
        if args.output
        else kb_path.with_name("kb_master_enriched.xlsx")
    )
    cache_path = (
        Path(args.cache)
        if args.cache
        else kb_path.with_name(".enrich_cache.json")
    )

    model_name = config["models"][args.model_key]
    gen_cfg = load_generation_config(config)
    gen = OllamaGenerator(
        model_name,
        strict=bool(config.get("runtime", {}).get("strict_mode", False)),
        timeout=gen_cfg["timeout"],
        max_retries=gen_cfg["max_retries"],
        retry_backoff_sec=gen_cfg["retry_backoff_sec"],
        keep_alive=str(config.get("runtime", {}).get("ollama_keep_alive", "0s")),
        options=dict(config.get("runtime", {}).get("ollama_options", {})),
    )

    df = pd.read_excel(kb_path)
    if args.limit:
        df = df.head(args.limit).copy()

    cache: dict[str, str] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(cache, dict):
                cache = {}
        except Exception:
            cache = {}

    descriptions: list[str] = []
    new_count = 0
    for idx, row in df.iterrows():
        fields_block = _row_text(row)
        if not fields_block.strip():
            descriptions.append("")
            continue
        key = _row_hash(fields_block, model_name)
        cached = cache.get(key)
        if isinstance(cached, str) and cached.strip():
            descriptions.append(cached)
            continue
        try:
            text = await _generate_one(gen, fields_block)
        except Exception as exc:
            logger.warning("Row %s enrichment failed (%s); leaving empty.", idx, exc)
            text = ""
        descriptions.append(text)
        if text:
            cache[key] = text
            new_count += 1
            # Persist cache incrementally so a crash mid-run doesn't lose work.
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        if (idx + 1) % 10 == 0:
            logger.info("Enriched %s/%s rows (%s new).", idx + 1, len(df), new_count)

    df["LLM_Description"] = descriptions
    df.to_excel(out_path, index=False)
    logger.info("Wrote %s rows with LLM_Description to %s (%s newly generated).", len(df), out_path, new_count)


if __name__ == "__main__":
    asyncio.run(main())
