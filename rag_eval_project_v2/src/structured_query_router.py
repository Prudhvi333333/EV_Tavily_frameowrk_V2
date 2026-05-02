from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.generator import OllamaGenerator, load_generation_config
from src.retriever import RetrievedDoc
from src.utils.config_loader import resolve_path
from src.utils.logger import get_logger


_KB_TEXT_FIELDS = [
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

_LIST_TRIGGERS = (
    "list ",
    "list me",
    "which ",
    "what are ",
    "show all",
    "show the",
    "name the",
    "name all",
    "give me",
    "enumerate",
    "all suppliers",
    "all companies",
    "all firms",
    "all manufacturers",
    "all areas",
    "all plants",
    "all facilities",
    "all tier",
    "supplier network",
    "full network",
    "linked to",
    "tied to",
)

_COUNT_TRIGGERS = (
    "how many",
    "count of",
    "number of",
    "total number",
    "total count",
)

_FILTER_SPEC_SCHEMA_HINT = """
Return ONLY a single JSON object with these keys (use null when not applicable):
{
  "intent": "list" | "count" | "filter",
  "location": string | null,                     // e.g. "Georgia"
  "min_employment": integer | null,              // numeric threshold; null if not specified
  "max_employment": integer | null,
  "facility_type": string | null,                // e.g. "Manufacturing Plant", "R&D", "Headquarters"
  "ev_relevance": "yes" | "no" | "indirect" | null,
  "oem": string | null,                          // e.g. "Rivian", "Hyundai", "Kia", "Ford", "GM"
  "tier": "Tier 1" | "Tier 2" | "Tier 3" | "Tier 1/2" | "Tier 2/3" | "OEM" | null,
  "category_keyword": string | null,             // free-form keyword to substring-match against Category
  "role_keyword": string | null                  // free-form keyword to substring-match against EV Supply Chain Role
}
""".strip()


@dataclass
class RouteResult:
    used_router: bool = False
    intent_kind: str = ""             # "list" | "count" | "filter" | ""
    filter_spec: dict[str, Any] = field(default_factory=dict)
    matched_rows: list[dict[str, Any]] = field(default_factory=list)
    kb_context: str = ""
    pseudo_docs: list[RetrievedDoc] = field(default_factory=list)
    fallback_reason: str = ""         # reason routing was skipped or failed


class StructuredQueryRouter:
    """Detect structured (list/count/filter) queries and answer them with a deterministic
    pandas filter on the KB master spreadsheet rather than vector retrieval.

    Why: vector search over 205 flattened rows misses on count/aggregate questions
    ("Georgia companies with >1000 employees"). The KB is a structured table — treat
    structured queries as table queries.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger("structured_router", config)
        router_cfg = dict(config.get("structured_router", {}))
        self.enabled = bool(router_cfg.get("enabled", True))
        self.max_rows = int(router_cfg.get("max_rows_in_context", 20))
        self.fallback_on_parse_error = bool(router_cfg.get("fallback_on_parse_error", True))
        llm_model = str(router_cfg.get("llm_model", "")).strip()
        # Default to the evaluator's judge model — it's already loaded and cheap (llama3:8b).
        if not llm_model:
            llm_model = str(config.get("evaluation", {}).get("judge", {}).get("model", "llama3:8b"))
        self.llm_model = llm_model
        judge_cfg = dict(config.get("evaluation", {}).get("judge", {}))
        self._llm_base_url = str(judge_cfg.get("ollama_base_url", "")).strip() or None
        self._llm_api_key_env = str(judge_cfg.get("api_key_env", "OLLAMA_API_KEY"))
        self._llm: OllamaGenerator | None = None

        kb_path = resolve_path(config, config["paths"]["kb_input"])
        # Mirror kb_loader: fall back to kb_master.xlsx if the enriched file is missing.
        if not kb_path.exists():
            fallback = kb_path.parent / "kb_master.xlsx"
            if fallback.exists():
                kb_path = fallback
            else:
                raise FileNotFoundError(
                    f"Router KB file not found at {kb_path} and fallback {fallback} also missing."
                )
        self.kb_df = pd.read_excel(kb_path)
        self._normalize_kb()

    # ----- public API -----------------------------------------------------

    async def route(self, question: str, intent: dict[str, Any] | None = None) -> RouteResult:
        if not self.enabled:
            return RouteResult(fallback_reason="router_disabled")

        kind = self._heuristic_kind(question, intent or {})
        if not kind:
            return RouteResult(fallback_reason="no_structured_trigger")

        try:
            spec = await self._extract_filter_spec(question, kind)
        except Exception as exc:
            self.logger.warning("Router LLM extraction failed: %s", exc)
            if self.fallback_on_parse_error:
                return RouteResult(fallback_reason=f"llm_error:{type(exc).__name__}")
            raise

        if not spec:
            return RouteResult(fallback_reason="empty_spec")

        rows_df = self._apply_filters(spec)
        if rows_df.empty:
            # Empty match is still routable — the answer is "0/none" with [STRUCTURED_KB]
            # context that explicitly states the filters that returned nothing. This
            # prevents the generator from hallucinating in the *opposite* direction.
            context = self._empty_context(spec)
            return RouteResult(
                used_router=True,
                intent_kind=kind,
                filter_spec=spec,
                matched_rows=[],
                kb_context=context,
                pseudo_docs=[],
            )

        capped = rows_df.head(self.max_rows)
        rows_payload = capped.to_dict(orient="records")
        context = self._rows_to_context(rows_payload, total=len(rows_df), spec=spec, kind=kind)
        pseudo_docs = self._rows_to_pseudo_docs(capped)

        return RouteResult(
            used_router=True,
            intent_kind=kind,
            filter_spec=spec,
            matched_rows=rows_payload,
            kb_context=context,
            pseudo_docs=pseudo_docs,
        )

    # ----- heuristic gate -------------------------------------------------

    @staticmethod
    def _heuristic_kind(question: str, intent: dict[str, Any]) -> str:
        q = (question or "").casefold()
        if any(t in q for t in _COUNT_TRIGGERS):
            return "count"
        if any(t in q for t in _LIST_TRIGGERS):
            return "list"
        if intent.get("list_query"):
            return "list"
        if intent.get("multi_hop") and ("how many" in q or "list" in q or "all " in q):
            return "list"
        return ""

    # ----- LLM filter extraction -----------------------------------------

    async def _extract_filter_spec(self, question: str, kind: str) -> dict[str, Any]:
        llm = self._ensure_llm()
        prompt = (
            "You convert questions about Georgia's EV automotive supply chain into a "
            "structured filter spec for a KB query. The KB columns include Company, "
            "Category (Tier 1/Tier 2/Tier 2-3/OEM/OEM Supply Chain), Updated Location, "
            "Primary Facility Type (Manufacturing Plant, R&D, Headquarters, etc.), "
            "EV Supply Chain Role (Battery Cell, Battery Pack, Materials, Thermal "
            "Management, Power Electronics, Vehicle Assembly, Charging Infrastructure, "
            "General Automotive, etc.), Primary OEMs (Rivian, Hyundai, Kia, Ford, GM, "
            "Multiple OEMs, etc.), Employment (integer headcount), and "
            "EV / Battery Relevant (yes/no/indirect).\n\n"
            f"{_FILTER_SPEC_SCHEMA_HINT}\n\n"
            f"Detected query kind: {kind}.\n"
            f"Question: {question}\n\n"
            "JSON:"
        )
        system = (
            "You output ONLY a single valid JSON object. No prose, no code fences, no comments. "
            "Use null for any field not explicitly present in the question. Be conservative — "
            "do NOT invent constraints that the question did not state."
        )
        raw = await llm.generate(prompt=prompt, system=system, temperature=0.0)
        spec = self._parse_json(raw)
        if not isinstance(spec, dict):
            return {}
        # Drop unknown keys defensively.
        allowed = {
            "intent", "location", "min_employment", "max_employment",
            "facility_type", "ev_relevance", "oem", "tier",
            "category_keyword", "role_keyword",
        }
        return {k: v for k, v in spec.items() if k in allowed}

    @staticmethod
    def _parse_json(raw: str) -> Any:
        if not raw:
            return None
        text = str(raw).strip()
        # Strip code fences if the model added any.
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
            if text.endswith("```"):
                text = text[: -3]
            text = text.strip()
        # Fall back to extracting the first {...} object.
        if not text.startswith("{"):
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                return None
            text = m.group(0)
        try:
            return json.loads(text)
        except Exception:
            return None

    # ----- pandas filter --------------------------------------------------

    def _apply_filters(self, spec: dict[str, Any]) -> pd.DataFrame:
        df = self.kb_df

        location = _clean_str(spec.get("location"))
        if location and not _is_implicit_state(location):
            # Every KB row is in Georgia; treating "Georgia"/"GA" as a real substring
            # filter would zero out the result set since the Updated Location column
            # stores city/county pairs (e.g., "Warrenton, Warren County") with no
            # state token. Skip the filter when it asks for the implicit state.
            df = df[df["__location_norm"].str.contains(re.escape(location.casefold()), na=False)]

        min_emp = _to_int(spec.get("min_employment"))
        if min_emp is not None:
            df = df[df["__employment_int"].fillna(-1) >= min_emp]

        max_emp = _to_int(spec.get("max_employment"))
        if max_emp is not None:
            df = df[df["__employment_int"].fillna(10**9) <= max_emp]

        facility = _clean_str(spec.get("facility_type"))
        if facility:
            df = df[df["__facility_norm"].str.contains(re.escape(facility.casefold()), na=False)]

        ev_rel = _clean_str(spec.get("ev_relevance"))
        if ev_rel:
            ev_rel_l = ev_rel.casefold()
            if ev_rel_l in {"yes", "true"}:
                df = df[df["__ev_norm"].isin(["yes"])]
            elif ev_rel_l in {"no", "false"}:
                df = df[df["__ev_norm"].isin(["no"])]
            elif ev_rel_l == "indirect":
                df = df[df["__ev_norm"].isin(["indirect"])]

        oem = _clean_str(spec.get("oem"))
        if oem:
            # OEM mentions can appear in OEMs column ("Hyundai Kia Rivian"), Company
            # ("Rivian Automotive"), Industry Group, or Role ("Hyundai Transys ...").
            # Match across the searchable concat to capture full network linkage.
            oem_pat = re.escape(oem.casefold())
            df = df[df["__searchable_norm"].str.contains(oem_pat, na=False)]

        tier = _clean_str(spec.get("tier"))
        if tier:
            df = df[df["__category_norm"].str.contains(re.escape(tier.casefold()), na=False)]

        category_kw = _clean_str(spec.get("category_keyword"))
        if category_kw:
            df = df[df["__category_norm"].str.contains(re.escape(category_kw.casefold()), na=False)]

        role_kw = _clean_str(spec.get("role_keyword"))
        if role_kw:
            df = df[df["__role_norm"].str.contains(re.escape(role_kw.casefold()), na=False)]

        return df.reset_index(drop=True)

    # ----- context rendering ---------------------------------------------

    def _rows_to_context(
        self,
        rows: list[dict[str, Any]],
        total: int,
        spec: dict[str, Any],
        kind: str,
    ) -> str:
        header = (
            "[STRUCTURED_KB]\n"
            f"intent={kind} filter={json.dumps(_compact_spec(spec), separators=(',', ':'))} "
            f"matched={total} returned={len(rows)}\n"
            "These rows are the COMPLETE deterministic match against the KB. "
            "Do not invent or extrapolate beyond them."
        )
        blocks: list[str] = [header]
        for i, row in enumerate(rows, start=1):
            lines = [f"[ROW {i}]"]
            for field_ in _KB_TEXT_FIELDS:
                if field_ in row:
                    val = _clean_str(row.get(field_))
                    if val:
                        lines.append(f"{field_}: {val}")
            blocks.append("\n".join(lines))
        if total > len(rows):
            blocks.append(f"[NOTE] {total - len(rows)} additional row(s) matched but were truncated for context budget.")
        return "\n\n".join(blocks)

    def _empty_context(self, spec: dict[str, Any]) -> str:
        return (
            "[STRUCTURED_KB]\n"
            f"intent=filter filter={json.dumps(_compact_spec(spec), separators=(',', ':'))} "
            f"matched=0 returned=0\n"
            "No KB rows match the specified filters."
        )

    def _rows_to_pseudo_docs(self, df: pd.DataFrame) -> list[RetrievedDoc]:
        out: list[RetrievedDoc] = []
        for idx, row in df.iterrows():
            text = "\n".join(
                f"{f}: {_clean_str(row.get(f))}"
                for f in _KB_TEXT_FIELDS
                if _clean_str(row.get(f))
            )
            out.append(
                RetrievedDoc(
                    id=f"router_{idx + 1}",
                    text=text,
                    metadata={
                        "company": _clean_str(row.get("Company")),
                        "category": _clean_str(row.get("Category")),
                        "location": _clean_str(row.get("Updated Location")),
                        "role": _clean_str(row.get("EV Supply Chain Role")),
                        "source": "structured_router",
                    },
                    score=1.0,
                    semantic_score=1.0,
                    bm25_score=0.0,
                )
            )
        return out

    # ----- internals ------------------------------------------------------

    def _normalize_kb(self) -> None:
        df = self.kb_df.copy()
        df["__location_norm"] = df.get("Updated Location", "").astype(str).str.casefold()
        df["__facility_norm"] = df.get("Primary Facility Type", "").astype(str).str.casefold()
        df["__ev_norm"] = df.get("EV / Battery Relevant", "").astype(str).str.casefold().str.strip()
        df["__oem_norm"] = df.get("Primary OEMs", "").astype(str).str.casefold()
        df["__company_norm"] = df.get("Company", "").astype(str).str.casefold()
        df["__category_norm"] = df.get("Category", "").astype(str).str.casefold()
        df["__role_norm"] = df.get("EV Supply Chain Role", "").astype(str).str.casefold()
        df["__industry_norm"] = df.get("Industry Group", "").astype(str).str.casefold()
        df["__product_norm"] = df.get("Product / Service", "").astype(str).str.casefold()
        df["__employment_int"] = pd.to_numeric(df.get("Employment"), errors="coerce")
        # Single concatenated lowercase blob used for OEM-link searches: an OEM name
        # might appear in the company name (Rivian Automotive) or the role description
        # (Hyundai Transys Georgia Powertrain), not just the OEMs column.
        df["__searchable_norm"] = (
            df["__company_norm"].fillna("") + " | "
            + df["__oem_norm"].fillna("") + " | "
            + df["__industry_norm"].fillna("") + " | "
            + df["__role_norm"].fillna("") + " | "
            + df["__product_norm"].fillna("")
        )
        self.kb_df = df

    def _ensure_llm(self) -> OllamaGenerator:
        if self._llm is None:
            gen_cfg = load_generation_config(self.config)
            self._llm = OllamaGenerator(
                self.llm_model,
                base_url=self._llm_base_url,
                api_key_env=self._llm_api_key_env,
                strict=bool(self.config.get("runtime", {}).get("strict_mode", False)),
                timeout=gen_cfg["timeout"],
                max_retries=gen_cfg["max_retries"],
                retry_backoff_sec=gen_cfg["retry_backoff_sec"],
                keep_alive=str(self.config.get("runtime", {}).get("ollama_keep_alive", "0s")),
                options=dict(self.config.get("runtime", {}).get("ollama_options", {})),
            )
        return self._llm


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "null"}:
        return ""
    return text


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(float(value))
    except Exception:
        return None


def _compact_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in spec.items() if v not in (None, "", [], {})}


def _is_implicit_state(value: str) -> bool:
    return value.strip().casefold() in {"georgia", "ga", "ga.", "state of georgia"}
