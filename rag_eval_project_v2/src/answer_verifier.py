"""Post-generation factual-claim verification pass.

Why: even with stricter prompts, the generator occasionally invents OEM links or
mis-states employment numbers when the context is dense. The verifier re-reads the
context with a fresh prompt that asks the model to (a) check each factual claim
against the context and (b) rewrite the answer keeping only supported claims.

The verifier runs at most one pass to keep cost bounded. If the model returns an
empty or malformed correction we fall back to the original answer.
"""

from __future__ import annotations

from typing import Any

from src.generator import OllamaGenerator, load_generation_config, strip_reasoning
from src.utils.logger import get_logger


_VERIFIER_SYSTEM = (
    "You are a factual checker. Your job is to compare a draft answer against a "
    "context block and return a corrected answer that contains ONLY claims supported "
    "by the context. You never add new facts, never speculate, and never copy "
    "reasoning traces. Stay grounded."
)

_VERIFIER_PROMPT = (
    "Below is a question, the context the original generator was given, and a draft "
    "answer. Identify each factual claim in the draft answer (entity names, OEM links, "
    "employment numbers, tier classifications, counts) and decide if the context "
    "directly supports it.\n\n"
    "Then output the FINAL corrected answer following these rules:\n"
    "- Keep only claims supported by the context.\n"
    "- If the draft is a numbered list, drop any entries not present in the context "
    "and re-number the remaining entries from 1; end with a single line 'Total: N'.\n"
    "- If after pruning there is nothing supported, return exactly: "
    "'Insufficient context to answer with confidence.'\n"
    "- Do NOT include the reasoning. Output ONLY the corrected answer text.\n\n"
    "Question:\n{question}\n\n"
    "Context:\n{context}\n\n"
    "Draft answer:\n{draft}\n\n"
    "Corrected answer:"
)


class AnswerVerifier:
    """Wraps a single Ollama-backed verification pass."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        verifier_cfg = dict(config.get("verifier", {}))
        self.enabled = bool(verifier_cfg.get("enabled", True))
        self.max_passes = max(0, int(verifier_cfg.get("max_passes", 1)))
        # Default to the primary generator model so we don't pay an extra model-load tax.
        default_model = config.get("models", {}).get("gemma", "gemma4:31b-chat")
        configured = str(verifier_cfg.get("model", "") or "").strip()
        self.model_name = configured or default_model
        self.logger = get_logger("verifier", config)
        self._llm: OllamaGenerator | None = None

    async def verify(self, question: str, context: str, draft: str) -> tuple[str, bool]:
        """Return (final_answer, was_modified). If verification is disabled or yields
        nothing usable, return the original draft unchanged."""
        if not self.enabled or self.max_passes <= 0:
            return draft, False
        if not str(draft or "").strip():
            return draft, False
        if not str(context or "").strip():
            # No context to verify against — leave the draft alone (this happens for
            # NO_RAG; verifier only adds value when there IS a context to check).
            return draft, False

        llm = self._ensure_llm()
        prompt = _VERIFIER_PROMPT.format(question=question, context=context, draft=draft)
        try:
            raw = await llm.generate(prompt=prompt, system=_VERIFIER_SYSTEM, temperature=0.0)
        except Exception as exc:
            self.logger.warning("Verifier call failed: %s; returning draft unchanged.", exc)
            return draft, False

        corrected = strip_reasoning(raw)
        if not corrected.strip():
            return draft, False

        # Heuristic: if the corrected answer is suspiciously short compared to the draft
        # AND the draft was a list, prefer the draft. The verifier sometimes over-prunes.
        draft_lines = [ln for ln in draft.splitlines() if ln.strip()]
        corrected_lines = [ln for ln in corrected.splitlines() if ln.strip()]
        looks_like_list = any(ln.lstrip().startswith(("1.", "1)", "- ", "* ")) for ln in draft_lines)
        if looks_like_list and len(corrected_lines) < max(2, len(draft_lines) // 4):
            self.logger.info(
                "Verifier shrank draft from %d to %d lines; keeping draft to avoid over-pruning.",
                len(draft_lines),
                len(corrected_lines),
            )
            return draft, False

        return corrected, corrected.strip() != draft.strip()

    def _ensure_llm(self) -> OllamaGenerator:
        if self._llm is None:
            gen_cfg = load_generation_config(self.config)
            self._llm = OllamaGenerator(
                self.model_name,
                strict=bool(self.config.get("runtime", {}).get("strict_mode", False)),
                timeout=gen_cfg["timeout"],
                max_retries=gen_cfg["max_retries"],
                retry_backoff_sec=gen_cfg["retry_backoff_sec"],
                keep_alive=str(self.config.get("runtime", {}).get("ollama_keep_alive", "0s")),
                options=dict(self.config.get("runtime", {}).get("ollama_options", {})),
            )
        return self._llm
