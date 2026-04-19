from __future__ import annotations

import os
from enum import Enum
from typing import Any

import httpx


class PipelineMode(str, Enum):
    RAG = "rag"
    NO_RAG = "no_rag"
    RAG_PRETRAINED = "rag_pretrained"
    RAG_PRETRAINED_WEB = "rag_pretrained_web"


class OllamaGenerator:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        timeout: float = 120.0,
        strict: bool = True,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self.strict = strict
        self._checked = False
        self._available = False

    async def _check_availability(self) -> bool:
        if self._checked:
            return self._available
        self._checked = True
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=2.0, write=2.0, pool=2.0)) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                if resp.status_code != 200:
                    self._available = False
                    return False
                names = {m.get("name", "") for m in resp.json().get("models", [])}
                expected = {self.model, f"{self.model}:latest"}
                self._available = bool(names & expected)
        except Exception:
            self._available = False
        return self._available

    async def generate(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        if not await self._check_availability():
            message = (
                f"Ollama model '{self.model}' is unavailable on {self.base_url}. "
                "Run `ollama list` and ensure the model is installed."
            )
            if self.strict:
                raise RuntimeError(message)
            return ""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            timeout = httpx.Timeout(connect=5.0, read=self.timeout, write=20.0, pool=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            data = resp.json()
            return str(data.get("response", "")).strip()
        except Exception as e:
            if self.strict:
                raise RuntimeError(f"Ollama generation failed for model '{self.model}'.") from e
            return ""


class GeminiGenerator:
    def __init__(self, model: str, timeout: float = 120.0, strict: bool = True) -> None:
        self.model = model
        self.timeout = timeout
        self.strict = strict
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()

    async def generate(self, prompt: str, system: str = "") -> str:
        if not self.api_key:
            if self.strict:
                raise RuntimeError("GEMINI_API_KEY is not set for Gemini model usage.")
            return ""
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system or "You are a helpful assistant."}]},
            "contents": [{"parts": [{"text": prompt}]}],
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                if self.strict:
                    raise RuntimeError("Gemini returned no candidates.")
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            text = " ".join(str(p.get("text", "")).strip() for p in parts)
            if text.strip():
                return text.strip()
            if self.strict:
                raise RuntimeError("Gemini returned empty output.")
            return ""
        except Exception as e:
            if self.strict:
                raise RuntimeError("Gemini request failed.") from e
            return ""


def build_prompt(
    question: str,
    mode: PipelineMode,
    kb_context: str = "",
    web_context: str = "",
    few_shot_examples: str = "",
    prompt_mode: str = "chain_of_thought",
) -> tuple[str, str]:
    reasoning_line = {
        "standard": (
            "Use crisp factual language and avoid filler. If evidence is missing, say so explicitly."
        ),
        "chain_of_thought": (
            "Internally follow this checklist before answering: understand intent, map evidence, verify, answer. "
            "Do not output hidden reasoning; output only final answer with evidence references."
        ),
        "few_shot": (
            "Use the examples to match answer style and structure. Stay grounded and concise."
        ),
    }.get(prompt_mode, "Answer with grounded, concise statements.")

    examples_block = f"{few_shot_examples}\n\n" if few_shot_examples else ""

    if mode == PipelineMode.RAG:
        system = (
            "You are a data analyst for Georgia's EV supply chain.\n"
            "Answer ONLY from provided context.\n"
            "Never use external or pretrained knowledge.\n"
            "If answer is not in context, return exactly: This information is not available in the knowledge base.\n"
            f"{reasoning_line}"
        )
        user = (
            f"{examples_block}"
            "Context:\n"
            f"{kb_context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return system, user

    if mode == PipelineMode.NO_RAG:
        system = (
            "You are an automotive supply chain expert.\n"
            "Answer from pretrained knowledge only.\n"
            "Always provide a direct, best-effort answer to the question.\n"
            "Do not return refusal-style or uncertainty-only responses.\n"
            "For list questions, provide a structured list with company, role, and product/service.\n"
            "If some details are uncertain, still provide the most likely information and continue.\n"
            f"{reasoning_line}"
        )
        user = f"{examples_block}Question: {question}\nAnswer:"
        return system, user

    if mode == PipelineMode.RAG_PRETRAINED:
        system = (
            "You are an EV supply chain analyst.\n"
            "Use context as PRIMARY source.\n"
            "Use pretrained knowledge only when context is insufficient.\n"
            "Tag each factual sentence: [KB] or [PRETRAINED].\n"
            "If unknown, state uncertainty explicitly.\n"
            f"{reasoning_line}"
        )
        user = (
            f"{examples_block}"
            "Context:\n"
            f"{kb_context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return system, user

    system = (
        "You are an EV research analyst with three sources.\n"
        "Priority order: [KB] > [WEB] > [PRETRAINED].\n"
        "Tag each factual sentence with its source tag.\n"
        "When sources conflict, prioritize KB and mention the conflict.\n"
        "If web context is empty, continue with [KB]/[PRETRAINED] only.\n"
        f"{reasoning_line}"
    )
    user = (
        f"{examples_block}"
        "KB Context:\n"
        f"{kb_context}\n\n"
        "Web Results:\n"
        f"{web_context or 'WEB_UNAVAILABLE'}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return system, user


class ModelGenerator:
    def __init__(self, model_key: str, config: dict[str, Any], few_shot_builder: Any | None = None) -> None:
        self.model_key = model_key
        self.config = config
        self.few_shot_builder = few_shot_builder
        self.model_name = config["models"][model_key]
        self.prompt_mode = config.get("prompting", {}).get("mode", "chain_of_thought")
        self.strict_mode = bool(config.get("runtime", {}).get("strict_mode", False))
        if model_key in {"qwen", "gemma"}:
            self.client = OllamaGenerator(self.model_name, strict=self.strict_mode)
        elif model_key == "gemini":
            self.client = GeminiGenerator(self.model_name, strict=self.strict_mode)
        else:
            self.client = OllamaGenerator(self.model_name, strict=self.strict_mode)

    async def generate_with_mode(
        self,
        question: str,
        pipeline_mode: PipelineMode,
        kb_context: str = "",
        web_context: str = "",
    ) -> str:
        few_shot = ""
        if self.few_shot_builder is not None:
            n = int(self.config.get("prompting", {}).get("few_shot_examples", 2))
            few_shot = self.few_shot_builder.get_examples(question, pipeline_mode.value, n=n)
        system, user = build_prompt(
            question=question,
            mode=pipeline_mode,
            kb_context=kb_context,
            web_context=web_context,
            few_shot_examples=few_shot,
            prompt_mode=self.prompt_mode,
        )
        if isinstance(self.client, OllamaGenerator):
            return await self.client.generate(prompt=user, system=system)
        return await self.client.generate(prompt=user, system=system)
