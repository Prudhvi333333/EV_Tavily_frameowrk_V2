from __future__ import annotations

import asyncio
import os
from enum import Enum
from typing import Any

import httpx

from src.utils.ollama import resolve_ollama_base_url

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
        max_retries: int = 2,
        retry_backoff_sec: float = 2.0,
        keep_alive: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = resolve_ollama_base_url(base_url)
        self.timeout = timeout
        self.strict = strict
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_sec = max(0.1, float(retry_backoff_sec))
        self.keep_alive = str(keep_alive or os.getenv("OLLAMA_KEEP_ALIVE", "0s")).strip()
        self.options = dict(options or {})
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
            "options": {**self.options, "temperature": temperature},
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        last_error: Exception | None = None
        timeout = httpx.Timeout(connect=5.0, read=self.timeout, write=20.0, pool=5.0)
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(url, json=payload)
                    resp.raise_for_status()
                data = resp.json()
                return str(data.get("response", "")).strip()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else "unknown"
                body = ""
                try:
                    body = (e.response.text or "").strip()[:240] if e.response is not None else ""
                except Exception:
                    body = ""
                last_error = RuntimeError(
                    f"Ollama HTTP {status} for model '{self.model}' at {self.base_url}. Body: {body}"
                )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_backoff_sec * (attempt + 1))
                    continue
        if self.strict:
            raise RuntimeError(f"Ollama generation failed for model '{self.model}'.") from last_error
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


class OpenRouterGenerator:
    def __init__(
        self,
        model: str,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str | None = None,
        timeout: float = 120.0,
        strict: bool = True,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self.timeout = timeout
        self.strict = strict
        self.api_key = os.getenv(api_key_env, "").strip()

    async def generate(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        if not self.api_key:
            if self.strict:
                raise RuntimeError(f"{self.api_key_env} is not set for OpenRouter usage.")
            return ""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system or "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
            return (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        except Exception as e:
            if self.strict:
                raise RuntimeError(f"OpenRouter generation failed for model '{self.model}'.") from e
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
            "Use crisp factual language and avoid filler."
        ),
        "chain_of_thought": (
            "Think step-by-step internally and then provide only the final answer."
        ),
        "few_shot": (
            "Use the examples to match answer style and structure. Stay grounded and concise."
        ),
    }.get(prompt_mode, "Answer with grounded, concise statements.")

    examples_block = f"{few_shot_examples}\n\n" if few_shot_examples else ""

    if mode == PipelineMode.RAG:
        system = (
            "You are a data analyst for Georgia's EV supply chain.\n"
            "Use provided context as the main evidence source.\n"
            "If context is incomplete, still provide the best possible answer and clearly mark uncertain parts.\n"
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
            "For list questions, provide a structured list with company, role, and product/service.\n"
            "If some details are uncertain, provide the most likely answer and state uncertainty briefly.\n"
            f"{reasoning_line}"
        )
        user = f"{examples_block}Question: {question}\nAnswer:"
        return system, user

    if mode == PipelineMode.RAG_PRETRAINED:
        system = (
            "You are an EV supply chain analyst.\n"
            "Use context as the primary source and pretrained knowledge only to fill gaps.\n"
            "Do not fabricate facts not supported by context or clearly stated as background knowledge.\n"
            "When useful, mention whether a claim comes from context or general knowledge in natural language.\n"
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
        "Prioritize sources in this order: KB context, then web context, then pretrained knowledge.\n"
        "When sources conflict, prefer KB and mention the conflict briefly.\n"
        "If web context is empty, continue with KB and pretrained knowledge without fabricating web claims.\n"
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
        self.ollama_keep_alive = str(config.get("runtime", {}).get("ollama_keep_alive", "0s"))
        self.ollama_options = dict(config.get("runtime", {}).get("ollama_options", {}))
        if model_key in {"qwen", "gemma"}:
            self.client = OllamaGenerator(
                self.model_name,
                strict=self.strict_mode,
                keep_alive=self.ollama_keep_alive,
                options=self.ollama_options,
            )
        elif model_key == "gemini":
            self.client = GeminiGenerator(self.model_name, strict=self.strict_mode)
        else:
            self.client = OllamaGenerator(
                self.model_name,
                strict=self.strict_mode,
                keep_alive=self.ollama_keep_alive,
                options=self.ollama_options,
            )

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
