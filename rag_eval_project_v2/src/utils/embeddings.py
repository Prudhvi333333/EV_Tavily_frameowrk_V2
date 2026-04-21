from __future__ import annotations

import os
from typing import Any

import httpx
import numpy as np

from src.utils.ollama import resolve_ollama_base_url

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, local_files_only: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)

    def encode(
        self,
        texts: str | list[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = False,
    ) -> Any:
        arr = self.model.encode(
            [texts] if isinstance(texts, str) else texts,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )
        if convert_to_numpy:
            return arr
        if isinstance(texts, str):
            return arr[0].tolist()
        return arr.tolist()

    def encode_with_task(
        self,
        texts: str | list[str],
        task: str = "generic",
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = False,
    ) -> Any:
        # Sentence-transformers models generally do not require task prefixes.
        _ = task
        return self.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=convert_to_numpy,
        )


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        timeout: float = 180.0,
        batch_size: int = 32,
        keep_alive: str | None = None,
        use_instruction_prefixes: bool = False,
        query_prefix: str = "search_query:",
        document_prefix: str = "search_document:",
        max_retries: int = 2,
        retry_backoff_sec: float = 2.0,
    ) -> None:
        self.model = model
        self.base_url = resolve_ollama_base_url(base_url)
        self.timeout = timeout
        self.batch_size = max(int(batch_size), 1)
        self.keep_alive = str(keep_alive or os.getenv("OLLAMA_KEEP_ALIVE", "0s")).strip()
        self.use_instruction_prefixes = bool(use_instruction_prefixes)
        self.query_prefix = str(query_prefix).strip() or "search_query:"
        self.document_prefix = str(document_prefix).strip() or "search_document:"
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_sec = max(0.1, float(retry_backoff_sec))
        self._verified = False

    def encode(
        self,
        texts: str | list[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = False,
    ) -> Any:
        return self.encode_with_task(
            texts,
            task="generic",
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=convert_to_numpy,
        )

    def encode_with_task(
        self,
        texts: str | list[str],
        task: str = "generic",
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = False,
    ) -> Any:
        single = isinstance(texts, str)
        items = [str(texts)] if single else [str(x) for x in list(texts)]
        items = self._apply_task_prefix(items, task=task)
        if not items:
            arr = np.zeros((0, 0), dtype=np.float32)
            return arr if convert_to_numpy else []

        self._ensure_model_available()

        all_vecs: list[list[float]] = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i : i + self.batch_size]
            vecs = self._embed_batch(batch)
            all_vecs.extend(vecs)

        arr = np.array(all_vecs, dtype=np.float32)
        if normalize_embeddings and arr.size:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms

        if convert_to_numpy:
            return arr
        if single:
            return arr[0].tolist()
        return arr.tolist()

    def _apply_task_prefix(self, items: list[str], task: str) -> list[str]:
        if not self.use_instruction_prefixes:
            return items

        task_norm = str(task or "").strip().lower()
        if task_norm == "query":
            prefix = self.query_prefix
        elif task_norm == "document":
            prefix = self.document_prefix
        else:
            return items

        prefix_l = prefix.casefold()
        out: list[str] = []
        for text in items:
            original = str(text)
            stripped = original.lstrip()
            if stripped.casefold().startswith(prefix_l):
                out.append(original)
            else:
                out.append(f"{prefix} {original}".strip())
        return out

    def _ensure_model_available(self) -> None:
        if self._verified:
            return
        url = f"{self.base_url}/api/tags"
        timeout = httpx.Timeout(connect=4.0, read=10.0, write=10.0, pool=4.0)
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                names = {m.get("name", "") for m in resp.json().get("models", [])}
        except Exception as e:
            raise RuntimeError(
                f"Ollama is not reachable at {self.base_url}. Start Ollama before running."
            ) from e

        expected = {self.model, f"{self.model}:latest"}
        if not (names & expected):
            raise RuntimeError(
                f"Ollama embedding model '{self.model}' is missing. Run: ollama pull {self.model}"
            )
        self._verified = True

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        timeout = httpx.Timeout(connect=8.0, read=self.timeout, write=30.0, pool=8.0)
        payload = {"model": self.model, "input": batch}
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        url = f"{self.base_url}/api/embed"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 404:
                        return self._embed_batch_legacy(batch)
                    resp.raise_for_status()
                    data = resp.json()
                embeddings = data.get("embeddings")
                if not isinstance(embeddings, list) or not embeddings:
                    raise RuntimeError(f"Ollama embed API returned invalid response for model '{self.model}'.")
                return embeddings
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    wait_s = self.retry_backoff_sec * (attempt + 1)
                    # No async loop in this sync path; simple blocking wait is acceptable for retry.
                    import time

                    time.sleep(wait_s)
                    continue
        raise RuntimeError(
            f"Ollama embed API failed for model '{self.model}' after retries."
        ) from last_error

    def _embed_batch_legacy(self, batch: list[str]) -> list[list[float]]:
        timeout = httpx.Timeout(connect=5.0, read=self.timeout, write=20.0, pool=5.0)
        out: list[list[float]] = []
        with httpx.Client(timeout=timeout) as client:
            for text in batch:
                resp = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text,
                        **({"keep_alive": self.keep_alive} if self.keep_alive else {}),
                    },
                )
                resp.raise_for_status()
                emb = resp.json().get("embedding")
                if not isinstance(emb, list):
                    raise RuntimeError(
                        f"Ollama legacy embedding API returned invalid response for model '{self.model}'."
                    )
                out.append(emb)
        return out


def load_embedder_from_config(config: dict[str, Any]) -> Any:
    emb_cfg = config.get("embeddings", {})
    provider = str(emb_cfg.get("provider", "sentence_transformers")).strip().lower()
    model_name = emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    local_only = bool(emb_cfg.get("local_files_only", True))
    batch_size = int(emb_cfg.get("batch_size", 32))
    timeout = float(emb_cfg.get("timeout", 180))
    max_retries = int(emb_cfg.get("max_retries", 2))
    retry_backoff_sec = float(emb_cfg.get("retry_backoff_sec", 2.0))
    keep_alive = str(emb_cfg.get("keep_alive", os.getenv("OLLAMA_KEEP_ALIVE", "0s"))).strip()
    base_url = emb_cfg.get("ollama_base_url")
    prefixes_cfg = emb_cfg.get("instruction_prefixes", {})
    prefix_enabled = bool(prefixes_cfg.get("enabled", False))
    only_for_nomic = bool(prefixes_cfg.get("only_for_nomic", True))
    if only_for_nomic and "nomic-embed-text" not in str(model_name).casefold():
        prefix_enabled = False
    query_prefix = str(prefixes_cfg.get("query", "search_query:"))
    document_prefix = str(prefixes_cfg.get("document", "search_document:"))

    if provider == "ollama":
        return OllamaEmbedder(
            model=model_name,
            base_url=base_url,
            timeout=timeout,
            batch_size=batch_size,
            keep_alive=keep_alive,
            use_instruction_prefixes=prefix_enabled,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
        )
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name=model_name, local_files_only=local_only)
    raise ValueError(f"Unsupported embeddings.provider='{provider}'. Use 'ollama' or 'sentence_transformers'.")


def encode_for_task(
    embedder: Any,
    texts: str | list[str],
    task: str,
    normalize_embeddings: bool = True,
    convert_to_numpy: bool = False,
) -> Any:
    encode_with_task = getattr(embedder, "encode_with_task", None)
    if callable(encode_with_task):
        return encode_with_task(
            texts,
            task=task,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=convert_to_numpy,
        )
    return embedder.encode(
        texts,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=convert_to_numpy,
    )
