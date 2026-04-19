from __future__ import annotations

import os
from typing import Any

import httpx
import numpy as np


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


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        timeout: float = 60.0,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        self.batch_size = max(int(batch_size), 1)
        self._verified = False

    def encode(
        self,
        texts: str | list[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = False,
    ) -> Any:
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
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
        timeout = httpx.Timeout(connect=5.0, read=self.timeout, write=20.0, pool=5.0)
        payload = {"model": self.model, "input": batch}
        url = f"{self.base_url}/api/embed"
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

    def _embed_batch_legacy(self, batch: list[str]) -> list[list[float]]:
        timeout = httpx.Timeout(connect=5.0, read=self.timeout, write=20.0, pool=5.0)
        out: list[list[float]] = []
        with httpx.Client(timeout=timeout) as client:
            for text in batch:
                resp = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
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
    base_url = emb_cfg.get("ollama_base_url")

    if provider == "ollama":
        return OllamaEmbedder(model=model_name, base_url=base_url, batch_size=batch_size)
    if provider == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name=model_name, local_files_only=local_only)
    raise ValueError(f"Unsupported embeddings.provider='{provider}'. Use 'ollama' or 'sentence_transformers'.")

