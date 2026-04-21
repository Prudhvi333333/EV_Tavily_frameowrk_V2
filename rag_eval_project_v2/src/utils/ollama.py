from __future__ import annotations

import os
from urllib.parse import urlparse


def _normalize_ollama_url(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Empty Ollama endpoint value.")
    if "://" not in text:
        text = f"http://{text}"
    parsed = urlparse(text)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid Ollama endpoint: {raw!r}")
    return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")


def resolve_ollama_base_url(config_value: str | None = None) -> str:
    """
    Resolve Ollama endpoint with runtime env override support.

    Precedence:
    1) OLLAMA_HOST
    2) OLLAMA_BASE_URL
    3) Config-provided value
    4) Default localhost endpoint
    """
    candidates = [
        os.getenv("OLLAMA_HOST", "").strip(),
        os.getenv("OLLAMA_BASE_URL", "").strip(),
        str(config_value or "").strip(),
        "http://localhost:11434",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        return _normalize_ollama_url(candidate)
    return "http://localhost:11434"
