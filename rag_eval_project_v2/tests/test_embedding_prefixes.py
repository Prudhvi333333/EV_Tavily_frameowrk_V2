from __future__ import annotations

from src.utils.embeddings import OllamaEmbedder, encode_for_task


def _fake_embed_batch(captured: list[str]):
    def _run(batch: list[str]) -> list[list[float]]:
        captured.extend(batch)
        return [[float(len(t)), 1.0, 0.5] for t in batch]

    return _run


def test_ollama_embedder_applies_query_prefix_when_enabled() -> None:
    captured: list[str] = []
    embedder = OllamaEmbedder(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
        use_instruction_prefixes=True,
        query_prefix="search_query:",
        document_prefix="search_document:",
    )
    embedder._ensure_model_available = lambda: None
    embedder._embed_batch = _fake_embed_batch(captured)

    _ = encode_for_task(
        embedder,
        ["Which suppliers provide electrolyte materials in Georgia?"],
        task="query",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    assert captured
    assert captured[0].startswith("search_query:")


def test_ollama_embedder_avoids_double_prefix() -> None:
    captured: list[str] = []
    embedder = OllamaEmbedder(
        model="nomic-embed-text",
        base_url="http://127.0.0.1:11434",
        use_instruction_prefixes=True,
        query_prefix="search_query:",
        document_prefix="search_document:",
    )
    embedder._ensure_model_available = lambda: None
    embedder._embed_batch = _fake_embed_batch(captured)

    _ = encode_for_task(
        embedder,
        ["search_document: Existing prefixed doc"],
        task="document",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    assert captured[0] == "search_document: Existing prefixed doc"
