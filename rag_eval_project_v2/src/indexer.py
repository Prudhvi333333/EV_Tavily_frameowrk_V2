from __future__ import annotations

import pickle
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from src.utils.embeddings import encode_for_task, load_embedder_from_config
from src.utils.config_loader import resolve_path
from src.utils.logger import get_logger


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class HybridIndex:
    collection: Any
    documents: list[dict[str, Any]]
    embedder: Any
    bm25: BM25Okapi
    tokenized_corpus: list[list[str]]


def build_or_load_index(config: dict[str, Any], documents: list[dict[str, Any]]) -> HybridIndex:
    logger = get_logger("indexer", config)
    chroma_dir = resolve_path(config, config["paths"]["chroma_dir"])
    chroma_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = resolve_path(config, config["paths"]["progress_dir"])
    progress_dir.mkdir(parents=True, exist_ok=True)

    embedder = load_embedder_from_config(config)
    ids = [d["id"] for d in documents]
    texts = [d["text"] for d in documents]
    metadatas = [d["metadata"] for d in documents]
    probe_vec = encode_for_task(
        embedder,
        [texts[0]],
        task="document",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    embedding_dim = int(len(probe_vec))

    embed_manifest_path = Path(progress_dir) / "embedding_manifest.json"
    current_manifest = {
        "provider": str(config.get("embeddings", {}).get("provider", "sentence_transformers")),
        "model": str(config.get("embeddings", {}).get("model", "")),
        "dim": embedding_dim,
        "instruction_prefixes": dict(
            config.get("embeddings", {}).get("instruction_prefixes", {})
            if isinstance(config.get("embeddings", {}).get("instruction_prefixes", {}), dict)
            else {}
        ),
    }
    previous_manifest = {}
    if embed_manifest_path.exists():
        try:
            previous_manifest = json.loads(embed_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            previous_manifest = {}

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection_name = "ev_kb_chunks"
    force_rebuild = previous_manifest != current_manifest
    try:
        existing = client.get_collection(collection_name)
        if force_rebuild or existing.count() != len(documents):
            client.delete_collection(collection_name)
            collection = client.create_collection(
                collection_name,
                metadata={"hnsw:space": "cosine", "embedding_dim": embedding_dim},
            )
        else:
            collection = existing
    except Exception:
        collection = client.create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine", "embedding_dim": embedding_dim},
        )

    if collection.count() == 0:
        logger.info("Building fresh Chroma index for %s documents.", len(documents))
        vectors = encode_for_task(
            embedder,
            texts,
            task="document",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        collection.add(ids=ids, embeddings=vectors.tolist(), documents=texts, metadatas=metadatas)
    else:
        logger.info("Using existing Chroma index with %s documents.", collection.count())

    tokenized_corpus = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    bm25_dump = Path(progress_dir) / "bm25_index.pkl"
    with bm25_dump.open("wb") as f:
        pickle.dump(
            {
                "ids": ids,
                "texts": texts,
                "tokenized_corpus": tokenized_corpus,
                "avgdl": float(np.mean([len(x) for x in tokenized_corpus])),
            },
            f,
        )
    embed_manifest_path.write_text(json.dumps(current_manifest, indent=2), encoding="utf-8")

    return HybridIndex(
        collection=collection,
        documents=documents,
        embedder=embedder,
        bm25=bm25,
        tokenized_corpus=tokenized_corpus,
    )
