from __future__ import annotations

from typing import Any


class LlamaIndexSemanticBackend:
    def __init__(self, collection: Any, strict_mode: bool = True) -> None:
        self.collection = collection
        self.strict_mode = strict_mode
        try:
            from llama_index.core.vector_stores import VectorStoreQuery
            from llama_index.vector_stores.chroma import ChromaVectorStore
        except Exception as exc:
            raise RuntimeError(
                "LlamaIndex retrieval backend is enabled but dependencies are missing. "
                "Install: llama-index-core and llama-index-vector-stores-chroma."
            ) from exc

        self._vector_store_query_cls = VectorStoreQuery
        self._vector_store = ChromaVectorStore(chroma_collection=collection)

    def query(self, vector: list[float], n_results: int) -> list[dict[str, Any]]:
        query = self._vector_store_query_cls(
            query_embedding=vector,
            similarity_top_k=max(int(n_results), 1),
            mode="default",
        )
        try:
            result = self._vector_store.query(query)
        except Exception as exc:
            raise RuntimeError("LlamaIndex vector store query failed.") from exc

        ids = list(getattr(result, "ids", []) or [])
        similarities = list(getattr(result, "similarities", []) or [])
        distances = list(getattr(result, "distances", []) or [])
        texts: list[str] = []
        metas: list[dict[str, Any]] = []

        nodes = list(getattr(result, "nodes", []) or [])
        if nodes:
            for node in nodes:
                if hasattr(node, "get_content"):
                    texts.append(str(node.get_content()))
                else:
                    texts.append(str(getattr(node, "text", "")))
                metas.append(dict(getattr(node, "metadata", {}) or {}))
                if not ids:
                    ids.append(str(getattr(node, "node_id", "")))
        else:
            raw_texts = getattr(result, "documents", None)
            raw_metas = getattr(result, "metadatas", None)
            if raw_texts is not None:
                texts = [str(x) for x in list(raw_texts)]
            if raw_metas is not None:
                metas = [dict(x or {}) for x in list(raw_metas)]

        length = min(
            len(texts) if texts else len(ids),
            len(metas) if metas else (len(texts) if texts else len(ids)),
        )
        if length == 0:
            return []

        if not texts:
            texts = [""] * length
        if not metas:
            metas = [{} for _ in range(length)]
        if not ids:
            ids = ["" for _ in range(length)]

        out: list[dict[str, Any]] = []
        for idx in range(length):
            sim = 0.0
            if idx < len(similarities):
                sim = float(similarities[idx])
            elif idx < len(distances):
                sim = 1.0 / (1.0 + float(distances[idx]))
            out.append(
                {
                    "doc_id": str(ids[idx]),
                    "text": str(texts[idx]),
                    "metadata": dict(metas[idx] or {}),
                    "similarity": max(0.0, min(1.0, sim)),
                }
            )
        return out

