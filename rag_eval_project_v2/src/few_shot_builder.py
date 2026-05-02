from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.embeddings import encode_for_task, load_embedder_from_config


_LIST_QUESTION_TRIGGERS = (
    "list", "which", "how many", "count", "names", "name the", "show all",
    "all suppliers", "all companies", "all firms", "all areas", "all plants",
    "all facilities", "supplier network", "linked to", "tied to", "show me",
    "give me", "enumerate", "what are",
)

_LIST_ANSWER_RE = re.compile(r"(^\s*\d+[.)]\s)|(^\s*-\s)|(^\s*\*\s)|(\|)", re.MULTILINE)


def _is_list_question(question: str) -> bool:
    q = (question or "").casefold()
    return any(t in q for t in _LIST_QUESTION_TRIGGERS)


def _is_list_answer(answer: str) -> bool:
    if not answer:
        return False
    return bool(_LIST_ANSWER_RE.search(answer))


class FewShotBuilder:
    def __init__(self, train_df: pd.DataFrame, config: dict[str, Any]) -> None:
        self.train_df = train_df.reset_index(drop=True)
        self.config = config
        self.embedder = load_embedder_from_config(config)
        self.train_questions = self.train_df["Question"].astype(str).tolist()
        self.train_vectors = encode_for_task(
            self.embedder,
            self.train_questions,
            task="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        self._list_indices: list[int] = [
            i for i, a in enumerate(self.train_df.get("Human validated answers", []).astype(str).tolist())
            if _is_list_answer(a)
        ]

    def get_examples(self, question: str, pipeline_mode: str, n: int = 2) -> str:
        if len(self.train_df) == 0 or n <= 0:
            return ""
        query_vec = encode_for_task(
            self.embedder,
            [question],
            task="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        sims = cosine_similarity(query_vec, self.train_vectors)[0]
        q_norm = " ".join(question.casefold().split())

        # Build a similarity-ranked candidate list excluding the query itself.
        ranked = [int(i) for i in np.argsort(sims)[::-1].tolist()]
        seen: set[int] = set()
        candidates: list[int] = []
        for idx in ranked:
            if idx in seen:
                continue
            row = self.train_df.iloc[idx]
            if " ".join(str(row["Question"]).casefold().split()) == q_norm:
                continue
            candidates.append(idx)
            seen.add(idx)

        # For list/count questions, ensure at least floor(n/2) of the chosen examples
        # have list-style golden answers — a small bias that materially improves the
        # generator's adherence to the numbered-list output format on structured queries.
        is_list_q = _is_list_question(question)
        min_list_examples = max(1, n // 2) if is_list_q else 0

        chosen: list[int] = []
        if min_list_examples > 0:
            list_pool = [
                idx for idx in candidates
                if idx in set(self._list_indices)
            ]
            for idx in list_pool[:min_list_examples]:
                chosen.append(idx)

        for idx in candidates:
            if len(chosen) >= n:
                break
            if idx not in chosen:
                chosen.append(idx)

        blocks: list[str] = []
        for counter, idx in enumerate(chosen[:n], start=1):
            row = self.train_df.iloc[idx]
            q = str(row["Question"])
            a = str(row["Human validated answers"])
            a = self._prepare_example_answer(a, pipeline_mode=pipeline_mode)
            blocks.append(f"Example {counter}:\nQuestion: {q}\nAnswer: {a}")
        return "\n\n".join(blocks)

    def _prepare_example_answer(self, golden_answer: str, pipeline_mode: str) -> str:
        raw = str(golden_answer or "").strip()
        if not raw:
            return "No validated answer available."

        lines = [x.strip() for x in raw.splitlines() if x.strip()]
        if not lines:
            return raw

        # Preserve full list structure when the answer is enumerative — clipping at 6
        # lines truncates supplier-network answers and teaches the model to truncate
        # too. Cap at 15 lines instead.
        has_structured_list = any("|" in ln for ln in lines) or _is_list_answer(raw)
        max_lines = 15 if has_structured_list else 4
        selected = lines[:max_lines]
        text = "\n".join(selected)

        limit = 1600 if pipeline_mode in {"rag", "rag_pretrained", "rag_pretrained_web"} else 1300
        if len(text) > limit:
            text = text[:limit].rsplit(" ", 1)[0].rstrip() + " ..."
        return text
