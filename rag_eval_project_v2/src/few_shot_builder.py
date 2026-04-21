from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.embeddings import encode_for_task, load_embedder_from_config


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
        ranked = np.argsort(sims)[::-1].tolist()
        blocks: list[str] = []
        counter = 1
        for idx in ranked:
            if counter > n:
                break
            row = self.train_df.iloc[int(idx)]
            q = str(row["Question"])
            if " ".join(q.casefold().split()) == q_norm:
                continue
            a = str(row["Human validated answers"])
            a = self._prepare_example_answer(a, pipeline_mode=pipeline_mode)
            blocks.append(f"Example {counter}:\nQuestion: {q}\nAnswer: {a}")
            counter += 1
        return "\n\n".join(blocks)

    def format_for_list_question(self, example: dict[str, str]) -> str:
        answer = str(example.get("answer", ""))
        first_num = re.search(r"\b(\d+)\b", answer)
        if first_num:
            return f"There are {first_num.group(1)} companies: [formatted list follows]."
        return "There are N companies: [formatted list follows]."

    def _format_rag_template(self, golden_answer: str) -> str:
        return self._prepare_example_answer(golden_answer, pipeline_mode="rag")

    def _prepare_example_answer(self, golden_answer: str, pipeline_mode: str) -> str:
        raw = str(golden_answer or "").strip()
        if not raw:
            return "No validated answer available."

        lines = [x.strip() for x in raw.splitlines() if x.strip()]
        if not lines:
            return raw

        has_structured_list = any("|" in ln for ln in lines)
        max_lines = 6 if has_structured_list else 4
        selected = lines[:max_lines]
        text = "\n".join(selected)

        # Keep examples concise to prevent prompt bloat while preserving real factual content.
        limit = 1100 if pipeline_mode in {"rag", "rag_pretrained", "rag_pretrained_web"} else 900
        if len(text) > limit:
            text = text[:limit].rsplit(" ", 1)[0].rstrip() + " ..."
        return text
