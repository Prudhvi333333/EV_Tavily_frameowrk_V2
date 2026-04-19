from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.embeddings import load_embedder_from_config


class FewShotBuilder:
    def __init__(self, train_df: pd.DataFrame, config: dict[str, Any]) -> None:
        self.train_df = train_df.reset_index(drop=True)
        self.config = config
        self.embedder = load_embedder_from_config(config)
        self.train_questions = self.train_df["Question"].astype(str).tolist()
        self.train_vectors = self.embedder.encode(self.train_questions, normalize_embeddings=True)

    def get_examples(self, question: str, pipeline_mode: str, n: int = 2) -> str:
        if len(self.train_df) == 0 or n <= 0:
            return ""
        query_vec = self.embedder.encode([question], normalize_embeddings=True)
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
            if pipeline_mode in {"rag", "rag_pretrained", "rag_pretrained_web"}:
                a = self._format_rag_template(a)
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
        lines = [x.strip() for x in golden_answer.splitlines() if x.strip()]
        if not lines:
            return "Answer with grounded bullet points and clear source-based statements."
        has_list = any("|" in ln for ln in lines)
        if has_list:
            return (
                "There are N matching entities.\n"
                "1) <Company> [<Tier>] | Role: <EV Supply Chain Role> | Product: <Product / Service>\n"
                "2) <Company> [<Tier>] | Role: <EV Supply Chain Role> | Product: <Product / Service>"
            )
        return "Provide a concise factual answer and explicitly mention uncertainty when context is missing."
