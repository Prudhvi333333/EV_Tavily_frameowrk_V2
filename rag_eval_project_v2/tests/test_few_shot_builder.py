from __future__ import annotations

import numpy as np
import pandas as pd

import src.few_shot_builder as fsb
from src.few_shot_builder import FewShotBuilder


class _FakeEmbedder:
    def __init__(self) -> None:
        self.tasks: list[str] = []

    def encode_with_task(self, texts, task="generic", normalize_embeddings=True, convert_to_numpy=False):
        self.tasks.append(str(task))
        items = [texts] if isinstance(texts, str) else list(texts)
        vectors = []
        for text in items:
            vec = np.zeros(8, dtype=float)
            for i, ch in enumerate(str(text).lower()):
                vec[(ord(ch) + i) % 8] += 1.0
            norm = np.linalg.norm(vec) or 1.0
            vectors.append(vec / norm if normalize_embeddings else vec)
        arr = np.asarray(vectors, dtype=float)
        if convert_to_numpy:
            return arr
        return arr.tolist()


def test_few_shot_examples_use_grounded_answers_without_placeholders(monkeypatch) -> None:
    fake = _FakeEmbedder()
    monkeypatch.setattr(fsb, "load_embedder_from_config", lambda cfg: fake)
    train_df = pd.DataFrame(
        [
            {
                "Question": "List Tier 1 suppliers in Georgia",
                "Human validated answers": (
                    "There are 2 suppliers.\n"
                    "F&P Georgia Manufacturing [Tier 1] | Role: Battery Pack | Product: Pack assemblies\n"
                    "Hitachi Astemo Americas Inc. [Tier 1] | Role: Battery Cell | Product: Cell components"
                ),
            },
            {
                "Question": "Which companies supply thermal systems?",
                "Human validated answers": "ZF Gainesville LLC | Role: Thermal Management",
            },
        ]
    )
    builder = FewShotBuilder(train_df, {"embeddings": {"provider": "ollama", "model": "nomic-embed-text"}})
    examples = builder.get_examples("Show Tier 1 suppliers", pipeline_mode="rag", n=1)

    assert "There are N matching entities" not in examples
    assert "<Company>" not in examples
    assert "F&P Georgia Manufacturing" in examples or "Hitachi Astemo Americas Inc." in examples
    assert "query" in fake.tasks

