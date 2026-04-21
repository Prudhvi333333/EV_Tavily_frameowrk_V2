from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.splitter import load_split


def test_load_split_appends_extra_test_questions_without_changing_train(tmp_path: Path) -> None:
    train_path = tmp_path / "train_questions.xlsx"
    test_path = tmp_path / "test_questions.xlsx"
    master_path = tmp_path / "questions_master.xlsx"
    extra_path = tmp_path / "extra_test_questions.xlsx"

    train_df = pd.DataFrame(
        [
            {"Num": 2, "Question": "Train Q1", "Use Case Category": "A", "Human validated answers": "A1"},
            {"Num": 4, "Question": "Train Q2", "Use Case Category": "B", "Human validated answers": "A2"},
        ]
    )
    test_df = pd.DataFrame(
        [
            {"Num": 1, "Question": "Test Q1", "Use Case Category": "A", "Human validated answers": "T1"},
            {"Num": 3, "Question": "Test Q2", "Use Case Category": "B", "Human validated answers": "T2"},
        ]
    )
    extra_df = pd.DataFrame(
        [
            {"Question": "Test Q1", "Use Case Category": "A"},  # duplicate of base test -> should be skipped
            {"Question": "Extra Q1"},  # missing fields -> defaults should be applied
            {"Num": 20, "Question": "Extra Q2", "Use Case Category": "C", "Human validated answers": "E2"},
        ]
    )

    train_df.to_excel(train_path, index=False)
    test_df.to_excel(test_path, index=False)
    # master path is not used in this test branch, but kept for complete config shape.
    train_df.to_excel(master_path, index=False)
    extra_df.to_excel(extra_path, index=False)

    cfg = {
        "paths": {
            "train_questions": str(train_path.resolve()),
            "test_questions": str(test_path.resolve()),
            "questions_input": str(master_path.resolve()),
            "extra_test_questions": str(extra_path.resolve()),
            "logs_dir": str((tmp_path / "logs").resolve()),
        }
    }

    loaded_train, loaded_test = load_split(cfg)
    assert len(loaded_train) == 2
    assert len(loaded_test) == 4
    assert "Extra Q1" in loaded_test["Question"].tolist()
    assert "Extra Q2" in loaded_test["Question"].tolist()
    assert loaded_test["Question"].str.casefold().nunique() == len(loaded_test)

    extra_q1 = loaded_test.loc[loaded_test["Question"] == "Extra Q1"].iloc[0]
    assert str(extra_q1["Use Case Category"]) == "Extra Test"
    assert str(extra_q1["Human validated answers"]) == ""
    assert pd.notna(extra_q1["Num"])
