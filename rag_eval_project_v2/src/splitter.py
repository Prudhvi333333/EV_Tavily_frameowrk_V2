from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config_loader import resolve_path
from src.utils.logger import get_logger


def split_questions(
    questions_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_logger("splitter", config)
    split_cfg = config["split"]
    train_size = int(split_cfg["train"])
    test_size = int(split_cfg["test"])
    stratify_col = split_cfg["stratify_column"]
    strategy = split_cfg["strategy"]
    total_size = train_size + test_size
    if len(questions_df) != total_size:
        logger.warning(
            "Question count (%s) does not match train+test (%s); split uses configured ratio.",
            len(questions_df),
            total_size,
        )

    if strategy == "stratified":
        train_df, test_df = train_test_split(
            questions_df,
            test_size=test_size / max(total_size, 1),
            random_state=42,
            stratify=questions_df[stratify_col],
        )
    else:
        train_df, test_df = train_test_split(
            questions_df,
            test_size=test_size / max(total_size, 1),
            random_state=42,
        )

    train_df = train_df.sort_values("Num").reset_index(drop=True)
    test_df = test_df.sort_values("Num").reset_index(drop=True)

    train_path = resolve_path(config, config["paths"]["train_questions"])
    test_path = resolve_path(config, config["paths"]["test_questions"])
    train_df.to_excel(train_path, index=False)
    test_df.to_excel(test_path, index=False)

    train_dist = train_df[stratify_col].value_counts().to_dict()
    test_dist = test_df[stratify_col].value_counts().to_dict()
    logger.info("Train distribution: %s", train_dist)
    logger.info("Test distribution: %s", test_dist)

    missing_in_test = sorted(set(train_dist) - set(test_dist))
    if missing_in_test:
        logger.warning("Categories missing in test split: %s", missing_in_test)
    else:
        logger.info("All categories present in test split.")
    return train_df, test_df


def load_split(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_logger("splitter", config)
    train_path = resolve_path(config, config["paths"]["train_questions"])
    test_path = resolve_path(config, config["paths"]["test_questions"])
    master_path = resolve_path(config, config["paths"]["questions_input"])

    if train_path.exists() and test_path.exists():
        train_df = pd.read_excel(train_path)
        test_df = pd.read_excel(test_path)
        logger.info("Loaded existing split from disk.")
    else:
        logger.info("No split files found, generating new split.")
        questions_df = pd.read_excel(master_path)
        train_df, test_df = split_questions(questions_df, config)

    logger.info("Test set question IDs: %s", test_df["Num"].tolist())
    return train_df, test_df

