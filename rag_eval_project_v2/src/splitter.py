from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config_loader import resolve_path
from src.utils.logger import get_logger


def _normalize_question(text: Any) -> str:
    return " ".join(str(text or "").casefold().split())


def _load_extra_test_questions(
    config: dict[str, Any],
    logger: Any,
    base_test_df: pd.DataFrame,
) -> pd.DataFrame:
    extra_path_raw = str(config.get("paths", {}).get("extra_test_questions", "")).strip()
    if not extra_path_raw:
        return base_test_df

    extra_path = resolve_path(config, extra_path_raw)
    if not extra_path.exists():
        logger.warning("Extra test questions file not found: %s (skipping)", extra_path)
        return base_test_df

    suffix = extra_path.suffix.casefold()
    if suffix == ".csv":
        extra_df = pd.read_csv(extra_path)
    else:
        extra_df = pd.read_excel(extra_path)

    if extra_df.empty:
        logger.info("Extra test questions file is empty: %s", extra_path)
        return base_test_df
    if "Question" not in extra_df.columns:
        raise ValueError(f"Extra test questions file must contain 'Question' column: {extra_path}")

    out = extra_df.copy()
    if "Human validated answers" not in out.columns:
        out["Human validated answers"] = ""
    out["Human validated answers"] = out["Human validated answers"].fillna("").astype(str)

    if "Use Case Category" not in out.columns:
        out["Use Case Category"] = "Extra Test"
    out["Use Case Category"] = out["Use Case Category"].fillna("Extra Test").astype(str)

    max_num = pd.to_numeric(base_test_df.get("Num"), errors="coerce").dropna().max()
    next_num = int(max_num) + 1 if pd.notna(max_num) else 1
    if "Num" not in out.columns:
        out["Num"] = list(range(next_num, next_num + len(out)))
    else:
        num_series = pd.to_numeric(out["Num"], errors="coerce")
        missing_mask = num_series.isna()
        out["Num"] = num_series
        if missing_mask.any():
            needed = int(missing_mask.sum())
            out.loc[missing_mask, "Num"] = list(range(next_num, next_num + needed))

    existing_questions = {_normalize_question(q) for q in base_test_df["Question"].astype(str).tolist()}
    seen = set(existing_questions)
    dedup_rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        q = str(row.get("Question", "")).strip()
        if not q:
            continue
        key = _normalize_question(q)
        if key in seen:
            continue
        seen.add(key)
        dedup_rows.append(row.to_dict())

    if not dedup_rows:
        logger.info("No new extra test questions remained after dedupe.")
        return base_test_df

    merged = pd.concat([base_test_df, pd.DataFrame(dedup_rows)], ignore_index=True)
    merged["Num"] = pd.to_numeric(merged["Num"], errors="coerce")
    merged = merged.sort_values("Num", na_position="last").reset_index(drop=True)
    logger.info(
        "Appended extra test questions: +%s from %s (test now=%s)",
        len(dedup_rows),
        extra_path,
        len(merged),
    )
    return merged


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

    test_df = _load_extra_test_questions(config, logger, test_df)
    logger.info("Test set question IDs: %s", test_df["Num"].tolist())
    return train_df, test_df
