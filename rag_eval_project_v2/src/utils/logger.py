from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .config_loader import resolve_path


def get_logger(name: str, config: dict[str, Any]) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    log_dir = resolve_path(config, config["paths"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(Path(log_dir) / f"{name}_{log_name}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger

