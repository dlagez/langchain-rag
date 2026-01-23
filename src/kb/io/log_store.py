from __future__ import annotations

import logging
from pathlib import Path


def log_path(log_root: Path, kb_id: str) -> Path:
    path = log_root / kb_id
    path.mkdir(parents=True, exist_ok=True)
    return path / "ingest.log"


def configure_logging(log_root: Path, kb_id: str, level: int = logging.INFO) -> None:
    log_file = log_path(log_root, kb_id)
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
