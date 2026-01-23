from __future__ import annotations

from pathlib import Path


def kb_index_dir(index_root: Path, kb_id: str) -> Path:
    path = index_root / kb_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def qdrant_path(index_root: Path, kb_id: str) -> Path:
    return kb_index_dir(index_root, kb_id) / "qdrant"


def bm25_path(index_root: Path, kb_id: str) -> Path:
    return kb_index_dir(index_root, kb_id) / "bm25.pkl"


def collection_name(base: str, kb_id: str, *, use_remote: bool) -> str:
    if use_remote:
        return f"{base}_{kb_id}"
    return base
