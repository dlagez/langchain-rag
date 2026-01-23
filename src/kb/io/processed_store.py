from __future__ import annotations

from pathlib import Path


def processed_path(processed_root: Path, kb_id: str, rel_path: Path) -> Path:
    rel_txt = rel_path.with_suffix(".txt")
    return (processed_root / kb_id / rel_txt).resolve()


def write_processed_text(processed_root: Path, kb_id: str, rel_path: Path, text: str) -> Path:
    path = processed_path(processed_root, kb_id, rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
