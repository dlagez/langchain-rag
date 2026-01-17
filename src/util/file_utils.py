from __future__ import annotations

from pathlib import Path

from .text_utils import _maybe_repair_mojibake


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_text_with_fallback(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = path.read_text(encoding="gb18030")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
    return _maybe_repair_mojibake(text)
