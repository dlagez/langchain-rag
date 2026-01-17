from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from .document_utils import _docs_from_text
from .file_utils import _write_text
from .ocr_utils import _LazyOCR, _extract_text_from_file
from .text_utils import _normalize_text

logger = logging.getLogger(__name__)


def _load_cached_text(source_path: Path, processed_path: Path) -> str | None:
    if not processed_path.exists():
        return None
    try:
        if processed_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns:
            return _normalize_text(
                processed_path.read_text(encoding="utf-8", errors="replace")
            )
    except OSError:
        return None
    return None


def process_sources(
    source_dir: Path,
    processed_dir: Path,
    ocr: _LazyOCR,
    image_dpi: int = 200,
) -> list[Document]:
    source_dir = Path(source_dir)
    processed_dir = Path(processed_dir)
    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    docs: list[Document] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue

        rel = path.relative_to(source_dir)
        out_path = (processed_dir / rel).with_suffix(".txt")

        cached_text = _load_cached_text(path, out_path)
        if cached_text:
            cached_docs = _docs_from_text(
                path,
                cached_text,
                use_page_markers=path.suffix.lower() == ".pdf",
                force_page=path.suffix.lower() == ".pdf",
            )
            docs.extend(cached_docs)
            continue

        try:
            extracted_docs, text = _extract_text_from_file(
                path, ocr, image_dpi=image_dpi
            )
        except ValueError:
            logger.info("Skipping unsupported file: %s", path)
            continue
        except Exception as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue

        if text:
            _write_text(out_path, text)
        if extracted_docs:
            docs.extend(extracted_docs)

    if not docs:
        raise SystemExit("No supported files found in data/source.")
    return docs
