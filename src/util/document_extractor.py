from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .document_utils import _docs_from_text
from .extractors import _extract_doc_text, _extract_docx_text, _extract_excel_text
from .file_utils import _read_text_with_fallback
from .ocr_common import PdfImageTool, _extract_pdf_text
from .text_utils import _join_pages, _normalize_text

logger = logging.getLogger(__name__)


class DocumentTextExtractor:
    def __init__(self, ocr: Any, *, image_dpi: int = 200) -> None:
        self._ocr = ocr
        self._image_dpi = image_dpi

    def _extract_pdf_pages(self, pdf_path: Path) -> list[str]:
        text_pages = _extract_pdf_text(pdf_path)
        if text_pages is not None:
            return text_pages

        tool = PdfImageTool(image_dpi=self._image_dpi)
        page_texts: list[str] = []
        for _, image_bytes, filename in tool.iter_pdf_images(pdf_path):
            lines = self._ocr.get().ocr_image_bytes(image_bytes, filename=filename)
            page_texts.append("\n".join(lines))
        return page_texts

    def extract_text(self, path: Path) -> tuple[list[Document], str]:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            text = _normalize_text(_read_text_with_fallback(path))
            return _docs_from_text(path, text), text
        if suffix == ".pdf":
            start = time.perf_counter()
            pages = self._extract_pdf_pages(path)
            duration = time.perf_counter() - start
            logger.info("OCR processed %s in %.2fs", path, duration)
            text = _join_pages(pages)
            return _docs_from_text(path, text, use_page_markers=True, force_page=True), text
        if suffix == ".docx":
            text = _extract_docx_text(path)
            return (
                _docs_from_text(path, text, use_page_markers=True, force_page=True),
                text,
            )
        if suffix in (".xls", ".xlsx"):
            text = _extract_excel_text(path)
            return _docs_from_text(path, text), text
        if suffix == ".csv":
            text = _normalize_text(_read_text_with_fallback(path))
            return _docs_from_text(path, text), text
        if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
            start = time.perf_counter()
            lines = self._ocr.get().ocr_image_path(path)
            duration = time.perf_counter() - start
            logger.info("OCR processed %s in %.2fs", path, duration)
            text = _normalize_text("\n".join(lines))
            return _docs_from_text(path, text), text
        if suffix == ".doc":
            text = _extract_doc_text(path)
            return (
                _docs_from_text(path, text, use_page_markers=True, force_page=True),
                text,
            )
        raise ValueError(f"Unsupported file type: {path}")
