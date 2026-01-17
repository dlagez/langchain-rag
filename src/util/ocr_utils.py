from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from langchain_core.documents import Document

from ppocr_pdf_tool import LocalPPOCRTool, RemotePPOCRTool

from .document_utils import _docs_from_text
from .extractors import _extract_doc_text, _extract_docx_text, _extract_excel_text
from .file_utils import _read_text_with_fallback
from .text_utils import _join_pages, _normalize_text

logger = logging.getLogger(__name__)


class _LazyOCR:
    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
        self._ocr: LocalPPOCRTool | RemotePPOCRTool | None = None

    def get(self) -> LocalPPOCRTool | RemotePPOCRTool:
        if self._ocr is None:
            mode = os.getenv("PPOCR_MODE", "local").strip().lower()
            if mode in ("remote", "http"):
                self._ocr = RemotePPOCRTool()
            elif mode == "auto":
                try:
                    self._ocr = LocalPPOCRTool(**self._kwargs)
                except SystemExit:
                    self._ocr = RemotePPOCRTool()
            else:
                self._ocr = LocalPPOCRTool(**self._kwargs)
        return self._ocr


def _extract_text_from_file(
    path: Path,
    ocr: _LazyOCR,
    image_dpi: int = 200,
) -> tuple[list[Document], str]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = _normalize_text(_read_text_with_fallback(path))
        return _docs_from_text(path, text), text
    if suffix == ".pdf":
        pages = LocalPPOCRTool._extract_pdf_text(path)
        if pages is None:
            start = time.perf_counter()
            pages = ocr.get().ocr_pdf(path, image_dpi=image_dpi)
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
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        start = time.perf_counter()
        lines = ocr.get().ocr_image_path(path)
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
