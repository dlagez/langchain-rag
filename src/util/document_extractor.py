from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from .document_utils import _build_metadata, _docs_from_text
from .extractors import (
    _excel_rows_to_text,
    _extract_doc_text,
    _extract_docx_text,
    _extract_excel_rows,
)
from .file_utils import _read_text_with_fallback
from .ocr_common import PdfImageTool, _extract_pdf_text
from .text_utils import _join_pages, _normalize_text

logger = logging.getLogger(__name__)


class DocumentTextExtractor:
    """统一文档解析入口：支持文本/Office/PDF/图片/OCR，产出 Document 列表与原文文本。"""
    def __init__(
        self,
        ocr: Any,
        *,
        image_dpi: int = 200,
        ocr_concurrency: int = 1,
        ocr_page_timeout: float | None = None,
        ocr_max_retries: int = 0,
        ocr_retry_backoff_ms: int = 0,
    ) -> None:
        self._ocr = ocr
        self._image_dpi = image_dpi
        self._ocr_concurrency = max(1, int(ocr_concurrency))
        self._ocr_page_timeout = ocr_page_timeout if (ocr_page_timeout and ocr_page_timeout > 0) else None
        self._ocr_max_retries = max(0, int(ocr_max_retries))
        self._ocr_retry_backoff_ms = max(0, int(ocr_retry_backoff_ms))

    def _call_ocr_image_bytes(self, ocr_client: Any, image_bytes: bytes, filename: str) -> list[str]:
        if self._ocr_page_timeout is None:
            return ocr_client.ocr_image_bytes(image_bytes, filename=filename)
        try:
            return ocr_client.ocr_image_bytes(
                image_bytes,
                filename=filename,
                timeout=self._ocr_page_timeout,
            )
        except TypeError:
            return ocr_client.ocr_image_bytes(image_bytes, filename=filename)

    def _call_ocr_image_path(self, ocr_client: Any, image_path: Path) -> list[str]:
        if self._ocr_page_timeout is None:
            return ocr_client.ocr_image_path(image_path)
        try:
            return ocr_client.ocr_image_path(image_path, timeout=self._ocr_page_timeout)
        except TypeError:
            return ocr_client.ocr_image_path(image_path)

    def _ocr_with_retry_bytes(self, ocr_client: Any, image_bytes: bytes, filename: str) -> list[str]:
        attempt = 0
        while True:
            try:
                return self._call_ocr_image_bytes(ocr_client, image_bytes, filename)
            except Exception:
                if attempt >= self._ocr_max_retries:
                    raise
                attempt += 1
                if self._ocr_retry_backoff_ms > 0:
                    time.sleep((self._ocr_retry_backoff_ms / 1000.0) * (2 ** (attempt - 1)))

    def _ocr_with_retry_path(self, ocr_client: Any, image_path: Path) -> list[str]:
        attempt = 0
        while True:
            try:
                return self._call_ocr_image_path(ocr_client, image_path)
            except Exception:
                if attempt >= self._ocr_max_retries:
                    raise
                attempt += 1
                if self._ocr_retry_backoff_ms > 0:
                    time.sleep((self._ocr_retry_backoff_ms / 1000.0) * (2 ** (attempt - 1)))

    def _extract_pdf_pages(self, pdf_path: Path) -> list[str]:
        text_pages = _extract_pdf_text(pdf_path)
        if text_pages is not None:
            return text_pages

        tool = PdfImageTool(image_dpi=self._image_dpi)
        page_items = list(tool.iter_pdf_images(pdf_path))
        if not page_items:
            return []

        ocr_client = self._ocr.get()
        page_texts: list[str] = [""] * len(page_items)

        if self._ocr_concurrency <= 1 or len(page_items) == 1:
            for page_no, image_bytes, filename in page_items:
                try:
                    lines = self._ocr_with_retry_bytes(ocr_client, image_bytes, filename)
                    page_texts[page_no] = "\n".join(lines)
                except Exception as exc:
                    logger.warning(
                        "OCR failed for %s page %s (%s): %s",
                        pdf_path,
                        page_no + 1,
                        filename,
                        exc,
                    )
                    page_texts[page_no] = ""
            return page_texts

        max_workers = min(self._ocr_concurrency, len(page_items))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_page = {
                executor.submit(self._ocr_with_retry_bytes, ocr_client, image_bytes, filename): (
                    page_no,
                    filename,
                )
                for page_no, image_bytes, filename in page_items
            }
            for future in as_completed(future_to_page):
                page_no, filename = future_to_page[future]
                try:
                    lines = future.result()
                    page_texts[page_no] = "\n".join(lines)
                except Exception as exc:
                    logger.warning(
                        "OCR failed for %s page %s (%s): %s",
                        pdf_path,
                        page_no + 1,
                        filename,
                        exc,
                    )
                    page_texts[page_no] = ""
        return page_texts

    def extract_text(self, path: Path) -> tuple[list[Document], str]:
        suffix = path.suffix.lower()
        # Plain text
        if suffix == ".txt":
            text = _normalize_text(_read_text_with_fallback(path))
            return _docs_from_text(path, text), text
        # PDF: try embedded text first, fall back to OCR per page
        if suffix == ".pdf":
            start = time.perf_counter()
            pages = self._extract_pdf_pages(path)
            duration = time.perf_counter() - start
            logger.info("OCR processed %s in %.2fs", path, duration)
            text = _join_pages(pages)
            return _docs_from_text(path, text, use_page_markers=True, force_page=True), text
        # DOCX: parse paragraphs/tables with page markers
        if suffix == ".docx":
            text = _extract_docx_text(path)
            return (
                _docs_from_text(path, text, use_page_markers=True, force_page=True),
                text,
            )
        # Excel: row-level extraction with sheet/row metadata
        if suffix in (".xls", ".xlsx"):
            rows = _extract_excel_rows(path)
            text = _excel_rows_to_text(rows)
            docs: list[Document] = []
            for row in rows:
                if row.get("is_header"):
                    continue
                line = (row.get("text") or "").strip()
                if not line:
                    continue
                metadata = _build_metadata(path)
                metadata["sheet"] = row.get("sheet")
                metadata["row"] = row.get("row")
                metadata["is_header"] = bool(row.get("is_header"))
                docs.append(Document(page_content=line, metadata=metadata))
            return docs, text
        # CSV: treat as plain text
        if suffix == ".csv":
            text = _normalize_text(_read_text_with_fallback(path))
            return _docs_from_text(path, text), text
        # Image: OCR the image and wrap into a single document
        if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
            start = time.perf_counter()
            lines = self._ocr_with_retry_path(self._ocr.get(), path)
            duration = time.perf_counter() - start
            logger.info("OCR processed %s in %.2fs", path, duration)
            text = _normalize_text("\n".join(lines))
            return _docs_from_text(path, text), text
        # DOC: convert with available backends (textract/Word/LibreOffice)
        if suffix == ".doc":
            text = _extract_doc_text(path)
            return (
                _docs_from_text(path, text, use_page_markers=True, force_page=True),
                text,
            )
        raise ValueError(f"Unsupported file type: {path}")
