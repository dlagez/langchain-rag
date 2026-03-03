from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from util.ocr_utils import _LazyOCR, _extract_text_from_file

from app.settings import Settings


def build_ocr(settings: Settings) -> _LazyOCR:
    return _LazyOCR(
        backend="remote",
        url=settings.ocr_url,
        timeout=settings.ocr_timeout,
        file_field=settings.ocr_file_field,
    )


def extract_documents(
    path: Path,
    ocr: _LazyOCR,
    settings: Settings,
    *,
    image_dpi: int = 200,
) -> tuple[list[Document], str]:
    return _extract_text_from_file(
        path,
        ocr,
        image_dpi=image_dpi,
        ocr_concurrency=settings.ocr_concurrency,
        ocr_page_timeout=settings.ocr_page_timeout,
        ocr_max_retries=settings.ocr_max_retries,
        ocr_retry_backoff_ms=settings.ocr_retry_backoff_ms,
    )
