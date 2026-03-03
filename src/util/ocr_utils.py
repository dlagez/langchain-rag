from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from .ocr_common import _env_first

if TYPE_CHECKING:
    from .local_ppocr import LocalPPOCRClient
    from .remote_ocr import RemoteOCRClient


class _LazyOCR:
    def __init__(self, backend: str | None = None, **kwargs) -> None:
        self._kwargs = kwargs
        self._backend = (
            backend
            or _env_first(
                ["OCR_BACKEND", "OCR_ENGINE", "PPOCR_BACKEND", "PPOCR_ENGINE"],
                "remote",
            )
        ).strip().lower()
        self._ocr: RemoteOCRClient | LocalPPOCRClient | None = None

    def get(self) -> RemoteOCRClient | LocalPPOCRClient:
        if self._ocr is None:
            if self._backend in {"local", "ppocr", "paddle", "paddleocr"}:
                from .local_ppocr import LocalPPOCRClient

                self._ocr = LocalPPOCRClient(**self._kwargs)
            elif self._backend in {"remote", "server", "http"}:
                from .remote_ocr import RemoteOCRClient

                self._ocr = RemoteOCRClient(**self._kwargs)
            else:
                raise ValueError(f"Unknown OCR backend: {self._backend}")
        return self._ocr


def _extract_text_from_file(
    path: Path,
    ocr: _LazyOCR,
    image_dpi: int = 200,
    ocr_concurrency: int = 1,
    ocr_page_timeout: float | None = None,
    ocr_max_retries: int = 0,
    ocr_retry_backoff_ms: int = 0,
) -> tuple[list[Document], str]:
    from .document_extractor import DocumentTextExtractor

    extractor = DocumentTextExtractor(
        ocr,
        image_dpi=image_dpi,
        ocr_concurrency=ocr_concurrency,
        ocr_page_timeout=ocr_page_timeout,
        ocr_max_retries=ocr_max_retries,
        ocr_retry_backoff_ms=ocr_retry_backoff_ms,
    )
    return extractor.extract_text(path)
