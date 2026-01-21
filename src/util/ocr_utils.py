from __future__ import annotations

import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from langchain_core.documents import Document

from .document_utils import _docs_from_text
from .extractors import _extract_doc_text, _extract_docx_text, _extract_excel_text
from .file_utils import _read_text_with_fallback
from .text_utils import _join_pages, _normalize_text

logger = logging.getLogger(__name__)


def _env_first(names: list[str], default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _extract_pdf_text(path: Path) -> list[str] | None:
    try:
        import fitz  # type: ignore
    except Exception:
        return None
    pages: list[str] = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            text = page.get_text("text").strip()
            pages.append(text)
    if not pages:
        return None
    if any(page for page in pages):
        return pages
    return None


class PdfImageTool:
    def __init__(self, *, image_dpi: int = 200) -> None:
        self.image_dpi = image_dpi

    def iter_pdf_images(self, pdf_path: Path):
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            import fitz  # type: ignore
        except Exception:
            fitz = None

        if fitz is not None:
            with fitz.open(str(pdf_path)) as doc:
                for index, page in enumerate(doc):
                    zoom = self.image_dpi / 72.0
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    image_bytes = pix.tobytes("png")
                    filename = f"{pdf_path.stem}_page_{index + 1}.png"
                    yield index, image_bytes, filename
            return

        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise SystemExit(
                "Missing PDF renderer. Install pymupdf (fitz) or pdf2image + poppler."
            ) from exc

        images = convert_from_path(str(pdf_path), dpi=self.image_dpi)
        for index, image in enumerate(images):
            from io import BytesIO

            buffer = BytesIO()
            image.save(buffer, format="PNG")
            filename = f"{pdf_path.stem}_page_{index + 1}.png"
            yield index, buffer.getvalue(), filename


class RemoteOCRClient:
    def __init__(
        self,
        *,
        url: str | None = None,
        timeout: float | None = None,
        file_field: str | None = None,
    ) -> None:
        self._url = (
            url
            or _env_first(
                ["OCR_URL", "PPOCR_URL", "PADDLEX_OCR_URL"],
                "http://10.0.22.109:8081/ocr",
            )
        ).strip()
        self._timeout = float(
            timeout
            if timeout is not None
            else _env_first(["OCR_TIMEOUT", "PPOCR_TIMEOUT", "PADDLEX_OCR_TIMEOUT"], "30")
        )
        self._file_field = (
            file_field
            or _env_first(["OCR_FILE_FIELD", "PPOCR_FILE_FIELD", "PADDLEX_OCR_FILE_FIELD"], "file")
        ).strip()

    def _post_multipart(self, image_bytes: bytes, filename: str) -> dict[str, Any]:
        boundary = f"----ocrboundary{os.urandom(8).hex()}"
        mime = "image/png"
        guess, _ = mimetypes.guess_type(filename)
        if guess:
            mime = guess
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{self._file_field}"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
        footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
        body = header + image_bytes + footer

        req = request.Request(
            self._url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OCR HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"OCR request failed: {exc}") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OCR server returned invalid JSON.") from exc

    def _extract_texts(self, payload: Any) -> list[str]:
        texts: list[str] = []

        def _from_line(line: Any) -> str | None:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                return None
            value = line[1]
            if not isinstance(value, (list, tuple)) or not value:
                return None
            text = value[0]
            return text if isinstance(text, str) else None

        def _walk(value: Any) -> None:
            if isinstance(value, str):
                if value.strip():
                    texts.append(value.strip())
                return
            if isinstance(value, dict):
                if "result" in value:
                    _walk(value["result"])
                    return
                for item in value.values():
                    _walk(item)
                return
            if isinstance(value, list):
                if value and isinstance(value[0], (list, tuple)):
                    line_texts = []
                    for line in value:
                        text = _from_line(line)
                        if text:
                            line_texts.append(text)
                    if line_texts:
                        texts.extend(line_texts)
                        return
                for item in value:
                    _walk(item)

        _walk(payload)
        return texts

    def ocr_image_bytes(self, image_bytes: bytes, filename: str = "image.png") -> list[str]:
        response = self._post_multipart(image_bytes, filename)
        return self._extract_texts(response)

    def ocr_image_path(self, image_path: Path) -> list[str]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.ocr_image_bytes(image_path.read_bytes(), filename=image_path.name)

    def ocr_pdf(self, pdf_path: Path, image_dpi: int = 200) -> list[str]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        text_pages = _extract_pdf_text(pdf_path)
        if text_pages is not None:
            return text_pages

        tool = PdfImageTool(image_dpi=image_dpi)
        page_texts: list[str] = []
        for _, image_bytes, filename in tool.iter_pdf_images(pdf_path):
            lines = self.ocr_image_bytes(image_bytes, filename=filename)
            page_texts.append("\n".join(lines))
        return page_texts


class _LazyOCR:
    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
        self._ocr: RemoteOCRClient | None = None

    def get(self) -> RemoteOCRClient:
        if self._ocr is None:
            self._ocr = RemoteOCRClient(**self._kwargs)
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
    if suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
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
