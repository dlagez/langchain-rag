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


def _env_bool(names: list[str], default: bool) -> bool:
    for name in names:
        value = os.getenv(name)
        if value is None:
            continue
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _env_optional_path(names: list[str]) -> Path | None:
    raw = _env_first(names, "").strip()
    if not raw:
        return None
    return Path(raw)


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

        def _flatten_rec_texts(value: Any) -> list[str]:
            if not isinstance(value, (list, tuple)):
                return []
            flat: list[str] = []
            for item in value:
                if isinstance(item, str):
                    stripped = item.strip()
                    if stripped:
                        flat.append(stripped)
                    continue
                if isinstance(item, (list, tuple)) and item:
                    first = item[0]
                    if isinstance(first, str):
                        stripped = first.strip()
                if stripped:
                    flat.append(stripped)
            return flat

        def _find_rec_texts(value: Any) -> tuple[bool, list[str]]:
            if isinstance(value, dict):
                if "rec_texts" in value:
                    return True, _flatten_rec_texts(value.get("rec_texts"))
                for key in ("res", "result", "data", "output", "outputs", "results", "pages"):
                    if key in value:
                        found, found_texts = _find_rec_texts(value[key])
                        if found:
                            return True, found_texts
                return False, []
            if isinstance(value, list):
                found_any = False
                merged: list[str] = []
                for item in value:
                    found, found_texts = _find_rec_texts(item)
                    if found:
                        found_any = True
                        merged.extend(found_texts)
                return found_any, merged
            return False, []

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

        if isinstance(payload, list):
            merged: list[str] = []
            found_any = False
            for item in payload:
                if isinstance(item, dict) and "rec_texts" in item:
                    found_any = True
                    merged.extend(_flatten_rec_texts(item.get("rec_texts")))
                    continue
                found, found_texts = _find_rec_texts(item)
                if found:
                    found_any = True
                    merged.extend(found_texts)
            if found_any:
                return merged

        found, rec_texts = _find_rec_texts(payload)
        if found:
            return rec_texts

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


class LocalPPOCRClient:
    def __init__(
        self,
        *,
        output_dir: Path | None = None,
        save_img: bool | None = None,
        save_json: bool | None = None,
        print_result: bool | None = None,
        disable_model_source_check: bool | None = None,
        predict_kwargs: dict[str, Any] | None = None,
        **paddle_kwargs: Any,
    ) -> None:
        self._output_dir = output_dir or _env_optional_path(
            ["OCR_OUTPUT_DIR", "PPOCR_OUTPUT_DIR", "PADDLEX_OCR_OUTPUT_DIR"]
        )
        self._save_img = (
            save_img
            if save_img is not None
            else _env_bool(["OCR_SAVE_IMG", "PPOCR_SAVE_IMG"], False)
        )
        self._save_json = (
            save_json
            if save_json is not None
            else _env_bool(["OCR_SAVE_JSON", "PPOCR_SAVE_JSON"], False)
        )
        self._print_result = (
            print_result
            if print_result is not None
            else _env_bool(["OCR_PRINT_RESULT", "PPOCR_PRINT_RESULT"], False)
        )
        if self._output_dir is None and (self._save_img or self._save_json):
            self._output_dir = Path("ppocr_results")
        self._disable_model_source_check = (
            disable_model_source_check
            if disable_model_source_check is not None
            else _env_bool(
                [
                    "PPOCR_DISABLE_MODEL_SOURCE_CHECK",
                    "OCR_DISABLE_MODEL_SOURCE_CHECK",
                ],
                False,
            )
        )
        self._predict_kwargs = predict_kwargs or {}
        self._paddle_kwargs = self._build_paddle_kwargs(paddle_kwargs)
        self._ocr = None

    def _build_paddle_kwargs(self, paddle_kwargs: dict[str, Any]) -> dict[str, Any]:
        defaults = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        env_overrides: dict[str, Any] = {}
        env_lang = os.getenv("PPOCR_LANG") or os.getenv("OCR_LANG")
        if env_lang:
            env_overrides["lang"] = env_lang
        env_version = os.getenv("PPOCR_VERSION") or os.getenv("OCR_VERSION")
        if env_version:
            env_overrides["ocr_version"] = env_version

        for key in (
            "use_doc_orientation_classify",
            "use_doc_unwarping",
            "use_textline_orientation",
        ):
            env_key = f"PPOCR_{key.upper()}"
            alt_env_key = f"OCR_{key.upper()}"
            if os.getenv(env_key) is not None or os.getenv(alt_env_key) is not None:
                env_overrides[key] = _env_bool(
                    [env_key, alt_env_key],
                    defaults[key],
                )

        merged = dict(paddle_kwargs)
        for key, value in env_overrides.items():
            merged.setdefault(key, value)
        for key, value in defaults.items():
            merged.setdefault(key, value)
        return merged

    def _get_ocr(self):
        if self._ocr is None:
            if self._disable_model_source_check:
                os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR  # type: ignore

            self._ocr = PaddleOCR(**self._paddle_kwargs)
        return self._ocr

    def _maybe_save_results(self, results: list[Any]) -> None:
        if not results:
            return
        if self._output_dir is None:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        for res in results:
            if self._print_result and hasattr(res, "print"):
                try:
                    res.print()
                except Exception:
                    pass
            if self._save_img and hasattr(res, "save_to_img"):
                try:
                    res.save_to_img(str(self._output_dir))
                except Exception:
                    pass
            if self._save_json and hasattr(res, "save_to_json"):
                try:
                    res.save_to_json(str(self._output_dir))
                except Exception:
                    pass

    def _extract_texts_from_result(self, res: Any) -> list[str]:
        def _normalize(item: Any) -> str | None:
            if isinstance(item, str):
                value = item.strip()
                return value or None
            if isinstance(item, (list, tuple)) and item:
                first = item[0]
                if isinstance(first, str):
                    value = first.strip()
                    return value or None
            return None

        def _flatten(items: Any) -> list[str]:
            if not isinstance(items, (list, tuple)):
                return []
            texts: list[str] = []
            for item in items:
                value = _normalize(item)
                if value:
                    texts.append(value)
            return texts

        if hasattr(res, "json"):
            payload = res.json
            if isinstance(payload, dict):
                data = payload.get("res", payload)
                if isinstance(data, dict) and "rec_texts" in data:
                    return _flatten(data.get("rec_texts"))
        if isinstance(res, dict) and "rec_texts" in res:
            return _flatten(res.get("rec_texts"))
        try:
            rec_texts = res["rec_texts"]
        except Exception:
            rec_texts = None
        if rec_texts is not None:
            return _flatten(rec_texts)
        return []

    def _predict(self, input_value: Any) -> list[Any]:
        ocr = self._get_ocr()
        return ocr.predict(input=input_value, **self._predict_kwargs)

    def ocr_image_bytes(self, image_bytes: bytes, filename: str = "image.png") -> list[str]:
        import cv2  # type: ignore
        import numpy as np

        array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image bytes: {filename}")
        results = self._predict(image)
        self._maybe_save_results(results)
        texts: list[str] = []
        for res in results:
            texts.extend(self._extract_texts_from_result(res))
        return texts

    def ocr_image_path(self, image_path: Path) -> list[str]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        results = self._predict(str(image_path))
        self._maybe_save_results(results)
        texts: list[str] = []
        for res in results:
            texts.extend(self._extract_texts_from_result(res))
        return texts

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
                self._ocr = LocalPPOCRClient(**self._kwargs)
            elif self._backend in {"remote", "server", "http"}:
                self._ocr = RemoteOCRClient(**self._kwargs)
            else:
                raise ValueError(f"Unknown OCR backend: {self._backend}")
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
