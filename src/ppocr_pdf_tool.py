from __future__ import annotations

import base64
from io import BytesIO
import json
import mimetypes
import os
from pathlib import Path
from typing import Iterable
from urllib import error, request


_DEFAULT_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
    "ASsJTYQAAAAASUVORK5CYII="
)


class PdfImageTool:
    def __init__(self, image_dpi: int = 200) -> None:
        self.image_dpi = image_dpi

    def iter_pdf_images(self, pdf_path: Path) -> Iterable[tuple[int, bytes, str]]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            import fitz  # type: ignore
        except ImportError:
            fitz = None

        if fitz is not None:
            zoom = self.image_dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            with fitz.open(str(pdf_path)) as doc:
                for index in range(len(doc)):
                    page = doc.load_page(index)
                    pix = page.get_pixmap(matrix=matrix)
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
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            filename = f"{pdf_path.stem}_page_{index + 1}.png"
            yield index, buffer.getvalue(), filename


class LocalPPOCRTool:
    def __init__(
        self,
        lang: str = "ch",
        use_angle_cls: bool = True,
        use_gpu: bool | None = None,
        **kwargs,
    ) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "Missing paddleocr. Install paddleocr and a compatible paddlepaddle."
            ) from exc
        if use_gpu is None:
            use_gpu = self._infer_use_gpu()
        self._ocr = PaddleOCR(
            lang=lang, use_angle_cls=use_angle_cls, use_gpu=use_gpu, **kwargs
        )

    @staticmethod
    def _infer_use_gpu() -> bool:
        env = os.getenv("PPOCR_USE_GPU")
        if env:
            return env.strip().lower() in ("1", "true", "yes", "on")
        return False

    def ocr_image_bytes(self, image_bytes: bytes) -> list[str]:
        image = self._decode_image(image_bytes)
        result = self._ocr.ocr(image, cls=True)
        return self._extract_texts(result)

    @staticmethod
    def ocr_image_bytes_remote(
        image_bytes: bytes | None = None,
        *,
        image_path: Path | None = None,
        filename: str | None = None,
        url: str | None = None,
        base_url: str | None = None,
        endpoint: str | None = None,
        request_format: str | None = None,
        timeout: float | None = None,
        file_field: str | None = None,
    ) -> dict | list:
        image_bytes, filename = LocalPPOCRTool._resolve_remote_image(
            image_bytes, image_path, filename
        )
        url = LocalPPOCRTool._resolve_remote_url(url, base_url, endpoint)
        request_format = LocalPPOCRTool._resolve_request_format(request_format)
        timeout_value = LocalPPOCRTool._resolve_timeout(timeout)
        file_field = LocalPPOCRTool._resolve_file_field(file_field)

        errors: list[str] = []

        if request_format in ("auto", "multipart"):
            status, body, err = LocalPPOCRTool._post_multipart(
                url, image_bytes, filename, file_field, timeout_value
            )
            parsed = LocalPPOCRTool._parse_json(body) if body else None
            if LocalPPOCRTool._is_success(status, parsed):
                return parsed
            errors.append(
                f"multipart {LocalPPOCRTool._format_error(status, err, body)}"
            )

        if request_format in ("auto", "json"):
            for payload in LocalPPOCRTool._json_payloads(image_bytes):
                status, body, err = LocalPPOCRTool._post_json(
                    url, payload, timeout_value
                )
                parsed = LocalPPOCRTool._parse_json(body) if body else None
                if LocalPPOCRTool._is_success(status, parsed):
                    return parsed
                errors.append(
                    f"json {LocalPPOCRTool._format_error(status, err, body)}"
                )

        error_details = "\n".join(errors) if errors else "No response"
        raise RuntimeError(f"PPOCR request failed. url={url}\n{error_details}")

    def ocr_image_path(self, image_path: Path) -> list[str]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.ocr_image_bytes(image_path.read_bytes())

    def ocr_pdf(self, pdf_path: Path, image_dpi: int = 200) -> list[str]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        text_pages = self._extract_pdf_text(pdf_path)
        if text_pages is not None:
            return text_pages

        tool = PdfImageTool(image_dpi=image_dpi)
        page_texts: list[str] = []
        for _, image_bytes, _ in tool.iter_pdf_images(pdf_path):
            texts = self.ocr_image_bytes(image_bytes)
            page_texts.append("\n".join(texts))
        return page_texts

    @staticmethod
    def _decode_image(image_bytes: bytes):
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "Missing image deps. Install opencv-python and numpy for local OCR."
            ) from exc

        data = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image bytes.")
        return image

    @staticmethod
    def _resolve_remote_image(
        image_bytes: bytes | None,
        image_path: Path | None,
        filename: str | None,
    ) -> tuple[bytes, str]:
        if image_path is None:
            env_path = os.getenv("PPOCR_IMAGE_PATH")
            if env_path:
                image_path = Path(env_path)
        if image_path is not None:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"PPOCR image not found: {image_path}")
            image_bytes = image_path.read_bytes()
            filename = filename or image_path.name
        if image_bytes is None:
            image_bytes = base64.b64decode(_DEFAULT_PNG_BASE64)
        if not filename:
            filename = "inline.png"
        return image_bytes, filename

    @staticmethod
    def _resolve_remote_url(
        url: str | None,
        base_url: str | None,
        endpoint: str | None,
    ) -> str:
        if url:
            return url
        env_url = os.getenv("PPOCR_URL")
        if env_url:
            return env_url
        if base_url is None:
            base_url = os.getenv("PPOCR_BASE_URL", "http://10.0.22.109:8001")
        if endpoint is None:
            endpoint = os.getenv("PPOCR_ENDPOINT", "/ocr")
        base_url = base_url.rstrip("/")
        return LocalPPOCRTool._build_url(base_url, endpoint)

    @staticmethod
    def _build_url(base_url: str, endpoint: str) -> str:
        if not endpoint:
            return base_url
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return f"{base_url}{endpoint}"

    @staticmethod
    def _resolve_request_format(request_format: str | None) -> str:
        if request_format is None:
            request_format = os.getenv("PPOCR_REQUEST_FORMAT", "auto")
        request_format = request_format.strip().lower()
        if request_format not in ("auto", "multipart", "json"):
            return "auto"
        return request_format

    @staticmethod
    def _resolve_timeout(timeout: float | None) -> float:
        if timeout is None:
            timeout = float(os.getenv("PPOCR_TIMEOUT", "20"))
        return float(timeout)

    @staticmethod
    def _resolve_file_field(file_field: str | None) -> str:
        if not file_field:
            return os.getenv("PPOCR_FILE_FIELD", "file")
        return file_field

    @staticmethod
    def _post_json(url: str, payload: dict, timeout: float):
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        return LocalPPOCRTool._send_request(req, timeout)

    @staticmethod
    def _post_multipart(
        url: str,
        image_bytes: bytes,
        filename: str,
        file_field: str,
        timeout: float,
    ):
        boundary = f"----ppocrboundary{os.urandom(8).hex()}"
        mime = "image/png"
        guess, _ = mimetypes.guess_type(filename)
        if guess:
            mime = guess
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
        footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
        body = header + image_bytes + footer

        req = request.Request(
            url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return LocalPPOCRTool._send_request(req, timeout)

    @staticmethod
    def _send_request(req, timeout: float):
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read(), None
        except error.HTTPError as exc:
            return exc.code, exc.read(), exc
        except error.URLError as exc:
            return None, None, exc

    @staticmethod
    def _json_payloads(image_bytes: bytes) -> list[dict]:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{encoded}"
        return [
            {"images": [encoded]},
            {"images": [data_url]},
            {"image": encoded},
            {"image": data_url},
        ]

    @staticmethod
    def _parse_json(body: bytes):
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _body_preview(body: bytes, limit: int = 2000) -> str:
        if not body:
            return "<empty>"
        text = body.decode("utf-8", errors="replace")
        if len(text) > limit:
            return f"{text[:limit]}...[truncated]"
        return text

    @staticmethod
    def _looks_successful(parsed) -> bool:
        if parsed is None:
            return False
        if isinstance(parsed, dict):
            error_code = parsed.get("error_code")
            if error_code not in (None, 0, "0"):
                return False
            if parsed.get("error"):
                return False
        return True

    @staticmethod
    def _is_success(status: int | None, parsed) -> bool:
        return (
            status is not None
            and 200 <= status < 300
            and LocalPPOCRTool._looks_successful(parsed)
        )

    @staticmethod
    def _format_error(status: int | None, err, body: bytes | None) -> str:
        preview = LocalPPOCRTool._body_preview(body or b"")
        return f"status={status} err={err} body={preview}"

    @staticmethod
    def _is_line_item(item) -> bool:
        return (
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and isinstance(item[1], (list, tuple))
            and item[1]
            and isinstance(item[1][0], str)
        )

    @classmethod
    def _extract_texts(cls, result) -> list[str]:
        if not result:
            return []

        pages = result
        if isinstance(pages, list) and pages and cls._is_line_item(pages[0]):
            pages = [pages]

        texts: list[str] = []
        for page in pages:
            if not isinstance(page, list):
                continue
            for line in page:
                if cls._is_line_item(line):
                    texts.append(line[1][0])
        return texts

    @staticmethod
    def _extract_pdf_text(pdf_path: Path) -> list[str] | None:
        try:
            import fitz  # type: ignore
        except ImportError:
            return None

        pages: list[str] = []
        with fitz.open(str(pdf_path)) as doc:
            for page in doc:
                text = page.get_text("text").strip()
                pages.append(text)
        if not pages:
            return None
        if any(page for page in pages):
            return pages
        return None


class RemotePPOCRTool:
    def __init__(self) -> None:
        pass

    def ocr_image_bytes(self, image_bytes: bytes) -> list[str]:
        payload = LocalPPOCRTool.ocr_image_bytes_remote(image_bytes=image_bytes)
        return self._extract_remote_texts(payload)

    def ocr_image_path(self, image_path: Path) -> list[str]:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        payload = LocalPPOCRTool.ocr_image_bytes_remote(image_path=image_path)
        return self._extract_remote_texts(payload)

    def ocr_pdf(self, pdf_path: Path, image_dpi: int = 200) -> list[str]:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        text_pages = LocalPPOCRTool._extract_pdf_text(pdf_path)
        if text_pages is not None:
            return text_pages

        tool = PdfImageTool(image_dpi=image_dpi)
        page_texts: list[str] = []
        for _, image_bytes, _ in tool.iter_pdf_images(pdf_path):
            lines = self.ocr_image_bytes(image_bytes)
            page_texts.append("\n".join(lines))
        return page_texts

    @classmethod
    def _extract_remote_texts(cls, payload) -> list[str]:
        if payload is None:
            return []
        if isinstance(payload, list):
            if payload and all(isinstance(item, dict) for item in payload):
                texts: list[str] = []
                for item in payload:
                    value = item.get("text") or item.get("word")
                    if isinstance(value, str):
                        texts.append(value)
                if texts:
                    return texts
            if payload and all(isinstance(item, str) for item in payload):
                return [item for item in payload if item]
            return LocalPPOCRTool._extract_texts(payload)
        if isinstance(payload, dict):
            for key in ("result", "results", "data", "ocr_result"):
                if key in payload:
                    return cls._extract_remote_texts(payload[key])
            text = payload.get("text")
            if isinstance(text, str):
                return [text]
            texts = payload.get("texts")
            if isinstance(texts, list):
                return [item for item in texts if isinstance(item, str)]
            lines = payload.get("lines") or payload.get("words")
            if isinstance(lines, list):
                extracted: list[str] = []
                for item in lines:
                    if isinstance(item, str):
                        extracted.append(item)
                    elif isinstance(item, dict):
                        value = item.get("text") or item.get("word")
                        if isinstance(value, str):
                            extracted.append(value)
                    elif LocalPPOCRTool._is_line_item(item):
                        extracted.append(item[1][0])
                if extracted:
                    return extracted
        return []
