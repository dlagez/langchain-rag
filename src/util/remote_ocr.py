from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib import error, request

from .ocr_common import _env_first
from .prompt_logger import log_event


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
        request_info = {
            "url": self._url,
            "timeout": self._timeout,
            "file_field": self._file_field,
            "filename": filename,
            "mime": mime,
            "bytes_len": len(image_bytes),
        }
        try:
            with request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            log_event(
                "ocr",
                request=request_info,
                response={"status": exc.code, "raw": detail},
                error=f"OCR HTTP {exc.code}",
            )
            raise RuntimeError(f"OCR HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            log_event(
                "ocr",
                request=request_info,
                response=None,
                error=str(exc),
            )
            raise RuntimeError(f"OCR request failed: {exc}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            log_event(
                "ocr",
                request=request_info,
                response={"raw": raw},
                error="invalid_json",
            )
            raise RuntimeError("OCR server returned invalid JSON.") from exc
        log_event(
            "ocr",
            request=request_info,
            response=payload,
        )
        return payload

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

    def ocr_image_path(self, image_path):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return self.ocr_image_bytes(image_path.read_bytes(), filename=image_path.name)
