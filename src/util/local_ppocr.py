from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .ocr_common import _env_bool, _env_optional_path


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
