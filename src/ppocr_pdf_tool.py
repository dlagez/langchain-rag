from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from typing import Iterable


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
