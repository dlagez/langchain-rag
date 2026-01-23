from __future__ import annotations

import os
from pathlib import Path


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
