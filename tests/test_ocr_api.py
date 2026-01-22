import json
import mimetypes
import os
import unittest
from pathlib import Path
from urllib import error, request

from dotenv import load_dotenv


def call_remote_ocr(
    image_path: Path,
    *,
    url: str,
    timeout: float | None = None,
    file_field: str = "file",
) -> tuple[int | None, dict | list | None, str]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image_bytes = image_path.read_bytes()
    filename = image_path.name
    boundary = f"----ocrboundary{os.urandom(8).hex()}"
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
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            print(raw)
            status = resp.status
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except Exception as exc:
        return None, None, f"OCR request failed: {exc}"

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    return status, parsed, raw


class TestRemoteOCRClientReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv()

    def test_remote_ocr_real_data(self) -> None:
        image_path = os.getenv("OCR_IMAGE_PATH") or os.getenv("PPOCR_IMAGE_PATH")
        if not image_path:
            self.skipTest("Set OCR_IMAGE_PATH to run remote OCR test.")
        path = Path(image_path)
        if not path.exists():
            self.fail(f"OCR_IMAGE_PATH not found: {path}")

        url = os.getenv("OCR_URL") or os.getenv("PPOCR_URL")
        timeout = os.getenv("OCR_TIMEOUT") or os.getenv("PPOCR_TIMEOUT")
        file_field = os.getenv("OCR_FILE_FIELD") or os.getenv("PPOCR_FILE_FIELD")
        expected_substr = os.getenv("OCR_EXPECT_SUBSTR")
        if not url:
            self.skipTest("Set OCR_URL to run remote OCR test.")

        status, parsed, raw = call_remote_ocr(
            path,
            url=url,
            timeout=float(timeout) if timeout else None,
            file_field=file_field or "file",
        )
        self.assertTrue(status and 200 <= status < 300, f"OCR request failed: {status}")
        self.assertIsNotNone(parsed, f"OCR response is not JSON: {raw[:200]}")

        if expected_substr and parsed is not None:
            raw_json = json.dumps(parsed, ensure_ascii=False)
            self.assertIn(expected_substr, raw_json)


if __name__ == "__main__":
    unittest.main()
