import argparse
import datetime
import json
import mimetypes
import os
import sys
import unittest
from pathlib import Path
from urllib import error, request

from dotenv import load_dotenv


def _apply_cli_env_overrides():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image", dest="image")
    parser.add_argument("--url", dest="url")
    parser.add_argument("--timeout", dest="timeout")
    parser.add_argument("--file-field", dest="file_field")
    parser.add_argument("--output-dir", dest="output_dir")
    args, remaining = parser.parse_known_args()

    if args.image:
        os.environ["OCR_IMAGE_PATH"] = args.image
    if args.url:
        os.environ["OCR_URL"] = args.url
    if args.timeout:
        os.environ["OCR_TIMEOUT"] = args.timeout
    if args.file_field:
        os.environ["OCR_FILE_FIELD"] = args.file_field
    if args.output_dir:
        os.environ["OCR_OUTPUT_DIR"] = args.output_dir

    return remaining


class TestOCRApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv()
        cls.url = os.getenv("OCR_URL", "http://10.0.22.109:8081/ocr")
        cls.timeout = float(os.getenv("OCR_TIMEOUT", "30"))
        cls.file_field = os.getenv("OCR_FILE_FIELD", "file")
        cls.output_dir = Path(os.getenv("OCR_OUTPUT_DIR", "ppocr_results"))
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_image_bytes(self):
        image_path = os.getenv("OCR_IMAGE_PATH")
        if not image_path:
            self.fail("OCR_IMAGE_PATH is required for OCR API test.")
        path = Path(image_path)
        if not path.exists():
            self.fail(f"OCR_IMAGE_PATH not found: {path}")
        return path.read_bytes(), path.name

    def _post_multipart(self, image_bytes: bytes, filename: str):
        boundary = f"----ocrboundary{os.urandom(8).hex()}"
        mime = "image/png"
        guess, _ = mimetypes.guess_type(filename)
        if guess:
            mime = guess
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{self.file_field}"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
        footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
        body = header + image_bytes + footer

        req = request.Request(
            self.url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, resp.read(), None
        except error.HTTPError as exc:
            return exc.code, exc.read(), exc
        except error.URLError as exc:
            return None, None, exc

    def _parse_json(self, body: bytes):
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_result(self, filename: str, parsed, body: bytes):
        stem = Path(filename).stem or "ocr_result"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"{stem}_{timestamp}.json"
        if parsed is None:
            payload = {"raw": body.decode("utf-8", errors="replace")}
        else:
            payload = parsed
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved result to {out_path}")

    def test_ocr_endpoint(self) -> None:
        image_bytes, filename = self._load_image_bytes()
        status, body, err = self._post_multipart(image_bytes, filename)
        parsed = self._parse_json(body) if body else None
        if parsed is not None:
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
        elif body:
            print(body.decode("utf-8", errors="replace"))
        if status and 200 <= status < 300 and parsed is not None:
            self._write_result(filename, parsed, body)
            return
        detail = body[:200] if body else b""
        self.fail(f"OCR request failed. url={self.url} status={status} err={err} body={detail}")


if __name__ == "__main__":
    remaining_args = _apply_cli_env_overrides()
    unittest.main(argv=[sys.argv[0]] + remaining_args)


# python -m unittest tests.test_ocr_api
