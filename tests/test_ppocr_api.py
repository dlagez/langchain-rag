import argparse
import base64
import datetime
import json
import mimetypes
import os
import sys
import unittest
from pathlib import Path
from urllib import error, request

from dotenv import load_dotenv


_DEFAULT_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMA"
    "ASsJTYQAAAAASUVORK5CYII="
)


def _apply_cli_env_overrides():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--image", dest="image")
    parser.add_argument("--url", dest="url")
    parser.add_argument("--base-url", dest="base_url")
    parser.add_argument("--endpoint", dest="endpoint")
    parser.add_argument("--format", dest="request_format")
    parser.add_argument("--timeout", dest="timeout")
    parser.add_argument("--file-field", dest="file_field")
    parser.add_argument("--output-dir", dest="output_dir")
    args, remaining = parser.parse_known_args()

    if args.image:
        os.environ["PPOCR_IMAGE_PATH"] = args.image
    if args.url:
        os.environ["PPOCR_URL"] = args.url
    if args.base_url:
        os.environ["PPOCR_BASE_URL"] = args.base_url
    if args.endpoint:
        os.environ["PPOCR_ENDPOINT"] = args.endpoint
    if args.request_format:
        os.environ["PPOCR_REQUEST_FORMAT"] = args.request_format
    if args.timeout:
        os.environ["PPOCR_TIMEOUT"] = args.timeout
    if args.file_field:
        os.environ["PPOCR_FILE_FIELD"] = args.file_field
    if args.output_dir:
        os.environ["PPOCR_OUTPUT_DIR"] = args.output_dir

    return remaining


class TestPPOCRApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv()
        cls.base_url = os.getenv("PPOCR_BASE_URL", "http://10.0.22.109:8001").rstrip("/")
        cls.endpoint = os.getenv("PPOCR_ENDPOINT", "/ocr")
        cls.url = os.getenv("PPOCR_URL") or cls._build_url(cls.base_url, cls.endpoint)
        cls.request_format = os.getenv("PPOCR_REQUEST_FORMAT", "auto").lower()
        cls.timeout = float(os.getenv("PPOCR_TIMEOUT", "10"))
        cls.file_field = os.getenv("PPOCR_FILE_FIELD", "file")
        cls.output_dir = Path(os.getenv("PPOCR_OUTPUT_DIR", "ppocr_results"))
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _build_url(base_url: str, endpoint: str) -> str:
        if not endpoint:
            return base_url
        if not endpoint.startswith("/"):
            endpoint = f"/{endpoint}"
        return f"{base_url}{endpoint}"

    def _load_image_bytes(self):
        image_path = os.getenv("PPOCR_IMAGE_PATH")
        if image_path:
            path = Path(image_path)
            if not path.exists():
                self.fail(f"PPOCR_IMAGE_PATH not found: {path}")
            return path.read_bytes(), path.name
        return base64.b64decode(_DEFAULT_PNG_BASE64), "inline.png"

    def _post_json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        return self._send_request(req)

    def _post_multipart(self, image_bytes: bytes, filename: str):
        boundary = f"----ppocrboundary{os.urandom(8).hex()}"
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
        return self._send_request(req)

    def _send_request(self, req):
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, resp.read(), None
        except error.HTTPError as exc:
            return exc.code, exc.read(), exc
        except error.URLError as exc:
            return None, None, exc

    def _json_payloads(self, image_bytes: bytes):
        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:image/png;base64,{encoded}"
        return [
            {"images": [encoded]},
            {"images": [data_url]},
            {"image": encoded},
            {"image": data_url},
        ]

    def _parse_json(self, body: bytes):
        if not body:
            return None
        try:
            return json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None

    def _body_preview(self, body: bytes, limit: int = 2000) -> str:
        if not body:
            return "<empty>"
        text = body.decode("utf-8", errors="replace")
        if len(text) > limit:
            return f"{text[:limit]}...[truncated]"
        return text

    def _looks_successful(self, parsed):
        if parsed is None:
            return False
        if isinstance(parsed, dict):
            error_code = parsed.get("error_code")
            if error_code not in (None, 0, "0"):
                return False
            if parsed.get("error"):
                return False
        return True

    def _write_result(self, filename: str, parsed, body: bytes):
        stem = Path(filename).stem or "ppocr_result"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.output_dir / f"{stem}_{timestamp}.json"
        if parsed is None:
            payload = {"raw": self._body_preview(body, limit=200000)}
        else:
            payload = parsed
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved result to {out_path}")

    def test_ppocr_endpoint(self) -> None:
        image_bytes, filename = self._load_image_bytes()
        errors = []

        if self.request_format in ("auto", "multipart"):
            status, body, err = self._post_multipart(image_bytes, filename)
            parsed = self._parse_json(body) if body else None
            print(
                f"multipart status={status} err={err} body={self._body_preview(body)}"
            )
            if status and 200 <= status < 300 and self._looks_successful(parsed):
                self._write_result(filename, parsed, body)
                return
            errors.append(
                f"multipart status={status} err={err} body={body[:200] if body else b''}"
            )

        error_details = "\n".join(errors) if errors else "No response"
        self.fail(f"PPOCR request failed. url={self.url}\n{error_details}")


if __name__ == "__main__":
    remaining_args = _apply_cli_env_overrides()
    unittest.main(argv=[sys.argv[0]] + remaining_args)
