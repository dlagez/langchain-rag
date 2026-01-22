import os
import unittest
from pathlib import Path

from dotenv import load_dotenv

from util.ocr_utils import LocalPPOCRClient


class TestLocalPPOCRPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv()

    def test_local_ocr(self) -> None:
        image_path = os.getenv("OCR_IMAGE_PATH") or os.getenv("PPOCR_IMAGE_PATH")
        if not image_path:
            self.skipTest("Set OCR_IMAGE_PATH to run local OCR test.")
        path = Path(image_path)
        if not path.exists():
            self.fail(f"OCR_IMAGE_PATH not found: {path}")

        ocr = LocalPPOCRClient(save_img=True, save_json=True)
        lines = ocr.ocr_image_path(path)
        self.assertTrue(lines)


if __name__ == "__main__":
    unittest.main()
