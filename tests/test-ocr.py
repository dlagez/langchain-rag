from pathlib import Path

# Initialize PaddleOCR instance
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# Run OCR inference on a sample image 
result = ocr.predict(
    input=r"D:\code3\langchain-rag\data\source\S5819\form\PixPin_2025-12-04_10-03-02.jpg")

print("type(res)=", type(result), "len=", len(result) if isinstance(result, list) else "NA")


# Visualize the results and save the JSON results
output_dir = Path(__file__).resolve().parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
for res in result:
    res.print()
    res.save_to_img(str(output_dir))
    res.save_to_json(str(output_dir))
