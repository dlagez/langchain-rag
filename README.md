# LangChain RAG Starter

Minimal RAG project skeleton using LangChain + Google Gemini embeddings + Qdrant.

## Quickstart (PowerShell)
1. Activate venv: `.\.venv\Scripts\Activate.ps1`
2. Install deps: `python -m pip install -r requirements.txt`
3. Create `.env` with your key: `GOOGLE_API_KEY=...`
4. (Optional) Set `GOOGLE_MODEL` to a supported model, e.g. `gemini-2.5-flash`
5. Put source files in `data/source/`
6. Run: `python src/rag_demo.py "你的问题"`

Notes:
- The vector store persists to `index/`. Delete that folder or use `--rebuild` to rebuild.
- Processed text outputs are written to `data/processed/`.

## Qdrant (local)
The vector store uses Qdrant in local mode by default, persisted under `index/qdrant`.
To use a running Qdrant server instead, set `QDRANT_URL` (and `QDRANT_API_KEY` if needed).

## Local PPOCR
`src/ppocr_pdf_tool.py` provides local OCR utilities using PaddleOCR:
- `PdfImageTool` renders PDF pages to PNG bytes.
- `LocalPPOCRTool` runs OCR on images or PDFs and returns extracted text.

Notes:
- PDF rendering needs either `pymupdf` (fitz) or `pdf2image` + poppler installed.
- PaddleOCR requires a compatible `paddlepaddle` build for your platform.
- To enable GPU OCR, install `paddlepaddle-gpu` and set `PPOCR_USE_GPU=1` in `.env`.

## Source Processing
`src/rag_demo.py` now reads raw files from `data/source/`, converts them to text,
and writes the processed output into `data/processed/`.
These two folders are the default input/output locations for the processing tools.

Supported types:
- `txt`, `pdf`, `docx`, `doc` (via textract/Word/LibreOffice), `xls`, `xlsx`, `csv`, common images

## Environment variables
Required:
- `GOOGLE_API_KEY`: Gemini Developer API key.

Optional:
- `GOOGLE_MODEL`: chat model name (default: `gemini-2.5-flash`).
- `GOOGLE_EMBEDDING_MODEL`: embedding model name (default: `text-embedding-004`).
- `RAG_CHUNK_SIZE`: chunk size (default: `800`).
- `RAG_CHUNK_OVERLAP`: chunk overlap (default: `100`).
- `RAG_ALPHA`: hybrid retrieval mix (default: `0.7`).
- `QDRANT_PATH`: local Qdrant storage path (default: `index/qdrant`).
- `QDRANT_COLLECTION`: collection name for vectors (default: `rag_docs`).
- `QDRANT_URL`: connect to a Qdrant server instead of local mode.
- `QDRANT_API_KEY`: API key for Qdrant server.
- `PPOCR_USE_GPU`: set to `1` to enable GPU OCR when supported (default: off).
- `PPOCR_URL`: full PPOCR endpoint URL (default: built from base + endpoint).
- `PPOCR_BASE_URL`: PPOCR base URL (default: `http://10.0.22.109:8001`).
- `PPOCR_ENDPOINT`: PPOCR path (default: `/ocr`).
- `PPOCR_REQUEST_FORMAT`: `auto`/`multipart`/`json` (default: `auto`).
- `PPOCR_TIMEOUT`: request timeout seconds (default: `10`).
- `PPOCR_FILE_FIELD`: multipart file field name (default: `file`).
- `PPOCR_IMAGE_PATH`: default image path for PPOCR API tests/requests.

Copy-ready example:
```bash
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemini-2.5-flash
GOOGLE_EMBEDDING_MODEL=text-embedding-004
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=100
RAG_ALPHA=0.7
QDRANT_PATH=index/qdrant
QDRANT_COLLECTION=rag_docs
PPOCR_USE_GPU=0
PPOCR_BASE_URL=http://10.0.22.109:8001
PPOCR_ENDPOINT=/ocr
PPOCR_REQUEST_FORMAT=auto
PPOCR_TIMEOUT=10
PPOCR_FILE_FIELD=file
# PPOCR_URL=http://10.0.22.109:8001/ocr
# PPOCR_IMAGE_PATH=data/source/sample.png
```


INFO: HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents "HTTP/1.1 200 OK"
