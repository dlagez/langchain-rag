# LangChain RAG Starter

Minimal RAG project skeleton using LangChain + Google Gemini embeddings + Qdrant.

## Quickstart (PowerShell)
1. Activate venv: `.\.venv\Scripts\Activate.ps1`
2. Install deps: `python -m pip install -r requirements.txt`
3. Create `.env` with your key: `GOOGLE_API_KEY=...`
4. (Optional) Set `GOOGLE_MODEL` to a supported model, e.g. `gemini-1.5-flash`
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

Supported types:
- `txt`, `pdf`, `docx`, `doc` (via textract/Word/LibreOffice), `xls`, `xlsx`, `csv`, common images

## Environment variables
Required:
- `GOOGLE_API_KEY`: Gemini Developer API key.

Optional:
- `GOOGLE_MODEL`: chat model name.
- `GOOGLE_EMBEDDING_MODEL`: embedding model name.
- `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`: chunking config.
- `RAG_ALPHA`: hybrid retrieval mix (0-1).
- `QDRANT_PATH`: local Qdrant storage path.
- `QDRANT_COLLECTION`: collection name for vectors.
- `QDRANT_URL`: connect to a Qdrant server instead of local mode.
- `QDRANT_API_KEY`: API key for Qdrant server.
- `PPOCR_USE_GPU`: set to `1` to enable GPU OCR when supported.


INFO: HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents "HTTP/1.1 200 OK"
