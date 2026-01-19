# LangChain RAG Starter

Minimal RAG project skeleton using LangChain + Google Gemini or Alibaba Bailian
(DashScope) embeddings + Qdrant.

## Quickstart (PowerShell)
1. Activate venv: `.\.venv\Scripts\Activate.ps1`
2. Install deps: `python -m pip install -r requirements.txt`
3. Create `.env` with your key (Google: `GOOGLE_API_KEY=...` or Bailian:
   `BAILIAN_API_KEY=...` + `RAG_PROVIDER=bailian`)
4. (Optional) Set your model (`GOOGLE_MODEL` or `BAILIAN_MODEL`)
5. Put source files in `data/source/<process_id>/form/` and
   `data/source/<process_id>/attachment/`
6. Run: `python src/rag-contract.py "your question"`

Notes:
- The vector store persists to `index/`. Delete that folder or use `--rebuild` to rebuild.
- Processed text outputs are written to `data/processed/`.

## Provider switching
Set `RAG_PROVIDER` to select a default for both the LLM and embeddings.
Use `RAG_LLM_PROVIDER` or `RAG_EMBEDDING_PROVIDER` to override either one.

Example (Bailian):
```bash
RAG_PROVIDER=bailian
BAILIAN_API_KEY=your_key_here
BAILIAN_MODEL=qwen-plus
BAILIAN_EMBEDDING_MODEL=text-embedding-v2
```

Example (mix providers):
```bash
RAG_LLM_PROVIDER=bailian
RAG_EMBEDDING_PROVIDER=google
```

## Qdrant (local)
The vector store uses Qdrant in local mode by default, persisted under `index/qdrant`.
To use a running Qdrant server instead, set `QDRANT_URL` (and `QDRANT_API_KEY` if needed).

## Vector store structure
Data is stored in a single Qdrant collection (name from `QDRANT_COLLECTION`, default
`contract_approval_rag`). Each chunk becomes one point with a vector and payload.

Payload shape:
- `page_content`: chunk text
- `metadata`: fields like `source`, `doc_id`, `source_type`, `field`, `page`,
  `chunk_id`, `doc_type_hint`, `process_id`, `created_at`, `filename`

To inspect stored chunks directly (no re-chunking), use:
```bash
python tests/inspect_chunks.py --limit 5 --max-chars 400
```
More examples:
```bash
# filter by source filename
python tests/inspect_chunks.py --source-contains 合同 --limit 3 --max-chars 0

# filter by exact doc_id
python tests/inspect_chunks.py --doc-id "attachment/合同文件.pdf" --limit 3 --max-chars 0

# filter by doc_id substring
python tests/inspect_chunks.py --doc-id-contains "合同12.12.pdf" --limit 3 --max-chars 0
```

Notes:
- `index/manifest.json` includes `doc_count` for the current collection.

## Local PPOCR
`src/ppocr_pdf_tool.py` provides local OCR utilities using PaddleOCR:
- `PdfImageTool` renders PDF pages to PNG bytes.
- `LocalPPOCRTool` runs OCR on images or PDFs and returns extracted text.

Notes:
- PDF rendering needs either `pymupdf` (fitz) or `pdf2image` + poppler installed.
- PaddleOCR requires a compatible `paddlepaddle` build for your platform.
- To enable GPU OCR, install `paddlepaddle-gpu` and set `PPOCR_USE_GPU=1` in `.env`.

## Source Processing
`src/rag-contract.py` reads raw files from `data/source/<process_id>/form|attachment`
(files under `data/source/` root are ignored),
converts them to text,
and writes the processed output into `data/processed/`.
These two folders are the default input/output locations for the processing tools.

Directory layout:
```
data/source/
  <process_id>/
    form/
      ...
    attachment/
      ...
```

Supported types:
- `txt`, `pdf`, `docx`, `doc` (via textract/Word/LibreOffice), `xls`, `xlsx`, `csv`, common images

## Chunking strategy
Structured chunking is applied before indexing:
- Contract-like attachments are split by chapter/section/article headings when possible,
  with length-based fallback using overlap.
- Checklist-like attachments are grouped by item lines (denser, shorter chunks).
- Signature pages are isolated when signature keywords are detected.

Key knobs:
- `RAG_CHUNK_SIZE` sets the target max chunk length for contract-style text.
- `RAG_CHUNK_OVERLAP` is used when length-based splitting is required.

## Environment variables
Required (choose provider):
- `GOOGLE_API_KEY`: Gemini Developer API key.
- `BAILIAN_API_KEY` / `DASHSCOPE_API_KEY`: Alibaba Bailian API key.

Optional:
- `RAG_PROVIDER`: default provider for LLM + embeddings (`google`/`bailian`).
- `RAG_LLM_PROVIDER`: override LLM provider (`google`/`bailian`).
- `RAG_EMBEDDING_PROVIDER`: override embeddings provider (`google`/`bailian`).
- `RAG_FORM_RETRIEVAL_QUERY`: retrieval-only query for form extraction (falls back to `RAG_FORM_QUESTION`).
- `RAG_ATTACHMENT_RETRIEVAL_QUERY`: retrieval-only query for attachment extraction (falls back to `RAG_ATTACHMENT_QUESTION`).
- `RAG_LOG_LEVEL`: application log level (default: `INFO`).
- `RAG_LOG_REQUESTS`: set to `1` to save provider request logs (currently Bailian, includes request URL).
- `RAG_LOG_DIR`: directory to save full request logs (default: `data/log`).
- `GOOGLE_MODEL`: chat model name (default: `gemini-2.5-flash`).
- `GOOGLE_EMBEDDING_MODEL`: embedding model name (default: `text-embedding-004`).
- `BAILIAN_MODEL`: chat model name (default: `qwen-plus`).
- `BAILIAN_EMBEDDING_MODEL`: embedding model name (default: `text-embedding-v2`).
- `DASHSCOPE_MODEL` / `DASHSCOPE_EMBEDDING_MODEL`: aliases for Bailian models.
- `RAG_CHUNK_SIZE`: target max chunk length for structured chunking (default: `800`).
- `RAG_CHUNK_OVERLAP`: overlap used for length-based splits (default: `100`).
- `RAG_ALPHA`: hybrid retrieval mix (default: `0.7`).
- `QDRANT_PATH`: local Qdrant storage path (default: `index/qdrant`).
- `QDRANT_COLLECTION`: collection name for vectors (default: `contract_approval_rag`).
- `QDRANT_URL`: connect to a Qdrant server instead of local mode.
- `QDRANT_API_KEY`: API key for Qdrant server.
- `QUESTION`: default prompt text when no CLI question is provided.
- `PROCESS_ID` / `RAG_PROCESS_ID`: process instance ID for metadata filtering (required when multiple process folders exist).
- `RAG_CREATED_AT` / `CREATED_AT`: optional created_at metadata value.
- `PPOCR_USE_GPU`: set to `1` to enable GPU OCR when supported (default: off).
- `PPOCR_URL`: full PPOCR endpoint URL (default: built from base + endpoint).
- `PPOCR_BASE_URL`: PPOCR base URL (default: `http://10.0.22.109:8001`).
- `PPOCR_ENDPOINT`: PPOCR path (default: `/ocr`).
- `PPOCR_REQUEST_FORMAT`: `auto`/`multipart`/`json` (default: `auto`).
- `PPOCR_TIMEOUT`: request timeout seconds (default: `20`).
- `PPOCR_FILE_FIELD`: multipart file field name (default: `file`).
- `PPOCR_IMAGE_PATH`: default image path for PPOCR API tests/requests.
- `PPOCR_MODE`: OCR backend (`local`/`remote`/`auto`, default: `local`).

Copy-ready example (Google):
```bash
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemini-2.5-flash
PPOCR_TIMEOUT=60

RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=100
RAG_ALPHA=0.7
QDRANT_PATH=index/qdrant
QDRANT_COLLECTION=contract_approval_rag
QUESTION="What is the VAT rate stated in the contract?\nOnly use values explicitly present."
PROCESS_ID=your_process_id
PPOCR_USE_GPU=0
PPOCR_BASE_URL=http://10.0.22.109:8001
PPOCR_ENDPOINT=/ocr
PPOCR_REQUEST_FORMAT=auto
PPOCR_FILE_FIELD=file
# PPOCR_URL=http://10.0.22.109:8001/ocr
# PPOCR_IMAGE_PATH=data/source/sample.png
```


INFO: HTTP Request: POST https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents "HTTP/1.1 200 OK"
