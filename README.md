# Personal KB P0 (Bailian + Remote OCR)

This project implements a P0 personal knowledge base with:
- Ingestion from absolute paths
- Parsing + OCR
- Chunking + Vector + BM25 retrieval
- Answers with citations
- Local embedding service (preferred) with Bailian fallback

## Requirements & environment
- Python 3.10+ (tested on 3.12) and pip.
- Network access to required services:
  - Bailian (DashScope) API when `LLM_PROVIDER=bailian`.
  - Local embedding service (OpenAI-compatible `/v1/embeddings`) when `EMBEDDING_PROVIDER=local`.
  - Remote OCR service (`OCR_URL`) for PDF/image OCR.
- Optional: remote Qdrant (`QDRANT_URL`). Default uses local storage under `index/<kb_id>/qdrant`.
- Web UI loads HTMX from CDN; offline usage requires vendoring HTMX into `src/app/static/`.

## Quickstart (PowerShell)
1) Activate venv: `\.\.venv\Scripts\Activate.ps1`
2) Install deps: `python -m pip install -r requirements.txt`
3) Create `.env` (see `.env.example`)
4) Ingest a folder: `python -m app.cli ingest --root "D:\docs\projectA" --kb default`
5) re-ingest: `python -m app.cli ingest --root "D:\docs\projectA" --kb default`
6) Ask a question: `python -m app.cli query --kb default "????"`

## Web UI
Start the UI server:
```
python -m app.web
```
Then open `http://localhost:8000` in a browser.

## How to add a folder
You add a folder by running ingest with an absolute path:
```
python -m app.cli ingest --root "D:\docs\projectA" --kb kb_project_a
```
- `--root` must be an absolute path.
- `--kb` is optional; if omitted, a kb_id is generated and saved to `data/manifest/kb_paths.json`.

To add multiple folders, run ingest per folder:
```
python -m app.cli ingest --root "D:\docs\projectA" --kb kb_project_a
python -m app.cli ingest --root "D:\docs\projectB" --kb kb_project_b
```

## Query
```
python -m app.cli query --kb kb_project_a "????"
```
Or resolve kb_id from root path:
```
python -m app.cli query --root "D:\docs\projectA" "????"
```

## Supported file types
- txt, pdf, docx, doc, xls, xlsx, csv
- images: png, jpg, jpeg, bmp, tif, tiff, webp

## Provider selection
Control providers via env:
- `LLM_PROVIDER=bailian`
- `EMBEDDING_PROVIDER=local` or `EMBEDDING_PROVIDER=bailian`

## Embedding priority
- If `EMBEDDING_PROVIDER=local`, embeddings use the local service (`/v1/embeddings`).
- If `EMBEDDING_PROVIDER=bailian`, embeddings use Bailian.

## OCR
- OCR uses a remote service via `OCR_URL` (multipart upload, default field `file`).
- Local OCR (PaddleOCR) is not wired in P0; keep using remote OCR unless you extend the code.

## Storage layout
- `data/manifest/<kb_id>/` : file status + ingest report + ingest.log
- `data/processed/<kb_id>/` : parsed text
- `index/<kb_id>/` : Qdrant + BM25 index

## Metadata fields (keep in sync with code)
These fields are stored in chunk metadata and/or payload. When you add or change
fields in code, update this section too.

- `source` : original file path (absolute)
- `source_type` : file type suffix (txt/pdf/docx/xlsx/etc.)
- `doc_id` : relative path inside the KB root (e.g. `系统运维/bug反馈表.xlsx`)
- `chunk_id` : chunk sequence id within the document
- `page` : page number for paged sources (pdf/doc/docx)
- `kb_id` : knowledge base id

Optional / table-specific (when implemented):
- `sheet` : Excel sheet name
- `row` : Excel row number (1-based)
- `is_header` : whether the row is a header

## Environment variables (P0)
```
# LLM (Bailian)
LLM_PROVIDER=bailian
BAILIAN_API_KEY=***
BAILIAN_MODEL=qwen-plus

# Embedding (local preferred)
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_BASE_URL=http://10.0.22.109:8002/v1
LOCAL_EMBEDDING_MODEL=/home/zp/models/bge-m3
# Bailian fallback
BAILIAN_EMBEDDING_MODEL=text-embedding-v2

# OCR (remote)
OCR_URL=http://your-ocr-host/ocr
OCR_TIMEOUT=30
OCR_FILE_FIELD=file

# Prompt logging
PROMPT_LOG_EMBEDDING=1
PROMPT_LOG_RETRIEVAL=1

# Logging
RAG_LOG_LEVEL=INFO

# Chunking / BM25
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=100
RAG_ALPHA=0.7
RAG_BM25_ENABLED=1
RAG_BM25_K1=1.2
RAG_BM25_B=0.75
RAG_BM25_MAX_DOC_TOKENS=1024
RAG_BM25_FETCH_K=24
RAG_BM25_MAX_QUERY_TOKENS=128
RAG_TOP_K=10
RAG_MAX_CONTEXT_CHARS=12000
RAG_EMBEDDING_BATCH_SIZE=16

# Qdrant
QDRANT_COLLECTION=kb_chunks
QDRANT_PATH=index/qdrant
QDRANT_URL=
QDRANT_API_KEY=
```
