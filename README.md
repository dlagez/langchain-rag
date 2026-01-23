# Personal KB P0 (Bailian + Remote OCR)

This project implements a P0 personal knowledge base with:
- Ingestion from absolute paths
- Parsing + OCR
- Chunking + Vector + BM25 retrieval
- Answers with citations
- Local embedding service (preferred) with Bailian fallback

## Quickstart (PowerShell)
1) Activate venv: `\.\.venv\Scripts\Activate.ps1`
2) Install deps: `python -m pip install -r requirements.txt`
3) Create `.env` (see `.env.example`)
4) Ingest a folder: `python -m app.cli ingest --root "D:\docs\projectA" --kb kb_project_a`
5) Ask a question: `python -m app.cli query --kb kb_project_a "????"`

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

## Storage layout
- `data/manifest/<kb_id>/` : file status + ingest report
- `data/processed/<kb_id>/` : parsed text
- `index/<kb_id>/` : Qdrant + BM25 index
- `data/log/<kb_id>/` : ingest logs

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

# Chunking / BM25 / Qdrant
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=100
RAG_BM25_ENABLED=1
QDRANT_COLLECTION=kb_chunks
```
