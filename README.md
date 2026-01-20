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
- `RAG_ATTACHMENT_KEYWORD_QUERY`: keyword-only query for BM25; if empty, falls back to retrieval query/question.
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
- `RAG_BM25_ENABLED`: enable BM25 retrieval for hybrid fusion (default: `1`).
- `RAG_BM25_K1`: BM25 k1 parameter (default: `1.2`).
- `RAG_BM25_B`: BM25 b parameter (default: `0.75`).
- `RAG_BM25_MAX_DOC_TOKENS`: cap tokens per doc for BM25 indexing (default: `1024`).
- `RAG_BM25_FETCH_K`: BM25 candidate pool size (default: `fetch_k`).
- `RAG_BM25_MAX_QUERY_TOKENS`: cap tokens for BM25 query (default: `128`).
- `RAG_TOC_FILTER_ENABLED`: enable table-of-contents filtering in retrieval (default: `1`).
- `RAG_TOC_MIN_LINES`: minimum line count to consider a chunk as TOC-like (default: `5`).
- `RAG_TOC_LINE_RATIO`: ratio of TOC-like lines to flag a chunk (default: `0.45`).
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
RAG_BM25_ENABLED=1
RAG_BM25_K1=1.2
RAG_BM25_B=0.75
RAG_BM25_MAX_DOC_TOKENS=1024
RAG_BM25_FETCH_K=24
RAG_BM25_MAX_QUERY_TOKENS=128
RAG_TOC_FILTER_ENABLED=1
RAG_TOC_MIN_LINES=5
RAG_TOC_LINE_RATIO=0.45
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


2026-01-21：本地 BM25 + Qdrant dense
RAG_BM25_ENABLED: 是否启用 BM25 召回；1 开启、0 关闭。关闭后只走向量检索。
RAG_BM25_K1: BM25 的 k1 参数，控制词频对分数的影响力度；越大越强调高词频。
RAG_BM25_B: BM25 的 b 参数，控制文档长度归一化；越大越惩罚长文档。
RAG_BM25_MAX_DOC_TOKENS: BM25 建索引时每个文档的最大分词数量上限；防止超长文本拖慢索引与检索。
RAG_BM25_FETCH_K: BM25 候选池大小；越大召回越多，但融合/重排成本也更高。
RAG_BM25_MAX_QUERY_TOKENS: BM25 查询的最大分词数量上限；防止超长 query 影响速度与噪声。

BM25 使用逻辑（当前实现）

建索引：在 build_or_load_vectorstore 里用已切分的 splits 构建本地 BM25 索引（与向量索引的 chunk 对齐），持久化到 bm25.pkl。
召回：在 retrieve_documents 里用 BM25 对 keyword_query 或 question 进行检索，取 RAG_BM25_FETCH_K 个候选。
过滤：BM25 召回后按 process_id + source_type=attachment 过滤，再叠加 scope 过滤（若启用）。
融合：与向量召回结果合并去重，按 alpha 做归一化线性融合，统一 rerank，最终取 k。
追踪来源：融合后文档会标记 bm25 / vector 召回来源，便于日志排查。
注意事项 / 限制

分词策略：中文用 2–3 字 n‑gram，英文用字母数字 token（>1）。不做停用词过滤，短 query 可能噪声偏大。
上限截断：
RAG_BM25_MAX_DOC_TOKENS 会截断每个 chunk 的索引 token 数，过低会损失召回。
RAG_BM25_MAX_QUERY_TOKENS 截断过长 query，避免噪声/慢检索。
索引一致性：BM25 索引与向量索引必须和当前 chunk 一致；当文档或切分参数变更时需 --rebuild 或删除 index/。
流程依赖：process_id 缺失会阻止检索；BM25 只对 attachment 生效，不用于表单直取。
资源占用：BM25 索引常驻内存（docs + postings），大文档集会增加内存和加载时间。
