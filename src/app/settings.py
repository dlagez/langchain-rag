from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    data_dir: Path
    manifest_dir: Path
    processed_dir: Path
    log_dir: Path
    index_dir: Path

    bailian_api_key: str | None
    bailian_model: str
    bailian_embedding_model: str
    llm_provider: str
    embedding_provider: str
    local_embedding_base_url: str | None
    local_embedding_model: str | None

    ocr_url: str
    ocr_timeout: float
    ocr_file_field: str

    chunk_size: int
    chunk_overlap: int
    alpha: float
    bm25_enabled: bool
    bm25_k1: float
    bm25_b: float
    bm25_max_doc_tokens: int
    bm25_fetch_k: int
    bm25_max_query_tokens: int

    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_collection: str

    top_k: int
    max_context_chars: int
    embedding_batch_size: int

    @classmethod
    def from_env(cls, root_dir: Path) -> "Settings":
        root_dir = root_dir.resolve()
        data_dir = root_dir / "data"
        llm_provider = (
            os.getenv("LLM_PROVIDER")
            or os.getenv("RAG_LLM_PROVIDER")
            or os.getenv("RAG_PROVIDER")
            or "bailian"
        ).strip().lower()
        embedding_provider = (
            os.getenv("EMBEDDING_PROVIDER")
            or os.getenv("RAG_EMBEDDING_PROVIDER")
            or os.getenv("RAG_PROVIDER")
            or ""
        ).strip().lower()
        local_embedding_base_url = os.getenv("LOCAL_EMBEDDING_BASE_URL")
        local_embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL")
        if not embedding_provider:
            if local_embedding_base_url:
                embedding_provider = "local"
            else:
                embedding_provider = "bailian"
        return cls(
            root_dir=root_dir,
            data_dir=data_dir,
            manifest_dir=data_dir / "manifest",
            processed_dir=data_dir / "processed",
            log_dir=data_dir / "log",
            index_dir=root_dir / "index",
            bailian_api_key=os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
            bailian_model=os.getenv("BAILIAN_MODEL") or os.getenv("DASHSCOPE_MODEL") or "qwen-plus",
            bailian_embedding_model=(
                os.getenv("BAILIAN_EMBEDDING_MODEL")
                or os.getenv("DASHSCOPE_EMBEDDING_MODEL")
                or "text-embedding-v2"
            ),
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            local_embedding_base_url=local_embedding_base_url,
            local_embedding_model=local_embedding_model,
            ocr_url=os.getenv("OCR_URL", "http://10.0.22.109:8081/ocr"),
            ocr_timeout=_env_float("OCR_TIMEOUT", 30.0),
            ocr_file_field=os.getenv("OCR_FILE_FIELD", "file"),
            chunk_size=_env_int("RAG_CHUNK_SIZE", 800),
            chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 100),
            alpha=_env_float("RAG_ALPHA", 0.7),
            bm25_enabled=_env_bool("RAG_BM25_ENABLED", True),
            bm25_k1=_env_float("RAG_BM25_K1", 1.2),
            bm25_b=_env_float("RAG_BM25_B", 0.75),
            bm25_max_doc_tokens=_env_int("RAG_BM25_MAX_DOC_TOKENS", 1024),
            bm25_fetch_k=_env_int("RAG_BM25_FETCH_K", 24),
            bm25_max_query_tokens=_env_int("RAG_BM25_MAX_QUERY_TOKENS", 128),
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "kb_chunks"),
            top_k=_env_int("RAG_TOP_K", 6),
            max_context_chars=_env_int("RAG_MAX_CONTEXT_CHARS", 12000),
            embedding_batch_size=_env_int("RAG_EMBEDDING_BATCH_SIZE", 16),
        )
