from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from domain.contract.rag_contract_utils import (
    _assign_chunk_ids,
    _chunk_documents_for_index,
)
from util.util import (
    _build_index_manifest,
    _collection_exists,
    _get_qdrant_client,
    _load_manifest,
    _manifest_matches,
    _qdrant_location,
    _recreate_collection,
    _save_manifest,
    _upsert_documents,
)


# 是RAG 服务层的基础服务，负责初始化模型、配置日志、构建/加载向量索引等“运行支撑逻辑”。核心作用：
# 模型与提供商配置：解析环境变量，统一 google / bailian 的 provider 与模型名（resolve_embedding_config / resolve_llm_config）
# 构建 Embeddings / LLM：根据 provider 构造对应的 LangChain 实例（_build_*, build_llm）
# 日志与请求记录：配置日志级别、可选记录请求内容，兼容 DashScope/百炼 SDK 的日志过滤
# 向量库构建/复用：build_or_load_vectorstore 里做分块、嵌入、Qdrant collection 创建/复用、manifest 校验等

_PROVIDER_ALIASES = {
    "alibaba": "bailian",
    "aliyun": "bailian",
    "bailian": "bailian",
    "dashscope": "bailian",
    "tongyi": "bailian",
    "qwen": "bailian",
    "gemini": "google",
    "genai": "google",
    "google": "google",
    "google-genai": "google",
}

_DASHSCOPE_HTTP_PATCHED = False
_REQUEST_LOG_COUNTER = 0
_REQUEST_LOG_LOCK = threading.Lock()
_REQUEST_LOG_STATE = threading.local()


def _normalize_provider(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def _resolve_provider(
    env_key: str, fallback_key: str | None, default: str
) -> str:
    value = os.getenv(env_key)
    if not value and fallback_key:
        value = os.getenv(fallback_key)
    normalized = _normalize_provider(value)
    return normalized or default


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _request_log_dir() -> Path:
    raw = os.getenv("RAG_LOG_DIR", "data/log")
    path = Path(raw)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _next_request_log_path() -> Path:
    global _REQUEST_LOG_COUNTER
    with _REQUEST_LOG_LOCK:
        _REQUEST_LOG_COUNTER += 1
        counter = _REQUEST_LOG_COUNTER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return _request_log_dir() / f"request_{timestamp}_{counter}.txt"


def _append_request_log(label: str, payload: str) -> None:
    if not _env_flag("RAG_LOG_REQUESTS"):
        return
    path = getattr(_REQUEST_LOG_STATE, "path", None)
    if label == "Request url":
        path = _next_request_log_path()
        _REQUEST_LOG_STATE.path = path
    if path is None:
        path = _next_request_log_path()
        _REQUEST_LOG_STATE.path = path
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{label}: {payload}\n")


class _DashscopeLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for prefix in ("Request url: ", "Request body: ", "Response: "):
            if message.startswith(prefix):
                body = message[len(prefix) :].strip()
                _append_request_log(prefix.strip(": "), body)
                return False
        return True


def _using_bailian() -> bool:
    embedding_provider = _resolve_provider(
        "RAG_EMBEDDING_PROVIDER", "RAG_PROVIDER", ""
    )
    llm_provider = _resolve_provider("RAG_LLM_PROVIDER", "RAG_PROVIDER", "")
    return embedding_provider == "bailian" or llm_provider == "bailian"


def _patch_bailian_http_logging() -> None:
    global _DASHSCOPE_HTTP_PATCHED
    if _DASHSCOPE_HTTP_PATCHED:
        return
    try:
        from dashscope.api_entities import http_request as _dashscope_http
    except Exception:
        return

    original = _dashscope_http.HttpRequest._handle_request

    def _handle_request_with_logging(self):
        logging.getLogger("dashscope").debug("Request url: %s", self.url)
        return original(self)

    _dashscope_http.HttpRequest._handle_request = _handle_request_with_logging
    _DASHSCOPE_HTTP_PATCHED = True


def _configure_request_logging() -> None:
    log_requests = _env_flag("RAG_LOG_REQUESTS")
    if _using_bailian():
        if log_requests:
            _patch_bailian_http_logging()
        logger = logging.getLogger("dashscope")
        logger.addFilter(_DashscopeLogFilter())
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        logger.setLevel(logging.DEBUG if log_requests else logging.INFO)
        logger.propagate = True


def configure_logging() -> None:
    level_name = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
    level = logging._nameToLevel.get(level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    _configure_request_logging()


def _bailian_api_key() -> str:
    api_key = os.getenv("BAILIAN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Bailian API key missing. Set BAILIAN_API_KEY or DASHSCOPE_API_KEY."
        )
    return api_key


def _build_google_embeddings(model: str) -> Embeddings:
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except Exception as exc:
        raise SystemExit(
            "Google embeddings require langchain-google-genai."
        ) from exc
    return GoogleGenerativeAIEmbeddings(model=model)


def _build_google_llm(model: str, temperature: float = 0) -> Any:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as exc:
        raise SystemExit(
            "Google LLM requires langchain-google-genai."
        ) from exc
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def _build_bailian_embeddings(model: str) -> Embeddings:
    api_key = _bailian_api_key()
    try:
        from langchain_community.embeddings import DashScopeEmbeddings
    except Exception as exc:
        raise SystemExit(
            "Bailian embeddings require langchain-community and dashscope."
        ) from exc
    return DashScopeEmbeddings(model=model, dashscope_api_key=api_key)


def _build_bailian_llm(model: str) -> Any:
    api_key = _bailian_api_key()
    try:
        from langchain_community.chat_models import ChatTongyi
    except Exception as exc:
        raise SystemExit(
            "Bailian LLM requires langchain-community and dashscope."
        ) from exc
    return ChatTongyi(model=model, api_key=api_key)


def resolve_embedding_config() -> tuple[str, str]:
    provider = _resolve_provider(
        "RAG_EMBEDDING_PROVIDER", "RAG_PROVIDER", "google"
    )
    if provider == "google":
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
        return provider, model
    if provider == "bailian":
        model = (
            os.getenv("BAILIAN_EMBEDDING_MODEL")
            or os.getenv("DASHSCOPE_EMBEDDING_MODEL")
            or "text-embedding-v2"
        )
        return provider, model
    raise SystemExit(f"Unsupported embedding provider: {provider}")


def resolve_llm_config() -> tuple[str, str]:
    provider = _resolve_provider("RAG_LLM_PROVIDER", "RAG_PROVIDER", "google")
    if provider == "google":
        model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
        return provider, model
    if provider == "bailian":
        model = (
            os.getenv("BAILIAN_MODEL")
            or os.getenv("DASHSCOPE_MODEL")
            or "qwen-plus"
        )
        return provider, model
    raise SystemExit(f"Unsupported LLM provider: {provider}")


def _build_embeddings(provider: str, model: str) -> Embeddings:
    if provider == "google":
        return _build_google_embeddings(model)
    if provider == "bailian":
        return _build_bailian_embeddings(model)
    raise SystemExit(f"Unsupported embedding provider: {provider}")


def build_llm(provider: str, model: str) -> Any:
    if provider == "google":
        return _build_google_llm(model, temperature=0)
    if provider == "bailian":
        return _build_bailian_llm(model)
    raise SystemExit(f"Unsupported LLM provider: {provider}")


def build_or_load_vectorstore(
    docs: list[Document],
    persist_dir: Path,
    processed_dir: Path,
    force_rebuild: bool = False,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
):
    if not embedding_provider or not embedding_model:
        embedding_provider, embedding_model = resolve_embedding_config()
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    collection_name = os.getenv("QDRANT_COLLECTION", "contract_approval_rag")
    qdrant_location = _qdrant_location(persist_dir)

    embeddings = _build_embeddings(embedding_provider, embedding_model)
    manifest = _build_index_manifest(
        processed_dir, embedding_model, chunk_size, chunk_overlap
    )
    manifest["embedding_provider"] = embedding_provider
    manifest["collection_name"] = collection_name
    manifest["qdrant_location"] = qdrant_location
    manifest["ingestion_schema"] = "contract_approval_v4_attachment_only"
    manifest["chunking_strategy"] = "structured_v1"
    manifest["chunking_params"] = {
        "contract_min": max(200, int(chunk_size * 0.5)),
        "contract_max": chunk_size,
        "overlap": chunk_overlap,
        "checklist_min": min(200, min(600, chunk_size)),
        "checklist_max": min(600, chunk_size),
    }

    client = _get_qdrant_client(persist_dir)

    if not force_rebuild:
        stored_manifest = _load_manifest(persist_dir)
        if _manifest_matches(stored_manifest, manifest) and _collection_exists(
            client, collection_name
        ):
            return client, collection_name, embeddings

    chunked_docs = _chunk_documents_for_index(
        docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = _assign_chunk_ids(chunked_docs)
    if not splits:
        raise SystemExit("No content left after splitting documents.")

    vectors = embeddings.embed_documents([doc.page_content for doc in splits])
    if not vectors:
        raise SystemExit("Embedding model returned no vectors.")
    vector_size = len(vectors[0])
    _recreate_collection(client, collection_name, vector_size)
    _upsert_documents(client, collection_name, splits, vectors)
    _save_manifest(persist_dir, manifest, doc_count=len(splits))
    return client, collection_name, embeddings
