from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _prompt_dir() -> Path:
    base = _root_dir() / "data" / "prompt"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _safe_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "length": len(value),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(v) for v in value]
    return str(value)


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


def should_log_embedding() -> bool:
    return _env_bool("PROMPT_LOG_EMBEDDING", True)


def should_log_retrieval() -> bool:
    return _env_bool("PROMPT_LOG_RETRIEVAL", True)


def log_event(
    kind: str,
    *,
    request: Any | None = None,
    response: Any | None = None,
    error: str | None = None,
) -> Path:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "request": _safe_value(request),
        "response": _safe_value(response),
        "error": error,
    }
    filename = f"{_utc_timestamp()}_{kind}_{uuid.uuid4().hex[:8]}.json"
    path = _prompt_dir() / filename
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def log_embedding_call(
    *,
    provider: str,
    model: str,
    count: int,
    sources: list[str] | None,
    purpose: str,
    error: str | None = None,
) -> Path | None:
    if not should_log_embedding():
        return None
    request_payload = {
        "provider": provider,
        "model": model,
        "count": count,
        "sources": sources or [],
        "purpose": purpose,
    }
    response_payload = None if error else {"status": "ok"}
    return log_event(
        f"embedding_{purpose}",
        request=request_payload,
        response=response_payload,
        error=error,
    )


def log_retrieval_event(
    *,
    kb_id: str,
    query: str,
    top_k: int,
    fetch_k: int,
    vector_hits: list[dict],
    bm25_hits: list[dict],
    merged_hits: list[dict],
    filter_info: dict | None = None,
) -> Path | None:
    if not should_log_retrieval():
        return None
    request_payload = {
        "kb_id": kb_id,
        "query": query,
        "top_k": top_k,
        "fetch_k": fetch_k,
        "filter": filter_info or {},
    }
    response_payload = {
        "vector_hits": vector_hits,
        "bm25_hits": bm25_hits,
        "merged_hits": merged_hits,
    }
    return log_event("retrieval", request=request_payload, response=response_payload)
