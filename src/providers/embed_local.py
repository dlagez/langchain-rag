from __future__ import annotations

import json
from typing import Any
from urllib import error, request


def _ensure_base_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise RuntimeError("LOCAL_EMBEDDING_BASE_URL is empty")
    if not base.endswith("/v1") and not base.endswith("/v1/"):
        base = f"{base}/v1"
    return base.rstrip("/")


def _extract_embeddings(payload: Any) -> list[list[float]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    vectors: list[list[float]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        emb = item.get("embedding")
        if isinstance(emb, list) and emb and all(isinstance(v, (int, float)) for v in emb):
            vectors.append([float(v) for v in emb])
    return vectors


def embed_texts(
    texts: list[str],
    *,
    model: str,
    base_url: str,
) -> list[list[float]]:
    if not texts:
        return []
    if not model:
        raise RuntimeError("LOCAL_EMBEDDING_MODEL is required")

    base = _ensure_base_url(base_url)
    url = f"{base}/embeddings"

    payload = {"model": model, "input": texts}
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Local embedding HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"Local embedding request failed: {exc}") from exc

    try:
        response = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Local embedding response is not valid JSON") from exc

    vectors = _extract_embeddings(response)
    if not vectors:
        raise RuntimeError("Local embedding response missing vectors")
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Local embedding count mismatch: got {len(vectors)} expected {len(texts)}"
        )
    return vectors
