from __future__ import annotations

from typing import Any


def _is_vector(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    if not value:
        return False
    return all(isinstance(item, (int, float)) for item in value)


def _extract_embeddings(payload: Any) -> list[list[float]]:
    if payload is None:
        return []

    if isinstance(payload, list):
        if payload and _is_vector(payload[0]):
            return [list(vec) for vec in payload]
        if payload and isinstance(payload[0], dict):
            vectors = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                value = item.get("embedding") or item.get("vector") or item.get("values")
                if _is_vector(value):
                    vectors.append(list(value))
            if vectors:
                return vectors

    if isinstance(payload, dict):
        for key in ("embeddings", "embedding", "data", "result", "output"):
            value = payload.get(key)
            vectors = _extract_embeddings(value)
            if vectors:
                return vectors

    if hasattr(payload, "embeddings"):
        vectors = _extract_embeddings(getattr(payload, "embeddings"))
        if vectors:
            return vectors

    if hasattr(payload, "output"):
        vectors = _extract_embeddings(getattr(payload, "output"))
        if vectors:
            return vectors

    return []


def _extract_status(payload: Any) -> tuple[int | None, str | None]:
    status = None
    message = None
    if hasattr(payload, "status_code"):
        try:
            status = int(getattr(payload, "status_code"))
        except Exception:
            status = None
    if hasattr(payload, "message"):
        message = str(getattr(payload, "message"))
    if isinstance(payload, dict):
        if status is None and "status_code" in payload:
            try:
                status = int(payload.get("status_code"))
            except Exception:
                status = None
        if message is None and "message" in payload:
            message = str(payload.get("message"))
    return status, message


def embed_texts(
    texts: list[str],
    *,
    model: str,
    api_key: str,
    batch_size: int = 16,
) -> list[list[float]]:
    if not texts:
        return []
    try:
        import dashscope
    except Exception as exc:
        raise RuntimeError(f"dashscope not available: {exc}") from exc

    dashscope.api_key = api_key

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = dashscope.Embeddings.call(model=model, input=batch)
        status, message = _extract_status(response)
        if status is not None and status >= 400:
            raise RuntimeError(f"Embedding call failed: {status} {message or ''}")
        payload = response
        if hasattr(response, "output"):
            payload = response.output
        embeddings = _extract_embeddings(payload)
        if not embeddings:
            embeddings = _extract_embeddings(response)
        if not embeddings:
            raise RuntimeError("Embedding response missing vectors.")
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch: got {len(embeddings)} expected {len(batch)}"
            )
        vectors.extend(embeddings)

    return vectors
