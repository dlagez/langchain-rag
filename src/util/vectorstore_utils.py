from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams



def _fingerprint_processed(processed_dir: Path) -> str:
    if not processed_dir.exists():
        return ""
    items = []
    for path in sorted(processed_dir.rglob("*.txt")):
        try:
            stat = path.stat()
        except OSError:
            continue
        rel = path.relative_to(processed_dir).as_posix()
        items.append(f"{rel}:{stat.st_mtime_ns}:{stat.st_size}")
    if not items:
        return ""
    payload = "\n".join(items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _build_index_manifest(
    processed_dir: Path,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    return {
        "fingerprint": _fingerprint_processed(processed_dir),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


def _manifest_matches(stored: dict, expected: dict) -> bool:
    if not stored:
        return False
    for key, value in expected.items():
        if stored.get(key) != value:
            return False
    return True


def _save_manifest(persist_dir: Path, manifest: dict, doc_count: int) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = dict(manifest)
    manifest_payload["doc_count"] = doc_count
    (persist_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_manifest(persist_dir: Path) -> dict:
    manifest_path = persist_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _qdrant_location(persist_dir: Path) -> str:
    url = os.getenv("QDRANT_URL")
    if url:
        return url
    return os.getenv("QDRANT_PATH", str(persist_dir / "qdrant"))


def _get_qdrant_client(persist_dir: Path) -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if url:
        return QdrantClient(url=url, api_key=api_key)
    path = _qdrant_location(persist_dir)
    return QdrantClient(path=path)


def _collection_exists(client: QdrantClient, name: str) -> bool:
    try:
        client.get_collection(collection_name=name)
    except Exception:
        return False
    return True


def _recreate_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    try:
        client.delete_collection(collection_name=name)
    except Exception:
        pass
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _upsert_documents(
    client: QdrantClient,
    collection_name: str,
    docs: list[Document],
    vectors: list[list[float]],
    batch_size: int = 64,
) -> None:
    total = len(docs)
    for start in range(0, total, batch_size):
        points: list[PointStruct] = []
        for idx in range(start, min(start + batch_size, total)):
            payload = {
                "page_content": docs[idx].page_content,
                "metadata": docs[idx].metadata,
            }
            points.append(PointStruct(id=idx, vector=vectors[idx], payload=payload))
        if points:
            client.upsert(collection_name=collection_name, points=points, wait=True)


def _search_qdrant(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    limit: int,
    query_filter=None,
):
    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return getattr(response, "points", response)
    raise RuntimeError("Unsupported Qdrant client: missing search/query_points.")


def _docs_from_search_results(results) -> tuple[list[Document], np.ndarray]:
    docs: list[Document] = []
    scores: list[float] = []
    for point in results:
        payload = point.payload or {}
        content = payload.get("page_content") or ""
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        docs.append(Document(page_content=content, metadata=metadata))
        scores.append(float(point.score or 0.0))
    return docs, np.array(scores, dtype=np.float32)


