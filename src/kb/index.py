from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from app.settings import Settings
from kb.io.index_store import bm25_path, collection_name, qdrant_path
from providers.embed_bailian import embed_texts as embed_bailian
from providers.embed_local import embed_texts as embed_local
from util.prompt_logger import log_embedding_call
from util.vectorstore_utils import _bm25_search, _build_bm25_index, _save_bm25_index


def get_client_and_collection(settings: Settings, kb_id: str) -> tuple[QdrantClient, str]:
    use_remote = bool(settings.qdrant_url)
    if use_remote:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    else:
        client = QdrantClient(path=str(qdrant_path(settings.index_dir, kb_id)))
    collection = collection_name(settings.qdrant_collection, kb_id, use_remote=use_remote)
    return client, collection


def collection_exists(client: QdrantClient, name: str) -> bool:
    try:
        client.get_collection(collection_name=name)
    except Exception:
        return False
    return True


def ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    if collection_exists(client, name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _kb_doc_filter(kb_id: str, doc_id: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="kb_id", match=MatchValue(value=kb_id)),
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
        ]
    )


def delete_doc_chunks(
    client: QdrantClient, collection: str, *, kb_id: str, doc_id: str
) -> None:
    if not collection_exists(client, collection):
        return
    try:
        client.delete(collection_name=collection, points_selector=_kb_doc_filter(kb_id, doc_id))
    except Exception:
        return


def upsert_documents(
    client: QdrantClient,
    collection: str,
    docs: list[Document],
    vectors: list[list[float]],
) -> None:
    points: list[PointStruct] = []
    for doc, vector in zip(docs, vectors):
        metadata = doc.metadata or {}
        payload = {
            "page_content": doc.page_content,
            "metadata": metadata,
            "kb_id": metadata.get("kb_id"),
            "doc_id": metadata.get("doc_id"),
            "source": metadata.get("source"),
            "source_type": metadata.get("source_type"),
        }
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
    if points:
        client.upsert(collection_name=collection, points=points, wait=True)


def index_chunks(
    *,
    client: QdrantClient,
    collection: str,
    docs: list[Document],
    settings: Settings,
) -> int:
    if not docs:
        return 0
    texts = [doc.page_content for doc in docs]
    source_names = _source_names(docs)
    vectors = _embed_texts(texts, settings, source_names=source_names, purpose="index")
    vector_size = len(vectors[0]) if vectors else 0
    ensure_collection(client, collection, vector_size)
    upsert_documents(client, collection, docs, vectors)
    return len(docs)


def _embed_texts(
    texts: list[str],
    settings: Settings,
    *,
    source_names: list[str] | None,
    purpose: str,
) -> list[list[float]]:
    provider = settings.embedding_provider
    if provider == "local":
        if not settings.local_embedding_base_url:
            raise SystemExit("LOCAL_EMBEDDING_BASE_URL is required for local embedding.")
        if not settings.local_embedding_model:
            raise SystemExit("LOCAL_EMBEDDING_MODEL is required for local embedding.")
        try:
            vectors = embed_local(
                texts,
                model=settings.local_embedding_model,
                base_url=settings.local_embedding_base_url,
            )
        except Exception as exc:
            log_embedding_call(
                provider="local",
                model=settings.local_embedding_model,
                count=len(texts),
                sources=source_names,
                purpose=purpose,
                error=str(exc),
            )
            raise
        log_embedding_call(
            provider="local",
            model=settings.local_embedding_model,
            count=len(texts),
            sources=source_names,
            purpose=purpose,
        )
        return vectors
    if provider in {"bailian", "dashscope"}:
        if not settings.bailian_api_key:
            raise SystemExit("BAILIAN_API_KEY is required for embedding.")
        try:
            vectors = embed_bailian(
                texts,
                model=settings.bailian_embedding_model,
                api_key=settings.bailian_api_key,
                batch_size=settings.embedding_batch_size,
            )
        except Exception as exc:
            log_embedding_call(
                provider="bailian",
                model=settings.bailian_embedding_model,
                count=len(texts),
                sources=source_names,
                purpose=purpose,
                error=str(exc),
            )
            raise
        log_embedding_call(
            provider="bailian",
            model=settings.bailian_embedding_model,
            count=len(texts),
            sources=source_names,
            purpose=purpose,
        )
        return vectors
    raise SystemExit(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def _source_names(docs: list[Document]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for doc in docs:
        source = (doc.metadata or {}).get("source")
        if not source:
            continue
        name = Path(str(source)).name
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _scroll_points(client: QdrantClient, collection: str) -> Iterable:
    next_offset = None
    while True:
        response = client.scroll(
            collection_name=collection,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", None)
        if points is None:
            points, next_offset = response
        else:
            next_offset = getattr(response, "next_page_offset", None)
        if not points:
            break
        for point in points:
            yield point
        if next_offset is None:
            break


def load_all_docs(client: QdrantClient, collection: str) -> list[Document]:
    docs: list[Document] = []
    if not collection_exists(client, collection):
        return docs
    for point in _scroll_points(client, collection):
        payload = point.payload or {}
        content = payload.get("page_content") or ""
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def rebuild_bm25(client: QdrantClient, collection: str, settings: Settings, kb_id: str) -> int:
    docs = load_all_docs(client, collection)
    if not docs:
        return 0
    index = _build_bm25_index(
        docs,
        k1=settings.bm25_k1,
        b=settings.bm25_b,
        max_doc_tokens=settings.bm25_max_doc_tokens,
    )
    path = bm25_path(settings.index_dir, kb_id)
    _save_bm25_index(path.parent, index)
    return len(docs)


def bm25_search(
    kb_id: str,
    query: str,
    limit: int,
    *,
    settings: Settings,
) -> tuple[list[Document], np.ndarray]:
    path = bm25_path(settings.index_dir, kb_id)
    index = None
    try:
        from util.vectorstore_utils import _load_bm25_index

        index = _load_bm25_index(path.parent)
    except Exception:
        index = None
    if index is None:
        return [], np.array([], dtype=np.float32)

    def _filter(doc: Document) -> bool:
        return doc.metadata.get("kb_id") == kb_id

    return _bm25_search(
        index,
        query,
        limit,
        filter_fn=_filter,
        max_query_tokens=settings.bm25_max_query_tokens,
    )


def count_points(client: QdrantClient, collection: str) -> int:
    try:
        result = client.count(collection_name=collection, exact=True)
    except Exception:
        return 0
    return int(getattr(result, "count", 0) or 0)
