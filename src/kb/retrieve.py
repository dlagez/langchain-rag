from __future__ import annotations

from collections import OrderedDict

import numpy as np
from langchain_core.documents import Document
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from app.settings import Settings
from kb.index import bm25_search, get_client_and_collection
from providers.embed_bailian import embed_texts as embed_bailian
from providers.embed_local import embed_texts as embed_local
from util.prompt_logger import log_embedding_call
from util.document_utils import _doc_signature
from util.vectorstore_utils import _docs_from_search_results, _fusion_rerank, _search_qdrant


def _kb_filter(kb_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="kb_id", match=MatchValue(value=kb_id))])


def _merge_results(
    vector_docs: list[Document],
    vector_scores: np.ndarray,
    bm25_docs: list[Document],
    bm25_scores: np.ndarray,
    *,
    alpha: float,
    k: int,
) -> list[Document]:
    if not bm25_docs:
        return vector_docs[:k]

    combined: OrderedDict[str, dict] = OrderedDict()

    for doc, score in zip(vector_docs, vector_scores):
        sig = _doc_signature(doc)
        combined[sig] = {"doc": doc, "dense": float(score), "sparse": None}

    for doc, score in zip(bm25_docs, bm25_scores):
        sig = _doc_signature(doc)
        entry = combined.get(sig)
        if entry is None:
            combined[sig] = {"doc": doc, "dense": None, "sparse": float(score)}
        else:
            entry["sparse"] = float(score)

    docs = [item["doc"] for item in combined.values()]
    dense_scores = [item["dense"] for item in combined.values()]
    sparse_scores = [item["sparse"] for item in combined.values()]

    reranked, _ = _fusion_rerank(docs, dense_scores, sparse_scores, k, alpha)
    return reranked


def retrieve_documents(
    *,
    kb_id: str,
    question: str,
    settings: Settings,
    top_k: int | None = None,
) -> list[Document]:
    query = question.strip()
    if not query:
        return []

    client, collection = get_client_and_collection(settings, kb_id)
    query_vec = _embed_query(query, settings)
    k = top_k or settings.top_k
    fetch_k = max(k, settings.bm25_fetch_k)

    results = _search_qdrant(
        client,
        collection,
        query_vector=query_vec,
        limit=fetch_k,
        query_filter=_kb_filter(kb_id),
    )
    vector_docs, vector_scores = _docs_from_search_results(results)

    if not settings.bm25_enabled:
        return vector_docs[:k]

    bm25_docs, bm25_scores = bm25_search(
        kb_id,
        query,
        fetch_k,
        settings=settings,
    )
    return _merge_results(vector_docs, vector_scores, bm25_docs, bm25_scores, alpha=settings.alpha, k=k)


def _embed_query(query: str, settings: Settings) -> list[float]:
    provider = settings.embedding_provider
    if provider == "local":
        if not settings.local_embedding_base_url:
            raise SystemExit("LOCAL_EMBEDDING_BASE_URL is required for local embedding.")
        if not settings.local_embedding_model:
            raise SystemExit("LOCAL_EMBEDDING_MODEL is required for local embedding.")
        try:
            vectors = embed_local(
                [query],
                model=settings.local_embedding_model,
                base_url=settings.local_embedding_base_url,
            )
        except Exception as exc:
            log_embedding_call(
                provider="local",
                model=settings.local_embedding_model,
                count=1,
                sources=[],
                purpose="query",
                error=str(exc),
            )
            raise
        log_embedding_call(
            provider="local",
            model=settings.local_embedding_model,
            count=1,
            sources=[],
            purpose="query",
        )
        return vectors[0]
    if provider in {"bailian", "dashscope"}:
        if not settings.bailian_api_key:
            raise SystemExit("BAILIAN_API_KEY is required for retrieval.")
        try:
            vectors = embed_bailian(
                [query],
                model=settings.bailian_embedding_model,
                api_key=settings.bailian_api_key,
                batch_size=1,
            )
        except Exception as exc:
            log_embedding_call(
                provider="bailian",
                model=settings.bailian_embedding_model,
                count=1,
                sources=[],
                purpose="query",
                error=str(exc),
            )
            raise
        log_embedding_call(
            provider="bailian",
            model=settings.bailian_embedding_model,
            count=1,
            sources=[],
            purpose="query",
        )
        return vectors[0]
    raise SystemExit(f"Unsupported EMBEDDING_PROVIDER: {provider}")
