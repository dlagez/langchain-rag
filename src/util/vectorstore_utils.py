from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from .keyword_utils import _keyword_score, _query_terms


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


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score == min_score:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


def _keyword_scores_for_docs(
    docs: list[Document],
    cjk_keywords: list[str],
    latin_keywords: list[str],
) -> np.ndarray:
    scores = np.zeros(len(docs), dtype=np.float32)
    for idx, doc in enumerate(docs):
        scores[idx] = _keyword_score(doc.page_content, cjk_keywords, latin_keywords)
    return scores


def _hybrid_rerank(
    query: str,
    docs: list[Document],
    vector_scores: np.ndarray,
    k: int,
    alpha: float,
) -> list[Document]:
    if vector_scores.size == 0 or not docs:
        return []
    cjk, latin = _query_terms(query)
    keyword_scores = _keyword_scores_for_docs(docs, cjk, latin)

    if keyword_scores.size > 0 and float(keyword_scores.max()) > 0:
        combined = _normalize_scores(vector_scores) * alpha + _normalize_scores(
            keyword_scores
        ) * (1 - alpha)
        order = np.argsort(combined)[::-1][:k]
    else:
        order = np.argsort(vector_scores)[::-1][:k]

    return [docs[idx] for idx in order]


@dataclass
class BM25Index:
    docs: list[Document]
    doc_lens: list[int]
    avgdl: float
    idf: dict[str, float]
    postings: dict[str, list[tuple[int, int]]]
    k1: float
    b: float
    cjk_min: int
    cjk_max: int
    max_doc_tokens: int


_BM25_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_BM25_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _bm25_tokenize(
    text: str,
    *,
    max_tokens: int | None = None,
    cjk_min: int = 2,
    cjk_max: int = 3,
) -> list[str]:
    if not text:
        return []
    tokens: list[str] = []
    for chunk in _BM25_CJK_RE.findall(text):
        limit = min(cjk_max, len(chunk))
        for size in range(cjk_min, limit + 1):
            for idx in range(len(chunk) - size + 1):
                tokens.append(chunk[idx : idx + size])
    for token in _BM25_WORD_RE.findall(text):
        if len(token) > 1:
            tokens.append(token.lower())
    if max_tokens and len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokens


def _bm25_index_path(persist_dir: Path) -> Path:
    return persist_dir / "bm25.pkl"


def _save_bm25_index(persist_dir: Path, index: BM25Index) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "docs": index.docs,
        "doc_lens": index.doc_lens,
        "avgdl": index.avgdl,
        "idf": index.idf,
        "postings": index.postings,
        "k1": index.k1,
        "b": index.b,
        "cjk_min": index.cjk_min,
        "cjk_max": index.cjk_max,
        "max_doc_tokens": index.max_doc_tokens,
    }
    with _bm25_index_path(persist_dir).open("wb") as handle:
        pickle.dump(payload, handle)


def _load_bm25_index(persist_dir: Path) -> BM25Index | None:
    path = _bm25_index_path(persist_dir)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if isinstance(payload, BM25Index):
        if not hasattr(payload, "max_doc_tokens"):
            payload.max_doc_tokens = 1024
        return payload
    if not isinstance(payload, dict):
        return None
    try:
        return BM25Index(
            docs=payload["docs"],
            doc_lens=payload["doc_lens"],
            avgdl=payload["avgdl"],
            idf=payload["idf"],
            postings=payload["postings"],
            k1=payload["k1"],
            b=payload["b"],
            cjk_min=payload.get("cjk_min", 2),
            cjk_max=payload.get("cjk_max", 3),
            max_doc_tokens=payload.get("max_doc_tokens", 1024),
        )
    except Exception:
        return None


def _build_bm25_index(
    docs: list[Document],
    *,
    k1: float = 1.2,
    b: float = 0.75,
    max_doc_tokens: int = 1024,
    cjk_min: int = 2,
    cjk_max: int = 3,
) -> BM25Index:
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    df: Counter[str] = Counter()
    doc_lens: list[int] = []
    for idx, doc in enumerate(docs):
        tokens = _bm25_tokenize(
            doc.page_content,
            max_tokens=max_doc_tokens,
            cjk_min=cjk_min,
            cjk_max=cjk_max,
        )
        if not tokens:
            doc_lens.append(0)
            continue
        tf = Counter(tokens)
        doc_lens.append(sum(tf.values()))
        for term, freq in tf.items():
            postings[term].append((idx, freq))
        for term in tf.keys():
            df[term] += 1
    total_docs = len(docs)
    idf = {
        term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }
    avgdl = float(np.mean(doc_lens)) if doc_lens else 0.0
    return BM25Index(
        docs=docs,
        doc_lens=doc_lens,
        avgdl=avgdl,
        idf=idf,
        postings=dict(postings),
        k1=k1,
        b=b,
        cjk_min=cjk_min,
        cjk_max=cjk_max,
        max_doc_tokens=max_doc_tokens,
    )


def _bm25_scores(
    index: BM25Index,
    query: str,
    *,
    max_query_tokens: int = 128,
) -> np.ndarray:
    tokens = _bm25_tokenize(
        query,
        max_tokens=max_query_tokens,
        cjk_min=index.cjk_min,
        cjk_max=index.cjk_max,
    )
    if not tokens or not index.docs:
        return np.zeros(len(index.docs), dtype=np.float32)
    scores = np.zeros(len(index.docs), dtype=np.float32)
    avgdl = index.avgdl or 1.0
    for term in set(tokens):
        idf = index.idf.get(term)
        if idf is None:
            continue
        for doc_idx, tf in index.postings.get(term, []):
            dl = index.doc_lens[doc_idx]
            denom = tf + index.k1 * (1 - index.b + index.b * (dl / avgdl))
            if denom <= 0:
                continue
            scores[doc_idx] += idf * (tf * (index.k1 + 1) / denom)
    return scores


def _bm25_search(
    index: BM25Index,
    query: str,
    limit: int,
    *,
    filter_fn: Callable[[Document], bool] | None = None,
    max_query_tokens: int = 128,
) -> tuple[list[Document], np.ndarray]:
    if not query or not index.docs or limit <= 0:
        return [], np.array([], dtype=np.float32)
    scores = _bm25_scores(index, query, max_query_tokens=max_query_tokens)
    if scores.size == 0:
        return [], np.array([], dtype=np.float32)
    if filter_fn is None:
        indices = range(len(index.docs))
    else:
        indices = [idx for idx, doc in enumerate(index.docs) if filter_fn(doc)]
    if not indices:
        return [], np.array([], dtype=np.float32)
    candidates = [idx for idx in indices if scores[idx] > 0]
    if not candidates:
        return [], np.array([], dtype=np.float32)
    candidates.sort(key=lambda idx: scores[idx], reverse=True)
    top = candidates[:limit]
    docs = [index.docs[idx] for idx in top]
    top_scores = np.array([scores[idx] for idx in top], dtype=np.float32)
    return docs, top_scores


def _fusion_rerank(
    docs: list[Document],
    dense_scores,
    sparse_scores,
    k: int,
    alpha: float,
) -> tuple[list[Document], np.ndarray]:
    if not docs:
        return [], np.array([], dtype=np.float32)
    if len(dense_scores) != len(docs) or len(sparse_scores) != len(docs):
        raise ValueError("fusion scores must align with docs")

    def _normalize_optional(values) -> np.ndarray:
        present = [val for val in values if val is not None]
        if not present:
            return np.zeros(len(values), dtype=np.float32)
        arr = np.array(present, dtype=np.float32)
        min_score = float(np.min(arr))
        max_score = float(np.max(arr))
        if max_score == min_score:
            return np.zeros(len(values), dtype=np.float32)
        out = np.zeros(len(values), dtype=np.float32)
        for idx, val in enumerate(values):
            if val is None:
                continue
            out[idx] = (float(val) - min_score) / (max_score - min_score)
        return out

    dense_norm = _normalize_optional(dense_scores)
    sparse_norm = _normalize_optional(sparse_scores)
    combined = dense_norm * float(alpha) + sparse_norm * (1 - float(alpha))
    order = np.argsort(combined)[::-1]
    if k > 0:
        order = order[:k]
    return [docs[idx] for idx in order], combined[order]


