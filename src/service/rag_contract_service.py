from __future__ import annotations

import logging
import os
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import numpy as np

from domain.contract.rag_contract_utils import (
    SourceScope,
    _collect_source_names,
    _filter_docs_and_scores_by_scope,
    _filter_docs_by_scope,
    _format_source_scope,
    _tag_retriever,
    _unique_sources_with_retriever,
)
from util.document_utils import _format_source
from util.util import (
    _bm25_search,
    _doc_signature,
    _docs_from_search_results,
    _format_context,
    _fusion_rerank,
    _response_text,
    _search_qdrant,
)
from util.rag_utils import _extract_token_usage


def save_prompt(label: str, prompt: str, answer: str, prompt_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prompt_path = prompt_dir / f"prompt_{timestamp}_{label}.txt"
    payload = f"{prompt}\n\n---\nAnswer:\n{answer}"
    prompt_path.write_text(payload, encoding="utf-8")
    return prompt_path


def record_llm_stats(
    llm_stats: dict[str, dict[str, int | float | None]],
    label: str,
    response: Any,
    elapsed_seconds: float,
) -> None:
    usage = _extract_token_usage(response)
    llm_stats[label] = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "seconds": round(elapsed_seconds, 3),
    }


def _is_toc_like(
    text: str,
    *,
    min_lines: int,
    ratio_threshold: float,
) -> bool:
    if not text:
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < min_lines:
        return False
    toc_hits = 0
    for line in lines:
        if re.search(r"[\.·]{4,}", line) or "……" in line:
            toc_hits += 1
            continue
        if re.search(r"\b\d{1,4}$", line) and re.search(r"(第\s*\d+.*条)|(\d+\.\d+)", line):
            toc_hits += 1
            continue
        if re.search(r"^第\s*\d+", line) and re.search(r"\b\d{1,4}$", line):
            toc_hits += 1
            continue
    if not lines:
        return False
    return (toc_hits / len(lines)) >= ratio_threshold


def _apply_toc_filter(
    docs: list[Document],
    scores: np.ndarray,
    *,
    label: str,
    enabled: bool,
    min_lines: int,
    ratio_threshold: float,
) -> tuple[list[Document], np.ndarray]:
    if not enabled or not docs:
        return docs, scores
    kept_docs: list[Document] = []
    kept_scores: list[float] = []
    removed = 0
    for doc, score in zip(docs, scores):
        if _is_toc_like(
            doc.page_content,
            min_lines=min_lines,
            ratio_threshold=ratio_threshold,
        ):
            removed += 1
            continue
        kept_docs.append(doc)
        kept_scores.append(float(score))
    if kept_docs:
        if removed:
            logging.info(
                "TOC filter removed %s/%s docs for %s retrieval",
                removed,
                len(docs),
                label,
            )
        return kept_docs, np.array(kept_scores, dtype=np.float32)
    if removed:
        logging.info(
            "TOC filter removed all %s docs; keeping original results.",
            label,
        )
    return docs, scores


def retrieve_documents(
    query: str,
    client,
    collection_name: str,
    raw_docs: list[Document],
    embedder: Embeddings,
    bm25_index=None,
    k: int = 6,
    fetch_k: int = 24,
    alpha: float = 0.7,
    source_scope: "SourceScope | None" = None,
    keyword_query: str | None = None,
    process_id: str | None = None,
) -> tuple[list[Document], str]:
    if not process_id:
        raise SystemExit("process_id is required for retrieval.")
    logging.info(
        "Retrieval request: query_len=%s collection=%s process_id=%s k=%s fetch_k=%s",
        len(query or ""),
        collection_name,
        process_id,
        k,
        fetch_k,
    )
    fetch_k = max(fetch_k, k)
    toc_enabled = os.getenv("RAG_TOC_FILTER_ENABLED", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    toc_min_lines = int(os.getenv("RAG_TOC_MIN_LINES", "5"))
    toc_ratio = float(os.getenv("RAG_TOC_LINE_RATIO", "0.45"))
    query_vec = embedder.embed_query(query)
    query_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.source_type",
                match=MatchValue(value="attachment"),
            ),
            FieldCondition(
                key="metadata.process_id",
                match=MatchValue(value=process_id),
            ),
        ]
    )
    attachment_results = _search_qdrant(
        client,
        collection_name,
        query_vec,
        limit=fetch_k,
        query_filter=query_filter,
    )
    logging.info(
        "Qdrant returned %s points for collection=%s process_id=%s",
        len(attachment_results) if attachment_results is not None else 0,
        collection_name,
        process_id,
    )
    attachment_docs, attachment_scores = _docs_from_search_results(
        attachment_results
    )
    if attachment_docs:
        sample_sources = ", ".join(
            _format_source(doc) for doc in attachment_docs[:10]
        )
        logging.info(
            "Qdrant sample sources (%s shown): %s",
            min(10, len(attachment_docs)),
            sample_sources,
        )
    dense_docs = attachment_docs
    dense_scores = attachment_scores
    if source_scope and source_scope.is_active():
        logging.info(
            "Applying scope filter: %s",
            _format_source_scope(source_scope),
        )
        dense_docs, dense_scores = _filter_docs_and_scores_by_scope(
            dense_docs, dense_scores, source_scope
        )
        logging.info(
            "Scope filter kept %s/%s docs",
            len(dense_docs),
            len(attachment_docs),
        )
    dense_docs, dense_scores = _apply_toc_filter(
        dense_docs,
        dense_scores,
        label="vector",
        enabled=toc_enabled,
        min_lines=toc_min_lines,
        ratio_threshold=toc_ratio,
    )

    bm25_docs: list[Document] = []
    bm25_scores = dense_scores[:0]
    bm25_query = (keyword_query or query or "").strip()
    bm25_fetch_k = max(
        int(os.getenv("RAG_BM25_FETCH_K", str(fetch_k))),
        k,
    )
    bm25_max_query_tokens = int(
        os.getenv("RAG_BM25_MAX_QUERY_TOKENS", "128")
    )
    if bm25_index is not None and bm25_query:
        def _bm25_filter(doc: Document) -> bool:
            meta = doc.metadata or {}
            if meta.get("source_type") != "attachment":
                return False
            if process_id and meta.get("process_id") != process_id:
                return False
            return True

        bm25_docs, bm25_scores = _bm25_search(
            bm25_index,
            bm25_query,
            bm25_fetch_k,
            filter_fn=_bm25_filter,
            max_query_tokens=bm25_max_query_tokens,
        )
        if bm25_docs:
            sample_sources = ", ".join(
                _format_source(doc) for doc in bm25_docs[:10]
            )
            logging.info(
                "BM25 sample sources (%s shown): %s",
                min(10, len(bm25_docs)),
                sample_sources,
            )
        if source_scope and source_scope.is_active():
            bm25_before = len(bm25_docs)
            bm25_docs, bm25_scores = _filter_docs_and_scores_by_scope(
                bm25_docs, bm25_scores, source_scope
            )
            logging.info(
                "BM25 scope filter kept %s/%s docs",
                len(bm25_docs),
                bm25_before,
            )
        bm25_docs, bm25_scores = _apply_toc_filter(
            bm25_docs,
            bm25_scores,
            label="bm25",
            enabled=toc_enabled,
            min_lines=toc_min_lines,
            ratio_threshold=toc_ratio,
        )

    if not dense_docs and not bm25_docs:
        return [], "none"
    if dense_docs and not bm25_docs:
        top_docs = dense_docs[:k]
        _tag_retriever(top_docs, "vector")
        logging.info(
            "Top-k sources: %s",
            ", ".join(_format_source(doc) for doc in top_docs),
        )
        return top_docs, "vector"
    if bm25_docs and not dense_docs:
        top_docs = bm25_docs[:k]
        _tag_retriever(top_docs, "bm25")
        logging.info(
            "Top-k sources: %s",
            ", ".join(_format_source(doc) for doc in top_docs),
        )
        return top_docs, "bm25"

    entries: dict[str, dict[str, object]] = {}

    def _add_docs(docs: list[Document], scores, label: str) -> None:
        for doc, score in zip(docs, scores):
            key = _doc_signature(doc)
            entry = entries.get(key)
            if entry is None:
                entry = {"doc": doc, "dense": None, "bm25": None}
                entries[key] = entry
            entry[label] = float(score)

    _add_docs(dense_docs, dense_scores, "dense")
    _add_docs(bm25_docs, bm25_scores, "bm25")

    keys = list(entries.keys())
    merged_docs = [entries[key]["doc"] for key in keys]
    dense_vals = [entries[key]["dense"] for key in keys]
    bm25_vals = [entries[key]["bm25"] for key in keys]
    reranked_docs, _ = _fusion_rerank(
        merged_docs, dense_vals, bm25_vals, k, alpha
    )
    for doc in reranked_docs:
        key = _doc_signature(doc)
        entry = entries.get(key, {})
        if entry.get("dense") is not None:
            _tag_retriever([doc], "vector")
        if entry.get("bm25") is not None:
            _tag_retriever([doc], "bm25")
    logging.info(
        "Top-k sources: %s",
        ", ".join(_format_source(doc) for doc in reranked_docs),
    )
    return reranked_docs, "hybrid"


def run_extraction(
    label: str,
    question: str,
    scope: SourceScope,
    *,
    llm,
    args,
    prompt_dir: Path,
    client,
    collection_name: str,
    attachment_docs: list[Document],
    embedder: Embeddings,
    bm25_index=None,
    llm_stats: dict[str, dict[str, int | float | None]],
    alpha: float,
    active_process_id: str,
    direct_docs: list[Document] | None = None,
    keyword_query: str | None = None,
    retrieval_query: str | None = None,
) -> str:
    print("-" * 60)
    if direct_docs is not None:
        docs = _tag_retriever(direct_docs, f"direct:{label}")
        if not docs:
            print(f"[{label}] No direct documents found.")
            return ""
        context = _format_context(docs)
        prompt = (
            "Use the following context to answer the question. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"{context}\n\nQuestion: {question}\nAnswer:"
        )
        start_time = time.perf_counter()
        response = llm.invoke(prompt)
        elapsed = time.perf_counter() - start_time
        record_llm_stats(llm_stats, label, response, elapsed)
        answer_text = _response_text(response.content)
        prompt_path = save_prompt(label, prompt, answer_text, prompt_dir)
        print(f"[{label}] Prompt saved to {prompt_path}")
        print(f"[{label}] Answer:\n {answer_text}")
        print(f"[{label}] Retrieval: direct")
        print(f"[{label}] Sources:")
        for source in _unique_sources_with_retriever(docs):
            print("-", source)
        return answer_text

    if scope.is_active():
        print(f"[{label}] Retrieval scope: {_format_source_scope(scope)}")
        scope_docs = _filter_docs_by_scope(attachment_docs, scope)
        matched_files = _collect_source_names(scope_docs)
        if matched_files:
            print(f"[{label}] Scope files:")
            for name in matched_files[:20]:
                print("-", name)
        else:
            print(f"[{label}] Scope files: <none>")
            print(
                f"[{label}] No files matched the retrieval scope; aborting search."
            )
            return ""

    docs, strategy = retrieve_documents(
        retrieval_query or question,
        client,
        collection_name,
        attachment_docs,
        embedder,
        bm25_index=bm25_index,
        k=args.k,
        fetch_k=args.fetch_k,
        alpha=alpha,
        source_scope=scope,
        keyword_query=keyword_query,
        process_id=active_process_id,
    )

    if not docs:
        print(f"[{label}] No relevant documents found.")
        return ""

    context = _format_context(docs)
    prompt = (
        "Use the following context to answer the question. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )
    start_time = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start_time
    record_llm_stats(llm_stats, label, response, elapsed)
    answer_text = _response_text(response.content)
    prompt_path = save_prompt(label, prompt, answer_text, prompt_dir)

    print(f"[{label}] Prompt saved to {prompt_path}")
    print(f"[{label}] Answer:\n {answer_text}")
    print(f"[{label}] Retrieval: {strategy}")
    print(f"[{label}] Sources:")
    for source in _unique_sources_with_retriever(docs):
        print("-", source)
    return answer_text


def sum_metric(
    llm_stats: dict[str, dict[str, int | float | None]], key: str
) -> float:
    values = [
        stats.get(key)
        for stats in llm_stats.values()
        if isinstance(stats.get(key), (int, float))
    ]
    return round(sum(values), 3) if values else 0.0


def sum_tokens(
    llm_stats: dict[str, dict[str, int | float | None]],
    key: str,
    labels: tuple[str, str, str] = ("form", "attachment", "compare"),
) -> tuple[int | None, list[str]]:
    total = 0
    missing: list[str] = []
    for label in labels:
        stats = llm_stats.get(label, {})
        val = stats.get(key)
        if isinstance(val, (int, float)):
            total += int(val)
        else:
            missing.append(label)
    return (total if total > 0 else None), missing
    if scope.is_active():
        print(f"[{label}] Retrieval scope: {_format_source_scope(scope)}")
        scope_docs = _filter_docs_by_scope(attachment_docs, scope)
        matched_files = _collect_source_names(scope_docs)
        if matched_files:
            print(f"[{label}] Scope files:")
            for name in matched_files[:20]:
                print("-", name)
        else:
            print(f"[{label}] Scope files: <none>")
            print(
                f"[{label}] No files matched the retrieval scope; aborting search."
            )
            return ""

    docs, strategy = retrieve_documents(
        retrieval_query or question,
        client,
        collection_name,
        attachment_docs,
        embedder,
        k=args.k,
        fetch_k=args.fetch_k,
        alpha=alpha,
        source_scope=scope,
        process_id=active_process_id,
    )

    if not docs:
        print(f"[{label}] No relevant documents found.")
        return ""

    context = _format_context(docs)
    prompt = (
        "Use the following context to answer the question. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )
    start_time = time.perf_counter()
    response = llm.invoke(prompt)
    elapsed = time.perf_counter() - start_time
    record_llm_stats(llm_stats, label, response, elapsed)
    answer_text = _response_text(response.content)
    prompt_path = save_prompt(label, prompt, answer_text, prompt_dir)

    print(f"[{label}] Prompt saved to {prompt_path}")
    print(f"[{label}] Answer:\n {answer_text}")
    print(f"[{label}] Retrieval: {strategy}")
    print(f"[{label}] Sources:")
    for source in _unique_sources_with_retriever(docs):
        print("-", source)
    return answer_text
