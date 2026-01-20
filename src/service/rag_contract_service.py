from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

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
    _docs_from_search_results,
    _format_context,
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


def retrieve_documents(
    query: str,
    client,
    collection_name: str,
    raw_docs: list[Document],
    embedder: Embeddings,
    k: int = 6,
    fetch_k: int = 24,
    alpha: float = 0.7,
    source_scope: "SourceScope | None" = None,
    keyword_query: str | None = None,
    process_id: str | None = None,
) -> tuple[list[Document], str]:
    if not process_id:
        raise SystemExit("process_id is required for retrieval.")
    fetch_k = max(fetch_k, k)
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
    attachment_docs, attachment_scores = _docs_from_search_results(
        attachment_results
    )
    docs = attachment_docs
    if source_scope and source_scope.is_active():
        docs, _ = _filter_docs_and_scores_by_scope(
            docs, attachment_scores, source_scope
        )
    if not docs:
        return [], "none"
    top_docs = docs[:k]
    logging.info(
        "Top-k sources: %s",
        ", ".join(_format_source(doc) for doc in top_docs),
    )
    return top_docs, "vector"


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
    llm_stats: dict[str, dict[str, int | float | None]],
    alpha: float,
    active_process_id: str,
    direct_docs: list[Document] | None = None,
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
