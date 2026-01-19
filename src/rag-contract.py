from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from service.rag_services import (
    build_llm,
    build_or_load_vectorstore,
    configure_logging,
    resolve_embedding_config,
    resolve_llm_config,
)
from domain.contract.rag_contract_utils import (
    SourceScope,
    _build_contract_documents,
    _collect_source_names,
    _filter_docs_and_scores_by_scope,
    _filter_docs_by_scope,
    _format_source_scope,
    _match_source_names,
    _source_type_filter,
    _tag_retriever,
    _unique_sources_with_retriever,
)
from util.util import (
    _docs_from_search_results,
    _format_context,
    _LazyOCR,
    _response_text,
    _search_qdrant,
    process_sources,
)


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
) -> tuple[list[Document], str]:
    fetch_k = max(fetch_k, k)
    query_vec = embedder.embed_query(query)
    attachment_results = _search_qdrant(
        client,
        collection_name,
        query_vec,
        limit=fetch_k,
        query_filter=_source_type_filter("attachment"),
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
    return docs[:k], "vector"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG demo")
    parser.add_argument("question", nargs="?", help="Question to ask.")
    parser.add_argument("--k", type=int, default=6, help="Top-k chunks to return.")
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=24,
        help="Candidate pool size for hybrid retrieval.",
    )
    parser.add_argument(
        "--image-dpi", type=int, default=200, help="DPI for OCR PDF rendering."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector index even if cached.",
    )
    parser.add_argument(
        "--process-id",
        help="Process instance ID for metadata (overrides env PROCESS_ID).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    configure_logging()
    args = _parse_args()

    root = Path(__file__).resolve().parents[1]
    source_dir = root / "data" / "source"
    processed_dir = root / "data" / "processed"
    persist_dir = root / "index"
    embedding_provider, embedding_model = resolve_embedding_config()
    llm_provider, llm_model = resolve_llm_config()

    # 解析源文件（含 OCR）并构建结构化文档
    ocr_tool = _LazyOCR()
    # OCR + 文本抽取 -> 原始文档切分
    raw_source_docs = process_sources(
        source_dir, processed_dir, ocr_tool, image_dpi=args.image_dpi
    )
    process_id = (
        args.process_id
        or os.getenv("PROCESS_ID")
        or os.getenv("RAG_PROCESS_ID")
    )
    created_at = os.getenv("RAG_CREATED_AT") or os.getenv("CREATED_AT")
    # 表单/附件结构化处理，补充元数据
    structured_docs, inferred_process_id = _build_contract_documents(
        raw_source_docs,
        source_dir,
        process_id=process_id,
        created_at=created_at,
    )
    if not inferred_process_id:
        logging.warning(
            "process_id is missing; set PROCESS_ID or --process-id for filtering."
        )
    attachment_docs = [
        doc
        for doc in structured_docs
        if (doc.metadata or {}).get("source_type") == "attachment"
    ]
    # 构建或加载向量索引（仅附件）
    client, collection_name, embedder = build_or_load_vectorstore(
        attachment_docs,
        persist_dir,
        processed_dir,
        force_rebuild=args.rebuild,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
    logging.info(
        "Embedding provider: %s (%s)", embedding_provider, embedding_model
    )

    # 初始化 LLM
    llm = build_llm(llm_provider, llm_model)
    logging.info("LLM provider: %s (%s)", llm_provider, llm_model)

    form_question = os.getenv("RAG_FORM_QUESTION", "").strip()
    attachment_question = os.getenv("RAG_ATTACHMENT_QUESTION", "").strip()
    compare_question = os.getenv("RAG_COMPARE_QUESTION", "").strip()
    form_retrieval_query = (
        os.getenv("RAG_FORM_RETRIEVAL_QUERY", "").strip() or form_question
    )
    attachment_retrieval_query = (
        os.getenv("RAG_ATTACHMENT_RETRIEVAL_QUERY", "").strip()
        or attachment_question
    )
    missing = []
    if not form_question:
        missing.append("RAG_FORM_QUESTION")
    if not attachment_question:
        missing.append("RAG_ATTACHMENT_QUESTION")
    if not compare_question:
        missing.append("RAG_COMPARE_QUESTION")
    if missing:
        logging.error(
            "Missing required env vars: %s", ", ".join(missing)
        )
        return

    alpha = float(os.getenv("RAG_ALPHA", "0.7"))
    source_names = _collect_source_names(attachment_docs)
    prompt_dir = root / "data" / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    def _save_prompt(label: str, prompt: str, answer: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        prompt_path = prompt_dir / f"prompt_{timestamp}_{label}.txt"
        payload = f"{prompt}\n\n---\nAnswer:\n{answer}"
        prompt_path.write_text(payload, encoding="utf-8")
        return prompt_path

    def _run_extraction(
        label: str,
        question: str,
        scope: SourceScope,
        *,
        direct_docs: list[Document] | None = None,
        retrieval_query: str | None = None,
    ) -> str:
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
            response = llm.invoke(prompt)
            answer_text = _response_text(response.content)
            prompt_path = _save_prompt(label, prompt, answer_text)
            print(f"[{label}] Prompt saved to {prompt_path}")
            print(f"[{label}] Answer:\n {answer_text}")
            print(f"[{label}] Retrieval: direct")
            print(f"[{label}] Sources:")
            for source in _unique_sources_with_retriever(docs):
                print("-", source)
            return answer_text

        if scope.is_active():
            print(
                f"[{label}] Retrieval scope: {_format_source_scope(scope)}"
            )
            matched_files = _match_source_names(scope, source_names)
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
        response = llm.invoke(prompt)
        answer_text = _response_text(response.content)
        prompt_path = _save_prompt(label, prompt, answer_text)

        print(f"[{label}] Prompt saved to {prompt_path}")
        print(f"[{label}] Answer:\n {answer_text}")
        print(f"[{label}] Retrieval: {strategy}")
        print(f"[{label}] Sources:")
        for source in _unique_sources_with_retriever(docs):
            print("-", source)
        return answer_text

    form_scope = SourceScope(prefix="表单")
    attachment_scope = SourceScope(prefix="附件", include_terms=("合同",))

    form_direct_docs = _filter_docs_by_scope(raw_source_docs, form_scope)
    form_answer = _run_extraction(
        "form",
        form_question,
        form_scope,
        direct_docs=form_direct_docs,
        retrieval_query=form_retrieval_query,
    )
    attachment_answer = _run_extraction(
        "attachment",
        attachment_question,
        attachment_scope,
        retrieval_query=attachment_retrieval_query,
    )

    compare_prompt = (
        "Use the following extracted information to answer the question. "
        "If the answer is not in the extracted information, say you do not know.\n\n"
        f"Form extraction:\n{form_answer or '<none>'}\n\n"
        f"Attachment extraction:\n{attachment_answer or '<none>'}\n\n"
        f"Question: {compare_question}\nAnswer:"
    )
    compare_response = llm.invoke(compare_prompt)
    compare_answer = _response_text(compare_response.content)
    compare_prompt_path = _save_prompt("compare", compare_prompt, compare_answer)
    print(f"[compare] Prompt saved to {compare_prompt_path}")
    print(f"[compare] Answer:\n {compare_answer}")


if __name__ == "__main__":
    main()
