from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
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
from domain.contract.contract_attachment_selector import ContractAttachmentSelector
from util.util import (
    _docs_from_search_results,
    _format_context,
    _LazyOCR,
    _response_text,
    _search_qdrant,
    process_sources,
)
from util.document_utils import _format_source


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


def _filter_docs_by_process_id(
    docs: list[Document], process_id: str | None
) -> list[Document]:
    if not process_id:
        return []
    filtered = []
    for doc in docs:
        metadata = doc.metadata or {}
        if metadata.get("process_id") == process_id:
            filtered.append(doc)
            continue
        source = metadata.get("source")
        if source and process_id in Path(source).parts:
            filtered.append(doc)
    return filtered


def _resolve_active_process_id(
    docs: list[Document], explicit_process_id: str | None
) -> str:
    process_ids = {
        doc.metadata.get("process_id")
        for doc in docs
        if doc.metadata and doc.metadata.get("process_id")
    }
    if explicit_process_id:
        if process_ids and explicit_process_id not in process_ids:
            logging.warning(
                "Requested process_id %s not found in documents.",
                explicit_process_id,
            )
        return explicit_process_id
    if len(process_ids) == 1:
        return next(iter(process_ids))
    if not process_ids:
        raise SystemExit(
            "process_id missing. Use data/source/<process_id>/form|attachment "
            "or set PROCESS_ID / --process-id."
        )
    raise SystemExit(
        "Multiple process_id values detected; set PROCESS_ID / --process-id to select one."
    )


def _list_process_files(process_dir: Path) -> list[str]:
    base = process_dir
    if not base.exists():
        return []
    files = []
    for path in sorted(base.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(base).as_posix())
    return files


def _resolve_process_dir(
    base_dir: Path, process_id: str | None
) -> tuple[Path, str]:
    if process_id:
        process_dir = base_dir / process_id
        if not process_dir.exists():
            raise SystemExit(f"Process dir not found: {process_dir}")
        return process_dir, process_id

    def _has_category_dirs(path: Path) -> bool:
        names = {child.name.lower() for child in path.iterdir() if child.is_dir()}
        return bool(
            names.intersection(
                {"form", "forms", "attachment", "attachments", "表单", "附件"}
            )
        )

    candidates = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and _has_category_dirs(path)
    ]
    if len(candidates) == 1:
        return candidates[0], candidates[0].name
    if not candidates:
        raise SystemExit(
            f"No process directories found under {base_dir}. "
            "Expected data/source/<process_id>/form|attachment."
        )
    names = ", ".join(path.name for path in candidates)
    raise SystemExit(
        f"Multiple process directories found: {names}. "
        "Set PROCESS_ID or --process-id to choose one."
    )


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
    base_source_dir = root / "data" / "source"
    processed_dir = root / "data" / "processed"
    persist_dir = root / "index"
    embedding_provider, embedding_model = resolve_embedding_config()
    llm_provider, llm_model = resolve_llm_config()

    process_id = (
        args.process_id
        or os.getenv("PROCESS_ID")
        or os.getenv("RAG_PROCESS_ID")
    )
    source_dir, process_id = _resolve_process_dir(base_source_dir, process_id)
    processed_dir = processed_dir / process_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    # 解析源文件（含 OCR）并构建结构化文档
    ocr_tool = _LazyOCR()
    # OCR + 文本抽取 -> 原始文档切分
    raw_source_docs = process_sources(
        source_dir, processed_dir, ocr_tool, image_dpi=args.image_dpi
    )
    logging.info(
        "Loaded raw documents: %s (source_dir=%s)",
        len(raw_source_docs),
        source_dir,
    )
    created_at = os.getenv("RAG_CREATED_AT") or os.getenv("CREATED_AT")
    # 表单/附件结构化处理，补充元数据
    structured_docs, inferred_process_id = _build_contract_documents(
        raw_source_docs,
        source_dir,
        process_id=process_id,
        created_at=created_at,
    )
    logging.info("Structured documents: %s", len(structured_docs))
    active_process_id = _resolve_active_process_id(
        structured_docs, process_id or inferred_process_id
    )
    logging.info("Active process_id: %s", active_process_id)
    base_log_dir = Path(os.getenv("RAG_LOG_DIR", "data/log"))
    if not base_log_dir.is_absolute():
        base_log_dir = root / base_log_dir
    if active_process_id in base_log_dir.parts:
        process_log_dir = base_log_dir
    else:
        process_log_dir = base_log_dir / active_process_id
    process_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RAG_LOG_DIR"] = str(process_log_dir)
    logging.info("Log dir: %s", process_log_dir)
    process_files = _list_process_files(source_dir)
    if process_files:
        logging.info(
            "Process files (%s): %s",
            len(process_files),
            ", ".join(process_files[:50]),
        )
        if len(process_files) > 50:
            logging.info(
                "Process files: %s more not shown",
                len(process_files) - 50,
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
    form_docs_count = sum(
        1
        for doc in structured_docs
        if (doc.metadata or {}).get("source_type") == "form"
    )
    logging.info(
        "Docs by type: form=%s attachment=%s",
        form_docs_count,
        len(attachment_docs),
    )
    selector = ContractAttachmentSelector()
    contract_names = selector.select_contract_names(attachment_docs, top_k=1)
    if contract_names:
        attachment_docs = [
            doc
            for doc in attachment_docs
            if Path((doc.metadata or {}).get("source") or "").name
            in contract_names
        ]
        logging.info("Selected contract file: %s", contract_names[0])
    else:
        logging.warning(
            "No contract file detected; using all attachment documents."
        )

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
        "Vector store: collection=%s, persist_dir=%s",
        collection_name,
        persist_dir,
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
    logging.info(
        "Retrieval queries: form_len=%s attachment_len=%s",
        len(form_retrieval_query),
        len(attachment_retrieval_query),
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
    prompt_base_dir = root / "data" / "prompt"
    prompt_dir = prompt_base_dir / active_process_id
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
    if contract_names:
        attachment_scope = SourceScope(names=tuple(contract_names))
    else:
        attachment_scope = SourceScope(prefix="附件", include_terms=("合同",))

    form_direct_docs = _filter_docs_by_scope(raw_source_docs, form_scope)
    form_direct_docs = _filter_docs_by_process_id(
        form_direct_docs, active_process_id
    )
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
