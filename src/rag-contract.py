from __future__ import annotations

import argparse
import json
import time
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
    _filter_docs_by_scope,
)
from domain.contract.contract_attachment_selector import ContractAttachmentSelector
from util.util import (
    _format_context,
    _response_text,
    _LazyOCR,
    process_sources,
)
from util.rag_utils import (
    _filter_docs_by_process_id,
    _list_process_files,
    _resolve_process_dir,
    _resolve_active_process_id,
)
from service.rag_contract_service import (
    record_llm_stats,
    run_extraction,
    save_prompt,
    sum_metric,
    sum_tokens,
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
    run_started = time.perf_counter()

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
    prompt_base_dir = root / "data" / "prompt"
    prompt_dir = prompt_base_dir / active_process_id
    prompt_dir.mkdir(parents=True, exist_ok=True)
    llm_stats: dict[str, dict[str, int | float | None]] = {}

    form_scope = SourceScope(prefix="表单")
    if contract_names:
        attachment_scope = SourceScope(names=tuple(contract_names))
    else:
        attachment_scope = SourceScope(prefix="附件", include_terms=("合同",))

    form_direct_docs = _filter_docs_by_scope(raw_source_docs, form_scope)
    form_direct_docs = _filter_docs_by_process_id(
        form_direct_docs, active_process_id
    )
    form_answer = run_extraction(
        "form",
        form_question,
        form_scope,
        llm=llm,
        args=args,
        prompt_dir=prompt_dir,
        client=client,
        collection_name=collection_name,
        attachment_docs=attachment_docs,
        embedder=embedder,
        llm_stats=llm_stats,
        alpha=alpha,
        active_process_id=active_process_id,
        direct_docs=form_direct_docs,
        retrieval_query=form_retrieval_query,
    )
    attachment_answer = run_extraction(
        "attachment",
        attachment_question,
        attachment_scope,
        llm=llm,
        args=args,
        prompt_dir=prompt_dir,
        client=client,
        collection_name=collection_name,
        attachment_docs=attachment_docs,
        embedder=embedder,
        llm_stats=llm_stats,
        alpha=alpha,
        active_process_id=active_process_id,
        retrieval_query=attachment_retrieval_query,
    )

    compare_prompt = (
        "Use the following extracted information to answer the question. "
        "If the answer is not in the extracted information, say you do not know.\n\n"
        f"Form extraction:\n{form_answer or '<none>'}\n\n"
        f"Attachment extraction:\n{attachment_answer or '<none>'}\n\n"
        f"Question: {compare_question}\nAnswer:"
    )
    compare_start = time.perf_counter()
    compare_response = llm.invoke(compare_prompt)
    compare_elapsed = time.perf_counter() - compare_start
    record_llm_stats(llm_stats, "compare", compare_response, compare_elapsed)
    compare_answer = _response_text(compare_response.content)
    compare_prompt_path = save_prompt("compare", compare_prompt, compare_answer, prompt_dir)
    print(f"[compare] Prompt saved to {compare_prompt_path}")
    print(f"[compare] Answer:\n {compare_answer}")

    token_report = {
        "form": llm_stats.get("form", {}),
        "attachment": llm_stats.get("attachment", {}),
        "compare": llm_stats.get("compare", {}),
    }
    total_prompt, missing_prompt = sum_tokens(llm_stats, "prompt_tokens")
    total_completion, missing_completion = sum_tokens(llm_stats, "completion_tokens")
    total_tokens, missing_total = sum_tokens(llm_stats, "total_tokens")
    token_report["total"] = {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_tokens,
        "seconds": sum_metric(llm_stats, "seconds"),
        "missing_prompt_tokens": missing_prompt,
        "missing_completion_tokens": missing_completion,
        "missing_total_tokens": missing_total,
    }
    token_report["rectification_seconds"] = llm_stats.get("compare", {}).get(
        "seconds"
    )
    token_report["elapsed_seconds"] = round(
        time.perf_counter() - run_started, 3
    )

    metrics_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    metrics_path = (
        prompt_dir / f"metrics_{metrics_timestamp}_rag_usage.json"
    )
    metrics_path.write_text(
        json.dumps(token_report, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"[metrics] Usage saved to {metrics_path}")


if __name__ == "__main__":
    main()
