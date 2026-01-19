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
    _merge_docs_with_retriever,
    _regex_attachment_fallback,
    _source_type_filter,
    _tag_retriever,
    _unique_sources_with_retriever,
)
from util.util import (
    _docs_from_search_results,
    _docs_have_keyword_hits,
    _format_context,
    _hybrid_rerank,
    _keyword_fallback,
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
    """
    核心召回流程：
    1) 向量检索：分别检索表单与附件，融合向量分数。
    2) 范围过滤：根据 SourceScope 限定候选来源。
    3) 关键词/正则回退：在原始文档中进行字面命中兜底。
    4) 融合与裁剪：合并召回结果并返回最终 Top-k 及策略标识。
    """
    fetch_k = max(fetch_k, k)
    # 兜底召回使用原始文档（先做范围过滤，减少无关文本）
    fallback_docs = _filter_docs_by_scope(raw_docs, source_scope)
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
    attachment_docs = _tag_retriever(attachment_docs, "vector:attachment")
    print(f"Retrievers: attachment={len(attachment_docs)}")
    docs = attachment_docs
    vector_scores = attachment_scores
    docs, vector_scores = _filter_docs_and_scores_by_scope(
        docs, vector_scores, source_scope
    )
    if source_scope and source_scope.is_active() and not docs:
        print("No documents matched the current retrieval scope.")
    # 结合关键词分数与向量分数进行融合重排
    docs = _hybrid_rerank(query, docs, vector_scores, k=k, alpha=alpha)
    # 将关键词检索作为备选召回路径（默认用原始 query）
    if keyword_query is None:
        keyword_query = query
    # 兜底召回：正则/关键词规则补充召回结果
    # 正则回退：优先命中附件正文中的关键信息
    regex_docs = _regex_attachment_fallback(
        fallback_docs, keyword_query, limit=min(6, k), source_scope=source_scope
    )
    regex_docs = _tag_retriever(regex_docs, "regex")
    # 关键词回退：字面匹配兜底
    keyword_docs = _keyword_fallback(
        fallback_docs, keyword_query, limit=min(3, k)
    )
    keyword_docs = _tag_retriever(keyword_docs, "keyword")
    had_keyword = bool(keyword_docs)
    had_regex = bool(regex_docs)
    # 正则/关键词结果先行合并
    if had_regex:
        keyword_docs = _merge_docs_with_retriever(regex_docs, keyword_docs)
    # 优先合并兜底命中，保证召回稳定性
    if keyword_docs:
        if docs:
            # 先合并兜底结果，再裁剪到 Top-k
            docs = _merge_docs_with_retriever(keyword_docs, docs)[:k]
            if had_regex and had_keyword:
                return docs, "hybrid+keyword+regex"
            if had_regex:
                return docs, "hybrid+regex"
            return docs, "hybrid+keyword"
        if had_regex and not had_keyword:
            return keyword_docs, "regex_fallback"
        return keyword_docs, "keyword_fallback"
    # 无任何命中则返回空
    if not docs:
        return [], "none"

    # 关键词命中为空时，优先回退到关键词召回结果
    if not _docs_have_keyword_hits(docs, keyword_query):
        if keyword_docs:
            return keyword_docs, "keyword_fallback"

    # 返回融合后的向量召回结果
    return docs, "hybrid"


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
    embedding_provider, embedding_model = resolve_embedding_config()
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
    llm_provider, llm_model = resolve_llm_config()
    llm = build_llm(llm_provider, llm_model)
    logging.info("LLM provider: %s (%s)", llm_provider, llm_model)

    form_question = os.getenv("RAG_FORM_QUESTION", "").strip()
    attachment_question = os.getenv("RAG_ATTACHMENT_QUESTION", "").strip()
    compare_question = os.getenv("RAG_COMPARE_QUESTION", "").strip()
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
            print(f"[{label}] Question: {question}")
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
            question,
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

        print(f"[{label}] Question: {question}")
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
        "form", form_question, form_scope, direct_docs=form_direct_docs
    )
    attachment_answer = _run_extraction(
        "attachment", attachment_question, attachment_scope
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
    print(f"[compare] Question: {compare_question}")
    print(f"[compare] Prompt saved to {compare_prompt_path}")
    print(f"[compare] Answer:\n {compare_answer}")


if __name__ == "__main__":
    main()
