from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from util.rag_contract_utils import (
    SourceScope,
    _assign_chunk_ids,
    _build_contract_documents,
    _build_keyword_query,
    _build_retrieval_query,
    _chunk_documents_for_index,
    _collect_source_names,
    _filter_docs_and_scores_by_scope,
    _filter_docs_by_scope,
    _format_source_scope,
    _infer_source_scope,
    _match_source_names,
    _merge_docs_with_retriever,
    _regex_attachment_fallback,
    _source_type_filter,
    _tag_retriever,
    _unique_sources_with_retriever,
)
from util.util import (
    _build_index_manifest,
    _collection_exists,
    _docs_from_search_results,
    _docs_have_keyword_hits,
    _format_context,
    _get_qdrant_client,
    _hybrid_rerank,
    _keyword_fallback,
    _LazyOCR,
    _load_manifest,
    _manifest_matches,
    _qdrant_location,
    _recreate_collection,
    _response_text,
    _save_manifest,
    _search_qdrant,
    _upsert_documents,
    process_sources,
)


def retrieve_documents(
    query: str,
    client,
    collection_name: str,
    raw_docs: list[Document],
    embedder: GoogleGenerativeAIEmbeddings,
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
    # 向量检索：分别对表单与附件进行召回
    query_vec = embedder.embed_query(query)
    # 表单类向量检索
    form_results = _search_qdrant(
        client,
        collection_name,
        query_vec,
        limit=fetch_k,
        query_filter=_source_type_filter("form"),
    )
    # 附件类向量检索
    attachment_results = _search_qdrant(
        client,
        collection_name,
        query_vec,
        limit=fetch_k,
        query_filter=_source_type_filter("attachment"),
    )
    form_docs, form_scores = _docs_from_search_results(form_results)
    attachment_docs, attachment_scores = _docs_from_search_results(
        attachment_results
    )
    # 标记召回来源，便于调试
    form_docs = _tag_retriever(form_docs, "vector:form")
    attachment_docs = _tag_retriever(attachment_docs, "vector:attachment")
    print(
        f"Retrievers: form={len(form_docs)}, attachment={len(attachment_docs)}"
    )
    # 合并不同来源的候选文档
    docs = form_docs + attachment_docs
    if form_scores.size == 0:
        vector_scores = attachment_scores
    elif attachment_scores.size == 0:
        vector_scores = form_scores
    else:
        vector_scores = np.concatenate([form_scores, attachment_scores])
    # 按检索范围过滤候选与分数
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


def get_vectorstore(
    docs: list[Document],
    persist_dir: Path,
    processed_dir: Path,
    force_rebuild: bool = False,
):
    """
    根据文档构建/复用向量库：生成索引清单、检查缓存一致性，
    必要时重建集合并写入向量，最终返回 client/collection/embeddings。
    """
    embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    # 理论上用于文本分块时的重叠长度，保证相邻 chunk 之间有上下文衔接（减少边界信息丢失）
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    collection_name = os.getenv("QDRANT_COLLECTION", "contract_approval_rag")
    qdrant_location = _qdrant_location(persist_dir)

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    # 生成“索引清单（manifest）”，用来判断能不能复用已建好的向量库，避免每次都重建。
    # collection_name、qdrant_location、ingestion_schema：记录这次索引用的集合名、存储位置、以及你的自定义版本号/结构标识。
    # 后面会把这份 manifest 和磁盘上已有的 manifest 对比；如果一致，就直接复用现有向量库，不重建。
    manifest = _build_index_manifest(
        processed_dir, embedding_model, chunk_size, chunk_overlap
    )
    manifest["collection_name"] = collection_name
    manifest["qdrant_location"] = qdrant_location
    manifest["ingestion_schema"] = "contract_approval_v3"
    manifest["chunking_strategy"] = "structured_v1"
    manifest["chunking_params"] = {
        "contract_min": max(200, int(chunk_size * 0.5)),
        "contract_max": chunk_size,
        "overlap": chunk_overlap,
        "checklist_min": min(200, min(600, chunk_size)),
        "checklist_max": min(600, chunk_size),
    }

    client = _get_qdrant_client(persist_dir)

    # 若索引配置与缓存一致，直接复用已有向量库
    if not force_rebuild:
        stored_manifest = _load_manifest(persist_dir)
        if _manifest_matches(stored_manifest, manifest) and _collection_exists(
            client, collection_name
        ):
            return client, collection_name, embeddings

    # 分块、向量化并写入向量库，为分块添加 chunk_id 便于追踪
    chunked_docs = _chunk_documents_for_index(
        docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    splits = _assign_chunk_ids(chunked_docs)
    if not splits:
        raise SystemExit("No content left after splitting documents.")

    # 向量化并写入向量库
    vectors = embeddings.embed_documents([doc.page_content for doc in splits])
    if not vectors:
        raise SystemExit("Embedding model returned no vectors.")
    vector_size = len(vectors[0])
    # 重建向量集合并写入
    _recreate_collection(client, collection_name, vector_size)
    _upsert_documents(client, collection_name, splits, vectors)
    _save_manifest(persist_dir, manifest, doc_count=len(splits))
    return client, collection_name, embeddings


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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    root = Path(__file__).resolve().parents[1]
    source_dir = root / "data" / "source"
    processed_dir = root / "data" / "processed"
    persist_dir = root / "index"

    # 解析源文件（含 OCR）并构建结构化文档
    ocr_tool = _LazyOCR()
    # OCR + 文本抽取 -> 原始文档切分
    raw_docs = process_sources(
        source_dir, processed_dir, ocr_tool, image_dpi=args.image_dpi
    )
    process_id = (
        args.process_id
        or os.getenv("PROCESS_ID")
        or os.getenv("RAG_PROCESS_ID")
    )
    created_at = os.getenv("RAG_CREATED_AT") or os.getenv("CREATED_AT")
    # 表单/附件结构化处理，补充元数据
    raw_docs, inferred_process_id = _build_contract_documents(
        raw_docs,
        source_dir,
        process_id=process_id,
        created_at=created_at,
    )
    if not inferred_process_id:
        logging.warning(
            "process_id is missing; set PROCESS_ID or --process-id for filtering."
        )
    # 构建或加载向量索引
    client, collection_name, embedder = get_vectorstore(
        raw_docs, persist_dir, processed_dir, force_rebuild=args.rebuild
    )

    # 初始化 LLM
    model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    question = args.question or os.getenv("QUESTION") or "hello"
    question = question.strip()
    if not question:
        return

    alpha = float(os.getenv("RAG_ALPHA", "0.7"))
    # 解析问题，生成检索/关键词查询并确定检索范围
    retrieval_query = _build_retrieval_query(question)
    keyword_query = _build_keyword_query(question, retrieval_query)
    source_names = _collect_source_names(raw_docs)
    source_scope = _infer_source_scope(question, source_names, raw_docs)
    print(f"Retrieval query:\n {retrieval_query}")
    if keyword_query != retrieval_query:
        print(f"Keyword query:\n {keyword_query}")
    # 打印检索范围，便于排查召回问题
    if source_scope.is_active():
        print(f"Retrieval scope: {_format_source_scope(source_scope)}")
        matched_files = _match_source_names(source_scope, source_names)
        if matched_files:
            print("Scope files:")
            for name in matched_files[:20]:
                print("-", name)
        else:
            print("Scope files: <none>")
            print("No files matched the retrieval scope; aborting search.")
            return
    # 检索召回并生成上下文
    # 核心召回：融合向量 + 关键词/正则回退
    docs, strategy = retrieve_documents(
        retrieval_query,
        client,
        collection_name,
        raw_docs,
        embedder,
        k=args.k,
        fetch_k=args.fetch_k,
        alpha=alpha,
        source_scope=source_scope,
        keyword_query=keyword_query,
    )

    if not docs:
        print("No relevant documents found.")
        return

    # 拼接上下文并调用 LLM 生成答案
    context = _format_context(docs)
    # 构造提示词并请求模型回答
    prompt = (
        "Use the following context to answer the question. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )
    prompt_dir = root / "data" / "prompt"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_path = prompt_dir / f"prompt_{timestamp}.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    response = llm.invoke(prompt)

    print(f"Prompt saved to {prompt_path}")
    print("Answer:\n", _response_text(response.content))
    print(f"\nRetrieval: {strategy}")
    print("\nSources:")
    for source in _unique_sources_with_retriever(docs):
        print("-", source)


if __name__ == "__main__":
    main()
