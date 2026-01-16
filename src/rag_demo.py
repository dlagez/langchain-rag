from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    _unique_sources,
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
) -> tuple[list[Document], str]:
    fetch_k = max(fetch_k, k)
    query_vec = embedder.embed_query(query)
    results = _search_qdrant(client, collection_name, query_vec, limit=fetch_k)
    docs, vector_scores = _docs_from_search_results(results)
    docs = _hybrid_rerank(query, docs, vector_scores, k=k, alpha=alpha)
    if not docs:
        fallback = _keyword_fallback(raw_docs, query, limit=min(3, k))
        return (fallback, "keyword_fallback") if fallback else ([], "none")

    if not _docs_have_keyword_hits(docs, query):
        fallback = _keyword_fallback(raw_docs, query, limit=min(3, k))
        if fallback:
            return fallback, "keyword_fallback"

    return docs, "hybrid"


def get_vectorstore(
    docs: list[Document],
    persist_dir: Path,
    processed_dir: Path,
    force_rebuild: bool = False,
):
    embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    collection_name = os.getenv("QDRANT_COLLECTION", "rag_docs")
    qdrant_location = _qdrant_location(persist_dir)

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    manifest = _build_index_manifest(
        processed_dir, embedding_model, chunk_size, chunk_overlap
    )
    manifest["collection_name"] = collection_name
    manifest["qdrant_location"] = qdrant_location

    client = _get_qdrant_client(persist_dir)

    if not force_rebuild:
        stored_manifest = _load_manifest(persist_dir)
        if _manifest_matches(stored_manifest, manifest) and _collection_exists(
            client, collection_name
        ):
            return client, collection_name, embeddings

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splits = splitter.split_documents(docs)
    if not splits:
        raise SystemExit("No content left after splitting documents.")

    vectors = embeddings.embed_documents([doc.page_content for doc in splits])
    if not vectors:
        raise SystemExit("Embedding model returned no vectors.")
    vector_size = len(vectors[0])
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
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    root = Path(__file__).resolve().parents[1]
    source_dir = root / "data" / "source"
    processed_dir = root / "data" / "processed"
    persist_dir = root / "index"

    ocr_tool = _LazyOCR()
    raw_docs = process_sources(
        source_dir, processed_dir, ocr_tool, image_dpi=args.image_dpi
    )
    client, collection_name, embedder = get_vectorstore(
        raw_docs, persist_dir, processed_dir, force_rebuild=args.rebuild
    )

    model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    question = args.question or os.getenv("QUESTION") or "投标人基本情况是什么？."
    question = question.strip()
    if not question:
        return

    alpha = float(os.getenv("RAG_ALPHA", "0.7"))
    docs, strategy = retrieve_documents(
        question,
        client,
        collection_name,
        raw_docs,
        embedder,
        k=args.k,
        fetch_k=args.fetch_k,
        alpha=alpha,
    )

    if not docs:
        print("No relevant documents found.")
        return

    context = _format_context(docs)
    prompt = (
        "Use the following context to answer the question. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = llm.invoke(prompt)

    print("Answer:\n", _response_text(response.content))
    print(f"\nRetrieval: {strategy}")
    print("\nSources:")
    for source in _unique_sources(docs):
        print("-", source)


if __name__ == "__main__":
    main()
