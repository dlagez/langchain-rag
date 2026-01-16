from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _cjk_ngrams(text: str, min_size: int = 2, max_size: int = 4) -> list[str]:
    ngrams: list[str] = []
    for chunk in _CJK_RE.findall(text):
        limit = min(max_size, len(chunk))
        for size in range(min_size, limit + 1):
            for idx in range(len(chunk) - size + 1):
                ngrams.append(chunk[idx : idx + size])
    seen = set()
    deduped: list[str] = []
    for term in ngrams:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped


def _query_keywords(query: str) -> list[str]:
    strong = _cjk_ngrams(query, min_size=3, max_size=5)
    if strong:
        return strong
    return _cjk_ngrams(query, min_size=2, max_size=4)


def _docs_have_keyword_hits(docs: list[Document], query: str) -> bool:
    keywords = _query_keywords(query)
    if not keywords:
        return True
    for doc in docs:
        text = doc.page_content
        if any(kw in text for kw in keywords):
            return True
    return False


def _keyword_fallback(docs: list[Document], query: str, limit: int = 3) -> list[Document]:
    keywords = _query_keywords(query)
    if not keywords:
        return []
    scored = []
    for doc in docs:
        text = doc.page_content
        score = 0
        for kw in keywords:
            if kw in text:
                score += len(kw)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda item: (-item[0], item[1].metadata.get("page") or 0))
    return [doc for _, doc in scored[:limit]]


def _extract_texts(value) -> list[str]:
    texts = []
    if isinstance(value, str):
        texts.append(value)
        return texts
    if isinstance(value, list):
        if value and isinstance(value[0], (list, tuple)) and len(value[0]) >= 2:
            for line in value:
                if (
                    isinstance(line, (list, tuple))
                    and len(line) >= 2
                    and isinstance(line[1], (list, tuple))
                    and line[1]
                ):
                    text = line[1][0]
                    if isinstance(text, str):
                        texts.append(text)
            if texts:
                return texts
        for item in value:
            texts.extend(_extract_texts(item))
        return texts
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            texts.append(value["text"])
        if isinstance(value.get("texts"), list):
            texts.extend(t for t in value["texts"] if isinstance(t, str))
        for key in ("result", "results"):
            if key in value:
                texts.extend(_extract_texts(value[key]))
        return texts
    return texts


def _documents_from_json(payload, source: Path):
    docs = []
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        for page in payload["results"]:
            page_texts = _extract_texts(page)
            if page_texts:
                text = "\n".join(page_texts)
            else:
                text = json.dumps(page, ensure_ascii=False)
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(source), "page": page.get("page")},
                )
            )
        if docs:
            return docs

    page_texts = _extract_texts(payload)
    text = "\n".join(page_texts) if page_texts else json.dumps(payload, ensure_ascii=False)
    return [Document(page_content=text, metadata={"source": str(source)})]


def load_documents(data_dir: Path):
    paths = sorted(
        list(data_dir.glob("*.txt"))
        + list(data_dir.glob("*.json"))
    )
    if not paths:
        raise SystemExit(
            "No .txt or .json files found in ppocr_results/. Add some and retry."
        )

    docs = []
    for path in paths:
        if path.suffix.lower() == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())
            continue
        if path.suffix.lower() == ".json":
            try:
                raw = path.read_text(encoding="utf-8", errors="replace")
                payload = json.loads(raw)
            except json.JSONDecodeError:
                docs.append(
                    Document(page_content=raw, metadata={"source": str(path)})
                )
                continue
            docs.extend(_documents_from_json(payload, path))
    return docs


def collection_exists(client: QdrantClient, name: str) -> bool:
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False


def get_vectorstore(docs, persist_dir: Path):
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    client = QdrantClient(path=str(persist_dir))
    collection_name = "rag_demo"

    if collection_exists(client, collection_name):
        return Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    splits = splitter.split_documents(docs)
    if not splits:
        raise SystemExit("No content left after splitting documents.")

    vector_size = len(embeddings.embed_documents([splits[0].page_content])[0])
    from qdrant_client.http import models as rest

    client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(
            size=vector_size,
            distance=rest.Distance.COSINE,
        ),
    )
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    vectorstore.add_documents(splits)
    return vectorstore


def main() -> None:
    load_dotenv()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "ppocr_results"
    persist_dir = root / "qdrant"

    raw_docs = load_documents(data_dir)
    vectorstore = get_vectorstore(raw_docs, persist_dir)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20},
    )
    model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    question = "如果要暂停工作，有哪几种情形."
    docs = retriever.invoke(question)
    if not _docs_have_keyword_hits(docs, question):
        fallback_docs = _keyword_fallback(raw_docs, question, limit=3)
        if fallback_docs:
            docs = fallback_docs
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    response = llm.invoke(prompt)

    print("Answer:\n", (response.content or "").strip())
    print("\nSources:")
    for doc in docs:
        print("-", doc.metadata.get("source", "unknown"))


if __name__ == "__main__":
    main()
