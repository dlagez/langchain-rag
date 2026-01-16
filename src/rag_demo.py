from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Qdrant
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient


def load_documents(data_dir: Path):
    paths = sorted(data_dir.glob("*.txt"))
    if not paths:
        raise SystemExit("No .txt files found in data/. Add some and retry.")

    docs = []
    for path in paths:
        loader = TextLoader(str(path), encoding="utf-8")
        docs.extend(loader.load())
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
    return Qdrant.from_documents(
        documents=splits,
        embedding=embeddings,
        client=client,
        collection_name=collection_name,
    )


def main() -> None:
    load_dotenv()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    persist_dir = root / "qdrant"

    docs = load_documents(data_dir)
    vectorstore = get_vectorstore(docs, persist_dir)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    question = "Summarize the key points in the documents."
    result = qa.invoke({"query": question})

    print("Answer:\n", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "unknown"))


if __name__ == "__main__":
    main()
