from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.documents import Document

from .text_utils import _normalize_text, _split_pages


def _build_metadata(path: Path, page: int | None = None) -> dict:
    metadata = {
        "source": str(path),
        "source_type": path.suffix.lower().lstrip("."),
    }
    if page is not None:
        metadata["page"] = page
    return metadata


def _docs_from_text(
    path: Path,
    text: str,
    use_page_markers: bool = False,
    force_page: bool = False,
) -> list[Document]:
    text = _normalize_text(text)
    if not text:
        return []
    pages = _split_pages(text, use_page_markers=use_page_markers)
    docs: list[Document] = []
    use_page = force_page or (use_page_markers and len(pages) > 1)
    for idx, page_text in enumerate(pages, start=1):
        page_text = page_text.strip()
        if not page_text:
            continue
        metadata = _build_metadata(path, page=idx if use_page else None)
        docs.append(Document(page_content=page_text, metadata=metadata))
    return docs


def _format_source(doc: Document) -> str:
    source = doc.metadata.get("source") or "unknown"
    name = Path(source).name
    page = doc.metadata.get("page")
    if page is not None:
        return f"{name}#p{page}"
    return name


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for doc in docs:
        label = _format_source(doc)
        blocks.append(f"[{label}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _unique_sources(docs: list[Document]) -> list[str]:
    seen = set()
    sources: list[str] = []
    for doc in docs:
        label = _format_source(doc)
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


def _response_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return str(content)


def _doc_signature(doc: Document) -> tuple:
    source = doc.metadata.get("source")
    page = doc.metadata.get("page")
    content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
    return (source, page, content_hash)


def _merge_docs(primary: list[Document], secondary: list[Document]) -> list[Document]:
    seen = set()
    merged: list[Document] = []
    for doc in primary + secondary:
        key = _doc_signature(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
    return merged
