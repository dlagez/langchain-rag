from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document


_HEADING_PATTERNS = (
    re.compile(r"^第[一二三四五六七八九十百千0-9]+[章节条款].*$"),
    re.compile(r"^附件[一二三四五六七八九十0-9]+.*$"),
    re.compile(r"^(chapter|section|article|appendix)\b.*$", re.IGNORECASE),
    re.compile(r"^\d+(?:\.\d+){0,3}\s+\S+.*$"),
)


def _is_heading(line: str) -> bool:
    if not line:
        return False
    if len(line) > 120:
        return False
    for pattern in _HEADING_PATTERNS:
        if pattern.match(line):
            return True
    return False


def _split_structured(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines()]
    sections: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                current.append("")
            continue
        if _is_heading(line) and current:
            sections.append("\n".join(current).strip())
            current = [line]
            continue
        current.append(line)
    if current:
        sections.append("\n".join(current).strip())
    return [section for section in sections if section]


def _split_by_length(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    if chunk_size <= 0:
        yield text
        return
    step = max(1, chunk_size - max(0, overlap))
    idx = 0
    length = len(text)
    while idx < length:
        yield text[idx : idx + chunk_size]
        idx += step


def chunk_documents(
    docs: list[Document],
    *,
    kb_id: str,
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    chunks: list[Document] = []
    chunk_id = 0
    for doc in docs:
        base_meta = dict(doc.metadata or {})
        base_meta["kb_id"] = kb_id
        base_meta["doc_id"] = doc_id
        text = doc.page_content or ""
        for section in _split_structured(text):
            for piece in _split_by_length(section, chunk_size, chunk_overlap):
                piece = piece.strip()
                if not piece:
                    continue
                meta = dict(base_meta)
                meta["chunk_id"] = chunk_id
                chunks.append(Document(page_content=piece, metadata=meta))
                chunk_id += 1
    return chunks
