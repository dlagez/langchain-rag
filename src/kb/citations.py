from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from kb.models import Citation, FileRecord


def _record_deleted(records: dict[str, FileRecord], source: str) -> bool:
    record = records.get(source)
    if record is None:
        return False
    return bool(record.source_deleted or record.status == "deleted")


def build_citations(
    docs: list[Document],
    *,
    file_records: dict[str, FileRecord] | None = None,
) -> list[Citation]:
    file_records = file_records or {}
    seen: set[tuple[str, int | None]] = set()
    citations: list[Citation] = []
    for doc in docs:
        metadata = doc.metadata or {}
        source = str(metadata.get("source") or "unknown")
        page = metadata.get("page")
        doc_id = metadata.get("doc_id")
        key = (source, page if isinstance(page, int) else None)
        if key in seen:
            continue
        seen.add(key)
        source_deleted = _record_deleted(file_records, source) or not Path(source).exists()
        citations.append(
            Citation(
                source=source,
                page=page if isinstance(page, int) else None,
                doc_id=str(doc_id) if doc_id is not None else None,
                source_deleted=source_deleted,
            )
        )
    return citations
