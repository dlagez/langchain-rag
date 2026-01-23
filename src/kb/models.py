from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FileRecord:
    abs_path: str
    rel_path: str
    hash: str
    mtime_ns: int
    status: str
    source_deleted: bool
    error: str | None
    updated_at: str

    @classmethod
    def from_dict(cls, payload: dict) -> "FileRecord":
        return cls(
            abs_path=str(payload.get("abs_path", "")),
            rel_path=str(payload.get("rel_path", "")),
            hash=str(payload.get("hash", "")),
            mtime_ns=int(payload.get("mtime_ns", 0)),
            status=str(payload.get("status", "")),
            source_deleted=bool(payload.get("source_deleted", False)),
            error=payload.get("error"),
            updated_at=str(payload.get("updated_at", "")),
        )

    def to_dict(self) -> dict:
        return {
            "abs_path": self.abs_path,
            "rel_path": self.rel_path,
            "hash": self.hash,
            "mtime_ns": self.mtime_ns,
            "status": self.status,
            "source_deleted": self.source_deleted,
            "error": self.error,
            "updated_at": self.updated_at,
        }


@dataclass
class IndexManifest:
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    doc_count: int
    chunk_count: int
    updated_at: str

    @classmethod
    def from_dict(cls, payload: dict) -> "IndexManifest":
        return cls(
            embedding_model=str(payload.get("embedding_model", "")),
            chunk_size=int(payload.get("chunk_size", 0)),
            chunk_overlap=int(payload.get("chunk_overlap", 0)),
            doc_count=int(payload.get("doc_count", 0)),
            chunk_count=int(payload.get("chunk_count", 0)),
            updated_at=str(payload.get("updated_at", "")),
        )

    def to_dict(self) -> dict:
        return {
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "doc_count": self.doc_count,
            "chunk_count": self.chunk_count,
            "updated_at": self.updated_at,
        }


@dataclass
class Citation:
    source: str
    page: int | None
    doc_id: str | None
    source_deleted: bool

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "page": self.page,
            "doc_id": self.doc_id,
            "source_deleted": self.source_deleted,
        }


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    context_used: str


@dataclass
class IngestItem:
    abs_path: str
    status: str
    reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "abs_path": self.abs_path,
            "status": self.status,
            "reason": self.reason,
        }


@dataclass
class IngestReport:
    kb_id: str
    root_path: str
    ingested: list[IngestItem]
    skipped: list[IngestItem]
    errors: list[IngestItem]
    created_at: str

    def to_dict(self) -> dict:
        return {
            "kb_id": self.kb_id,
            "root_path": self.root_path,
            "created_at": self.created_at,
            "ingested": [item.to_dict() for item in self.ingested],
            "skipped": [item.to_dict() for item in self.skipped],
            "errors": [item.to_dict() for item in self.errors],
        }


def new_ingest_report(kb_id: str, root_path: str) -> IngestReport:
    return IngestReport(
        kb_id=kb_id,
        root_path=root_path,
        ingested=[],
        skipped=[],
        errors=[],
        created_at=_utc_now(),
    )


def now_iso() -> str:
    return _utc_now()
