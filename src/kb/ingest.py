from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from app.settings import Settings
from kb.chunk import chunk_documents
from kb.index import (
    count_points,
    delete_doc_chunks,
    get_client_and_collection,
    index_chunks,
    rebuild_bm25,
)
from kb.io.log_store import configure_logging
from kb.io.manifest_store import load_file_records, save_file_records, save_index_manifest, save_ingest_report
from kb.io.processed_store import write_processed_text
from kb.models import FileRecord, IndexManifest, IngestItem, new_ingest_report, now_iso
from kb.parse import build_ocr, extract_documents
from kb.registry import resolve_kb_id

logger = logging.getLogger(__name__)

_ALLOWED_SUFFIXES = {
    ".txt",
    ".pdf",
    ".docx",
    ".doc",
    ".xls",
    ".xlsx",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _iter_files(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*")) if path.is_file()]


def _log_level_from_env() -> int:
    level_name = (os.getenv("RAG_LOG_LEVEL") or "INFO").upper()
    try:
        return getattr(logging, level_name)
    except AttributeError:
        return logging.INFO


def ingest_kb(root_path: Path, *, kb_id: str | None, settings: Settings) -> str:
    root_path = Path(root_path)
    if not root_path.is_absolute():
        raise SystemExit("Root path must be an absolute path.")
    if not root_path.exists():
        raise SystemExit(f"Root path not found: {root_path}")

    kb_id = resolve_kb_id(root_path, settings.manifest_dir, kb_id=kb_id)

    configure_logging(settings.log_dir, kb_id, level=_log_level_from_env())

    report = new_ingest_report(kb_id, str(root_path))

    records = load_file_records(settings.manifest_dir, kb_id)
    record_map = {record.abs_path: record for record in records}

    ocr = build_ocr(settings)

    seen_paths: set[str] = set()
    any_indexed = False

    client = None
    collection = None

    def ensure_client():
        nonlocal client, collection
        if client is None:
            client, collection = get_client_and_collection(settings, kb_id)
        return client, collection

    for path in _iter_files(root_path):
        abs_path = str(path.resolve())
        seen_paths.add(abs_path)
        rel_path = path.relative_to(root_path)
        suffix = path.suffix.lower()

        if suffix not in _ALLOWED_SUFFIXES:
            report.skipped.append(
                IngestItem(abs_path=abs_path, status="skipped", reason="unsupported")
            )
            record_map[abs_path] = FileRecord(
                abs_path=abs_path,
                rel_path=rel_path.as_posix(),
                hash="",
                mtime_ns=path.stat().st_mtime_ns,
                status="skipped",
                source_deleted=False,
                error="unsupported",
                updated_at=now_iso(),
            )
            continue

        mtime_ns = path.stat().st_mtime_ns
        file_hash = _hash_file(path)
        existing = record_map.get(abs_path)

        if (
            existing
            and existing.hash == file_hash
            and existing.mtime_ns == mtime_ns
            and existing.status == "indexed"
            and not existing.source_deleted
        ):
            report.skipped.append(
                IngestItem(abs_path=abs_path, status="skipped", reason="unchanged")
            )
            continue

        doc_id = rel_path.as_posix()
        try:
            docs, text = extract_documents(path, ocr, image_dpi=200)
        except Exception as exc:
            logger.warning("Parse failed for %s: %s", path, exc)
            report.errors.append(
                IngestItem(abs_path=abs_path, status="error", reason=str(exc))
            )
            record_map[abs_path] = FileRecord(
                abs_path=abs_path,
                rel_path=doc_id,
                hash=file_hash,
                mtime_ns=mtime_ns,
                status="error",
                source_deleted=False,
                error=str(exc),
                updated_at=now_iso(),
            )
            continue

        if not docs:
            report.errors.append(
                IngestItem(abs_path=abs_path, status="error", reason="no_text")
            )
            record_map[abs_path] = FileRecord(
                abs_path=abs_path,
                rel_path=doc_id,
                hash=file_hash,
                mtime_ns=mtime_ns,
                status="error",
                source_deleted=False,
                error="no_text",
                updated_at=now_iso(),
            )
            continue

        chunks = chunk_documents(
            docs,
            kb_id=kb_id,
            doc_id=doc_id,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        if not chunks:
            report.errors.append(
                IngestItem(abs_path=abs_path, status="error", reason="no_chunks")
            )
            record_map[abs_path] = FileRecord(
                abs_path=abs_path,
                rel_path=doc_id,
                hash=file_hash,
                mtime_ns=mtime_ns,
                status="error",
                source_deleted=False,
                error="no_chunks",
                updated_at=now_iso(),
            )
            continue

        client, collection = ensure_client()
        if existing and existing.status == "indexed":
            delete_doc_chunks(client, collection, kb_id=kb_id, doc_id=doc_id)

        if settings.embedding_provider in {"bailian", "dashscope"} and not settings.bailian_api_key:
            raise SystemExit("BAILIAN_API_KEY is required for indexing.")
        if settings.embedding_provider == "local":
            if not settings.local_embedding_base_url:
                raise SystemExit("LOCAL_EMBEDDING_BASE_URL is required for indexing.")
            if not settings.local_embedding_model:
                raise SystemExit("LOCAL_EMBEDDING_MODEL is required for indexing.")

        indexed_count = index_chunks(
            client=client,
            collection=collection,
            docs=chunks,
            settings=settings,
        )
        if indexed_count:
            any_indexed = True

        if text:
            write_processed_text(settings.processed_dir, kb_id, rel_path, text)

        record_map[abs_path] = FileRecord(
            abs_path=abs_path,
            rel_path=doc_id,
            hash=file_hash,
            mtime_ns=mtime_ns,
            status="indexed",
            source_deleted=False,
            error=None,
            updated_at=now_iso(),
        )
        report.ingested.append(
            IngestItem(abs_path=abs_path, status="indexed", reason=None)
        )

    for abs_path, record in list(record_map.items()):
        if abs_path in seen_paths:
            continue
        if record.source_deleted:
            continue
        record.source_deleted = True
        record.status = "deleted"
        record.updated_at = now_iso()
        report.skipped.append(
            IngestItem(abs_path=abs_path, status="deleted", reason="source_missing")
        )

    save_file_records(settings.manifest_dir, kb_id, list(record_map.values()))
    save_ingest_report(settings.manifest_dir, kb_id, report)

    if any_indexed:
        client, collection = ensure_client()
        if settings.bm25_enabled:
            rebuild_bm25(client, collection, settings, kb_id)
        chunk_count = count_points(client, collection)
        doc_count = len(
            [
                record
                for record in record_map.values()
                if record.status == "indexed" and not record.source_deleted
            ]
        )
        embedding_model = (
            settings.local_embedding_model
            if settings.embedding_provider == "local"
            else settings.bailian_embedding_model
        )
        manifest = IndexManifest(
            embedding_model=embedding_model or "",
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            doc_count=doc_count,
            chunk_count=chunk_count,
            updated_at=now_iso(),
        )
        save_index_manifest(settings.manifest_dir, kb_id, manifest)

    return kb_id
