from __future__ import annotations

import json
from pathlib import Path

from ..models import FileRecord, IndexManifest, IngestReport


def kb_manifest_dir(manifest_root: Path, kb_id: str) -> Path:
    path = manifest_root / kb_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_file_records(manifest_root: Path, kb_id: str) -> list[FileRecord]:
    path = kb_manifest_dir(manifest_root, kb_id) / "files.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    records = []
    for item in payload:
        if isinstance(item, dict):
            records.append(FileRecord.from_dict(item))
    return records


def save_file_records(manifest_root: Path, kb_id: str, records: list[FileRecord]) -> None:
    path = kb_manifest_dir(manifest_root, kb_id) / "files.json"
    payload = [record.to_dict() for record in records]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_ingest_report(manifest_root: Path, kb_id: str, report: IngestReport) -> None:
    path = kb_manifest_dir(manifest_root, kb_id) / "ingest_report.json"
    payload = report.to_dict()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index_manifest(manifest_root: Path, kb_id: str) -> IndexManifest | None:
    path = kb_manifest_dir(manifest_root, kb_id) / "index_manifest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return IndexManifest.from_dict(payload)


def save_index_manifest(manifest_root: Path, kb_id: str, manifest: IndexManifest) -> None:
    path = kb_manifest_dir(manifest_root, kb_id) / "index_manifest.json"
    payload = manifest.to_dict()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
