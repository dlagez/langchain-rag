from __future__ import annotations

import hashlib
import json
from pathlib import Path


_KB_PATHS = "kb_paths.json"


def _sanitize_kb_id(value: str) -> str:
    cleaned = []
    for ch in value.strip().lower():
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    kb_id = "".join(cleaned).strip("_")
    return kb_id or "kb"


def _unique_kb_id(base: str, existing: set[str], root_path: Path) -> str:
    if base not in existing:
        return base
    digest = hashlib.sha1(str(root_path).encode("utf-8")).hexdigest()[:6]
    candidate = f"{base}_{digest}"
    if candidate not in existing:
        return candidate
    idx = 2
    while True:
        candidate = f"{base}_{digest}_{idx}"
        if candidate not in existing:
            return candidate
        idx += 1


def load_kb_paths(manifest_dir: Path) -> dict[str, str]:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / _KB_PATHS
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def save_kb_paths(manifest_dir: Path, mapping: dict[str, str]) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_dir / _KB_PATHS
    payload = json.dumps(mapping, ensure_ascii=False, indent=2)
    path.write_text(payload, encoding="utf-8")


def resolve_kb_id(
    root_path: Path, manifest_dir: Path, kb_id: str | None = None
) -> str:
    root_path = root_path.resolve()
    mapping = load_kb_paths(manifest_dir)
    existing_ids = set(mapping.values())
    root_key = str(root_path)

    if kb_id:
        mapping[root_key] = kb_id
        save_kb_paths(manifest_dir, mapping)
        return kb_id

    if root_key in mapping:
        return mapping[root_key]

    base = _sanitize_kb_id(root_path.name)
    kb_id = _unique_kb_id(base, existing_ids, root_path)
    mapping[root_key] = kb_id
    save_kb_paths(manifest_dir, mapping)
    return kb_id
