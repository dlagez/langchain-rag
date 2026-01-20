from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def _filter_docs_by_process_id(
    docs: list[Document], process_id: str | None
) -> list[Document]:
    if not process_id:
        return []
    filtered: list[Document] = []
    for doc in docs:
        metadata = doc.metadata or {}
        if metadata.get("process_id") == process_id:
            filtered.append(doc)
            continue
        source = metadata.get("source")
        if source and process_id in Path(source).parts:
            filtered.append(doc)
    return filtered


def _extract_token_usage(response: Any) -> dict[str, int | None]:
    usage = None
    for attr in ("usage_metadata", "response_metadata", "metadata"):
        value = getattr(response, attr, None)
        if isinstance(value, dict):
            if attr == "usage_metadata":
                usage = value
                break
            nested = value.get("usage_metadata")
            if isinstance(nested, dict):
                usage = nested
                break
            nested = value.get("usage")
            if isinstance(nested, dict):
                usage = nested
                break
            nested = value.get("token_usage")
            if isinstance(nested, dict):
                usage = nested
                break
    if usage is None and isinstance(response, dict):
        usage = response.get("usage") or response.get("token_usage")

    def _pick_int(keys: tuple[str, ...]) -> int | None:
        for key in keys:
            val = usage.get(key) if isinstance(usage, dict) else None
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                return int(val)
        return None

    prompt_tokens = _pick_int(("prompt_tokens", "input_tokens", "prompt_tokens_total"))
    completion_tokens = _pick_int(
        ("completion_tokens", "output_tokens", "generated_tokens")
    )
    total_tokens = _pick_int(("total_tokens", "total", "tokens"))
    if (
        total_tokens is None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _resolve_active_process_id(
    docs: list[Document], explicit_process_id: str | None
) -> str:
    process_ids = {
        doc.metadata.get("process_id")
        for doc in docs
        if doc.metadata and doc.metadata.get("process_id")
    }
    if explicit_process_id:
        if process_ids and explicit_process_id not in process_ids:
            logging.warning(
                "Requested process_id %s not found in documents.",
                explicit_process_id,
            )
        return explicit_process_id
    if len(process_ids) == 1:
        return next(iter(process_ids))
    if not process_ids:
        raise SystemExit(
            "process_id missing. Use data/source/<process_id>/form|attachment "
            "or set PROCESS_ID / --process-id."
        )
    raise SystemExit(
        "Multiple process_id values detected; set PROCESS_ID / --process-id to select one."
    )


def _list_process_files(process_dir: Path) -> list[str]:
    if not process_dir.exists():
        return []
    files: list[str] = []
    for path in sorted(process_dir.rglob("*")):
        if path.is_file():
            files.append(path.relative_to(process_dir).as_posix())
    return files


def _resolve_process_dir(
    base_dir: Path, process_id: str | None
) -> tuple[Path, str]:
    if process_id:
        process_dir = base_dir / process_id
        if not process_dir.exists():
            raise SystemExit(f"Process dir not found: {process_dir}")
        return process_dir, process_id

    def _has_category_dirs(path: Path) -> bool:
        names = {child.name.lower() for child in path.iterdir() if child.is_dir()}
        return bool(
            names.intersection(
                {"form", "forms", "attachment", "attachments", "表单", "附件"}
            )
        )

    candidates = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and _has_category_dirs(path)
    ]
    if len(candidates) == 1:
        return candidates[0], candidates[0].name
    if not candidates:
        raise SystemExit(
            f"No process directories found under {base_dir}. "
            "Expected data/source/<process_id>/form|attachment."
        )
    names = ", ".join(path.name for path in candidates)
    raise SystemExit(
        f"Multiple process directories found: {names}. "
        "Set PROCESS_ID or --process-id to choose one."
    )


__all__ = [
    "_extract_token_usage",
    "_filter_docs_by_process_id",
    "_resolve_active_process_id",
    "_list_process_files",
    "_resolve_process_dir",
]
