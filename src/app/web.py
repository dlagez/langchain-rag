from __future__ import annotations

from pathlib import Path
from typing import Any
import re

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from app.settings import Settings
from kb.answer import answer_question
from kb.ingest import ingest_kb
from kb.io.manifest_store import load_file_records, load_index_manifest
from kb.registry import load_kb_paths, resolve_kb_id
from kb.retrieve import retrieve_documents

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Personal KB")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _settings() -> Settings:
    return Settings.from_env(ROOT)


def _list_kbs(settings: Settings) -> list[dict[str, Any]]:
    manifest_root = settings.manifest_dir
    kb_paths = load_kb_paths(manifest_root)
    roots_by_kb: dict[str, list[str]] = {}
    for root, kb_id in kb_paths.items():
        roots_by_kb.setdefault(kb_id, []).append(root)

    items: list[dict[str, Any]] = []
    if not manifest_root.exists():
        return items
    for path in sorted(manifest_root.iterdir()):
        if not path.is_dir():
            continue
        kb_id = path.name
        index_manifest = load_index_manifest(manifest_root, kb_id)
        items.append(
            {
                "kb_id": kb_id,
                "roots": roots_by_kb.get(kb_id, []),
                "doc_count": index_manifest.doc_count if index_manifest else None,
                "chunk_count": index_manifest.chunk_count if index_manifest else None,
                "updated_at": index_manifest.updated_at if index_manifest else None,
            }
        )
    return items


def _resolve_kb_ids(settings: Settings, kb_ids: list[str] | str | None) -> list[str]:
    if kb_ids is None:
        values: list[str] = []
    elif isinstance(kb_ids, str):
        values = [kb_id.strip() for kb_id in kb_ids.split(",") if kb_id.strip()]
    else:
        values = [kb_id.strip() for kb_id in kb_ids if kb_id and kb_id.strip()]

    valid: list[str] = []
    for value in values:
        manifest_dir = settings.manifest_dir / value
        if manifest_dir.exists() and manifest_dir.is_dir():
            valid.append(value)

    if valid:
        return valid

    items = _list_kbs(settings)
    if items:
        return [items[0]["kb_id"]]
    return []


def _default_kb_id(settings: Settings) -> str:
    items = _list_kbs(settings)
    if items:
        return items[0]["kb_id"]
    return ""


def _doc_to_dict(doc) -> dict[str, Any]:
    meta = doc.metadata or {}
    return {
        "content": doc.page_content,
        "source": meta.get("source"),
        "doc_id": meta.get("doc_id"),
        "page": meta.get("page"),
        "sheet": meta.get("sheet"),
        "row": meta.get("row"),
        "chunk_id": meta.get("chunk_id"),
        "source_type": meta.get("source_type"),
    }


def _facet_files(docs: list[dict[str, Any]]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for doc in docs:
        source = doc.get("source") or ""
        name = Path(str(source)).name if source else "unknown"
        counts[name] = counts.get(name, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


_FILE_EXTENSIONS = (
    "txt",
    "pdf",
    "docx",
    "doc",
    "xls",
    "xlsx",
    "csv",
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tif",
    "tiff",
    "webp",
)
_FILE_MENTION_RE = re.compile(
    r"([\w\u4e00-\u9fff\-\.]+\.(" + "|".join(_FILE_EXTENSIONS) + r"))",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_MONTH_RE = re.compile(r"(\d{1,2})月(?:份)?")
_FILTER_SPLIT_RE = re.compile(r"[,\n;，；|]+")


def _question_tokens(question: str) -> list[str]:
    if not question:
        return []
    tokens: set[str] = set()
    for token in _TOKEN_RE.findall(question):
        token = token.strip().lower()
        if len(token) >= 2:
            tokens.add(token)
    for chunk in _CJK_RE.findall(question):
        if len(chunk) < 2:
            continue
        limit = min(3, len(chunk))
        for size in range(2, limit + 1):
            for idx in range(len(chunk) - size + 1):
                tokens.add(chunk[idx : idx + size])
    for match in _MONTH_RE.findall(question):
        tokens.add(f"{match}月")
        tokens.add(f"{match}月份")
    return list(tokens)


def _score_file_name(question: str, file_name: str, tokens: list[str]) -> int:
    if not question or not file_name:
        return 0
    q = question.lower()
    name = file_name.lower()
    base = Path(name).stem
    score = 0
    if name in q:
        score += 30
    if base and base in q:
        score += 20
    for token in tokens:
        if token in base:
            score += 3
        elif token in name:
            score += 1
    return score


def _suggest_file_names(
    settings: Settings, *, kb_ids: list[str], question: str, limit: int = 8
) -> tuple[list[str], str | None]:
    if not kb_ids:
        return [], "暂无可用文件"

    records: list = []
    for kb_id in kb_ids:
        records.extend(load_file_records(settings.manifest_dir, kb_id))

    if not records:
        return [], "暂无可用文件"

    names: dict[str, str] = {}
    for record in records:
        if record.status != "indexed" or record.source_deleted:
            continue
        name = Path(record.abs_path).name
        if not name:
            continue
        updated_at = record.updated_at or ""
        if name not in names or updated_at > names[name]:
            names[name] = updated_at

    if not names:
        return [], "暂无可用文件"

    question = question.strip()
    if not question:
        return [], "输入问题后自动推荐可过滤的文件名"

    tokens = _question_tokens(question)
    mentions = {match[0].lower() for match in _FILE_MENTION_RE.findall(question)}

    scored: list[tuple[int, str, str]] = []
    for name, updated_at in names.items():
        score = _score_file_name(question, name, tokens)
        if name.lower() in mentions:
            score += 50
        if score > 0:
            scored.append((score, updated_at, name))

    if not scored:
        # No obvious match: show latest files to allow manual selection.
        recent = sorted(names.items(), key=lambda item: item[1], reverse=True)
        return [name for name, _ in recent[:limit]], "未识别到文件名，以下为最近文件"

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [name for _, _, name in scored[:limit]], None


def _parse_file_filters(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = [item.strip() for item in _FILTER_SPLIT_RE.split(raw) if item.strip()]
    if not parts:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in parts:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _match_any_file_filter(source: str, filters: list[str]) -> bool:
    if not source or not filters:
        return False
    source_name = Path(source).name.lower()
    source_path = source.lower()
    for item in filters:
        if not item:
            continue
        value = item.lower()
        if source_name == value:
            return True
        if ("/" in value or "\\" in value) and source_path == value:
            return True
    return False


def _retrieve_multi(
    settings: Settings,
    *,
    kb_ids: list[str],
    question: str,
    top_k: int | None,
) -> list[Document]:
    if not kb_ids:
        return []
    if len(kb_ids) == 1:
        return retrieve_documents(
            kb_id=kb_ids[0],
            question=question,
            settings=settings,
            top_k=top_k,
        )
    k_total = top_k or settings.top_k
    per_k = max(1, int((k_total + len(kb_ids) - 1) / len(kb_ids)))
    docs: list[Document] = []
    for kb_id in kb_ids:
        docs.extend(
            retrieve_documents(
                kb_id=kb_id,
                question=question,
                settings=settings,
                top_k=per_k,
            )
        )
    return docs[:k_total]


@app.get("/", response_class=HTMLResponse)
def root() -> RedirectResponse:
    return RedirectResponse(url="/kb")


@app.get("/kb", response_class=HTMLResponse)
def kb_home(request: Request) -> HTMLResponse:
    settings = _settings()
    items = _list_kbs(settings)
    return templates.TemplateResponse(
        "kb.html",
        {
            "request": request,
            "kbs": items,
        },
    )


@app.get("/ingest", response_class=HTMLResponse)
def ingest_page(request: Request, kb_id: str | None = None) -> HTMLResponse:
    settings = _settings()
    return templates.TemplateResponse(
        "ingest.html",
        {
            "request": request,
            "kb_id": kb_id or "",
        },
    )


@app.post("/ingest", response_class=HTMLResponse)
def ingest_action(
    request: Request,
    root_path: str = Form(...),
    kb_id: str | None = Form(None),
) -> HTMLResponse:
    settings = _settings()
    kb_id = ingest_kb(Path(root_path), kb_id=kb_id, settings=settings)
    return templates.TemplateResponse(
        "ingest_done.html",
        {
            "request": request,
            "kb_id": kb_id,
            "root_path": root_path,
        },
    )


@app.get("/query", response_class=HTMLResponse)
def query_page(request: Request, kb_id: str | None = None) -> HTMLResponse:
    settings = _settings()
    default_kb = _default_kb_id(settings)
    kb_list = _list_kbs(settings)
    return templates.TemplateResponse(
        "query.html",
        {
            "request": request,
            "default_kb_id": default_kb,
            "kbs": kb_list,
            "default_top_k": settings.top_k,
        },
    )


@app.post("/search", response_class=HTMLResponse)
def search_action(
    request: Request,
    kb_id: list[str] | None = Form(None),
    question: str = Form(...),
    top_k: int = Form(0),
    file_name: str | None = Form(None),
) -> HTMLResponse:
    settings = _settings()
    kb_ids = _resolve_kb_ids(settings, kb_id)
    if not kb_ids:
        return templates.TemplateResponse(
            "partials/results.html",
            {
                "request": request,
                "answer": "尚未发现可用知识库，请先入库。",
                "citations": [],
                "docs": [],
                "facets": [],
                "kb_id": "",
                "question": question,
            },
        )
    top_k_value = top_k or None
    docs = _retrieve_multi(
        settings,
        kb_ids=kb_ids,
        question=question,
        top_k=top_k_value,
    )

    filters = _parse_file_filters(file_name)
    if filters:
        lowered_filters = [item.lower() for item in filters]
        docs = [
            doc
            for doc in docs
            if _match_any_file_filter(
                str(Path((doc.metadata or {}).get("source") or "")).lower(),
                lowered_filters,
            )
        ]

    doc_dicts = [_doc_to_dict(doc) for doc in docs]

    file_records: dict[str, Any] = {}
    for kb in kb_ids:
        for record in load_file_records(settings.manifest_dir, kb):
            file_records[record.abs_path] = record
    result = answer_question(
        question=question,
        docs=docs,
        settings=settings,
        file_records=file_records,
    )

    facets = _facet_files(doc_dicts)

    return templates.TemplateResponse(
        "partials/results.html",
        {
            "request": request,
            "answer": result.answer,
            "citations": result.citations,
            "docs": doc_dicts,
            "facets": facets,
            "kb_id": kb_id,
            "question": question,
        },
    )


@app.get("/suggest_files", response_class=HTMLResponse)
def suggest_files(
    request: Request,
    kb_id: list[str] | None = None,
    question: str = "",
    limit: int = 8,
) -> HTMLResponse:
    settings = _settings()
    kb_ids = _resolve_kb_ids(settings, kb_id)
    suggestions, message = _suggest_file_names(
        settings, kb_ids=kb_ids, question=question, limit=limit
    )
    return templates.TemplateResponse(
        "partials/file_suggestions.html",
        {
            "request": request,
            "suggestions": suggestions,
            "message": message,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.web:app", host="0.0.0.0", port=8000, reload=True)
