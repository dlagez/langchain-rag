from __future__ import annotations

from pathlib import Path
from typing import Any

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
    kb_list = _list_kbs(settings)
    kb_id = kb_id or (kb_list[0]["kb_id"] if kb_list else "")
    return templates.TemplateResponse(
        "query.html",
        {
            "request": request,
            "kb_id": kb_id,
        },
    )


@app.post("/search", response_class=HTMLResponse)
def search_action(
    request: Request,
    kb_id: str = Form(...),
    question: str = Form(...),
    top_k: int = Form(0),
    file_name: str | None = Form(None),
) -> HTMLResponse:
    settings = _settings()
    top_k_value = top_k or None
    docs = retrieve_documents(kb_id=kb_id, question=question, settings=settings, top_k=top_k_value)

    if file_name:
        lowered = file_name.strip().lower()
        if lowered:
            docs = [
                doc
                for doc in docs
                if lowered in str(Path((doc.metadata or {}).get("source") or "")).lower()
            ]

    doc_dicts = [_doc_to_dict(doc) for doc in docs]

    file_records = {r.abs_path: r for r in load_file_records(settings.manifest_dir, kb_id)}
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.web:app", host="0.0.0.0", port=8000, reload=True)
