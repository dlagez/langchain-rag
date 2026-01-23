from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.settings import Settings
from kb.citations import build_citations
from kb.models import AnswerResult, FileRecord
from providers.llm_bailian import generate_answer
from util.document_utils import _format_context


def _trim_context(docs: list[Document], max_chars: int) -> str:
    if max_chars <= 0:
        return _format_context(docs)
    chunks = []
    total = 0
    for doc in docs:
        label = doc.metadata.get("source") or "unknown"
        page = doc.metadata.get("page")
        label_name = Path(label).name
        if page is not None:
            label_name = f"{label_name}#p{page}"
        block = f"[{label_name}]\n{doc.page_content}\n"
        if total + len(block) > max_chars:
            break
        chunks.append(block)
        total += len(block)
    return "\n".join(chunks).strip()


def answer_question(
    *,
    question: str,
    docs: list[Document],
    settings: Settings,
    file_records: dict[str, FileRecord] | None = None,
) -> AnswerResult:
    if settings.llm_provider not in {"bailian", "dashscope"}:
        raise SystemExit(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
    if not docs:
        return AnswerResult(
            answer="无法从知识库中找到依据",
            citations=[],
            context_used="",
        )

    citations = build_citations(docs, file_records=file_records)
    context = _trim_context(docs, settings.max_context_chars)

    if not settings.bailian_api_key:
        return AnswerResult(
            answer="无法从知识库中找到依据",
            citations=citations,
            context_used=context,
        )

    system = (
        "You are a helpful assistant. Answer only using the provided context. "
        "If the context does not contain the answer, respond with: 无法从知识库中找到依据。"
    )
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in Chinese. Cite sources using (filename#page) when possible."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    try:
        answer = generate_answer(
            model=settings.bailian_model,
            api_key=settings.bailian_api_key,
            messages=messages,
        )
    except Exception:
        answer = ""
    if not answer:
        answer = "无法从知识库中找到依据"

    return AnswerResult(answer=answer, citations=citations, context_used=context)
