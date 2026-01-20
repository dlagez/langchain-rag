from __future__ import annotations

from pathlib import Path
import re

from langchain_core.documents import Document

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_ATTACHMENT_RE = re.compile(r"附件\s*([0-9]+)")
_ATTACHMENT_CN_RE = re.compile(r"附件\s*([一二三四五六七八九十])")
_CN_NUM_MAP = {
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
}


def _cjk_ngrams(text: str, min_size: int = 2, max_size: int = 4) -> list[str]:
    ngrams: list[str] = []
    for chunk in _CJK_RE.findall(text):
        limit = min(max_size, len(chunk))
        for size in range(min_size, limit + 1):
            for idx in range(len(chunk) - size + 1):
                ngrams.append(chunk[idx : idx + size])
    seen = set()
    deduped: list[str] = []
    for term in ngrams:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped


def _latin_keywords(text: str) -> list[str]:
    tokens = [token.lower() for token in _WORD_RE.findall(text) if len(token) > 1]
    return list(dict.fromkeys(tokens))


def _query_terms(query: str) -> tuple[list[str], list[str]]:
    cjk = _cjk_ngrams(query, min_size=2, max_size=5)
    if len(cjk) > 64:
        cjk = cjk[:64]
    latin = _latin_keywords(query)
    if len(latin) > 32:
        latin = latin[:32]
    return cjk, latin


def _extract_source_hint(query: str) -> str | None:
    match = _ATTACHMENT_RE.search(query)
    if match:
        return f"附件{match.group(1)}"
    match = _ATTACHMENT_CN_RE.search(query)
    if match:
        number = _CN_NUM_MAP.get(match.group(1))
        if number:
            return f"附件{number}"
    return None


def _filter_docs_by_source_hint(
    docs: list[Document], source_hint: str | None
) -> list[Document]:
    if not source_hint:
        return docs
    filtered: list[Document] = []
    for doc in docs:
        source = doc.metadata.get("source") or ""
        if source_hint in Path(source).name:
            filtered.append(doc)
    return filtered or docs


def _keyword_score(
    text: str, cjk_keywords: list[str], latin_keywords: list[str]
) -> int:
    score = 0
    for kw in cjk_keywords:
        if kw in text:
            score += len(kw)
    if latin_keywords:
        lowered = text.lower()
        for kw in latin_keywords:
            if kw in lowered:
                score += len(kw)
    return score


def _docs_have_keyword_hits(docs: list[Document], query: str) -> bool:
    cjk, latin = _query_terms(query)
    if not cjk and not latin:
        return True
    for doc in docs:
        if _keyword_score(doc.page_content, cjk, latin):
            return True
    return False


def _keyword_fallback(
    docs: list[Document], query: str, limit: int = 3
) -> list[Document]:
    cjk, latin = _query_terms(query)
    if not cjk and not latin:
        return []
    scored = []
    for doc in docs:
        score = _keyword_score(doc.page_content, cjk, latin)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda item: (-item[0], item[1].metadata.get("page") or 0))
    return [doc for _, doc in scored[:limit]]
