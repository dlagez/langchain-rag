from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _cjk_ngrams(text: str, min_size: int = 2, max_size: int = 5) -> list[str]:
    ngrams: list[str] = []
    for chunk in _CJK_RE.findall(text):
        limit = min(max_size, len(chunk))
        for size in range(min_size, limit + 1):
            for idx in range(len(chunk) - size + 1):
                ngrams.append(chunk[idx : idx + size])
    seen = set()
    deduped: list[str] = []
    for term in ngrams:
        if term in seen:
            continue
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


def _keyword_score(text: str, cjk_keywords: list[str], latin_keywords: list[str]) -> int:
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
