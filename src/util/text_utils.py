from __future__ import annotations


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _garbled_score(text: str) -> int:
    score = 0
    for ch in text:
        code = ord(ch)
        if ch == "\ufffd":
            score += 2
        elif 0x00C0 <= code <= 0x00FF:
            score += 1
    return score


def _cjk_count(text: str) -> int:
    return sum(1 for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF)


def _looks_garbled(text: str) -> bool:
    if not text:
        return False
    garbled = _garbled_score(text)
    cjk = _cjk_count(text)
    if garbled < 20:
        return False
    return cjk == 0 or garbled > cjk * 2


def _maybe_repair_mojibake(text: str) -> str:
    if not _looks_garbled(text):
        return text
    original_garbled = _garbled_score(text)
    original_cjk = _cjk_count(text)
    for encoding in ("utf-8", "gb18030"):
        try:
            repaired = text.encode("latin1").decode(encoding)
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        if not repaired:
            continue
        if _garbled_score(repaired) < original_garbled and _cjk_count(
            repaired
        ) >= original_cjk:
            return repaired
    return text


def _join_pages(pages: list[str]) -> str:
    chunks = []
    for idx, page_text in enumerate(pages, start=1):
        chunks.append(f"=== Page {idx} ===")
        chunks.append(page_text.strip())
    return "\n\n".join(chunks).strip()


def _split_pages(text: str, use_page_markers: bool = False) -> list[str]:
    if not use_page_markers or "=== Page " not in text:
        return [text.strip()] if text.strip() else []
    parts = []
    for block in text.split("=== Page "):
        block = block.strip()
        if not block:
            continue
        _, _, content = block.partition("===")
        parts.append(content.strip())
    return parts


def _ensure_page_markers(text: str) -> str:
    if not text:
        return text
    if "=== Page " in text:
        return text
    return f"=== Page 1 ===\n\n{text}"
