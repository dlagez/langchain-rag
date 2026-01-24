from __future__ import annotations

import re
from datetime import datetime, timedelta


_DATE_TOKEN = "DATE"


def _append_date_tokens(text: str) -> str:
    """在文本末尾追加标准化日期 token，便于检索命中日期/月份。"""
    if not text:
        return text
    tokens: set[str] = set()

    def _add_date(year: int, month: int, day: int | None = None) -> None:
        if year < 1900 or year > 2100:
            return
        if month < 1 or month > 12:
            return
        tokens.add(f"{_DATE_TOKEN}={year:04d}-{month:02d}")
        if day is not None and 1 <= day <= 31:
            tokens.add(f"{_DATE_TOKEN}={year:04d}-{month:02d}-{day:02d}")

    def _year2(y: int) -> int:
        return 2000 + y

    # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
    for match in re.finditer(r"(20\d{2})[./-](\d{1,2})[./-](\d{1,2})", text):
        _add_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    # YYYY-MM or YYYY/MM or YYYY.MM
    for match in re.finditer(r"(20\d{2})[./-](\d{1,2})", text):
        _add_date(int(match.group(1)), int(match.group(2)), None)

    # YYYY年M月D日
    for match in re.finditer(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", text):
        _add_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    # YYYY年M月
    for match in re.finditer(r"(20\d{2})年(\d{1,2})月", text):
        _add_date(int(match.group(1)), int(match.group(2)), None)

    # YY年M月
    for match in re.finditer(r"(\d{2})年(\d{1,2})月", text):
        _add_date(_year2(int(match.group(1))), int(match.group(2)), None)

    # YY/MM/DD or YY-M-D
    for match in re.finditer(r"\b(\d{2})[/-](\d{1,2})[/-](\d{1,2})\b", text):
        _add_date(_year2(int(match.group(1))), int(match.group(2)), int(match.group(3)))

    # YYYYMMDD
    for match in re.finditer(r"\b(20\d{2})(\d{2})(\d{2})\b", text):
        _add_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))

    # YYYYMM
    for match in re.finditer(r"\b(20\d{2})(\d{2})\b", text):
        _add_date(int(match.group(1)), int(match.group(2)), None)

    # Excel serial date (5 digits or more)
    for match in re.finditer(r"\b(\d{5})(?:\.\d+)?\b", text):
        serial = int(match.group(1))
        if 30000 <= serial <= 60000:
            base = datetime(1899, 12, 30)
            dt = base + timedelta(days=serial)
            _add_date(dt.year, dt.month, dt.day)

    if not tokens:
        return text

    suffix = " ".join(f"[{token}]" for token in sorted(tokens))
    return f"{text} {suffix}".strip()
