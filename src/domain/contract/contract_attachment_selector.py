from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from langchain_core.documents import Document


_NAME_STRONG_TERMS = (
    "建设工程施工合同",
    "施工合同",
    "合同协议书",
    "合同条款",
    "合同"
)
_NAME_EXCLUDE_TERMS = (
    "核对",
    "对照",
    "清单",
    "评审",
    "审查",
    "检查项",
    "问题",
    "纪要",
    "说明",
    "申请",
    "审批表",
    "协议"
)
_HEAD_FIELDS = (
    "合同编号",
    "签订地点",
    "签订日期",
    "项目名称",
)
_TAIL_SIGN_TERMS = (
    "法定代表人",
    "授权代表",
    "签字",
    "签章",
    "盖章",
    "公章",
    "开户银行",
    "账号",
)
_PARTY_A_TERMS = ("甲方", "发包人")
_PARTY_B_TERMS = ("乙方", "承包人")
_CLAUSE_RE = re.compile(r"第\s*[一二三四五六七八九十百零0-9]+\s*条")


@dataclass(frozen=True)
class _ContractCandidate:
    name: str
    head_text: str
    tail_text: str

# 把所有附件按“文件名”分组，抽取每个文件的“头部/尾部”文本，然后按一组合同特征打分
# 文件名包含“合同”、头部出现甲乙方/合同字段、条款编号、尾部签章等；
# 文件名/内容命中排除词会扣分或直接剔除）。
# 没有最低分阈值：只要不被排除，就会进候选，即使得分是 0 或负数也会被选为 top_k（默认 1）。
class ContractAttachmentSelector:
    def __init__(
        self,
        *,
        head_min: int = 3000,
        head_max: int = 8000,
        tail_min: int = 3000,
        tail_max: int = 8000,
        party_window: int = 50,
    ) -> None:
        self._head_min = head_min
        self._head_max = head_max
        self._tail_min = tail_min
        self._tail_max = tail_max
        self._party_window = party_window

    def select_contract_names(
        self,
        docs: list[Document],
        *,
        top_k: int = 1,
    ) -> list[str]:
        names, _ = self.select_contract_names_with_report(
            docs, top_k=top_k
        )
        return names

    def select_contract_names_with_report(
        self,
        docs: list[Document],
        *,
        top_k: int = 1,
    ) -> tuple[list[str], list[dict[str, object]]]:
        candidates = self._build_candidates(docs)
        scored: list[tuple[int, str]] = []
        report: list[dict[str, object]] = []
        for candidate in candidates:
            score, detail = self._score_candidate_detail(candidate)
            report.append(detail)
            if score is None:
                continue
            scored.append((score, candidate.name))
        if not scored or top_k <= 0:
            return [], report
        scored.sort(key=lambda item: (-item[0], len(item[1])))
        return [name for _, name in scored[:top_k]], report

    def _build_candidates(self, docs: list[Document]) -> list[_ContractCandidate]:
        grouped: dict[str, list[tuple[tuple[int, int, int], Document]]] = {}
        for idx, doc in enumerate(docs):
            metadata = doc.metadata or {}
            if metadata.get("source_type") != "attachment":
                continue
            source = metadata.get("source")
            if not source:
                continue
            name = Path(source).name
            sort_key = self._doc_sort_key(metadata, idx)
            grouped.setdefault(name, []).append((sort_key, doc))
        candidates: list[_ContractCandidate] = []
        for name, items in grouped.items():
            items.sort(key=lambda item: item[0])
            ordered_docs = [doc for _, doc in items]
            head_text, tail_text = self._extract_head_tail(ordered_docs)
            candidates.append(
                _ContractCandidate(name=name, head_text=head_text, tail_text=tail_text)
            )
        return candidates

    @staticmethod
    def _doc_sort_key(metadata: dict, fallback: int) -> tuple[int, int, int]:
        page = metadata.get("page")
        if isinstance(page, int):
            return (0, page, fallback)
        return (1, fallback, 0)

    def _extract_head_tail(self, docs: list[Document]) -> tuple[str, str]:
        total_len = self._total_length(docs)
        head_limit = self._effective_limit(
            total_len, self._head_min, self._head_max
        )
        tail_limit = self._effective_limit(
            total_len, self._tail_min, self._tail_max
        )
        head_text = self._collect_head(docs, head_limit)
        tail_text = self._collect_tail(docs, tail_limit)
        return head_text, tail_text

    @staticmethod
    def _total_length(docs: list[Document]) -> int:
        total = 0
        first = True
        for doc in docs:
            text = doc.page_content or ""
            if not text:
                continue
            if not first:
                total += 1
            first = False
            total += len(text)
        return total

    @staticmethod
    def _effective_limit(total_len: int, min_len: int, max_len: int) -> int:
        if total_len <= min_len:
            return total_len
        return min(max_len, total_len)

    @staticmethod
    def _collect_head(docs: list[Document], limit: int) -> str:
        if limit <= 0:
            return ""
        parts: list[str] = []
        size = 0
        for doc in docs:
            text = doc.page_content or ""
            if not text:
                continue
            prefix = "\n" if parts else ""
            chunk = prefix + text
            remaining = limit - size
            if remaining <= 0:
                break
            parts.append(chunk[:remaining])
            size += min(len(chunk), remaining)
            if size >= limit:
                break
        return "".join(parts)

    @staticmethod
    def _collect_tail(docs: list[Document], limit: int) -> str:
        if limit <= 0:
            return ""
        tail_text = ""
        for doc in docs:
            text = doc.page_content or ""
            if not text:
                continue
            if tail_text:
                text = "\n" + text
            if len(text) >= limit:
                tail_text = text[-limit:]
            else:
                tail_text = (tail_text + text)[-limit:]
        return tail_text

    def _score_candidate(self, candidate: _ContractCandidate) -> int | None:
        name = candidate.name
        if self._contains_any(name, _NAME_EXCLUDE_TERMS):
            return None
        head = candidate.head_text
        tail = candidate.tail_text
        combined = "\n".join(item for item in (head, tail) if item)
        score = 0

        strong_name_hits = self._count_hits(name, _NAME_STRONG_TERMS)
        if strong_name_hits:
            score += min(strong_name_hits, 2) * 6
        elif "合同" in name:
            score += 2

        if self._has_party_pair(head):
            score += 10

        score += self._count_hits(head, _HEAD_FIELDS) * 3

        clause_hits = len(_CLAUSE_RE.findall(combined))
        if clause_hits >= 3:
            score += 4
        if clause_hits >= 6:
            score += 3

        tail_hits = self._count_hits(tail, _TAIL_SIGN_TERMS)
        if tail_hits:
            score += min(tail_hits, 4) * 2
            if tail_hits >= 3:
                score += 4

        penalty_hits = self._count_hits(combined, _NAME_EXCLUDE_TERMS)
        if penalty_hits:
            score -= penalty_hits * 5

        return score

    def _score_candidate_detail(
        self, candidate: _ContractCandidate
    ) -> tuple[int | None, dict[str, object]]:
        name = candidate.name
        name_exclude_terms = self._match_terms(name, _NAME_EXCLUDE_TERMS)
        head = candidate.head_text
        tail = candidate.tail_text
        combined = "\n".join(item for item in (head, tail) if item)
        detail: dict[str, object] = {
            "name": name,
            "score": None,
            "excluded_by_name": bool(name_exclude_terms),
            "name_exclude_terms": name_exclude_terms,
            "head_len": len(head),
            "tail_len": len(tail),
        }
        if name_exclude_terms:
            return None, detail

        score = 0
        strong_name_terms = self._match_terms(name, _NAME_STRONG_TERMS)
        strong_name_hits = len(strong_name_terms)
        if strong_name_hits:
            score += min(strong_name_hits, 2) * 6
        elif "鍚堝悓" in name:
            score += 2

        has_party_pair = self._has_party_pair(head)
        if has_party_pair:
            score += 10

        head_terms = self._match_terms(head, _HEAD_FIELDS)
        score += len(head_terms) * 3

        clause_hits = len(_CLAUSE_RE.findall(combined))
        clause_bonus = 0
        if clause_hits >= 3:
            clause_bonus += 4
        if clause_hits >= 6:
            clause_bonus += 3
        if clause_hits >= 10:
            clause_bonus += 6
        if clause_hits >= 20:
            clause_bonus += 6
        if clause_hits >= 40:
            clause_bonus += 6
        score += clause_bonus

        tail_terms = self._match_terms(tail, _TAIL_SIGN_TERMS)
        if tail_terms:
            score += min(len(tail_terms), 4) * 2
            if len(tail_terms) >= 3:
                score += 4

        penalty_terms = self._match_terms(combined, _NAME_EXCLUDE_TERMS)
        penalty_weight = 5
        if clause_hits >= 10 or (strong_name_hits and clause_hits >= 3):
            penalty_weight = 1
        elif clause_hits >= 3:
            penalty_weight = 2
        if penalty_terms:
            score -= len(penalty_terms) * penalty_weight

        detail.update(
            {
                "score": score,
                "strong_name_terms": strong_name_terms,
                "name_contains_contract": "鍚堝悓" in name,
                "has_party_pair": has_party_pair,
                "head_terms": head_terms,
                "clause_hits": clause_hits,
                "clause_bonus": clause_bonus,
                "tail_terms": tail_terms,
                "penalty_terms": penalty_terms,
                "penalty_weight": penalty_weight,
            }
        )
        return score, detail

    def _has_party_pair(self, text: str) -> bool:
        if not text:
            return False
        positions_a = self._find_positions(text, _PARTY_A_TERMS)
        positions_b = self._find_positions(text, _PARTY_B_TERMS)
        for pos_a in positions_a:
            for pos_b in positions_b:
                if abs(pos_a - pos_b) <= self._party_window:
                    return True
        return False

    @staticmethod
    def _find_positions(text: str, terms: tuple[str, ...]) -> list[int]:
        positions: list[int] = []
        for term in terms:
            start = 0
            while True:
                idx = text.find(term, start)
                if idx == -1:
                    break
                positions.append(idx)
                start = idx + len(term)
        return positions

    @staticmethod
    def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _count_hits(text: str, terms: tuple[str, ...]) -> int:
        return sum(1 for term in terms if term in text)

    @staticmethod
    def _match_terms(text: str, terms: tuple[str, ...]) -> list[str]:
        if not text:
            return []
        return [term for term in terms if term in text]
