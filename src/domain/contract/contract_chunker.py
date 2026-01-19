from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from langchain_core.documents import Document

# 合同领域的分块器，专门把 OCR/抽取后的合同相关文档切成更适合检索/向量化的片段。核心作用：
# 根据文件名/内容判断文档类型：合同正文、检查清单、其它
# 按中文合同结构（章/条/节）、表格行、句子长度等规则切分
# 对签署页、清单项等做特殊处理
# 生成带有元数据的 Document（例如 doc_type_hint、page、has_signature 等）

_CONTRACT_NAME_HINTS = (
    "建设工程施工合同",
    "施工合同",
    "合同协议书",
    "合同条款",
    "合同",
)
_CHECKLIST_NAME_HINTS = (
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
)
_SIGNATURE_TERMS = (
    "签字",
    "签章",
    "盖章",
    "公章",
    "法定代表人",
    "授权代表",
    "开户银行",
    "账号",
)
_CHECKBOX_TERMS = ("是否", "√", "×", "☑", "☒", "□", "■", "✔", "✘", "勾选")
_CONTRACT_TITLE_HINTS = (
    "通用条款",
    "专用条款",
    "合同价款",
    "结算",
    "争议解决",
    "违约责任",
)

_CHAPTER_RE = re.compile(r"^\s*第[一二三四五六七八九十百零0-9]+[章节]")
_CLAUSE_RE = re.compile(r"^\s*第[一二三四五六七八九十百零0-9]+条")
_SECTION_RE = re.compile(r"^\s*[一二三四五六七八九十]+、")
_SUBSECTION_RE = re.compile(r"^\s*[（(][一二三四五六七八九十]+[)）]")
_NUM_TITLE_RE = re.compile(r"^\s*\d+(\.\d+)+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；;])")


@dataclass(frozen=True)
class _Segment:
    text: str
    page: int | None
    order: int


class ContractChunker:
    def __init__(
        self,
        *,
        contract_min: int = 400,
        contract_max: int = 900,
        overlap: int = 120,
        checklist_min: int = 200,
        checklist_max: int = 600,
        checklist_min_lines: int = 5,
        checklist_max_lines: int = 15,
    ) -> None:
        self._contract_min = contract_min
        self._contract_max = contract_max
        self._overlap = overlap
        self._checklist_min = checklist_min
        self._checklist_max = checklist_max
        self._checklist_min_lines = checklist_min_lines
        self._checklist_max_lines = checklist_max_lines

    def chunk_documents(self, docs: list[Document]) -> list[Document]:
        grouped: dict[str, list[_Segment]] = {}
        base_meta: dict[str, dict] = {}
        output: list[Document] = []
        for order, doc in enumerate(docs):
            metadata = doc.metadata or {}
            source = metadata.get("source")
            if not source:
                continue
            source_type = metadata.get("source_type")
            if source_type == "form":
                if not (doc.page_content or "").strip():
                    continue
                meta = dict(metadata)
                meta["doc_type_hint"] = "form"
                if meta.get("page") is None:
                    meta.pop("page", None)
                output.append(
                    Document(page_content=doc.page_content.strip(), metadata=meta)
                )
                continue
            if source not in base_meta:
                base_meta[source] = dict(metadata)
            grouped.setdefault(source, []).append(
                _Segment(
                    text=doc.page_content or "",
                    page=metadata.get("page"),
                    order=order,
                )
            )

        for source, segments in grouped.items():
            metadata = dict(base_meta.get(source, {}))

            name = Path(source).name
            ordered = sorted(
                segments,
                key=lambda seg: (0, seg.page, seg.order)
                if isinstance(seg.page, int)
                else (1, seg.order, 0),
            )
            doc_type = self._infer_doc_type(name, ordered)

            if doc_type == "checklist":
                output.extend(self._chunk_checklist(name, ordered, metadata))
            elif doc_type == "contract":
                output.extend(self._chunk_contract(name, ordered, metadata))
            else:
                output.extend(self._chunk_generic(name, ordered, metadata))

        return output

    def _infer_doc_type(self, name: str, segments: list[_Segment]) -> str:
        if self._contains_any(name, _CHECKLIST_NAME_HINTS):
            return "checklist"
        text = "\n".join(seg.text for seg in segments if seg.text)
        if self._contains_any(name, _CONTRACT_NAME_HINTS):
            return "contract"
        if self._has_contract_structure(text):
            return "contract"
        if self._looks_like_checklist(text):
            return "checklist"
        return "other"

    def _chunk_contract(
        self, name: str, segments: list[_Segment], metadata: dict
    ) -> list[Document]:
        signature_chunks: list[Document] = []
        body_segments: list[_Segment] = []
        tail_threshold = max(0, len(segments) - 3)
        for idx, segment in enumerate(segments):
            text = segment.text.strip()
            if not text:
                continue
            if idx >= tail_threshold and self._contains_any(text, _SIGNATURE_TERMS):
                signature_chunks.extend(
                    self._build_chunks(
                        text,
                        metadata,
                        doc_type="contract",
                        page=segment.page,
                        max_len=self._contract_max,
                        min_len=self._contract_min,
                        overlap=self._overlap,
                        has_signature=True,
                    )
                )
            else:
                body_segments.append(segment)

        body_text = "\n".join(seg.text for seg in body_segments if seg.text.strip())
        contract_chunks = self._split_contract_text(body_text)
        docs = [
            Document(
                page_content=chunk,
                metadata=self._with_metadata(
                    metadata,
                    doc_type="contract",
                    page=None,
                    has_signature=False,
                ),
            )
            for chunk in contract_chunks
        ]
        return docs + signature_chunks

    def _chunk_checklist(
        self, name: str, segments: list[_Segment], metadata: dict
    ) -> list[Document]:
        text = "\n".join(seg.text for seg in segments if seg.text.strip())
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        chunks = self._split_checklist_lines(lines)
        has_checkbox = self._contains_any(text, _CHECKBOX_TERMS)
        output = []
        for chunk in chunks:
            output.append(
                Document(
                    page_content=chunk,
                    metadata=self._with_metadata(
                        metadata,
                        doc_type="checklist",
                    page=None,
                    has_signature=False,
                    has_checkbox=has_checkbox,
                ),
            )
            )
        return output

    def _chunk_generic(
        self, name: str, segments: list[_Segment], metadata: dict
    ) -> list[Document]:
        text = "\n".join(seg.text for seg in segments if seg.text.strip())
        chunks = self._split_generic_text(text)
        return [
            Document(
                page_content=chunk,
                metadata=self._with_metadata(
                    metadata,
                    doc_type="other",
                    page=None,
                    has_signature=False,
                ),
            )
            for chunk in chunks
        ]

    def _split_contract_text(self, text: str) -> list[str]:
        if not text.strip():
            return []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections = self._split_by_structure(lines, contract_mode=True)
        chunks: list[str] = []
        for section in sections:
            if len(section) <= self._contract_max:
                chunks.append(section)
            else:
                chunks.extend(
                    self._split_by_sentences(
                        section,
                        max_len=self._contract_max,
                        min_len=self._contract_min,
                        overlap=self._overlap,
                    )
                )
        return self._merge_short_chunks(
            chunks, min_len=self._contract_min, max_len=self._contract_max
        )

    def _split_generic_text(self, text: str) -> list[str]:
        if not text.strip():
            return []
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        sections = self._split_by_structure(lines, contract_mode=False)
        chunks: list[str] = []
        for section in sections:
            if len(section) <= self._contract_max:
                chunks.append(section)
            else:
                chunks.extend(
                    self._split_by_sentences(
                        section,
                        max_len=self._contract_max,
                        min_len=self._contract_min,
                        overlap=self._overlap,
                    )
                )
        return self._merge_short_chunks(
            chunks, min_len=self._contract_min, max_len=self._contract_max
        )

    def _split_by_structure(
        self, lines: list[str], *, contract_mode: bool
    ) -> list[str]:
        sections: list[str] = []
        current: list[str] = []
        idx = 0
        while idx < len(lines):
            line = lines[idx]
            if self._is_table_line(line):
                if current:
                    sections.append("\n".join(current).strip())
                    current = []
                table_lines = [line]
                idx += 1
                while idx < len(lines) and self._is_table_line(lines[idx]):
                    table_lines.append(lines[idx])
                    idx += 1
                sections.append("\n".join(table_lines).strip())
                continue
            if self._is_heading(line, contract_mode=contract_mode):
                if current:
                    sections.append("\n".join(current).strip())
                    current = []
            current.append(line)
            idx += 1
        if current:
            sections.append("\n".join(current).strip())
        return [section for section in sections if section]

    def _split_checklist_lines(self, lines: list[str]) -> list[str]:
        chunks: list[str] = []
        current: list[str] = []
        line_count = 0
        char_count = 0
        for line in lines:
            if self._is_item_start(line) and current and line_count >= self._checklist_min_lines:
                chunks.append("\n".join(current).strip())
                current = []
                line_count = 0
                char_count = 0
            current.append(line)
            line_count += 1
            char_count += len(line)
            if (
                char_count >= self._checklist_max
                or line_count >= self._checklist_max_lines
            ):
                chunks.append("\n".join(current).strip())
                current = []
                line_count = 0
                char_count = 0
        if current:
            chunks.append("\n".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    def _split_by_sentences(
        self, text: str, *, max_len: int, min_len: int, overlap: int
    ) -> list[str]:
        sentences = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > max_len:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._split_by_length(sentence, max_len, overlap))
                continue
            if len(current) + len(sentence) > max_len:
                if current:
                    chunks.append(current)
                if overlap > 0 and current:
                    current = current[-overlap:] + sentence
                else:
                    current = sentence
            else:
                current += sentence
        if current:
            chunks.append(current)
        return self._merge_short_chunks(chunks, min_len=min_len, max_len=max_len)

    def _split_by_length(self, text: str, max_len: int, overlap: int) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= max_len:
            return [text]
        step = max_len - overlap if max_len > overlap else max_len
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_len, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = end - overlap if overlap > 0 else end
        return chunks

    def _build_chunks(
        self,
        text: str,
        metadata: dict,
        *,
        doc_type: str,
        page: int | None,
        max_len: int,
        min_len: int,
        overlap: int,
        has_signature: bool,
    ) -> list[Document]:
        chunks = self._split_by_sentences(
            text, max_len=max_len, min_len=min_len, overlap=overlap
        )
        output = []
        for chunk in chunks:
            output.append(
                Document(
                    page_content=chunk,
                    metadata=self._with_metadata(
                        metadata,
                        doc_type=doc_type,
                        page=page,
                        has_signature=has_signature,
                    ),
                )
            )
        return output

    def _with_metadata(
        self,
        metadata: dict,
        *,
        doc_type: str,
        page: int | None,
        has_signature: bool,
        has_checkbox: bool | None = None,
    ) -> dict:
        meta = dict(metadata)
        meta["doc_type_hint"] = doc_type
        if has_signature:
            meta["signature_page"] = True
        if has_checkbox is not None:
            meta["has_checkbox_like"] = bool(has_checkbox)
        if page is None:
            meta.pop("page", None)
        else:
            meta["page"] = page
        return meta

    def _has_contract_structure(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        for line in lines[:200]:
            if _CHAPTER_RE.match(line) or _CLAUSE_RE.match(line):
                return True
            if _SECTION_RE.match(line) or _SUBSECTION_RE.match(line):
                return True
            if any(term in line for term in _CONTRACT_TITLE_HINTS):
                return True
        return False

    def _looks_like_checklist(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        item_hits = sum(1 for line in lines if self._is_item_start(line))
        ratio = item_hits / max(1, len(lines))
        return item_hits >= 3 and ratio >= 0.08

    @staticmethod
    def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    @staticmethod
    def _is_table_line(line: str) -> bool:
        if "\t" in line:
            return True
        if len(line) >= 20 and re.search(r"\s{2,}", line):
            return True
        return False

    @staticmethod
    def _is_item_start(line: str) -> bool:
        return bool(
            re.match(r"^\s*\d+[\.、]", line)
            or re.match(r"^\s*[（(]?\d+[)）]", line)
            or re.match(r"^[一二三四五六七八九十]+[、.]", line)
        )

    def _is_heading(self, line: str, *, contract_mode: bool) -> bool:
        if contract_mode:
            if _CHAPTER_RE.match(line) or _CLAUSE_RE.match(line):
                return True
            if _SECTION_RE.match(line) or _SUBSECTION_RE.match(line):
                return True
            if any(term in line for term in _CONTRACT_TITLE_HINTS) and len(line) <= 24:
                return True
        else:
            if _NUM_TITLE_RE.match(line) or _SECTION_RE.match(line):
                return True
            if line.endswith((":", "：")) and len(line) <= 30:
                return True
        return False

    @staticmethod
    def _merge_short_chunks(
        chunks: list[str], *, min_len: int, max_len: int
    ) -> list[str]:
        merged: list[str] = []
        for chunk in chunks:
            if not chunk:
                continue
            if merged and len(chunk) < min_len and len(merged[-1]) + len(chunk) <= max_len:
                merged[-1] = merged[-1].rstrip() + "\n" + chunk.lstrip()
            else:
                merged.append(chunk)
        return merged
