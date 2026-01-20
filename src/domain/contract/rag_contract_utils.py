from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re

import numpy as np
from langchain_core.documents import Document
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from .contract_attachment_selector import ContractAttachmentSelector
from .contract_chunker import ContractChunker
from util.document_utils import _doc_signature, _format_source
from util.keyword_utils import _extract_source_hint

# 合同 RAG 的领域工具集，把原始文档整理、过滤、分块、召回标签等“合同业务逻辑”集中在这里。主要职责：
# 来源范围控制：SourceScope + _infer_source_scope/_filter_docs_by_scope/_filter_docs_and_scores_by_scope，按“表单/附件/文件名/关键词”等限定检索范围
# 召回结果标注与合并：_tag_retriever/_merge_docs_with_retriever/_unique_sources_with_retriever，给 doc 打上召回来源、合并去重
# 附件/表单结构化：_build_contract_documents，把原始文本转为结构化 Document（字段、段落、页码、source_type 等）
# 正则兜底检索：_regex_attachment_fallback，基于关键词正则从附件文本里兜底召回
# 索引分块与ID：_chunk_documents_for_index、_assign_chunk_ids 用合同分块策略切片并打 chunk_id
# 检索过滤：_source_type_filter 生成 Qdrant 的过滤条件

def _source_type_filter(source_type: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="metadata.source_type", match=MatchValue(value=source_type)
            )
        ]
    )


def _normalize_retriever_labels(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)]


def _tag_retriever(docs: list[Document], label: str) -> list[Document]:
    for doc in docs:
        if doc.metadata is None or not isinstance(doc.metadata, dict):
            doc.metadata = {}
        labels = _normalize_retriever_labels(doc.metadata.get("_retriever"))
        if label not in labels:
            labels.append(label)
        doc.metadata["_retriever"] = labels
    return docs


def _merge_retriever_labels(target: Document, incoming: Document) -> None:
    if target.metadata is None or not isinstance(target.metadata, dict):
        target.metadata = {}
    incoming_labels = _normalize_retriever_labels(
        incoming.metadata.get("_retriever") if incoming.metadata else None
    )
    if not incoming_labels:
        return
    target_labels = _normalize_retriever_labels(
        target.metadata.get("_retriever")
    )
    merged = list(dict.fromkeys(target_labels + incoming_labels))
    target.metadata["_retriever"] = merged


def _merge_docs_with_retriever(
    primary: list[Document], secondary: list[Document]
) -> list[Document]:
    seen: dict[tuple, Document] = {}
    merged: list[Document] = []
    for doc in primary + secondary:
        key = _doc_signature(doc)
        existing = seen.get(key)
        if existing is None:
            merged.append(doc)
            seen[key] = doc
        else:
            _merge_retriever_labels(existing, doc)
    return merged


def _format_retriever(doc: Document) -> str:
    labels = _normalize_retriever_labels(
        doc.metadata.get("_retriever") if doc.metadata else None
    )
    if not labels:
        return "unknown"
    return "+".join(sorted(set(labels)))


def _unique_sources_with_retriever(docs: list[Document]) -> list[str]:
    seen: dict[str, set[str]] = {}
    order: list[str] = []
    for doc in docs:
        label = _format_source(doc)
        retriever = _format_retriever(doc)
        if label not in seen:
            seen[label] = set()
            order.append(label)
        seen[label].add(retriever)
    output: list[str] = []
    for label in order:
        retrievers = "+".join(sorted(seen[label]))
        output.append(f"{label} ({retrievers})")
    return output


_KEYWORD_PUNCTUATION = " ,.;:?!、。，；：？！"
_PERCENT_RE = re.compile(r"^\d{1,2}[%％]$")


def _regex_keywords_from_query(keyword_query: str) -> list[str]:
    if not keyword_query:
        return []
    tokens: list[str] = []
    for raw in keyword_query.split():
        token = raw.strip(_KEYWORD_PUNCTUATION)
        if not token:
            continue
        if "?" in raw or "？" in raw:
            if len(token) > 8:
                continue
            token = token.replace("?", "").replace("？", "")
        if len(token) < 2 and not re.search(r"\d", token):
            continue
        tokens.append(token)
    return list(dict.fromkeys(tokens))


def _regex_keyword_patterns(tokens: list[str]) -> list[str]:
    patterns: list[str] = []
    for token in tokens:
        if _PERCENT_RE.fullmatch(token):
            number = re.search(r"\d{1,2}", token)
            if number:
                patterns.append(f"{number.group(0)}\\s*[%％]")
            continue
        patterns.append(re.escape(token))
    return patterns


def _regex_attachment_fallback(
    docs: list[Document],
    keyword_query: str,
    limit: int,
    source_scope: "SourceScope | None",
) -> list[Document]:
    tokens = _regex_keywords_from_query(keyword_query)
    if not tokens:
        return []
    patterns = _regex_keyword_patterns(tokens)
    if not patterns:
        return []
    pattern = re.compile("|".join(patterns))
    scored: list[tuple[int, Document]] = []
    for doc in docs:
        metadata = doc.metadata or {}
        source = metadata.get("source") or ""
        name = Path(source).name
        source_type = _normalize_source_type(metadata.get("source_type"))
        if source_type is None:
            source_type = _infer_source_category(source)
        if source_type != "attachment":
            continue
        if source_scope and source_scope.is_active():
            if not name or not _doc_in_scope(
                name, source_scope, source_type, metadata, text
            ):
                continue
        text = doc.page_content
        if not text:
            continue
        matches = pattern.findall(text)
        if matches:
            scored.append((len(matches), doc))
    scored.sort(key=lambda item: (-item[0], item[1].metadata.get("page") or 0))
    return [doc for _, doc in scored[:limit]]



_FORM_PREFIX = "表单"
_ATTACHMENT_PREFIX = "附件"
_FORM_DIR_NAMES = {"表单", "form", "forms"}
_ATTACHMENT_DIR_NAMES = {"附件", "attachment", "attachments"}
_FORM_FIELD_MAX_LEN = 28
_FORM_FIELD_HINTS = (
    "编号",
    "流程",
    "项目",
    "合同",
    "税率",
    "金额",
    "类型",
    "意见",
    "备注",
)
_PROCESS_ID_FIELD_HINTS = (
    "流程",
    "实例",
    "编号",
    "单号",
    "表单编号",
    "process",
    "instance",
    "id",
)
_VALUE_HINTS = ("下载", "查看")
_VALUE_FILE_RE = re.compile(
    r"\.(pdf|docx|doc|xlsx|xls|csv|png|jpg|jpeg|bmp|tif|tiff)\b",
    re.IGNORECASE,
)
_VALUE_NUMERIC_RE = re.compile(r"^[\d\s,\.\-/%()％（）]+$")


@dataclass(frozen=True)
class SourceScope:
    names: tuple[str, ...] = ()
    prefix: str | None = None
    include_terms: tuple[str, ...] = ()
    exclude_terms: tuple[str, ...] = ()

    def is_active(self) -> bool:
        return bool(self.names or self.prefix or self.include_terms or self.exclude_terms)


def _collect_source_names(raw_docs: list[Document]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for doc in raw_docs:
        source = doc.metadata.get("source") if doc.metadata else None
        if not source:
            continue
        name = Path(source).name
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return sorted(names)


def _match_source_names(scope: SourceScope, source_names: list[str]) -> list[str]:
    return [name for name in source_names if _doc_in_scope(name, scope)]


def _infer_explicit_sources(prompt: str, source_names: list[str]) -> list[str]:
    matches: list[str] = []
    for name in source_names:
        if name and name in prompt:
            matches.append(name)
            continue
        stem = Path(name).stem
        if stem and stem in prompt:
            matches.append(name)
    if not matches:
        return []
    max_len = max(len(item) for item in matches)
    return [item for item in matches if len(item) == max_len]


def _infer_source_scope(
    prompt: str,
    source_names: list[str],
    raw_docs: list[Document] | None = None,
) -> SourceScope:
    explicit = _infer_explicit_sources(prompt, source_names)
    if explicit:
        return SourceScope(names=tuple(explicit))

    include_terms: list[str] = []
    exclude_terms: list[str] = []
    prefix: str | None = None

    if "表单" in prompt or "审批表" in prompt:
        prefix = "表单"

    if any(term in prompt for term in ("附件", "合同", "招标文件", "通知书", "协议")):
        if prefix is None:
            prefix = "附件"

    if raw_docs is not None and "合同" in prompt and prefix != _FORM_PREFIX:
        contract_name_candidates = [
            name
            for name in source_names
            if "合同" in name
        ]
        if contract_name_candidates:
            return SourceScope(names=tuple(contract_name_candidates))
        selector = ContractAttachmentSelector()
        contract_names = selector.select_contract_names(raw_docs, top_k=1)
        if contract_names:
            return SourceScope(names=tuple(contract_names))

    if "招标文件" in prompt:
        prefix = "附件"
        include_terms = ["招标文件"]
    elif "通知书" in prompt:
        prefix = "附件"
        include_terms = ["通知书"]
    elif "协议" in prompt and "合同" not in prompt:
        prefix = "附件"
        include_terms = ["协议"]
    elif "附件" in prompt and "合同" in prompt:
        prefix = "附件"
        include_terms = ["合同"]
        exclude_terms = ["招标文件"]
    elif "合同" in prompt and prefix == "附件":
        include_terms = ["合同"]
        exclude_terms = ["招标文件"]

    hint = _extract_source_hint(prompt)
    if hint and hint not in include_terms:
        include_terms.append(hint)

    include_terms = list(dict.fromkeys(include_terms))
    exclude_terms = list(dict.fromkeys(exclude_terms))

    return SourceScope(
        prefix=prefix,
        include_terms=tuple(include_terms),
        exclude_terms=tuple(exclude_terms),
    )


def _format_source_scope(scope: SourceScope) -> str:
    parts: list[str] = []
    if scope.names:
        parts.append("files=" + ", ".join(scope.names))
    if scope.prefix:
        parts.append(f"prefix={scope.prefix}")
    if scope.include_terms:
        parts.append("include=" + ", ".join(scope.include_terms))
    if scope.exclude_terms:
        parts.append("exclude=" + ", ".join(scope.exclude_terms))
    return "; ".join(parts)


def _doc_in_scope(
    name: str,
    scope: SourceScope,
    source_type: str | None = None,
    metadata: dict | None = None,
    content: str | None = None,
) -> bool:
    if scope.names and name not in scope.names:
        return False
    if scope.prefix:
        normalized_prefix = _normalize_source_type(scope.prefix)
        normalized_type = _normalize_source_type(source_type)
        if normalized_prefix and normalized_type:
            if normalized_prefix != normalized_type:
                return False
        elif not name.startswith(scope.prefix):
            return False
    if scope.include_terms and not _include_terms_match(
        scope.include_terms, name, metadata, content
    ):
        return False
    if scope.exclude_terms and _exclude_terms_match(
        scope.exclude_terms, name, metadata, content
    ):
        return False
    return True


def _include_terms_match(
    terms: tuple[str, ...],
    name: str,
    metadata: dict | None,
    content: str | None,
) -> bool:
    if not terms:
        return True
    if name and all(term in name for term in terms):
        return True
    if metadata:
        filename = metadata.get("filename")
        if isinstance(filename, str) and all(term in filename for term in terms):
            return True
        doc_type = metadata.get("doc_type_hint") or metadata.get("doc_type")
        if doc_type and len(terms) == 1:
            term = terms[0]
            if term in {"合同", "contract"} and doc_type == "contract":
                return True
            if term in {"清单", "checklist"} and doc_type == "checklist":
                return True
    if content and all(term in content for term in terms):
        return True
    return False


def _exclude_terms_match(
    terms: tuple[str, ...],
    name: str,
    metadata: dict | None,
    content: str | None,
) -> bool:
    if not terms:
        return False
    if name and any(term in name for term in terms):
        return True
    if metadata:
        filename = metadata.get("filename")
        if isinstance(filename, str) and any(term in filename for term in terms):
            return True
    if content and any(term in content for term in terms):
        return True
    return False



def _filter_docs_by_scope(
    docs: list[Document], scope: SourceScope | None
) -> list[Document]:
    if scope is None or not scope.is_active():
        return docs
    filtered: list[Document] = []
    for doc in docs:
        source = doc.metadata.get("source") or ""
        name = Path(source).name
        source_type = _normalize_source_type(doc.metadata.get("source_type"))
        if source_type is None:
            source_type = _infer_source_category(source)
        if name and _doc_in_scope(
            name, scope, source_type, doc.metadata, doc.page_content
        ):
            filtered.append(doc)
    return filtered



def _filter_docs_and_scores_by_scope(
    docs: list[Document],
    scores: np.ndarray,
    scope: SourceScope | None,
) -> tuple[list[Document], np.ndarray]:
    if scope is None or not scope.is_active():
        return docs, scores
    indices = []
    for idx, doc in enumerate(docs):
        source = doc.metadata.get("source") or ""
        name = Path(source).name
        source_type = _normalize_source_type(doc.metadata.get("source_type"))
        if source_type is None:
            source_type = _infer_source_category(source)
        if name and _doc_in_scope(
            name, scope, source_type, doc.metadata, doc.page_content
        ):
            indices.append(idx)
    if not indices:
        return [], scores[:0]
    filtered_docs = [docs[idx] for idx in indices]
    if scores.size == 0:
        return filtered_docs, scores
    try:
        filtered_scores = scores[indices]
    except Exception:
        filtered_scores = np.array([scores[idx] for idx in indices], dtype=scores.dtype)
    return filtered_docs, filtered_scores



def _normalize_lines(text: str) -> list[str]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    return [line.strip() for line in cleaned.split("\n") if line.strip()]


def _looks_like_field_name(line: str) -> bool:
    if not line or len(line) > _FORM_FIELD_MAX_LEN:
        return False
    if _VALUE_FILE_RE.search(line):
        return False
    if any(token in line for token in _VALUE_HINTS):
        return False
    if _VALUE_NUMERIC_RE.fullmatch(line):
        return False
    if re.search(r"\d", line) and len(line) <= 8:
        return False
    return True


def _is_strong_field_name(line: str) -> bool:
    if line.endswith((":", "：")):
        return True
    return any(hint in line for hint in _FORM_FIELD_HINTS)


def _parse_form_fields(text: str) -> list[tuple[str, str]]:
    lines = _normalize_lines(text)
    if not lines:
        return []
    fields: list[tuple[str, str]] = []
    current_field: str | None = None
    current_values: list[str] = []
    for line in lines:
        if current_field is None:
            current_field = line
            continue
        if _looks_like_field_name(line):
            if current_values:
                value = "\n".join(current_values).strip()
                if value:
                    fields.append((current_field, value))
                current_field = line
                current_values = []
                continue
            if _is_strong_field_name(line):
                value = "\n".join(current_values).strip()
                if value:
                    fields.append((current_field, value))
                current_field = line
                current_values = []
                continue
        current_values.append(line)

    if current_field:
        value = "\n".join(current_values).strip()
        if value:
            fields.append((current_field, value))
    return fields


def _split_attachment_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    if len(blocks) > 1:
        return blocks
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [text.strip()] if text.strip() else []


def _doc_id_from_source(source: str, source_dir: Path) -> str:
    try:
        source_path = Path(source).resolve()
        source_dir = source_dir.resolve()
        # Prefer full process-aware path: source/<process_id>/attachment/<file>
        if source_dir.parent.exists():
            return str(source_path.relative_to(source_dir.parent))
        return str(source_path.relative_to(source_dir))
    except Exception:
        return Path(source).name


def _infer_process_id_from_fields(
    form_fields: list[tuple[str, str]],
) -> str | None:
    for field, value in form_fields:
        if any(hint in field for hint in _PROCESS_ID_FIELD_HINTS):
            token = value.strip().split()[0] if value else ""
            if token:
                return token
    return None


def _build_contract_documents(
    raw_docs: list[Document],
    source_dir: Path,
    process_id: str | None,
    created_at: str | None,
) -> tuple[list[Document], str | None]:
    grouped: dict[str, list[Document]] = {}
    for doc in raw_docs:
        source = doc.metadata.get("source") if doc.metadata else None
        if not source:
            continue
        grouped.setdefault(source, []).append(doc)

    form_fields_by_source: dict[str, list[tuple[str, str]]] = {}
    inferred_process_id: str | None = None
    for source, docs in grouped.items():
        source_type = _infer_source_category(source, source_dir)
        if source_type == "form":
            combined = "\n".join(doc.page_content for doc in docs)
            form_fields_by_source[source] = _parse_form_fields(combined)
        if inferred_process_id is None:
            inferred_process_id = _process_id_from_source(source, source_dir)

    if inferred_process_id is None and process_id is None:
        candidates = []
        for fields in form_fields_by_source.values():
            candidate = _infer_process_id_from_fields(fields)
            if candidate:
                candidates.append(candidate)
        if candidates:
            inferred_process_id = candidates[0]
            if len(set(candidates)) > 1:
                logging.warning(
                    "Multiple process_id values detected; using %s.",
                    inferred_process_id,
                )
    if inferred_process_id and process_id and inferred_process_id != process_id:
        logging.warning(
            "Process ID mismatch between folder (%s) and input (%s); using folder value.",
            inferred_process_id,
            process_id,
        )

    output_docs: list[Document] = []
    for source, docs in grouped.items():
        name = Path(source).name
        source_type = _infer_source_category(source, source_dir)
        doc_id = _doc_id_from_source(source, source_dir)
        base_meta: dict[str, object] = {"source": source, "doc_id": doc_id}
        process_id_value = _process_id_from_source(source, source_dir)
        if process_id_value is None:
            process_id_value = inferred_process_id or process_id
        if process_id_value:
            base_meta["process_id"] = process_id_value
        if created_at:
            base_meta["created_at"] = created_at

        if source_type == "form":
            fields = form_fields_by_source.get(source, [])
            if not fields:
                combined = "\n".join(doc.page_content for doc in docs).strip()
                if combined:
                    fields = [("raw_text", combined)]
            for field, value in fields:
                content = f"{field}: {value}".strip()
                metadata = dict(base_meta)
                metadata["source_type"] = "form"
                metadata["field"] = field
                output_docs.append(Document(page_content=content, metadata=metadata))
            continue

        filename = name
        for doc in docs:
            page = doc.metadata.get("page") if doc.metadata else None
            if page is None:
                paragraphs = _split_attachment_paragraphs(doc.page_content)
            else:
                paragraphs = [doc.page_content]

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                metadata = dict(base_meta)
                metadata["source_type"] = "attachment"
                metadata["filename"] = filename
                if page is not None:
                    metadata["page"] = page
                output_docs.append(
                    Document(page_content=paragraph, metadata=metadata)
                )

    return output_docs, inferred_process_id


def _normalize_source_type(value: str | None) -> str | None:
    if not value:
        return None
    raw = value.strip()
    lowered = raw.lower()
    if raw in _FORM_DIR_NAMES or lowered in _FORM_DIR_NAMES:
        return "form"
    if raw in _ATTACHMENT_DIR_NAMES or lowered in _ATTACHMENT_DIR_NAMES:
        return "attachment"
    if lowered in {"form", "attachment"}:
        return lowered
    return None


def _infer_source_category(
    source: str, source_dir: Path | None = None
) -> str | None:
    if source_dir is not None:
        parts = _relative_parts(source, source_dir)
        if len(parts) >= 2:
            category = _normalize_source_type(parts[1])
            if category:
                return category
    try:
        parts = Path(source).parts
    except Exception:
        return None
    for part in reversed(parts[:-1]):
        category = _normalize_source_type(part)
        if category:
            return category
    return None


def _process_id_from_source(source: str, source_dir: Path) -> str | None:
    parts = _relative_parts(source, source_dir)
    if not parts:
        return None
    if len(parts) == 1:
        return None
    if _normalize_source_type(parts[0]):
        return None
    return parts[0]


def _relative_parts(source: str, source_dir: Path) -> tuple[str, ...]:
    try:
        rel = Path(source).resolve().relative_to(source_dir.resolve())
    except Exception:
        return ()
    return rel.parts

def _assign_chunk_ids(docs: list[Document]) -> list[Document]:
    counters: dict[tuple[object, ...], int] = {}
    for doc in docs:
        if doc.metadata is None:
            doc.metadata = {}
        key = (
            doc.metadata.get("doc_id"),
            doc.metadata.get("field"),
            doc.metadata.get("page"),
        )
        counters.setdefault(key, 0)
        doc.metadata["chunk_id"] = counters[key]
        counters[key] += 1
    return docs


def _chunk_documents_for_index(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    contract_min = max(200, int(chunk_size * 0.5))
    contract_max = chunk_size
    overlap = max(0, chunk_overlap)
    checklist_max = min(600, contract_max)
    checklist_min = min(200, checklist_max)
    chunker = ContractChunker(
        contract_min=contract_min,
        contract_max=contract_max,
        overlap=overlap,
        checklist_min=checklist_min,
        checklist_max=checklist_max,
    )
    return chunker.chunk_documents(docs)
