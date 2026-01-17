from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import os
import re
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

from util.util import (
    _build_index_manifest,
    _collection_exists,
    _docs_from_search_results,
    _docs_have_keyword_hits,
    _doc_signature,
    _extract_source_hint,
    _format_source,
    _format_context,
    _get_qdrant_client,
    _hybrid_rerank,
    _keyword_fallback,
    _LazyOCR,
    _load_manifest,
    _manifest_matches,
    _qdrant_location,
    _recreate_collection,
    _response_text,
    _save_manifest,
    _search_qdrant,
    _upsert_documents,
    process_sources,
)


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


_KEYWORD_PUNCTUATION = (
    " ,.;:?!、。，；：？！"
)
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
        source_type = metadata.get("source_type")
        if source_type != "attachment" and not name.startswith(
            _ATTACHMENT_PREFIX
        ):
            continue
        if source_scope and source_scope.is_active():
            if not name or not _doc_in_scope(name, source_scope):
                continue
        text = doc.page_content
        if not text:
            continue
        matches = pattern.findall(text)
        if matches:
            scored.append((len(matches), doc))
    scored.sort(key=lambda item: (-item[0], item[1].metadata.get("page") or 0))
    return [doc for _, doc in scored[:limit]]


def retrieve_documents(
    query: str,
    client,
    collection_name: str,
    raw_docs: list[Document],
    embedder: GoogleGenerativeAIEmbeddings,
    k: int = 6,
    fetch_k: int = 24,
    alpha: float = 0.7,
    source_scope: "SourceScope | None" = None,
    keyword_query: str | None = None,
) -> tuple[list[Document], str]:
    fetch_k = max(fetch_k, k)
    fallback_docs = _filter_docs_by_scope(raw_docs, source_scope)
    query_vec = embedder.embed_query(query)
    form_results = _search_qdrant(
        client,
        collection_name,
        query_vec,
        limit=fetch_k,
        query_filter=_source_type_filter("form"),
    )
    attachment_results = _search_qdrant(
        client,
        collection_name,
        query_vec,
        limit=fetch_k,
        query_filter=_source_type_filter("attachment"),
    )
    form_docs, form_scores = _docs_from_search_results(form_results)
    attachment_docs, attachment_scores = _docs_from_search_results(
        attachment_results
    )
    form_docs = _tag_retriever(form_docs, "vector:form")
    attachment_docs = _tag_retriever(attachment_docs, "vector:attachment")
    print(
        f"Retrievers: form={len(form_docs)}, attachment={len(attachment_docs)}"
    )
    docs = form_docs + attachment_docs
    if form_scores.size == 0:
        vector_scores = attachment_scores
    elif attachment_scores.size == 0:
        vector_scores = form_scores
    else:
        vector_scores = np.concatenate([form_scores, attachment_scores])
    docs, vector_scores = _filter_docs_and_scores_by_scope(
        docs, vector_scores, source_scope
    )
    if source_scope and source_scope.is_active() and not docs:
        print("No documents matched the current retrieval scope.")
    docs = _hybrid_rerank(query, docs, vector_scores, k=k, alpha=alpha)
    if keyword_query is None:
        keyword_query = query
    regex_docs = _regex_attachment_fallback(
        fallback_docs, keyword_query, limit=min(6, k), source_scope=source_scope
    )
    regex_docs = _tag_retriever(regex_docs, "regex")
    keyword_docs = _keyword_fallback(
        fallback_docs, keyword_query, limit=min(3, k)
    )
    keyword_docs = _tag_retriever(keyword_docs, "keyword")
    had_keyword = bool(keyword_docs)
    had_regex = bool(regex_docs)
    if had_regex:
        keyword_docs = _merge_docs_with_retriever(regex_docs, keyword_docs)
    if keyword_docs:
        if docs:
            docs = _merge_docs_with_retriever(keyword_docs, docs)[:k]
            if had_regex and had_keyword:
                return docs, "hybrid+keyword+regex"
            if had_regex:
                return docs, "hybrid+regex"
            return docs, "hybrid+keyword"
        if had_regex and not had_keyword:
            return keyword_docs, "regex_fallback"
        return keyword_docs, "keyword_fallback"
    if not docs:
        return [], "none"

    if not _docs_have_keyword_hits(docs, keyword_query):
        if keyword_docs:
            return keyword_docs, "keyword_fallback"

    return docs, "hybrid"


_QUESTION_WORDS = ("什么", "多少", "几", "如何", "是否", "哪里", "哪", "谁")
_DOMAIN_HINTS = ("增值税", "税率", "计税", "征收率", "发票", "项目计税类型")
_RULE_MARKERS = ("规则", "不得", "仅", "若", "否则", "返回", "输出", "识别")
_IGNORE_LINE_RE = re.compile(r"^\s*(\d+[\.\)]|[-*•])\s*")
_IGNORE_STARTS = ("你是", "你的任务", "请严格", "注意", "说明")
_KEYWORD_HINTS = (
    "增值税税率",
    "税率",
    "征收率",
    "计税方法",
    "一般计税",
    "简易计算",
    "增值税专用发票",
    "增值税普通发票",
)

_FORM_PREFIX = "表单"
_ATTACHMENT_PREFIX = "附件"
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
_VALUE_NUMERIC_RE = re.compile(
    r"^[\d\s,\.\-/%()％（）]+$"
)


@dataclass(frozen=True)
class SourceScope:
    names: tuple[str, ...] = ()
    prefix: str | None = None
    include_terms: tuple[str, ...] = ()
    exclude_terms: tuple[str, ...] = ()

    def is_active(self) -> bool:
        return bool(self.names or self.prefix or self.include_terms or self.exclude_terms)


def _is_rule_like(line: str) -> bool:
    if _IGNORE_LINE_RE.match(line):
        return True
    if any(line.startswith(prefix) for prefix in _IGNORE_STARTS):
        return True
    if any(marker in line for marker in _RULE_MARKERS) and not line.endswith(("?", "？")):
        return True
    return False


def _build_retrieval_query(prompt: str) -> str:
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    if not lines:
        return prompt.strip()

    candidates: list[str] = []
    for line in lines:
        if _is_rule_like(line):
            continue
        if line.endswith(("?", "？")) or any(word in line for word in _QUESTION_WORDS):
            candidates.append(line)
            continue
        if any(hint in line for hint in _DOMAIN_HINTS) and len(line) <= 80:
            candidates.append(line)

    if candidates:
        seen = set()
        deduped = []
        for line in candidates:
            if line not in seen:
                seen.add(line)
                deduped.append(line)
        return " ".join(deduped[:3])

    return lines[-1]


def _extract_prompt_keywords(prompt: str) -> list[str]:
    matches: list[str] = []
    for hint in _KEYWORD_HINTS:
        if hint in prompt:
            matches.append(hint)
    for hint in _DOMAIN_HINTS:
        if hint in prompt and hint not in matches:
            matches.append(hint)
    for item in re.findall(r"\d{1,2}%|\d{1,2}\s*%", prompt):
        token = item.replace(" ", "")
        if token not in matches:
            matches.append(token)
    return matches


def _build_keyword_query(prompt: str, base_query: str) -> str:
    keywords = _extract_prompt_keywords(prompt)
    if not keywords:
        return base_query
    combined = [base_query] if base_query else []
    combined.extend(keywords[:8])
    return " ".join(combined)


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


_CONTRACT_NAME_STRONG = (
    "建设工程施工合同",
    "施工合同",
    "合同协议书",
    "合同条款",
)
_CONTRACT_NAME_EXCLUDES = (
    "条款核对",
    "核对",
    "对照",
    "招标文件",
    "投标文件",
    "中标通知书",
    "答疑",
    "澄清",
    "控制价",
    "报告书",
    "清单",
    "汇总",
    "说明",
    "补遗",
    "更正",
    "图纸",
    "纪要",
)
_CONTRACT_TEXT_STRONG = (
    "发包人",
    "承包人",
    "合同价款",
    "签订地点",
    "签订日期",
    "通用合同条款",
    "专用合同条款",
    "工程名称",
    "工期",
)
_CONTRACT_TEXT_EXCLUDES = (
    "投标人",
    "招标人",
    "评标委员会",
    "招标文件",
    "答疑",
    "澄清",
)


def _contract_text_sample(docs: list[Document], limit: int = 4000) -> str:
    if not docs:
        return ""
    chunks: list[str] = []
    total = 0
    for doc in docs:
        text = doc.page_content or ""
        if not text:
            continue
        if total + len(text) > limit:
            remaining = max(0, limit - total)
            if remaining:
                chunks.append(text[:remaining])
            break
        chunks.append(text)
        total += len(text)
        if total >= limit:
            break
    return "\n".join(chunks)


def _score_contract_candidate(name: str, text: str) -> int:
    score = 0
    strong_hit = False
    for term in _CONTRACT_NAME_STRONG:
        if term in name:
            score += 6
            strong_hit = True
    if not strong_hit and "合同" in name:
        score += 2
    for term in _CONTRACT_NAME_EXCLUDES:
        if term in name:
            score -= 8
    for term in _CONTRACT_TEXT_STRONG:
        if term in text:
            score += 2
    if "发包人" in text and "承包人" in text:
        score += 4
    if "合同价款" in text:
        score += 3
    for term in _CONTRACT_TEXT_EXCLUDES:
        if term in text:
            score -= 3
    return score


def _select_contract_names(raw_docs: list[Document]) -> list[str]:
    if not raw_docs:
        return []
    grouped: dict[str, list[Document]] = {}
    for doc in raw_docs:
        metadata = doc.metadata or {}
        if metadata.get("source_type") != "attachment":
            continue
        source = metadata.get("source")
        if not source:
            continue
        name = Path(source).name
        grouped.setdefault(name, []).append(doc)
    scored: list[tuple[int, str]] = []
    for name, docs in grouped.items():
        text_sample = _contract_text_sample(docs)
        score = _score_contract_candidate(name, text_sample)
        if score > 0:
            scored.append((score, name))
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    max_score = scored[0][0]
    return [name for score, name in scored if score >= max_score - 2]


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

    if raw_docs is not None and "合同" in prompt:
        contract_names = _select_contract_names(raw_docs)
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


def _doc_in_scope(name: str, scope: SourceScope) -> bool:
    if scope.names and name not in scope.names:
        return False
    if scope.prefix and not name.startswith(scope.prefix):
        return False
    if scope.include_terms and not all(term in name for term in scope.include_terms):
        return False
    if scope.exclude_terms and any(term in name for term in scope.exclude_terms):
        return False
    return True


def _filter_docs_by_scope(
    docs: list[Document], scope: SourceScope | None
) -> list[Document]:
    if scope is None or not scope.is_active():
        return docs
    filtered: list[Document] = []
    for doc in docs:
        source = doc.metadata.get("source") or ""
        name = Path(source).name
        if name and _doc_in_scope(name, scope):
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
        if name and _doc_in_scope(name, scope):
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
    blocks = [
        block.strip() for block in re.split(r"\n{2,}", text) if block.strip()
    ]
    if len(blocks) > 1:
        return blocks
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [text.strip()] if text.strip() else []


def _doc_id_from_source(source: str, source_dir: Path) -> str:
    try:
        return str(Path(source).resolve().relative_to(source_dir.resolve()))
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
    for source, docs in grouped.items():
        name = Path(source).name
        if name.startswith(_FORM_PREFIX):
            combined = "\n".join(doc.page_content for doc in docs)
            form_fields_by_source[source] = _parse_form_fields(combined)

    inferred_process_id = process_id
    if inferred_process_id is None:
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

    output_docs: list[Document] = []
    for source, docs in grouped.items():
        name = Path(source).name
        doc_id = _doc_id_from_source(source, source_dir)
        base_meta: dict[str, object] = {"source": source, "doc_id": doc_id}
        if inferred_process_id:
            base_meta["process_id"] = inferred_process_id
        if created_at:
            base_meta["created_at"] = created_at

        if name.startswith(_FORM_PREFIX):
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
                output_docs.append(
                    Document(page_content=content, metadata=metadata)
                )
            continue

        filename = name
        if name.startswith(_ATTACHMENT_PREFIX):
            filename = name[len(_ATTACHMENT_PREFIX) :]

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


def get_vectorstore(
    docs: list[Document],
    persist_dir: Path,
    processed_dir: Path,
    force_rebuild: bool = False,
):
    embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
    chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    collection_name = os.getenv("QDRANT_COLLECTION", "contract_approval_rag")
    qdrant_location = _qdrant_location(persist_dir)

    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    manifest = _build_index_manifest(
        processed_dir, embedding_model, chunk_size, chunk_overlap
    )
    manifest["collection_name"] = collection_name
    manifest["qdrant_location"] = qdrant_location
    manifest["ingestion_schema"] = "contract_approval_v2"

    client = _get_qdrant_client(persist_dir)

    if not force_rebuild:
        stored_manifest = _load_manifest(persist_dir)
        if _manifest_matches(stored_manifest, manifest) and _collection_exists(
            client, collection_name
        ):
            return client, collection_name, embeddings

    splits = _assign_chunk_ids(docs)
    if not splits:
        raise SystemExit("No content left after splitting documents.")

    vectors = embeddings.embed_documents([doc.page_content for doc in splits])
    if not vectors:
        raise SystemExit("Embedding model returned no vectors.")
    vector_size = len(vectors[0])
    _recreate_collection(client, collection_name, vector_size)
    _upsert_documents(client, collection_name, splits, vectors)
    _save_manifest(persist_dir, manifest, doc_count=len(splits))
    return client, collection_name, embeddings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG demo")
    parser.add_argument("question", nargs="?", help="Question to ask.")
    parser.add_argument("--k", type=int, default=6, help="Top-k chunks to return.")
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=24,
        help="Candidate pool size for hybrid retrieval.",
    )
    parser.add_argument(
        "--image-dpi", type=int, default=200, help="DPI for OCR PDF rendering."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector index even if cached.",
    )
    parser.add_argument(
        "--process-id",
        help="Process instance ID for metadata (overrides env PROCESS_ID).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    root = Path(__file__).resolve().parents[1]
    source_dir = root / "data" / "source"
    processed_dir = root / "data" / "processed"
    persist_dir = root / "index"

    ocr_tool = _LazyOCR()
    raw_docs = process_sources(
        source_dir, processed_dir, ocr_tool, image_dpi=args.image_dpi
    )
    process_id = (
        args.process_id
        or os.getenv("PROCESS_ID")
        or os.getenv("RAG_PROCESS_ID")
    )
    created_at = os.getenv("RAG_CREATED_AT") or os.getenv("CREATED_AT")
    raw_docs, inferred_process_id = _build_contract_documents(
        raw_docs,
        source_dir,
        process_id=process_id,
        created_at=created_at,
    )
    if not inferred_process_id:
        logging.warning(
            "process_id is missing; set PROCESS_ID or --process-id for filtering."
        )
    client, collection_name, embedder = get_vectorstore(
        raw_docs, persist_dir, processed_dir, force_rebuild=args.rebuild
    )

    model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    # question = args.question or os.getenv("QUESTION") or """
    #     项目计税类型
    #     用于识别合同中适用的增值税计税方式。
    #     识别规则：
    #     若合同中明确约定一般计税或简易计税，直接提取；
    #     若未明确约定，通过税率（6%、9%、13%）或征收率（3%、5%）进行推断；
    #     若仍未明确，结合项目类型并依据税法知识库进行推断；
    #     无法唯一判断时，返回“未明确”。
    
    #     分别找出：施工合同审批表项目计税类型是什么？合同里面的项目计税类型是什么？"""
    
    # question = args.question or os.getenv("QUESTION") or "合同里面的项目计税类型你是怎么得出来的，我在合同里面没有找到项目计税类型相关信息。我需要对比施工合同审批表.txt，和合同里面的计税类型是否一致"

    question = args.question or os.getenv("QUESTION") or """
        你是一名合同税务信息抽取助手。
        你的任务是从合同文本中识别并提取“增值税税率”。

        识别规则：
        1. 仅提取合同中明确出现的增值税税率数值，如：6%、9%、13%。
        2. 若合同中出现“征收率”（如3%、5%），请区分其与税率的概念，不得将征收率作为税率输出。
        3. 若合同中同时出现多个税率，仅在合同明确区分适用范围时分别列出；否则标记为“多税率，需人工确认”。
        4. 若合同未明确出现任何增值税税率，不得根据项目类型或经验进行推断，直接返回“未明确”。

        附件合同里面的增值税税率是多少？"""
    
    # 分别找出：施工合同审批表的增值税税率是多少？合同里面的增值税税率是多少？
    question = question.strip()
    if not question:
        return

    alpha = float(os.getenv("RAG_ALPHA", "0.7"))
    retrieval_query = _build_retrieval_query(question)
    keyword_query = _build_keyword_query(question, retrieval_query)
    source_names = _collect_source_names(raw_docs)
    source_scope = _infer_source_scope(question, source_names, raw_docs)
    print(f"Retrieval query:\n {retrieval_query}")
    if keyword_query != retrieval_query:
        print(f"Keyword query:\n {keyword_query}")
    if source_scope.is_active():
        print(f"Retrieval scope: {_format_source_scope(source_scope)}")
        matched_files = _match_source_names(source_scope, source_names)
        if matched_files:
            print("Scope files:")
            for name in matched_files[:20]:
                print("-", name)
        else:
            print("Scope files: <none>")
            print("No files matched the retrieval scope; aborting search.")
            return
    docs, strategy = retrieve_documents(
        retrieval_query,
        client,
        collection_name,
        raw_docs,
        embedder,
        k=args.k,
        fetch_k=args.fetch_k,
        alpha=alpha,
        source_scope=source_scope,
        keyword_query=keyword_query,
    )

    if not docs:
        print("No relevant documents found.")
        return

    context = _format_context(docs)
    prompt = (
        "Use the following context to answer the question. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )
    response = llm.invoke(prompt)

    print("Question:\n", question)
    print("Answer:\n", _response_text(response.content))
    print(f"\nRetrieval: {strategy}")
    print("\nSources:")
    for source in _unique_sources_with_retriever(docs):
        print("-", source)


if __name__ == "__main__":
    main()
