from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from util.vectorstore_utils import _get_qdrant_client  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print vector store schema and payload keys."
    )
    parser.add_argument(
        "--collection",
        help="Override QDRANT_COLLECTION.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=200,
        help="Max points to scan for payload keys (0=all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Scroll batch size.",
    )
    return parser.parse_args()


def _stringify_distance(value) -> str | None:
    if value is None:
        return None
    return getattr(value, "value", None) or str(value)


def _extract_vector_configs(info) -> list[tuple[str, int | None, str | None]]:
    params = getattr(getattr(info, "config", None), "params", None)
    if not params:
        return []
    vectors = getattr(params, "vectors", None) or getattr(params, "vector", None)
    if vectors is None:
        return []

    configs: list[tuple[str, int | None, str | None]] = []
    if hasattr(vectors, "size"):
        configs.append(
            ("default", vectors.size, _stringify_distance(vectors.distance))
        )
        return configs

    items = None
    if hasattr(vectors, "params_map"):
        items = getattr(vectors, "params_map", None)
    elif isinstance(vectors, dict):
        items = vectors
    if not items:
        return configs

    for name, cfg in items.items():
        size = getattr(cfg, "size", None)
        distance = _stringify_distance(getattr(cfg, "distance", None))
        configs.append((str(name), size, distance))
    return configs


def _print_payload_schema(info) -> None:
    payload_schema = getattr(info, "payload_schema", None) or {}
    if not payload_schema:
        print("Payload schema (indexed fields): <none>")
        return
    print("Payload schema (indexed fields):")
    for key, schema in payload_schema.items():
        data_type = getattr(schema, "data_type", None)
        if data_type is None and isinstance(schema, dict):
            data_type = schema.get("data_type")
        data_type = getattr(data_type, "value", None) or data_type
        print(f"- {key}: {data_type}")


def main() -> None:
    load_dotenv()
    args = _parse_args()

    collection = (
        args.collection
        or os.getenv("QDRANT_COLLECTION")
        or "contract_approval_rag"
    )
    client = _get_qdrant_client(ROOT / "index")

    info = client.get_collection(collection_name=collection)
    print(f"Collection: {collection}")
    vector_configs = _extract_vector_configs(info)
    if vector_configs:
        print("Vector config:")
        for name, size, distance in vector_configs:
            print(f"- {name}: size={size} distance={distance}")
    else:
        print("Vector config: <unknown>")

    count = getattr(info, "points_count", None)
    if count is not None:
        print(f"Points count: {count}")

    _print_payload_schema(info)

    payload_keys: set[str] = set()
    metadata_keys: set[str] = set()
    metadata_types: dict[str, set[str]] = defaultdict(set)

    scanned = 0
    next_offset = None
    max_scan = args.sample if args.sample and args.sample > 0 else None
    while True:
        limit = max(1, args.batch_size)
        if max_scan is not None:
            remaining = max_scan - scanned
            if remaining <= 0:
                break
            limit = min(limit, remaining)

        response = client.scroll(
            collection_name=collection,
            limit=limit,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", None)
        if points is None:
            points, next_offset = response
        else:
            next_offset = getattr(response, "next_page_offset", None)

        if not points:
            break

        for point in points:
            scanned += 1
            payload = point.payload or {}
            payload_keys.update(payload.keys())
            metadata = payload.get("metadata") or {}
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    metadata_keys.add(key)
                    metadata_types[key].add(type(value).__name__)

        if next_offset is None:
            break

    print(f"Scanned points: {scanned}")
    print(
        "Payload keys observed: "
        + (", ".join(sorted(payload_keys)) if payload_keys else "<none>")
    )
    print(
        "Metadata keys observed: "
        + (", ".join(sorted(metadata_keys)) if metadata_keys else "<none>")
    )
    if metadata_types:
        print("Metadata value types:")
        for key in sorted(metadata_types):
            types = ", ".join(sorted(metadata_types[key]))
            print(f"- {key}: {types}")


if __name__ == "__main__":
    main()

# (.venv) C:\Users\roc\code\langchain-rag>python tests/print_schema.py
# Collection: rag_docs 当前查的是集合名 rag_docs（来自 .env 里的 QDRANT_COLLECTION）
# Vector config:
# - default: size=768 distance=Cosine 向量维度是 768，检索用余弦相似度。
# Points count: 11119 向量库里一共有 11119 个 chunk（point）
# Payload schema (indexed fields): <none>
# Scanned points: 200 只扫描了 200 条样本，所以字段不一定齐全。
# Payload keys observed: metadata, page_content 每条记录就两块——文本 page_content + 元数据 metadata。

# Metadata keys observed: chunk_id, chunk_type, doc_id, doc_type_hint, field, filename, has_checkbox_like, signature_page, source, source_type
# Metadata value types:
# - chunk_id: int
# - chunk_type: str
# - doc_id: str
# - doc_type_hint: str
# - field: str
# - filename: str
# - has_checkbox_like: bool
# - signature_page: bool
# - source: str
# - source_type: str