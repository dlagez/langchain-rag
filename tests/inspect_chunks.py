from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from util.vectorstore_utils import _get_qdrant_client  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect chunk sizes and contents."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max chunks to print (0=all).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Max chars per chunk (0=all).",
    )
    parser.add_argument(
        "--source-contains",
        help="Only show chunks whose source filename contains this string.",
    )
    parser.add_argument(
        "--doc-id-contains",
        help="Only show chunks whose metadata doc_id contains this string.",
    )
    parser.add_argument(
        "--doc-id",
        help="Only show chunks whose metadata doc_id matches this string exactly.",
    )
    parser.add_argument(
        "--chunk-id",
        type=int,
        help="Only show chunks whose metadata chunk_id matches.",
    )
    parser.add_argument(
        "--point-id",
        help="Only show the point with this id (int or uuid).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Scroll batch size.",
    )
    parser.add_argument(
        "--collection",
        help="Override QDRANT_COLLECTION.",
    )
    return parser.parse_args()


def _doc_source_name(metadata: dict) -> str:
    source = metadata.get("source") or "unknown"
    return Path(source).name


def _metadata_matches_chunk_id(metadata: dict, target: int) -> bool:
    if "chunk_id" not in metadata:
        return False
    value = metadata.get("chunk_id")
    return str(value) == str(target)


def _metadata_matches_doc_id(
    metadata: dict, target: str, *, contains: bool
) -> bool:
    if "doc_id" not in metadata:
        return False
    value = metadata.get("doc_id")
    if value is None:
        return False
    value = str(value)
    if contains:
        return target in value
    return value == target


def _parse_point_id(value: str):
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.lstrip("-").isdigit():
        try:
            return int(text)
        except ValueError:
            return text
    return text


def _format_meta(metadata: dict, point_id) -> str:
    md = metadata
    parts = []
    if point_id is not None:
        parts.append(f"point_id={point_id}")
    source = md.get("source")
    if source:
        parts.append(f"source={Path(source).name}")
    if "source_type" in md:
        parts.append(f"type={md['source_type']}")
    if "doc_type_hint" in md:
        parts.append(f"doc_type={md['doc_type_hint']}")
    if "field" in md:
        parts.append(f"field={md['field']}")
    if "page" in md:
        parts.append(f"page={md['page']}")
    if "chunk_id" in md:
        parts.append(f"chunk_id={md['chunk_id']}")
    if "doc_id" in md:
        parts.append(f"doc_id={md['doc_id']}")
    return ", ".join(parts)


def _count_points(client, collection: str) -> int | None:
    try:
        result = client.count(collection_name=collection, exact=True)
    except Exception:
        return None
    return getattr(result, "count", None)


def main() -> None:
    load_dotenv()
    args = _parse_args()

    collection = (
        args.collection
        or os.getenv("QDRANT_COLLECTION")
        or "contract_approval_rag"
    )
    client = _get_qdrant_client(ROOT / "index")
    total = _count_points(client, collection)
    if total is not None:
        print(f"Total chunks (collection count): {total}")

    if args.point_id:
        point_id = _parse_point_id(args.point_id)
        if hasattr(client, "retrieve"):
            points = client.retrieve(
                collection_name=collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        else:
            points = []

        if not points:
            print(f"No point found for id: {args.point_id}")
            return

        sizes = []
        printed = 0
        scanned = 0
        for point in points:
            scanned += 1
            payload = point.payload or {}
            text = payload.get("page_content") or ""
            metadata = payload.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            size = len(text)
            sizes.append(size)

            if args.source_contains:
                if args.source_contains not in _doc_source_name(metadata):
                    continue
            if args.chunk_id is not None:
                if not _metadata_matches_chunk_id(metadata, args.chunk_id):
                    continue
            if args.doc_id:
                if not _metadata_matches_doc_id(
                    metadata, args.doc_id, contains=False
                ):
                    continue
            if args.doc_id_contains:
                if not _metadata_matches_doc_id(
                    metadata, args.doc_id_contains, contains=True
                ):
                    continue

            if args.limit and printed >= args.limit:
                continue

            display = text
            if args.max_chars and len(display) > args.max_chars:
                display = display[: args.max_chars].rstrip() + "..."
            print("")
            print(f"--- Chunk {printed + 1} ({size} chars) ---")
            print(_format_meta(metadata, getattr(point, "id", None)))
            print(display)
            printed += 1

        if sizes:
            print(
                "Size stats (chars) for scanned: "
                f"min={min(sizes)} max={max(sizes)} avg={mean(sizes):.1f}"
            )
        print(f"Scanned chunks: {scanned}")
        if printed == 0:
            print("No chunks matched the current filters.")
        return

    sizes = []
    printed = 0
    scanned = 0
    next_offset = None
    stop = False
    while True:
        response = client.scroll(
            collection_name=collection,
            limit=max(1, args.batch_size),
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
            text = payload.get("page_content") or ""
            metadata = payload.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            size = len(text)
            sizes.append(size)

            if args.source_contains:
                if args.source_contains not in _doc_source_name(metadata):
                    continue
            if args.chunk_id is not None:
                if not _metadata_matches_chunk_id(metadata, args.chunk_id):
                    continue
            if args.doc_id:
                if not _metadata_matches_doc_id(
                    metadata, args.doc_id, contains=False
                ):
                    continue
            if args.doc_id_contains:
                if not _metadata_matches_doc_id(
                    metadata, args.doc_id_contains, contains=True
                ):
                    continue

            if args.limit and printed >= args.limit:
                continue

            display = text
            if args.max_chars and len(display) > args.max_chars:
                display = display[: args.max_chars].rstrip() + "..."
            print("")
            print(f"--- Chunk {printed + 1} ({size} chars) ---")
            print(_format_meta(metadata, getattr(point, "id", None)))
            print(display)
            printed += 1
            if args.limit and printed >= args.limit:
                stop = True
                break

        if stop:
            break
        if next_offset is None:
            break

    if sizes:
        print(
            "Size stats (chars) for scanned: "
            f"min={min(sizes)} max={max(sizes)} avg={mean(sizes):.1f}"
        )
    print(f"Scanned chunks: {scanned}")
    if printed == 0:
        print("No chunks matched the current filters.")


if __name__ == "__main__":
    main()

# python tests/inspect_chunks.py --limit 1 --max-chars 400
# python tests/inspect_chunks.py --source-contains 合同 --limit 3 --max-chars 0
# python tests/inspect_chunks.py --chunk-id 0 --max-chars 400
# python tests/inspect_chunks.py --point-id 100 --collection rag_docs
# python tests/inspect_chunks.py --doc-id "attachment/高新三路声环境提升工程预留段建设项目工程总承包（EPC）项目合同12.12.pdf" --limit 3 --max-chars 0
# python tests/inspect_chunks.py --doc-id-contains "合同12.12.pdf" --limit 3 --max-chars 0



# --source-contains 合同 只看文件名包含“合同”的 chunk
# --collection your_collection 覆盖 QDRANT_COLLECTION
# --batch-size 100 调整 scroll 批大小
