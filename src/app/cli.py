from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from app.settings import Settings
from kb.answer import answer_question
from kb.ingest import ingest_kb
from kb.io.manifest_store import load_file_records
from kb.registry import load_kb_paths
from kb.retrieve import retrieve_documents


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Personal KB P0 CLI (Bailian + remote OCR)")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest files into a KB")
    ingest.add_argument("--root", required=True, help="Absolute path to source root")
    ingest.add_argument("--kb", help="KB id to use (optional)")

    query = sub.add_parser("query", help="Query a KB")
    query.add_argument("--kb", help="KB id to use")
    query.add_argument("--root", help="Root path to resolve KB id")
    query.add_argument("--top-k", type=int, default=0, help="Top-K results")
    query.add_argument("question", help="Question to ask")

    return parser.parse_args()


def _resolve_kb_id_for_query(root_path: str | None, kb_id: str | None, settings: Settings) -> str:
    if kb_id:
        return kb_id
    if not root_path:
        raise SystemExit("Provide --kb or --root to locate the KB.")
    root = Path(root_path).resolve()
    mapping = load_kb_paths(settings.manifest_dir)
    kb = mapping.get(str(root))
    if not kb:
        raise SystemExit(f"KB not found for root path: {root}")
    return kb


def main() -> None:
    args = _parse_args()
    settings = Settings.from_env(ROOT)

    if args.command == "ingest":
        kb_id = ingest_kb(Path(args.root), kb_id=args.kb, settings=settings)
        print(f"Ingest completed. kb_id={kb_id}")
        return

    if args.command == "query":
        kb_id = _resolve_kb_id_for_query(args.root, args.kb, settings)
        top_k = args.top_k or None
        docs = retrieve_documents(kb_id=kb_id, question=args.question, settings=settings, top_k=top_k)
        file_records = {r.abs_path: r for r in load_file_records(settings.manifest_dir, kb_id)}
        result = answer_question(
            question=args.question,
            docs=docs,
            settings=settings,
            file_records=file_records,
        )
        print(result.answer)
        if result.citations:
            print("\nCitations:")
            for cite in result.citations:
                suffix = f"#p{cite.page}" if cite.page is not None else ""
                deleted = " (source_deleted)" if cite.source_deleted else ""
                print(f"- {Path(cite.source).name}{suffix}{deleted}")
        return


if __name__ == "__main__":
    main()
