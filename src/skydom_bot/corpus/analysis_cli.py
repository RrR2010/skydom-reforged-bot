"""CLI for offline screen-corpus board analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from skydom_bot.corpus.analysis import analyze_corpus, write_analysis_outputs


def build_parser() -> argparse.ArgumentParser:
    """Build corpus-analysis CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="skydom-analyze-corpus",
        description="Run the current board detector against every captured screen.",
    )
    parser.add_argument("--corpus", type=Path, default=Path("corpus"))
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--pitch-delta", type=float, default=3.0)
    parser.add_argument(
        "--export-cells",
        action="store_true",
        help="Export detected cell crops under corpus/derived/cells/.",
    )
    return parser


def main() -> int:
    """Analyze the corpus and write review-oriented reports."""
    args = build_parser().parse_args()
    records = analyze_corpus(
        args.corpus,
        confidence_threshold=args.confidence,
        pitch_delta_threshold=args.pitch_delta,
        export_cells=args.export_cells,
    )
    jsonl_path, csv_path, markdown_path = write_analysis_outputs(
        args.corpus,
        records,
    )

    ok = sum(record.status == "OK" for record in records)
    review = sum(record.status == "REVIEW" for record in records)
    failed = sum(record.status == "FAIL" for record in records)

    print(
        f"Corpus: {len(records)} | OK: {ok} | REVIEW: {review} | FAIL: {failed}"
    )
    print(f"Markdown: {markdown_path}")
    print(f"CSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
