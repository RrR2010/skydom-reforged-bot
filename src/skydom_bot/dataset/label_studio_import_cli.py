"""CLI for importing Label Studio Target Storage annotations."""

from __future__ import annotations

import argparse
from pathlib import Path

from skydom_bot.dataset.label_studio_import import import_label_studio_annotations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge Label Studio human annotations into dataset records."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Override Label Studio Target Storage annotations directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = import_label_studio_annotations(args.dataset, args.annotations)

    print(f"Annotation files seen: {summary.files_seen}")
    print(f"Tasks with sample_id: {summary.tasks_with_sample_id}")
    print(f"Records updated: {summary.records_updated}")
    print(f"Skipped without complete annotation: {summary.skipped_without_annotation}")
    print(f"Skipped missing record: {summary.skipped_missing_record}")
    print(f"Invalid files: {summary.invalid_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
