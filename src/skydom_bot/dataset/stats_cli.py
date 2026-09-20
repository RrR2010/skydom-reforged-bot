"""CLI for dataset statistics and integrity validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from skydom_bot.dataset.stats import collect_dataset_statistics


def build_parser() -> argparse.ArgumentParser:
    """Build dataset statistics arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize human labels and validate canonical dataset records."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when validation issues are found.",
    )
    return parser


def main() -> int:
    """Print dataset coverage, class distributions, and integrity issues."""
    args = build_parser().parse_args()
    stats = collect_dataset_statistics(args.dataset)

    print(f"Dataset: {args.dataset}")
    print(f"Records: {stats.total}")
    print(f"Labeled: {stats.labeled}")
    print(f"Unlabeled: {stats.unlabeled}")
    print(f"Invalid labels/records: {stats.invalid}")
    print(f"With capture provenance: {stats.with_capture_provenance}")
    print(f"Legacy without capture provenance: {stats.without_capture_provenance}")

    for field, counts in stats.distributions.items():
        rendered = ", ".join(f"{name}={count}" for name, count in counts.items())
        print(f"{field}: {rendered or '-'}")

    if stats.issues:
        print("Validation issues:")
        for issue in stats.issues:
            try:
                display_path = issue.path.relative_to(args.dataset)
            except ValueError:
                display_path = issue.path
            print(f"  {display_path}: {issue.message}")

    return 1 if args.strict and stats.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
