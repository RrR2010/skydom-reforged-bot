"""CLI for dataset statistics and training-readiness guidance."""

from __future__ import annotations

import argparse
from pathlib import Path

from skydom_bot.dataset.report import write_collection_report
from skydom_bot.dataset.stats import collect_dataset_statistics


def build_parser() -> argparse.ArgumentParser:
    """Build dataset statistics arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize dataset health and write training-readiness guidance."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Markdown report path (default: <dataset>/dataset-stats.md).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip Markdown report generation.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Also print detailed pair distributions and observed combinations.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when validation issues are found.",
    )
    return parser


def _render_counts(counts: dict[str, int]) -> str:
    """Render one compact distribution line."""
    return ", ".join(f"{name}={count}" for name, count in counts.items()) or "-"


def main() -> int:
    """Print compact dataset health and optionally write a Markdown report."""
    args = build_parser().parse_args()
    stats = collect_dataset_statistics(args.dataset)

    provenance = (
        stats.labeled_with_capture_provenance / stats.labeled
        if stats.labeled
        else 0.0
    )

    print(f"Dataset: {args.dataset}")
    print(
        f"Records: {stats.total} | Labeled: {stats.labeled} | "
        f"Unlabeled: {stats.unlabeled} | Invalid: {stats.invalid}"
    )
    print(
        f"Capture provenance: {stats.labeled_with_capture_provenance}/"
        f"{stats.labeled} labeled ({provenance:.0%})"
    )
    print(f"Color: {_render_counts(stats.distributions['color'])}")
    print(f"Kind: {_render_counts(stats.distributions['kind'])}")
    print(f"Blocker: {_render_counts(stats.distributions['blocker'])}")
    print(f"Power-up: {_render_counts(stats.distributions['powerup'])}")

    if args.details:
        print("\nPair distributions:")
        for pair_name, rows in stats.pair_distributions.items():
            print(f"  {pair_name}:")
            if not rows:
                print("    -")
                continue
            for first_value, second_counts in rows.items():
                print(f"    {first_value}: {_render_counts(second_counts)}")

        print("\nObserved complete combinations:")
        if not stats.combinations:
            print("  -")
        else:
            for combination, count in stats.combinations.items():
                print(f"  {combination}: {count}")

    if not args.no_report:
        report_path = args.report or args.dataset / "dataset-stats.md"
        write_collection_report(report_path, stats)
        print(f"Training-readiness report: {report_path}")

    if stats.issues:
        print(f"Validation issues: {len(stats.issues)}")
        if args.details:
            for issue in stats.issues:
                try:
                    display_path = issue.path.relative_to(args.dataset)
                except ValueError:
                    display_path = issue.path
                print(f"  {display_path}: {issue.message}")

    return 1 if args.strict and stats.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
