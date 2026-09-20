"""CLI for dataset statistics and integrity validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from skydom_bot.dataset.stats import collect_dataset_statistics


def build_parser() -> argparse.ArgumentParser:
    """Build dataset statistics arguments."""
    parser = argparse.ArgumentParser(
        description="Summarize human labels, multidimensional coverage, and integrity."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--target",
        type=int,
        default=30,
        help="Desired samples per actionable power-up/color combination (default: 30).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when validation issues are found.",
    )
    return parser


def _render_dimensions(dimensions: tuple[tuple[str, str], ...]) -> str:
    """Render a compact label-combination description."""
    return ", ".join(f"{field}={value}" for field, value in dimensions)


def main() -> int:
    """Print dataset coverage, collection priorities, and integrity issues."""
    args = build_parser().parse_args()
    stats = collect_dataset_statistics(
        args.dataset,
        target_per_combination=args.target,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Records: {stats.total}")
    print(f"Labeled: {stats.labeled}")
    print(f"Unlabeled: {stats.unlabeled}")
    print(f"Invalid labels/records: {stats.invalid}")
    print(f"With capture provenance: {stats.with_capture_provenance}")
    print(f"Legacy without capture provenance: {stats.without_capture_provenance}")

    print("\nMarginal distributions:")
    for field, counts in stats.distributions.items():
        rendered = ", ".join(f"{name}={count}" for name, count in counts.items())
        print(f"  {field}: {rendered or '-'}")

    print("\nPair distributions:")
    for pair_name, rows in stats.pair_distributions.items():
        print(f"  {pair_name}:")
        if not rows:
            print("    -")
            continue
        for first_value, second_counts in rows.items():
            rendered = ", ".join(
                f"{second_value}={count}"
                for second_value, count in second_counts.items()
            )
            print(f"    {first_value}: {rendered}")

    print("\nPower-up/color collection priorities:")
    actionable = [item for item in stats.powerup_color_gaps if item.gap > 0]
    if not actionable:
        print("  All tracked combinations meet the target.")
    else:
        for item in actionable:
            print(
                f"  [{item.status}] {_render_dimensions(item.dimensions)}: "
                f"{item.count}/{item.target} (gap {item.gap})"
            )

    print("\nObserved complete combinations:")
    if not stats.combinations:
        print("  -")
    else:
        for combination, count in stats.combinations.items():
            print(f"  {combination}: {count}")

    if stats.issues:
        print("\nValidation issues:")
        for issue in stats.issues:
            try:
                display_path = issue.path.relative_to(args.dataset)
            except ValueError:
                display_path = issue.path
            print(f"  {display_path}: {issue.message}")

    return 1 if args.strict and stats.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
