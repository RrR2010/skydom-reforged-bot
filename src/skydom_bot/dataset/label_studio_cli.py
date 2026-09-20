"""CLI for exporting the crop dataset to Label Studio storage folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from skydom_bot.dataset.label_studio import export_label_studio_storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export incremental Label Studio source tasks and target folders."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = export_label_studio_storage(args.dataset, args.output)

    print(f"Task definitions total: {summary.total}")
    print(f"New task files created: {summary.created}")
    print(f"Existing task files kept: {summary.existing}")
    print(f"Label config: {summary.config_path}")
    print(f"Source tasks directory: {summary.source_tasks_dir}")
    print(f"Target annotations directory: {summary.target_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
