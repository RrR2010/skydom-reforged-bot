"""CLI for exporting the crop dataset to Label Studio."""

from __future__ import annotations

import argparse
from pathlib import Path

from skydom_bot.dataset.label_studio import export_label_studio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export local tile crops and bootstrap predictions to Label Studio."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path, tasks_path, count = export_label_studio(
        args.dataset,
        args.output,
    )
    print(f"Tasks exported: {count}")
    print(f"Label config: {config_path}")
    print(f"Tasks JSON: {tasks_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
