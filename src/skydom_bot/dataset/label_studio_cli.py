"""CLI for exporting the crop dataset to Label Studio storage folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from skydom_bot.dataset.label_studio import export_label_studio_storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export incremental Label Studio task batches and storage folders."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional override for the Label Studio input directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = export_label_studio_storage(args.dataset, args.output)

    print(f"Samples total: {summary.total_samples}")
    print(f"New samples exported: {summary.new_samples}")
    print(f"Existing samples kept: {summary.existing_samples}")
    print(f"Images copied into input: {summary.copied_images}")
    print(f"Label config: {summary.config_path}")
    print(f"Source storage directory: {summary.input_dir}")
    print(f"Target annotations directory: {summary.output_dir}")
    if summary.batch_path is not None:
        print(f"New task batch: {summary.batch_path}")
    else:
        print("New task batch: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
