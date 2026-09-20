"""CLI for manually triggered screen-corpus capture."""

from __future__ import annotations

import argparse
from pathlib import Path

from skydom_bot.capture import capture_screen
from skydom_bot.corpus.capture import (
    load_region,
    parse_metadata,
    parse_region,
    save_capture,
    save_region,
    select_region_interactively,
)


def build_parser() -> argparse.ArgumentParser:
    """Build corpus-capture CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="skydom-capture-screen",
        description="Capture one useful-game-area screenshot into the local corpus.",
    )
    parser.add_argument(
        "stage_id",
        nargs="?",
        help="Stage/level identifier for this capture, for example level-27.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("corpus"),
        help="Local corpus root (default: corpus).",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=None,
        help="Monitor index. Defaults to the configured monitor or 1.",
    )
    parser.add_argument(
        "--region",
        help="Override/configure game region as X,Y,W,H in monitor-local pixels.",
    )
    parser.add_argument(
        "--meta",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Optional capture metadata. Repeat as needed.",
    )
    parser.add_argument(
        "--configure",
        action="store_true",
        help="Interactively select and save the useful game region, then exit.",
    )
    return parser


def main() -> int:
    """Configure the game region or save one manually triggered screenshot."""
    args = build_parser().parse_args()

    if args.configure:
        monitor = args.monitor or 1
        image = capture_screen(monitor)
        region = (
            parse_region(args.region)
            if args.region
            else select_region_interactively(image)
        )
        region.validate(image)
        path = save_region(args.corpus, region, monitor=monitor)
        print(
            f"Configured game region: "
            f"{region.x},{region.y},{region.width},{region.height} "
            f"(monitor {monitor})"
        )
        print(f"Config: {path}")
        return 0

    if not args.stage_id:
        raise SystemExit("stage_id is required unless --configure is used.")

    if args.region:
        configured_monitor = args.monitor or 1
        region = parse_region(args.region)
        save_region(args.corpus, region, monitor=configured_monitor)
    else:
        configured_monitor, region = load_region(args.corpus)

    monitor = args.monitor or configured_monitor
    metadata = parse_metadata(args.meta)
    record = save_capture(
        args.corpus,
        stage_id=args.stage_id,
        monitor=monitor,
        region=region,
        metadata=metadata,
    )

    print(f"Captured: {record.capture_id}")
    print(f"Stage: {record.stage_id}")
    print(f"Image: {args.corpus / record.image}")
    if record.metadata:
        print(
            "Metadata: "
            + ", ".join(
                f"{key}={value}" for key, value in record.metadata.items()
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
