"""CLI for collecting raw cell crops into the local training dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from skydom_bot.capture import capture_screen
from skydom_bot.dataset.collector import DatasetCollector
from skydom_bot.vision.board_detector import BoardDetector
from skydom_bot.vision.recognizer import ClassicalTileRecognizer


def _read_rgb(path: Path):
    """Read a saved screenshot as RGB."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_parser() -> argparse.ArgumentParser:
    """Build dataset collection arguments."""
    parser = argparse.ArgumentParser(
        description="Collect de-duplicated tile crops with bootstrap metadata."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path)
    source.add_argument("--screen", action="store_true", help="Capture a monitor (default when --image is omitted).")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    return parser


def main() -> int:
    """Detect the board and persist every active cell as an unlabeled sample."""
    args = build_parser().parse_args()
    image = _read_rgb(args.image) if args.image else capture_screen(args.monitor)

    geometry = BoardDetector().detect(image)
    estimates = ClassicalTileRecognizer().recognize_board(image, geometry)
    source_name = str(args.image) if args.image else f"screen:monitor-{args.monitor}"

    records = DatasetCollector(args.dataset).collect(
        image,
        geometry,
        estimates,
        source=source_name,
    )

    unlabeled = sum(record.labels is None for record in records)
    print(f"Collected/touched samples: {len(records)}")
    print(f"Unlabeled samples: {unlabeled}")
    print(f"Dataset: {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
