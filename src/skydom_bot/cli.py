"""Command-line diagnostic entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from skydom_bot.capture import capture_screen
from skydom_bot.debug.report import format_geometry_summary, save_board_overlay
from skydom_bot.vision.board_detector import BoardDetector

UInt8Image = NDArray[np.uint8]


def _read_rgb(path: Path) -> UInt8Image:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Inspect Skydom board geometry from an image or screen capture.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Input screenshot path.")
    source.add_argument("--screen", action="store_true", help="Capture the primary monitor.")
    parser.add_argument("--monitor", type=int, default=1, help="MSS monitor index used with --screen.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/board-overlay.png"))
    return parser


def main() -> int:
    """Detect and print board geometry, then write a diagnostic overlay."""
    args = build_parser().parse_args()
    image = _read_rgb(args.image) if args.image else capture_screen(args.monitor)
    geometry = BoardDetector().detect(image)

    print(format_geometry_summary(geometry))
    print(f"Overlay: {save_board_overlay(image, geometry, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
