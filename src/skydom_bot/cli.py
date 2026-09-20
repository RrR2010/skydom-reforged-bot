"""Command-line diagnostic entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from skydom_bot.capture import capture_screen
from skydom_bot.debug.overlay import draw_board_overlay
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

    print(f"Board: {geometry.rows}x{geometry.cols}")
    print(f"Bounds: x={geometry.bounds.x}, y={geometry.bounds.y}, w={geometry.bounds.width}, h={geometry.bounds.height}")
    print(f"Pitch: {geometry.pitch_x:.2f} x {geometry.pitch_y:.2f}")
    print(f"Active cells: {len(geometry.cells)}")
    print(f"Confidence: {geometry.confidence:.3f}")
    print("Topology:")
    print(geometry.topology_text())

    overlay = draw_board_overlay(image, geometry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Overlay: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
