"""Shared textual and image reports for board-geometry inspection."""

from __future__ import annotations

from pathlib import Path

import cv2
from numpy.typing import NDArray
import numpy as np

from skydom_bot.debug.overlay import draw_board_overlay
from skydom_bot.domain.board import BoardGeometry

UInt8Image = NDArray[np.uint8]


def format_geometry_summary(geometry: BoardGeometry) -> str:
    """Return the canonical CLI-style summary for detected board geometry."""
    bounds = geometry.bounds
    return "\n".join(
        (
            f"Board: {geometry.rows}x{geometry.cols}",
            f"Bounds: x={bounds.x}, y={bounds.y}, w={bounds.width}, h={bounds.height}",
            f"Pitch: {geometry.pitch_x:.2f} x {geometry.pitch_y:.2f}",
            f"Active cells: {len(geometry.cells)}",
            f"Confidence: {geometry.confidence:.3f}",
            "Topology:",
            geometry.topology_text(),
        )
    )


def save_board_overlay(
    image_rgb: UInt8Image,
    geometry: BoardGeometry,
    output: Path,
) -> Path:
    """Write the standard board overlay and return its path."""
    overlay = draw_board_overlay(image_rgb, geometry)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return output
