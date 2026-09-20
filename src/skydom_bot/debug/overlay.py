"""Diagnostic overlays for computer-vision development."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from skydom_bot.domain.board import BoardGeometry

UInt8Image = NDArray[np.uint8]


def draw_board_overlay(image_rgb: UInt8Image, geometry: BoardGeometry) -> UInt8Image:
    """Return an RGB copy annotated with board bounds, cells, and confidence."""
    canvas = image_rgb.copy()
    bounds = geometry.bounds
    cv2.rectangle(canvas, (bounds.x, bounds.y), (bounds.right - 1, bounds.bottom - 1), (255, 255, 255), 2)

    for cell in geometry.cells:
        cv2.rectangle(
            canvas,
            (cell.bounds.x, cell.bounds.y),
            (cell.bounds.right - 1, cell.bounds.bottom - 1),
            (255, 255, 255),
            1,
        )
        cv2.circle(canvas, (cell.center.x, cell.center.y), 3, (255, 255, 255), -1)
        cv2.putText(
            canvas,
            f"{cell.row},{cell.col}",
            (cell.bounds.x + 3, cell.bounds.y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"{geometry.rows}x{geometry.cols}  cells={len(geometry.cells)}  conf={geometry.confidence:.2f}",
        (bounds.x, max(16, bounds.y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas
