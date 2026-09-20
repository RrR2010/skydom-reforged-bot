"""Synthetic regression tests for automatic board geometry detection."""

from __future__ import annotations

import cv2
import numpy as np

from skydom_bot.vision.board_detector import BoardDetector


def _synthetic_board(rows: int = 8, cols: int = 7, pitch: int = 60) -> np.ndarray:
    image = np.full((620, 1000, 3), 225, dtype=np.uint8)
    x0, y0 = 280, 50
    active = {(r, c) for r in range(rows - 1) for c in range(cols)}
    active.update((rows - 1, c) for c in range(1, cols - 1))

    # Two close dark-blue shades recreate the repeated cell boundaries used by
    # pitch autocorrelation while all cells stay inside the HSV board mask.
    colors = ((52, 55, 133), (44, 48, 117))
    for row, col in active:
        xa, ya = x0 + col * pitch, y0 + row * pitch
        cv2.rectangle(image, (xa, ya), (xa + pitch - 1, ya + pitch - 1), colors[(row + col) % 2], -1)
        cv2.circle(image, (xa + pitch // 2, ya + pitch // 2), pitch // 4, (230, 70, 10), -1)
    return image


def test_detects_irregular_board_without_hard_coded_dimensions() -> None:
    geometry = BoardDetector().detect(_synthetic_board())

    assert geometry.rows == 8
    assert geometry.cols == 7
    assert len(geometry.cells) == 54
    assert not geometry.has_cell(7, 0)
    assert not geometry.has_cell(7, 6)
    assert geometry.has_cell(7, 1)
    assert geometry.has_cell(7, 5)
    assert abs(geometry.pitch_x - 60) < 1.5
    assert abs(geometry.pitch_y - 60) < 1.5
