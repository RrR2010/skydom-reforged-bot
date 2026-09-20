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

    # Two close dark-blue shades recreate repeated cell boundaries while all
    # cells remain inside the HSV board mask.
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


def test_large_piece_does_not_make_a_real_cell_disappear() -> None:
    image = _synthetic_board()
    pitch = 60
    x0, y0 = 280, 50
    row, col = 3, 4
    xa, ya = x0 + col * pitch, y0 + row * pitch

    cv2.rectangle(image, (xa + 6, ya + 6), (xa + pitch - 7, ya + pitch - 7), (245, 95, 5), -1)

    geometry = BoardDetector().detect(image)

    assert geometry.has_cell(row, col)
    assert len(geometry.cells) == 54


def test_fragmented_board_is_grouped_across_ice_like_occlusion() -> None:
    image = _synthetic_board(rows=9, cols=9, pitch=60)
    x0, y0, pitch = 280, 50, 60

    # Paint a cyan blocker across several central cells. This deliberately
    # disconnects the normal dark-blue background into multiple components.
    for row in (1, 2, 3):
        for col in (3, 4, 5):
            xa, ya = x0 + col * pitch, y0 + row * pitch
            cv2.rectangle(image, (xa, ya), (xa + pitch - 1, ya + pitch - 1), (90, 220, 240), -1)
            cv2.rectangle(image, (xa + 8, ya + 8), (xa + pitch - 9, ya + pitch - 9), (230, 70, 10), -1)

    geometry = BoardDetector().detect(image)

    assert geometry.rows == 9
    assert geometry.cols == 9
    assert geometry.has_cell(2, 4)


def test_pitch_prefers_fundamental_over_two_cell_harmonic() -> None:
    detector = BoardDetector()
    image = _synthetic_board(rows=8, cols=8, pitch=60)

    geometry = detector.detect(image)

    assert abs(geometry.pitch_x - 60) < 1.5
    assert abs(geometry.pitch_y - 60) < 1.5


def test_diagnostics_expose_pitch_signals_and_cell_evidence() -> None:
    detector = BoardDetector()
    image = _synthetic_board(rows=8, cols=8, pitch=60)

    geometry, diagnostics = detector.detect_with_diagnostics(image)

    assert diagnostics.crop_rgb.shape[:2] == (geometry.bounds.height, geometry.bounds.width)
    assert diagnostics.pitch_x.selected_pitch > 0
    assert diagnostics.pitch_y.selected_pitch > 0
    assert diagnostics.pitch_x.profile.ndim == 1
    assert diagnostics.pitch_y.profile.ndim == 1
    assert diagnostics.occupancy.shape == (geometry.rows, geometry.cols)
    assert float(diagnostics.occupancy.max()) <= 1.0
    assert float(diagnostics.occupancy.min()) >= 0.0
