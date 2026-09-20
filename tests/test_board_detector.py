"""Synthetic regression tests for automatic board geometry detection."""

from __future__ import annotations

import cv2
import numpy as np

from skydom_bot.domain.board import Rect
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
    assert diagnostics.evidence_state.shape == diagnostics.occupancy.shape
    assert diagnostics.cardinal_support.shape == diagnostics.occupancy.shape
    assert diagnostics.reconciled_topology.shape == diagnostics.occupancy.shape


def test_structural_reconciliation_promotes_only_surrounded_uncertain_cell() -> None:
    detector = BoardDetector()
    occupancy = np.array(
        [
            [0.00, 0.90, 0.00],
            [0.90, 0.42, 0.90],
            [0.00, 0.90, 0.00],
        ],
        dtype=np.float32,
    )

    state, support, topology = detector._reconcile_topology(occupancy)

    assert int(state[1, 1]) == 1
    assert int(support[1, 1]) == 4
    assert bool(topology[1, 1])


def test_structural_reconciliation_does_not_fill_irregular_edge_gap() -> None:
    detector = BoardDetector()
    occupancy = np.array(
        [
            [0.95, 0.95, 0.95],
            [0.95, 0.32, 0.00],
            [0.95, 0.29, 0.00],
        ],
        dtype=np.float32,
    )

    state, support, topology = detector._reconcile_topology(occupancy)

    assert int(state[1, 1]) == 1
    assert int(support[1, 1]) == 2
    assert not bool(topology[1, 1])


def test_secondary_mini_board_is_rejected_and_primary_grid_is_normalized() -> None:
    detector = BoardDetector()
    topology = np.zeros((9, 14), dtype=np.bool_)

    # Main 9x9 player board occupies columns 5..13 and is one large component.
    topology[:, 5:14] = True

    # Mini opponent preview projects sparsely on the player's larger grid scale.
    topology[4:8, 0:2] = True
    topology[8, 1:4] = True

    occupancy = topology.astype(np.float32)
    # Mini-board evidence is weaker because its smaller cells are sampled on
    # the player's larger logical pitch.
    occupancy[4:8, 0:2] = 0.45
    occupancy[8, 1:4] = 0.45

    labels, sizes, means, selected = detector._select_primary_topology(topology, occupancy)

    assert max(sizes) == 81
    assert max(means) >= 0.99
    assert int(np.count_nonzero(selected)) == 81
    assert not bool(selected[5, 0])
    assert bool(selected[5, 5])

    occupancy = selected.astype(np.float32)
    bounds = Rect(260, 206, 915, 596)
    final_bounds, final_occupancy, final_topology, pitch_x, pitch_y = detector._crop_to_selected_topology(
        bounds,
        occupancy,
        selected,
        915 / 14,
        596 / 9,
    )

    assert final_topology.shape == (9, 9)
    assert final_occupancy.shape == (9, 9)
    assert final_bounds.x > bounds.x
    assert abs(pitch_x - 65.36) < 0.2
    assert abs(pitch_y - 66.22) < 0.2


def test_sparse_disconnected_islands_can_form_one_board_candidate() -> None:
    detector = BoardDetector()
    image = np.full((900, 1600, 3), 225, dtype=np.uint8)
    pitch = 60
    board_color = (52, 55, 133)

    # Main block.
    for row in range(2, 7):
        for col in range(3, 6):
            x = 500 + col * pitch
            y = 100 + row * pitch
            cv2.rectangle(image, (x, y), (x + pitch - 1, y + pitch - 1), board_color, -1)

    # Detached one-cell and two-cell islands one pitch away from the main block.
    islands = [
        (0, 4),
        (2, 0),
        (2, 1),
        (2, 7),
        (2, 8),
        (4, 0),
        (4, 1),
        (4, 7),
        (4, 8),
    ]
    for row, col in islands:
        x = 500 + col * pitch
        y = 100 + row * pitch
        cv2.rectangle(image, (x, y), (x + pitch - 1, y + pitch - 1), board_color, -1)

    geometry = detector.detect(image)

    assert geometry.cols >= 9
    assert len(geometry.cells) >= 15


def test_small_high_evidence_island_is_preserved() -> None:
    detector = BoardDetector()
    topology = np.zeros((7, 9), dtype=np.bool_)
    topology[2:7, 3:6] = True
    topology[0, 4] = True

    occupancy = np.zeros((7, 9), dtype=np.float32)
    occupancy[topology] = 0.96

    labels, sizes, means, selected = detector._select_primary_topology(topology, occupancy)

    assert sorted(sizes) == [1, 15]
    assert min(means) > 0.90
    assert bool(selected[0, 4])
