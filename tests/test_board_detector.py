"""Synthetic regression tests for automatic board geometry detection."""

from __future__ import annotations

import cv2
import numpy as np

from skydom_bot.domain.board import Point, Rect
from skydom_bot.vision.board_detector import BoardDetector, PitchDiagnostics


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
    assert diagnostics.center_tile_evidence.shape == diagnostics.occupancy.shape
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

    labels, sizes, means, strong_fractions, scale, local_votes, local_scaled, selected = detector._select_primary_topology(
        topology,
        occupancy,
    )

    assert max(sizes) == 81
    assert max(means) >= 0.99
    assert max(strong_fractions) >= 0.99
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

    labels, sizes, means, strong_fractions, scale, local_votes, local_scaled, selected = detector._select_primary_topology(
        topology,
        occupancy,
    )

    assert sorted(sizes) == [1, 15]
    assert min(means) > 0.90
    assert min(strong_fractions) == 1.0
    assert bool(selected[0, 4])


def test_assisted_reconciliation_recovers_blocked_edge_cell_with_tile_evidence() -> None:
    detector = BoardDetector()
    occupancy = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.00, 0.92, 0.31],
            [0.00, 0.97, 0.99],
        ],
        dtype=np.float32,
    )
    center = np.array(
        [
            [0.00, 0.00, 0.00],
            [0.00, 0.90, 0.82],
            [0.00, 0.88, 0.91],
        ],
        dtype=np.float32,
    )

    state, support, topology = detector._reconcile_topology(occupancy, center)

    assert int(state[1, 2]) == 1
    assert int(support[1, 2]) == 2
    assert bool(topology[1, 2])


def test_assisted_reconciliation_does_not_fill_gap_without_tile_evidence() -> None:
    detector = BoardDetector()
    occupancy = np.array(
        [
            [0.95, 0.95, 0.95],
            [0.95, 0.32, 0.00],
            [0.95, 0.95, 0.00],
        ],
        dtype=np.float32,
    )
    center = np.zeros_like(occupancy)

    _, support, topology = detector._reconcile_topology(occupancy, center)

    assert int(support[1, 1]) >= 2
    assert not bool(topology[1, 1])


def test_small_island_with_one_weak_blocked_cell_is_preserved() -> None:
    detector = BoardDetector()
    topology = np.zeros((7, 9), dtype=np.bool_)
    topology[2:7, 3:6] = True
    topology[4, 1] = True
    topology[5:7, 0:2] = True

    occupancy = np.zeros((7, 9), dtype=np.float32)
    occupancy[topology] = 0.93

    # Recreate the observed failure: a legitimate 5-cell island has one weak
    # blocker-covered cell, pulling its mean below the old 0.80 cutoff while
    # four of five cells remain individually strong.
    occupancy[4, 1] = 0.18

    labels, sizes, means, strong_fractions, scale, local_votes, local_scaled, selected = detector._select_primary_topology(
        topology,
        occupancy,
    )

    small_label = 1 + sizes.index(5)
    assert means[small_label - 1] < 0.80
    assert strong_fractions[small_label - 1] >= 0.60
    assert bool(selected[4, 1])
    assert bool(selected[6, 1])



def _draw_grid_region(
    image: np.ndarray,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    pitch: int,
) -> None:
    """Draw repeated high-contrast grid boundaries for scale tests."""
    image[y0 : y0 + height, x0 : x0 + width] = (48, 52, 125)
    for x in range(x0, x0 + width + 1, pitch):
        cv2.line(image, (x, y0), (x, y0 + height - 1), (230, 230, 230), 2)
    for y in range(y0, y0 + height + 1, pitch):
        cv2.line(image, (x0, y), (x0 + width - 1, y), (230, 230, 230), 2)


def test_high_evidence_half_scale_opponent_grid_is_rejected() -> None:
    detector = BoardDetector()
    pitch = 60
    rows, cols = 9, 14
    bounds = Rect(0, 0, cols * pitch, rows * pitch)
    image = np.full((bounds.height, bounds.width, 3), 225, dtype=np.uint8)

    topology = np.zeros((rows, cols), dtype=np.bool_)
    topology[:, 5:14] = True
    topology[4:8, 0:4] = True

    occupancy = np.zeros((rows, cols), dtype=np.float32)
    occupancy[topology] = 0.95

    # The player board repeats at the expected 60 px pitch.
    _draw_grid_region(
        image,
        x0=5 * pitch,
        y0=0,
        width=9 * pitch,
        height=9 * pitch,
        pitch=pitch,
    )

    # The opponent preview occupies four projected player cells but internally
    # repeats every 30 px. Its occupancy is deliberately high, reproducing the
    # aliasing case that defeated the old strong-fraction heuristic.
    _draw_grid_region(
        image,
        x0=0,
        y0=4 * pitch,
        width=4 * pitch,
        height=4 * pitch,
        pitch=pitch // 2,
    )

    _, sizes, _, strong_fractions, scale, local_votes, local_scaled, selected = detector._select_primary_topology(
        topology,
        occupancy,
        image_rgb=image,
        bounds=bounds,
        pitch_x=float(pitch),
        pitch_y=float(pitch),
    )

    mini_label = 1 + sizes.index(16)
    mini_scale = scale[mini_label - 1]

    assert strong_fractions[mini_label - 1] == 1.0
    assert mini_scale.status == "scaled-replica"
    assert mini_scale.estimated_scale_ratio == 0.5
    assert not bool(selected[5, 1])
    assert bool(selected[5, 6])


def test_one_axis_scale_conflict_is_ambiguous_and_kept() -> None:
    detector = BoardDetector()
    pitch = 60
    bounds = Rect(0, 0, 9 * pitch, 7 * pitch)
    image = np.full((bounds.height, bounds.width, 3), 225, dtype=np.uint8)

    topology = np.zeros((7, 9), dtype=np.bool_)
    topology[2:7, 3:8] = True
    topology[0:4, 0:2] = True

    occupancy = np.zeros((7, 9), dtype=np.float32)
    occupancy[topology] = 0.95

    # Only the vertical direction contains a half-pitch pattern. The horizontal
    # span is too narrow to prove scale independently, so the component must be
    # surfaced as ambiguous rather than rejected.
    _draw_grid_region(
        image,
        x0=0,
        y0=0,
        width=2 * pitch,
        height=4 * pitch,
        pitch=pitch // 2,
    )
    _, sizes, _, _, scale, local_votes, local_scaled, selected = detector._select_primary_topology(
        topology,
        occupancy,
        image_rgb=image,
        bounds=bounds,
        pitch_x=float(pitch),
        pitch_y=float(pitch),
    )

    small_label = 1 + sizes.index(8)
    small_scale = scale[small_label - 1]

    assert small_scale.status in {"ambiguous", "insufficient-evidence"}
    assert bool(selected[1, 0])



def test_human_anchor_rejects_ambiguous_secondary_grid_for_match_fallback() -> None:
    detector = BoardDetector()
    pitch = 60
    rows, cols = 9, 14
    bounds = Rect(0, 0, cols * pitch, rows * pitch)
    image = np.full((bounds.height, bounds.width, 3), 225, dtype=np.uint8)

    topology = np.zeros((rows, cols), dtype=np.bool_)
    topology[:, 5:14] = True
    topology[4:8, 0:4] = True

    occupancy = np.zeros((rows, cols), dtype=np.float32)
    occupancy[topology] = 0.95

    _draw_grid_region(
        image,
        x0=5 * pitch,
        y0=0,
        width=9 * pitch,
        height=9 * pitch,
        pitch=pitch,
    )

    # Secondary region: half-pitch vertically aligned grid lines on X, but
    # normal player-pitch horizontal lines on Y. Automatic evidence therefore
    # conflicts across axes and must remain conservative.
    x0, y0 = 0, 4 * pitch
    width = height = 4 * pitch
    image[y0 : y0 + height, x0 : x0 + width] = (48, 52, 125)
    for x in range(x0, x0 + width + 1, pitch // 2):
        cv2.line(image, (x, y0), (x, y0 + height - 1), (230, 230, 230), 2)
    for y in range(y0, y0 + height + 1, pitch):
        cv2.line(image, (x0, y), (x0 + width - 1, y), (230, 230, 230), 2)

    _, sizes, _, _, auto_scale, auto_votes, auto_local_scaled, auto_selected = detector._select_primary_topology(
        topology,
        occupancy,
        image_rgb=image,
        bounds=bounds,
        pitch_x=float(pitch),
        pitch_y=float(pitch),
    )
    mini_label = 1 + sizes.index(16)

    assert auto_scale[mini_label - 1].status == "ambiguous"
    assert bool(auto_selected[5, 1])

    _, _, _, _, manual_scale, manual_votes, manual_local_scaled, manual_selected = detector._select_primary_topology(
        topology,
        occupancy,
        image_rgb=image,
        bounds=bounds,
        pitch_x=float(pitch),
        pitch_y=float(pitch),
        preferred_board_point=Point(6 * pitch + 10, pitch + 10),
    )

    assert manual_scale[mini_label - 1].status == "ambiguous"
    assert not bool(manual_selected[5, 1])
    assert bool(manual_selected[5, 6])



def test_connected_half_scale_region_is_removed_by_local_scale_voting() -> None:
    detector = BoardDetector()
    pitch = 60
    rows, cols = 9, 14
    bounds = Rect(0, 0, cols * pitch, rows * pitch)
    image = np.full((bounds.height, bounds.width, 3), 225, dtype=np.uint8)

    # Reproduce the real failure mode: the projected mini-board touches the
    # player board, so connected-component analysis sees one single component.
    topology = np.zeros((rows, cols), dtype=np.bool_)
    topology[:, 5:14] = True
    topology[5:9, 0:5] = True

    occupancy = np.zeros((rows, cols), dtype=np.float32)
    occupancy[topology] = 0.95

    _draw_grid_region(
        image,
        x0=5 * pitch,
        y0=0,
        width=9 * pitch,
        height=9 * pitch,
        pitch=pitch,
    )
    _draw_grid_region(
        image,
        x0=0,
        y0=5 * pitch,
        width=5 * pitch,
        height=4 * pitch,
        pitch=pitch // 2,
    )

    (
        labels,
        sizes,
        _,
        _,
        _,
        local_votes,
        local_scaled,
        selected,
    ) = detector._select_primary_topology(
        topology,
        occupancy,
        image_rgb=image,
        bounds=bounds,
        pitch_x=float(pitch),
        pitch_y=float(pitch),
    )

    assert len(sizes) == 1
    assert sizes[0] == int(np.count_nonzero(topology))
    assert int(labels[6, 2]) == int(labels[6, 6]) == 1
    assert int(local_votes[6, 2]) >= detector.config.local_scale_min_votes
    assert bool(local_scaled[6, 2])
    assert not bool(selected[6, 2])
    assert bool(selected[6, 6])



def test_local_subgrid_seed_expands_through_low_vote_boundary_cells() -> None:
    detector = BoardDetector()

    topology = np.ones((4, 6), dtype=np.bool_)
    votes = np.array(
        [
            [4, 4, 2, 2, 0, 0],
            [6, 6, 3, 3, 0, 0],
            [5, 5, 3, 3, 0, 0],
            [3, 3, 2, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    opportunities = np.full_like(votes, 6, dtype=np.uint8)
    required = np.maximum(
        detector.config.local_scale_min_votes,
        np.ceil(
            opportunities.astype(np.float32)
            * detector.config.local_scale_vote_ratio
        ).astype(np.uint8),
    )
    seeds = topology & (votes >= required)
    support = topology & (
        votes >= detector.config.local_scale_expand_min_vote
    )

    count, labels = cv2.connectedComponents(
        support.astype(np.uint8),
        connectivity=4,
    )
    scaled = seeds.copy()
    for label in range(1, count):
        component = labels == label
        size = int(np.count_nonzero(component))
        seed_count = int(np.count_nonzero(seeds & component))
        if (
            seed_count > 0
            and seed_count / size
            >= detector.config.local_scale_expand_min_seed_fraction
        ):
            scaled |= component

    assert bool(scaled[0, 2])
    assert bool(scaled[2, 3])
    assert not bool(scaled[0, 4])
    assert not bool(scaled[3, 3])



def _pitch_diag(axis: str, peaks: tuple[int, ...], values: dict[int, float]):
    size = max(values) + 2
    scores = np.full(size, np.nan, dtype=np.float64)
    profile = np.zeros(size + 160, dtype=np.float32)
    for lag, value in values.items():
        scores[lag] = value
    return PitchDiagnostics(
        axis=axis,
        edge_energy=np.zeros((1, 1), dtype=np.float32),
        profile=profile,
        lags=np.arange(len(scores), dtype=np.int32),
        scores=scores,
        peaks=peaks,
        selected_pitch=float(min(peaks)),
    )


def test_shared_pitch_rejects_one_axis_alias_and_uses_cross_axis_support() -> None:
    detector = BoardDetector()

    x = _pitch_diag(
        "x",
        (58, 65, 130),
        {58: 0.33, 65: 0.44, 130: 0.35},
    )
    y = _pitch_diag(
        "y",
        (65, 131),
        {58: 0.25, 65: 0.40, 130: 0.46, 131: 0.47},
    )

    # Populate profiles so _autocorrelation_at has deterministic synthetic
    # periodic support at the intended lags.
    rng = np.random.default_rng(42)
    base = rng.normal(size=420).astype(np.float32)
    for diag, period in ((x, 65), (y, 65)):
        pattern = np.resize(base[:period], diag.profile.shape)
        diag.profile[:] = pattern

    assert detector._shared_pitch(x, y) == 65.0


def test_shared_pitch_prefers_fundamental_when_double_harmonic_is_supported() -> None:
    detector = BoardDetector()

    x = _pitch_diag("x", (60, 120), {60: 0.42, 120: 0.48})
    y = _pitch_diag("y", (60, 120), {60: 0.40, 120: 0.46})

    rng = np.random.default_rng(7)
    period = 60
    pattern = rng.normal(size=period).astype(np.float32)
    x.profile[:] = np.resize(pattern, x.profile.shape)
    y.profile[:] = np.resize(pattern, y.profile.shape)

    assert detector._shared_pitch(x, y) == 60.0



def test_sparse_board_group_can_be_selected_by_spatial_footprint() -> None:
    detector = BoardDetector()
    image = np.full((900, 1600, 3), 225, dtype=np.uint8)
    board_color = (52, 55, 133)

    # Nine small islands are close enough to be grouped. Their combined masked
    # area stays below the dense-component threshold, while the group footprint
    # is large enough to describe a plausible sparse playfield.
    size = 45
    gap = 55
    step = size + gap
    for row in range(3):
        for col in range(3):
            x = 420 + col * step
            y = 180 + row * step
            cv2.rectangle(
                image,
                (x, y),
                (x + size - 1, y + size - 1),
                board_color,
                -1,
            )

    mask = detector._board_background_mask(image)
    bounds, component = detector._board_component(mask)

    image_area = image.shape[0] * image.shape[1]
    masked_area = int(np.count_nonzero(component))

    assert masked_area < image_area * detector.config.min_component_area_ratio
    assert bounds.width * bounds.height >= (
        image_area * detector.config.min_component_footprint_ratio
    )
    assert bounds.width >= 240
    assert bounds.height >= 240
