"""Automatic detection of regular-but-possibly-irregular Match-3 boards."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from skydom_bot.domain.board import BoardGeometry, Cell, Point, Rect

UInt8Image = NDArray[np.uint8]
FloatImage = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class BoardDetectorConfig:
    """Tunable thresholds for Skydom board geometry detection."""

    hue_min: int = 108
    hue_max: int = 132
    saturation_min: int = 90
    saturation_max: int = 230
    value_min: int = 55
    value_max: int = 190

    # Ice uses a cyan/light-blue overlay that can fully hide the normal board
    # background around a cell. Treat it as alternate board-surface evidence.
    ice_hue_min: int = 90
    ice_hue_max: int = 110
    ice_saturation_min: int = 35
    ice_value_min: int = 100

    min_component_area_ratio: float = 0.015
    min_component_footprint_ratio: float = 0.030
    min_fragment_area_ratio: float = 0.001
    component_join_gap_ratio: float = 0.07
    component_projection_overlap: float = 0.50

    pitch_min_px: int = 28
    pitch_max_px: int = 140
    pitch_peak_ratio: float = 0.75

    occupancy_threshold: float = 0.50
    occupancy_uncertain_threshold: float = 0.25
    structural_required_cardinal_neighbors: int = 4
    assisted_required_cardinal_neighbors: int = 2
    center_tile_radius_ratio: float = 0.30
    center_tile_saturation_min: int = 105
    center_tile_value_min: int = 125
    center_tile_evidence_threshold: float = 0.45
    secondary_component_keep_ratio: float = 0.35
    secondary_component_min_mean_evidence: float = 0.80
    secondary_component_min_strong_fraction: float = 0.60

    # Competitive modes can render a miniature opponent board next to the
    # player board. At exact 1:2 or 1:3 scale ratios, several miniature cells
    # can alias into one logical player cell and still produce strong occupancy.
    # Validate secondary components for repeated sub-grid structure before
    # accepting high occupancy as same-scale board evidence.
    secondary_component_scale_min_cells: int = 6
    secondary_component_scale_min_span_cells: float = 3.0
    secondary_component_subgrid_divisors: tuple[int, ...] = (2, 3)
    secondary_component_subgrid_min_score: float = 0.30
    secondary_component_subgrid_relative_score: float = 0.85
    secondary_component_subgrid_lag_tolerance_px: int = 2

    # A miniature board can become 4-connected to the player board after
    # projection onto the player's logical grid. Detect scale locally in
    # overlapping neighborhoods so a P/2 or P/3 region can still be separated
    # even when connected-component analysis sees only one component.
    local_scale_window_radius: int = 1
    local_scale_min_active_cells: int = 4
    local_scale_min_votes: int = 2
    local_scale_vote_ratio: float = 0.50
    local_scale_expand_min_vote: int = 1
    local_scale_expand_min_seed_fraction: float = 0.20

    cell_corner_ratio: float = 0.16


@dataclass(frozen=True, slots=True)
class PitchDiagnostics:
    """Intermediate 1D signals used to estimate one grid axis."""

    axis: str
    edge_energy: FloatImage
    profile: NDArray[np.float32]
    lags: NDArray[np.int32]
    scores: NDArray[np.float64]
    peaks: tuple[int, ...]
    selected_pitch: float


@dataclass(frozen=True, slots=True)
class ComponentScaleDiagnostics:
    """Scale-consistency evidence for one logical topology component."""

    label: int
    cells: int
    axes_tested: int
    expected_scores: tuple[float, ...]
    subgrid_scores: tuple[float, ...]
    subgrid_divisors: tuple[int | None, ...]
    estimated_scale_ratio: float | None
    status: str


@dataclass(frozen=True, slots=True)
class BoardDetectionDiagnostics:
    """Intermediate images and arrays for visual inspection of detection."""

    board_mask: UInt8Image
    component_mask: UInt8Image
    ice_mask: UInt8Image
    evidence_mask: UInt8Image
    bounds: Rect
    crop_rgb: UInt8Image
    crop_gray: FloatImage
    pitch_x: PitchDiagnostics
    pitch_y: PitchDiagnostics
    shared_pitch: float
    occupancy: NDArray[np.float32]
    evidence_state: NDArray[np.uint8]
    cardinal_support: NDArray[np.uint8]
    center_tile_evidence: NDArray[np.float32]
    reconciled_topology: NDArray[np.bool_]
    topology_components: NDArray[np.int32]
    topology_component_sizes: tuple[int, ...]
    topology_component_mean_evidence: tuple[float, ...]
    topology_component_strong_fraction: tuple[float, ...]
    topology_component_scale: tuple[ComponentScaleDiagnostics, ...]
    ambiguous_topology_components: tuple[int, ...]
    local_subgrid_votes: NDArray[np.uint8]
    local_subgrid_topology: NDArray[np.bool_]
    selected_topology: NDArray[np.bool_]


class BoardDetectionError(RuntimeError):
    """Raised when a board cannot be inferred with sufficient confidence."""


class BoardDetector:
    """Detect board bounds, cell pitch, dimensions, and active cells."""

    def __init__(self, config: BoardDetectorConfig | None = None) -> None:
        self.config = config or BoardDetectorConfig()

    def detect(
        self,
        image_rgb: UInt8Image,
        *,
        preferred_board_point: Point | None = None,
    ) -> BoardGeometry:
        """Detect board geometry from an RGB image.

        preferred_board_point is an optional human-selection hook. A match-level
        UI may ask the player to click the correct board once, retain that pixel
        point for the match, and pass it on subsequent detections.
        """
        geometry, _ = self.detect_with_diagnostics(
            image_rgb,
            preferred_board_point=preferred_board_point,
        )
        return geometry

    def detect_with_diagnostics(
        self,
        image_rgb: UInt8Image,
        *,
        preferred_board_point: Point | None = None,
    ) -> tuple[BoardGeometry, BoardDetectionDiagnostics]:
        """Detect geometry and preserve the intermediate perception pipeline.

        This method exists primarily for observability. The production detector
        and the visual debugger execute the exact same operations, so a debug
        plot cannot silently drift away from the real algorithm.
        """
        self._validate_image(image_rgb)

        board_mask = self._board_background_mask(image_rgb)
        bounds, component_mask = self._board_component(board_mask)

        crop = image_rgb[bounds.y : bounds.bottom, bounds.x : bounds.right]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        pitch_x_diag = self._pitch_diagnostics(crop, axis=1)
        pitch_y_diag = self._pitch_diagnostics(crop, axis=0)

        # Both axes describe the same square-cell grid. Select one shared
        # fundamental instead of averaging two independently chosen peaks.
        # This rejects one-axis aliases such as X=58 when both axes support 65
        # and 130 is merely the 2x harmonic.
        pitch = self._shared_pitch(pitch_x_diag, pitch_y_diag)
        cols = max(1, int(round(bounds.width / pitch)))
        rows = max(1, int(round(bounds.height / pitch)))
        pitch_x = bounds.width / cols
        pitch_y = bounds.height / rows

        ice_mask = self._ice_surface_mask(image_rgb)
        evidence_mask = cv2.bitwise_or(component_mask, ice_mask)
        occupancy = self._cell_occupancy_grid(evidence_mask, bounds, rows, cols, pitch_x, pitch_y)
        center_tile_evidence = self._center_tile_evidence_grid(
            image_rgb,
            bounds,
            rows,
            cols,
            pitch_x,
            pitch_y,
        )
        evidence_state, cardinal_support, reconciled_topology = self._reconcile_topology(
            occupancy,
            center_tile_evidence,
        )
        (
            topology_components,
            component_sizes,
            component_mean_evidence,
            component_strong_fraction,
            component_scale,
            local_subgrid_votes,
            local_subgrid_topology,
            selected_topology,
        ) = self._select_primary_topology(
            reconciled_topology,
            occupancy,
            image_rgb=image_rgb,
            bounds=bounds,
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            preferred_board_point=preferred_board_point,
        )

        (
            final_bounds,
            final_occupancy,
            final_topology,
            final_pitch_x,
            final_pitch_y,
        ) = self._crop_to_selected_topology(
            bounds,
            occupancy,
            selected_topology,
            pitch_x,
            pitch_y,
        )
        cells = self._cells_from_topology(
            final_occupancy,
            final_topology,
            final_bounds,
            final_pitch_x,
            final_pitch_y,
        )
        if not cells:
            raise BoardDetectionError("Board component found, but no active cells were inferred.")

        confidence = self._confidence(final_bounds, final_pitch_x, final_pitch_y, cells)
        geometry = BoardGeometry(
            bounds=final_bounds,
            rows=final_topology.shape[0],
            cols=final_topology.shape[1],
            pitch_x=final_pitch_x,
            pitch_y=final_pitch_y,
            cells=tuple(cells),
            confidence=confidence,
        )
        diagnostics = BoardDetectionDiagnostics(
            board_mask=board_mask,
            component_mask=component_mask,
            ice_mask=ice_mask,
            evidence_mask=evidence_mask,
            bounds=bounds,
            crop_rgb=crop,
            crop_gray=crop_gray,
            pitch_x=pitch_x_diag,
            pitch_y=pitch_y_diag,
            shared_pitch=pitch,
            occupancy=occupancy,
            evidence_state=evidence_state,
            cardinal_support=cardinal_support,
            center_tile_evidence=center_tile_evidence,
            reconciled_topology=reconciled_topology,
            topology_components=topology_components,
            topology_component_sizes=component_sizes,
            topology_component_mean_evidence=component_mean_evidence,
            topology_component_strong_fraction=component_strong_fraction,
            topology_component_scale=component_scale,
            ambiguous_topology_components=tuple(
                item.label for item in component_scale if item.status == "ambiguous"
            ),
            local_subgrid_votes=local_subgrid_votes,
            local_subgrid_topology=local_subgrid_topology,
            selected_topology=selected_topology,
        )
        return geometry, diagnostics

    @staticmethod
    def _validate_image(image_rgb: UInt8Image) -> None:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 image with shape (height, width, 3).")

    def _board_background_mask(self, image_rgb: UInt8Image) -> UInt8Image:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        c = self.config
        return cv2.inRange(
            hsv,
            np.array([c.hue_min, c.saturation_min, c.value_min], dtype=np.uint8),
            np.array([c.hue_max, c.saturation_max, c.value_max], dtype=np.uint8),
        )

    def _ice_surface_mask(self, image_rgb: UInt8Image) -> UInt8Image:
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        c = self.config
        return cv2.inRange(
            hsv,
            np.array([c.ice_hue_min, c.ice_saturation_min, c.ice_value_min], dtype=np.uint8),
            np.array([c.ice_hue_max, 255, 255], dtype=np.uint8),
        )

    def _board_component(self, mask: UInt8Image) -> tuple[Rect, UInt8Image]:
        """Group nearby board-colored fragments without painting over occlusion.

        Some levels deliberately split the playable board into separate islands.
        Those islands can be one full cell pitch apart and individually much
        smaller than the whole board. We therefore admit smaller fragments and
        join aligned neighbors across a larger gap. Later logical-topology
        selection rejects unrelated or miniature boards more safely than an
        aggressive early area threshold can.
        """
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            raise BoardDetectionError("No board-colored connected component was found.")

        image_area = mask.shape[0] * mask.shape[1]
        fragment_min_area = image_area * self.config.min_fragment_area_ratio
        fragments = [
            label
            for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= fragment_min_area
        ]
        if not fragments:
            raise BoardDetectionError("No plausible board fragments were found.")

        parent = {label: label for label in fragments}

        def find(label: int) -> int:
            while parent[label] != label:
                parent[label] = parent[parent[label]]
                label = parent[label]
            return label

        def union(a: int, b: int) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        def interval_gap(a0: int, a1: int, b0: int, b1: int) -> int:
            if a1 < b0:
                return b0 - a1
            if b1 < a0:
                return a0 - b1
            return 0

        max_gap = int(round(min(mask.shape) * self.config.component_join_gap_ratio))
        min_overlap = self.config.component_projection_overlap

        for index, a in enumerate(fragments):
            ax, ay, aw, ah = map(int, stats[a, :4])
            ar, ab = ax + aw, ay + ah

            for b in fragments[index + 1 :]:
                bx, by, bw, bh = map(int, stats[b, :4])
                br, bb = bx + bw, by + bh

                x_overlap = max(0, min(ar, br) - max(ax, bx))
                y_overlap = max(0, min(ab, bb) - max(ay, by))
                x_ratio = x_overlap / max(1, min(aw, bw))
                y_ratio = y_overlap / max(1, min(ah, bh))
                vertical_gap = interval_gap(ay, ab, by, bb)
                horizontal_gap = interval_gap(ax, ar, bx, br)

                if (
                    x_ratio >= min_overlap
                    and vertical_gap <= max_gap
                    or y_ratio >= min_overlap
                    and horizontal_gap <= max_gap
                ):
                    union(a, b)

        groups: dict[int, list[int]] = {}
        for label in fragments:
            groups.setdefault(find(label), []).append(label)

        candidates: list[tuple[int, int, list[int], Rect]] = []
        for group in groups.values():
            area = sum(int(stats[label, cv2.CC_STAT_AREA]) for label in group)

            left = min(int(stats[label, cv2.CC_STAT_LEFT]) for label in group)
            top = min(int(stats[label, cv2.CC_STAT_TOP]) for label in group)
            right = max(
                int(stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH])
                for label in group
            )
            bottom = max(
                int(stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT])
                for label in group
            )
            bounds = Rect(left, top, right - left, bottom - top)
            footprint_area = bounds.width * bounds.height

            # Dense boards contribute enough masked pixels directly. Sparse
            # island layouts may contain little board-colored surface overall
            # while still spanning a large, coherent playfield. Accept either
            # kind of evidence instead of forcing all levels through one fill-
            # ratio assumption.
            dense_enough = (
                area >= image_area * self.config.min_component_area_ratio
            )
            broad_enough = (
                footprint_area
                >= image_area * self.config.min_component_footprint_ratio
            )
            if not dense_enough and not broad_enough:
                continue

            candidates.append((footprint_area, area, group, bounds))

        if not candidates:
            raise BoardDetectionError(
                "No grouped component is large or spatially broad enough to be the board."
            )

        # Footprint is the more stable signal for sparse levels. Pixel area is
        # retained as a deterministic tie-breaker for similarly sized groups.
        _, _, selected_labels, bounds = max(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        component = np.isin(labels, selected_labels).astype(np.uint8) * 255
        return bounds, component

    def _pitch_diagnostics(self, crop_rgb: UInt8Image, axis: int) -> PitchDiagnostics:
        """Build the edge profile and normalized autocorrelation for one axis."""
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        derivative = np.abs(np.diff(gray, axis=axis))
        profile = derivative.mean(axis=1 - axis).astype(np.float32)
        profile -= profile.mean()

        maximum = min(self.config.pitch_max_px, max(self.config.pitch_min_px + 1, len(profile) // 3))
        minimum = min(self.config.pitch_min_px, maximum - 1)
        if maximum <= minimum:
            raise BoardDetectionError("Board component is too small to estimate a cell pitch.")

        scores = np.full(maximum + 1, np.nan, dtype=np.float64)
        for lag in range(minimum, maximum + 1):
            left = profile[:-lag]
            right = profile[lag:]
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator > 0:
                scores[lag] = float(np.dot(left, right) / denominator)

        finite = scores[np.isfinite(scores)]
        if finite.size == 0:
            raise BoardDetectionError("Could not calculate grid autocorrelation.")

        peak_floor = float(finite.max()) * self.config.pitch_peak_ratio
        peaks = tuple(
            lag
            for lag in range(minimum + 1, maximum)
            if (
                np.isfinite(scores[lag])
                and scores[lag] >= peak_floor
                and scores[lag] >= scores[lag - 1]
                and scores[lag] >= scores[lag + 1]
            )
        )
        pitch = min(peaks) if peaks else int(np.nanargmax(scores))
        if not np.isfinite(scores[pitch]) or scores[pitch] <= 0:
            raise BoardDetectionError("Could not find a periodic grid signal in the board image.")

        return PitchDiagnostics(
            axis="x" if axis == 1 else "y",
            edge_energy=derivative,
            profile=profile,
            lags=np.arange(len(scores), dtype=np.int32),
            scores=scores,
            peaks=peaks,
            selected_pitch=float(pitch),
        )

    def _shared_pitch(
        self,
        pitch_x: PitchDiagnostics,
        pitch_y: PitchDiagnostics,
    ) -> float:
        """Choose one square-grid pitch jointly from both axis signals.

        Per-axis peak picking can disagree when level geometry creates a strong
        non-grid spacing on only one axis. Candidate lags are clustered from
        both axes, scored by their weakest normalized cross-axis support, and
        then reduced to the smallest well-supported fundamental when a larger
        candidate is an integer harmonic.

        This preserves the existing preference for 60 over 120 while rejecting
        one-axis aliases such as 58 when both axes strongly support 65.
        """
        candidates = sorted(set(pitch_x.peaks) | set(pitch_y.peaks))
        if not candidates:
            candidates = [
                int(round(pitch_x.selected_pitch)),
                int(round(pitch_y.selected_pitch)),
            ]

        def axis_max(diag: PitchDiagnostics) -> float:
            finite = diag.scores[np.isfinite(diag.scores)]
            positive = finite[finite > 0]
            return float(positive.max()) if positive.size else 1.0

        max_x = axis_max(pitch_x)
        max_y = axis_max(pitch_y)
        tolerance = self.config.secondary_component_subgrid_lag_tolerance_px

        scored: list[tuple[float, float, int]] = []
        for candidate in candidates:
            score_x = self._autocorrelation_at(
                pitch_x.profile,
                candidate,
                tolerance=tolerance,
            )
            score_y = self._autocorrelation_at(
                pitch_y.profile,
                candidate,
                tolerance=tolerance,
            )
            norm_x = max(0.0, score_x / max_x)
            norm_y = max(0.0, score_y / max_y)
            weakest = min(norm_x, norm_y)
            mean = (norm_x + norm_y) / 2.0
            scored.append((weakest, mean, candidate))

        strongest_weakest = max(item[0] for item in scored)
        support_floor = strongest_weakest * self.config.pitch_peak_ratio
        supported = [
            item
            for item in scored
            if item[0] >= support_floor and item[0] > 0
        ]
        if not supported:
            supported = [max(scored, key=lambda item: (item[0], item[1]))]

        supported_lags = sorted(item[2] for item in supported)

        # Prefer a smaller candidate only when another supported candidate is
        # clearly its integer harmonic. Mere numerical smallness is not enough.
        for candidate in supported_lags:
            for harmonic in supported_lags:
                if harmonic <= candidate:
                    continue
                ratio = harmonic / candidate
                nearest = round(ratio)
                if nearest >= 2 and abs(ratio - nearest) <= 0.08:
                    return float(candidate)

        best = max(
            supported,
            key=lambda item: (item[0], item[1], -abs(item[2] - np.median(supported_lags))),
        )
        return float(best[2])

    def _estimate_pitch(self, crop_rgb: UInt8Image, axis: int) -> float:
        """Estimate the cell spacing for one axis."""
        return self._pitch_diagnostics(crop_rgb, axis).selected_pitch

    def _corner_occupancy(self, cell_mask: UInt8Image) -> float:
        """Measure board-surface evidence where game pieces rarely occlude it."""
        height, width = cell_mask.shape
        sample_w = max(2, int(round(width * self.config.cell_corner_ratio)))
        sample_h = max(2, int(round(height * self.config.cell_corner_ratio)))
        patches = (
            cell_mask[:sample_h, :sample_w],
            cell_mask[:sample_h, width - sample_w :],
            cell_mask[height - sample_h :, :sample_w],
            cell_mask[height - sample_h :, width - sample_w :],
        )
        pixels = sum(patch.size for patch in patches)
        return float(sum(np.count_nonzero(patch) for patch in patches) / pixels) if pixels else 0.0

    def _cell_occupancy_grid(
        self,
        evidence_mask: UInt8Image,
        bounds: Rect,
        rows: int,
        cols: int,
        pitch_x: float,
        pitch_y: float,
    ) -> NDArray[np.float32]:
        """Return per-cell board-surface evidence before thresholding."""
        local = evidence_mask[bounds.y : bounds.bottom, bounds.x : bounds.right]
        occupancy = np.zeros((rows, cols), dtype=np.float32)

        for row in range(rows):
            for col in range(cols):
                x0 = int(round(col * pitch_x))
                x1 = int(round((col + 1) * pitch_x))
                y0 = int(round(row * pitch_y))
                y1 = int(round((row + 1) * pitch_y))
                occupancy[row, col] = self._corner_occupancy(local[y0:y1, x0:x1])

        return occupancy

    def _center_tile_evidence_grid(
        self,
        image_rgb: UInt8Image,
        bounds: Rect,
        rows: int,
        cols: int,
        pitch_x: float,
        pitch_y: float,
    ) -> NDArray[np.float32]:
        """Measure colorful tile-like evidence near each logical cell center.

        Corner evidence is excellent for normal cells because pieces rarely
        cover the corners, but blockers such as chains can obscure those same
        corners. A second, independent cue samples a conservative center disk
        and asks how much of it is bright and saturated like a game piece.

        This cue never creates cells by itself. It can only assist an already
        uncertain position that also has structural support from neighboring
        cells.
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        local = hsv[bounds.y : bounds.bottom, bounds.x : bounds.right]
        scores = np.zeros((rows, cols), dtype=np.float32)

        for row in range(rows):
            for col in range(cols):
                x0 = int(round(col * pitch_x))
                x1 = int(round((col + 1) * pitch_x))
                y0 = int(round(row * pitch_y))
                y1 = int(round((row + 1) * pitch_y))
                region = local[y0:y1, x0:x1]
                if region.size == 0:
                    continue

                height, width = region.shape[:2]
                radius = max(
                    2,
                    int(
                        round(
                            min(width, height)
                            * self.config.center_tile_radius_ratio
                        )
                    ),
                )
                center = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(center, (width // 2, height // 2), radius, 255, -1)

                colorful = cv2.inRange(
                    region,
                    np.array(
                        [
                            0,
                            self.config.center_tile_saturation_min,
                            self.config.center_tile_value_min,
                        ],
                        dtype=np.uint8,
                    ),
                    np.array([179, 255, 255], dtype=np.uint8),
                )
                sample_pixels = max(1, int(np.count_nonzero(center)))
                scores[row, col] = float(
                    np.count_nonzero(cv2.bitwise_and(center, colorful))
                    / sample_pixels
                )

        return scores

    def _reconcile_topology(
        self,
        occupancy: NDArray[np.float32],
        center_tile_evidence: NDArray[np.float32] | None = None,
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.bool_]]:
        """Reconcile ambiguous visual evidence with conservative grid structure.

        Evidence is intentionally kept ternary before becoming topology:

        - 2 = strong: visual evidence is already sufficient.
        - 1 = uncertain: plausible cell, but below the strong threshold.
        - 0 = absent: too little evidence to infer a cell.

        The primary promotion rule remains deliberately conservative: an
        uncertain cell surrounded by strong cardinal neighbors is accepted.

        A secondary rule handles blockers that obscure the corners used by
        occupancy detection. An uncertain cell with at least two strong
        cardinal neighbors may also be promoted when its center contains strong
        tile-like saturated/bright evidence. This is evidence fusion rather
        than a relaxed global threshold: structure and an independent visual
        cue must agree.
        """
        strong = occupancy >= self.config.occupancy_threshold
        uncertain = (
            (occupancy >= self.config.occupancy_uncertain_threshold)
            & ~strong
        )

        state = np.zeros(occupancy.shape, dtype=np.uint8)
        state[uncertain] = 1
        state[strong] = 2

        rows, cols = occupancy.shape
        support = np.zeros(occupancy.shape, dtype=np.uint8)
        for row in range(rows):
            for col in range(cols):
                neighbors = (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                )
                support[row, col] = sum(
                    1
                    for nr, nc in neighbors
                    if 0 <= nr < rows and 0 <= nc < cols and strong[nr, nc]
                )

        promoted_structural = uncertain & (
            support >= self.config.structural_required_cardinal_neighbors
        )

        assisted = np.zeros_like(strong)
        if center_tile_evidence is not None:
            assisted = (
                uncertain
                & (support >= self.config.assisted_required_cardinal_neighbors)
                & (
                    center_tile_evidence
                    >= self.config.center_tile_evidence_threshold
                )
            )

        reconciled = strong | promoted_structural | assisted
        return state, support, reconciled

    @staticmethod
    def _autocorrelation_at(
        profile: NDArray[np.float32],
        lag: int,
        *,
        tolerance: int,
    ) -> float:
        """Return the strongest normalized autocorrelation near one lag."""
        best = float("-inf")
        for candidate in range(max(1, lag - tolerance), lag + tolerance + 1):
            if candidate >= len(profile):
                continue
            left = profile[:-candidate]
            right = profile[candidate:]
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator <= 0:
                continue
            best = max(best, float(np.dot(left, right) / denominator))
        return best if np.isfinite(best) else 0.0

    def _component_scale_diagnostics(
        self,
        image_rgb: UInt8Image,
        bounds: Rect,
        labels: NDArray[np.int32],
        label: int,
        *,
        pitch_x: float,
        pitch_y: float,
        cells: int,
    ) -> ComponentScaleDiagnostics:
        """Detect harmonic sub-grids inside one logical board component.

        A miniature opponent board can alias onto the player grid when two or
        three miniature cells fit inside one player cell. The projected
        occupancy may still look strong, so this check works in pixel space:
        repeated edge structure at P/2 or P/3 is compared with the expected
        player pitch P independently on both axes.

        Rejection requires agreement across at least two measurable axes. A
        one-axis conflict is surfaced as ambiguous instead of being discarded,
        preserving narrow or unusual legitimate player-board islands.
        """
        positions = np.argwhere(labels == label)
        if positions.size == 0 or cells < self.config.secondary_component_scale_min_cells:
            return ComponentScaleDiagnostics(
                label=label,
                cells=cells,
                axes_tested=0,
                expected_scores=(),
                subgrid_scores=(),
                subgrid_divisors=(),
                estimated_scale_ratio=None,
                status="insufficient-evidence",
            )

        min_row, min_col = positions.min(axis=0)
        max_row, max_col = positions.max(axis=0)
        x0 = bounds.x + int(round(float(min_col) * pitch_x))
        x1 = bounds.x + int(round(float(max_col + 1) * pitch_x))
        y0 = bounds.y + int(round(float(min_row) * pitch_y))
        y1 = bounds.y + int(round(float(max_row + 1) * pitch_y))
        crop = image_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            return ComponentScaleDiagnostics(
                label=label,
                cells=cells,
                axes_tested=0,
                expected_scores=(),
                subgrid_scores=(),
                subgrid_divisors=(),
                estimated_scale_ratio=None,
                status="insufficient-evidence",
            )

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        expected_scores: list[float] = []
        subgrid_scores: list[float] = []
        subgrid_divisors: list[int | None] = []
        detected_divisors: list[int] = []

        for axis, expected_pitch in ((1, pitch_x), (0, pitch_y)):
            profile_length = gray.shape[axis]
            if (
                profile_length
                < expected_pitch * self.config.secondary_component_scale_min_span_cells
            ):
                continue

            derivative = np.abs(np.diff(gray, axis=axis))
            profile = derivative.mean(axis=1 - axis).astype(np.float32)
            profile -= profile.mean()
            expected_lag = max(1, int(round(expected_pitch)))
            expected_score = self._autocorrelation_at(
                profile,
                expected_lag,
                tolerance=self.config.secondary_component_subgrid_lag_tolerance_px,
            )

            best_score = float("-inf")
            best_divisor: int | None = None
            for divisor in self.config.secondary_component_subgrid_divisors:
                lag = max(1, int(round(expected_pitch / divisor)))
                score = self._autocorrelation_at(
                    profile,
                    lag,
                    tolerance=self.config.secondary_component_subgrid_lag_tolerance_px,
                )
                if score > best_score:
                    best_score = score
                    best_divisor = divisor

            best_score = best_score if np.isfinite(best_score) else 0.0
            expected_scores.append(expected_score)
            subgrid_scores.append(best_score)

            is_subgrid = (
                best_divisor is not None
                and best_score >= self.config.secondary_component_subgrid_min_score
                and best_score
                >= expected_score * self.config.secondary_component_subgrid_relative_score
            )
            subgrid_divisors.append(best_divisor if is_subgrid else None)
            if is_subgrid and best_divisor is not None:
                detected_divisors.append(best_divisor)

        axes_tested = len(expected_scores)
        if axes_tested < 2:
            status = "insufficient-evidence"
            ratio = None
        elif (
            len(detected_divisors) == axes_tested
            and len(set(detected_divisors)) == 1
        ):
            ratio = 1.0 / detected_divisors[0]
            status = "scaled-replica"
        elif detected_divisors:
            ratio = float(np.median([1.0 / item for item in detected_divisors]))
            status = "ambiguous"
        else:
            ratio = 1.0
            status = "same-scale"

        return ComponentScaleDiagnostics(
            label=label,
            cells=cells,
            axes_tested=axes_tested,
            expected_scores=tuple(expected_scores),
            subgrid_scores=tuple(subgrid_scores),
            subgrid_divisors=tuple(subgrid_divisors),
            estimated_scale_ratio=ratio,
            status=status,
        )

    def _local_subgrid_votes(
        self,
        image_rgb: UInt8Image,
        bounds: Rect,
        topology: NDArray[np.bool_],
        *,
        pitch_x: float,
        pitch_y: float,
    ) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
        """Detect embedded scaled-grid regions with overlapping local windows.

        Connected-component selection alone cannot separate two boards when
        their projected logical cells touch. Each active cell therefore votes
        through a small neighborhood. A neighborhood is considered scaled only
        when both pixel axes agree on the same P/2 or P/3 divisor.

        Overlapping windows provide spatial consensus: isolated false harmonic
        responses from tile artwork receive too few votes, while a real
        miniature grid produces repeated votes across neighboring cells.
        """
        rows, cols = topology.shape
        votes = np.zeros((rows, cols), dtype=np.uint8)
        opportunities = np.zeros((rows, cols), dtype=np.uint8)
        radius = self.config.local_scale_window_radius

        for row in range(rows):
            for col in range(cols):
                if not bool(topology[row, col]):
                    continue

                r0 = max(0, row - radius)
                r1 = min(rows, row + radius + 1)
                c0 = max(0, col - radius)
                c1 = min(cols, col + radius + 1)
                window_topology = topology[r0:r1, c0:c1]
                active_cells = int(np.count_nonzero(window_topology))
                if active_cells < self.config.local_scale_min_active_cells:
                    continue

                x0 = bounds.x + int(round(c0 * pitch_x))
                x1 = bounds.x + int(round(c1 * pitch_x))
                y0 = bounds.y + int(round(r0 * pitch_y))
                y1 = bounds.y + int(round(r1 * pitch_y))
                crop = image_rgb[y0:y1, x0:x1]
                if crop.size == 0:
                    continue

                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
                detected_divisors: list[int] = []

                for axis, expected_pitch in ((1, pitch_x), (0, pitch_y)):
                    derivative = np.abs(np.diff(gray, axis=axis))
                    profile = derivative.mean(axis=1 - axis).astype(np.float32)
                    profile -= profile.mean()

                    expected_score = self._autocorrelation_at(
                        profile,
                        max(1, int(round(expected_pitch))),
                        tolerance=self.config.secondary_component_subgrid_lag_tolerance_px,
                    )

                    best_score = float("-inf")
                    best_divisor: int | None = None
                    for divisor in self.config.secondary_component_subgrid_divisors:
                        score = self._autocorrelation_at(
                            profile,
                            max(1, int(round(expected_pitch / divisor))),
                            tolerance=self.config.secondary_component_subgrid_lag_tolerance_px,
                        )
                        if score > best_score:
                            best_score = score
                            best_divisor = divisor

                    if (
                        best_divisor is not None
                        and np.isfinite(best_score)
                        and best_score >= self.config.secondary_component_subgrid_min_score
                        and best_score
                        >= expected_score
                        * self.config.secondary_component_subgrid_relative_score
                    ):
                        detected_divisors.append(best_divisor)

                scaled_window = (
                    len(detected_divisors) == 2
                    and len(set(detected_divisors)) == 1
                )

                for rr in range(r0, r1):
                    for cc in range(c0, c1):
                        if not bool(topology[rr, cc]):
                            continue
                        opportunities[rr, cc] = min(
                            255,
                            int(opportunities[rr, cc]) + 1,
                        )
                        if scaled_window:
                            votes[rr, cc] = min(255, int(votes[rr, cc]) + 1)

        required = np.maximum(
            self.config.local_scale_min_votes,
            np.ceil(
                opportunities.astype(np.float32)
                * self.config.local_scale_vote_ratio
            ).astype(np.uint8),
        )
        seeds = (
            topology
            & (opportunities > 0)
            & (votes >= required)
        )

        # Strict consensus identifies reliable seed cells, but edge cells of a
        # miniature board naturally receive fewer overlapping windows. Expand
        # each seeded region through contiguous cells that received any local
        # sub-grid evidence, provided a meaningful fraction of that support
        # component is made of strict seeds. This completes the mini-board
        # boundary without dilating blindly into zero-vote player cells.
        support = (
            topology
            & (votes >= self.config.local_scale_expand_min_vote)
        )
        count, support_labels = cv2.connectedComponents(
            support.astype(np.uint8),
            connectivity=4,
        )
        scaled = seeds.copy()
        for label in range(1, count):
            component = support_labels == label
            component_size = int(np.count_nonzero(component))
            if component_size == 0:
                continue
            seed_count = int(np.count_nonzero(seeds & component))
            if (
                seed_count > 0
                and seed_count / component_size
                >= self.config.local_scale_expand_min_seed_fraction
            ):
                scaled |= component

        return votes, scaled.astype(np.bool_)

    def _select_primary_topology(
        self,
        topology: NDArray[np.bool_],
        occupancy: NDArray[np.float32],
        *,
        image_rgb: UInt8Image | None = None,
        bounds: Rect | None = None,
        pitch_x: float | None = None,
        pitch_y: float | None = None,
        preferred_board_point: Point | None = None,
    ) -> tuple[
        NDArray[np.int32],
        tuple[int, ...],
        tuple[float, ...],
        tuple[float, ...],
        tuple[ComponentScaleDiagnostics, ...],
        NDArray[np.uint8],
        NDArray[np.bool_],
        NDArray[np.bool_],
    ]:
        """Separate player-board islands from scaled opponent-board replicas.

        Large/comparable components remain safe to keep. Secondary components
        with enough pixels are additionally checked for harmonic sub-grid
        structure. A confirmed P/2 or P/3 grid is rejected even when occupancy
        is high; ambiguous scale evidence is retained and surfaced to callers
        so a future match-level UI can request one human choice.
        """
        count, labels = cv2.connectedComponents(
            topology.astype(np.uint8),
            connectivity=4,
        )
        if count <= 1:
            return (
                labels.astype(np.int32),
                (),
                (),
                (),
                (),
                np.zeros(topology.shape, dtype=np.uint8),
                np.zeros(topology.shape, dtype=np.bool_),
                topology.copy(),
            )

        sizes = tuple(
            int(np.count_nonzero(labels == label))
            for label in range(1, count)
        )
        means = tuple(
            float(occupancy[labels == label].mean())
            if np.any(labels == label)
            else 0.0
            for label in range(1, count)
        )
        strong_fractions = tuple(
            float(
                np.count_nonzero(
                    occupancy[labels == label]
                    >= self.config.occupancy_threshold
                )
                / max(1, sizes[label - 1])
            )
            for label in range(1, count)
        )
        if not sizes:
            return (
                labels.astype(np.int32),
                (),
                (),
                (),
                (),
                np.zeros(topology.shape, dtype=np.uint8),
                np.zeros(topology.shape, dtype=np.bool_),
                topology.copy(),
            )

        largest = max(sizes)
        largest_label = 1 + sizes.index(largest)
        primary_label = largest_label
        human_anchor_active = False

        if (
            preferred_board_point is not None
            and bounds is not None
            and pitch_x is not None
            and pitch_y is not None
            and bounds.x <= preferred_board_point.x < bounds.right
            and bounds.y <= preferred_board_point.y < bounds.bottom
        ):
            anchor_col = int((preferred_board_point.x - bounds.x) / pitch_x)
            anchor_row = int((preferred_board_point.y - bounds.y) / pitch_y)
            if (
                0 <= anchor_row < labels.shape[0]
                and 0 <= anchor_col < labels.shape[1]
            ):
                anchor_label = int(labels[anchor_row, anchor_col])
                if anchor_label > 0:
                    primary_label = anchor_label
                    human_anchor_active = True

        scale_diagnostics: list[ComponentScaleDiagnostics] = []
        keep_labels: set[int] = set()

        for label, (size, mean_evidence, strong_fraction) in enumerate(
            zip(sizes, means, strong_fractions),
            start=1,
        ):
            comparable_size = (
                not human_anchor_active
                and size >= largest * self.config.secondary_component_keep_ratio
            )
            visual_support = (
                mean_evidence >= self.config.secondary_component_min_mean_evidence
                or strong_fraction
                >= self.config.secondary_component_min_strong_fraction
            )

            if (
                image_rgb is not None
                and bounds is not None
                and pitch_x is not None
                and pitch_y is not None
                and label != primary_label
                and not comparable_size
            ):
                scale = self._component_scale_diagnostics(
                    image_rgb,
                    bounds,
                    labels.astype(np.int32),
                    label,
                    pitch_x=pitch_x,
                    pitch_y=pitch_y,
                    cells=size,
                )
            else:
                scale = ComponentScaleDiagnostics(
                    label=label,
                    cells=size,
                    axes_tested=0,
                    expected_scores=(),
                    subgrid_scores=(),
                    subgrid_divisors=(),
                    estimated_scale_ratio=1.0 if label == primary_label else None,
                    status=(
                        "primary"
                        if label == primary_label
                        else "comparable-size"
                        if comparable_size
                        else "not-tested"
                    ),
                )

            scale_diagnostics.append(scale)

            if label == primary_label or comparable_size:
                keep_labels.add(label)
                continue
            if scale.status == "scaled-replica":
                continue
            if human_anchor_active and scale.status == "ambiguous":
                continue
            if visual_support:
                keep_labels.add(label)

        selected = np.isin(labels, tuple(keep_labels))

        local_subgrid_votes = np.zeros(topology.shape, dtype=np.uint8)
        local_subgrid_topology = np.zeros(topology.shape, dtype=np.bool_)
        if (
            image_rgb is not None
            and bounds is not None
            and pitch_x is not None
            and pitch_y is not None
        ):
            local_subgrid_votes, local_subgrid_topology = self._local_subgrid_votes(
                image_rgb,
                bounds,
                selected.astype(np.bool_),
                pitch_x=pitch_x,
                pitch_y=pitch_y,
            )

            if preferred_board_point is not None:
                anchor_col = int((preferred_board_point.x - bounds.x) / pitch_x)
                anchor_row = int((preferred_board_point.y - bounds.y) / pitch_y)
                if (
                    0 <= anchor_row < topology.shape[0]
                    and 0 <= anchor_col < topology.shape[1]
                ):
                    # Never erase the human-selected region. The local scale map
                    # is a rejection cue only for spatially separate candidates.
                    anchor_mask = np.zeros(topology.shape, dtype=np.uint8)
                    anchor_mask[anchor_row, anchor_col] = 1
                    kernel = np.ones((3, 3), dtype=np.uint8)
                    protected = cv2.dilate(anchor_mask, kernel, iterations=1) > 0
                    local_subgrid_topology &= ~protected

            selected &= ~local_subgrid_topology

        return (
            labels.astype(np.int32),
            sizes,
            means,
            strong_fractions,
            tuple(scale_diagnostics),
            local_subgrid_votes,
            local_subgrid_topology,
            selected.astype(np.bool_),
        )

    @staticmethod
    def _crop_to_selected_topology(
        bounds: Rect,
        occupancy: NDArray[np.float32],
        topology: NDArray[np.bool_],
        pitch_x: float,
        pitch_y: float,
    ) -> tuple[Rect, NDArray[np.float32], NDArray[np.bool_], float, float]:
        """Normalize the logical grid around the retained board component."""
        positions = np.argwhere(topology)
        if positions.size == 0:
            return bounds, occupancy, topology, pitch_x, pitch_y

        min_row, min_col = positions.min(axis=0)
        max_row, max_col = positions.max(axis=0)

        x0 = int(round(float(min_col) * pitch_x))
        x1 = int(round(float(max_col + 1) * pitch_x))
        y0 = int(round(float(min_row) * pitch_y))
        y1 = int(round(float(max_row + 1) * pitch_y))

        cropped_occupancy = occupancy[min_row : max_row + 1, min_col : max_col + 1].copy()
        cropped_topology = topology[min_row : max_row + 1, min_col : max_col + 1].copy()
        final_bounds = Rect(
            bounds.x + x0,
            bounds.y + y0,
            x1 - x0,
            y1 - y0,
        )
        rows, cols = cropped_topology.shape
        final_pitch_x = final_bounds.width / max(1, cols)
        final_pitch_y = final_bounds.height / max(1, rows)
        return (
            final_bounds,
            cropped_occupancy,
            cropped_topology,
            final_pitch_x,
            final_pitch_y,
        )

    def _cells_from_topology(
        self,
        occupancy: NDArray[np.float32],
        topology: NDArray[np.bool_],
        bounds: Rect,
        pitch_x: float,
        pitch_y: float,
    ) -> list[Cell]:
        """Convert reconciled topology into cell geometry."""
        cells: list[Cell] = []
        rows, cols = occupancy.shape

        for row in range(rows):
            for col in range(cols):
                if not bool(topology[row, col]):
                    continue

                score = float(occupancy[row, col])
                x0 = int(round(col * pitch_x))
                x1 = int(round((col + 1) * pitch_x))
                y0 = int(round(row * pitch_y))
                y1 = int(round((row + 1) * pitch_y))
                gx0, gy0 = bounds.x + x0, bounds.y + y0
                width, height = x1 - x0, y1 - y0
                cells.append(
                    Cell(
                        row=row,
                        col=col,
                        center=Point(gx0 + width // 2, gy0 + height // 2),
                        bounds=Rect(gx0, gy0, width, height),
                        occupancy=score,
                    )
                )

        return cells

    @staticmethod
    def _confidence(bounds: Rect, pitch_x: float, pitch_y: float, cells: list[Cell]) -> float:
        square_score = 1.0 - min(1.0, abs(pitch_x - pitch_y) / max(pitch_x, pitch_y))
        occupancy_score = min(1.0, float(np.median([cell.occupancy for cell in cells])) / 0.80)
        size_score = 1.0 if min(bounds.width, bounds.height) >= min(pitch_x, pitch_y) * 3 else 0.5
        return float(np.clip(0.55 * square_score + 0.35 * occupancy_score + 0.10 * size_score, 0.0, 1.0))
