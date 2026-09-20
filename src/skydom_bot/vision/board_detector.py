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
    min_fragment_area_ratio: float = 0.001
    component_join_gap_ratio: float = 0.07
    component_projection_overlap: float = 0.50

    pitch_min_px: int = 28
    pitch_max_px: int = 140
    pitch_peak_ratio: float = 0.75

    occupancy_threshold: float = 0.50
    occupancy_uncertain_threshold: float = 0.25
    structural_required_cardinal_neighbors: int = 4
    secondary_component_keep_ratio: float = 0.35
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
    occupancy: NDArray[np.float32]
    evidence_state: NDArray[np.uint8]
    cardinal_support: NDArray[np.uint8]
    reconciled_topology: NDArray[np.bool_]
    topology_components: NDArray[np.int32]
    topology_component_sizes: tuple[int, ...]
    selected_topology: NDArray[np.bool_]


class BoardDetectionError(RuntimeError):
    """Raised when a board cannot be inferred with sufficient confidence."""


class BoardDetector:
    """Detect board bounds, cell pitch, dimensions, and active cells."""

    def __init__(self, config: BoardDetectorConfig | None = None) -> None:
        self.config = config or BoardDetectorConfig()

    def detect(self, image_rgb: UInt8Image) -> BoardGeometry:
        """Detect board geometry from an RGB image."""
        geometry, _ = self.detect_with_diagnostics(image_rgb)
        return geometry

    def detect_with_diagnostics(
        self,
        image_rgb: UInt8Image,
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

        # Both axes describe the same square-cell grid. Combining them prevents
        # one partially occluded axis from dominating the result.
        pitch = float(np.median([pitch_x_diag.selected_pitch, pitch_y_diag.selected_pitch]))
        cols = max(1, int(round(bounds.width / pitch)))
        rows = max(1, int(round(bounds.height / pitch)))
        pitch_x = bounds.width / cols
        pitch_y = bounds.height / rows

        ice_mask = self._ice_surface_mask(image_rgb)
        evidence_mask = cv2.bitwise_or(component_mask, ice_mask)
        occupancy = self._cell_occupancy_grid(evidence_mask, bounds, rows, cols, pitch_x, pitch_y)
        evidence_state, cardinal_support, reconciled_topology = self._reconcile_topology(occupancy)
        topology_components, component_sizes, selected_topology = self._select_primary_topology(
            reconciled_topology
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
            occupancy=occupancy,
            evidence_state=evidence_state,
            cardinal_support=cardinal_support,
            reconciled_topology=reconciled_topology,
            topology_components=topology_components,
            topology_component_sizes=component_sizes,
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

        candidates: list[tuple[int, list[int], Rect]] = []
        for group in groups.values():
            area = sum(int(stats[label, cv2.CC_STAT_AREA]) for label in group)
            if area < image_area * self.config.min_component_area_ratio:
                continue

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
            candidates.append((area, group, Rect(left, top, right - left, bottom - top)))

        if not candidates:
            raise BoardDetectionError("No grouped component is large enough to be the board.")

        _, selected_labels, bounds = max(candidates, key=lambda item: item[0])
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

    def _reconcile_topology(
        self,
        occupancy: NDArray[np.float32],
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.bool_]]:
        """Reconcile ambiguous visual evidence with conservative grid structure.

        Evidence is intentionally kept ternary before becoming topology:

        - 2 = strong: visual evidence is already sufficient.
        - 1 = uncertain: plausible cell, but below the strong threshold.
        - 0 = absent: too little evidence to infer a cell.

        An uncertain cell is promoted only when all configured cardinal
        neighbors are strong. This is a deliberately conservative hysteresis
        rule: it recovers a transiently occluded interior cell without filling
        ordinary notches or edge gaps in an irregular board.
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

        promoted = uncertain & (
            support >= self.config.structural_required_cardinal_neighbors
        )
        reconciled = strong | promoted
        return state, support, reconciled

    def _select_primary_topology(
        self,
        topology: NDArray[np.bool_],
    ) -> tuple[NDArray[np.int32], tuple[int, ...], NDArray[np.bool_]]:
        """Separate independent logical board components and reject tiny replicas.

        Competitive levels can show a miniature opponent board near the real
        player board. Both share the same visual theme, so color segmentation
        alone can merge them into one large bounding box. After pitch inference,
        however, the miniature board projects into a sparse secondary component
        on the player's logical grid.

        We label 4-connected topology components and keep the largest component
        plus any other component whose size is comparable. This removes small
        scaled replicas without assuming the real board must be the only island.
        """
        count, labels = cv2.connectedComponents(
            topology.astype(np.uint8),
            connectivity=4,
        )
        if count <= 1:
            return labels.astype(np.int32), (), topology.copy()

        sizes = tuple(
            int(np.count_nonzero(labels == label))
            for label in range(1, count)
        )
        if not sizes:
            return labels.astype(np.int32), (), topology.copy()

        largest = max(sizes)
        keep_labels = {
            label
            for label, size in enumerate(sizes, start=1)
            if size >= largest * self.config.secondary_component_keep_ratio
        }
        selected = np.isin(labels, tuple(keep_labels))
        return labels.astype(np.int32), sizes, selected.astype(np.bool_)

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
