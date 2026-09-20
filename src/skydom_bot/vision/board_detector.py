"""Automatic detection of regular-but-possibly-irregular Match-3 boards."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from skydom_bot.domain.board import BoardGeometry, Cell, Point, Rect

UInt8Image = NDArray[np.uint8]


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

    min_component_area_ratio: float = 0.03
    min_fragment_area_ratio: float = 0.005
    component_join_gap_ratio: float = 0.03
    component_projection_overlap: float = 0.50

    pitch_min_px: int = 28
    pitch_max_px: int = 140
    pitch_peak_ratio: float = 0.75

    occupancy_threshold: float = 0.50
    cell_corner_ratio: float = 0.16


class BoardDetectionError(RuntimeError):
    """Raised when a board cannot be inferred with sufficient confidence."""


class BoardDetector:
    """Detect board bounds, cell pitch, dimensions, and active cells."""

    def __init__(self, config: BoardDetectorConfig | None = None) -> None:
        self.config = config or BoardDetectorConfig()

    def detect(self, image_rgb: UInt8Image) -> BoardGeometry:
        """Detect board geometry from an RGB image.

        The detector separates two different questions:
        1. Where is the board as a whole?
        2. Which logical grid positions are playable cells?

        The first question uses the stable dark-blue board background and can
        group disconnected fragments caused by blockers such as ice. The second
        uses multiple surface cues (normal board background + ice) at cell
        corners, where game pieces usually occlude the least.
        """
        self._validate_image(image_rgb)

        board_mask = self._board_background_mask(image_rgb)
        bounds, component_mask = self._board_component(board_mask)

        crop = image_rgb[bounds.y : bounds.bottom, bounds.x : bounds.right]
        pitch_x = self._estimate_pitch(crop, axis=1)
        pitch_y = self._estimate_pitch(crop, axis=0)

        # Both axes describe the same square-cell grid. Combining them prevents
        # one partially occluded axis from dominating the result.
        pitch = float(np.median([pitch_x, pitch_y]))
        cols = max(1, int(round(bounds.width / pitch)))
        rows = max(1, int(round(bounds.height / pitch)))
        pitch_x = bounds.width / cols
        pitch_y = bounds.height / rows

        evidence_mask = cv2.bitwise_or(component_mask, self._ice_surface_mask(image_rgb))
        cells = self._extract_cells(evidence_mask, bounds, rows, cols, pitch_x, pitch_y)
        if not cells:
            raise BoardDetectionError("Board component found, but no active cells were inferred.")

        confidence = self._confidence(bounds, pitch_x, pitch_y, cells)
        return BoardGeometry(
            bounds=bounds,
            rows=rows,
            cols=cols,
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            cells=tuple(cells),
            confidence=confidence,
        )

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

        Ice can split one physical board into several disconnected blue
        components. We therefore cluster large fragments that overlap strongly
        on one axis and are close on the other. Unlike morphological closing,
        the returned mask still contains only pixels observed in the original
        image.
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

    def _estimate_pitch(self, crop_rgb: UInt8Image, axis: int) -> float:
        """Estimate cell spacing from autocorrelation of image-edge energy.

        OpenCV provides the image operations (grayscale conversion and the
        surrounding image representation), but the autocorrelation itself is
        implemented here with NumPy. We seek the smallest strong local peak so
        a 2-cell harmonic (for example 130 px) does not beat the 1-cell period
        (65 px).
        """
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        derivative = np.abs(np.diff(gray, axis=axis))
        profile = derivative.mean(axis=1 - axis)
        profile -= profile.mean()

        maximum = min(self.config.pitch_max_px, max(self.config.pitch_min_px + 1, len(profile) // 3))
        minimum = min(self.config.pitch_min_px, maximum - 1)
        if maximum <= minimum:
            raise BoardDetectionError("Board component is too small to estimate a cell pitch.")

        scores = np.full(maximum + 1, -np.inf, dtype=np.float64)
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
        peaks = [
            lag
            for lag in range(minimum + 1, maximum)
            if (
                np.isfinite(scores[lag])
                and scores[lag] >= peak_floor
                and scores[lag] >= scores[lag - 1]
                and scores[lag] >= scores[lag + 1]
            )
        ]
        pitch = min(peaks) if peaks else int(np.argmax(scores))
        if not np.isfinite(scores[pitch]) or scores[pitch] <= 0:
            raise BoardDetectionError("Could not find a periodic grid signal in the board image.")
        return float(pitch)

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

    def _extract_cells(
        self,
        evidence_mask: UInt8Image,
        bounds: Rect,
        rows: int,
        cols: int,
        pitch_x: float,
        pitch_y: float,
    ) -> list[Cell]:
        local = evidence_mask[bounds.y : bounds.bottom, bounds.x : bounds.right]
        cells: list[Cell] = []

        for row in range(rows):
            for col in range(cols):
                x0 = int(round(col * pitch_x))
                x1 = int(round((col + 1) * pitch_x))
                y0 = int(round(row * pitch_y))
                y1 = int(round((row + 1) * pitch_y))
                region = local[y0:y1, x0:x1]
                occupancy = self._corner_occupancy(region)
                if occupancy < self.config.occupancy_threshold:
                    continue

                gx0, gy0 = bounds.x + x0, bounds.y + y0
                width, height = x1 - x0, y1 - y0
                cells.append(
                    Cell(
                        row=row,
                        col=col,
                        center=Point(gx0 + width // 2, gy0 + height // 2),
                        bounds=Rect(gx0, gy0, width, height),
                        occupancy=occupancy,
                    )
                )

        return cells

    @staticmethod
    def _confidence(bounds: Rect, pitch_x: float, pitch_y: float, cells: list[Cell]) -> float:
        square_score = 1.0 - min(1.0, abs(pitch_x - pitch_y) / max(pitch_x, pitch_y))
        occupancy_score = min(1.0, float(np.median([cell.occupancy for cell in cells])) / 0.80)
        size_score = 1.0 if min(bounds.width, bounds.height) >= min(pitch_x, pitch_y) * 3 else 0.5
        return float(np.clip(0.55 * square_score + 0.35 * occupancy_score + 0.10 * size_score, 0.0, 1.0))
