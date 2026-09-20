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
    """Tunable thresholds for the current Skydom board theme.

    The detector deliberately isolates theme-specific color assumptions here.
    Grid inference itself is resolution independent and supports missing cells.
    """

    hue_min: int = 108
    hue_max: int = 132
    saturation_min: int = 90
    saturation_max: int = 230
    value_min: int = 55
    value_max: int = 190
    min_component_area_ratio: float = 0.03
    pitch_min_px: int = 28
    pitch_max_px: int = 140
    occupancy_threshold: float = 0.035
    cell_inset_ratio: float = 0.08


class BoardDetectionError(RuntimeError):
    """Raised when a board cannot be inferred with sufficient confidence."""


class BoardDetector:
    """Detect the board component, cell pitch, dimensions, and active cells."""

    def __init__(self, config: BoardDetectorConfig | None = None) -> None:
        self.config = config or BoardDetectorConfig()

    def detect(self, image_rgb: UInt8Image) -> BoardGeometry:
        """Detect board geometry from an RGB image.

        The algorithm uses four steps:
        1. Segment the dark-blue board background in HSV.
        2. Keep the largest plausible connected component.
        3. Estimate cell pitch from periodic image-gradient autocorrelation.
        4. Divide the component into a regular logical grid and retain cells
           whose segmented-background occupancy indicates that they exist.
        """
        self._validate_image(image_rgb)
        mask = self._board_background_mask(image_rgb)
        bounds, component_mask = self._largest_component(mask)
        crop = image_rgb[bounds.y : bounds.bottom, bounds.x : bounds.right]
        pitch_x = self._estimate_pitch(crop, axis=1)
        pitch_y = self._estimate_pitch(crop, axis=0)

        # A single square-cell pitch is more stable when one axis is partially
        # obscured by pieces or an irregular final row.
        pitch = float(np.median([pitch_x, pitch_y]))
        cols = max(1, int(round(bounds.width / pitch)))
        rows = max(1, int(round(bounds.height / pitch)))
        pitch_x = bounds.width / cols
        pitch_y = bounds.height / rows

        cells = self._extract_cells(component_mask, bounds, rows, cols, pitch_x, pitch_y)
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

    def _largest_component(self, mask: UInt8Image) -> tuple[Rect, UInt8Image]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            raise BoardDetectionError("No board-colored connected component was found.")

        image_area = mask.shape[0] * mask.shape[1]
        min_area = image_area * self.config.min_component_area_ratio
        candidates: list[tuple[int, int]] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                candidates.append((area, label))
        if not candidates:
            raise BoardDetectionError("No connected component is large enough to be the board.")

        _, label = max(candidates)
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = np.where(labels == label, 255, 0).astype(np.uint8)
        return Rect(x, y, width, height), component

    def _estimate_pitch(self, crop_rgb: UInt8Image, axis: int) -> float:
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

        pitch = int(np.argmax(scores))
        if not np.isfinite(scores[pitch]) or scores[pitch] <= 0:
            raise BoardDetectionError("Could not find a periodic grid signal in the board image.")
        return float(pitch)

    def _extract_cells(
        self,
        component_mask: UInt8Image,
        bounds: Rect,
        rows: int,
        cols: int,
        pitch_x: float,
        pitch_y: float,
    ) -> list[Cell]:
        local = component_mask[bounds.y : bounds.bottom, bounds.x : bounds.right]
        cells: list[Cell] = []
        for row in range(rows):
            for col in range(cols):
                x0 = int(round(col * pitch_x))
                x1 = int(round((col + 1) * pitch_x))
                y0 = int(round(row * pitch_y))
                y1 = int(round((row + 1) * pitch_y))
                dx = max(1, int((x1 - x0) * self.config.cell_inset_ratio))
                dy = max(1, int((y1 - y0) * self.config.cell_inset_ratio))
                region = local[y0 + dy : y1 - dy, x0 + dx : x1 - dx]
                occupancy = float(np.count_nonzero(region) / region.size) if region.size else 0.0
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
        occupancy_score = min(1.0, float(np.median([cell.occupancy for cell in cells])) / 0.20)
        size_score = 1.0 if min(bounds.width, bounds.height) >= min(pitch_x, pitch_y) * 3 else 0.5
        return float(np.clip(0.55 * square_score + 0.35 * occupancy_score + 0.10 * size_score, 0.0, 1.0))
