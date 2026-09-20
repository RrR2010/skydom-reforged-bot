"""Classical color-based tile recognition for the first M3 milestone."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from skydom_bot.domain.board import BoardGeometry, Cell
from skydom_bot.domain.tile import TileColor, TileObservation

UInt8Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class TileClassifierConfig:
    """Thresholds for extracting bright, saturated tile pixels."""

    center_radius_ratio: float = 0.38
    saturation_min: int = 105
    value_min: int = 125
    min_foreground_fraction: float = 0.08
    shape_inset_ratio: float = 0.04
    shape_component_min_largest_ratio: float = 0.12
    shape_component_min_area_ratio: float = 0.015
    shape_bridge_ratio: float = 0.18
    shape_background_delta_min: float = 22.0
    neutral_overlay_saturation_max: int = 85
    neutral_overlay_value_min: int = 220
    histogram_bins: int = 180
    min_class_confidence: float = 0.50


@dataclass(frozen=True, slots=True)
class TileDiagnostics:
    """Intermediate signals for one tile classification."""

    crop_rgb: UInt8Image
    crop_hsv: UInt8Image
    center_mask: UInt8Image
    foreground_mask: UInt8Image
    shape_foreground_mask: UInt8Image
    overlay_foreground_mask: UInt8Image
    neutral_overlay_mask: UInt8Image
    background_distance_mask: UInt8Image
    hue_histogram: NDArray[np.float64]
    class_scores: dict[TileColor, float]
    dominant_hue: float | None


class TileClassifier:
    """Classify the base tile color from each known board cell.

    This first recognizer intentionally answers only "what color family is this
    tile?". Shape/special-piece recognition is a separate problem and will be
    layered on later.
    """

    def __init__(self, config: TileClassifierConfig | None = None) -> None:
        self.config = config or TileClassifierConfig()

    def classify_board(
        self,
        image_rgb: UInt8Image,
        geometry: BoardGeometry,
    ) -> tuple[TileObservation, ...]:
        """Classify every active board cell."""
        return tuple(self.classify_cell(image_rgb, cell)[0] for cell in geometry.cells)

    def classify_cell(
        self,
        image_rgb: UInt8Image,
        cell: Cell,
    ) -> tuple[TileObservation, TileDiagnostics]:
        """Classify one cell and expose diagnostics for visual debugging."""
        crop = image_rgb[
            cell.bounds.y : cell.bounds.bottom,
            cell.bounds.x : cell.bounds.right,
        ].copy()
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        height, width = crop.shape[:2]

        center_mask = np.zeros((height, width), dtype=np.uint8)
        radius = max(2, int(round(min(width, height) * self.config.center_radius_ratio)))
        cv2.circle(center_mask, (width // 2, height // 2), radius, 255, -1)

        saturated = cv2.inRange(
            hsv,
            np.array([0, self.config.saturation_min, self.config.value_min], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        foreground = cv2.bitwise_and(center_mask, saturated)

        center_pixels = max(1, int(np.count_nonzero(center_mask)))
        foreground_pixels = int(np.count_nonzero(foreground))
        foreground_fraction = foreground_pixels / center_pixels

        hue_values = hsv[:, :, 0][foreground > 0]
        histogram = np.zeros(self.config.histogram_bins, dtype=np.float64)
        dominant_hue: float | None = None

        if hue_values.size:
            histogram = np.bincount(
                hue_values.astype(np.int32),
                minlength=self.config.histogram_bins,
            ).astype(np.float64)
            histogram /= max(1.0, histogram.sum())
            dominant_hue = float(np.argmax(histogram))

        scores = self._class_scores(histogram)
        color, confidence = self._select_color(scores, foreground_fraction)

        (
            shape_foreground,
            overlay_foreground,
            neutral_overlay,
            background_distance,
        ) = self._shape_masks(
            crop,
            hsv,
            saturated,
            color,
        )

        observation = TileObservation(
            row=cell.row,
            col=cell.col,
            color=color,
            confidence=confidence,
            dominant_hue=dominant_hue,
            foreground_fraction=foreground_fraction,
        )
        diagnostics = TileDiagnostics(
            crop_rgb=crop,
            crop_hsv=hsv,
            center_mask=center_mask,
            foreground_mask=foreground,
            shape_foreground_mask=shape_foreground,
            overlay_foreground_mask=overlay_foreground,
            neutral_overlay_mask=neutral_overlay,
            background_distance_mask=background_distance,
            hue_histogram=histogram,
            class_scores=scores,
            dominant_hue=dominant_hue,
        )
        return observation, diagnostics

    def _shape_masks(
        self,
        crop_rgb: UInt8Image,
        hsv: UInt8Image,
        saturated: UInt8Image,
        color: TileColor,
    ) -> tuple[UInt8Image, UInt8Image, UInt8Image, UInt8Image]:
        """Build stable base-shape and residual-overlay masks.

        Shape segmentation should describe the tile's own silhouette rather
        than every bright/saturated object in the cell. The predicted color
        therefore becomes an additional cue: retain same-family pixels across
        almost the full cell, clean tiny threshold artifacts, then keep the
        connected component nearest the cell center.

        Pixels that are bright/saturated but not part of that base component
        are preserved separately as overlay evidence. This is useful for
        blockers such as a yellow chain crossing a purple tile.
        """
        height, width = saturated.shape
        region = np.zeros((height, width), dtype=np.uint8)
        inset_x = max(1, int(round(width * self.config.shape_inset_ratio)))
        inset_y = max(1, int(round(height * self.config.shape_inset_ratio)))
        cv2.rectangle(
            region,
            (inset_x, inset_y),
            (max(inset_x, width - inset_x - 1), max(inset_y, height - inset_y - 1)),
            255,
            -1,
        )
        full_foreground = cv2.bitwise_and(region, saturated)

        # Estimate the local board/background color from corner samples, then
        # reject pixels too similar to that background. This matters most for
        # blue pieces because the board itself is also blue and otherwise fills
        # the shape mask.
        lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        patch_h = max(2, int(round(height * 0.14)))
        patch_w = max(2, int(round(width * 0.14)))
        corner_pixels = np.concatenate(
            (
                lab[:patch_h, :patch_w].reshape(-1, 3),
                lab[:patch_h, width - patch_w :].reshape(-1, 3),
                lab[height - patch_h :, :patch_w].reshape(-1, 3),
                lab[height - patch_h :, width - patch_w :].reshape(-1, 3),
            ),
            axis=0,
        )
        background_lab = np.median(corner_pixels, axis=0)
        delta = np.linalg.norm(lab - background_lab, axis=2)
        background_distance = np.where(
            delta >= self.config.shape_background_delta_min,
            255,
            0,
        ).astype(np.uint8)

        hue = hsv[:, :, 0]
        color_mask = self._color_hue_mask(hue, color)
        candidate = cv2.bitwise_and(full_foreground, color_mask)
        candidate = cv2.bitwise_and(candidate, background_distance)

        # White/neutral decorations are weak in the hue histogram but can be
        # strong evidence of a power-up or special overlay. Keep them separate
        # from the base color mask.
        neutral_overlay = cv2.inRange(
            hsv,
            np.array(
                [0, 0, self.config.neutral_overlay_value_min],
                dtype=np.uint8,
            ),
            np.array(
                [179, self.config.neutral_overlay_saturation_max, 255],
                dtype=np.uint8,
            ),
        )
        neutral_overlay = cv2.bitwise_and(neutral_overlay, region)

        kernel = np.ones((3, 3), dtype=np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.where(candidate > 0, 1, 0).astype(np.uint8),
            connectivity=8,
        )
        shape = np.zeros_like(candidate)
        if count > 1:
            areas = [
                int(stats[label, cv2.CC_STAT_AREA])
                for label in range(1, count)
            ]
            largest = max(areas, default=0)
            min_area = max(
                int(round(width * height * self.config.shape_component_min_area_ratio)),
                int(round(largest * self.config.shape_component_min_largest_ratio)),
            )

            # A blocker can cut the base tile exactly through its center. Using
            # only the component nearest the center therefore selected a tiny
            # remnant on chained pieces. Retain every substantial same-color
            # component, then bridge narrow blocker gaps morphologically.
            for label in range(1, count):
                if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                    shape[labels == label] = 255

            bridge = max(
                3,
                int(round(min(width, height) * self.config.shape_bridge_ratio)),
            )
            if bridge % 2 == 0:
                bridge += 1
            bridge_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (bridge, bridge),
            )
            shape = cv2.morphologyEx(shape, cv2.MORPH_CLOSE, bridge_kernel)

            # After bridging, keep the largest reconstructed base component.
            reconstructed_count, reconstructed_labels, reconstructed_stats, _ = (
                cv2.connectedComponentsWithStats(
                    np.where(shape > 0, 1, 0).astype(np.uint8),
                    connectivity=8,
                )
            )
            if reconstructed_count > 1:
                selected = 1 + int(
                    np.argmax(
                        reconstructed_stats[
                            1:reconstructed_count,
                            cv2.CC_STAT_AREA,
                        ]
                    )
                )
                shape = np.where(
                    reconstructed_labels == selected,
                    255,
                    0,
                ).astype(np.uint8)

        overlay = cv2.bitwise_and(full_foreground, cv2.bitwise_not(shape))
        return shape, overlay, neutral_overlay, background_distance

    @staticmethod
    def _color_hue_mask(hue: NDArray[np.uint8], color: TileColor) -> UInt8Image:
        """Return an expanded hue-family mask for silhouette extraction."""
        ranges = {
            TileColor.RED: ((170, 179), (0, 10)),
            TileColor.ORANGE: ((0, 24),),
            TileColor.YELLOW: ((15, 38),),
            TileColor.GREEN: ((34, 90),),
            TileColor.BLUE: ((82, 136),),
            TileColor.PURPLE: ((128, 172),),
        }
        if color is TileColor.UNKNOWN:
            return np.full(hue.shape, 255, dtype=np.uint8)

        result = np.zeros(hue.shape, dtype=np.uint8)
        for start, end in ranges[color]:
            result[(hue >= start) & (hue <= end)] = 255
        return result

    @staticmethod
    def _circular_range_sum(histogram: NDArray[np.float64], start: int, end: int) -> float:
        """Sum a hue interval, supporting OpenCV hue wraparound at 180."""
        if start <= end:
            return float(histogram[start : end + 1].sum())
        return float(histogram[start:].sum() + histogram[: end + 1].sum())

    def _class_scores(self, histogram: NDArray[np.float64]) -> dict[TileColor, float]:
        """Return hue-mass scores for the game's base color families."""
        return {
            TileColor.RED: self._circular_range_sum(histogram, 170, 8),
            TileColor.ORANGE: self._circular_range_sum(histogram, 9, 20),
            TileColor.YELLOW: self._circular_range_sum(histogram, 21, 35),
            TileColor.GREEN: self._circular_range_sum(histogram, 36, 85),
            TileColor.BLUE: self._circular_range_sum(histogram, 86, 132),
            TileColor.PURPLE: self._circular_range_sum(histogram, 133, 169),
        }

    def _select_color(
        self,
        scores: dict[TileColor, float],
        foreground_fraction: float,
    ) -> tuple[TileColor, float]:
        if foreground_fraction < self.config.min_foreground_fraction or not scores:
            return TileColor.UNKNOWN, 0.0

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_color, best_score = ranked[0]

        # Treat the winning hue mass as the confidence baseline. Large gradient
        # pieces (especially orange) legitimately spread across adjacent red and
        # orange hue bins; penalizing a small runner-up margin turned correct
        # orange pieces into UNKNOWN even when orange still held the plurality.
        confidence = float(np.clip(best_score, 0.0, 1.0))
        if confidence < self.config.min_class_confidence:
            return TileColor.UNKNOWN, confidence
        return best_color, confidence
