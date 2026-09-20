"""Tests for first-stage tile color recognition."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from skydom_bot.domain.board import Cell, Point, Rect
from skydom_bot.domain.tile import TileColor
from skydom_bot.vision.tile_classifier import TileClassifier


def _rgb_from_hsv(h: int, s: int = 230, v: int = 240) -> tuple[int, int, int]:
    hsv = np.uint8([[[h, s, v]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(channel) for channel in rgb)


def _cell_image(hue: int) -> tuple[np.ndarray, Cell]:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:] = (35, 42, 108)
    cv2.circle(image, (40, 40), 25, _rgb_from_hsv(hue), -1)
    cell = Cell(0, 0, Point(40, 40), Rect(0, 0, 80, 80), 1.0)
    return image, cell


@pytest.mark.parametrize(
    ("hue", "expected"),
    [
        (2, TileColor.RED),
        (15, TileColor.ORANGE),
        (28, TileColor.YELLOW),
        (60, TileColor.GREEN),
        (105, TileColor.BLUE),
        (150, TileColor.PURPLE),
        (177, TileColor.RED),
    ],
)
def test_classifies_base_color_families(hue: int, expected: TileColor) -> None:
    image, cell = _cell_image(hue)

    observation, diagnostics = TileClassifier().classify_cell(image, cell)

    assert observation.color is expected
    assert observation.confidence >= 0.48
    assert diagnostics.hue_histogram.shape == (180,)


def test_returns_unknown_when_cell_has_too_little_foreground() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:] = (35, 42, 108)
    cell = Cell(0, 0, Point(40, 40), Rect(0, 0, 80, 80), 1.0)

    observation, _ = TileClassifier().classify_cell(image, cell)

    assert observation.color is TileColor.UNKNOWN


def test_orange_gradient_is_not_rejected_as_ambiguous() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:] = (35, 42, 108)
    cell = Cell(0, 0, Point(40, 40), Rect(0, 0, 80, 80), 1.0)

    # Recreate the game's orange gradient: a red-orange lower region and a
    # brighter orange upper region. Orange remains the plurality, but red is a
    # substantial runner-up.
    orange = _rgb_from_hsv(16)
    red_orange = _rgb_from_hsv(5)
    cv2.circle(image, (40, 40), 25, orange, -1)
    # Keep the red-orange band substantial but smaller than the orange body.
    # The previous fixture started at y=40 and accidentally painted more than
    # half of the circular tile red, contradicting the test's own premise.
    cv2.rectangle(image, (15, 50), (65, 65), red_orange, -1)

    observation, diagnostics = TileClassifier().classify_cell(image, cell)

    assert diagnostics.class_scores[TileColor.ORANGE] > diagnostics.class_scores[TileColor.RED]
    assert diagnostics.class_scores[TileColor.RED] > 0.15
    assert observation.color is TileColor.ORANGE
    assert observation.confidence >= 0.50


def test_orange_plurality_confidence_matches_real_gradient_case() -> None:
    classifier = TileClassifier()
    scores = {
        TileColor.RED: 0.43,
        TileColor.ORANGE: 0.54,
        TileColor.YELLOW: 0.02,
        TileColor.GREEN: 0.00,
        TileColor.BLUE: 0.01,
        TileColor.PURPLE: 0.00,
    }

    color, confidence = classifier._select_color(scores, foreground_fraction=0.93)

    assert color is TileColor.ORANGE
    assert confidence == pytest.approx(0.54)


def test_shape_foreground_is_not_clipped_by_color_circle() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:] = (35, 42, 108)
    cell = Cell(0, 0, Point(40, 40), Rect(0, 0, 80, 80), 1.0)

    green = _rgb_from_hsv(60)
    # Elongated object deliberately extends beyond the circular color mask.
    cv2.ellipse(image, (40, 40), (32, 14), -30, 0, 360, green, -1)

    _, diagnostics = TileClassifier().classify_cell(image, cell)

    assert np.count_nonzero(diagnostics.shape_foreground_mask) > np.count_nonzero(
        diagnostics.foreground_mask
    )
    assert diagnostics.shape_foreground_mask[20, 58] > 0 or diagnostics.shape_foreground_mask[60, 22] > 0


def test_different_color_overlay_is_separated_from_base_shape() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:] = (35, 42, 108)
    cell = Cell(0, 0, Point(40, 40), Rect(0, 0, 80, 80), 1.0)

    purple = _rgb_from_hsv(150)
    yellow = _rgb_from_hsv(28)
    cv2.rectangle(image, (18, 18), (62, 62), purple, -1)
    cv2.line(image, (10, 65), (70, 15), yellow, 8)

    observation, diagnostics = TileClassifier().classify_cell(image, cell)

    assert observation.color is TileColor.PURPLE
    assert np.count_nonzero(diagnostics.shape_foreground_mask) > 0
    assert np.count_nonzero(diagnostics.overlay_foreground_mask) > 0


def test_blocker_crossing_center_keeps_large_base_components() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:] = (35, 42, 108)
    cell = Cell(0, 0, Point(40, 40), Rect(0, 0, 80, 80), 1.0)

    purple = _rgb_from_hsv(150)
    yellow = _rgb_from_hsv(28)
    cv2.rectangle(image, (15, 15), (65, 65), purple, -1)
    cv2.line(image, (6, 70), (74, 10), yellow, 11)

    observation, diagnostics = TileClassifier().classify_cell(image, cell)

    assert observation.color is TileColor.PURPLE
    # The reconstructed mask should contain both sides of the tile, not just
    # one ~878 px half split by the synthetic blocker.
    assert np.count_nonzero(diagnostics.shape_foreground_mask) > 1800
    assert np.count_nonzero(diagnostics.overlay_foreground_mask) > 200
