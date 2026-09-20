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
