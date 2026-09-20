"""Tests for the recognizer-neutral semantic contract."""

from __future__ import annotations

import cv2
import numpy as np

from skydom_bot.domain.board import BoardGeometry, Cell, Point, Rect
from skydom_bot.domain.tile import TileColor
from skydom_bot.domain.tile_semantics import TileBlocker, TileKind, TilePowerup
from skydom_bot.vision.recognizer import ClassicalTileRecognizer


def test_classical_recognizer_emits_semantic_contract() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:] = (35, 42, 108)
    cv2.circle(image, (40, 40), 24, (20, 220, 40), -1)

    cell = Cell(0, 0, Point(40, 40), Rect(0, 0, 80, 80), 1.0)
    geometry = BoardGeometry(
        bounds=Rect(0, 0, 80, 80),
        rows=1,
        cols=1,
        pitch_x=80.0,
        pitch_y=80.0,
        cells=(cell,),
        confidence=1.0,
    )

    result = ClassicalTileRecognizer().recognize_board(image, geometry)[0]

    assert result.row == 0
    assert result.col == 0
    assert result.color is TileColor.GREEN
    assert result.kind in {TileKind.NORMAL, TileKind.UNKNOWN}
    assert result.blocker in {TileBlocker.NONE, TileBlocker.UNKNOWN}
    assert result.powerup in {TilePowerup.NONE, TilePowerup.UNKNOWN}
