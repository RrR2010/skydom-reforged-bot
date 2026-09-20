"""Stable tile-recognition contract and classical baseline implementation."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from skydom_bot.domain.board import BoardGeometry
from skydom_bot.domain.tile_state import TileStateEstimate
from skydom_bot.vision.shape_features import extract_shape_features
from skydom_bot.vision.tile_classifier import TileClassifier
from skydom_bot.vision.tile_semantics import TileAppearance, TileSemanticClassifier

UInt8Image = NDArray[np.uint8]


class TileRecognizer(Protocol):
    """Recognizer contract consumed by world-state code."""

    def recognize_board(
        self,
        image_rgb: UInt8Image,
        geometry: BoardGeometry,
    ) -> tuple[TileStateEstimate, ...]:
        """Return one semantic estimate for every active logical cell."""
        ...


class ClassicalTileRecognizer:
    """Expose the current interpretable CV stack through the stable contract."""

    def __init__(
        self,
        tile_classifier: TileClassifier | None = None,
        semantic_classifier: TileSemanticClassifier | None = None,
    ) -> None:
        self.tile_classifier = tile_classifier or TileClassifier()
        self.semantic_classifier = semantic_classifier or TileSemanticClassifier()

    def recognize_board(
        self,
        image_rgb: UInt8Image,
        geometry: BoardGeometry,
    ) -> tuple[TileStateEstimate, ...]:
        """Run color, shape, and semantic inference for the whole board."""
        appearances: list[TileAppearance] = []
        for cell in geometry.cells:
            observation, diagnostics = self.tile_classifier.classify_cell(image_rgb, cell)
            shape = extract_shape_features(
                diagnostics.crop_rgb,
                diagnostics.shape_foreground_mask,
            )
            appearances.append(TileAppearance(observation, diagnostics, shape))

        semantics = self.semantic_classifier.classify_board(tuple(appearances))
        return tuple(
            TileStateEstimate(
                row=appearance.observation.row,
                col=appearance.observation.col,
                color=appearance.observation.color,
                color_confidence=appearance.observation.confidence,
                kind=semantic.kind,
                kind_confidence=semantic.kind_confidence,
                blocker=semantic.blocker,
                blocker_confidence=semantic.blocker_confidence,
                powerup=semantic.powerup,
                powerup_confidence=semantic.powerup_confidence,
            )
            for appearance, semantic in zip(appearances, semantics)
        )
