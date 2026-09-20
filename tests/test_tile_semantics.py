"""Tests for interpretable tile semantic classification."""

from __future__ import annotations

import numpy as np

from skydom_bot.domain.tile import TileColor, TileObservation
from skydom_bot.domain.tile_semantics import TileBlocker, TileKind, TilePowerup
from skydom_bot.vision.shape_features import ShapeDiagnostics, ShapeFeatures
from skydom_bot.vision.tile_classifier import TileDiagnostics
from skydom_bot.vision.tile_semantics import TileAppearance, TileSemanticClassifier


def _appearance(
    *,
    row: int,
    color: TileColor,
    area: float,
    circularity: float,
    oriented_aspect: float,
    solidity: float,
    centroid_offset: float,
    residual_fraction: float,
) -> TileAppearance:
    size = 20
    residual = np.zeros((size, size), dtype=np.uint8)
    residual.flat[: int(round(size * size * residual_fraction))] = 255
    empty_rgb = np.zeros((size, size, 3), dtype=np.uint8)
    empty_mask = np.zeros((size, size), dtype=np.uint8)

    observation = TileObservation(
        row=row,
        col=0,
        color=color,
        confidence=0.95,
        dominant_hue=60.0,
        foreground_fraction=0.8,
    )
    diagnostics = TileDiagnostics(
        crop_rgb=empty_rgb,
        crop_hsv=empty_rgb,
        center_mask=empty_mask,
        foreground_mask=empty_mask,
        shape_foreground_mask=empty_mask,
        overlay_foreground_mask=residual,
        neutral_overlay_mask=np.zeros_like(empty_mask),
        background_distance_mask=np.zeros_like(empty_mask),
        hue_histogram=np.zeros(180, dtype=np.float64),
        class_scores={},
        dominant_hue=60.0,
    )
    features = ShapeFeatures(
        component_count=1,
        hole_count=0,
        area_fraction=area,
        perimeter=100.0,
        circularity=circularity,
        aspect_ratio=1.0,
        oriented_aspect_ratio=oriented_aspect,
        orientation_deg=0.0,
        extent=0.8,
        solidity=solidity,
        centroid_offset=centroid_offset,
    )
    shape = ShapeDiagnostics(
        contour_mask=empty_mask,
        contour_overlay=empty_rgb,
        features=features,
    )
    return TileAppearance(observation, diagnostics, shape)


def test_clean_elongated_piece_is_recognized_as_carrot_not_chain() -> None:
    normals = (
        _appearance(
            row=0,
            color=TileColor.GREEN,
            area=0.50,
            circularity=0.78,
            oriented_aspect=1.05,
            solidity=0.96,
            centroid_offset=0.02,
            residual_fraction=0.01,
        ),
        _appearance(
            row=1,
            color=TileColor.GREEN,
            area=0.51,
            circularity=0.77,
            oriented_aspect=1.08,
            solidity=0.95,
            centroid_offset=0.02,
            residual_fraction=0.01,
        ),
    )
    carrot = _appearance(
        row=2,
        color=TileColor.GREEN,
        area=0.39,
        circularity=0.63,
        oriented_aspect=2.02,
        solidity=0.91,
        centroid_offset=0.021,
        residual_fraction=0.01,
    )

    results = TileSemanticClassifier().classify_board(normals + (carrot,))
    result = results[-1]

    assert result.kind is TileKind.CARROT
    assert result.blocker is TileBlocker.NONE


def test_different_color_chain_is_recognized_from_residual_and_shape_degradation() -> None:
    normals = (
        _appearance(
            row=0,
            color=TileColor.PURPLE,
            area=0.48,
            circularity=0.73,
            oriented_aspect=1.01,
            solidity=0.94,
            centroid_offset=0.01,
            residual_fraction=0.01,
        ),
        _appearance(
            row=1,
            color=TileColor.PURPLE,
            area=0.49,
            circularity=0.72,
            oriented_aspect=1.02,
            solidity=0.95,
            centroid_offset=0.01,
            residual_fraction=0.01,
        ),
    )
    chained = _appearance(
        row=2,
        color=TileColor.PURPLE,
        area=0.21,
        circularity=0.56,
        oriented_aspect=1.82,
        solidity=0.82,
        centroid_offset=0.147,
        residual_fraction=0.22,
    )

    result = TileSemanticClassifier().classify_board(normals + (chained,))[-1]

    assert result.blocker is TileBlocker.CHAIN
    assert result.kind is not TileKind.CARROT


def test_same_color_chain_can_be_detected_from_peer_anomaly() -> None:
    normals = (
        _appearance(
            row=0,
            color=TileColor.YELLOW,
            area=0.56,
            circularity=0.77,
            oriented_aspect=1.01,
            solidity=0.95,
            centroid_offset=0.015,
            residual_fraction=0.01,
        ),
        _appearance(
            row=1,
            color=TileColor.YELLOW,
            area=0.55,
            circularity=0.76,
            oriented_aspect=1.02,
            solidity=0.95,
            centroid_offset=0.016,
            residual_fraction=0.01,
        ),
        _appearance(
            row=2,
            color=TileColor.YELLOW,
            area=0.57,
            circularity=0.78,
            oriented_aspect=1.01,
            solidity=0.94,
            centroid_offset=0.014,
            residual_fraction=0.01,
        ),
    )
    chained = _appearance(
        row=3,
        color=TileColor.YELLOW,
        area=0.67,
        circularity=0.58,
        oriented_aspect=1.02,
        solidity=0.88,
        centroid_offset=0.015,
        residual_fraction=0.03,
    )

    result = TileSemanticClassifier().classify_board(normals + (chained,))[-1]

    assert result.blocker is TileBlocker.CHAIN


def test_bright_neutral_overlay_is_reported_as_unknown_powerup_candidate() -> None:
    appearance = _appearance(
        row=0,
        color=TileColor.ORANGE,
        area=0.55,
        circularity=0.80,
        oriented_aspect=1.05,
        solidity=0.96,
        centroid_offset=0.02,
        residual_fraction=0.01,
    )

    neutral = np.zeros((20, 20), dtype=np.uint8)
    neutral.flat[:40] = 255  # 10% of the cell
    diagnostics = TileDiagnostics(
        crop_rgb=appearance.diagnostics.crop_rgb,
        crop_hsv=appearance.diagnostics.crop_hsv,
        center_mask=appearance.diagnostics.center_mask,
        foreground_mask=appearance.diagnostics.foreground_mask,
        shape_foreground_mask=appearance.diagnostics.shape_foreground_mask,
        overlay_foreground_mask=appearance.diagnostics.overlay_foreground_mask,
        neutral_overlay_mask=neutral,
        background_distance_mask=appearance.diagnostics.background_distance_mask,
        hue_histogram=appearance.diagnostics.hue_histogram,
        class_scores=appearance.diagnostics.class_scores,
        dominant_hue=appearance.diagnostics.dominant_hue,
    )
    appearance = TileAppearance(
        appearance.observation,
        diagnostics,
        appearance.shape,
    )

    result = TileSemanticClassifier().classify_board((appearance,))[0]

    assert result.powerup is TilePowerup.UNKNOWN
    assert result.blocker is TileBlocker.NONE
