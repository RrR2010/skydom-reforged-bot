"""Tests for classical shape descriptors."""

import cv2
import numpy as np

from skydom_bot.vision.shape_features import extract_shape_features


def _blank() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((80, 80, 3), dtype=np.uint8),
        np.zeros((80, 80), dtype=np.uint8),
    )


def test_ring_exposes_internal_hole() -> None:
    crop, mask = _blank()
    cv2.circle(mask, (40, 40), 24, 255, -1)
    cv2.circle(mask, (40, 40), 10, 0, -1)

    diagnostics = extract_shape_features(crop, mask)

    assert diagnostics.features.component_count == 1
    assert diagnostics.features.hole_count >= 1
    assert diagnostics.features.circularity > 0.70


def test_elongated_shape_has_non_square_aspect_ratio() -> None:
    crop, mask = _blank()
    cv2.ellipse(mask, (40, 40), (27, 10), 30, 0, 360, 255, -1)

    diagnostics = extract_shape_features(crop, mask)

    assert diagnostics.features.component_count == 1
    assert (
        diagnostics.features.aspect_ratio > 1.35
        or diagnostics.features.aspect_ratio < 0.74
    )


def test_fragmented_shape_counts_multiple_components() -> None:
    crop, mask = _blank()
    cv2.circle(mask, (25, 40), 10, 255, -1)
    cv2.circle(mask, (55, 40), 10, 255, -1)

    diagnostics = extract_shape_features(crop, mask)

    assert diagnostics.features.component_count == 2
