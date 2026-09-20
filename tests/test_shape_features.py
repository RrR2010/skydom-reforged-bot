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
    assert diagnostics.features.oriented_aspect_ratio > 2.0


def test_fragmented_shape_counts_multiple_components() -> None:
    crop, mask = _blank()
    cv2.circle(mask, (25, 40), 10, 255, -1)
    cv2.circle(mask, (55, 40), 10, 255, -1)

    diagnostics = extract_shape_features(crop, mask)

    assert diagnostics.features.component_count == 2


def test_tiny_threshold_holes_are_ignored() -> None:
    crop, mask = _blank()
    cv2.rectangle(mask, (15, 15), (65, 65), 255, -1)
    mask[25, 25] = 0
    mask[35, 35] = 0
    mask[45, 45] = 0

    diagnostics = extract_shape_features(crop, mask)

    assert diagnostics.features.hole_count == 0


def test_rotated_shape_keeps_elongation_in_oriented_box() -> None:
    crop, mask = _blank()
    box = cv2.boxPoints(((40.0, 40.0), (54.0, 18.0), 45.0)).astype(np.int32)
    cv2.fillConvexPoly(mask, box, 255)

    diagnostics = extract_shape_features(crop, mask)

    assert 0.8 < diagnostics.features.aspect_ratio < 1.2
    assert diagnostics.features.oriented_aspect_ratio > 2.5
