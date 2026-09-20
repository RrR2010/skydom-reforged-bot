"""Geometric descriptors for tile appearance analysis."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

UInt8Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ShapeFeatures:
    """Classical contour descriptors extracted from a binary foreground mask."""

    component_count: int
    hole_count: int
    area_fraction: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    extent: float
    solidity: float
    centroid_offset: float


@dataclass(frozen=True, slots=True)
class ShapeDiagnostics:
    """Intermediate contour data for visual debugging."""

    contour_mask: UInt8Image
    contour_overlay: UInt8Image
    features: ShapeFeatures


def extract_shape_features(
    crop_rgb: UInt8Image,
    foreground_mask: UInt8Image,
) -> ShapeDiagnostics:
    """Extract interpretable geometry from a segmented tile foreground.

    The descriptors intentionally stay model-free. They answer questions such
    as: is the object ring-like, elongated, fragmented, compact, or irregular?
    Those observations will later support special-piece semantics.
    """
    mask = np.where(foreground_mask > 0, 255, 0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        empty = ShapeFeatures(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return ShapeDiagnostics(mask, crop_rgb.copy(), empty)

    hierarchy_array = hierarchy[0] if hierarchy is not None else np.empty((0, 4), dtype=np.int32)
    outer_indices = [
        index
        for index in range(len(contours))
        if hierarchy is None or hierarchy_array[index][3] == -1
    ]
    hole_count = sum(
        1
        for index in range(len(contours))
        if hierarchy is not None and hierarchy_array[index][3] != -1
    )

    outer_contours = [contours[index] for index in outer_indices]
    largest = max(outer_contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    perimeter = float(cv2.arcLength(largest, True))
    x, y, width, height = cv2.boundingRect(largest)
    bbox_area = max(1.0, float(width * height))
    hull = cv2.convexHull(largest)
    hull_area = max(1.0, float(cv2.contourArea(hull)))

    circularity = (
        float(4.0 * np.pi * area / (perimeter * perimeter))
        if perimeter > 0
        else 0.0
    )
    aspect_ratio = float(width / max(1, height))
    extent = float(area / bbox_area)
    solidity = float(area / hull_area)

    moments = cv2.moments(largest)
    if moments["m00"] > 0:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
    else:
        cx = crop_rgb.shape[1] / 2.0
        cy = crop_rgb.shape[0] / 2.0

    image_cx = crop_rgb.shape[1] / 2.0
    image_cy = crop_rgb.shape[0] / 2.0
    diagonal = max(1.0, float(np.hypot(crop_rgb.shape[1], crop_rgb.shape[0])))
    centroid_offset = float(np.hypot(cx - image_cx, cy - image_cy) / diagonal)

    area_fraction = float(
        sum(cv2.contourArea(contour) for contour in outer_contours)
        / max(1, mask.size)
    )

    overlay = crop_rgb.copy()
    cv2.drawContours(overlay, outer_contours, -1, (255, 255, 255), 1)
    cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (255, 255, 255), 1)
    cv2.circle(overlay, (int(round(cx)), int(round(cy))), 2, (255, 255, 255), -1)

    features = ShapeFeatures(
        component_count=len(outer_contours),
        hole_count=hole_count,
        area_fraction=area_fraction,
        perimeter=perimeter,
        circularity=circularity,
        aspect_ratio=aspect_ratio,
        extent=extent,
        solidity=solidity,
        centroid_offset=centroid_offset,
    )
    return ShapeDiagnostics(mask, overlay, features)
