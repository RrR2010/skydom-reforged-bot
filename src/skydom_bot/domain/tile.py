"""Domain models for recognized Match-3 tiles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TileColor(str, Enum):
    """Color family used by the base Match-3 pieces."""

    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    PURPLE = "purple"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TileObservation:
    """Per-cell color observation produced by the vision layer."""

    row: int
    col: int
    color: TileColor
    confidence: float
    dominant_hue: float | None
    foreground_fraction: float
