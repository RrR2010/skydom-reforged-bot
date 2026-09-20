"""Semantic tile states layered on top of color and shape perception."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TileKind(str, Enum):
    """Semantic role of the visible base object in one board cell."""

    NORMAL = "normal"
    CARROT = "carrot"
    UNKNOWN = "unknown"


class TilePowerup(str, Enum):
    """Power-up/special modifier independent from base kind and blocker."""

    NONE = "none"
    FLYER = "flyer"
    ROW = "row"
    COLUMN = "column"
    BOMB = "bomb"
    COLOR_REMOVER = "color-remover"
    UNKNOWN = "unknown"


class TileBlocker(str, Enum):
    """Overlay/blocker state that changes how a tile can be used."""

    NONE = "none"
    CHAIN = "chain"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TileSemanticObservation:
    """High-level interpretation derived from several visual features."""

    row: int
    col: int
    kind: TileKind
    kind_confidence: float
    blocker: TileBlocker
    blocker_confidence: float
    powerup: TilePowerup
    powerup_confidence: float
    residual_fraction: float
    neutral_overlay_fraction: float
    shape_anomaly: float
