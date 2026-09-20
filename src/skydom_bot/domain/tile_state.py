"""Recognizer-neutral semantic state for one logical board cell."""

from __future__ import annotations

from dataclasses import dataclass

from skydom_bot.domain.tile import TileColor
from skydom_bot.domain.tile_semantics import TileBlocker, TileKind, TilePowerup


@dataclass(frozen=True, slots=True)
class TileStateEstimate:
    """Final semantic estimate independent from the recognition implementation."""

    row: int
    col: int
    color: TileColor
    color_confidence: float
    kind: TileKind
    kind_confidence: float
    blocker: TileBlocker
    blocker_confidence: float
    powerup: TilePowerup
    powerup_confidence: float
