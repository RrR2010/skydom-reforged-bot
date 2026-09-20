"""Interpretable semantic classification for Match-3 tile roles and blockers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skydom_bot.domain.tile import TileColor, TileObservation
from skydom_bot.domain.tile_semantics import (
    TileBlocker,
    TileKind,
    TileSemanticObservation,
)
from skydom_bot.vision.shape_features import ShapeDiagnostics
from skydom_bot.vision.tile_classifier import TileDiagnostics


@dataclass(frozen=True, slots=True)
class TileAppearance:
    """All lower-level evidence needed for board-relative semantic analysis."""

    observation: TileObservation
    diagnostics: TileDiagnostics
    shape: ShapeDiagnostics


@dataclass(frozen=True, slots=True)
class TileSemanticConfig:
    """Conservative thresholds for semantic labels."""

    carrot_min_oriented_aspect: float = 1.65
    carrot_max_centroid_offset: float = 0.09
    carrot_min_solidity: float = 0.86
    carrot_max_residual_fraction: float = 0.06

    chain_min_residual_fraction: float = 0.08
    chain_min_shape_anomaly: float = 2.25
    chain_low_solidity: float = 0.90
    chain_low_circularity: float = 0.64

    min_peer_count: int = 2
    robust_scale_floor: float = 0.035


class TileSemanticClassifier:
    """Fuse base shape, residual overlay, and board-relative anomaly evidence.

    The classifier intentionally stays conservative. It assigns semantic labels
    only when several interpretable signals agree; ambiguous cells remain
    UNKNOWN rather than forcing a brittle rule.
    """

    def __init__(self, config: TileSemanticConfig | None = None) -> None:
        self.config = config or TileSemanticConfig()

    @staticmethod
    def _residual_fraction(appearance: TileAppearance) -> float:
        mask = appearance.diagnostics.overlay_foreground_mask
        return float(np.count_nonzero(mask) / max(1, mask.size))

    @staticmethod
    def _feature_vector(appearance: TileAppearance) -> np.ndarray:
        f = appearance.shape.features
        return np.array(
            [
                f.area_fraction,
                f.circularity,
                f.oriented_aspect_ratio,
                f.solidity,
                f.centroid_offset,
            ],
            dtype=np.float64,
        )

    def _shape_anomaly(
        self,
        appearance: TileAppearance,
        peers: tuple[TileAppearance, ...],
    ) -> float:
        if len(peers) < self.config.min_peer_count:
            return 0.0

        matrix = np.stack([self._feature_vector(peer) for peer in peers], axis=0)
        median = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - median), axis=0)

        # MAD is robust to one chained/special piece in an otherwise normal
        # color family. A small floor prevents perfectly identical synthetic
        # samples from producing infinite anomaly.
        scale = np.maximum(1.4826 * mad, self.config.robust_scale_floor)
        distance = np.abs(self._feature_vector(appearance) - median) / scale
        return float(np.median(distance))

    def classify_board(
        self,
        appearances: tuple[TileAppearance, ...],
    ) -> tuple[TileSemanticObservation, ...]:
        """Classify semantics using same-color peers as adaptive prototypes."""
        by_color: dict[TileColor, list[TileAppearance]] = {}
        for appearance in appearances:
            by_color.setdefault(appearance.observation.color, []).append(appearance)

        results: list[TileSemanticObservation] = []
        for appearance in appearances:
            color = appearance.observation.color
            peers = tuple(
                peer
                for peer in by_color.get(color, [])
                if peer is not appearance
            )
            results.append(self._classify_one(appearance, peers))
        return tuple(results)

    def _classify_one(
        self,
        appearance: TileAppearance,
        peers: tuple[TileAppearance, ...],
    ) -> TileSemanticObservation:
        f = appearance.shape.features
        residual = self._residual_fraction(appearance)
        anomaly = self._shape_anomaly(appearance, peers)

        carrot_signals = (
            f.oriented_aspect_ratio >= self.config.carrot_min_oriented_aspect,
            f.centroid_offset <= self.config.carrot_max_centroid_offset,
            f.solidity >= self.config.carrot_min_solidity,
            residual <= self.config.carrot_max_residual_fraction,
        )
        carrot_score = sum(carrot_signals) / len(carrot_signals)

        if all(carrot_signals):
            kind = TileKind.CARROT
            kind_confidence = float(
                min(
                    1.0,
                    0.55
                    + 0.20 * min(1.0, (f.oriented_aspect_ratio - 1.0) / 1.5)
                    + 0.15 * f.solidity
                    + 0.10 * (1.0 - min(1.0, f.centroid_offset / 0.15)),
                )
            )
        elif anomaly < 1.5 and residual < self.config.chain_min_residual_fraction:
            kind = TileKind.NORMAL
            kind_confidence = float(np.clip(1.0 - anomaly / 3.0, 0.5, 0.95))
        else:
            kind = TileKind.UNKNOWN
            kind_confidence = float(max(0.0, carrot_score - 0.25))

        chain_signals = (
            residual >= self.config.chain_min_residual_fraction,
            anomaly >= self.config.chain_min_shape_anomaly,
            f.solidity <= self.config.chain_low_solidity,
            f.circularity <= self.config.chain_low_circularity,
        )
        chain_votes = sum(chain_signals)

        # Different-color chains usually create a strong residual overlay. A
        # same-color chain may merge into the base mask, so allow shape anomaly
        # + degraded compactness to identify it even with weak residual.
        if chain_votes >= 2 and (
            chain_signals[0]
            or (chain_signals[1] and (chain_signals[2] or chain_signals[3]))
        ):
            blocker = TileBlocker.CHAIN
            blocker_confidence = float(min(1.0, 0.45 + 0.14 * chain_votes))
        elif residual < 0.03 and anomaly < 1.5:
            blocker = TileBlocker.NONE
            blocker_confidence = float(np.clip(1.0 - anomaly / 3.0, 0.55, 0.98))
        else:
            blocker = TileBlocker.UNKNOWN
            blocker_confidence = float(0.35 + 0.10 * chain_votes)

        # A clean carrot is an objective piece, not a chain merely because its
        # elongated geometry is anomalous relative to normal green pieces.
        if kind is TileKind.CARROT and residual <= self.config.carrot_max_residual_fraction:
            blocker = TileBlocker.NONE
            blocker_confidence = max(blocker_confidence, 0.90)

        return TileSemanticObservation(
            row=appearance.observation.row,
            col=appearance.observation.col,
            kind=kind,
            kind_confidence=kind_confidence,
            blocker=blocker,
            blocker_confidence=blocker_confidence,
            residual_fraction=residual,
            shape_anomaly=anomaly,
        )
