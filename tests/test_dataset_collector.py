"""Tests for de-duplicated raw tile dataset collection."""

from __future__ import annotations

import json

import numpy as np

from skydom_bot.dataset.collector import DatasetCollector
from skydom_bot.domain.board import BoardGeometry, Cell, Point, Rect
from skydom_bot.domain.tile import TileColor
from skydom_bot.domain.tile_semantics import TileBlocker, TileKind, TilePowerup
from skydom_bot.domain.tile_state import TileStateEstimate


def _fixture():
    image = np.zeros((40, 80, 3), dtype=np.uint8)
    image[:, :40] = (10, 200, 20)
    image[:, 40:] = (220, 80, 10)

    cells = (
        Cell(0, 0, Point(20, 20), Rect(0, 0, 40, 40), 1.0),
        Cell(0, 1, Point(60, 20), Rect(40, 0, 40, 40), 1.0),
    )
    geometry = BoardGeometry(
        bounds=Rect(0, 0, 80, 40),
        rows=1,
        cols=2,
        pitch_x=40.0,
        pitch_y=40.0,
        cells=cells,
        confidence=1.0,
    )
    estimates = (
        TileStateEstimate(
            0, 0, TileColor.GREEN, 0.99,
            TileKind.NORMAL, 0.90,
            TileBlocker.NONE, 0.95,
            TilePowerup.NONE, 0.95,
        ),
        TileStateEstimate(
            0, 1, TileColor.ORANGE, 0.95,
            TileKind.UNKNOWN, 0.40,
            TileBlocker.NONE, 0.80,
            TilePowerup.UNKNOWN, 0.70,
        ),
    )
    return image, geometry, estimates


def test_collector_saves_images_and_keeps_predictions_separate_from_labels(tmp_path) -> None:
    image, geometry, estimates = _fixture()

    records = DatasetCollector(tmp_path).collect(
        image,
        geometry,
        estimates,
        source="unit-test",
    )

    assert len(records) == 2
    assert all(record.labels is None for record in records)
    assert records[0].suggested["color"]["value"] == "green"

    record_path = tmp_path / "records" / f"{records[0].sample_id}.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["labels"] is None
    assert (tmp_path / payload["image"]).exists()


def test_collector_preserves_existing_human_labels_on_recollection(tmp_path) -> None:
    image, geometry, estimates = _fixture()
    collector = DatasetCollector(tmp_path)
    first = collector.collect(image, geometry, estimates, source="first")[0]

    record_path = tmp_path / "records" / f"{first.sample_id}.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["labels"] = {
        "color": "green",
        "kind": "normal",
        "blocker": "none",
        "powerup": "none",
    }
    record_path.write_text(json.dumps(payload), encoding="utf-8")

    second = collector.collect(image, geometry, estimates, source="second")[0]

    assert second.labels == payload["labels"]
