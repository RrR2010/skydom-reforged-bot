"""Tests for manual screen-corpus capture and offline analysis."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from skydom_bot.corpus import analysis
from skydom_bot.corpus.capture import (
    CaptureRegion,
    crop_region,
    load_region,
    parse_metadata,
    parse_region,
    save_capture,
    save_region,
)


def test_region_configuration_round_trip_and_crop(tmp_path) -> None:
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    region = parse_region("10,20,80,50")

    save_region(tmp_path, region, monitor=2)
    monitor, loaded = load_region(tmp_path)
    cropped = crop_region(image, loaded)

    assert monitor == 2
    assert loaded == CaptureRegion(10, 20, 80, 50)
    assert cropped.shape == (50, 80, 3)


def test_parse_metadata_accepts_repeated_key_value_pairs() -> None:
    assert parse_metadata(
        ["mode=competitive", "goal=carrot", "note=rare layout"]
    ) == {
        "mode": "competitive",
        "goal": "carrot",
        "note": "rare layout",
    }


def test_save_capture_writes_game_crop_and_metadata(tmp_path) -> None:
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    image[20:70, 30:110] = (10, 20, 30)

    record = save_capture(
        tmp_path,
        stage_id="phase 27",
        monitor=1,
        region=CaptureRegion(30, 20, 80, 50),
        metadata={"mode": "normal"},
        image_rgb=image,
    )

    assert record.stage_id == "phase 27"
    assert record.capture_id.startswith("phase-27__")
    assert record.metadata == {"mode": "normal"}
    assert (tmp_path / record.image).is_file()

    payload = json.loads(
        (tmp_path / "metadata" / f"{record.capture_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["stage_id"] == "phase 27"
    assert payload["region"] == {
        "x": 30,
        "y": 20,
        "width": 80,
        "height": 50,
    }


def test_analyze_corpus_builds_review_queue_from_detector_signals(
    tmp_path,
    monkeypatch,
) -> None:
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    record = save_capture(
        tmp_path,
        stage_id="competitive-1",
        monitor=1,
        region=CaptureRegion(0, 0, 160, 100),
        metadata={"mode": "competitive"},
        image_rgb=image,
    )

    geometry = SimpleNamespace(
        rows=9,
        cols=9,
        cells=tuple(range(72)),
        pitch_x=65.0,
        pitch_y=65.2,
        confidence=0.99,
    )
    diagnostics = SimpleNamespace(
        ambiguous_topology_components=(),
        local_subgrid_topology=np.array([[True, False]], dtype=np.bool_),
    )

    class FakeDetector:
        def detect_with_diagnostics(self, image_rgb):
            return geometry, diagnostics

    monkeypatch.setattr(analysis, "BoardDetector", FakeDetector)

    results = analysis.analyze_corpus(tmp_path)
    assert len(results) == 1
    assert results[0].capture_id == record.capture_id
    assert results[0].status == "REVIEW"
    assert results[0].reasons == ("scaled-subgrid-detected",)

    jsonl_path, csv_path, markdown_path = analysis.write_analysis_outputs(
        tmp_path,
        results,
    )
    assert jsonl_path.is_file()
    assert csv_path.is_file()
    assert markdown_path.is_file()
    assert "competitive-1" in markdown_path.read_text(encoding="utf-8")
    assert "scaled-subgrid-detected" in markdown_path.read_text(encoding="utf-8")
