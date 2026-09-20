"""Tests for Label Studio export format."""

from __future__ import annotations

import json

from skydom_bot.dataset.label_studio import export_label_studio


def test_export_creates_local_file_task_and_bootstrap_predictions(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    records = dataset / "records"
    records.mkdir(parents=True)
    (dataset / "images").mkdir()

    payload = {
        "sample_id": "abc123",
        "image": "images/abc123.png",
        "row": 2,
        "col": 3,
        "source": "tile-debugger",
        "suggested": {
            "color": {"value": "blue", "confidence": 0.99},
            "kind": {"value": "carrot", "confidence": 0.90},
            "blocker": {"value": "none", "confidence": 0.95},
            "powerup": {"value": "none", "confidence": 0.98},
        },
        "labels": None,
    }
    (records / "abc123.json").write_text(json.dumps(payload), encoding="utf-8")

    config_path, tasks_path, count = export_label_studio(dataset)

    assert count == 1
    assert config_path.exists()
    task = json.loads(tasks_path.read_text(encoding="utf-8"))[0]
    assert task["data"]["image"] == "/data/local-files/?d=images/abc123.png"
    assert task["data"]["sample_id"] == "abc123"
    assert task["predictions"][0]["model_version"] == "classical-bootstrap"
    results = task["predictions"][0]["result"]
    assert {item["from_name"] for item in results} == {
        "color", "kind", "blocker", "powerup"
    }
