"""Tests for Label Studio storage export format."""

from __future__ import annotations

import json

from skydom_bot.dataset.label_studio import export_label_studio_storage


def _write_record(records, sample_id: str = "abc123") -> None:
    payload = {
        "sample_id": sample_id,
        "image": f"images/{sample_id}.png",
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
    (records / f"{sample_id}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_export_creates_per_sample_storage_task_and_target_dir(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    records = dataset / "records"
    records.mkdir(parents=True)
    (dataset / "images").mkdir()
    _write_record(records)

    summary = export_label_studio_storage(dataset)

    assert summary.total == 1
    assert summary.created == 1
    assert summary.existing == 0
    assert summary.config_path.exists()
    assert summary.target_dir.exists()

    task_path = summary.source_tasks_dir / "abc123.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    assert task["data"]["image"] == "/data/local-files/?d=dataset/images/abc123.png"
    assert task["data"]["sample_id"] == "abc123"
    assert task["predictions"][0]["model_version"] == "classical-bootstrap"
    results = task["predictions"][0]["result"]
    assert {item["from_name"] for item in results} == {
        "color", "kind", "blocker", "powerup"
    }


def test_export_is_incremental_and_does_not_rewrite_existing_task(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    records = dataset / "records"
    records.mkdir(parents=True)
    (dataset / "images").mkdir()
    _write_record(records, "first")

    first = export_label_studio_storage(dataset)
    first_task = first.source_tasks_dir / "first.json"
    original = first_task.read_text(encoding="utf-8")

    # Simulate an already-published immutable source task.
    first_task.write_text(original + "\n", encoding="utf-8")
    _write_record(records, "second")

    second = export_label_studio_storage(dataset)

    assert second.total == 2
    assert second.created == 1
    assert second.existing == 1
    assert first_task.read_text(encoding="utf-8") == original + "\n"
    assert (second.source_tasks_dir / "second.json").exists()
