"""Tests for Label Studio batch storage export format."""

from __future__ import annotations

import json

from skydom_bot.dataset.label_studio import LABEL_CONFIG, export_label_studio_storage


def _write_record(records, images, sample_id: str = "abc123") -> None:
    payload = {
        "sample_id": sample_id,
        "image": f"input/images/{sample_id}.png",
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
    (images / f"{sample_id}.png").write_bytes(b"png")


def test_label_config_contains_non_applicable_and_adjacent_clear_choices() -> None:
    assert '<Choice value="none"/>' in LABEL_CONFIG
    assert '<Choice value="adjacent-clear"/>' in LABEL_CONFIG

    color_section = LABEL_CONFIG.split('<Choices name="color"', 1)[1].split("</Choices>", 1)[0]
    kind_section = LABEL_CONFIG.split('<Choices name="kind"', 1)[1].split("</Choices>", 1)[0]
    assert '<Choice value="none"/>' in color_section
    assert '<Choice value="none"/>' in kind_section


def test_export_creates_batch_input_images_and_target_dir(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    records = dataset / "records"
    images = dataset / "input" / "images"
    records.mkdir(parents=True)
    images.mkdir(parents=True)
    _write_record(records, images)

    summary = export_label_studio_storage(dataset)

    assert summary.total_samples == 1
    assert summary.new_samples == 1
    assert summary.existing_samples == 0
    assert summary.copied_images == 0
    assert summary.config_path.exists()
    assert summary.output_dir.exists()
    assert summary.batch_path is not None
    assert summary.batch_path.name == "batch-tasks-0001.json"
    assert (summary.images_dir / "abc123.png").exists()

    tasks = json.loads(summary.batch_path.read_text(encoding="utf-8"))
    assert len(tasks) == 1
    task = tasks[0]
    assert task["data"]["image"] == "/data/local-files/?d=input/images/abc123.png"
    assert task["data"]["sample_id"] == "abc123"
    assert task["predictions"][0]["model_version"] == "classical-bootstrap"
    results = task["predictions"][0]["result"]
    assert {item["from_name"] for item in results} == {
        "color", "kind", "blocker", "powerup"
    }


def test_export_is_incremental_and_creates_new_batch_only_for_new_samples(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    records = dataset / "records"
    images = dataset / "input" / "images"
    records.mkdir(parents=True)
    images.mkdir(parents=True)
    _write_record(records, images, "first")

    first = export_label_studio_storage(dataset)
    assert first.batch_path is not None
    first_content = first.batch_path.read_text(encoding="utf-8")

    _write_record(records, images, "second")
    second = export_label_studio_storage(dataset)

    assert second.total_samples == 2
    assert second.new_samples == 1
    assert second.existing_samples == 1
    assert second.batch_path is not None
    assert second.batch_path.name == "batch-tasks-0002.json"
    assert first.batch_path.read_text(encoding="utf-8") == first_content

    second_tasks = json.loads(second.batch_path.read_text(encoding="utf-8"))
    assert [task["data"]["sample_id"] for task in second_tasks] == ["second"]


def test_export_with_no_new_samples_does_not_create_empty_batch(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    records = dataset / "records"
    images = dataset / "input" / "images"
    records.mkdir(parents=True)
    images.mkdir(parents=True)
    _write_record(records, images, "only")

    first = export_label_studio_storage(dataset)
    second = export_label_studio_storage(dataset)

    assert first.batch_path is not None
    assert second.batch_path is None
    assert second.new_samples == 0
    assert second.existing_samples == 1


def test_export_migrates_legacy_crop_once_and_normalizes_record_path(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    records = dataset / "records"
    legacy_images = dataset / "images"
    records.mkdir(parents=True)
    legacy_images.mkdir()

    payload = {
        "sample_id": "legacy",
        "image": "images/legacy.png",
        "row": 1,
        "col": 2,
        "source": "legacy",
        "suggested": {},
        "labels": None,
    }
    record_path = records / "legacy.json"
    record_path.write_text(json.dumps(payload), encoding="utf-8")
    (legacy_images / "legacy.png").write_bytes(b"png")

    summary = export_label_studio_storage(dataset)

    assert summary.copied_images == 1
    assert (dataset / "input" / "images" / "legacy.png").exists()
    normalized = json.loads(record_path.read_text(encoding="utf-8"))
    assert normalized["image"] == "input/images/legacy.png"
