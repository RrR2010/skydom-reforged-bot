"""Tests for importing Label Studio human annotations."""

from __future__ import annotations

import json

from skydom_bot.dataset.label_studio_import import import_label_studio_annotations


def _annotation_task(sample_id: str, *, updated_at: str = "2026-09-20T02:00:00Z"):
    return {
        "id": 10,
        "data": {
            "image": f"/data/local-files/?d=input/images/{sample_id}.png",
            "sample_id": sample_id,
        },
        "annotations": [
            {
                "id": 100,
                "updated_at": updated_at,
                "result": [
                    {"from_name": "color", "to_name": "image", "type": "choices", "value": {"choices": ["blue"]}},
                    {"from_name": "kind", "to_name": "image", "type": "choices", "value": {"choices": ["carrot"]}},
                    {"from_name": "blocker", "to_name": "image", "type": "choices", "value": {"choices": ["none"]}},
                    {"from_name": "powerup", "to_name": "image", "type": "choices", "value": {"choices": ["none"]}},
                ],
            }
        ],
    }


def test_import_merges_human_labels_without_overwriting_suggested(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    records = dataset / "records"
    annotations = dataset / "output" / "annotations"
    records.mkdir(parents=True)
    annotations.mkdir(parents=True)

    record = {
        "sample_id": "abc",
        "image": "input/images/abc.png",
        "source": "test",
        "suggested": {"color": {"value": "green", "confidence": 0.75}},
        "labels": None,
    }
    record_path = records / "abc.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    (annotations / "task_10.json").write_text(json.dumps(_annotation_task("abc")), encoding="utf-8")

    summary = import_label_studio_annotations(dataset)

    assert summary.records_updated == 1
    updated = json.loads(record_path.read_text(encoding="utf-8"))
    assert updated["suggested"] == record["suggested"]
    assert updated["labels"] == {
        "color": "blue",
        "kind": "carrot",
        "blocker": "none",
        "powerup": "none",
    }


def test_import_prefers_latest_annotation(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    records = dataset / "records"
    annotations = dataset / "output" / "annotations"
    records.mkdir(parents=True)
    annotations.mkdir(parents=True)

    record_path = records / "abc.json"
    record_path.write_text(json.dumps({"sample_id": "abc", "labels": None, "suggested": {}}), encoding="utf-8")

    task = _annotation_task("abc", updated_at="2026-09-20T02:00:00Z")
    newer = _annotation_task("abc", updated_at="2026-09-20T03:00:00Z")["annotations"][0]
    for result in newer["result"]:
        if result["from_name"] == "color":
            result["value"]["choices"] = ["purple"]
    task["annotations"].append(newer)
    (annotations / "task_10.json").write_text(json.dumps(task), encoding="utf-8")

    import_label_studio_annotations(dataset)
    updated = json.loads(record_path.read_text(encoding="utf-8"))
    assert updated["labels"]["color"] == "purple"


def test_import_skips_partial_annotation(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    records = dataset / "records"
    annotations = dataset / "output" / "annotations"
    records.mkdir(parents=True)
    annotations.mkdir(parents=True)

    (records / "abc.json").write_text(json.dumps({"sample_id": "abc", "labels": None}), encoding="utf-8")
    task = _annotation_task("abc")
    task["annotations"][0]["result"] = task["annotations"][0]["result"][:2]
    (annotations / "task_10.json").write_text(json.dumps(task), encoding="utf-8")

    summary = import_label_studio_annotations(dataset)
    assert summary.records_updated == 0
    assert summary.skipped_without_annotation == 1
