"""Import human Label Studio annotations into canonical dataset records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_LABEL_FIELDS = ("color", "kind", "blocker", "powerup")


@dataclass(frozen=True, slots=True)
class LabelStudioImportSummary:
    """Counts produced by one annotation import pass."""

    files_seen: int
    tasks_with_sample_id: int
    records_updated: int
    skipped_without_annotation: int
    skipped_missing_record: int
    invalid_files: int


def _latest_annotation(task: dict[str, Any]) -> dict[str, Any] | None:
    annotations = task.get("annotations")
    if not isinstance(annotations, list):
        return None

    valid = [item for item in annotations if isinstance(item, dict)]
    if not valid:
        return None

    def sort_key(item: dict[str, Any]) -> str:
        updated = item.get("updated_at")
        created = item.get("created_at")
        return str(updated or created or "")

    return max(valid, key=sort_key)


def _labels_from_annotation(annotation: dict[str, Any]) -> dict[str, str] | None:
    result = annotation.get("result")
    if not isinstance(result, list):
        return None

    labels: dict[str, str] = {}
    for item in result:
        if not isinstance(item, dict) or item.get("type") != "choices":
            continue
        from_name = item.get("from_name")
        if from_name not in _LABEL_FIELDS:
            continue
        value = item.get("value")
        if not isinstance(value, dict):
            continue
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            continue
        choice = choices[0]
        if isinstance(choice, str):
            labels[str(from_name)] = choice

    if not labels or any(name not in labels for name in _LABEL_FIELDS):
        return None
    return labels


def _sample_id(task: dict[str, Any]) -> str | None:
    data = task.get("data")
    if not isinstance(data, dict):
        return None
    sample_id = data.get("sample_id")
    return sample_id if isinstance(sample_id, str) and sample_id else None


def import_label_studio_annotations(
    dataset_root: Path,
    annotations_dir: Path | None = None,
) -> LabelStudioImportSummary:
    """Merge submitted Label Studio choices into dataset record labels."""
    dataset_root = dataset_root.resolve()
    annotations_dir = (
        annotations_dir or dataset_root / "output" / "annotations"
    ).resolve()
    records_dir = dataset_root / "records"

    files_seen = 0
    tasks_with_sample_id = 0
    records_updated = 0
    skipped_without_annotation = 0
    skipped_missing_record = 0
    invalid_files = 0

    for path in sorted(annotations_dir.glob("*.json")):
        files_seen += 1
        try:
            task = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            invalid_files += 1
            continue

        if not isinstance(task, dict):
            invalid_files += 1
            continue

        sample_id = _sample_id(task)
        if sample_id is None:
            invalid_files += 1
            continue
        tasks_with_sample_id += 1

        annotation = _latest_annotation(task)
        labels = _labels_from_annotation(annotation) if annotation else None
        if labels is None:
            skipped_without_annotation += 1
            continue

        record_path = records_dir / f"{sample_id}.json"
        if not record_path.exists():
            skipped_missing_record += 1
            continue

        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            invalid_files += 1
            continue
        if not isinstance(record, dict):
            invalid_files += 1
            continue

        record["labels"] = labels
        record_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records_updated += 1

    return LabelStudioImportSummary(
        files_seen=files_seen,
        tasks_with_sample_id=tasks_with_sample_id,
        records_updated=records_updated,
        skipped_without_annotation=skipped_without_annotation,
        skipped_missing_record=skipped_missing_record,
        invalid_files=invalid_files,
    )
