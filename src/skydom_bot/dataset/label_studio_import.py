"""Import human Label Studio annotations into canonical dataset records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skydom_bot.domain.tile import TileColor
from skydom_bot.domain.tile_semantics import TileBlocker, TileKind, TilePowerup


_LABEL_FIELDS = ("color", "kind", "blocker", "powerup")
_ALLOWED_VALUES: dict[str, frozenset[str]] = {
    "color": frozenset(item.value for item in TileColor),
    "kind": frozenset(item.value for item in TileKind),
    "blocker": frozenset(item.value for item in TileBlocker),
    "powerup": frozenset(item.value for item in TilePowerup),
}


@dataclass(frozen=True, slots=True)
class LabelStudioImportSummary:
    """Counts produced by one annotation import pass."""

    files_seen: int
    tasks_with_sample_id: int
    records_updated: int
    skipped_without_annotation: int
    skipped_missing_record: int
    invalid_files: int


def _annotation_sort_key(annotation: dict[str, Any]) -> str:
    """Return a sortable timestamp for selecting the latest annotation."""
    updated = annotation.get("updated_at")
    created = annotation.get("created_at")
    return str(updated or created or "")


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
    if any(labels[name] not in _ALLOWED_VALUES[name] for name in _LABEL_FIELDS):
        return None
    return labels


def _sample_id_from_task(task: dict[str, Any]) -> str | None:
    data = task.get("data")
    if not isinstance(data, dict):
        return None
    sample_id = data.get("sample_id")
    return sample_id if isinstance(sample_id, str) and sample_id else None


def _annotation_candidates(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Normalize supported Label Studio export shapes.

    Local Files Target Storage writes one extensionless JSON file per
    annotation, with `result` at the root and the source task nested under
    `task`. Manual/task exports instead commonly place `data` at the root and
    annotations under `annotations`. Support both shapes.
    """
    candidates: list[tuple[str, dict[str, Any]]] = []

    # Local Files Target Storage format: annotation at root, task nested.
    nested_task = payload.get("task")
    if isinstance(nested_task, dict) and isinstance(payload.get("result"), list):
        sample_id = _sample_id_from_task(nested_task)
        if sample_id is not None:
            candidates.append((sample_id, payload))
        return candidates

    # Task export format: task at root, annotations nested.
    sample_id = _sample_id_from_task(payload)
    annotations = payload.get("annotations")
    if sample_id is not None and isinstance(annotations, list):
        for annotation in annotations:
            if isinstance(annotation, dict):
                candidates.append((sample_id, annotation))

    return candidates


def import_label_studio_annotations(
    dataset_root: Path,
    annotations_dir: Path | None = None,
) -> LabelStudioImportSummary:
    """Merge submitted Label Studio choices into dataset record labels.

    Both extensionless Local Files Target Storage objects and conventional
    `.json` task exports are supported.
    """
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

    latest_by_sample: dict[str, dict[str, Any]] = {}

    if annotations_dir.exists():
        paths = sorted(path for path in annotations_dir.iterdir() if path.is_file())
    else:
        paths = []

    for path in paths:
        files_seen += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            invalid_files += 1
            continue

        if not isinstance(payload, dict):
            invalid_files += 1
            continue

        candidates = _annotation_candidates(payload)
        if not candidates:
            invalid_files += 1
            continue

        # Count source objects that successfully resolve to a dataset sample.
        tasks_with_sample_id += 1
        for sample_id, annotation in candidates:
            current = latest_by_sample.get(sample_id)
            if current is None or _annotation_sort_key(annotation) >= _annotation_sort_key(current):
                latest_by_sample[sample_id] = annotation

    for sample_id, annotation in latest_by_sample.items():
        labels = _labels_from_annotation(annotation)
        if labels is None:
            skipped_without_annotation += 1
            continue

        record_path = records_dir / f"{sample_id}.json"
        if not record_path.exists():
            skipped_missing_record += 1
            continue

        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
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
