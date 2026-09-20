"""Dataset statistics and validation for human-labeled tile records."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
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
class DatasetIssue:
    """One integrity problem found while scanning a dataset record."""

    path: Path
    message: str


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    """Aggregate counts from canonical dataset records."""

    total: int
    labeled: int
    unlabeled: int
    invalid: int
    distributions: dict[str, dict[str, int]]
    issues: tuple[DatasetIssue, ...]


def _validate_labels(labels: Any) -> tuple[dict[str, str] | None, str | None]:
    """Return normalized complete labels or a validation error."""
    if labels is None:
        return None, None
    if not isinstance(labels, dict):
        return None, "labels must be null or an object"

    missing = [field for field in _LABEL_FIELDS if field not in labels]
    extra = sorted(set(labels) - set(_LABEL_FIELDS))
    if missing:
        return None, f"labels missing fields: {', '.join(missing)}"
    if extra:
        return None, f"labels contain unknown fields: {', '.join(extra)}"

    normalized = {field: str(labels[field]) for field in _LABEL_FIELDS}
    for field, value in normalized.items():
        if value not in _ALLOWED_VALUES[field]:
            allowed = ", ".join(sorted(_ALLOWED_VALUES[field]))
            return None, f"invalid {field} label {value!r}; expected one of: {allowed}"
    return normalized, None


def collect_dataset_statistics(root: Path = Path("dataset")) -> DatasetStatistics:
    """Scan canonical records and summarize human ground-truth coverage.

    Bootstrap predictions under ``suggested`` are intentionally ignored.
    """
    records_dir = root / "records"
    counters = {field: Counter() for field in _LABEL_FIELDS}
    issues: list[DatasetIssue] = []
    total = labeled = unlabeled = invalid = 0

    if not records_dir.exists():
        return DatasetStatistics(
            total=0,
            labeled=0,
            unlabeled=0,
            invalid=0,
            distributions={field: {} for field in _LABEL_FIELDS},
            issues=(),
        )

    for path in sorted(records_dir.glob("*.json")):
        total += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            invalid += 1
            issues.append(DatasetIssue(path, f"could not read JSON: {exc}"))
            continue

        if not isinstance(payload, dict):
            invalid += 1
            issues.append(DatasetIssue(path, "record root must be an object"))
            continue

        sample_id = payload.get("sample_id")
        if sample_id != path.stem:
            issues.append(
                DatasetIssue(
                    path,
                    f"sample_id {sample_id!r} does not match filename {path.stem!r}",
                )
            )

        image = payload.get("image")
        if not isinstance(image, str) or not image:
            issues.append(DatasetIssue(path, "image path is missing or invalid"))
        elif not (root / image).is_file():
            issues.append(DatasetIssue(path, f"image does not exist: {image}"))

        labels, label_error = _validate_labels(payload.get("labels"))
        if label_error is not None:
            invalid += 1
            issues.append(DatasetIssue(path, label_error))
            continue
        if labels is None:
            unlabeled += 1
            continue

        labeled += 1
        for field, value in labels.items():
            counters[field][value] += 1

    distributions = {
        field: dict(sorted(counter.items()))
        for field, counter in counters.items()
    }
    return DatasetStatistics(
        total=total,
        labeled=labeled,
        unlabeled=unlabeled,
        invalid=invalid,
        distributions=distributions,
        issues=tuple(issues),
    )
