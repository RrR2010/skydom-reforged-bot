"""Dataset statistics and validation for human-labeled tile records."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from skydom_bot.domain.tile import TileColor
from skydom_bot.domain.tile_semantics import TileBlocker, TileKind, TilePowerup

_LABEL_FIELDS = ("color", "kind", "blocker", "powerup")
_PAIR_FIELDS = (
    ("powerup", "color"),
    ("blocker", "color"),
    ("kind", "color"),
    ("powerup", "blocker"),
    ("powerup", "kind"),
    ("kind", "blocker"),
)
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
    """Aggregate counts and diversity evidence from canonical dataset records."""

    total: int
    labeled: int
    unlabeled: int
    invalid: int
    with_capture_provenance: int
    without_capture_provenance: int
    labeled_with_capture_provenance: int
    distributions: dict[str, dict[str, int]]
    capture_distributions: dict[str, dict[str, int]]
    pair_distributions: dict[str, dict[str, dict[str, int]]]
    pair_capture_distributions: dict[str, dict[str, dict[str, int]]]
    combinations: dict[str, int]
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

    if normalized["kind"] == TileKind.CARROT.value and normalized["powerup"] not in {
        TilePowerup.NONE.value,
        TilePowerup.UNKNOWN.value,
    }:
        return None, "carrot tiles cannot have a powerup"

    return normalized, None


def _capture_ids(payload: dict[str, Any]) -> tuple[str, ...]:
    """Return normalized unique capture IDs from one canonical record."""
    value = payload.get("capture_ids")
    if not isinstance(value, list):
        return ()
    return tuple(
        dict.fromkeys(
            str(item)
            for item in value
            if isinstance(item, str) and item
        )
    )


def _pair_name(first: str, second: str) -> str:
    """Return the stable display/storage name for one field pair."""
    return f"{first}×{second}"


def _combination_key(labels: dict[str, str]) -> str:
    """Return a stable key for one complete four-dimensional label combination."""
    return "|".join(f"{field}={labels[field]}" for field in _LABEL_FIELDS)


def _empty_statistics() -> DatasetStatistics:
    """Return an empty statistics object with a stable shape."""
    return DatasetStatistics(
        total=0,
        labeled=0,
        unlabeled=0,
        invalid=0,
        with_capture_provenance=0,
        without_capture_provenance=0,
        labeled_with_capture_provenance=0,
        distributions={field: {} for field in _LABEL_FIELDS},
        capture_distributions={field: {} for field in _LABEL_FIELDS},
        pair_distributions={_pair_name(*pair): {} for pair in _PAIR_FIELDS},
        pair_capture_distributions={_pair_name(*pair): {} for pair in _PAIR_FIELDS},
        combinations={},
        issues=(),
    )


def collect_dataset_statistics(root: Path = Path("dataset")) -> DatasetStatistics:
    """Scan canonical records and summarize human ground truth and diversity.

    Bootstrap predictions under suggested are intentionally ignored.
    Capture IDs are treated as provenance groups, not as extra samples.
    """
    records_dir = root / "records"
    if not records_dir.exists():
        return _empty_statistics()

    counters = {field: Counter() for field in _LABEL_FIELDS}
    capture_sets: dict[str, dict[str, set[str]]] = {
        field: defaultdict(set) for field in _LABEL_FIELDS
    }
    pair_counts: dict[str, dict[str, Counter[str]]] = {
        _pair_name(*pair): defaultdict(Counter) for pair in _PAIR_FIELDS
    }
    pair_capture_sets: dict[str, dict[str, dict[str, set[str]]]] = {
        _pair_name(*pair): defaultdict(lambda: defaultdict(set))
        for pair in _PAIR_FIELDS
    }
    combination_counts: Counter[str] = Counter()
    issues: list[DatasetIssue] = []
    total = labeled = unlabeled = invalid = 0
    with_capture_provenance = without_capture_provenance = 0
    labeled_with_capture_provenance = 0

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

        record_capture_ids = _capture_ids(payload)
        if record_capture_ids:
            with_capture_provenance += 1
        else:
            without_capture_provenance += 1

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
        if record_capture_ids:
            labeled_with_capture_provenance += 1

        for field, value in labels.items():
            counters[field][value] += 1
            capture_sets[field][value].update(record_capture_ids)

        for first, second in _PAIR_FIELDS:
            pair_name = _pair_name(first, second)
            first_value = labels[first]
            second_value = labels[second]
            pair_counts[pair_name][first_value][second_value] += 1
            pair_capture_sets[pair_name][first_value][second_value].update(
                record_capture_ids
            )

        combination_counts[_combination_key(labels)] += 1

    distributions = {
        field: dict(sorted(counter.items()))
        for field, counter in counters.items()
    }
    capture_distributions = {
        field: {
            value: len(ids)
            for value, ids in sorted(values.items())
        }
        for field, values in capture_sets.items()
    }
    pair_distributions = {
        name: {
            first_value: dict(sorted(second_counts.items()))
            for first_value, second_counts in sorted(rows.items())
        }
        for name, rows in pair_counts.items()
    }
    pair_capture_distributions = {
        name: {
            first_value: {
                second_value: len(ids)
                for second_value, ids in sorted(second_values.items())
            }
            for first_value, second_values in sorted(rows.items())
        }
        for name, rows in pair_capture_sets.items()
    }

    return DatasetStatistics(
        total=total,
        labeled=labeled,
        unlabeled=unlabeled,
        invalid=invalid,
        with_capture_provenance=with_capture_provenance,
        without_capture_provenance=without_capture_provenance,
        labeled_with_capture_provenance=labeled_with_capture_provenance,
        distributions=distributions,
        capture_distributions=capture_distributions,
        pair_distributions=pair_distributions,
        pair_capture_distributions=pair_capture_distributions,
        combinations=dict(sorted(combination_counts.items())),
        issues=tuple(issues),
    )
