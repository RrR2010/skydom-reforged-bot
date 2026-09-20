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
_REAL_COLORS = tuple(
    item.value for item in TileColor if item not in {TileColor.NONE, TileColor.UNKNOWN}
)
_REAL_POWERUPS = tuple(
    item.value for item in TilePowerup if item not in {TilePowerup.NONE, TilePowerup.UNKNOWN}
)


@dataclass(frozen=True, slots=True)
class DatasetIssue:
    """One integrity problem found while scanning a dataset record."""

    path: Path
    message: str


@dataclass(frozen=True, slots=True)
class CoverageGap:
    """One underrepresented actionable label combination."""

    dimensions: tuple[tuple[str, str], ...]
    count: int
    target: int

    @property
    def gap(self) -> int:
        """Return the remaining samples needed to reach the target."""
        return max(0, self.target - self.count)

    @property
    def status(self) -> str:
        """Return a coarse collection-priority status."""
        if self.count == 0:
            return "critical"
        if self.count < 10:
            return "very-low"
        if self.count < 25:
            return "low"
        if self.count < 50:
            return "ok"
        return "well-represented"


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    """Aggregate counts and collection coverage from canonical dataset records."""

    total: int
    labeled: int
    unlabeled: int
    invalid: int
    with_capture_provenance: int
    without_capture_provenance: int
    distributions: dict[str, dict[str, int]]
    pair_distributions: dict[str, dict[str, dict[str, int]]]
    combinations: dict[str, int]
    powerup_color_gaps: tuple[CoverageGap, ...]
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
        distributions={field: {} for field in _LABEL_FIELDS},
        pair_distributions={_pair_name(*pair): {} for pair in _PAIR_FIELDS},
        combinations={},
        powerup_color_gaps=(),
        issues=(),
    )


def _build_powerup_color_gaps(
    pair_counts: dict[str, dict[str, Counter[str]]],
    *,
    target_per_combination: int,
) -> tuple[CoverageGap, ...]:
    """Build collection priorities for real power-up/color combinations."""
    matrix = pair_counts[_pair_name("powerup", "color")]
    gaps = [
        CoverageGap(
            dimensions=(("powerup", powerup), ("color", color)),
            count=matrix.get(powerup, Counter()).get(color, 0),
            target=target_per_combination,
        )
        for powerup in _REAL_POWERUPS
        for color in _REAL_COLORS
    ]
    return tuple(
        sorted(
            gaps,
            key=lambda item: (
                -item.gap,
                item.count,
                dict(item.dimensions)["powerup"],
                dict(item.dimensions)["color"],
            ),
        )
    )


def collect_dataset_statistics(
    root: Path = Path("dataset"),
    *,
    target_per_combination: int = 30,
) -> DatasetStatistics:
    """Scan canonical records and summarize human ground-truth coverage.

    Bootstrap predictions under suggested are intentionally ignored.
    target_per_combination is used only for collection-priority guidance.
    """
    if target_per_combination < 1:
        raise ValueError("target_per_combination must be >= 1")

    records_dir = root / "records"
    if not records_dir.exists():
        return _empty_statistics()

    counters = {field: Counter() for field in _LABEL_FIELDS}
    pair_counts: dict[str, dict[str, Counter[str]]] = {
        _pair_name(*pair): defaultdict(Counter) for pair in _PAIR_FIELDS
    }
    combination_counts: Counter[str] = Counter()
    issues: list[DatasetIssue] = []
    total = labeled = unlabeled = invalid = 0
    with_capture_provenance = without_capture_provenance = 0

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

        capture_ids = payload.get("capture_ids")
        if (
            isinstance(capture_ids, list)
            and any(isinstance(item, str) and item for item in capture_ids)
        ):
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
        for field, value in labels.items():
            counters[field][value] += 1

        for first, second in _PAIR_FIELDS:
            pair_counts[_pair_name(first, second)][labels[first]][labels[second]] += 1
        combination_counts[_combination_key(labels)] += 1

    distributions = {
        field: dict(sorted(counter.items()))
        for field, counter in counters.items()
    }
    pair_distributions = {
        name: {
            first_value: dict(sorted(second_counts.items()))
            for first_value, second_counts in sorted(rows.items())
        }
        for name, rows in pair_counts.items()
    }

    return DatasetStatistics(
        total=total,
        labeled=labeled,
        unlabeled=unlabeled,
        invalid=invalid,
        with_capture_provenance=with_capture_provenance,
        without_capture_provenance=without_capture_provenance,
        distributions=distributions,
        pair_distributions=pair_distributions,
        combinations=dict(sorted(combination_counts.items())),
        powerup_color_gaps=_build_powerup_color_gaps(
            pair_counts,
            target_per_combination=target_per_combination,
        ),
        issues=tuple(issues),
    )
