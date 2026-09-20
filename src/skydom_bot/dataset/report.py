"""Markdown reporting for dataset collection and training readiness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skydom_bot.dataset.stats import DatasetStatistics
from skydom_bot.domain.tile_semantics import TileBlocker, TilePowerup

_NON_ACTIONABLE_VALUES = {"none", "unknown"}
_COLORLESS_POWERUPS = {"color-remover"}

MIN_COMBINATION_SAMPLES = 3
BASELINE_COMBINATION_SAMPLES = 5
ROBUST_COMBINATION_SAMPLES = 10
DESIRED_CLASS_SAMPLES = 25
DESIRED_COLOR_SAMPLES = 30
MIN_INDEPENDENT_CAPTURES = 3
PROVENANCE_CONFIDENCE_THRESHOLD = 0.80


@dataclass(frozen=True, slots=True)
class CoverageBand:
    """Heuristic collection band for one visual combination."""

    name: str
    meaning: str


def coverage_band(count: int) -> CoverageBand:
    """Classify sample coverage without claiming formal statistical significance."""
    if count == 0:
        return CoverageBand("missing", "not observed")
    if count < MIN_COMBINATION_SAMPLES:
        return CoverageBand("insufficient", "too little variation evidence")
    if count < BASELINE_COMBINATION_SAMPLES:
        return CoverageBand("minimum", "enough to include in an early experiment")
    if count < ROBUST_COMBINATION_SAMPLES:
        return CoverageBand("baseline", "reasonable first-training coverage")
    return CoverageBand("robust-candidate", "stronger coverage; validate with learning curves")


def _observed_colors(stats: DatasetStatistics) -> list[str]:
    """Return real colors that have actually appeared in human labels."""
    return sorted(
        color
        for color, count in stats.distributions["color"].items()
        if color not in _NON_ACTIONABLE_VALUES and count > 0
    )


def _real_powerups() -> list[str]:
    """Return known concrete power-up classes."""
    return [
        item.value
        for item in TilePowerup
        if item.value not in _NON_ACTIONABLE_VALUES
    ]


def _real_blockers() -> list[str]:
    """Return known concrete blocker classes."""
    return [
        item.value
        for item in TileBlocker
        if item.value not in _NON_ACTIONABLE_VALUES
    ]


def _provenance_ratio(stats: DatasetStatistics) -> float:
    """Return labeled-sample provenance coverage."""
    if stats.labeled == 0:
        return 0.0
    return stats.labeled_with_capture_provenance / stats.labeled


def _capture_text(
    captures: int,
    *,
    provenance_trusted: bool,
) -> str:
    """Render capture diversity conservatively when provenance is incomplete."""
    if not provenance_trusted:
        return f"{captures} known (partial)"
    if captures < MIN_INDEPENDENT_CAPTURES:
        return f"{captures} (low)"
    return str(captures)


def _powerup_priority(
    total: int,
    combo_counts: list[int],
) -> str:
    """Prioritize power-up collection from total volume and color coverage."""
    if total == 0 or any(count == 0 for count in combo_counts):
        return "HIGH"
    if total < DESIRED_CLASS_SAMPLES or any(
        count < BASELINE_COMBINATION_SAMPLES for count in combo_counts
    ):
        return "MEDIUM"
    return "LOW"


def build_collection_report(stats: DatasetStatistics) -> str:
    """Build a Markdown report focused on training-readiness evidence."""
    colors = _observed_colors(stats)
    color_set = set(colors)
    powerup_counts = stats.distributions["powerup"]
    blocker_counts = stats.distributions["blocker"]
    color_counts = stats.distributions["color"]
    kind_counts = stats.distributions["kind"]

    powerup_color = stats.pair_distributions["powerup×color"]
    powerup_color_captures = stats.pair_capture_distributions["powerup×color"]
    kind_color = stats.pair_distributions["kind×color"]
    kind_color_captures = stats.pair_capture_distributions["kind×color"]

    provenance_ratio = _provenance_ratio(stats)
    provenance_trusted = provenance_ratio >= PROVENANCE_CONFIDENCE_THRESHOLD

    lines = [
        "# Dataset training-readiness report",
        "",
        "> The thresholds in this report are collection heuristics, not formal statistical guarantees. "
        "The final stopping rule should come from validation metrics and learning curves.",
        "",
        "## Snapshot",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Records | {stats.total} |",
        f"| Labeled | {stats.labeled} |",
        f"| Unlabeled | {stats.unlabeled} |",
        f"| Invalid | {stats.invalid} |",
        f"| Labeled with capture provenance | {stats.labeled_with_capture_provenance}/{stats.labeled} |",
        f"| Provenance coverage | {provenance_ratio:.0%} |",
        "",
        f"Observed real colors: **{', '.join(colors) if colors else 'none'}**.",
        "",
    ]

    if not provenance_trusted:
        lines.extend(
            [
                "**Capture-diversity warning:** provenance coverage is still too low to use "
                "independent-capture counts as a hard training-readiness criterion. Known capture "
                "counts are shown as partial evidence only.",
                "",
            ]
        )

    lines.extend(
        [
            "## Collection thresholds",
            "",
            "| Scope | Threshold | Interpretation |",
            "|---|---:|---|",
            f"| Combination | 0 | missing |",
            f"| Combination | 1–{MIN_COMBINATION_SAMPLES - 1} | insufficient |",
            f"| Combination | {MIN_COMBINATION_SAMPLES}–{BASELINE_COMBINATION_SAMPLES - 1} | minimum |",
            f"| Combination | {BASELINE_COMBINATION_SAMPLES}–{ROBUST_COMBINATION_SAMPLES - 1} | baseline |",
            f"| Combination | ≥{ROBUST_COMBINATION_SAMPLES} | robust-candidate |",
            f"| Power-up/blocker class | ~{DESIRED_CLASS_SAMPLES} | preferred first-training volume |",
            f"| Color | ~{DESIRED_COLOR_SAMPLES} | preferred overall representation |",
            f"| Independent captures | ≥{MIN_INDEPENDENT_CAPTURES} | minimum diversity evidence when provenance is reliable |",
            "",
            "## What to collect next",
            "",
            "### Power-ups",
            "",
            "For color-bearing power-ups, the report checks both total class volume and "
            "coverage across colors that have already appeared in the dataset.",
            "",
            "| Priority | Power-up | Total | Color coverage | Weakest combinations |",
            "|---|---|---:|---:|---|",
        ]
    )

    powerup_rows: list[tuple[int, str]] = []
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

    for powerup in _real_powerups():
        total = powerup_counts.get(powerup, 0)

        if powerup in _COLORLESS_POWERUPS:
            priority = "HIGH" if total < DESIRED_CLASS_SAMPLES else "LOW"
            detail = "color semantics not assessed"
            coverage = "n/a"
        else:
            combo_counts = [
                powerup_color.get(powerup, {}).get(color, 0)
                for color in colors
            ]
            priority = _powerup_priority(total, combo_counts)
            covered = sum(count > 0 for count in combo_counts)
            coverage = f"{covered}/{len(colors)}"

            weak = []
            for color, count in zip(colors, combo_counts):
                band = coverage_band(count)
                if band.name in {"missing", "insufficient", "minimum"}:
                    captures = powerup_color_captures.get(powerup, {}).get(color, 0)
                    weak.append(
                        f"{color}: {count} samples, "
                        f"{_capture_text(captures, provenance_trusted=provenance_trusted)} captures, "
                        f"{band.name}"
                    )
            detail = "; ".join(weak) if weak else "all observed colors at baseline or better"

        row = (
            f"| {priority} | {powerup} | {total}/{DESIRED_CLASS_SAMPLES} | "
            f"{coverage} | {detail} |"
        )
        powerup_rows.append((priority_order[priority], row))

    lines.extend(row for _, row in sorted(powerup_rows))

    lines.extend(
        [
            "",
            "### Carrots",
            "",
            f"Total carrot samples: **{kind_counts.get('carrot', 0)}**.",
            "",
            "| Color | Samples | Independent captures | Coverage |",
            "|---|---:|---:|---|",
        ]
    )
    for color in colors:
        count = kind_color.get("carrot", {}).get(color, 0)
        captures = kind_color_captures.get("carrot", {}).get(color, 0)
        band = coverage_band(count)
        lines.append(
            f"| {color} | {count} | "
            f"{_capture_text(captures, provenance_trusted=provenance_trusted)} | "
            f"{band.name} |"
        )

    lines.extend(
        [
            "",
            "### Blockers",
            "",
            "| Blocker | Samples | Preferred class volume | Status |",
            "|---|---:|---:|---|",
        ]
    )
    for blocker in _real_blockers():
        count = blocker_counts.get(blocker, 0)
        status = (
            "missing"
            if count == 0
            else "collect more"
            if count < DESIRED_CLASS_SAMPLES
            else "first-training volume reached"
        )
        lines.append(
            f"| {blocker} | {count} | {DESIRED_CLASS_SAMPLES} | {status} |"
        )

    lines.extend(
        [
            "",
            "### Overall color representation",
            "",
            "| Color | Samples | Preferred volume | Status |",
            "|---|---:|---:|---|",
        ]
    )
    for color in colors:
        count = color_counts.get(color, 0)
        status = "collect more" if count < DESIRED_COLOR_SAMPLES else "represented"
        lines.append(f"| {color} | {count} | {DESIRED_COLOR_SAMPLES} | {status} |")

    lines.extend(
        [
            "",
            "## Training strategy",
            "",
            "1. Reach at least **3 independent-looking samples per valid combination** before treating that combination as represented.",
            "2. Prefer **5+ samples per valid combination** for the first serious transfer-learning experiment.",
            f"3. Aim for roughly **{DESIRED_CLASS_SAMPLES} samples per rare semantic class** and **{DESIRED_COLOR_SAMPLES} per color**, without forcing redundant near-duplicates.",
            "4. Once these floors are reached, stop increasing counts blindly. Train the model and use per-head F1/recall, subgroup errors, and learning curves to decide where more data is valuable.",
            "5. Split train/validation/test by capture group, not by crop, whenever capture provenance is available. This reduces leakage from visually related cells from the same board state.",
            "",
            "## Marginal distributions",
            "",
        ]
    )

    for field, counts in stats.distributions.items():
        lines.extend(
            [
                f"### {field}",
                "",
                "| Value | Samples | Known independent captures |",
                "|---|---:|---:|",
            ]
        )
        for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            captures = stats.capture_distributions.get(field, {}).get(name, 0)
            lines.append(f"| {name} | {count} | {captures} |")
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- Colors that have never appeared in human labels are not automatically treated as missing collection targets.",
            "- color-remover is not forced into a color matrix until its color semantics are confirmed.",
            "- Power-up/blocker combinations remain observational; no structural compatibility rule is assumed yet.",
            "- Sample-count bands are operational heuristics. They should be revised after the first model produces real learning curves.",
            "",
        ]
    )

    return "\n".join(lines)


def write_collection_report(path: Path, stats: DatasetStatistics) -> None:
    """Write the training-readiness Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_collection_report(stats), encoding="utf-8")
