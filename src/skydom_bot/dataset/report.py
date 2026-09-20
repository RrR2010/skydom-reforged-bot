"""Markdown reporting for dataset collection guidance."""

from __future__ import annotations

from pathlib import Path

from skydom_bot.dataset.stats import DatasetStatistics

_NON_ACTIONABLE_VALUES = {"none", "unknown"}
_COLORLESS_POWERUPS = {"color-remover"}


def _observed_colors(stats: DatasetStatistics) -> list[str]:
    """Return real colors that have actually been observed in human labels."""
    return sorted(
        color
        for color, count in stats.distributions["color"].items()
        if color not in _NON_ACTIONABLE_VALUES and count > 0
    )


def _missing(values: list[str], observed: set[str]) -> str:
    """Render missing values for a Markdown table."""
    missing = [value for value in values if value not in observed]
    return ", ".join(missing) if missing else "—"


def _priority(count: int, target: int, coverage: int, coverage_total: int) -> str:
    """Return a collection priority from quantity and diversity coverage."""
    high_threshold = max(2, (target + 3) // 4)
    if count < high_threshold:
        return "HIGH"
    if count < target or coverage < coverage_total:
        return "MEDIUM"
    return "LOW"


def build_collection_report(
    stats: DatasetStatistics,
    *,
    target_per_class: int = 10,
) -> str:
    """Build a human-readable Markdown report focused on what to collect next."""
    if target_per_class < 1:
        raise ValueError("target_per_class must be >= 1")

    colors = _observed_colors(stats)
    color_set = set(colors)
    powerup_counts = stats.distributions["powerup"]
    blocker_counts = stats.distributions["blocker"]
    kind_counts = stats.distributions["kind"]
    powerup_color = stats.pair_distributions["powerup×color"]
    kind_color = stats.pair_distributions["kind×color"]

    lines = [
        "# Dataset collection report",
        "",
        "## Snapshot",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Records | {stats.total} |",
        f"| Labeled | {stats.labeled} |",
        f"| Unlabeled | {stats.unlabeled} |",
        f"| Invalid | {stats.invalid} |",
        f"| Observed real colors | {len(colors)} |",
        "",
        f"Observed real colors: **{', '.join(colors) if colors else 'none'}**.",
        "",
        "Colors that have never appeared are not treated as missing collection targets.",
        "",
        "## What to collect next",
        "",
        "### Power-ups",
        "",
        f"Quantity target: **{target_per_class} samples per power-up class**. "
        "Color coverage is evaluated separately across colors already observed in the dataset.",
        "",
        "| Priority | Power-up | Samples | Target | Color coverage | Missing observed colors |",
        "|---|---|---:|---:|---:|---|",
    ]

    rows: list[tuple[str, str, int, str, str]] = []
    for powerup, count in sorted(powerup_counts.items()):
        if powerup in _NON_ACTIONABLE_VALUES:
            continue

        if powerup in _COLORLESS_POWERUPS:
            coverage_text = "n/a"
            missing_text = "not assessed"
            priority = _priority(count, target_per_class, 0, 0)
        else:
            observed = set(powerup_color.get(powerup, {})) & color_set
            coverage_text = f"{len(observed)}/{len(colors)}"
            missing_text = _missing(colors, observed)
            priority = _priority(
                count,
                target_per_class,
                len(observed),
                len(colors),
            )

        rows.append((priority, powerup, count, coverage_text, missing_text))

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    rows.sort(key=lambda row: (priority_order[row[0]], row[2], row[1]))

    for priority, powerup, count, coverage_text, missing_text in rows:
        lines.append(
            f"| {priority} | {powerup} | {count} | {target_per_class} | "
            f"{coverage_text} | {missing_text} |"
        )

    real_blockers = {
        name: count
        for name, count in blocker_counts.items()
        if name not in _NON_ACTIONABLE_VALUES
    }
    lines.extend(["", "### Blockers", ""])
    if not real_blockers:
        lines.append(
            "**HIGH priority:** no blocker samples are labeled yet. "
            "Collect any chain or adjacent-clear example when encountered."
        )
    else:
        lines.extend(
            [
                "| Blocker | Samples |",
                "|---|---:|",
                *[
                    f"| {name} | {count} |"
                    for name, count in sorted(real_blockers.items())
                ],
            ]
        )

    carrot_count = kind_counts.get("carrot", 0)
    carrot_colors = set(kind_color.get("carrot", {})) & color_set
    lines.extend(
        [
            "",
            "### Carrots",
            "",
            f"Samples: **{carrot_count}**. "
            f"Color coverage: **{len(carrot_colors)}/{len(colors)}**.",
            "",
        ]
    )
    missing_carrot_colors = _missing(colors, carrot_colors)
    if missing_carrot_colors == "—":
        weakest = sorted(
            (
                (count, color)
                for color, count in kind_color.get("carrot", {}).items()
                if color in color_set
            )
        )
        if weakest:
            minimum = weakest[0][0]
            weakest_colors = ", ".join(
                color for count, color in weakest if count == minimum
            )
            lines.append(
                f"All observed colors are represented. Least represented: "
                f"**{weakest_colors}** ({minimum} samples each)."
            )
    else:
        lines.append(f"Missing observed colors: **{missing_carrot_colors}**.")

    lines.extend(["", "## Marginal distributions", ""])
    for field, counts in stats.distributions.items():
        lines.extend(
            [
                f"### {field}",
                "",
                "| Value | Samples |",
                "|---|---:|",
                *[
                    f"| {name} | {count} |"
                    for name, count in sorted(
                        counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ],
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- none and unknown are preserved in statistics but are not treated as collection classes.",
            "- A color is considered a collection target only after it has appeared at least once in human labels.",
            "- color-remover color coverage is intentionally not assessed until its color semantics are confirmed.",
            "- Power-up/blocker combinations remain observational only; no structural rule is assumed yet.",
            "",
        ]
    )

    return "\n".join(lines)


def write_collection_report(
    path: Path,
    stats: DatasetStatistics,
    *,
    target_per_class: int = 10,
) -> None:
    """Write the collection-oriented Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_collection_report(stats, target_per_class=target_per_class),
        encoding="utf-8",
    )
