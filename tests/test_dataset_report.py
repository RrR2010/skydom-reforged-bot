"""Tests for collection-oriented dataset Markdown reports."""

from __future__ import annotations

from skydom_bot.dataset.report import build_collection_report
from skydom_bot.dataset.stats import DatasetStatistics


def _stats() -> DatasetStatistics:
    return DatasetStatistics(
        total=128,
        labeled=109,
        unlabeled=19,
        invalid=0,
        with_capture_provenance=13,
        without_capture_provenance=115,
        distributions={
            "color": {
                "blue": 24,
                "green": 23,
                "orange": 22,
                "purple": 18,
                "unknown": 1,
                "yellow": 21,
            },
            "kind": {"carrot": 21, "normal": 88},
            "blocker": {"none": 109},
            "powerup": {
                "bomb": 3,
                "color-remover": 1,
                "column": 1,
                "flyer": 5,
                "none": 95,
                "row": 4,
            },
        },
        pair_distributions={
            "powerup×color": {
                "bomb": {"blue": 1, "orange": 2},
                "color-remover": {"unknown": 1},
                "column": {"purple": 1},
                "flyer": {
                    "blue": 1,
                    "green": 1,
                    "orange": 1,
                    "purple": 1,
                    "yellow": 1,
                },
                "none": {
                    "blue": 22,
                    "green": 21,
                    "orange": 18,
                    "purple": 16,
                    "yellow": 18,
                },
                "row": {"green": 1, "orange": 1, "yellow": 2},
            },
            "blocker×color": {"none": {}},
            "kind×color": {
                "carrot": {
                    "blue": 5,
                    "green": 3,
                    "orange": 6,
                    "purple": 2,
                    "yellow": 5,
                },
                "normal": {},
            },
            "powerup×blocker": {},
            "powerup×kind": {},
            "kind×blocker": {},
        },
        combinations={},
        powerup_color_gaps=(),
        issues=(),
    )


def test_report_uses_only_observed_colors_as_collection_targets() -> None:
    report = build_collection_report(_stats(), target_per_class=10)

    assert "Observed real colors: **blue, green, orange, purple, yellow**." in report
    assert "red" not in report


def test_report_separates_powerup_quantity_from_color_coverage() -> None:
    report = build_collection_report(_stats(), target_per_class=10)

    assert "| MEDIUM | flyer | 5 | 10 | 5/5 | — |" in report
    assert "| MEDIUM | bomb | 3 | 10 | 2/5 | green, purple, yellow |" in report
    assert "| HIGH | column | 1 | 10 | 1/5 | blue, green, orange, yellow |" in report


def test_report_does_not_force_color_coverage_for_color_remover() -> None:
    report = build_collection_report(_stats(), target_per_class=10)

    assert "| HIGH | color-remover | 1 | 10 | n/a | not assessed |" in report


def test_report_highlights_absent_blockers_and_weakest_carrot_color() -> None:
    report = build_collection_report(_stats(), target_per_class=10)

    assert "no blocker samples are labeled yet" in report
    assert "Least represented: **purple** (2 samples each)." in report
