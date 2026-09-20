"""Tests for collection-oriented dataset Markdown reports."""

from __future__ import annotations

from skydom_bot.dataset.report import build_collection_report, coverage_band
from skydom_bot.dataset.stats import DatasetStatistics


def _stats() -> DatasetStatistics:
    return DatasetStatistics(
        total=128,
        labeled=109,
        unlabeled=19,
        invalid=0,
        with_capture_provenance=13,
        without_capture_provenance=115,
        labeled_with_capture_provenance=13,
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
        capture_distributions={
            "color": {"blue": 3, "green": 2},
            "kind": {"carrot": 2, "normal": 3},
            "blocker": {"none": 3},
            "powerup": {"bomb": 2, "flyer": 3},
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
        pair_capture_distributions={
            "powerup×color": {
                "bomb": {"blue": 1, "orange": 2},
                "column": {"purple": 1},
                "flyer": {
                    "blue": 1,
                    "green": 1,
                    "orange": 1,
                    "purple": 1,
                    "yellow": 1,
                },
                "row": {"green": 1, "orange": 1, "yellow": 1},
            },
            "blocker×color": {},
            "kind×color": {
                "carrot": {
                    "blue": 2,
                    "green": 1,
                    "orange": 2,
                    "purple": 1,
                    "yellow": 2,
                }
            },
            "powerup×blocker": {},
            "powerup×kind": {},
            "kind×blocker": {},
        },
        combinations={},
        issues=(),
    )


def test_coverage_bands_match_collection_strategy() -> None:
    assert coverage_band(0).name == "missing"
    assert coverage_band(2).name == "insufficient"
    assert coverage_band(3).name == "minimum"
    assert coverage_band(5).name == "baseline"
    assert coverage_band(10).name == "robust-candidate"


def test_report_uses_only_observed_colors_as_collection_targets() -> None:
    report = build_collection_report(_stats())

    assert "Observed real colors: **blue, green, orange, purple, yellow**." in report
    assert "red" not in report


def test_report_prioritizes_missing_and_undercovered_powerup_color_combinations() -> None:
    report = build_collection_report(_stats())

    assert "| HIGH | column | 1/25 | 1/5 |" in report
    assert "blue: 0 samples" in report
    assert "| HIGH | bomb | 3/25 | 2/5 |" in report
    assert "green: 0 samples" in report
    assert "| MEDIUM | flyer | 5/25 | 5/5 |" in report


def test_report_does_not_force_color_matrix_for_color_remover() -> None:
    report = build_collection_report(_stats())

    assert "| HIGH | color-remover | 1/25 | n/a | color semantics not assessed |" in report


def test_report_warns_when_capture_provenance_is_incomplete() -> None:
    report = build_collection_report(_stats())

    assert "Capture-diversity warning" in report
    assert "known (partial)" in report


def test_report_includes_carrot_and_blocker_training_readiness() -> None:
    report = build_collection_report(_stats())

    assert "| purple | 2 | 1 known (partial) | insufficient |" in report
    assert "| chain | 0 | 25 | missing |" in report
    assert "| adjacent-clear | 0 | 25 | missing |" in report


def test_report_explains_learning_curve_stopping_rule() -> None:
    report = build_collection_report(_stats())

    assert "learning curves" in report
    assert "Split train/validation/test by capture group" in report
