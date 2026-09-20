"""Tests for dataset statistics and validation."""

from __future__ import annotations

import json

from skydom_bot.dataset.stats import collect_dataset_statistics


def _write_record(root, sample_id: str, *, labels) -> None:
    image_rel = f"input/images/{sample_id}.png"
    image_path = root / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fixture")

    records_dir = root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    (records_dir / f"{sample_id}.json").write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "image": image_rel,
                "row": 0,
                "col": 0,
                "source": "unit-test",
                "suggested": {
                    "color": {"value": "red", "confidence": 1.0},
                    "kind": {"value": "normal", "confidence": 1.0},
                    "blocker": {"value": "none", "confidence": 1.0},
                    "powerup": {"value": "none", "confidence": 1.0},
                },
                "labels": labels,
            }
        ),
        encoding="utf-8",
    )


def test_statistics_count_only_human_labels(tmp_path) -> None:
    _write_record(
        tmp_path,
        "labeled-a",
        labels={
            "color": "green",
            "kind": "normal",
            "blocker": "none",
            "powerup": "flyer",
        },
    )
    _write_record(tmp_path, "unlabeled", labels=None)
    _write_record(
        tmp_path,
        "labeled-b",
        labels={
            "color": "green",
            "kind": "none",
            "blocker": "chain",
            "powerup": "color-remover",
        },
    )

    stats = collect_dataset_statistics(tmp_path)

    assert stats.total == 3
    assert stats.labeled == 2
    assert stats.unlabeled == 1
    assert stats.invalid == 0
    assert stats.with_capture_provenance == 0
    assert stats.without_capture_provenance == 3
    assert stats.distributions["color"] == {"green": 2}
    assert stats.distributions["kind"] == {"none": 1, "normal": 1}
    assert stats.distributions["blocker"] == {"chain": 1, "none": 1}
    assert stats.distributions["powerup"] == {"color-remover": 1, "flyer": 1}


def test_statistics_reject_partial_or_unknown_human_labels(tmp_path) -> None:
    _write_record(
        tmp_path,
        "partial",
        labels={
            "color": "blue",
            "kind": "normal",
            "blocker": "none",
        },
    )
    _write_record(
        tmp_path,
        "invalid-value",
        labels={
            "color": "cyan",
            "kind": "normal",
            "blocker": "none",
            "powerup": "none",
        },
    )

    stats = collect_dataset_statistics(tmp_path)

    assert stats.total == 2
    assert stats.labeled == 0
    assert stats.unlabeled == 0
    assert stats.invalid == 2
    assert stats.with_capture_provenance == 0
    assert stats.without_capture_provenance == 2
    assert len(stats.issues) == 2


def test_statistics_report_missing_image_without_promoting_suggestion(tmp_path) -> None:
    records_dir = tmp_path / "records"
    records_dir.mkdir(parents=True)
    (records_dir / "sample.json").write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "image": "input/images/missing.png",
                "suggested": {
                    "color": {"value": "purple", "confidence": 1.0},
                },
                "labels": None,
            }
        ),
        encoding="utf-8",
    )

    stats = collect_dataset_statistics(tmp_path)

    assert stats.unlabeled == 1
    assert stats.labeled == 0
    assert stats.distributions["color"] == {}
    assert any("image does not exist" in issue.message for issue in stats.issues)


def test_statistics_accept_non_applicable_adjacent_clear_obstacle(tmp_path) -> None:
    _write_record(
        tmp_path,
        "obstacle",
        labels={
            "color": "none",
            "kind": "none",
            "blocker": "adjacent-clear",
            "powerup": "none",
        },
    )

    stats = collect_dataset_statistics(tmp_path)

    assert stats.invalid == 0
    assert stats.labeled == 1
    assert stats.distributions["color"] == {"none": 1}
    assert stats.distributions["kind"] == {"none": 1}
    assert stats.distributions["blocker"] == {"adjacent-clear": 1}


def test_statistics_build_pair_and_complete_combination_distributions(tmp_path) -> None:
    labels = {
        "color": "blue",
        "kind": "none",
        "blocker": "none",
        "powerup": "flyer",
    }
    _write_record(tmp_path, "blue-flyer-a", labels=labels)
    _write_record(tmp_path, "blue-flyer-b", labels=labels)
    _write_record(
        tmp_path,
        "red-bomb",
        labels={
            "color": "red",
            "kind": "none",
            "blocker": "chain",
            "powerup": "bomb",
        },
    )

    stats = collect_dataset_statistics(tmp_path)

    assert stats.pair_distributions["powerup×color"] == {
        "bomb": {"red": 1},
        "flyer": {"blue": 2},
    }
    assert stats.pair_distributions["powerup×blocker"] == {
        "bomb": {"chain": 1},
        "flyer": {"none": 2},
    }
    assert stats.combinations[
        "color=blue|kind=none|blocker=none|powerup=flyer"
    ] == 2
    assert stats.combinations[
        "color=red|kind=none|blocker=chain|powerup=bomb"
    ] == 1


def test_powerup_color_gaps_prioritize_missing_combinations(tmp_path) -> None:
    _write_record(
        tmp_path,
        "blue-flyer",
        labels={
            "color": "blue",
            "kind": "none",
            "blocker": "none",
            "powerup": "flyer",
        },
    )

    stats = collect_dataset_statistics(tmp_path, target_per_combination=3)

    by_dimensions = {
        tuple(item.dimensions): item
        for item in stats.powerup_color_gaps
    }
    blue_flyer = by_dimensions[(("powerup", "flyer"), ("color", "blue"))]
    red_flyer = by_dimensions[(("powerup", "flyer"), ("color", "red"))]

    assert blue_flyer.count == 1
    assert blue_flyer.gap == 2
    assert blue_flyer.status == "very-low"
    assert red_flyer.count == 0
    assert red_flyer.gap == 3
    assert red_flyer.status == "critical"
    assert stats.powerup_color_gaps[0].gap == 3


def test_statistics_reject_carrot_with_real_powerup(tmp_path) -> None:
    _write_record(
        tmp_path,
        "invalid-carrot-powerup",
        labels={
            "color": "orange",
            "kind": "carrot",
            "blocker": "none",
            "powerup": "row",
        },
    )

    stats = collect_dataset_statistics(tmp_path)

    assert stats.invalid == 1
    assert stats.labeled == 0
    assert any(
        "carrot tiles cannot have a powerup" in issue.message
        for issue in stats.issues
    )


def test_target_per_combination_must_be_positive(tmp_path) -> None:
    try:
        collect_dataset_statistics(tmp_path, target_per_combination=0)
    except ValueError as exc:
        assert str(exc) == "target_per_combination must be >= 1"
    else:
        raise AssertionError("expected ValueError")
