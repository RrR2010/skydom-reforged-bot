"""Tests for the compact screen-corpus capture form."""

from __future__ import annotations

from skydom_bot.corpus.capture_gui import (
    CaptureFormState,
    build_metadata,
    load_form_state,
    save_form_state,
)


def test_capture_form_state_round_trip(tmp_path) -> None:
    state = CaptureFormState(
        stage_id="level-27",
        mode="competitive",
        initial_moves="22",
        board_variant="sparse",
        has_ice=True,
        custom=[
            ("goal", "carrot"),
            ("note", "rare layout"),
            ("", ""),
            ("opponent", "yes"),
        ],
    )

    path = save_form_state(tmp_path, state)
    loaded = load_form_state(tmp_path)

    assert path.is_file()
    assert loaded == state


def test_build_metadata_keeps_canonical_keys_and_custom_values() -> None:
    state = CaptureFormState(
        stage_id="level-27",
        mode="normal",
        initial_moves="18",
        board_variant="wide",
        has_ice=False,
        custom=[
            ("goal", "ice"),
            ("mode", "must-not-override"),
            ("note", "second state"),
            ("", "ignored"),
        ],
    )

    assert build_metadata(state) == {
        "mode": "normal",
        "initial_moves": "18",
        "board_variant": "wide",
        "has_ice": "no",
        "goal": "ice",
        "note": "second state",
    }
