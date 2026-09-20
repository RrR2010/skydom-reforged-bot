"""Tests for the top-level CLI help command."""

from __future__ import annotations

from skydom_bot import help_cli


def test_root_help_lists_main_commands(capsys) -> None:
    assert help_cli.main() == 0
    output = capsys.readouterr().out
    assert "skydom-collect-tiles" in output
    assert "skydom-export-label-studio" in output
    assert "skydom-import-label-studio" in output


def test_workflow_topic_is_available() -> None:
    assert "skydom-collect-tiles" in help_cli._TOPICS["workflow"]
    assert "skydom-import-label-studio" in help_cli._TOPICS["workflow"]
