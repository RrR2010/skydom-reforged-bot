"""Tests for shared board inspection reporting."""

from skydom_bot.debug.report import format_geometry_summary
from skydom_bot.domain.board import BoardGeometry, Cell, Point, Rect


def test_geometry_summary_is_canonical_for_cli_and_debugger() -> None:
    geometry = BoardGeometry(
        bounds=Rect(10, 20, 120, 60),
        rows=1,
        cols=2,
        pitch_x=60.0,
        pitch_y=60.0,
        cells=(
            Cell(0, 0, Point(40, 50), Rect(10, 20, 60, 60), 0.95),
            Cell(0, 1, Point(100, 50), Rect(70, 20, 60, 60), 0.91),
        ),
        confidence=0.987,
    )

    summary = format_geometry_summary(geometry)

    assert "Board: 1x2" in summary
    assert "Bounds: x=10, y=20, w=120, h=60" in summary
    assert "Pitch: 60.00 x 60.00" in summary
    assert "Active cells: 2" in summary
    assert "Confidence: 0.987" in summary
    assert summary.endswith("Topology:\nXX")
