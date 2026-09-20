"""Tests for board domain behavior."""

from skydom_bot.domain.board import BoardGeometry, Cell, Point, Rect


def test_topology_text_preserves_missing_cells() -> None:
    cells = tuple(
        Cell(row=row, col=col, center=Point(0, 0), bounds=Rect(0, 0, 10, 10), occupancy=0.5)
        for row, cols in enumerate(((0, 1, 2), (1,)))
        for col in cols
    )
    geometry = BoardGeometry(Rect(0, 0, 30, 20), 2, 3, 10.0, 10.0, cells, 1.0)

    assert geometry.topology_text() == "XXX\n X "
    assert geometry.has_cell(1, 1)
    assert not geometry.has_cell(1, 0)
