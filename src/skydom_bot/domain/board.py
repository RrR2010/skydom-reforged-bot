"""Domain models for board geometry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Point:
    """Integer screen/image coordinate."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned rectangle using an exclusive right/bottom edge."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        """Return the exclusive right edge."""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Return the exclusive bottom edge."""
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class Cell:
    """One playable board cell."""

    row: int
    col: int
    center: Point
    bounds: Rect
    occupancy: float


@dataclass(frozen=True, slots=True)
class BoardGeometry:
    """Detected board bounds, regular grid pitch, and playable cells."""

    bounds: Rect
    rows: int
    cols: int
    pitch_x: float
    pitch_y: float
    cells: tuple[Cell, ...]
    confidence: float

    @property
    def cell_map(self) -> dict[tuple[int, int], Cell]:
        """Return cells indexed by logical row/column."""
        return {(cell.row, cell.col): cell for cell in self.cells}

    def has_cell(self, row: int, col: int) -> bool:
        """Return whether a logical position exists in the board topology."""
        return (row, col) in self.cell_map

    def topology_text(self, active: str = "X", empty: str = " ") -> str:
        """Render the irregular board topology as a compact text grid."""
        keys = self.cell_map
        return "\n".join(
            "".join(active if (row, col) in keys else empty for col in range(self.cols))
            for row in range(self.rows)
        )
