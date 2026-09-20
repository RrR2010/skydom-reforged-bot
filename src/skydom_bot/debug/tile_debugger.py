"""Interactive multi-selection debugger for tile recognition."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure
from numpy.typing import NDArray

from skydom_bot.capture import capture_screen
from skydom_bot.debug.report import format_geometry_summary, save_board_overlay
from skydom_bot.domain.board import BoardGeometry, Cell
from skydom_bot.domain.tile import TileColor, TileObservation
from skydom_bot.domain.tile_semantics import TileBlocker, TileKind, TileSemanticObservation
from skydom_bot.vision.board_detector import BoardDetector
from skydom_bot.vision.shape_features import ShapeDiagnostics, extract_shape_features
from skydom_bot.vision.tile_classifier import TileClassifier, TileDiagnostics
from skydom_bot.vision.tile_semantics import (
    TileAppearance,
    TileSemanticClassifier,
)

UInt8Image = NDArray[np.uint8]
CellKey = tuple[int, int]

_SYMBOLS = {
    TileColor.RED: "R",
    TileColor.ORANGE: "O",
    TileColor.YELLOW: "Y",
    TileColor.GREEN: "G",
    TileColor.BLUE: "B",
    TileColor.PURPLE: "P",
    TileColor.UNKNOWN: "?",
}


@dataclass(frozen=True, slots=True)
class TileSnapshot:
    """Cached perception result used by comparison and focus panels."""

    cell: Cell
    observation: TileObservation
    diagnostics: TileDiagnostics
    shape: ShapeDiagnostics
    semantics: TileSemanticObservation | None = None


def _read_rgb(path: Path) -> UInt8Image:
    """Read an image file as RGB."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def format_tile_matrix(
    geometry: BoardGeometry,
    observations: tuple[TileObservation, ...],
) -> str:
    """Render recognized colors in the same logical grid as board topology."""
    by_cell = {(item.row, item.col): item for item in observations}
    rows: list[str] = []
    for row in range(geometry.rows):
        chars: list[str] = []
        for col in range(geometry.cols):
            if not geometry.has_cell(row, col):
                chars.append(" ")
                continue
            item = by_cell.get((row, col))
            chars.append(_SYMBOLS[item.color] if item else "?")
        rows.append("".join(chars))
    return "\n".join(rows)


class TileDebugger:
    """Compare several detected cells while keeping one detailed focus cell."""

    max_selected = 6

    def __init__(
        self,
        image_rgb: UInt8Image,
        geometry: BoardGeometry,
        classifier: TileClassifier,
        observations: tuple[TileObservation, ...],
        compare_output: Path,
    ) -> None:
        self.image_rgb = image_rgb
        self.geometry = geometry
        self.classifier = classifier
        self.observations = observations
        self.compare_output = compare_output

        self.by_cell = {(item.row, item.col): item for item in observations}
        self.cells = geometry.cell_map
        self._cache: dict[CellKey, TileSnapshot] = {}
        self.semantic_classifier = TileSemanticClassifier()
        self._semantic_by_cell = self._build_semantic_map()

        initial = self._initial_cell()
        initial_key = self._key(initial)
        self.selected_keys: list[CellKey] = [initial_key]
        self.focused_key: CellKey | None = initial_key
        self.status_text = ""

        self.figure: Figure = plt.figure(figsize=(18, 10))
        grid = self.figure.add_gridspec(
            3,
            4,
            width_ratios=(1.20, 1.0, 1.0, 1.0),
            height_ratios=(1.0, 1.0, 1.15),
            left=0.025,
            right=0.985,
            top=0.91,
            bottom=0.065,
            wspace=0.20,
            hspace=0.32,
        )

        self.board_ax = self.figure.add_subplot(grid[:, 0])
        self.compare_axes = [
            self.figure.add_subplot(grid[row, col])
            for row in range(2)
            for col in range(1, 4)
        ]
        self.crop_ax = self.figure.add_subplot(grid[2, 1])
        self.feature_ax = self.figure.add_subplot(grid[2, 2])
        self.shape_ax = self.figure.add_subplot(grid[2, 3])

        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.render()

    @staticmethod
    def _key(cell: Cell) -> CellKey:
        return (cell.row, cell.col)

    def _initial_cell(self) -> Cell:
        """Start at the least-confident cell because it is most informative."""
        ranked = sorted(self.observations, key=lambda item: item.confidence)
        if ranked:
            key = (ranked[0].row, ranked[0].col)
            if key in self.cells:
                return self.cells[key]
        return next(iter(self.cells.values()))

    def _build_semantic_map(self) -> dict[CellKey, TileSemanticObservation]:
        """Analyze semantics against same-color peers across the whole board."""
        appearances: list[TileAppearance] = []
        keys: list[CellKey] = []

        for cell in self.geometry.cells:
            key = self._key(cell)
            observation, diagnostics = self.classifier.classify_cell(
                self.image_rgb,
                cell,
            )
            shape = extract_shape_features(
                diagnostics.crop_rgb,
                diagnostics.shape_foreground_mask,
            )
            appearances.append(TileAppearance(observation, diagnostics, shape))
            keys.append(key)
            self._cache[key] = TileSnapshot(
                cell=cell,
                observation=observation,
                diagnostics=diagnostics,
                shape=shape,
            )

        semantics = self.semantic_classifier.classify_board(tuple(appearances))
        mapping = dict(zip(keys, semantics))

        # Attach the semantic interpretation to the cached snapshots so every
        # panel reads one immutable object.
        for key, semantic in mapping.items():
            snapshot = self._cache[key]
            self._cache[key] = TileSnapshot(
                cell=snapshot.cell,
                observation=snapshot.observation,
                diagnostics=snapshot.diagnostics,
                shape=snapshot.shape,
                semantics=semantic,
            )
        return mapping

    def _snapshot(self, key: CellKey) -> TileSnapshot:
        """Return a cached cell analysis so redraws do not recompute vision."""
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        cell = self.cells[key]
        observation, diagnostics = self.classifier.classify_cell(self.image_rgb, cell)
        shape = extract_shape_features(
            diagnostics.crop_rgb,
            diagnostics.shape_foreground_mask,
        )
        snapshot = TileSnapshot(
            cell,
            observation,
            diagnostics,
            shape,
            self._semantic_by_cell.get(key),
        )
        self._cache[key] = snapshot
        return snapshot

    def _cell_from_event(self, event: MouseEvent) -> Cell | None:
        if event.inaxes is not self.board_ax or event.xdata is None or event.ydata is None:
            return None

        bounds = self.geometry.bounds
        x = float(event.xdata) + bounds.x
        y = float(event.ydata) + bounds.y
        for cell in self.geometry.cells:
            if (
                cell.bounds.x <= x < cell.bounds.right
                and cell.bounds.y <= y < cell.bounds.bottom
            ):
                return cell
        return None

    @staticmethod
    def _ctrl_pressed(event: MouseEvent) -> bool:
        key = (event.key or "").lower()
        return "control" in key or "ctrl" in key

    def _on_click(self, event: MouseEvent) -> None:
        cell = self._cell_from_event(event)
        if cell is None:
            return

        key = self._key(cell)
        if self._ctrl_pressed(event) or event.button == 3:
            self._toggle_selected(key)
        else:
            self.selected_keys = [key]
            self.focused_key = key
            self.status_text = ""
        self.render()

    def _toggle_selected(self, key: CellKey) -> None:
        if key in self.selected_keys:
            self.selected_keys.remove(key)
            if self.focused_key == key:
                self.focused_key = self.selected_keys[-1] if self.selected_keys else None
            self.status_text = ""
            return

        if len(self.selected_keys) >= self.max_selected:
            self.status_text = f"Selection limit: {self.max_selected} cells."
            return

        self.selected_keys.append(key)
        self.focused_key = key
        self.status_text = ""

    def _on_key(self, event: KeyEvent) -> None:
        key = (event.key or "").lower()

        if key == "c":
            self.selected_keys.clear()
            self.focused_key = None
            self.status_text = "Selection cleared."
            self.render()
            return

        if key == "s":
            self.save_comparison()
            return

        if key in {str(index) for index in range(1, self.max_selected + 1)}:
            index = int(key) - 1
            if index < len(self.selected_keys):
                self.focused_key = self.selected_keys[index]
                self.status_text = ""
                self.render()

    def save_comparison(self) -> None:
        """Save the current composite debugger view as a shareable PNG."""
        self.compare_output.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(self.compare_output, dpi=150, bbox_inches="tight")
        self.status_text = f"Saved: {self.compare_output}"
        print(self.status_text)
        self.render()

    def _render_board(self) -> None:
        self.board_ax.clear()
        bounds = self.geometry.bounds
        board_crop = self.image_rgb[bounds.y : bounds.bottom, bounds.x : bounds.right]
        self.board_ax.imshow(board_crop)
        self.board_ax.set_title(
            "Board crop — click = focus | Ctrl+click/right-click = compare\n"
            "C = clear | 1..6 = focus selected | S = save composite"
        )
        self.board_ax.set_axis_off()

        selected_order = {key: index + 1 for index, key in enumerate(self.selected_keys)}

        for cell in self.geometry.cells:
            observation = self.by_cell[(cell.row, cell.col)]
            key = self._key(cell)
            is_selected = key in selected_order
            is_focused = key == self.focused_key
            linewidth = 3.5 if is_focused else (2.0 if is_selected else 0.8)

            self.board_ax.add_patch(
                plt.Rectangle(
                    (
                        cell.bounds.x - bounds.x,
                        cell.bounds.y - bounds.y,
                    ),
                    cell.bounds.width,
                    cell.bounds.height,
                    fill=False,
                    linewidth=linewidth,
                )
            )

            prefix = f"{selected_order[key]}:" if is_selected else ""
            self.board_ax.text(
                cell.center.x - bounds.x,
                cell.center.y - bounds.y,
                f"{prefix}{_SYMBOLS[observation.color]}\n{observation.confidence:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    @staticmethod
    def _rgb_mask(mask: UInt8Image) -> UInt8Image:
        return np.repeat(mask[:, :, None], 3, axis=2)

    def _render_compare_card(self, ax: Axes, key: CellKey | None, index: int) -> None:
        ax.clear()
        ax.set_axis_off()

        if key is None:
            ax.text(
                0.5,
                0.5,
                f"Slot {index}\nCtrl+click a board cell",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        snapshot = self._snapshot(key)
        observation = snapshot.observation
        diagnostics = snapshot.diagnostics
        shape = snapshot.shape

        separator = np.full(
            (diagnostics.crop_rgb.shape[0], 3, 3),
            255,
            dtype=np.uint8,
        )
        compact = np.concatenate(
            (
                diagnostics.crop_rgb,
                separator,
                self._rgb_mask(diagnostics.shape_foreground_mask),
                separator,
                self._rgb_mask(diagnostics.overlay_foreground_mask),
            ),
            axis=1,
        )
        ax.imshow(compact)
        f = shape.features
        semantic = snapshot.semantics
        focus_marker = " [FOCUS]" if key == self.focused_key else ""
        semantic_text = (
            f"kind={semantic.kind.value}:{semantic.kind_confidence:.2f} "
            f"blocker={semantic.blocker.value}:{semantic.blocker_confidence:.2f} "
            f"anom={semantic.shape_anomaly:.2f} res={semantic.residual_fraction:.2f}"
            if semantic is not None
            else "semantics=n/a"
        )
        ax.set_title(
            f"{index}. cell ({observation.row},{observation.col}) "
            f"{_SYMBOLS[observation.color]} {observation.confidence:.2f}{focus_marker}\n"
            f"{semantic_text}\n"
            f"RGB | base | residual   "
            f"area={f.area_fraction:.2f} circ={f.circularity:.2f} "
            f"oAR={f.oriented_aspect_ratio:.2f} θ={f.orientation_deg:.0f}°\n"
            f"sol={f.solidity:.2f} holes={f.hole_count} "
            f"offset={f.centroid_offset:.3f}",
            fontsize=8.5,
        )

    def _render_compare_panel(self) -> None:
        for index, ax in enumerate(self.compare_axes, start=1):
            key = self.selected_keys[index - 1] if index <= len(self.selected_keys) else None
            self._render_compare_card(ax, key, index)

    def _render_empty_focus(self) -> None:
        for ax in (self.crop_ax, self.feature_ax, self.shape_ax):
            ax.clear()
            ax.set_axis_off()
            ax.text(
                0.5,
                0.5,
                "Select a cell to inspect",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    def _render_crop(self, snapshot: TileSnapshot) -> None:
        self.crop_ax.clear()
        observation = snapshot.observation
        diagnostics = snapshot.diagnostics

        mask_rgb = np.zeros_like(diagnostics.crop_rgb)
        mask_rgb[diagnostics.foreground_mask > 0] = diagnostics.crop_rgb[
            diagnostics.foreground_mask > 0
        ]
        separator = np.full(
            (diagnostics.crop_rgb.shape[0], 4, 3),
            255,
            dtype=np.uint8,
        )
        combined = np.concatenate((diagnostics.crop_rgb, separator, mask_rgb), axis=1)
        self.crop_ax.imshow(combined)
        self.crop_ax.set_axis_off()
        self.crop_ax.set_title(
            f"FOCUS cell ({observation.row},{observation.col}) — RGB | color foreground\n"
            f"{observation.color.value}  confidence={observation.confidence:.3f}  "
            f"foreground={observation.foreground_fraction:.2f}",
            fontsize=9,
        )

    def _render_features(self, snapshot: TileSnapshot) -> None:
        self.feature_ax.clear()
        diagnostics = snapshot.diagnostics

        hue_axis = np.arange(len(diagnostics.hue_histogram))
        self.feature_ax.plot(hue_axis, diagnostics.hue_histogram)
        if diagnostics.dominant_hue is not None:
            self.feature_ax.axvline(diagnostics.dominant_hue, linestyle="--")
        self.feature_ax.set_xlim(0, 179)
        self.feature_ax.set_xlabel("OpenCV hue", fontsize=8)
        self.feature_ax.set_ylabel("normalized mass", fontsize=8)
        self.feature_ax.tick_params(labelsize=7)
        self.feature_ax.grid(alpha=0.25)

        ranked = sorted(
            diagnostics.class_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        score_text = "  ".join(
            f"{_SYMBOLS[color]}={score:.2f}" for color, score in ranked
        )
        self.feature_ax.set_title(f"FOCUS hue histogram\n{score_text}", fontsize=9)

    def _render_shape(self, snapshot: TileSnapshot) -> None:
        self.shape_ax.clear()
        diagnostics = snapshot.diagnostics
        shape = snapshot.shape

        top = shape.contour_overlay
        separator = np.full((top.shape[0], 4, 3), 255, dtype=np.uint8)
        combined = np.concatenate(
            (
                top,
                separator,
                self._rgb_mask(shape.contour_mask),
                separator,
                self._rgb_mask(diagnostics.overlay_foreground_mask),
            ),
            axis=1,
        )
        self.shape_ax.imshow(combined)
        self.shape_ax.set_axis_off()

        f = shape.features
        semantic = snapshot.semantics
        semantic_line = (
            f"kind={semantic.kind.value}:{semantic.kind_confidence:.2f}  "
            f"blocker={semantic.blocker.value}:{semantic.blocker_confidence:.2f}  "
            f"anomaly={semantic.shape_anomaly:.2f} residual={semantic.residual_fraction:.2f}\n"
            if semantic is not None
            else ""
        )
        self.shape_ax.set_title(
            "FOCUS contour | base | residual\n"
            + semantic_line
            + f"components={f.component_count} holes={f.hole_count} "
            f"area={f.area_fraction:.2f} circ={f.circularity:.2f}\n"
            f"axisAR={f.aspect_ratio:.2f} orientedAR={f.oriented_aspect_ratio:.2f} "
            f"angle={f.orientation_deg:.1f}° extent={f.extent:.2f}\n"
            f"solidity={f.solidity:.2f} centroid_offset={f.centroid_offset:.3f}",
            fontsize=9,
        )

    def _render_focus_panel(self) -> None:
        if self.focused_key is None:
            self._render_empty_focus()
            return

        snapshot = self._snapshot(self.focused_key)
        self._render_crop(snapshot)
        self._render_features(snapshot)
        self._render_shape(snapshot)

    def render(self) -> None:
        """Redraw board, comparison cards, and detailed focus panels."""
        self._render_board()
        self._render_compare_panel()
        self._render_focus_panel()

        status = f" | {self.status_text}" if self.status_text else ""
        self.figure.suptitle(
            "Skydom Tile Debugger — multi-cell comparison"
            f" | selected {len(self.selected_keys)}/{self.max_selected}{status}",
            fontsize=14,
        )
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        """Open the interactive Matplotlib debugger."""
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare color/shape diagnostics for detected Match-3 tiles."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Input screenshot path.")
    source.add_argument("--screen", action="store_true", help="Capture a monitor.")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/board-overlay.png"),
        help="Standard board geometry overlay path.",
    )
    parser.add_argument(
        "--compare-output",
        type=Path,
        default=Path("artifacts/tile-compare.png"),
        help="PNG written when S is pressed in the interactive debugger.",
    )
    return parser


def main() -> int:
    """Detect geometry, classify tile colors, print matrices, and open debugger."""
    args = build_parser().parse_args()
    image = _read_rgb(args.image) if args.image else capture_screen(args.monitor)

    geometry = BoardDetector().detect(image)
    classifier = TileClassifier()
    observations = classifier.classify_board(image, geometry)

    print(format_geometry_summary(geometry))
    print(f"Overlay: {save_board_overlay(image, geometry, args.output)}")
    print("Tile colors:")
    print(format_tile_matrix(geometry, observations))

    unknown = sum(item.color is TileColor.UNKNOWN for item in observations)
    low_confidence = sum(item.confidence < 0.65 for item in observations)
    print(f"Unknown tiles: {unknown}")
    print(f"Low-confidence tiles (<0.65): {low_confidence}")

    TileDebugger(
        image,
        geometry,
        classifier,
        observations,
        compare_output=args.compare_output,
    ).show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
