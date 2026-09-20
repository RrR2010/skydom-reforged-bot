"""Interactive debugger for first-stage tile color recognition."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure
from numpy.typing import NDArray

from skydom_bot.capture import capture_screen
from skydom_bot.debug.report import format_geometry_summary, save_board_overlay
from skydom_bot.domain.board import BoardGeometry, Cell
from skydom_bot.domain.tile import TileColor, TileObservation
from skydom_bot.vision.board_detector import BoardDetector
from skydom_bot.vision.shape_features import ShapeDiagnostics, extract_shape_features
from skydom_bot.vision.tile_classifier import TileClassifier, TileDiagnostics

UInt8Image = NDArray[np.uint8]

_SYMBOLS = {
    TileColor.RED: "R",
    TileColor.ORANGE: "O",
    TileColor.YELLOW: "Y",
    TileColor.GREEN: "G",
    TileColor.BLUE: "B",
    TileColor.PURPLE: "P",
    TileColor.UNKNOWN: "?",
}


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
    """Click a detected cell to inspect its segmentation and color features."""

    def __init__(
        self,
        image_rgb: UInt8Image,
        geometry: BoardGeometry,
        classifier: TileClassifier,
        observations: tuple[TileObservation, ...],
    ) -> None:
        self.image_rgb = image_rgb
        self.geometry = geometry
        self.classifier = classifier
        self.observations = observations
        self.by_cell = {(item.row, item.col): item for item in observations}
        self.cells = geometry.cell_map

        self.figure: Figure = plt.figure(figsize=(15, 8))
        grid = self.figure.add_gridspec(
            2,
            3,
            left=0.04,
            right=0.98,
            top=0.91,
            bottom=0.08,
            wspace=0.18,
            hspace=0.30,
        )
        self.board_ax = self.figure.add_subplot(grid[:, 0])
        self.crop_ax = self.figure.add_subplot(grid[0, 1])
        self.feature_ax = self.figure.add_subplot(grid[1, 1])
        self.shape_ax = self.figure.add_subplot(grid[:, 2])

        self.selected = self._initial_cell()
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.render()

    def _initial_cell(self) -> Cell:
        """Start at the least-confident cell because it is most informative."""
        ranked = sorted(self.observations, key=lambda item: item.confidence)
        if ranked:
            key = (ranked[0].row, ranked[0].col)
            if key in self.cells:
                return self.cells[key]
        return next(iter(self.cells.values()))

    def _on_click(self, event: MouseEvent) -> None:
        if event.inaxes is not self.board_ax or event.xdata is None or event.ydata is None:
            return

        x, y = float(event.xdata), float(event.ydata)
        for cell in self.geometry.cells:
            if (
                cell.bounds.x <= x < cell.bounds.right
                and cell.bounds.y <= y < cell.bounds.bottom
            ):
                self.selected = cell
                self.render()
                return

    def _render_board(self) -> None:
        self.board_ax.clear()
        self.board_ax.imshow(self.image_rgb)
        self.board_ax.set_title("Recognized board — click any active cell")
        self.board_ax.set_axis_off()

        for cell in self.geometry.cells:
            observation = self.by_cell[(cell.row, cell.col)]
            selected = cell.row == self.selected.row and cell.col == self.selected.col
            linewidth = 3 if selected else 1
            self.board_ax.add_patch(
                plt.Rectangle(
                    (cell.bounds.x, cell.bounds.y),
                    cell.bounds.width,
                    cell.bounds.height,
                    fill=False,
                    linewidth=linewidth,
                )
            )
            self.board_ax.text(
                cell.center.x,
                cell.center.y,
                f"{_SYMBOLS[observation.color]}\n{observation.confidence:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    def _render_crop(self, observation: TileObservation, diagnostics: TileDiagnostics) -> None:
        self.crop_ax.clear()

        # Show RGB and segmentation side by side inside the same axis by
        # concatenating them. This keeps the interactive layout compact.
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
            f"Cell ({observation.row},{observation.col}) — RGB | extracted foreground\n"
            f"prediction={observation.color.value}  confidence={observation.confidence:.3f}  "
            f"color_foreground={observation.foreground_fraction:.2f}"
        )

    def _render_features(self, diagnostics: TileDiagnostics) -> None:
        self.feature_ax.clear()

        hue_axis = np.arange(len(diagnostics.hue_histogram))
        self.feature_ax.plot(hue_axis, diagnostics.hue_histogram)
        if diagnostics.dominant_hue is not None:
            self.feature_ax.axvline(diagnostics.dominant_hue, linestyle="--")
        self.feature_ax.set_xlim(0, 179)
        self.feature_ax.set_xlabel("OpenCV hue (0..179)")
        self.feature_ax.set_ylabel("normalized foreground hue mass")
        self.feature_ax.grid(alpha=0.25)

        ranked = sorted(
            diagnostics.class_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        score_text = "   ".join(
            f"{_SYMBOLS[color]}={score:.2f}" for color, score in ranked
        )
        self.feature_ax.set_title(
            "Hue histogram after spatial + saturation/value segmentation\n"
            f"{score_text}"
        )

    def _render_shape(self, diagnostics: ShapeDiagnostics) -> None:
        self.shape_ax.clear()

        top = diagnostics.contour_overlay
        mask_rgb = np.repeat(diagnostics.contour_mask[:, :, None], 3, axis=2)
        separator = np.full((top.shape[0], 4, 3), 255, dtype=np.uint8)
        combined = np.concatenate((top, separator, mask_rgb), axis=1)
        self.shape_ax.imshow(combined)
        self.shape_ax.set_axis_off()

        f = diagnostics.features
        self.shape_ax.set_title(
            "Shape descriptors — full-cell foreground, no circular color mask\n"
            "contour overlay | binary mask\n"
            f"components={f.component_count}  holes={f.hole_count}\n"
            f"area={f.area_fraction:.2f}  circularity={f.circularity:.2f}\n"
            f"aspect={f.aspect_ratio:.2f}  extent={f.extent:.2f}\n"
            f"solidity={f.solidity:.2f}  centroid_offset={f.centroid_offset:.3f}"
        )

    def render(self) -> None:
        """Redraw all panels for the selected logical cell."""
        observation, diagnostics = self.classifier.classify_cell(
            self.image_rgb,
            self.selected,
        )
        shape = extract_shape_features(
            diagnostics.crop_rgb,
            diagnostics.shape_foreground_mask,
        )
        self._render_board()
        self._render_crop(observation, diagnostics)
        self._render_features(diagnostics)
        self._render_shape(shape)
        self.figure.suptitle(
            "Skydom Tile Debugger — color + classical shape descriptors",
            fontsize=14,
        )
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        """Open the interactive Matplotlib debugger."""
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect first-stage color classification for every detected tile."
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

    TileDebugger(image, geometry, classifier, observations).show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
