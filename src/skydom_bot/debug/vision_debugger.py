"""Interactive Matplotlib debugger for the board-detection pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from numpy.typing import NDArray

from skydom_bot.capture import capture_screen
from skydom_bot.debug.overlay import draw_board_overlay
from skydom_bot.debug.report import format_geometry_summary, save_board_overlay
from skydom_bot.domain.board import BoardGeometry
from skydom_bot.vision.board_detector import (
    BoardDetectionDiagnostics,
    BoardDetector,
    PitchDiagnostics,
)

UInt8Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class DebugContext:
    """All data required to render each debugger step."""

    image_rgb: UInt8Image
    geometry: BoardGeometry
    diagnostics: BoardDetectionDiagnostics
    occupancy_threshold: float
    uncertain_threshold: float
    required_cardinal_neighbors: int


@dataclass(frozen=True, slots=True)
class DebugStep:
    """One visual explanation step in the perception pipeline."""

    title: str
    explanation: str
    render: Callable[[Axes, Axes, DebugContext], None]


def _read_rgb(path: Path) -> UInt8Image:
    """Read an image file as RGB."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _hide(ax: Axes) -> None:
    """Hide axis decorations for image views."""
    ax.set_axis_off()


def _show_mask(ax: Axes, mask: UInt8Image, title: str) -> None:
    """Render a binary mask with a consistent grayscale scale."""
    ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    _hide(ax)


def _render_original(left: Axes, right: Axes, ctx: DebugContext) -> None:
    left.imshow(ctx.image_rgb)
    b = ctx.diagnostics.bounds
    left.add_patch(
        plt.Rectangle((b.x, b.y), b.width, b.height, fill=False, linewidth=2)
    )
    left.set_title("Original screenshot + detected board bounds")
    _hide(left)

    right.imshow(ctx.diagnostics.crop_rgb)
    right.set_title(
        f"Board crop: {ctx.geometry.rows}x{ctx.geometry.cols} logical grid\n"
        f"{len(ctx.geometry.cells)} active cells"
    )
    _hide(right)


def _render_board_mask(left: Axes, right: Axes, ctx: DebugContext) -> None:
    _show_mask(left, ctx.diagnostics.board_mask, "HSV mask: normal board background")
    _show_mask(right, ctx.diagnostics.component_mask, "Grouped board fragments actually selected")


def _render_ice_and_evidence(left: Axes, right: Axes, ctx: DebugContext) -> None:
    _show_mask(left, ctx.diagnostics.ice_mask, "HSV mask: ice / cyan surface")
    _show_mask(
        right,
        ctx.diagnostics.evidence_mask,
        "Combined surface evidence = board background OR ice",
    )


def _render_gray(left: Axes, right: Axes, ctx: DebugContext) -> None:
    left.imshow(ctx.diagnostics.crop_rgb)
    left.set_title("RGB crop used for pitch detection")
    _hide(left)

    right.imshow(ctx.diagnostics.crop_gray, cmap="gray")
    right.set_title("Grayscale crop")
    _hide(right)


def _render_edge_profile(
    left: Axes,
    right: Axes,
    pitch: PitchDiagnostics,
    title: str,
) -> None:
    left.imshow(pitch.edge_energy, cmap="gray", aspect="auto")
    left.set_title(f"{title}: absolute first derivative / edge energy")
    left.set_xlabel("x")
    left.set_ylabel("y")

    right.plot(np.arange(len(pitch.profile)), pitch.profile)
    right.axhline(0, linewidth=0.8)
    right.set_title(f"{title}: 2D edge image collapsed into a 1D profile")
    right.set_xlabel("pixel position")
    right.set_ylabel("mean edge energy (centered)")
    right.grid(alpha=0.25)


def _render_edges_x(left: Axes, right: Axes, ctx: DebugContext) -> None:
    _render_edge_profile(left, right, ctx.diagnostics.pitch_x, "X-axis")


def _render_edges_y(left: Axes, right: Axes, ctx: DebugContext) -> None:
    _render_edge_profile(left, right, ctx.diagnostics.pitch_y, "Y-axis")


def _render_autocorrelation(
    left: Axes,
    right: Axes,
    pitch: PitchDiagnostics,
    title: str,
) -> None:
    valid = np.isfinite(pitch.scores)
    lags = pitch.lags[valid]
    scores = pitch.scores[valid]

    left.plot(lags, scores, marker=".", markersize=3)
    for peak in pitch.peaks:
        left.axvline(peak, linestyle="--", alpha=0.45)
    left.axvline(pitch.selected_pitch, linewidth=2)
    left.set_title(
        f"{title}: normalized autocorrelation\n"
        f"selected fundamental pitch = {pitch.selected_pitch:.0f}px"
    )
    left.set_xlabel("lag / displacement (px)")
    left.set_ylabel("similarity")
    left.grid(alpha=0.25)

    if pitch.peaks:
        peak_scores = [pitch.scores[peak] for peak in pitch.peaks]
        right.bar([str(peak) for peak in pitch.peaks], peak_scores)
        right.set_title("Strong local peaks (harmonics included)")
        right.set_xlabel("lag (px)")
        right.set_ylabel("autocorrelation")
    else:
        right.text(0.5, 0.5, "No local peaks\n(fallback to global maximum)", ha="center", va="center")
        _hide(right)


def _render_autocorrelation_x(left: Axes, right: Axes, ctx: DebugContext) -> None:
    _render_autocorrelation(left, right, ctx.diagnostics.pitch_x, "X-axis")


def _render_autocorrelation_y(left: Axes, right: Axes, ctx: DebugContext) -> None:
    _render_autocorrelation(left, right, ctx.diagnostics.pitch_y, "Y-axis")


def _render_occupancy(left: Axes, right: Axes, ctx: DebugContext) -> None:
    matrix = ctx.diagnostics.occupancy
    image = left.imshow(matrix, vmin=0.0, vmax=1.0)
    left.set_title("Per-cell corner evidence before thresholding")
    left.set_xlabel("column")
    left.set_ylabel("row")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            left.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(image, ax=left, fraction=0.046, pad=0.04)

    strong = matrix >= ctx.occupancy_threshold
    right.imshow(strong, cmap="gray", vmin=0, vmax=1)
    right.set_title(f"Strong visual cells >= {ctx.occupancy_threshold:.2f}")
    right.set_xlabel("column")
    right.set_ylabel("row")

    for row in range(strong.shape[0]):
        for col in range(strong.shape[1]):
            right.text(
                col,
                row,
                "X" if strong[row, col] else ".",
                ha="center",
                va="center",
                fontsize=9,
            )


def _render_evidence_classes(left: Axes, right: Axes, ctx: DebugContext) -> None:
    state = ctx.diagnostics.evidence_state
    left.imshow(state, vmin=0, vmax=2)
    left.set_title(
        "Ternary visual evidence\n"
        f"strong >= {ctx.occupancy_threshold:.2f}, "
        f"uncertain >= {ctx.uncertain_threshold:.2f}"
    )
    left.set_xlabel("column")
    left.set_ylabel("row")

    labels = {0: ".", 1: "?", 2: "X"}
    for row in range(state.shape[0]):
        for col in range(state.shape[1]):
            left.text(
                col,
                row,
                labels[int(state[row, col])],
                ha="center",
                va="center",
                fontsize=10,
            )

    matrix = ctx.diagnostics.occupancy
    right.imshow(matrix, vmin=0.0, vmax=1.0)
    right.set_title("Raw score retained underneath the class")
    right.set_xlabel("column")
    right.set_ylabel("row")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            right.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=8)


def _render_structural_reconciliation(left: Axes, right: Axes, ctx: DebugContext) -> None:
    support = ctx.diagnostics.cardinal_support
    state = ctx.diagnostics.evidence_state
    topology = ctx.diagnostics.reconciled_topology

    left.imshow(support, vmin=0, vmax=4)
    left.set_title("Strong cardinal-neighbor support (0..4)")
    left.set_xlabel("column")
    left.set_ylabel("row")
    for row in range(support.shape[0]):
        for col in range(support.shape[1]):
            left.text(col, row, str(int(support[row, col])), ha="center", va="center", fontsize=9)

    right.imshow(topology, cmap="gray", vmin=0, vmax=1)
    right.set_title(
        "Reconciled topology\n"
        f"uncertain cells promoted with >= {ctx.required_cardinal_neighbors} strong neighbors"
    )
    right.set_xlabel("column")
    right.set_ylabel("row")
    for row in range(topology.shape[0]):
        for col in range(topology.shape[1]):
            promoted = bool(topology[row, col]) and int(state[row, col]) == 1
            symbol = "P" if promoted else ("X" if topology[row, col] else ".")
            right.text(col, row, symbol, ha="center", va="center", fontsize=9)


def _render_component_selection(left: Axes, right: Axes, ctx: DebugContext) -> None:
    labels = ctx.diagnostics.topology_components
    selected = ctx.diagnostics.selected_topology
    sizes = ctx.diagnostics.topology_component_sizes
    means = ctx.diagnostics.topology_component_mean_evidence

    left.imshow(labels)
    stats = ", ".join(
        f"{index}: n={size}, mean={mean:.2f}"
        for index, (size, mean) in enumerate(zip(sizes, means), start=1)
    ) or "single component"
    left.set_title(
        "4-connected topology components before board selection\n"
        f"{stats}"
    )
    left.set_xlabel("column")
    left.set_ylabel("row")
    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            label = int(labels[row, col])
            left.text(
                col,
                row,
                "." if label == 0 else str(label),
                ha="center",
                va="center",
                fontsize=8,
            )

    right.imshow(selected, cmap="gray", vmin=0, vmax=1)
    right.set_title(
        "Retained player-board topology before normalization\n"
        "keep comparable components or small islands with strong same-scale evidence"
    )
    right.set_xlabel("column")
    right.set_ylabel("row")
    for row in range(selected.shape[0]):
        for col in range(selected.shape[1]):
            right.text(
                col,
                row,
                "X" if selected[row, col] else ".",
                ha="center",
                va="center",
                fontsize=8,
            )


def _render_final(left: Axes, right: Axes, ctx: DebugContext) -> None:
    overlay = draw_board_overlay(ctx.image_rgb, ctx.geometry)
    left.imshow(overlay)
    left.set_title("Final board geometry")
    _hide(left)

    right.text(
        0.04,
        0.96,
        (
            f"Board: {ctx.geometry.rows}x{ctx.geometry.cols}\n"
            f"Active cells: {len(ctx.geometry.cells)}\n"
            f"Pitch: {ctx.geometry.pitch_x:.2f} x {ctx.geometry.pitch_y:.2f}\n\n"
            f"{ctx.geometry.topology_text(active='X', empty='.')}"
        ),
        va="top",
        family="monospace",
        fontsize=12,
    )
    right.set_title("Discrete result")
    _hide(right)


def _steps() -> tuple[DebugStep, ...]:
    """Return the ordered visual explanation of the detector."""
    return (
        DebugStep(
            "1. Input and board bounds",
            "Start from the full screenshot. The rectangle is the board region inferred from grouped blue fragments.",
            _render_original,
        ),
        DebugStep(
            "2. Normal board mask",
            "HSV thresholding converts the screenshot into a binary question: which pixels look like the normal dark-blue board surface?",
            _render_board_mask,
        ),
        DebugStep(
            "3. Alternate surface evidence",
            "Ice is a second visual cue. It is not used to define the board alone, but it can support cell existence where the normal blue background is occluded.",
            _render_ice_and_evidence,
        ),
        DebugStep(
            "4. Pitch input",
            "Pitch estimation ignores tile identity. It uses repeated spatial structure in the board crop after grayscale conversion.",
            _render_gray,
        ),
        DebugStep(
            "5. X edge profile",
            "A first derivative emphasizes vertical boundaries. Averaging the 2D edge image creates a 1D signal with repeated peaks near cell boundaries.",
            _render_edges_x,
        ),
        DebugStep(
            "6. X autocorrelation",
            "The 1D signal is compared with shifted copies of itself. Strong similarity at ~one cell width reveals the grid period; larger multiples are harmonics.",
            _render_autocorrelation_x,
        ),
        DebugStep(
            "7. Y edge profile",
            "The same transformation is repeated vertically so both axes independently estimate the square-cell spacing.",
            _render_edges_y,
        ),
        DebugStep(
            "8. Y autocorrelation",
            "Strong local peaks are inspected and the smallest strong peak is chosen as the fundamental period.",
            _render_autocorrelation_y,
        ),
        DebugStep(
            "9. Cell evidence",
            "Each logical cell receives a continuous score from surface evidence in its corners. The right panel shows only cells that are visually strong by themselves.",
            _render_occupancy,
        ),
        DebugStep(
            "10. Strong / uncertain / absent",
            "Instead of forcing an immediate yes/no decision, visual evidence is kept ternary: strong (X), uncertain (?), or absent (.). This preserves ambiguity for the next stage.",
            _render_evidence_classes,
        ),
        DebugStep(
            "11. Structural reconciliation",
            "Uncertain cells are compared with the already-strong grid around them. The current rule is deliberately conservative: only an uncertain interior cell surrounded by strong cardinal neighbors is promoted (P).",
            _render_structural_reconciliation,
        ),
        DebugStep(
            "12. Board component selection",
            "A screen may contain another Match-3 board, such as an opponent preview. The reconciled logical grid is split into 4-connected components; small disconnected replicas are rejected while comparable islands are preserved.",
            _render_component_selection,
        ),
        DebugStep(
            "13. Final topology",
            "The selected player-board component is normalized to its own row/column bounds and becomes the discrete grid representation used by later solver layers.",
            _render_final,
        ),
    )


class VisionDebugger:
    """Interactive step-by-step Matplotlib viewer."""

    def __init__(self, context: DebugContext, start_step: int = 1) -> None:
        self.context = context
        self.steps = _steps()
        self.index = min(max(start_step, 1), len(self.steps)) - 1

        self.figure: Figure = plt.figure(figsize=(15, 8))
        grid = self.figure.add_gridspec(
            2,
            2,
            height_ratios=(12, 1.4),
            left=0.04,
            right=0.98,
            top=0.91,
            bottom=0.10,
            wspace=0.18,
        )
        self.left = self.figure.add_subplot(grid[0, 0])
        self.right = self.figure.add_subplot(grid[0, 1])
        self.info = self.figure.add_subplot(grid[1, :])
        self.info.set_axis_off()

        previous_ax = self.figure.add_axes((0.39, 0.02, 0.10, 0.045))
        next_ax = self.figure.add_axes((0.51, 0.02, 0.10, 0.045))
        self.previous_button = Button(previous_ax, "Previous")
        self.next_button = Button(next_ax, "Next")
        self.previous_button.on_clicked(lambda _: self.previous())
        self.next_button.on_clicked(lambda _: self.next())
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)

        self.render()

    def _on_key(self, event: object) -> None:
        key = getattr(event, "key", None)
        if key in {"right", "down", " ", "enter"}:
            self.next()
        elif key in {"left", "up", "backspace"}:
            self.previous()

    def next(self) -> None:
        """Advance one visual step."""
        self.index = min(len(self.steps) - 1, self.index + 1)
        self.render()

    def previous(self) -> None:
        """Go back one visual step."""
        self.index = max(0, self.index - 1)
        self.render()

    def render(self) -> None:
        """Redraw the current step."""
        self.left.clear()
        self.right.clear()
        self.info.clear()
        self.info.set_axis_off()

        step = self.steps[self.index]
        step.render(self.left, self.right, self.context)
        self.info.text(
            0.0,
            0.72,
            step.explanation,
            va="top",
            wrap=True,
            fontsize=10,
        )
        self.info.text(
            0.0,
            0.15,
            f"Step {self.index + 1}/{len(self.steps)}  |  Use Previous/Next or arrow keys",
            fontsize=9,
        )
        self.figure.suptitle(f"Skydom Vision Debugger — {step.title}", fontsize=14)
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        """Open the interactive Matplotlib window."""
        plt.show()


def _save_steps(context: DebugContext, directory: Path) -> None:
    """Export every debugger step as a PNG for sharing or offline review."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, step in enumerate(_steps(), start=1):
        figure, (left, right) = plt.subplots(1, 2, figsize=(15, 7))
        step.render(left, right, context)
        figure.suptitle(f"{index:02d}. {step.title}\n{step.explanation}", fontsize=12)
        figure.tight_layout()
        path = directory / f"{index:02d}-{step.title.lower().replace(' ', '-').replace('.', '')}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI arguments for the visual debugger."""
    parser = argparse.ArgumentParser(description="Visualize each stage of Skydom board detection.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Input screenshot path.")
    source.add_argument("--screen", action="store_true", help="Capture a monitor.")
    parser.add_argument("--monitor", type=int, default=1, help="MSS monitor index used with --screen.")
    parser.add_argument(
        "--save-steps",
        type=Path,
        help="Optional directory to export every visual step as a PNG.",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=12,
        help=(
            "1-based debugger step to open first. Defaults to 12, the latest "
            "board-selection concept introduced for multi-board competitive levels."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/board-overlay.png"),
        help="Standard board overlay path, also generated by this debugger.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open the interactive window. Useful together with --save-steps.",
    )
    return parser


def main() -> int:
    """Run the board detector once and inspect its intermediate representations."""
    args = build_parser().parse_args()
    image = _read_rgb(args.image) if args.image else capture_screen(args.monitor)

    detector = BoardDetector()
    geometry, diagnostics = detector.detect_with_diagnostics(image)
    context = DebugContext(
        image_rgb=image,
        geometry=geometry,
        diagnostics=diagnostics,
        occupancy_threshold=detector.config.occupancy_threshold,
        uncertain_threshold=detector.config.occupancy_uncertain_threshold,
        required_cardinal_neighbors=detector.config.structural_required_cardinal_neighbors,
    )

    print(format_geometry_summary(geometry))
    print(f"Overlay: {save_board_overlay(image, geometry, args.output)}")

    if args.save_steps:
        _save_steps(context, args.save_steps)
        print(f"Saved visual pipeline to: {args.save_steps}")

    if not args.no_window:
        VisionDebugger(context, start_step=args.start_step).show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
