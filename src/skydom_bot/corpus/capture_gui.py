"""Compact Tkinter form for repeated screen-corpus capture."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from skydom_bot.corpus.capture import load_region, save_capture


@dataclass(slots=True)
class CaptureFormState:
    """Values retained by the capture form between captures and launches."""

    stage_id: str = ""
    mode: str = ""
    initial_moves: str = ""
    board_variant: str = ""
    has_ice: bool = False
    custom: list[tuple[str, str]] = field(
        default_factory=lambda: [("", "") for _ in range(4)]
    )


def form_state_path(root: Path) -> Path:
    """Return the local persisted capture-form state path."""
    return root / "capture-form.json"


def load_form_state(root: Path) -> CaptureFormState:
    """Load the last form values, falling back to an empty state."""
    path = form_state_path(root)
    if not path.is_file():
        return CaptureFormState()

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CaptureFormState()

    custom = payload.get("custom", [])
    normalized_custom = [
        (str(item[0]), str(item[1]))
        for item in custom
        if isinstance(item, list) and len(item) == 2
    ][:4]
    normalized_custom.extend([("", "")] * (4 - len(normalized_custom)))

    return CaptureFormState(
        stage_id=str(payload.get("stage_id", "")),
        mode=str(payload.get("mode", "")),
        initial_moves=str(payload.get("initial_moves", "")),
        board_variant=str(payload.get("board_variant", "")),
        has_ice=bool(payload.get("has_ice", False)),
        custom=normalized_custom,
    )


def save_form_state(root: Path, state: CaptureFormState) -> Path:
    """Persist the current form values locally."""
    root.mkdir(parents=True, exist_ok=True)
    path = form_state_path(root)
    payload = asdict(state)
    payload["custom"] = [list(item) for item in state.custom]
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def build_metadata(state: CaptureFormState) -> dict[str, str]:
    """Build canonical and custom metadata for one capture."""
    metadata = {
        "mode": state.mode.strip(),
        "initial_moves": state.initial_moves.strip(),
        "board_variant": state.board_variant.strip(),
        "has_ice": "yes" if state.has_ice else "no",
    }
    for key, value in state.custom:
        key = key.strip()
        if key:
            metadata[key] = value.strip()
    return metadata


def build_parser() -> argparse.ArgumentParser:
    """Build capture-form command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="skydom-capture-gui",
        description="Open a compact persistent form for repeated corpus captures.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("corpus"),
        help="Local corpus root (default: corpus).",
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=None,
        help="Override the monitor saved in corpus/region.json.",
    )
    return parser


def _state_from_widgets(
    *,
    stage_id: str,
    mode: str,
    initial_moves: str,
    board_variant: str,
    has_ice: bool,
    custom: list[tuple[str, str]],
) -> CaptureFormState:
    """Normalize widget values into serializable state."""
    return CaptureFormState(
        stage_id=stage_id.strip(),
        mode=mode.strip(),
        initial_moves=initial_moves.strip(),
        board_variant=board_variant.strip(),
        has_ice=has_ice,
        custom=[(key.strip(), value.strip()) for key, value in custom],
    )


def run_form(root: Path, *, monitor_override: int | None = None) -> int:
    """Open the compact capture form and run its event loop."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    configured_monitor, region = load_region(root)
    monitor = monitor_override or configured_monitor
    initial = load_form_state(root)

    window = tk.Tk()
    window.title("Skydom Capture")
    window.resizable(False, False)
    window.attributes("-topmost", True)

    frame = ttk.Frame(window, padding=8)
    frame.grid(sticky="nsew")

    stage_var = tk.StringVar(value=initial.stage_id)
    mode_var = tk.StringVar(value=initial.mode)
    moves_var = tk.StringVar(value=initial.initial_moves)
    variant_var = tk.StringVar(value=initial.board_variant)
    ice_var = tk.BooleanVar(value=initial.has_ice)
    topmost_var = tk.BooleanVar(value=True)
    status_var = tk.StringVar(value=f"Ready · monitor {monitor}")

    custom_vars: list[tuple[tk.StringVar, tk.StringVar]] = [
        (tk.StringVar(value=key), tk.StringVar(value=value))
        for key, value in initial.custom
    ]

    row = 0

    def labeled_entry(label: str, variable: tk.StringVar) -> None:
        nonlocal row
        ttk.Label(frame, text=label).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 2)
        )
        row += 1
        ttk.Entry(frame, textvariable=variable, width=31).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(0, 6)
        )
        row += 1

    labeled_entry("Stage ID", stage_var)
    labeled_entry("Mode", mode_var)
    labeled_entry("Initial moves", moves_var)
    labeled_entry("Board variant", variant_var)

    ttk.Checkbutton(frame, text="Has ice", variable=ice_var).grid(
        row=row, column=0, sticky="w", pady=(0, 6)
    )
    ttk.Checkbutton(
        frame,
        text="Always on top",
        variable=topmost_var,
        command=lambda: window.attributes("-topmost", topmost_var.get()),
    ).grid(row=row, column=1, sticky="e", pady=(0, 6))
    row += 1

    ttk.Separator(frame, orient="horizontal").grid(
        row=row, column=0, columnspan=2, sticky="ew", pady=(2, 7)
    )
    row += 1

    ttk.Label(frame, text="Extra metadata").grid(
        row=row, column=0, columnspan=2, sticky="w", pady=(0, 3)
    )
    row += 1
    ttk.Label(frame, text="Name").grid(row=row, column=0, sticky="w")
    ttk.Label(frame, text="Value").grid(row=row, column=1, sticky="w")
    row += 1

    for key_var, value_var in custom_vars:
        ttk.Entry(frame, textvariable=key_var, width=14).grid(
            row=row, column=0, sticky="ew", padx=(0, 3), pady=2
        )
        ttk.Entry(frame, textvariable=value_var, width=16).grid(
            row=row, column=1, sticky="ew", pady=2
        )
        row += 1

    ttk.Separator(frame, orient="horizontal").grid(
        row=row, column=0, columnspan=2, sticky="ew", pady=(7, 7)
    )
    row += 1

    def current_state() -> CaptureFormState:
        return _state_from_widgets(
            stage_id=stage_var.get(),
            mode=mode_var.get(),
            initial_moves=moves_var.get(),
            board_variant=variant_var.get(),
            has_ice=ice_var.get(),
            custom=[
                (key_var.get(), value_var.get())
                for key_var, value_var in custom_vars
            ],
        )

    def capture() -> None:
        state = current_state()
        if not state.stage_id:
            messagebox.showerror("Skydom Capture", "Stage ID is required.")
            return

        try:
            save_form_state(root, state)
            record = save_capture(
                root,
                stage_id=state.stage_id,
                monitor=monitor,
                region=region,
                metadata=build_metadata(state),
            )
        except Exception as exc:
            status_var.set("Capture failed")
            messagebox.showerror("Skydom Capture", str(exc))
            return

        status_var.set(f"Captured · {record.captured_at[11:19]}")

    ttk.Button(frame, text="CAPTURE", command=capture).grid(
        row=row,
        column=0,
        columnspan=2,
        sticky="ew",
        ipady=5,
        pady=(0, 5),
    )
    row += 1

    ttk.Label(frame, textvariable=status_var, anchor="center").grid(
        row=row, column=0, columnspan=2, sticky="ew"
    )

    def close() -> None:
        save_form_state(root, current_state())
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", close)
    window.bind("<Control-Return>", lambda _event: capture())
    window.mainloop()
    return 0


def main() -> int:
    """Launch the persistent corpus-capture form."""
    args = build_parser().parse_args()
    return run_form(args.corpus, monitor_override=args.monitor)


if __name__ == "__main__":
    raise SystemExit(main())
