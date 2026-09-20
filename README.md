# Skydom Reforged Bot

Computer-vision and planning bot for a Match-3 game. The project is deliberately split into **perception → world state → solver → controller** so the game rules can be tested without moving the mouse and the vision layer can be tested from static screenshots.

## Current milestone: M1 + M2

The first implementation detects the board automatically from a screenshot and reconstructs its logical topology, including irregular shapes and partially occluded surfaces such as ice.

Implemented now:

- RGB screenshot input or optional live capture with `mss`;
- HSV segmentation of the normal board background and ice surfaces;
- grouping of disconnected board fragments;
- automatic cell-pitch estimation using gradient autocorrelation;
- dynamic row/column inference rather than hard-coded board dimensions;
- missing-cell detection for irregular boards;
- diagnostic overlay with logical `(row, col)` coordinates;
- interactive Matplotlib debugger for every perception stage;
- synthetic regression tests.

Tile recognition, game-state recognition, move solving, synchronization, and mouse input intentionally come later.

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev,capture]"
```

## Inspect a saved screenshot

```powershell
skydom-inspect --image .\samples\board.png --output .\artifacts\board-overlay.png
```

Expected console output is similar to:

```text
Board: 8x7
Active cells: 54
Topology:
XXXXXXX
XXXXXXX
XXXXXXX
XXXXXXX
XXXXXXX
XXXXXXX
XXXXXXX
 XXXXX
```

## Inspect the live screen

```powershell
skydom-inspect --screen --monitor 1
```

Live capture currently captures the whole monitor.

## Visual perception debugger

The visual debugger runs the exact same detection pipeline but retains its intermediate images and signals.

```powershell
skydom-debug-vision --screen --monitor 1
```

Navigate with **Previous / Next** or the arrow keys. The steps show:

1. original screenshot and board bounds;
2. HSV mask for the normal board background;
3. ice mask and combined surface evidence;
4. grayscale crop used by pitch detection;
5. X-axis edge-energy image and 1D profile;
6. X-axis normalized autocorrelation and detected peaks;
7. Y-axis edge-energy image and 1D profile;
8. Y-axis normalized autocorrelation and detected peaks;
9. per-cell evidence matrix and strong visual cells;
10. ternary visual evidence: strong / uncertain / absent;
11. structural reconciliation using cardinal-neighbor support;
12. final board topology.

To export every step as a PNG:

```powershell
skydom-debug-vision --screen --monitor 1 --save-steps .\artifacts\vision-steps
```

For headless export without opening a Matplotlib window:

```powershell
skydom-debug-vision --screen --monitor 1 --save-steps .\artifacts\vision-steps --no-window
```

The **cell evidence** and **structural reconciliation** steps are particularly useful when a hint animation, blocker, or special tile weakens a real cell. The detector keeps ambiguous visual evidence instead of discarding it immediately, then conservatively promotes an uncertain interior cell only when the surrounding grid strongly supports its existence.

## Tests

```powershell
pytest
```

## Next validation data

The detector is intentionally heuristic at this stage. Useful fixtures vary one factor at a time:

- a different board shape/level;
- browser zoom or window size;
- a board during a cascade/animation;
- hint/highlight states;
- the level-complete or level-failed screen.

These cases drive changes instead of embedding assumptions from a single level.
