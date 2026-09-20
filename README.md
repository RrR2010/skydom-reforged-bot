# Skydom Reforged Bot

Computer-vision and planning bot for a Match-3 game. The project is deliberately split into **perception → world state → solver → controller** so the game rules can be tested without moving the mouse and the vision layer can be tested from static screenshots.

## Current milestone: M1 + M2

The first implementation detects the board automatically from a screenshot and reconstructs its logical topology, including irregular shapes such as a 7-column board whose last row contains only 5 cells.

Implemented now:

- RGB screenshot input or optional live capture with `mss`;
- HSV segmentation of the current dark-blue Skydom board theme;
- automatic board bounding box detection;
- automatic cell-pitch estimation using gradient autocorrelation;
- dynamic row/column inference rather than hard-coded board dimensions;
- missing-cell detection for irregular boards;
- diagnostic overlay with logical `(row, col)` coordinates;
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

Live capture currently captures the whole monitor. Window-specific discovery/cropping is planned after the board detector has been validated against screenshots from different levels, resolutions, and browser zoom values.

## Tests

```powershell
pytest
```

## Next validation data

The detector is intentionally heuristic at this stage. The most useful next fixtures are screenshots that vary one factor at a time:

- a different board shape/level;
- browser zoom or window size;
- a board during a cascade/animation;
- the level-complete or level-failed screen.

These will drive the next changes instead of embedding assumptions from a single level.
