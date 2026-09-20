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

The visual debugger runs the exact same detection pipeline but retains its intermediate images and signals. It also prints the normal `skydom-inspect` summary and writes `artifacts/board-overlay.png`, so running both commands separately is unnecessary.

```powershell
skydom-debug-vision --screen --monitor 1
```

During iterative development the debugger opens at the first newly interesting step rather than always restarting at step 1. The current default is **step 10**. Override it whenever needed:

```powershell
skydom-debug-vision --screen --monitor 1 --start-step 1
skydom-debug-vision --screen --monitor 1 --start-step 11
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

The debugger therefore acts as the richer inspection command; `skydom-inspect` remains available as the lightweight non-interactive version.

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


## M3: first-stage tile recognition

The first tile recognizer deliberately classifies only the **color family** of each active cell. Shape and special-piece semantics come later.

```powershell
git pull
pip install -e ".[dev,capture]"
skydom-debug-tiles --screen --monitor 1
```

The command:

- detects board geometry;
- prints the same board summary/topology used by the geometry debugger;
- classifies every active cell as `R/O/Y/G/B/P/?`;
- prints unknown and low-confidence counts;
- opens an interactive board where clicking a cell reveals its crop, extracted foreground, hue histogram, dominant hue, and per-color score.

The current classifier uses a classical vision pipeline:

```text
cell crop
  -> center spatial mask
  -> HSV conversion
  -> high-saturation / high-value foreground mask
  -> hue histogram
  -> color-family score
  -> confidence / UNKNOWN fallback
```

This is intentionally simpler than a neural classifier. It makes segmentation and feature design directly observable and gives us a baseline before adding shape recognition or learned models.


### Competitive levels with an opponent mini-board

Some levels render a second, smaller Match-3 board for the opponent. A color-only board detector can accidentally merge both boards because they share the same visual theme.

The geometry pipeline now resolves this *after* pitch inference:

```text
segmented board-like pixels
  -> logical grid at detected player-cell pitch
  -> structural reconciliation
  -> 4-connected topology components
  -> keep the dominant component (and any comparably sized islands)
  -> normalize rows/columns around the retained player board
```

This avoids hard-coding a screen position for the player board. The vision debugger now includes a **Board component selection** step and opens there by default.

### Orange gradient handling

The orange square pieces span both red-orange and orange hue bins because of their strong vertical gradient. The classifier now uses the winning hue-family mass as its confidence baseline instead of heavily penalizing a strong runner-up color. This preserves an `UNKNOWN` fallback while allowing a plurality-orange tile to remain orange.


### Shape diagnostics

The tile debugger now also extracts classical geometric descriptors from the segmented foreground. This stage does not yet assign semantic labels such as `CARROT`, `CHAINED`, or `SPECIAL`; it exposes measurable shape evidence first.

For the selected cell, the debugger shows:

- largest-contour overlay and binary foreground mask;
- connected component count;
- internal hole count;
- area fraction;
- contour circularity;
- bounding-box aspect ratio;
- extent;
- solidity;
- centroid offset.

These features are useful for visually separating cases such as:

```text
yellow ring        -> hole_count > 0
elongated carrot   -> non-square aspect ratio / shifted centroid
fragmented overlay -> multiple components
compact base tile  -> high solidity / compact contour
```

The goal is to inspect real game variants before turning these descriptors into hard semantic rules.


### Separate masks for color and shape

Color recognition and shape recognition now deliberately use different spatial masks.

- **Color mask:** circular, centered, conservative. It samples the tile core and avoids borders, blockers, and neighboring cells.
- **Shape mask:** nearly full-cell rectangular support, with only a small inset to avoid the purple grid/frame. It preserves silhouettes that extend beyond the center, such as carrots, chains, and large special pieces.

This avoids clipping shape descriptors with a mask that was designed for an entirely different task.


### Stabilized shape segmentation

Real game captures showed that tiny per-cell visual differences could radically change raw binary masks. The shape pipeline now uses:

- the already classified tile color as an additional segmentation cue;
- nearly full-cell spatial support;
- morphological close/open cleanup;
- the connected component nearest the known cell center as the base tile silhouette;
- a separate residual-overlay mask for bright/saturated pixels excluded from that base component;
- significant-hole counting instead of counting every threshold artifact;
- rotation-aware bounding boxes via `cv2.minAreaRect`.

This makes descriptors less sensitive to highlights, antialiasing, and differently colored overlays such as chains. Axis-aligned aspect ratio is still reported, but `oriented_aspect_ratio` is the more useful elongation feature for diagonal pieces such as carrots.

The interactive board panel is now cropped to `BoardGeometry.bounds`. Click coordinates are translated back to full-screen coordinates internally, so the user sees only the relevant board without changing cell hit-testing.


### Multi-cell tile comparison

The tile debugger now supports side-by-side comparison of up to six cells while preserving a detailed focus view.

Controls:

```text
click                  focus only this cell
Ctrl+click             add/remove a cell from comparison
right-click            add/remove a cell from comparison
1..6                   focus one of the selected cells
C                      clear the comparison selection
S                      save the current composite view
```

The layout is split into three purposes:

- **board crop:** remains large and clickable, with selected cells numbered in selection order;
- **comparison cards:** up to six compact cards showing RGB, reconstructed base mask, residual overlay, and key shape metrics;
- **focus row:** full color crop, hue histogram/class scores, and detailed shape diagnostics for the currently focused cell.

Expensive per-cell vision diagnostics are cached in a `TileSnapshot`, so changing focus or redrawing the figure does not repeatedly recompute classification and contour features.

Pressing `S` writes:

```text
artifacts/tile-compare.png
```

or another path supplied with:

```powershell
skydom-debug-tiles --screen --monitor 1 --compare-output .\artifacts\my-comparison.png
```

This view is intended for comparing visually equivalent pieces, normal vs blocked variants, and several failure cases in one shareable screenshot.


## M3 semantic interpretation

The perception stack now adds a conservative semantic layer on top of color, shape, and overlay extraction.

Current semantic outputs are:

- `TileKind.NORMAL`
- `TileKind.CARROT`
- `TileKind.UNKNOWN`
- `TileBlocker.NONE`
- `TileBlocker.CHAIN`
- `TileBlocker.UNKNOWN`

The semantic classifier intentionally uses several signals rather than one hard shape threshold:

```text
base color
  + reconstructed base shape
  + residual overlay fraction
  + same-color peer comparison
  -> kind / blocker scores
```

Same-color peers act as adaptive prototypes. For each color family, the classifier computes a robust median feature vector and median-absolute-deviation scale across:

- area fraction;
- circularity;
- rotation-aware aspect ratio;
- solidity;
- centroid offset.

This lets a yellow chained tile be detected as an outlier even when the yellow chain merges into the yellow base mask and produces little different-color residual evidence.

Carrots are currently recognized only when several conservative geometric signals agree (elongated rotated shape, compact/solid silhouette, centered object, and little residual overlay). Ambiguous cells remain `UNKNOWN` instead of forcing a label.

The multi-cell debugger now displays, for each comparison card:

```text
kind=<label>:<confidence>
blocker=<label>:<confidence>
anom=<same-color shape anomaly>
res=<residual overlay fraction>
```

These scores are intended to stay visible while we collect more real game variants before broadening the semantic rules.


### Power-up extensibility and background-aware shape masks

Larger boards exposed two additional perception requirements:

1. blue base pieces can share hue with the dark-blue board background;
2. special/power-up pieces may contain large white or neutral decorations that are weak in hue space.

The tile classifier now estimates a local background color from cell-corner samples in CIELAB space and requires sufficient color distance from that background before a pixel can enter the base-shape mask. This prevents a blue tile's mask from expanding into the whole blue cell background.

It also exposes a separate `neutral_overlay_mask` for bright, low-saturation pixels. The semantic layer treats a sufficiently large neutral overlay as an **unknown power-up candidate** rather than forcing a concrete power-up type before enough examples exist.

Semantic state is now deliberately compositional:

```text
color
kind
blocker
powerup
```

For example:

```text
BLUE + CARROT + NONE + NONE
ORANGE + NORMAL + NONE + UNKNOWN_POWERUP
PURPLE + NORMAL + CHAIN + NONE
```

The exact power-up taxonomy remains open for future additions as new game pieces are observed.


## M4: dataset-first learned recognizer foundation

The project is now deliberately pivoting away from adding more appearance-specific thresholds to the classical tile recognizer.

The current split is:

```text
screen
  -> classical board geometry
  -> logical cells
  -> tile recognizer contract
       -> ClassicalTileRecognizer   (current baseline)
       -> LearnedTileRecognizer     (future)
       -> HybridTileRecognizer      (future)
  -> TileStateEstimate
```

`TileStateEstimate` is the model-neutral contract consumed by later board-state and solver code. It contains only semantic outputs:

- color + confidence;
- kind + confidence;
- blocker + confidence;
- power-up + confidence.

Masks, contours, HSV histograms, CNN tensors, and other implementation-specific diagnostics stay outside that contract. This lets the recognition implementation change without forcing the solver to change.

### Collecting training crops

A new local dataset collector stores every active board cell as a lossless PNG plus a JSON record:

```powershell
skydom-collect-tiles --screen --monitor 1
```

or:

```powershell
skydom-collect-tiles --image .\samples\board.png
```

The default structure is:

```text
dataset/
  images/
    <content-hash>.png
  records/
    <content-hash>.json
```

Crops are de-duplicated by a SHA-256-derived content ID.

Each record intentionally separates:

```text
suggested = current recognizer prediction
labels    = human-confirmed ground truth (initially null)
```

This separation is critical: the classical heuristic output must not silently become training truth, otherwise a learned model would merely reproduce its mistakes.

The local `dataset/` directory is git-ignored.

### Exporting interesting cells from the debugger

In `skydom-debug-tiles`, select up to six cells and press:

```text
E
```

Only the selected cells are exported to the local dataset. Existing human labels in their JSON records are preserved when the same crop is encountered again.

This makes the debugger a lightweight active-data-collection tool: unusual power-ups, blockers, uncertain pieces, and newly discovered object types can be collected while playing without saving full browser screenshots.

The next dataset milestone is human annotation plus a train/validation split. Only after enough representative crops exist will a learned classifier dependency be added.


### Label Studio integration

The Label Studio integration uses a storage-first, incremental layout:

```text
dataset/
  images/                  # canonical crops collected by the bot
  records/                 # canonical sample metadata
  input/                   # everything Label Studio reads
    batch-tasks-0001.json
    batch-tasks-0002.json
    images/
      <sample-id>.png
  output/
    annotations/           # Label Studio Target Storage
  label-studio-config.xml
```

Run:

```powershell
skydom-export-label-studio
```

The exporter copies any missing canonical crops into `dataset/input/images`
and creates a new immutable batch file containing only samples that have not
appeared in earlier `batch-tasks-*.json` files.

For example:

```text
first export:
  batch-tasks-0001.json -> 56 samples

later, after collecting 12 new samples:
  batch-tasks-0002.json -> 12 new samples
```

Existing batch files are never rewritten. If there are no new samples, no empty
batch is created.

Each task contains:

- image URL: `/data/local-files/?d=input/images/<sample-id>.png`;
- `sample_id`;
- row, column, and capture-source metadata;
- classical recognizer output as Label Studio `predictions`, never as human
  ground truth.

#### Local Label Studio startup

Keep Label Studio in its separate Python 3.12 environment and start it through
the repository launcher:

```powershell
.\scripts\start-label-studio.ps1
```

The launcher sets:

```text
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<repo>\dataset
```

The variables are process-local, but the launcher is persistent project
configuration, so they do not need to be recreated manually in every terminal.

#### Source Storage

Configure one Local Files Source Storage:

```text
Storage Title:
Tile Tasks

Absolute local path:
<repo>\dataset\input

Import Method:
Tasks

File Name Filter:
^batch-tasks-.*\.json$

Scan all sub-folders:
off
```

Because the JSON batches live directly in `input/` and the images live in
`input/images/`, one storage subtree contains both task definitions and media.
With subfolder scanning disabled, the image files are not imported as
standalone tasks.

After collecting a new board:

```powershell
skydom-collect-tiles --screen --monitor 1
skydom-export-label-studio
```

then synchronize the `Tile Tasks` Source Storage. Only the newly generated
batch file should create new Label Studio tasks.

#### Target Storage

Configure one Local Files Target Storage:

```text
Storage Title:
Human Annotations

Absolute local path:
<repo>\dataset\output\annotations

Can delete objects from storage:
off
```

Submitted annotations are therefore persisted outside Label Studio's internal
database and can later be normalized back into:

```text
dataset/records/<sample-id>.json -> labels
```

The labeling interface is stored at `dataset/label-studio-config.xml` and
exposes four independent single-choice dimensions: color, kind, blocker, and
power-up.
