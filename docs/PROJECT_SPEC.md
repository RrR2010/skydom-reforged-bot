# Skydom Reforged Bot — Project Specification

## Goal

Build an autonomous Match-3 bot that can perceive the game board from screenshots, reconstruct a semantic board state, evaluate legal moves, and eventually execute them safely.

## Architectural principle

The project is intentionally split into:

```text
capture
  -> board geometry
  -> cell recognition
  -> board state
  -> solver
  -> controller
```

The solver must never depend directly on OpenCV masks, contour features, or model tensors. Recognition implementations produce a stable semantic contract.

## Current perception stack

### Board geometry

`BoardDetector` in `src/skydom_bot/vision/board_detector.py` detects:

- board bounds;
- dynamic row/column count;
- X/Y pitch through autocorrelation;
- irregular topology / missing cells;
- sparse/disconnected player boards;
- opponent mini-board rejection;
- weak cells using visual + structural evidence.

The geometry pipeline is classical CV and is expected to remain classical unless a future failure mode justifies replacing it.

### Tile recognition contract

`TileStateEstimate` is the recognizer-neutral semantic output:

- `color` + confidence;
- `kind` + confidence;
- `blocker` + confidence;
- `powerup` + confidence.

`TileRecognizer` is the protocol consumed by later world-state code.

`ClassicalTileRecognizer` is the current baseline adapter. A future learned recognizer must implement the same contract.

### Current semantic dimensions

```text
TileColor: red, orange, yellow, green, blue, purple, unknown
TileKind: normal, carrot, unknown
TileBlocker: none, chain, unknown
TilePowerup: none, flyer, row, column, bomb, color-remover, unknown
```

The taxonomy is intentionally compositional so new power-ups or modifiers can be added without multiplying combined classes.

## Dataset pipeline

Canonical local structure:

```text
dataset/
  records/
  input/
    batch-tasks-0001.json
    batch-tasks-0002.json
    images/
      <sample-id>.png
  output/
    annotations/
  label-studio-config.xml
```

`DatasetCollector` writes crops directly into `dataset/input/images` and per-sample metadata into `dataset/records`.

`sample_id` is derived from image content, providing stable de-duplication.

Newly collected records also store `capture_ids`: stable hashes of the detected board capture. Re-encountering the same crop on different boards appends provenance instead of discarding it. This metadata is intended for leakage-safe train/validation grouping; older records without it remain valid but should not be used as held-out validation unless their provenance is recovered.

Each record deliberately separates:

```text
suggested = current recognizer prediction
labels    = human ground truth
```

Suggested predictions must never silently become training truth.

Dataset statistics/validation:

```powershell
skydom-dataset-stats
skydom-dataset-stats --strict
```

The command reports labeled/unlabeled totals and per-head human-label distributions for color/kind/blocker/powerup. It validates complete label objects, enum values, sample-id/file consistency, and referenced crop paths. `suggested` predictions are intentionally ignored for label counts.

## Label Studio integration

Source Storage:

```text
<repo>\dataset\input
Import Method: Tasks
File filter: ^batch-tasks-.*\.json$
Scan subfolders: off
```

Target Storage:

```text
<repo>\dataset\output\annotations
```

Launcher:

```powershell
.\scripts\start-label-studio.ps1
```

The launcher sets the local-files document root to `<repo>\dataset`.

Exporter:

```powershell
skydom-export-label-studio
```

It creates immutable incremental batches containing only unseen samples.

Importer:

```powershell
skydom-import-label-studio
```

It reads Label Studio Target Storage output, including the observed extensionless Local Files format, extracts the latest complete human annotation, and writes it into the matching `dataset/records/<sample_id>.json -> labels` without modifying `suggested`.

The end-to-end annotation loop has been validated on real data: Label Studio Target Storage -> importer -> canonical record labels.

## Learned recognizer direction

The next learned model should operate primarily on already detected cell crops rather than re-detecting the whole board.

Preferred initial experiment:

```text
cell crop
  -> lightweight image backbone
  -> independent heads:
       color
       kind
       blocker
       powerup
```

A hybrid experiment may concatenate classical features with the learned image embedding, but only after a pure learned baseline exists.

## Constraints

- Keep code/comments/docstrings in English.
- Prefer conservative `UNKNOWN` over forced wrong classification.
- Preserve debuggability and visual inspection tools.
- Do not hard-code board size or one level layout.
- New appearance types should extend semantic dimensions rather than create combined monolithic classes.
- Human labels are authoritative for training/evaluation.
