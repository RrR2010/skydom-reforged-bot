# Handoff — Skydom Reforged Bot

## Repository / branch

- Repository: `RrR2010/skydom-reforged-bot`
- Active branch: `feat/screen-corpus-capture`
- Active PR: `#1`
- User runs locally on Windows PowerShell.
- Bot environment: `.venv`.
- Label Studio environment: `.venv-labelstudio` with Python 3.12.

## User working style

- User knows Python basics and wants important CV/architecture concepts explained, not line-by-line code.
- Code/comments/docstrings should be English.
- User appreciates visual debuggers and wants them retained.
- User is actively playing new levels to surface variants and edge cases.
- Avoid getting trapped in heuristic rabbit holes; periodically reassess whether a learned/hybrid approach is more appropriate.

## Current state

### Geometry

`BoardDetector` is mature enough for current levels. It supports dynamic board sizes, irregular/sparse topology, weak structural cells, hints, blockers, and competitor mini-board rejection.

### Classical tile perception

Current pipeline provides:

- base color classification;
- reconstructed shape features;
- residual/neutral overlays;
- semantic baseline for normal/carrot, chain/no-chain, unknown power-up.

This classical layer is now a baseline/data-bootstrap tool, not the intended final semantic recognizer.

Observed power-up labels now in the taxonomy: `flyer`, `row`, `column`, `bomb`, and `color-remover`, plus `none` and `unknown`.

A new whole-cell obstacle has been observed: it has no matchable color, cannot participate in swaps/matches, and is removed when an adjacent match touches it. Its synchronized semantic representation is `color=none`, `kind=none`, `blocker=adjacent-clear`, `powerup=none`. `unknown` is reserved for genuine recognition uncertainty rather than known non-applicability.

### Debugger

`skydom-debug-tiles` supports multi-cell comparison and exporting selected cells with `E`.

### Screen corpus capture

A full-game-area corpus workflow is available independently of the tile dataset.

- `skydom-capture-screen --configure` stores the reusable monitor-local game region.
- `skydom-capture-screen <stage>` performs one CLI capture.
- `skydom-capture-gui` opens a narrow always-on-top Tkinter form for repeated manual capture.
- The GUI persists its last values in `corpus/capture-form.json`.
- Canonical GUI metadata keys are `mode`, `initial_moves`, `board_variant`, and `has_ice`; four additional arbitrary key/value rows are available and cannot overwrite canonical keys.
- Repeated captures with identical stage/metadata accumulate because each image receives a timestamped `capture_id`.
- `skydom-analyze-corpus` evaluates captured screens offline with the current board detector.

### Dataset

Canonical structure after latest refactor:

```text
dataset/
  records/
  input/
    batch-tasks-0001.json
    images/
  output/
    annotations/
  label-studio-config.xml
```

`DatasetCollector` now writes directly to `dataset/input/images`.

Latest confirmed local validation before the capture-provenance/taxonomy commits: full `pytest` passed with 50 tests. Re-run the suite after pulling the latest changes. Real dataset statistics at that point were 109 labeled / 0 unlabeled / 0 invalid records. The blocker head was still entirely `none`, so more blocker examples remain a priority.

Dataset statistics/validation is available with:

```powershell
skydom-dataset-stats
skydom-dataset-stats --strict
```

It counts human-labeled vs unlabeled records, reports class distributions for all four semantic heads, validates complete label objects and known enum values, checks sample-id/file consistency and crop existence, reports capture-provenance coverage, and never treats `suggested` as ground truth.

New collections store one or more `capture_ids` per sample. The ID hashes the detected board crop + topology, so all crops from one board capture can later be kept on the same side of the train/validation split. Existing legacy records without `capture_ids` remain usable for training but should not be treated as leakage-safe held-out validation unless recollected/provenance is recovered.

Legacy `dataset/images` is deprecated. The exporter can migrate it once; after verification it may be deleted.

### Label Studio

Start with:

```powershell
.\scripts\start-label-studio.ps1
```

Source Storage:

```text
path: <repo>\dataset\input
method: Tasks
filter: ^batch-tasks-.*\.json$
scan subfolders: off
```

Target Storage:

```text
path: <repo>\dataset\output\annotations
delete objects: off
```

User has confirmed Source Storage works and images load correctly.

### Exporter

`skydom-export-label-studio` creates incremental immutable batches. Existing sample IDs are discovered from previous batches, so only unseen records enter a new batch.

### Importer

Latest implementation adds:

```powershell
skydom-import-label-studio
```

Real Local Files Target Storage output has now been observed. Files are extensionless numeric filenames (for example `3`, `4`, ...) containing one annotation object at the root. The sample id lives under `task.data.sample_id`, while label results live in the root `result` array. The importer supports this shape as well as the earlier conventional task-export shape (`data` + `annotations[]`).

It selects the latest annotation per sample and writes a complete 4-field `labels` object into the canonical record while preserving `suggested`.

The real end-to-end annotation loop is now confirmed working. Example verified record `018c99bad20be95d1ef1` received human labels while retaining the original bootstrap predictions.

## Key real-world observations

- Orange tiles have a red/orange hue gradient.
- Blue tile hue can overlap with the dark-blue board background.
- Chains can occlude/split base shapes and may share color with the base piece.
- Carrots are deliberate geometric outliers and should not become chains merely because peer-shape anomaly is high.
- Power-ups can contain large white/neutral decorations.
- Whole-cell adjacent-clear obstacles exist and should not inherit a pseudo-color from HSV appearance.
- Some levels have a smaller opponent board.
- Some levels have sparse/disconnected playable topology.

## Recommended next agent sequence

1. Continue collecting and labeling representative normal, carrot, chain, and power-up cells, especially blockers and rare power-ups.
2. Re-run `skydom-dataset-stats --strict` periodically and watch capture-provenance coverage.
3. Add a board/capture-aware train/validation split using `capture_ids` to reduce leakage.
4. Implement the first learned multi-head cell classifier once the dataset is large enough.
5. Compare the learned recognizer against `ClassicalTileRecognizer` on the same held-out set.

## Do not regress

- Do not hard-code board dimensions.
- Do not collapse semantic dimensions into combined classes.
- Do not treat baseline predictions as ground truth.
- Do not make the solver depend on OpenCV internals.
- Do not require Label Studio inside the bot runtime environment.
- Do not delete or rewrite old task batches during routine export.
