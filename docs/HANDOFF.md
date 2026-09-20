# Handoff — Skydom Reforged Bot

## Repository / branch

- Repository: `RrR2010/skydom-reforged-bot`
- Active branch: `feat/board-geometry-m1-m2`
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

### Debugger

`skydom-debug-tiles` supports multi-cell comparison and exporting selected cells with `E`.

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

It expects Label Studio Target Storage JSON files containing:

- `data.sample_id`;
- `annotations[].result` with `choices` controls named color/kind/blocker/powerup.

It selects the latest annotation and writes a complete 4-field `labels` object into the canonical record while preserving `suggested`.

**Important:** importer has synthetic tests but has not yet been validated against the user's real Target Storage JSON files. This is the immediate next validation gate.

## Key real-world observations

- Orange tiles have a red/orange hue gradient.
- Blue tile hue can overlap with the dark-blue board background.
- Chains can occlude/split base shapes and may share color with the base piece.
- Carrots are deliberate geometric outliers and should not become chains merely because peer-shape anomaly is high.
- Power-ups can contain large white/neutral decorations.
- Some levels have a smaller opponent board.
- Some levels have sparse/disconnected playable topology.

## Recommended next agent sequence

1. Ask the user to pull and run tests after the importer commit.
2. Have the user submit 2–5 real Label Studio annotations.
3. Inspect one generated file under `dataset/output/annotations` if import fails.
4. Run `skydom-import-label-studio` and verify the matching records now contain human `labels`.
5. Add dataset statistics/validation tooling.
6. Continue collecting and labeling rather than tuning classical semantics.
7. Once dataset diversity is sufficient, implement the first learned multi-head cell classifier.

## Do not regress

- Do not hard-code board dimensions.
- Do not collapse semantic dimensions into combined classes.
- Do not treat baseline predictions as ground truth.
- Do not make the solver depend on OpenCV internals.
- Do not require Label Studio inside the bot runtime environment.
- Do not delete or rewrite old task batches during routine export.
