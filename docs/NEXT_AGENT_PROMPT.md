# Next Agent Transfer Prompt

Use the text below to start the next development session.

---

Continue development of repository `RrR2010/skydom-reforged-bot`.

Start by reading these files in order:

1. `docs/HANDOFF.md`
2. `docs/PROJECT_SPEC.md`
3. `docs/DECISIONS.md`
4. `docs/TODO.md`

Also inspect the active PR #1 on branch `feat/board-geometry-m1-m2` before making architectural changes.

Context:

- The project is an autonomous Match-3 bot for Skydom: Reforged.
- Board geometry is classical OpenCV and is already reasonably robust across dynamic board sizes, irregular/sparse topology, blockers, highlights, and opponent mini-boards.
- Tile semantics are moving away from handcrafted heuristics toward a learned cell-level recognizer.
- The stable recognition contract is `TileStateEstimate` with independent color/kind/blocker/powerup outputs and confidences.
- `ClassicalTileRecognizer` remains as an interpretable baseline/bootstrap model.
- Dataset collection, Label Studio source storage, target storage, export, and import are now working end-to-end on real data.
- Human labels are authoritative; bootstrap predictions must never be treated as ground truth.

Current dataset workflow:

```text
skydom-collect-tiles
  -> dataset/input/images + dataset/records
skydom-export-label-studio
  -> incremental batch-tasks-NNNN.json
Label Studio Source Storage Sync
  -> human annotation
Label Studio Target Storage
  -> dataset/output/annotations
skydom-import-label-studio
  -> dataset/records/<sample_id>.json labels
```

Label Studio Local Files Target Storage writes extensionless numeric files whose root object is the annotation; the source task is nested under `task`, and the sample id is `task.data.sample_id`. The importer already supports this format.

Current power-up taxonomy observed in gameplay:

- `none`
- `flyer` — flies to/eliminates a target of interest, often related to the level objective
- `row` — clears a row
- `column` — clears a column
- `bomb` — area-clearing bomb
- `color-remover` — removes pieces of the same color
- `unknown`

Immediate priorities:

1. Run/confirm the full test suite after the latest power-up taxonomy/config changes.
2. Add dataset statistics/validation tooling: labeled/unlabeled counts and class distributions for color/kind/blocker/powerup.
3. Continue collecting and annotating diverse examples rather than adding more one-off classical heuristics.
4. Add a train/validation split strategy that avoids leakage between crops from the same board/capture.
5. When enough labeled diversity exists, implement the first learned multi-head cell classifier and compare it against `ClassicalTileRecognizer` on the same held-out set.

Working preferences:

- Code/comments/docstrings in English.
- Explain important CV/ML/architecture concepts, but not line-by-line implementation details.
- Preserve visual debuggers as first-class development tools.
- The user is comfortable running PowerShell commands locally and testing changes.
- Implement autonomously where safe; ask for screenshots/files only when real game output is needed.
- Avoid a heuristic rabbit hole. Reassess whether a learned approach is more appropriate before adding appearance-specific thresholds.

Do not regress:

- no hard-coded board size;
- no solver dependency on OpenCV internals;
- no combined monolithic tile classes when compositional dimensions suffice;
- no bootstrap predictions silently promoted to labels;
- no Label Studio dependency inside the bot runtime environment;
- no rewriting/deleting prior task batches during normal incremental export.

Before starting new work, summarize the current state you inferred from the docs and state which TODO item you are taking next.

---
