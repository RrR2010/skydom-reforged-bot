# Project TODO / Roadmap

## Immediate validation

- [x] Pull latest branch and run full `pytest` after Label Studio importer changes (50 passed).
- [x] Submit several annotations in Label Studio and confirm Target Storage files appear in `dataset/output/annotations`.
- [x] Run `skydom-import-label-studio` against real Target Storage output.
- [x] Verify imported `labels` match the UI selections and `suggested` remains unchanged.
- [ ] Check behavior when an annotation is edited after first submission.
- [ ] Check whether Label Studio target files contain multiple annotations or only the latest annotation in this local-storage configuration.

## Dataset quality

- [ ] Annotate representative normal pieces of every color.
- [ ] Annotate carrots in multiple colors/orientations.
- [ ] Annotate chained pieces, especially same-color chain cases.
- [x] Identify initial real power-up types: flyer, row, column, bomb, color-remover.
- [x] Add observed `TilePowerup` enum values and synchronize Label Studio config.
- [ ] Capture animation/highlight/hint states and decide whether they are excluded, separately labeled, or temporally filtered.
- [x] Add dataset statistics command: counts by color/kind/blocker/powerup and unlabeled count.
- [x] Add board-capture provenance to newly collected records for leakage-safe grouping.
- [ ] Add train/validation split keyed by capture/board rather than random crop only, to reduce leakage.

## Learned recognition

- [ ] Establish first image-classification baseline after enough labels exist.
- [ ] Prefer multi-head classification over combined classes.
- [ ] Compare lightweight backbones (e.g. MobileNet/EfficientNet/ResNet-class models).
- [ ] Evaluate pure learned model before classical-feature fusion.
- [ ] Report per-head accuracy/confusion matrices.
- [ ] Compare learned model against `ClassicalTileRecognizer` on the same held-out set.
- [ ] Add `LearnedTileRecognizer` implementing the existing `TileRecognizer` protocol.
- [ ] Consider `HybridTileRecognizer` only if evidence shows benefit.

## Board state / solver

- [ ] Add `BoardState` domain object combining geometry and semantic tile estimates.
- [ ] Generate legal adjacent swaps only across active topology.
- [ ] Simulate color matches.
- [ ] Add objective/blocker semantics after base solver is correct.
- [ ] Add move scoring.

## Runtime automation

- [ ] Detect stable playing state vs cascade/animation.
- [ ] Detect won/lost/level transitions.
- [ ] Add temporal stabilization across frames.
- [ ] Add safe mouse actuator.
- [ ] Add click verification / recovery when UI state changes.

## Maintenance

- [ ] Decide when PR #1 is ready to split/merge; it now spans geometry, semantic baseline, debugger, dataset, and Label Studio integration.
- [ ] Keep visual debuggers as first-class diagnostics.
- [ ] Avoid further one-off heuristic tuning unless needed to bootstrap data collection.
