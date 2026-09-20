# Architecture Decision Log

## ADR-001 — Keep board geometry classical

**Decision:** retain OpenCV-based board detection for bounds, pitch, rows/columns, and topology.

**Reason:** geometry is already robust across irregular, sparse, highlighted, blocker-covered, and competitive boards. Replacing it with object detection would duplicate information that the grid detector already extracts well.

## ADR-002 — Learned recognition is cell-level

**Decision:** learned recognition receives logical cell crops rather than detecting tiles on the whole screenshot.

**Reason:** board geometry already supplies cell locations. The remaining problem is primarily semantic classification, not object localization.

## ADR-003 — Semantic state is compositional

**Decision:** represent a cell through independent dimensions: color, kind, blocker, powerup.

**Reason:** avoids class explosion such as `BLUE_CARROT_CHAINED_POWERUP_X` and permits future extensions independently.

## ADR-004 — Classical recognition becomes baseline, not destination

**Decision:** stop adding appearance-specific heuristic thresholds except for clear infrastructure bugs.

**Reason:** new pieces, power-ups, same-color blockers, highlights, and animation states make handcrafted semantic rules increasingly brittle.

## ADR-005 — Human ground truth stays separate from bootstrap predictions

**Decision:** dataset records contain `suggested` and `labels` separately.

**Reason:** training on heuristic outputs as if they were truth would reproduce baseline errors.

## ADR-006 — Label Studio instead of custom annotation UI

**Decision:** use Label Studio Community for annotation.

**Reason:** classification UI, storage integration, annotation persistence, and review workflows are already solved by a maintained open-source tool.

## ADR-007 — Storage-first Label Studio workflow

**Decision:** use `dataset/input` as Source Storage and `dataset/output/annotations` as Target Storage.

**Reason:** avoids repeated manual JSON import/export and scales incrementally as new boards are played.

## ADR-008 — Immutable incremental task batches

**Decision:** exporter creates `batch-tasks-NNNN.json` containing only unseen samples and never rewrites prior batches.

**Reason:** storage synchronization becomes append-only and predictable.

## ADR-009 — Canonical crop location is dataset/input/images

**Decision:** collector writes new crops directly to `dataset/input/images`.

**Reason:** removes the duplicate `dataset/images` -> `dataset/input/images` copy path. A temporary compatibility migration remains for legacy data only.

## ADR-010 — Separate Label Studio Python environment

**Decision:** `.venv-labelstudio` uses Python 3.12 and remains separate from the bot `.venv`.

**Reason:** Label Studio has a large dependency tree and was incompatible with the user's Python 3.14 environment because of `pkgutil.find_loader` removal through an upstream dependency.

## ADR-011 — Power-up labels use observed gameplay semantics

**Decision:** the current power-up taxonomy is `none`, `flyer`, `row`, `column`, `bomb`, `color-remover`, and `unknown`.

**Reason:** these categories have now been observed and annotated in real gameplay. The learned recognizer should predict these as one semantic head rather than creating combined tile classes.


## ADR-012 — Preserve board-capture provenance for dataset splitting

**Decision:** newly collected sample records store one or more `capture_ids`, derived from the detected board crop and logical topology.

**Reason:** random crop-level splitting can leak near-identical visual context from one board capture into both training and validation. Samples may also be de-duplicated across multiple captures, so provenance is accumulated rather than overwritten. Legacy records without capture provenance remain valid training data but are not considered safe held-out validation data unless provenance is recovered.
