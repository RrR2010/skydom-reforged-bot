"""Offline evaluation of captured game-screen corpora."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any

import cv2

from skydom_bot.vision.board_detector import BoardDetectionError, BoardDetector


@dataclass(frozen=True, slots=True)
class CorpusAnalysisRecord:
    """Offline board-detection result for one captured game screen."""

    capture_id: str
    stage_id: str
    status: str
    reasons: tuple[str, ...]
    rows: int | None
    cols: int | None
    active_cells: int | None
    pitch_x: float | None
    pitch_y: float | None
    confidence: float | None
    ambiguous_components: int
    scaled_cells: int
    image: str
    metadata: dict[str, str]


def _read_rgb(path: Path):
    """Read one corpus screenshot as RGB."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read corpus image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _load_capture_metadata(path: Path) -> dict[str, Any]:
    """Load one capture metadata JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Capture metadata root must be an object: {path}")
    return payload


def _review_reasons(
    *,
    confidence: float,
    pitch_x: float,
    pitch_y: float,
    ambiguous_components: int,
    scaled_cells: int,
    confidence_threshold: float,
    pitch_delta_threshold: float,
) -> list[str]:
    """Return conservative reasons to place one capture in the review queue."""
    reasons: list[str] = []
    if confidence < confidence_threshold:
        reasons.append(f"confidence<{confidence_threshold:.2f}")
    if abs(pitch_x - pitch_y) > pitch_delta_threshold:
        reasons.append(f"pitch-delta>{pitch_delta_threshold:.1f}px")
    if ambiguous_components:
        reasons.append("ambiguous-scale-component")
    if scaled_cells:
        reasons.append("scaled-subgrid-detected")
    return reasons


def _export_cells(
    root: Path,
    capture_id: str,
    image_rgb,
    geometry,
) -> int:
    """Export detector cell crops as derived, unlabeled corpus artifacts."""
    target = root / "derived" / "cells" / capture_id
    target.mkdir(parents=True, exist_ok=True)

    count = 0
    for cell in geometry.cells:
        rect = cell.bounds
        crop = image_rgb[
            rect.y : rect.y + rect.height,
            rect.x : rect.x + rect.width,
        ]
        if crop.size == 0:
            continue
        path = target / f"r{cell.row:02d}_c{cell.col:02d}.png"
        ok = cv2.imwrite(str(path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"Could not write derived cell crop: {path}")
        count += 1
    return count


def analyze_corpus(
    root: Path = Path("corpus"),
    *,
    confidence_threshold: float = 0.90,
    pitch_delta_threshold: float = 3.0,
    export_cells: bool = False,
) -> tuple[CorpusAnalysisRecord, ...]:
    """Run the current BoardDetector against every captured corpus screen."""
    metadata_dir = root / "metadata"
    if not metadata_dir.exists():
        return ()

    detector = BoardDetector()
    results: list[CorpusAnalysisRecord] = []

    for metadata_path in sorted(metadata_dir.glob("*.json")):
        capture = _load_capture_metadata(metadata_path)
        capture_id = str(capture["capture_id"])
        stage_id = str(capture.get("stage_id", ""))
        image_rel = str(capture["image"])
        metadata = {
            str(key): str(value)
            for key, value in dict(capture.get("metadata") or {}).items()
        }
        image_path = root / image_rel

        try:
            image = _read_rgb(image_path)
            geometry, diagnostics = detector.detect_with_diagnostics(image)
            ambiguous = len(diagnostics.ambiguous_topology_components)
            scaled_cells = int(diagnostics.local_subgrid_topology.sum())
            reasons = _review_reasons(
                confidence=geometry.confidence,
                pitch_x=geometry.pitch_x,
                pitch_y=geometry.pitch_y,
                ambiguous_components=ambiguous,
                scaled_cells=scaled_cells,
                confidence_threshold=confidence_threshold,
                pitch_delta_threshold=pitch_delta_threshold,
            )
            status = "REVIEW" if reasons else "OK"

            if export_cells:
                _export_cells(root, capture_id, image, geometry)

            results.append(
                CorpusAnalysisRecord(
                    capture_id=capture_id,
                    stage_id=stage_id,
                    status=status,
                    reasons=tuple(reasons),
                    rows=geometry.rows,
                    cols=geometry.cols,
                    active_cells=len(geometry.cells),
                    pitch_x=geometry.pitch_x,
                    pitch_y=geometry.pitch_y,
                    confidence=geometry.confidence,
                    ambiguous_components=ambiguous,
                    scaled_cells=scaled_cells,
                    image=image_rel,
                    metadata=metadata,
                )
            )
        except (BoardDetectionError, FileNotFoundError, ValueError, RuntimeError) as exc:
            results.append(
                CorpusAnalysisRecord(
                    capture_id=capture_id,
                    stage_id=stage_id,
                    status="FAIL",
                    reasons=(str(exc),),
                    rows=None,
                    cols=None,
                    active_cells=None,
                    pitch_x=None,
                    pitch_y=None,
                    confidence=None,
                    ambiguous_components=0,
                    scaled_cells=0,
                    image=image_rel,
                    metadata=metadata,
                )
            )

    return tuple(results)


def write_analysis_outputs(
    root: Path,
    records: tuple[CorpusAnalysisRecord, ...],
) -> tuple[Path, Path, Path]:
    """Write JSONL, CSV, and Markdown corpus-analysis outputs."""
    derived = root / "derived"
    derived.mkdir(parents=True, exist_ok=True)

    jsonl_path = derived / "analysis.jsonl"
    csv_path = derived / "analysis.csv"
    markdown_path = derived / "analysis.md"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = asdict(record)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    fieldnames = [
        "capture_id",
        "stage_id",
        "status",
        "reasons",
        "rows",
        "cols",
        "active_cells",
        "pitch_x",
        "pitch_y",
        "confidence",
        "ambiguous_components",
        "scaled_cells",
        "image",
        "metadata",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            payload = asdict(record)
            payload["reasons"] = "; ".join(record.reasons)
            payload["metadata"] = json.dumps(record.metadata, ensure_ascii=False)
            writer.writerow(payload)

    ok = sum(record.status == "OK" for record in records)
    review = sum(record.status == "REVIEW" for record in records)
    failed = sum(record.status == "FAIL" for record in records)

    lines = [
        "# Screen corpus analysis",
        "",
        f"Total: **{len(records)}** | OK: **{ok}** | REVIEW: **{review}** | FAIL: **{failed}**",
        "",
        "## Review queue",
        "",
        "| Status | Stage | Board | Cells | Pitch | Confidence | Reasons | Capture |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    queue = [record for record in records if record.status != "OK"]
    if not queue:
        lines.append("| — | — | — | — | — | — | No review items | — |")
    else:
        for record in queue:
            board = (
                f"{record.rows}x{record.cols}"
                if record.rows is not None and record.cols is not None
                else "—"
            )
            pitch = (
                f"{record.pitch_x:.2f}x{record.pitch_y:.2f}"
                if record.pitch_x is not None and record.pitch_y is not None
                else "—"
            )
            confidence = (
                f"{record.confidence:.3f}"
                if record.confidence is not None
                else "—"
            )
            lines.append(
                f"| {record.status} | {record.stage_id} | {board} | "
                f"{record.active_cells if record.active_cells is not None else '—'} | "
                f"{pitch} | {confidence} | {'; '.join(record.reasons)} | "
                f"{record.capture_id} |"
            )

    lines.extend(
        [
            "",
            "## All captures",
            "",
            "| Stage | Board | Cells | Confidence | Status | Capture |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for record in records:
        board = (
            f"{record.rows}x{record.cols}"
            if record.rows is not None and record.cols is not None
            else "—"
        )
        confidence = (
            f"{record.confidence:.3f}"
            if record.confidence is not None
            else "—"
        )
        lines.append(
            f"| {record.stage_id} | {board} | "
            f"{record.active_cells if record.active_cells is not None else '—'} | "
            f"{confidence} | {record.status} | {record.capture_id} |"
        )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl_path, csv_path, markdown_path
