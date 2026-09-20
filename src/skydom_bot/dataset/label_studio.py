"""Export the local crop dataset into Label Studio storage task files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LABEL_CONFIG = """<View>
  <Style>
    .lsf-main-content { max-width: 900px; margin: 0 auto; }
  </Style>

  <Image name="image" value="$image" zoom="true"/>

  <Header value="Color"/>
  <Choices name="color" toName="image" choice="single" showInLine="true" required="true">
    <Choice value="red"/>
    <Choice value="orange"/>
    <Choice value="yellow"/>
    <Choice value="green"/>
    <Choice value="blue"/>
    <Choice value="purple"/>
    <Choice value="unknown"/>
  </Choices>

  <Header value="Kind"/>
  <Choices name="kind" toName="image" choice="single" showInLine="true" required="true">
    <Choice value="normal"/>
    <Choice value="carrot"/>
    <Choice value="unknown"/>
  </Choices>

  <Header value="Blocker"/>
  <Choices name="blocker" toName="image" choice="single" showInLine="true" required="true">
    <Choice value="none"/>
    <Choice value="chain"/>
    <Choice value="unknown"/>
  </Choices>

  <Header value="Power-up"/>
  <Choices name="powerup" toName="image" choice="single" showInLine="true" required="true">
    <Choice value="none"/>
    <Choice value="unknown"/>
  </Choices>
</View>
"""


@dataclass(frozen=True, slots=True)
class LabelStudioExportSummary:
    """Paths and counts produced by one storage export."""

    config_path: Path
    source_tasks_dir: Path
    target_dir: Path
    created: int
    existing: int
    total: int


def _prediction_result(name: str, value: str) -> dict[str, Any]:
    return {
        "from_name": name,
        "to_name": "image",
        "type": "choices",
        "value": {"choices": [value]},
    }


def _prediction(record: dict[str, Any]) -> list[dict[str, Any]]:
    suggested = record.get("suggested")
    if not isinstance(suggested, dict):
        return []

    result: list[dict[str, Any]] = []
    confidences: list[float] = []
    for name in ("color", "kind", "blocker", "powerup"):
        item = suggested.get(name)
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if not isinstance(value, str):
            continue
        result.append(_prediction_result(name, value))
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

    if not result:
        return []

    score = sum(confidences) / len(confidences) if confidences else 0.0
    return [{
        "model_version": "classical-bootstrap",
        "score": score,
        "result": result,
    }]


def _task_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    image = record.get("image")
    sample_id = record.get("sample_id")
    if not isinstance(image, str) or not isinstance(sample_id, str):
        return None

    task: dict[str, Any] = {
        "data": {
            # LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT is the repository root,
            # while the Local Files storage itself points to the dataset
            # subdirectory. Therefore media URLs are repository-root-relative.
            "image": f"/data/local-files/?d=dataset/{image}",
            "sample_id": sample_id,
        },
        "meta": {
            "row": record.get("row"),
            "col": record.get("col"),
            "source": record.get("source"),
        },
    }
    predictions = _prediction(record)
    if predictions:
        task["predictions"] = predictions
    return task


def export_label_studio_storage(
    dataset_root: Path,
    output_dir: Path | None = None,
) -> LabelStudioExportSummary:
    """Create immutable per-sample source tasks plus an empty target directory.

    One JSON file per sample works naturally with Label Studio Local Files
    source storage: newly collected crops produce newly named task files, so
    subsequent storage syncs can discover them without rewriting old batches.
    Existing task files are deliberately left untouched.
    """
    output_dir = output_dir or dataset_root / "label_studio"
    source_tasks_dir = output_dir / "source" / "tasks"
    target_dir = output_dir / "target" / "annotations"
    source_tasks_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.xml"
    config_path.write_text(LABEL_CONFIG, encoding="utf-8")

    records_dir = dataset_root / "records"
    created = 0
    existing = 0
    total = 0

    for record_path in sorted(records_dir.glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        task = _task_from_record(record)
        if task is None:
            continue

        sample_id = task["data"]["sample_id"]
        task_path = source_tasks_dir / f"{sample_id}.json"
        total += 1

        if task_path.exists():
            existing += 1
            continue

        task_path.write_text(
            json.dumps(task, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        created += 1

    return LabelStudioExportSummary(
        config_path=config_path,
        source_tasks_dir=source_tasks_dir,
        target_dir=target_dir,
        created=created,
        existing=existing,
        total=total,
    )


def export_label_studio(
    dataset_root: Path,
    output_dir: Path | None = None,
) -> tuple[Path, Path, int]:
    """Backward-compatible wrapper for callers of the original exporter.

    The second path now points to the source task directory rather than one
    monolithic tasks.json file.
    """
    summary = export_label_studio_storage(dataset_root, output_dir)
    return summary.config_path, summary.source_tasks_dir, summary.total
