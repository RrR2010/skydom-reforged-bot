"""Export the local crop dataset into Label Studio storage batches."""

from __future__ import annotations

import json
import re
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

_BATCH_RE = re.compile(r"^batch-tasks-(\d{4})\.json$")


@dataclass(frozen=True, slots=True)
class LabelStudioExportSummary:
    """Paths and counts produced by one incremental storage export."""

    config_path: Path
    input_dir: Path
    images_dir: Path
    output_dir: Path
    batch_path: Path | None
    new_samples: int
    existing_samples: int
    total_samples: int
    copied_images: int


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
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str):
        return None

    task: dict[str, Any] = {
        "data": {
            "image": f"/data/local-files/?d=input/images/{sample_id}.png",
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


def _existing_exported_ids(input_dir: Path) -> set[str]:
    exported: set[str] = set()
    for batch_path in sorted(input_dir.glob("batch-tasks-*.json")):
        if not _BATCH_RE.match(batch_path.name):
            continue
        try:
            payload = json.loads(batch_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, list):
            continue
        for task in payload:
            if not isinstance(task, dict):
                continue
            data = task.get("data")
            if not isinstance(data, dict):
                continue
            sample_id = data.get("sample_id")
            if isinstance(sample_id, str):
                exported.add(sample_id)
    return exported


def _next_batch_path(input_dir: Path) -> Path:
    numbers = [
        int(match.group(1))
        for path in input_dir.glob("batch-tasks-*.json")
        if (match := _BATCH_RE.match(path.name))
    ]
    next_number = max(numbers, default=0) + 1
    return input_dir / f"batch-tasks-{next_number:04d}.json"


def _ensure_input_image(
    dataset_root: Path,
    images_dir: Path,
    record: dict[str, Any],
) -> bool:
    """Ensure an input image exists, migrating the legacy layout if necessary."""
    sample_id = record.get("sample_id")
    if not isinstance(sample_id, str):
        return False

    destination = images_dir / f"{sample_id}.png"
    if destination.exists():
        return False

    # Compatibility for datasets collected before input/images became the
    # canonical crop location. This is a one-time migration path.
    legacy = dataset_root / "images" / f"{sample_id}.png"
    if not legacy.exists():
        raise FileNotFoundError(
            f"Missing crop image for sample {sample_id}: expected {destination}"
        )

    destination.write_bytes(legacy.read_bytes())
    return True


def export_label_studio_storage(
    dataset_root: Path,
    output_dir: Path | None = None,
) -> LabelStudioExportSummary:
    """Create a new immutable task batch containing only unseen samples."""
    dataset_root = dataset_root.resolve()
    input_dir = (output_dir or dataset_root / "input").resolve()
    images_dir = input_dir / "images"
    target_dir = dataset_root / "output" / "annotations"

    input_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    config_path = dataset_root / "label-studio-config.xml"
    config_path.write_text(LABEL_CONFIG, encoding="utf-8")

    exported_ids = _existing_exported_ids(input_dir)
    records_dir = dataset_root / "records"
    new_tasks: list[dict[str, Any]] = []
    copied_images = 0
    total_samples = 0

    for record_path in sorted(records_dir.glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        task = _task_from_record(record)
        if task is None:
            continue

        total_samples += 1
        if _ensure_input_image(dataset_root, images_dir, record):
            copied_images += 1

        expected_image = (Path("input") / "images" / f"{task['data']['sample_id']}.png").as_posix()
        if record.get("image") != expected_image:
            record["image"] = expected_image
            record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        sample_id = task["data"]["sample_id"]
        if sample_id not in exported_ids:
            new_tasks.append(task)

    batch_path: Path | None = None
    if new_tasks:
        batch_path = _next_batch_path(input_dir)
        batch_path.write_text(
            json.dumps(new_tasks, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return LabelStudioExportSummary(
        config_path=config_path,
        input_dir=input_dir,
        images_dir=images_dir,
        output_dir=target_dir,
        batch_path=batch_path,
        new_samples=len(new_tasks),
        existing_samples=total_samples - len(new_tasks),
        total_samples=total_samples,
        copied_images=copied_images,
    )


def export_label_studio(
    dataset_root: Path,
    output_dir: Path | None = None,
) -> tuple[Path, Path, int]:
    """Backward-compatible wrapper around the storage exporter."""
    summary = export_label_studio_storage(dataset_root, output_dir)
    return summary.config_path, summary.input_dir, summary.total_samples
