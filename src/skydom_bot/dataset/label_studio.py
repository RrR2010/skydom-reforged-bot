"""Export the local crop dataset into Label Studio task format."""

from __future__ import annotations

import json
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


def export_label_studio(
    dataset_root: Path,
    output_dir: Path | None = None,
) -> tuple[Path, Path, int]:
    """Write Label Studio config and tasks from local dataset records."""
    output_dir = output_dir or dataset_root / "label_studio"
    output_dir.mkdir(parents=True, exist_ok=True)

    records_dir = dataset_root / "records"
    tasks: list[dict[str, Any]] = []

    for record_path in sorted(records_dir.glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        image = record.get("image")
        sample_id = record.get("sample_id")
        if not isinstance(image, str) or not isinstance(sample_id, str):
            continue

        task: dict[str, Any] = {
            "data": {
                "image": f"/data/local-files/?d={image}",
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
        tasks.append(task)

    config_path = output_dir / "config.xml"
    tasks_path = output_dir / "tasks.json"
    config_path.write_text(LABEL_CONFIG, encoding="utf-8")
    tasks_path.write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return config_path, tasks_path, len(tasks)
