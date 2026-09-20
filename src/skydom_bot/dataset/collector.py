"""Persist de-duplicated tile crops and metadata for future learned models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from skydom_bot.domain.board import BoardGeometry, Cell
from skydom_bot.domain.tile_state import TileStateEstimate

UInt8Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """One crop plus bootstrap predictions and optional human labels.

    Suggested values are explicitly not ground truth. They come from the
    current recognizer and exist only to accelerate later annotation.
    """

    sample_id: str
    image: str
    row: int
    col: int
    source: str
    suggested: dict[str, Any]
    labels: dict[str, str] | None


class DatasetCollector:
    """Write raw cell crops and JSON records without silently inventing labels."""

    def __init__(self, root: Path = Path("dataset")) -> None:
        self.root = root
        self.images_dir = root / "images"
        self.records_dir = root / "records"

    @staticmethod
    def _crop(image_rgb: UInt8Image, cell: Cell) -> UInt8Image:
        return image_rgb[
            cell.bounds.y : cell.bounds.bottom,
            cell.bounds.x : cell.bounds.right,
        ].copy()

    @staticmethod
    def _sample_id(crop_rgb: UInt8Image) -> str:
        """Use image bytes as a stable de-duplication key."""
        digest = sha256()
        digest.update(str(crop_rgb.shape).encode("ascii"))
        digest.update(crop_rgb.tobytes())
        return digest.hexdigest()[:20]

    @staticmethod
    def _suggested(estimate: TileStateEstimate) -> dict[str, Any]:
        return {
            "color": {
                "value": estimate.color.value,
                "confidence": estimate.color_confidence,
            },
            "kind": {
                "value": estimate.kind.value,
                "confidence": estimate.kind_confidence,
            },
            "blocker": {
                "value": estimate.blocker.value,
                "confidence": estimate.blocker_confidence,
            },
            "powerup": {
                "value": estimate.powerup.value,
                "confidence": estimate.powerup_confidence,
            },
        }

    def collect(
        self,
        image_rgb: UInt8Image,
        geometry: BoardGeometry,
        estimates: tuple[TileStateEstimate, ...],
        *,
        source: str,
        only_cells: set[tuple[int, int]] | None = None,
    ) -> tuple[DatasetRecord, ...]:
        """Save crops and records, returning records touched by this collection."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)

        estimate_map = {(item.row, item.col): item for item in estimates}
        records: list[DatasetRecord] = []

        for cell in geometry.cells:
            key = (cell.row, cell.col)
            if only_cells is not None and key not in only_cells:
                continue

            estimate = estimate_map.get(key)
            if estimate is None:
                continue

            crop = self._crop(image_rgb, cell)
            sample_id = self._sample_id(crop)
            image_rel = Path("images") / f"{sample_id}.png"
            image_path = self.root / image_rel
            record_path = self.records_dir / f"{sample_id}.json"

            if not image_path.exists():
                cv2.imwrite(
                    str(image_path),
                    cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                )

            existing_labels: dict[str, str] | None = None
            if record_path.exists():
                try:
                    existing = json.loads(record_path.read_text(encoding="utf-8"))
                    labels = existing.get("labels")
                    if isinstance(labels, dict):
                        existing_labels = {
                            str(name): str(value)
                            for name, value in labels.items()
                        }
                except (json.JSONDecodeError, OSError):
                    existing_labels = None

            record = DatasetRecord(
                sample_id=sample_id,
                image=image_rel.as_posix(),
                row=cell.row,
                col=cell.col,
                source=source,
                suggested=self._suggested(estimate),
                labels=existing_labels,
            )
            record_path.write_text(
                json.dumps(asdict(record), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            records.append(record)

        return tuple(records)
