"""Manual screen-corpus capture with persistent game-region configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import cv2
import numpy as np
from numpy.typing import NDArray

from skydom_bot.capture import capture_screen

UInt8Image = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class CaptureRegion:
    """Game-canvas rectangle in monitor-local pixels."""

    x: int
    y: int
    width: int
    height: int

    def validate(self, image: UInt8Image) -> None:
        """Validate this region against a captured monitor image."""
        height, width = image.shape[:2]
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Capture region width and height must be positive.")
        if self.x < 0 or self.y < 0:
            raise ValueError("Capture region x and y must be non-negative.")
        if self.x + self.width > width or self.y + self.height > height:
            raise ValueError(
                f"Capture region {self} exceeds monitor image {width}x{height}."
            )


@dataclass(frozen=True, slots=True)
class ScreenCaptureRecord:
    """Metadata for one manually triggered corpus screenshot."""

    capture_id: str
    stage_id: str
    captured_at: str
    monitor: int
    region: CaptureRegion
    image: str
    sha256: str
    metadata: dict[str, str]


_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _slug(value: str) -> str:
    """Return a filesystem-safe identifier while preserving useful stage text."""
    slug = _SLUG_RE.sub("-", value.strip()).strip("-._")
    return slug or "stage"


def parse_region(value: str) -> CaptureRegion:
    """Parse X,Y,W,H into a capture region."""
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError(
            "Region must be X,Y,W,H, for example 190,163,1016,534."
        ) from exc
    if len(parts) != 4:
        raise ValueError(
            "Region must be X,Y,W,H, for example 190,163,1016,534."
        )
    return CaptureRegion(*parts)


def parse_metadata(values: list[str]) -> dict[str, str]:
    """Parse repeated KEY=VALUE metadata arguments."""
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Metadata must be KEY=VALUE, got {value!r}.")
        key, item = value.split("=", maxsplit=1)
        key = key.strip()
        if not key:
            raise ValueError(f"Metadata key cannot be empty in {value!r}.")
        result[key] = item.strip()
    return result


def region_config_path(root: Path) -> Path:
    """Return the local region-configuration path for one corpus."""
    return root / "region.json"


def save_region(root: Path, region: CaptureRegion, *, monitor: int) -> Path:
    """Persist the reusable game-canvas region."""
    root.mkdir(parents=True, exist_ok=True)
    path = region_config_path(root)
    path.write_text(
        json.dumps(
            {
                "monitor": monitor,
                "region": asdict(region),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def load_region(root: Path) -> tuple[int, CaptureRegion]:
    """Load the reusable game-canvas region."""
    path = region_config_path(root)
    if not path.is_file():
        raise FileNotFoundError(
            f"No corpus region configured at {path}. "
            "Run skydom-capture-screen --configure first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    region = CaptureRegion(**payload["region"])
    return int(payload["monitor"]), region


def select_region_interactively(image_rgb: UInt8Image) -> CaptureRegion:
    """Let the user drag the useful game region once using OpenCV ROI selection."""
    title = "Select Skydom game area and press ENTER"
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    x, y, width, height = cv2.selectROI(
        title,
        bgr,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow(title)
    region = CaptureRegion(int(x), int(y), int(width), int(height))
    region.validate(image_rgb)
    return region


def crop_region(image_rgb: UInt8Image, region: CaptureRegion) -> UInt8Image:
    """Crop a validated monitor image to the configured game region."""
    region.validate(image_rgb)
    return image_rgb[
        region.y : region.y + region.height,
        region.x : region.x + region.width,
    ].copy()


def save_capture(
    root: Path,
    *,
    stage_id: str,
    monitor: int,
    region: CaptureRegion,
    metadata: Mapping[str, str] | None = None,
    image_rgb: UInt8Image | None = None,
) -> ScreenCaptureRecord:
    """Capture one game-region screenshot and persist image plus metadata."""
    source = capture_screen(monitor) if image_rgb is None else image_rgb
    crop = crop_region(source, region)

    captured_at = datetime.now(timezone.utc)
    timestamp = captured_at.strftime("%Y%m%dT%H%M%S.%fZ")
    stage_slug = _slug(stage_id)
    capture_id = f"{stage_slug}__{timestamp}"

    image_dir = root / "screens"
    metadata_dir = root / "metadata"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    image_rel = Path("screens") / f"{capture_id}.png"
    image_path = root / image_rel
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Could not encode corpus screenshot as PNG.")
    image_bytes = encoded.tobytes()
    image_path.write_bytes(image_bytes)

    record = ScreenCaptureRecord(
        capture_id=capture_id,
        stage_id=stage_id,
        captured_at=captured_at.isoformat(),
        monitor=monitor,
        region=region,
        image=image_rel.as_posix(),
        sha256=hashlib.sha256(image_bytes).hexdigest(),
        metadata=dict(metadata or {}),
    )
    (metadata_dir / f"{capture_id}.json").write_text(
        json.dumps(
            {
                **asdict(record),
                "region": asdict(region),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return record
