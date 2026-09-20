"""Screen capture adapter kept separate from perception logic."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

UInt8Image = NDArray[np.uint8]


def capture_screen(monitor: int = 1) -> UInt8Image:
    """Capture one monitor and return an RGB uint8 image.

    ``mss`` is an optional runtime dependency so static-image development and
    CI do not require desktop capture support.
    """
    try:
        import mss
    except ImportError as exc:
        raise RuntimeError("Screen capture requires `pip install -e .[capture]`.") from exc

    with mss.mss() as sct:
        monitors: list[Any] = sct.monitors
        if monitor <= 0 or monitor >= len(monitors):
            raise ValueError(f"Invalid monitor {monitor}; available monitors: 1..{len(monitors) - 1}.")
        bgra = np.asarray(sct.grab(monitors[monitor]), dtype=np.uint8)
    return bgra[:, :, [2, 1, 0]].copy()
