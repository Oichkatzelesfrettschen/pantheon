"""Pantheon+SH0ES data interface (compatibility layer).

The canonical implementation lives in `spandrel_core.pantheon`. Pantheon keeps a
thin wrapper here so existing imports (`spandrel.core.data_interface`) remain
stable while the core logic is promoted into `spandrel-core`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, cast

import numpy as np
from spandrel_core.pantheon import DataStats
from spandrel_core.pantheon import PantheonData as _PantheonData
from spandrel_core.pantheon import load_pantheon as _load

DATA_FILE = Path(__file__).parents[3] / "data" / "Pantheon+SH0ES.dat"

__all__ = ["DATA_FILE", "DataStats", "PantheonData", "load_pantheon"]


class PantheonData(_PantheonData):
    def __init__(
        self,
        filepath: Optional[Path] = None,
        z_min: float = 0.001,
        z_max: float = 2.5,
    ) -> None:
        super().__init__(filepath=Path(filepath) if filepath else DATA_FILE, z_min=z_min, z_max=z_max)


def load_pantheon(
    filepath: Optional[Path] = None,
    *,
    z_min: float = 0.001,
    z_max: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = _load(Path(filepath) if filepath else DATA_FILE, z_min=z_min, z_max=z_max)
    return cast(tuple[np.ndarray, np.ndarray, np.ndarray], out)
