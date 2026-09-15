from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CloudWindow:
    cloud_id: str
    cloud_date: date
    stock_code: str
    source_path: Path
    frame: pd.DataFrame


@dataclass(frozen=True)
class CloudRecord:
    cloud_id: str
    cloud_date: date
    stock_code: str
    source_path: Path
    point_count: int


Diagram = dict[int, np.ndarray]

