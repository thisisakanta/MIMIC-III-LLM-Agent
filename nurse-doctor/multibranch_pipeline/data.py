import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class PatientRecord:
    name: str
    y: float
    frame: pd.DataFrame


class InHospitalMortalityReaderCompat:
    """Reader compatible with old listfile/timeseries format."""

    def __init__(self, dataset_dir: str, listfile: str):
        self.dataset_dir = dataset_dir
        self.listfile = listfile
        self._meta = pd.read_csv(listfile)
        if "stay" not in self._meta.columns or "y_true" not in self._meta.columns:
            raise ValueError(f"Listfile {listfile} must contain stay,y_true columns")

    def __len__(self) -> int:
        return len(self._meta)

    def load_example(self, idx: int) -> PatientRecord:
        row = self._meta.iloc[idx]
        stay = row["stay"]
        y = float(row["y_true"])
        path = os.path.join(self.dataset_dir, stay)
        frame = pd.read_csv(path)
        return PatientRecord(name=stay, y=y, frame=frame)

    def load_all(self, max_patients: int = 0) -> List[PatientRecord]:
        n = len(self)
        if max_patients and max_patients > 0:
            n = min(n, max_patients)
        return [self.load_example(i) for i in range(n)]


def _empty_grid_frame(columns: List[str], time_steps: int) -> pd.DataFrame:
    # Mixed ICU columns include both numeric and categorical values, so keep an
    # object grid and let downstream preprocessing cast per-group as needed.
    out = pd.DataFrame(index=np.arange(time_steps), columns=columns, dtype=object)
    out.index.name = "bin"
    return out


def to_hourly_grid(frame: pd.DataFrame, max_hours: float = 48.0, timestep: float = 1.0) -> pd.DataFrame:
    """Convert raw event table into fixed bins [0, max_hours]."""
    if "Hours" not in frame.columns:
        raise ValueError("Input frame must contain Hours column")

    time_steps = int(max_hours / timestep) + 1
    cols = [c for c in frame.columns if c != "Hours"]
    result = _empty_grid_frame(cols, time_steps)

    work = frame.copy()
    work["Hours"] = pd.to_numeric(work["Hours"], errors="coerce")
    work = work.dropna(subset=["Hours"])
    work = work[(work["Hours"] >= 0.0) & (work["Hours"] <= max_hours)]
    work["bin"] = np.floor(work["Hours"] / timestep).astype(int)

    if work.empty:
        return result

    grouped = work.groupby("bin", sort=True)
    for b, g in grouped:
        if b < 0 or b >= time_steps:
            continue
        for c in cols:
            vals = g[c].dropna().astype(str)
            vals = vals[vals != ""]
            if vals.empty:
                continue
            numeric = pd.to_numeric(vals, errors="coerce")
            if numeric.notna().any():
                result.at[b, c] = float(numeric.dropna().mean())
            else:
                result.at[b, c] = vals.iloc[-1]

    return result


def load_split_records(data_root: str, split: str, max_patients: int = 0) -> List[PatientRecord]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of train/val/test")

    dataset_dir = os.path.join(data_root, "train" if split in {"train", "val"} else "test")
    listfile = os.path.join(data_root, f"{split}_listfile.csv")
    reader = InHospitalMortalityReaderCompat(dataset_dir=dataset_dir, listfile=listfile)
    return reader.load_all(max_patients=max_patients)


def build_grid_records(
    data_root: str,
    split: str,
    max_hours: float = 48.0,
    timestep: float = 1.0,
    max_patients: int = 0,
) -> List[PatientRecord]:
    records = load_split_records(data_root=data_root, split=split, max_patients=max_patients)
    out = []
    for rec in records:
        out.append(PatientRecord(name=rec.name, y=rec.y, frame=to_hourly_grid(rec.frame, max_hours=max_hours, timestep=timestep)))
    return out


def summarize_split(records: List[PatientRecord]) -> Dict[str, float]:
    ys = np.array([r.y for r in records], dtype=np.float32)
    return {
        "n": float(len(records)),
        "positive": float(ys.sum()),
        "positive_rate": float(ys.mean()) if len(ys) else 0.0,
    }
