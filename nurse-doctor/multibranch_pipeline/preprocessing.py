import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .data import PatientRecord


NEURO_COLS = [
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale total",
    "Glascow coma scale verbal response",
]

CARDIO_COLS = [
    "Heart Rate",
    "Diastolic blood pressure",
    "Mean blood pressure",
    "Systolic blood pressure",
]

RESP_COLS = [
    "Oxygen saturation",
    "Respiratory rate",
    "Fraction inspired oxygen",
]

META_VALUE_COLS = [
    "Glucose",
    "pH",
    "Temperature",
]

STATIC_COLS = ["Height", "Weight"]


CARDIO_RANGES = {
    "Heart Rate": (20.0, 260.0),
    "Diastolic blood pressure": (20.0, 200.0),
    "Mean blood pressure": (30.0, 220.0),
    "Systolic blood pressure": (40.0, 300.0),
}

RESP_RANGES = {
    "Oxygen saturation": (40.0, 100.0),
    "Respiratory rate": (2.0, 90.0),
    "Fraction inspired oxygen": (0.21, 1.0),
}


def _extract_leading_number(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    text = str(v).strip()
    if not text:
        return np.nan
    m = re.match(r"^(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1))
    return np.nan


@dataclass
class PreprocessorState:
    static_median: Dict[str, float]
    static_mean: Dict[str, float]
    static_std: Dict[str, float]
    neuro_fill: Dict[str, float]
    cardio_fill: Dict[str, float]
    resp_fill: Dict[str, float]
    meta_fill: Dict[str, float]


class GroupPreprocessor:
    def __init__(self, max_hours: float = 48.0, timestep: float = 1.0):
        self.max_hours = max_hours
        self.timestep = timestep
        self.time_steps = int(max_hours / timestep) + 1
        self.state: PreprocessorState | None = None

    def fit(self, records: List[PatientRecord]) -> None:
        if not records:
            raise ValueError("Cannot fit preprocessor on empty records")

        static_rows = []
        neuro_rows = []
        cardio_rows = []
        resp_rows = []
        meta_rows = []

        for rec in records:
            f = rec.frame
            static_rows.append(self._extract_static_raw(f))
            neuro_rows.append(self._extract_neuro_raw(f))
            cardio_rows.append(self._extract_numeric_raw(f, CARDIO_COLS))
            resp_rows.append(self._extract_resp_raw(f))
            meta_rows.append(self._extract_numeric_raw(f, META_VALUE_COLS))

        static_df = pd.DataFrame(static_rows)
        neuro_df = pd.concat(neuro_rows, axis=0, ignore_index=True)
        cardio_df = pd.concat(cardio_rows, axis=0, ignore_index=True)
        resp_df = pd.concat(resp_rows, axis=0, ignore_index=True)
        meta_df = pd.concat(meta_rows, axis=0, ignore_index=True)

        static_median = {c: float(static_df[c].median()) for c in STATIC_COLS}
        static_mean = {c: float(static_df[c].fillna(static_median[c]).mean()) for c in STATIC_COLS}
        static_std = {
            c: float(static_df[c].fillna(static_median[c]).std(ddof=0)) if float(static_df[c].fillna(static_median[c]).std(ddof=0)) > 1e-6 else 1.0
            for c in STATIC_COLS
        }

        self.state = PreprocessorState(
            static_median=static_median,
            static_mean=static_mean,
            static_std=static_std,
            neuro_fill={c: float(neuro_df[c].median()) for c in NEURO_COLS},
            cardio_fill={c: float(cardio_df[c].median()) for c in CARDIO_COLS},
            resp_fill={c: float(resp_df[c].median()) for c in RESP_COLS},
            meta_fill={c: float(meta_df[c].median()) for c in META_VALUE_COLS},
        )

    def save_state(self, output_path: str) -> None:
        if self.state is None:
            raise ValueError("Preprocessor is not fitted")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.state.__dict__, f, indent=2)

    def load_state(self, state_path: str) -> None:
        with open(state_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.state = PreprocessorState(**raw)

    def transform(self, rec: PatientRecord):
        if self.state is None:
            raise ValueError("Preprocessor is not fitted")

        f = rec.frame.copy()

        neuro = self._transform_neuro(f)
        cardio = self._transform_cardio(f)
        resp = self._transform_resp(f)
        metabolic = self._transform_metabolic(f)
        static = self._transform_static(f)

        return {
            "name": rec.name,
            "y": float(rec.y),
            "neuro": neuro.astype(np.float32),
            "cardio": cardio.astype(np.float32),
            "resp": resp.astype(np.float32),
            "meta": metabolic.astype(np.float32),
            "static": static.astype(np.float32),
        }

    def _extract_numeric_raw(self, frame: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for c in cols:
            out[c] = pd.to_numeric(frame.get(c), errors="coerce")
        return out

    def _extract_static_raw(self, frame: pd.DataFrame) -> Dict[str, float]:
        out = {}
        for c in STATIC_COLS:
            vals = pd.to_numeric(frame.get(c), errors="coerce").dropna()
            out[c] = float(vals.median()) if not vals.empty else np.nan
        return out

    def _extract_neuro_raw(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for c in NEURO_COLS:
            if c in frame.columns:
                if "total" in c.lower():
                    out[c] = pd.to_numeric(frame[c], errors="coerce")
                else:
                    out[c] = frame[c].map(_extract_leading_number)
            else:
                out[c] = np.nan
        return out

    def _extract_resp_raw(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for c in RESP_COLS:
            out[c] = pd.to_numeric(frame.get(c), errors="coerce")
        out = self._normalize_resp_units(out)
        return out

    def _transform_neuro(self, frame: pd.DataFrame) -> np.ndarray:
        df = self._extract_neuro_raw(frame)
        df = df.ffill()
        for c in NEURO_COLS:
            df[c] = df[c].fillna(self.state.neuro_fill[c])
        return df[NEURO_COLS].to_numpy()

    def _transform_cardio(self, frame: pd.DataFrame) -> np.ndarray:
        df = self._extract_numeric_raw(frame, CARDIO_COLS)
        for c, (lo, hi) in CARDIO_RANGES.items():
            df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
        df = df.ffill()
        for c in CARDIO_COLS:
            df[c] = df[c].fillna(self.state.cardio_fill[c])
        return df[CARDIO_COLS].to_numpy()

    def _normalize_resp_units(self, df: pd.DataFrame) -> pd.DataFrame:
        fio2 = df["Fraction inspired oxygen"]
        fio2 = fio2.where(fio2 <= 1.5, fio2 / 100.0)

        spo2 = df["Oxygen saturation"]
        spo2 = spo2.where(spo2 > 1.0, spo2 * 100.0)

        df["Fraction inspired oxygen"] = fio2
        df["Oxygen saturation"] = spo2
        return df

    def _transform_resp(self, frame: pd.DataFrame) -> np.ndarray:
        df = self._extract_resp_raw(frame)
        for c, (lo, hi) in RESP_RANGES.items():
            df.loc[(df[c] < lo) | (df[c] > hi), c] = np.nan
        df = df.ffill()
        for c in RESP_COLS:
            df[c] = df[c].fillna(self.state.resp_fill[c])
        df = df.rolling(window=3, min_periods=1).mean()
        return df[RESP_COLS].to_numpy()

    def _transform_metabolic(self, frame: pd.DataFrame) -> np.ndarray:
        vals = self._extract_numeric_raw(frame, META_VALUE_COLS)
        masks = vals.notna().astype(np.float32)

        deltas = pd.DataFrame(index=vals.index, columns=META_VALUE_COLS, dtype=float)
        for c in META_VALUE_COLS:
            delta = np.zeros(len(vals), dtype=np.float32)
            since_seen = 0.0
            for i, m in enumerate(masks[c].to_numpy()):
                if m > 0:
                    since_seen = 0.0
                else:
                    since_seen += float(self.timestep)
                delta[i] = since_seen
            deltas[c] = delta

        vals = vals.ffill()
        for c in META_VALUE_COLS:
            vals[c] = vals[c].fillna(self.state.meta_fill[c])

        # GRU-D-like input representation: value + mask + delta per channel.
        stacked = np.concatenate(
            [vals[META_VALUE_COLS].to_numpy(), masks[META_VALUE_COLS].to_numpy(), deltas[META_VALUE_COLS].to_numpy()],
            axis=1,
        )
        return stacked

    def _transform_static(self, frame: pd.DataFrame) -> np.ndarray:
        vals = self._extract_static_raw(frame)
        arr = np.array([vals[c] for c in STATIC_COLS], dtype=np.float32)
        for i, c in enumerate(STATIC_COLS):
            if np.isnan(arr[i]):
                arr[i] = self.state.static_median[c]
            arr[i] = (arr[i] - self.state.static_mean[c]) / self.state.static_std[c]
        return arr
