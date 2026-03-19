# train_mortality_lr_knn_imputer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline


# -------------------------
# Paths / Config
# -------------------------
DATA_DIR = Path("data") / "in-hospital-mortality"
TRAIN_LISTFILE = DATA_DIR / "train_listfile.csv"
VAL_LISTFILE   = DATA_DIR / "val_listfile.csv"
TEST_LISTFILE  = DATA_DIR / "test_listfile.csv"

FEATURE_COLS_JSON = DATA_DIR / "feature_cols.json"

CACHE_DIR = DATA_DIR / "cache_lr_knn_48h"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HOURS_LIMIT = 48.0

# Aggregate stats to compute per variable
AGG_FUNCS = ["mean", "std", "min", "max", "last"]


# -------------------------
# Helpers
# -------------------------
def load_feature_cols() -> List[str]:
    """
    feature_cols.json should contain a list of time-series variable names,
    e.g. ["Heart Rate", "Glucose", ...] or whatever you previously used.
    """
    if not FEATURE_COLS_JSON.exists():
        raise FileNotFoundError(f"Missing {FEATURE_COLS_JSON}")
    with open(FEATURE_COLS_JSON, "r") as f:
        cols = json.load(f)

    if not isinstance(cols, list) or not all(isinstance(x, str) for x in cols):
        raise ValueError("feature_cols.json must be a JSON list of strings (column names).")
    return cols


def read_listfile(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)

    # Typical IHM listfile: stay,y_true
    if "stay" not in df.columns:
        # sometimes first column is stay even if not named
        df = df.rename(columns={df.columns[0]: "stay"})
    if "y_true" not in df.columns:
        # sometimes label column is named "label"
        if "label" in df.columns:
            df = df.rename(columns={"label": "y_true"})
        else:
            raise ValueError(f"{path} must contain y_true (label) column.")
    return df[["stay", "y_true"]].copy()


def _safe_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_timeseries_48h(stay_filename: str, feature_cols: List[str]) -> pd.DataFrame:
    """
    Loads a single stay timeseries CSV from:
      data/in-hospital-mortality/train/<stay>
      data/in-hospital-mortality/val/<stay>
      data/in-hospital-mortality/test/<stay>

    Note: listfile stay path already includes split subfolder in many repos.
    In your setup it usually looks like: "56825_episode1_timeseries.csv"
    and the actual file lives inside split folder.
    We'll search train/val/test subfolders.
    """
    stay_filename = str(stay_filename)

    candidates = [
        DATA_DIR / "train" / stay_filename,
        DATA_DIR / "val" / stay_filename,
        DATA_DIR / "test" / stay_filename,
        DATA_DIR / stay_filename,  # fallback if files are directly under task dir
    ]
    fpath = next((p for p in candidates if p.exists()), None)
    if fpath is None:
        raise FileNotFoundError(f"Could not locate timeseries file for stay={stay_filename}")

    ts = pd.read_csv(fpath)

    # Most benchmark exports use "Hours"
    if "Hours" not in ts.columns:
        # sometimes it's "hours" or first column
        if "hours" in ts.columns:
            ts = ts.rename(columns={"hours": "Hours"})
        else:
            ts = ts.rename(columns={ts.columns[0]: "Hours"})

    # Limit to first 48 hours
    ts["Hours"] = pd.to_numeric(ts["Hours"], errors="coerce")
    ts = ts[ts["Hours"].notna() & (ts["Hours"] <= HOURS_LIMIT)].copy()

    # Keep only the configured feature columns that actually exist in the file
    keep_vars = [c for c in feature_cols if c in ts.columns]
    if not keep_vars:
        # Return empty; caller will produce all-NaN feature vector
        return ts[["Hours"]].copy()

    ts = _safe_to_numeric(ts, keep_vars)
    return ts[["Hours"] + keep_vars]


def featurize_stay_agg(ts: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    """
    Aggregate per variable across time within 0..48h:
      mean/std/min/max/last
    """
    feats: Dict[str, float] = {}

    present_vars = [c for c in feature_cols if c in ts.columns]
    for v in feature_cols:
        s = ts[v] if v in ts.columns else pd.Series([], dtype=float)

        # If variable missing from file or all NaN -> NaN features
        if len(s) == 0 or s.notna().sum() == 0:
            for fn in AGG_FUNCS:
                feats[f"{v}__{fn}"] = np.nan
            continue

        # Compute aggregations
        feats[f"{v}__mean"] = float(np.nanmean(s.values))
        feats[f"{v}__std"]  = float(np.nanstd(s.values, ddof=1)) if s.notna().sum() > 1 else 0.0
        feats[f"{v}__min"]  = float(np.nanmin(s.values))
        feats[f"{v}__max"]  = float(np.nanmax(s.values))

        # "last" = last observed non-null value in time
        s_last = s.dropna()
        feats[f"{v}__last"] = float(s_last.iloc[-1]) if len(s_last) else np.nan

    return feats


def build_dataset(listfile: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    X_rows = []
    y = listfile["y_true"].astype(int).values

    for stay in listfile["stay"].astype(str).tolist():
        ts = load_timeseries_48h(stay, feature_cols)
        feats = featurize_stay_agg(ts, feature_cols)
        feats["stay"] = stay
        X_rows.append(feats)

    X = pd.DataFrame(X_rows).set_index("stay")
    return X, y


def evaluate(model, X: np.ndarray, y: np.ndarray, name: str) -> None:
    prob = model.predict_proba(X)[:, 1]
    roc = roc_auc_score(y, prob)
    pr  = average_precision_score(y, prob)
    print(f"{name:>8} | ROC-AUC: {roc:.3f} | PR-AUC: {pr:.3f}")


# -------------------------
# Main
# -------------------------
def main():
    feature_cols = load_feature_cols()
    print(f"Loaded {len(feature_cols)} feature columns from {FEATURE_COLS_JSON}")

    train_lf = read_listfile(TRAIN_LISTFILE)
    val_lf   = read_listfile(VAL_LISTFILE)
    test_lf  = read_listfile(TEST_LISTFILE)

    print("Building datasets (aggregate features over first 48h)...")
    X_train_df, y_train = build_dataset(train_lf, feature_cols)
    X_val_df,   y_val   = build_dataset(val_lf, feature_cols)
    X_test_df,  y_test  = build_dataset(test_lf, feature_cols)

    # Align columns (in case some variables never appear in a split)
    all_cols = sorted(set(X_train_df.columns) | set(X_val_df.columns) | set(X_test_df.columns))
    X_train_df = X_train_df.reindex(columns=all_cols)
    X_val_df   = X_val_df.reindex(columns=all_cols)
    X_test_df  = X_test_df.reindex(columns=all_cols)

    print(f"Feature matrix shape: train={X_train_df.shape}, val={X_val_df.shape}, test={X_test_df.shape}")
    missing_train = float(np.isnan(X_train_df.values).mean() * 100)
    print(f"Train missingness (before impute): {missing_train:.2f}%")

    # Pipeline: KNN impute -> scale -> LR
    pipe = Pipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=5, weights="distance")),  # <-- KNN imputer here
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ]
    )

    print("Training LR + KNNImputer...")
    pipe.fit(X_train_df.values, y_train)

    print("\nEvaluation:")
    evaluate(pipe, X_train_df.values, y_train, "train")
    evaluate(pipe, X_val_df.values,   y_val,   "val")
    evaluate(pipe, X_test_df.values,  y_test,  "test")

    # Save feature list used
    (CACHE_DIR / "feature_columns_used.json").write_text(json.dumps(all_cols, indent=2))
    print(f"\nSaved feature column list to: {CACHE_DIR / 'feature_columns_used.json'}")


if __name__ == "__main__":
    main()