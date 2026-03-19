# train_mortality_lr_knn_imputer_missingind.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator


DATA_DIR = Path("data") / "in-hospital-mortality"
TRAIN_LISTFILE = DATA_DIR / "train_listfile.csv"
VAL_LISTFILE   = DATA_DIR / "val_listfile.csv"
TEST_LISTFILE  = DATA_DIR / "test_listfile.csv"
FEATURE_COLS_JSON = DATA_DIR / "feature_cols.json"

CACHE_DIR = DATA_DIR / "cache_lr_knn_48h_missingind"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HOURS_LIMIT = 48.0
AGG_FUNCS = ["mean", "std", "min", "max", "last"]
MISSING_THRESHOLD = 0.0  # keep indicators for any feature that ever missing


def load_feature_cols() -> List[str]:
    with open(FEATURE_COLS_JSON, "r") as f:
        cols = json.load(f)
    return cols


def read_listfile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "stay" not in df.columns:
        df = df.rename(columns={df.columns[0]: "stay"})
    if "y_true" not in df.columns:
        if "label" in df.columns:
            df = df.rename(columns={"label": "y_true"})
        else:
            raise ValueError(f"{path} missing y_true.")
    return df[["stay", "y_true"]].copy()


def _safe_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def load_timeseries_48h(stay_filename: str, feature_cols: List[str]) -> pd.DataFrame:
    stay_filename = str(stay_filename)
    candidates = [
        DATA_DIR / "train" / stay_filename,
        DATA_DIR / "val" / stay_filename,
        DATA_DIR / "test" / stay_filename,
        DATA_DIR / stay_filename,
    ]
    fpath = next((p for p in candidates if p.exists()), None)
    if fpath is None:
        raise FileNotFoundError(f"Timeseries not found for {stay_filename}")

    ts = pd.read_csv(fpath)
    if "Hours" not in ts.columns:
        if "hours" in ts.columns:
            ts = ts.rename(columns={"hours": "Hours"})
        else:
            ts = ts.rename(columns={ts.columns[0]: "Hours"})

    ts["Hours"] = pd.to_numeric(ts["Hours"], errors="coerce")
    ts = ts[ts["Hours"].notna() & (ts["Hours"] <= HOURS_LIMIT)].copy()

    keep_vars = [c for c in feature_cols if c in ts.columns]
    if keep_vars:
        ts = _safe_to_numeric(ts, keep_vars)
        return ts[["Hours"] + keep_vars]
    return ts[["Hours"]].copy()


def featurize_stay_agg(ts: pd.DataFrame, feature_cols: List[str]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    for v in feature_cols:
        s = ts[v] if v in ts.columns else pd.Series([], dtype=float)
        if len(s) == 0 or s.notna().sum() == 0:
            for fn in AGG_FUNCS:
                feats[f"{v}__{fn}"] = np.nan
            continue
        feats[f"{v}__mean"] = float(np.nanmean(s.values))
        feats[f"{v}__std"]  = float(np.nanstd(s.values, ddof=1)) if s.notna().sum() > 1 else 0.0
        feats[f"{v}__min"]  = float(np.nanmin(s.values))
        feats[f"{v}__max"]  = float(np.nanmax(s.values))
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


def main():
    feature_cols = load_feature_cols()

    train_lf = read_listfile(TRAIN_LISTFILE)
    val_lf   = read_listfile(VAL_LISTFILE)
    test_lf  = read_listfile(TEST_LISTFILE)

    print("Building datasets...")
    X_train_df, y_train = build_dataset(train_lf, feature_cols)
    X_val_df,   y_val   = build_dataset(val_lf, feature_cols)
    X_test_df,  y_test  = build_dataset(test_lf, feature_cols)

    all_cols = sorted(set(X_train_df.columns) | set(X_val_df.columns) | set(X_test_df.columns))
    X_train_df = X_train_df.reindex(columns=all_cols)
    X_val_df   = X_val_df.reindex(columns=all_cols)
    X_test_df  = X_test_df.reindex(columns=all_cols)

    print(f"Shapes: train={X_train_df.shape}, val={X_val_df.shape}, test={X_test_df.shape}")

    # ColumnTransformer:
    # - MissingIndicator creates binary features for missingness
    # - KNNImputer imputes numeric features
    # - Then scale and LR
    pre = ColumnTransformer(
        transformers=[
            ("num_impute", KNNImputer(n_neighbors=5, weights="distance"), list(range(len(all_cols)))),
            ("miss_ind", MissingIndicator(features="missing-only"), list(range(len(all_cols)))),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("scaler", StandardScaler(with_mean=False)),  # with_mean=False because MissingIndicator adds extra cols
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
        ]
    )

    print("Training LR + (MissingIndicator + KNNImputer)...")
    pipe.fit(X_train_df.values, y_train)

    print("\nEvaluation:")
    evaluate(pipe, X_train_df.values, y_train, "train")
    evaluate(pipe, X_val_df.values,   y_val,   "val")
    evaluate(pipe, X_test_df.values,  y_test,  "test")

    (CACHE_DIR / "feature_columns_used.json").write_text(json.dumps(all_cols, indent=2))
    print(f"\nSaved feature column list to: {CACHE_DIR / 'feature_columns_used.json'}")


if __name__ == "__main__":
    main()