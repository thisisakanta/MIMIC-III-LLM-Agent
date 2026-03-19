import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

TASK_DIR = "data/in-hospital-mortality"
FEATURE_CACHE_PATH = Path(TASK_DIR) / "feature_cols.json"


def read_timeseries(split: str, fname: str) -> pd.DataFrame:
    # val_listfile points to files located in train/
    folder = "train" if split == "val" else split
    ts_path = os.path.join(TASK_DIR, folder, fname)

    df = pd.read_csv(ts_path)
    if "Hours" in df.columns:
        df = df.drop(columns=["Hours"])
    return df


def build_feature_columns_from_train(train_listfile: pd.DataFrame) -> list[str]:
    # cache one-hot columns (from raw df with get_dummies)
    if FEATURE_CACHE_PATH.exists():
        with open(FEATURE_CACHE_PATH, "r") as f:
            return json.load(f)

    all_cols = set()
    for _, row in tqdm(
        train_listfile.iterrows(),
        total=len(train_listfile),
        desc="Inferring one-hot feature columns from ALL train files",
    ):
        df = read_timeseries("train", row["stay"])
        df2 = pd.get_dummies(df, dummy_na=True)
        all_cols.update(df2.columns)

    cols = sorted(all_cols)
    FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_CACHE_PATH, "w") as f:
        json.dump(cols, f)
    return cols


def df_to_aligned_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    df2 = pd.get_dummies(df, dummy_na=True)
    df2 = df2.reindex(columns=feature_cols)  # keep NaN
    return df2.to_numpy(dtype=np.float32)


def preprocess_matrix_ffill_and_missing(x: np.ndarray):
    """
    x: (T, F) float matrix; may contain NaNs.
    Returns:
      x_filled: (T, F) forward-filled numeric matrix (remaining NaNs -> 0)
      missing_frac: (F,) fraction of timepoints missing for each feature
      ever_missing: (F,) whether feature was ever missing (0/1)
    """
    # missing mask
    miss = np.isnan(x)
    missing_frac = miss.mean(axis=0).astype(np.float32)
    ever_missing = miss.any(axis=0).astype(np.float32)

    # forward fill along time axis
    x_filled = x.copy()
    T, F = x_filled.shape
    for j in range(F):
        last = np.nan
        for t in range(T):
            v = x_filled[t, j]
            if np.isnan(v):
                if not np.isnan(last):
                    x_filled[t, j] = last
            else:
                last = v

    # any remaining NaNs (e.g., feature never observed in this stay) -> 0
    x_filled = np.nan_to_num(x_filled, nan=0.0)

    return x_filled, missing_frac, ever_missing


def load_split(split: str, feature_cols: list[str]):
    listfile = pd.read_csv(os.path.join(TASK_DIR, f"{split}_listfile.csv"))
    X_list, y = [], []

    for _, row in tqdm(listfile.iterrows(), total=len(listfile), desc=f"Loading {split}"):
        fname = row["stay"]
        label = int(row["y_true"])

        df = read_timeseries(split, fname)
        X = df_to_aligned_matrix(df, feature_cols)

        X_list.append(X)
        y.append(label)

    return X_list, np.array(y, dtype=np.int64)


def aggregate_features_with_missingness(X_list: list[np.ndarray]) -> np.ndarray:
    """
    For each stay, produce a fixed vector:
      time summaries on forward-filled matrix: mean, std, min, max, last  (5*F)
      + missingness summaries: missing_frac (F) + ever_missing (F)       (+2*F)
    Total per stay = 7*F
    """
    feats = []
    for x in X_list:
        x_filled, missing_frac, ever_missing = preprocess_matrix_ffill_and_missing(x)

        mean = x_filled.mean(axis=0)
        std = x_filled.std(axis=0)
        minv = x_filled.min(axis=0)
        maxv = x_filled.max(axis=0)
        last = x_filled[-1]

        feats.append(np.concatenate([mean, std, minv, maxv, last, missing_frac, ever_missing]))

    return np.vstack(feats).astype(np.float32)


def eval_probs(y_true: np.ndarray, probs: np.ndarray, name: str):
    print(f"{name} ROC-AUC:", roc_auc_score(y_true, probs))
    print(f"{name} PR-AUC :", average_precision_score(y_true, probs))


def main():
    train_listfile = pd.read_csv(os.path.join(TASK_DIR, "train_listfile.csv"))

    feature_cols = build_feature_columns_from_train(train_listfile)
    print("Expanded feature columns:", len(feature_cols))
    print(f"Feature cache: {FEATURE_CACHE_PATH} ({'exists' if FEATURE_CACHE_PATH.exists() else 'not created'})")

    train_X, train_y = load_split("train", feature_cols)
    val_X, val_y = load_split("val", feature_cols)
    test_X, test_y = load_split("test", feature_cols)

    print("Train/Val/Test:", len(train_X), len(val_X), len(test_X))

    Xtr = aggregate_features_with_missingness(train_X)
    Xva = aggregate_features_with_missingness(val_X)
    Xte = aggregate_features_with_missingness(test_X)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)
    Xte = scaler.transform(Xte)

    clf = LogisticRegression(max_iter=4000, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr, train_y)

    eval_probs(val_y, clf.predict_proba(Xva)[:, 1], "VAL")
    eval_probs(test_y, clf.predict_proba(Xte)[:, 1], "TEST")


if __name__ == "__main__":
    main()