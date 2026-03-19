import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

RAW_TASK_DIR = "data/in-hospital-mortality"
CLEAN_TASK_DIR = "data/in-hospital-mortality-cleaned"

FEATURE_CACHE_PATH = Path(RAW_TASK_DIR) / "feature_cols.json"
MAX_HOURS = 48.0


def read_timeseries(split: str, fname: str) -> pd.DataFrame:
    """
    Read CLEANED timeseries.
    Val files physically live inside train folder.
    """
    folder = "train" if split == "val" else split
    ts_path = os.path.join(CLEAN_TASK_DIR, folder, fname)

    df = pd.read_csv(ts_path)
    df = df[df["Hours"] <= MAX_HOURS].copy()
    df = df.drop(columns=["Hours"])
    return df


def build_feature_columns(train_listfile: pd.DataFrame) -> list[str]:
    """
    Use cleaned train files to infer consistent column order.
    Only build once.
    """
    if FEATURE_CACHE_PATH.exists():
        with open(FEATURE_CACHE_PATH, "r") as f:
            return json.load(f)

    first_file = train_listfile.iloc[0]["stay"]
    df = read_timeseries("train", first_file)
    cols = sorted(df.columns.tolist())

    with open(FEATURE_CACHE_PATH, "w") as f:
        json.dump(cols, f)

    return cols


def df_to_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    df = df.reindex(columns=feature_cols)
    return df.to_numpy(dtype=np.float32)


def load_split(split: str, feature_cols: list[str]):
    listfile = pd.read_csv(os.path.join(RAW_TASK_DIR, f"{split}_listfile.csv"))

    X_list, y = [], []

    for _, row in tqdm(listfile.iterrows(), total=len(listfile), desc=f"Loading {split}"):
        fname = row["stay"]
        label = int(row["y_true"])

        df = read_timeseries(split, fname)
        X = df_to_matrix(df, feature_cols)

        X_list.append(X)
        y.append(label)

    return X_list, np.array(y)


def aggregate_features(X_list: list[np.ndarray]) -> np.ndarray:
    feats = []

    for x in X_list:
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        minv = np.nanmin(x, axis=0)
        maxv = np.nanmax(x, axis=0)

        df_tmp = pd.DataFrame(x)
        last = df_tmp.ffill().iloc[-1].to_numpy()

        feats.append(np.concatenate([mean, std, minv, maxv, last]))

    return np.vstack(feats)


def eval_probs(y_true, probs, name):
    print(f"{name} ROC-AUC:", roc_auc_score(y_true, probs))
    print(f"{name} PR-AUC :", average_precision_score(y_true, probs))


def main():
    train_listfile = pd.read_csv(os.path.join(RAW_TASK_DIR, "train_listfile.csv"))

    feature_cols = build_feature_columns(train_listfile)
    print("Number of base features:", len(feature_cols))

    train_X, train_y = load_split("train", feature_cols)
    val_X, val_y = load_split("val", feature_cols)
    test_X, test_y = load_split("test", feature_cols)

    Xtr = aggregate_features(train_X)
    Xva = aggregate_features(val_X)
    Xte = aggregate_features(test_X)


    # 1️Impute
    imputer = SimpleImputer(strategy="mean")
    Xtr = imputer.fit_transform(Xtr)
    Xva = imputer.transform(Xva)
    Xte = imputer.transform(Xte)

    # 2️Scale
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)
    Xte = scaler.transform(Xte)

    clf = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs"
    )
    clf.fit(Xtr, train_y)

    eval_probs(val_y, clf.predict_proba(Xva)[:, 1], "VAL")
    eval_probs(test_y, clf.predict_proba(Xte)[:, 1], "TEST")


if __name__ == "__main__":
    main()