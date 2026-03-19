# train_mortality_lr_mice_imputer_missingind_keepall.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Config (matches your repo)
# ----------------------------
TASK_DIR = Path("data") / "in-hospital-mortality"
FEATURE_JSON = TASK_DIR / "feature_cols.json"

LISTFILES = {
    "train": TASK_DIR / "train_listfile.csv",
    "val": TASK_DIR / "val_listfile.csv",
    "test": TASK_DIR / "test_listfile.csv",
}

# IMPORTANT: In your benchmark export, val stays are stored under train/
# We'll resolve paths robustly.
SPLIT_DIRS = {
    "train": TASK_DIR / "train",
    "val": TASK_DIR / "val",
    "test": TASK_DIR / "test",
}

# Clinical defaults / ranges file (from repo)
RANGES_CSV = Path("mimic3benchmark") / "resources" / "variable_ranges.csv"

# Output cache
OUT_DIR = TASK_DIR / "cache_lr_mice_48h_missingind_keepall"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 48h window
MAX_HOURS = 48.0

# Aggregation => 5 stats per base feature => 51 * 5 = 255
AGG_STATS = ["mean", "std", "min", "max", "last"]

# MICE settings (stable)
SEED = 42
MICE_MAX_ITER = 10
MICE_TOL = 1e-3

# Clipping quantiles to avoid MICE blowups (train-only)
CLIP_LOW_Q = 0.005
CLIP_HIGH_Q = 0.995

# LR settings
LR_MAX_ITER = 2000

np.random.seed(SEED)


# ----------------------------
# IO helpers
# ----------------------------
def load_feature_cols(path: Path) -> List[str]:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and "feature_cols" in obj:
        return list(obj["feature_cols"])
    if isinstance(obj, list):
        return list(obj)
    raise ValueError(f"Unrecognized feature_cols.json format at {path}")


def read_listfile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "stay" not in df.columns:
        df.rename(columns={df.columns[0]: "stay"}, inplace=True)
    if "y_true" not in df.columns:
        df.rename(columns={df.columns[1]: "y_true"}, inplace=True)
    df["stay"] = df["stay"].astype(str)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)
    return df[["stay", "y_true"]]


def resolve_timeseries_path(task_dir: Path, split: str, stay: str) -> Path:
    """
    Your export stores val files under train/. This tries:
      1) requested split folder
      2) train/val/test folders as fallbacks
    Supports .gz too.
    """
    stay = str(stay)
    order = [split, "train", "val", "test"]
    seen = set()
    order = [s for s in order if not (s in seen or seen.add(s))]

    for sp in order:
        p = task_dir / sp / stay
        if p.exists():
            return p
        pgz = task_dir / sp / (stay + ".gz")
        if pgz.exists():
            return pgz

    raise FileNotFoundError(f"Could not resolve timeseries file for stay={stay} (split={split})")


# ----------------------------
# Clinical defaults from variable_ranges.csv
# ----------------------------
def load_clinical_impute_defaults(ranges_csv: Path) -> Dict[str, float]:
    """
    Returns dict: VARIABLE -> IMPUTE (float)
    """
    if not ranges_csv.exists():
        return {}

    df = pd.read_csv(ranges_csv)
    # expected columns like: LEVEL2, OUTLIER LOW, VALID LOW, IMPUTE, VALID HIGH, OUTLIER HIGH
    # normalize column names
    cols = {c.lower().strip(): c for c in df.columns}
    var_col = None
    for candidate in ["level2", "variable"]:
        if candidate in cols:
            var_col = cols[candidate]
            break
    if var_col is None:
        # try exact known from benchmark
        if "LEVEL2" in df.columns:
            var_col = "LEVEL2"
        else:
            return {}

    if "IMPUTE" not in df.columns:
        # some variants rename to "IMPUTE" already; if not, bail
        return {}

    out: Dict[str, float] = {}
    for _, r in df.iterrows():
        v = str(r[var_col]).strip()
        if not v:
            continue
        imp = r["IMPUTE"]
        try:
            imp_f = float(imp)
            out[v] = imp_f
        except Exception:
            continue
    return out


# ----------------------------
# Feature engineering (48h aggregates)
# ----------------------------
def aggregate_48h_features(ts: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """
    For each base feature, compute [mean, std, min, max, last] over <=48h.
    Returns shape [D * 5]
    """
    if "Hours" not in ts.columns:
        raise ValueError("Timeseries CSV missing 'Hours' column.")

    ts = ts[ts["Hours"] <= MAX_HOURS].copy()
    ts = ts.sort_values("Hours")

    out_vals: List[float] = []
    for col in feature_cols:
        if col not in ts.columns:
            x = np.array([], dtype=np.float64)
        else:
            x = pd.to_numeric(ts[col], errors="coerce").to_numpy(dtype=np.float64)

        x = x[np.isfinite(x)]
        if x.size == 0:
            out_vals.extend([np.nan, np.nan, np.nan, np.nan, np.nan])
        else:
            out_vals.append(float(np.mean(x)))
            out_vals.append(float(np.std(x, ddof=0)))
            out_vals.append(float(np.min(x)))
            out_vals.append(float(np.max(x)))
            out_vals.append(float(x[-1]))

    return np.array(out_vals, dtype=np.float64)


def build_feature_names(feature_cols: List[str]) -> List[str]:
    names = []
    for col in feature_cols:
        for s in AGG_STATS:
            names.append(f"{col}__{s}")
    return names


def build_split_matrix(split: str, lf: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (X, y). X shape [N, 255]
    """
    X = np.zeros((len(lf), len(feature_cols) * len(AGG_STATS)), dtype=np.float64)
    y = lf["y_true"].to_numpy(dtype=int)

    for i, stay in enumerate(lf["stay"].tolist()):
        fp = resolve_timeseries_path(TASK_DIR, split, stay)
        ts = pd.read_csv(fp)
        X[i] = aggregate_48h_features(ts, feature_cols)

    return X, y


# ----------------------------
# Stabilization: clipping bounds from TRAIN
# ----------------------------
def make_clip_bounds_from_train(X_train: np.ndarray, low_q=0.005, high_q=0.995) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.nanquantile(X_train, low_q, axis=0)
    hi = np.nanquantile(X_train, high_q, axis=0)

    lo = np.where(np.isfinite(lo), lo, -np.inf)
    hi = np.where(np.isfinite(hi), hi, np.inf)

    swap = lo > hi
    lo[swap], hi[swap] = hi[swap], lo[swap]
    return lo, hi


def clip_matrix(X: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X[~np.isfinite(X)] = np.nan
    return np.clip(X, lo, hi)


# ----------------------------
# Metrics
# ----------------------------
def eval_split(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    if len(np.unique(y_true)) < 2:
        print(f"{name:>6} | ROC-AUC: nan | PR-AUC: nan")
        return
    roc = roc_auc_score(y_true, y_prob)
    pr = average_precision_score(y_true, y_prob)
    print(f"{name:>6} | ROC-AUC: {roc:.3f} | PR-AUC: {pr:.3f}")


# ----------------------------
# Main
# ----------------------------
def main():
    feature_cols = load_feature_cols(FEATURE_JSON)
    base_D = len(feature_cols)
    print(f"Loaded {base_D} base feature columns from {FEATURE_JSON}")

    feat_names = build_feature_names(feature_cols)
    if len(feat_names) != base_D * len(AGG_STATS):
        raise RuntimeError("Unexpected feature naming length mismatch.")

    # Load listfiles
    train_lf = read_listfile(LISTFILES["train"])
    val_lf = read_listfile(LISTFILES["val"])
    test_lf = read_listfile(LISTFILES["test"])

    # Build aggregate matrices
    print("Building 48h aggregate feature matrices...")
    X_train_raw, y_train = build_split_matrix("train", train_lf, feature_cols)
    X_val_raw, y_val = build_split_matrix("val", val_lf, feature_cols)
    X_test_raw, y_test = build_split_matrix("test", test_lf, feature_cols)

    print(f"Feature matrix shape: train={X_train_raw.shape}, val={X_val_raw.shape}, test={X_test_raw.shape}")

    # Missingness indicators from RAW (before any filling)
    miss_train = np.isnan(X_train_raw).astype(np.float32)
    miss_val = np.isnan(X_val_raw).astype(np.float32)
    miss_test = np.isnan(X_test_raw).astype(np.float32)

    train_missing_pct = float(np.isnan(X_train_raw).mean() * 100.0)
    print(f"Train missingness (raw, before fixes): {train_missing_pct:.2f}%")

    # Identify columns that are ALL missing in TRAIN
    all_missing_cols = np.where(np.all(np.isnan(X_train_raw), axis=0))[0].tolist()
    print(f"Columns all-missing in TRAIN: {len(all_missing_cols)} / {X_train_raw.shape[1]}")

    # Load clinical defaults
    clinical_defaults = load_clinical_impute_defaults(RANGES_CSV)
    if clinical_defaults:
        print(f"Loaded clinical IMPUTE defaults for {len(clinical_defaults)} variables from {RANGES_CSV}")
    else:
        print(f"WARNING: Could not load clinical defaults from {RANGES_CSV}. Will fall back to 0.0 defaults.")

    # Build per-aggregate-column defaults
    # For col "Temp__mean/min/max/last" default = IMPUTE(Temp)
    # For "Temp__std" default = 0.0
    agg_defaults = np.zeros(X_train_raw.shape[1], dtype=np.float64)

    for j, fname in enumerate(feat_names):
        base, stat = fname.split("__", 1)
        base_def = float(clinical_defaults.get(base, 0.0))
        if stat == "std":
            agg_defaults[j] = 0.0
        else:
            agg_defaults[j] = base_def

    # Prefill ONLY the all-missing-in-train columns (keep others NaN for MICE to handle)
    def prefill_all_missing_cols(X_raw: np.ndarray) -> np.ndarray:
        X = np.asarray(X_raw, dtype=np.float64).copy()
        X[~np.isfinite(X)] = np.nan
        if all_missing_cols:
            X[:, all_missing_cols] = np.where(
                np.isnan(X[:, all_missing_cols]),
                agg_defaults[np.newaxis, all_missing_cols],
                X[:, all_missing_cols],
            )
        return X

    X_train_prefill = prefill_all_missing_cols(X_train_raw)
    X_val_prefill = prefill_all_missing_cols(X_val_raw)
    X_test_prefill = prefill_all_missing_cols(X_test_raw)

    # Train-only clipping bounds, then clip all splits
    lo, hi = make_clip_bounds_from_train(X_train_prefill, low_q=CLIP_LOW_Q, high_q=CLIP_HIGH_Q)
    X_train_clip = clip_matrix(X_train_prefill, lo, hi)
    X_val_clip = clip_matrix(X_val_prefill, lo, hi)
    X_test_clip = clip_matrix(X_test_prefill, lo, hi)

    # MICE / IterativeImputer (Ridge estimator is much more stable than BayesianRidge here)
    mice = IterativeImputer(
        estimator=Ridge(alpha=1.0, random_state=SEED),
        max_iter=MICE_MAX_ITER,
        tol=MICE_TOL,
        initial_strategy="mean",
        imputation_order="ascending",
        skip_complete=True,
        random_state=SEED,
        sample_posterior=False,
    )

    # Fit on TRAIN only
    print("Training LR + (MICE/IterativeImputer) with KEEP-ALL + explicit missingness indicators...")
    X_train_imp = mice.fit_transform(X_train_clip)
    X_val_imp = mice.transform(X_val_clip)
    X_test_imp = mice.transform(X_test_clip)

    # Safety: if any NaNs remain (rare), mean-impute them
    safety = SimpleImputer(strategy="mean")
    X_train_imp = safety.fit_transform(X_train_imp)
    X_val_imp = safety.transform(X_val_imp)
    X_test_imp = safety.transform(X_test_imp)

    # Final safety: replace inf and clip again (prevents float issues)
    X_train_imp[~np.isfinite(X_train_imp)] = np.nan
    X_val_imp[~np.isfinite(X_val_imp)] = np.nan
    X_test_imp[~np.isfinite(X_test_imp)] = np.nan

    X_train_imp = safety.fit_transform(X_train_imp)
    X_val_imp = safety.transform(X_val_imp)
    X_test_imp = safety.transform(X_test_imp)

    X_train_imp = clip_matrix(X_train_imp, lo, hi)
    X_val_imp = clip_matrix(X_val_imp, lo, hi)
    X_test_imp = clip_matrix(X_test_imp, lo, hi)

    # Append missingness indicators (ALL 255 columns)
    X_train_final = np.concatenate([X_train_imp, miss_train], axis=1)
    X_val_final = np.concatenate([X_val_imp, miss_val], axis=1)
    X_test_final = np.concatenate([X_test_imp, miss_test], axis=1)

    # Scale (train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_final)
    X_val_s = scaler.transform(X_val_final)
    X_test_s = scaler.transform(X_test_final)

    # LR
    clf = LogisticRegression(
        max_iter=LR_MAX_ITER,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train_s, y_train)

    # Evaluate
    print("\nEvaluation:")
    p_train = clf.predict_proba(X_train_s)[:, 1]
    p_val = clf.predict_proba(X_val_s)[:, 1]
    p_test = clf.predict_proba(X_test_s)[:, 1]

    eval_split("train", y_train, p_train)
    eval_split("val", y_val, p_val)
    eval_split("test", y_test, p_test)

    # Save artifacts
    used_feature_names = feat_names + [f"miss__{n}" for n in feat_names]
    (OUT_DIR / "feature_columns_used.json").write_text(json.dumps(used_feature_names, indent=2))
    (OUT_DIR / "all_missing_cols_in_train.json").write_text(json.dumps(all_missing_cols, indent=2))

    # Save basic run metadata
    meta = {
        "model": "LogisticRegression",
        "imputer": "IterativeImputer(MICE) with Ridge(alpha=1.0)",
        "missing_indicators": "explicit missing mask for ALL 255 aggregate features",
        "clip_low_q": CLIP_LOW_Q,
        "clip_high_q": CLIP_HIGH_Q,
        "mice_max_iter": MICE_MAX_ITER,
        "mice_tol": MICE_TOL,
        "seed": SEED,
        "max_hours": MAX_HOURS,
        "agg_stats": AGG_STATS,
        "n_base_features": base_D,
        "n_agg_features": len(feat_names),
        "n_total_features_after_missing_indicators": len(used_feature_names),
        "n_all_missing_cols_in_train": len(all_missing_cols),
    }
    (OUT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved artifacts to: {OUT_DIR}")


if __name__ == "__main__":
    main()