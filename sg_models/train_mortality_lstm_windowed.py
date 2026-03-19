import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score


# -----------------------------
# Config
# -----------------------------
TASK_DIR = "data/in-hospital-mortality"

# Set to None for whole-stay, or 48 for first 48 hours
WINDOW_HOURS: Optional[float] = 48   # e.g. 48.0

# Safety cap: whole-stay sequences can be long; cap prevents huge padding.
# For whole-stay: 1000–2000 is usually safe; tune if needed.
MAX_LEN_CAP = 2000

BATCH_TRAIN = 64
BATCH_EVAL = 128
EPOCHS = 10
PATIENCE = 2
LR = 1e-3
HIDDEN = 96
DROPOUT = 0.2
NUM_WORKERS = 0  # mac safe

CACHE_DIR = Path(TASK_DIR) / f"cache_lstm_window_{'all' if WINDOW_HOURS is None else int(WINDOW_HOURS)}h"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CAT_MAPS_PATH = CACHE_DIR / "cat_maps.json"
NORM_STATS_PATH = CACHE_DIR / "norm_stats.json"
COLS_PATH = CACHE_DIR / "feature_cols.json"


# -----------------------------
# I/O helpers
# -----------------------------
def read_timeseries(split: str, fname: str) -> pd.DataFrame:
    # In this benchmark output, val_listfile references files in train/
    folder = "train" if split == "val" else split
    ts_path = os.path.join(TASK_DIR, folder, fname)
    return pd.read_csv(ts_path)


def load_listfile(split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(TASK_DIR, f"{split}_listfile.csv"))


# -----------------------------
# Time features + windowing
# -----------------------------
def add_time_features_and_window(df: pd.DataFrame, window_hours: Optional[float]) -> pd.DataFrame:
    """
    Keeps Hours, adds:
      - t_hours: time since start (Hours - first Hours)
      - dt_hours: delta time since previous row (clipped >=0)
    Then optionally windows to first N hours.
    """
    if "Hours" not in df.columns:
        # If Hours missing (unlikely here), just create zeros
        df = df.copy()
        df["Hours"] = 0.0

    df = df.copy()
    df["Hours"] = pd.to_numeric(df["Hours"], errors="coerce")

    # If Hours has NaNs, fill with forward fill then 0 (only for time index)
    df["Hours"] = df["Hours"].ffill().fillna(0.0)

    # Ensure sorted by time
    df = df.sort_values("Hours", kind="mergesort").reset_index(drop=True)

    t0 = float(df["Hours"].iloc[0]) if len(df) > 0 else 0.0
    df["t_hours"] = df["Hours"] - t0
    df["dt_hours"] = df["Hours"].diff().fillna(0.0)
    df["dt_hours"] = df["dt_hours"].clip(lower=0.0)

    # Windowing
    if window_hours is not None:
        df = df[df["t_hours"] <= float(window_hours)].reset_index(drop=True)

    # We do NOT use raw Hours as a feature; keep t_hours & dt_hours.
    # Drop Hours from features to avoid leakage of absolute charting offset.
    df = df.drop(columns=["Hours"])

    return df


# -----------------------------
# Feature columns
# -----------------------------
def infer_feature_cols() -> List[str]:
    """
    Build canonical columns from one train file AFTER time-features/windowing.
    This ensures t_hours, dt_hours are included.
    """
    if COLS_PATH.exists():
        return json.loads(COLS_PATH.read_text())

    lf = load_listfile("train")
    df = read_timeseries("train", lf.iloc[0]["stay"])
    df = add_time_features_and_window(df, WINDOW_HOURS)

    cols = list(df.columns)
    COLS_PATH.write_text(json.dumps(cols))
    return cols


def infer_cat_and_num_cols(cols: List[str]) -> Tuple[List[str], List[str]]:
    lf = load_listfile("train")
    df = read_timeseries("train", lf.iloc[0]["stay"])
    df = add_time_features_and_window(df, WINDOW_HOURS)

    cat_cols = [c for c in cols if df[c].dtype == object]
    num_cols = [c for c in cols if c not in cat_cols]
    return cat_cols, num_cols


# -----------------------------
# Encoding
# -----------------------------
def build_or_load_cat_maps(train_lf: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict[str, int]]:
    """
    Train-only categorical vocab.
    - __UNK__ = 0 (unseen)
    - __NA__  = 1 (missing)
    """
    if CAT_MAPS_PATH.exists():
        return json.loads(CAT_MAPS_PATH.read_text())

    maps: Dict[str, Dict[str, int]] = {c: {"__UNK__": 0, "__NA__": 1} for c in cat_cols}

    for _, row in tqdm(train_lf.iterrows(), total=len(train_lf), desc="Building categorical vocab (train)"):
        df = read_timeseries("train", row["stay"])
        df = add_time_features_and_window(df, WINDOW_HOURS)

        for c in cat_cols:
            raw = df[c]
            vals = raw.dropna().astype(str).unique()
            m = maps[c]
            for v in vals:
                if v not in m:
                    m[v] = len(m)

    CAT_MAPS_PATH.write_text(json.dumps(maps))
    return maps


def encode_df(
    df: pd.DataFrame,
    cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    cat_maps: Dict[str, Dict[str, int]],
) -> np.ndarray:
    """
    (T, F) float32
    - numeric: float, keep NaN for now
    - categorical: integer ids, missing -> __NA__
    """
    out = []
    for c in cols:
        if c in cat_cols:
            m = cat_maps[c]
            raw = df[c]
            ids = np.full(len(raw), m["__NA__"], dtype=np.float32)

            mask_notna = raw.notna().to_numpy()
            if mask_notna.any():
                s = raw[mask_notna].astype(str)
                ids[mask_notna] = s.map(lambda x: m.get(x, m["__UNK__"])).to_numpy(dtype=np.float32)

            out.append(ids)
        else:
            out.append(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32))

    return np.stack(out, axis=1)


# -----------------------------
# Train-only scaling (numeric only)
# -----------------------------
def build_or_load_norm_stats(
    train_lf: pd.DataFrame,
    cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    cat_maps: Dict[str, Dict[str, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    if NORM_STATS_PATH.exists():
        stats = json.loads(NORM_STATS_PATH.read_text())
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        return mean, std

    F = len(cols)
    sum_ = np.zeros(F, dtype=np.float64)
    sumsq = np.zeros(F, dtype=np.float64)
    count = np.zeros(F, dtype=np.float64)

    num_idx = np.array([i for i, c in enumerate(cols) if c in num_cols], dtype=np.int64)

    for _, row in tqdm(train_lf.iterrows(), total=len(train_lf), desc="Computing normalization stats (train)"):
        df = read_timeseries("train", row["stay"])
        df = add_time_features_and_window(df, WINDOW_HOURS)
        x = encode_df(df, cols, cat_cols, num_cols, cat_maps)

        xn = x[:, num_idx]
        mask = ~np.isnan(xn)

        sum_[num_idx] += np.where(mask, xn, 0.0).sum(axis=0)
        sumsq[num_idx] += np.where(mask, xn * xn, 0.0).sum(axis=0)
        count[num_idx] += mask.sum(axis=0)

    mean = np.zeros(F, dtype=np.float32)  # categoricals: no scaling
    std = np.ones(F, dtype=np.float32)

    c = np.maximum(count[num_idx], 1.0)
    m = (sum_[num_idx] / c).astype(np.float32)
    v = (sumsq[num_idx] / c - (m.astype(np.float64) ** 2)).astype(np.float32)
    v = np.maximum(v, 1e-6)
    s = np.sqrt(v).astype(np.float32)

    mean[num_idx] = m
    std[num_idx] = s

    NORM_STATS_PATH.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}))
    return mean, std


def standardize_and_fill(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = (x - mean[None, :]) / std[None, :]
    return np.nan_to_num(x, nan=0.0).astype(np.float32)


# -----------------------------
# Dataset / Dataloader
# -----------------------------
class MortalityDataset(Dataset):
    def __init__(self, split: str, cols, cat_cols, num_cols, cat_maps, mean, std):
        self.split = split
        self.cols = cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.cat_maps = cat_maps
        self.mean = mean
        self.std = std
        self.lf = load_listfile(split)

    def __len__(self):
        return len(self.lf)

    def __getitem__(self, idx):
        row = self.lf.iloc[idx]
        fname = row["stay"]
        y = float(row["y_true"])

        df = read_timeseries(self.split, fname)
        df = add_time_features_and_window(df, WINDOW_HOURS)

        x = encode_df(df, self.cols, self.cat_cols, self.num_cols, self.cat_maps)
        x = standardize_and_fill(x, self.mean, self.std)

        # whole-stay safety cap
        if x.shape[0] > MAX_LEN_CAP:
            x = x[:MAX_LEN_CAP, :]

        length = x.shape[0]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )


def collate_fn(batch):
    xs, lens, ys = zip(*batch)
    lengths = torch.stack(lens)
    y = torch.stack(ys)

    T_max = int(max(x.shape[0] for x in xs))
    F = int(xs[0].shape[1])

    x_pad = torch.zeros(len(xs), T_max, F, dtype=torch.float32)
    for i, x in enumerate(xs):
        x_pad[i, : x.shape[0], :] = x

    return x_pad, lengths, y


# -----------------------------
# Model
# -----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]
        return self.head(h_last).squeeze(1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, lengths, y in loader:
        x, lengths = x.to(device), lengths.to(device)
        logits = model(x, lengths)
        probs = torch.sigmoid(logits).cpu().numpy()
        ys.append(y.numpy())
        ps.append(probs)
    y_true = np.concatenate(ys)
    p = np.concatenate(ps)
    return roc_auc_score(y_true, p), average_precision_score(y_true, p)


# -----------------------------
# Train
# -----------------------------
def main():
    device = torch.device("cpu")

    cols = infer_feature_cols()
    cat_cols, num_cols = infer_cat_and_num_cols(cols)

    print("WINDOW_HOURS:", WINDOW_HOURS)
    print("Total features:", len(cols))
    print("Categorical cols:", cat_cols)
    print("Numeric cols:", num_cols)

    train_lf = load_listfile("train")
    cat_maps = build_or_load_cat_maps(train_lf, cat_cols)
    mean, std = build_or_load_norm_stats(train_lf, cols, cat_cols, num_cols, cat_maps)

    train_ds = MortalityDataset("train", cols, cat_cols, num_cols, cat_maps, mean, std)
    val_ds = MortalityDataset("val", cols, cat_cols, num_cols, cat_maps, mean, std)
    test_ds = MortalityDataset("test", cols, cat_cols, num_cols, cat_maps, mean, std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_EVAL, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_EVAL, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # class imbalance
    y_train = train_lf["y_true"].astype(int).to_numpy()
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)
    print("pos_weight:", float(pos_weight.item()))

    model = LSTMClassifier(input_dim=len(cols), hidden_dim=HIDDEN, dropout=DROPOUT).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_pr = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for x, lengths, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(loss.item())

        val_roc, val_pr = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={np.mean(losses):.4f}  VAL_ROC={val_roc:.4f}  VAL_PR={val_pr:.4f}")

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping (no PR-AUC improvement for {PATIENCE} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_roc, test_pr = evaluate(model, test_loader, device)
    print("TEST ROC-AUC:", test_roc)
    print("TEST PR-AUC :", test_pr)


if __name__ == "__main__":
    main()