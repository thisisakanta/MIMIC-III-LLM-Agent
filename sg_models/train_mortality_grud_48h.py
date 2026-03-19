# train_mortality_grud_48h.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


# ----------------------------
# Path resolver (val files sometimes stored under train/)
# ----------------------------
def resolve_timeseries_path(task_dir: Path, split: str, stay: str) -> Optional[Path]:
    """
    Some benchmark exports store val files inside train/. This resolver:
    1) tries the requested split folder first
    2) then falls back to train/val/test folders
    3) supports .gz files
    """
    stay = str(stay)

    # try requested split first, then others
    split_order = [split, "train", "val", "test"]
    seen = set()
    split_order = [s for s in split_order if not (s in seen or seen.add(s))]

    for sp in split_order:
        p = task_dir / sp / stay
        if p.exists():
            return p
        pgz = task_dir / sp / (stay + ".gz")
        if pgz.exists():
            return pgz
    return None


# ----------------------------
# Config
# ----------------------------
DATA_DIR = Path("data") / "in-hospital-mortality"
FEATURE_JSON = DATA_DIR / "feature_cols.json"

LISTFILES = {
    "train": DATA_DIR / "train_listfile.csv",
    "val": DATA_DIR / "val_listfile.csv",
    "test": DATA_DIR / "test_listfile.csv",
}

SPLIT_DIR = {
    "train": DATA_DIR / "train",
    "val": DATA_DIR / "val",
    "test": DATA_DIR / "test",
}

CACHE_DIR = DATA_DIR / "cache_grud_48h"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model/training params
MAX_HOURS = 48.0
DT_HOURS = 1.0
T_STEPS = int(MAX_HOURS / DT_HOURS)  # 48
BATCH_SIZE = 64
PATIENCE = 2
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 128
DROPOUT = 0.1
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------
# Utilities
# ----------------------------
def load_feature_cols(path: Path) -> List[str]:
    with open(path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "feature_cols" in obj:
        return list(obj["feature_cols"])
    if isinstance(obj, list):
        return list(obj)
    raise ValueError(f"Unrecognized feature_cols.json format: {path}")


def read_listfile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "stay" not in df.columns:
        df.rename(columns={df.columns[0]: "stay"}, inplace=True)
    if "y_true" not in df.columns:
        df.rename(columns={df.columns[1]: "y_true"}, inplace=True)
    df["stay"] = df["stay"].astype(str)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)
    return df[["stay", "y_true"]]


def compute_train_stats(
    train_list: pd.DataFrame,
    train_dir: Path,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-feature mean/std using ONLY observed values in train split.
    Returns (mean, std, emp_mean) where:
      mean/std used for standardization (z-score),
      emp_mean used by GRU-D for decay-to-mean.
    """
    sums = np.zeros(len(feature_cols), dtype=np.float64)
    sums_sq = np.zeros(len(feature_cols), dtype=np.float64)
    counts = np.zeros(len(feature_cols), dtype=np.float64)

    for stay in train_list["stay"].tolist():
        fp = train_dir / stay
        if not fp.exists():
            if not stay.endswith(".csv") and (train_dir / f"{stay}.csv").exists():
                fp = train_dir / f"{stay}.csv"
            else:
                continue

        df = pd.read_csv(fp)
        if "Hours" not in df.columns:
            raise ValueError(f"Missing 'Hours' column in {fp}")

        df = df[df["Hours"] <= MAX_HOURS].copy()
        for j, col in enumerate(feature_cols):
            if col not in df.columns:
                continue
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            m = np.isfinite(x)
            if m.any():
                sums[j] += x[m].sum()
                sums_sq[j] += (x[m] ** 2).sum()
                counts[j] += m.sum()

    emp_mean = np.divide(sums, np.maximum(counts, 1.0))
    var = np.divide(sums_sq, np.maximum(counts, 1.0)) - emp_mean**2
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)

    emp_mean[counts == 0] = 0.0
    std[counts == 0] = 1.0

    mean = emp_mean.copy()
    return mean.astype(np.float32), std.astype(np.float32), emp_mean.astype(np.float32)


def resample_to_hourly_grid(
    ts_df: pd.DataFrame,
    feature_cols: List[str],
    t_steps: int,
    dt_hours: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw event-time observations into fixed [T, D] grid (hourly bins).
    For each hour bin, take LAST observed value within that bin.
    """
    if "Hours" not in ts_df.columns:
        raise ValueError("Timeseries CSV must contain 'Hours' column.")

    ts_df = ts_df[ts_df["Hours"] <= MAX_HOURS].copy()

    bin_idx = np.floor(ts_df["Hours"].to_numpy(dtype=np.float32) / dt_hours).astype(int)
    bin_idx = np.clip(bin_idx, 0, t_steps - 1)
    ts_df["_bin"] = bin_idx

    X = np.full((t_steps, len(feature_cols)), np.nan, dtype=np.float32)

    ts_df = ts_df.sort_values("Hours")
    grouped = ts_df.groupby("_bin", sort=False)
    for b, g in grouped:
        for j, col in enumerate(feature_cols):
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=np.float32)
            finite = np.isfinite(vals)
            if finite.any():
                X[b, j] = vals[np.where(finite)[0][-1]]

    M = np.isfinite(X).astype(np.float32)
    return X, M


def compute_deltas(mask: np.ndarray, dt_hours: float) -> np.ndarray:
    """
    mask: [T, D] where 1 observed else 0
    delta[t,d] = time since last observation of feature d at time t (hours)
    """
    T, D = mask.shape
    delta = np.zeros((T, D), dtype=np.float32)
    delta[0] = (1.0 - mask[0]) * dt_hours
    for t in range(1, T):
        delta[t] = dt_hours + (1.0 - mask[t]) * delta[t - 1]
        delta[t] = mask[t] * dt_hours + (1.0 - mask[t]) * delta[t]
    return delta


# ----------------------------
# Dataset
# ----------------------------
@dataclass
class Sample:
    x: torch.Tensor  # [T, D]
    m: torch.Tensor  # [T, D]
    d: torch.Tensor  # [T, D]
    y: torch.Tensor  # [1]


class IHM48hDataset(Dataset):
    def __init__(
        self,
        listfile: pd.DataFrame,
        split_dir: Path,
        feature_cols: List[str],
        mean: np.ndarray,
        std: np.ndarray,
        emp_mean: np.ndarray,
        cache_path: Optional[Path] = None,
        task_dir: Optional[Path] = None,
        split: Optional[str] = None,
    ):
        """
        split_dir: Path to the split folder (e.g., data/in-hospital-mortality/train)
        task_dir : Path to the task root folder (e.g., data/in-hospital-mortality)
        split    : "train"/"val"/"test"
        """
        self.df = listfile.reset_index(drop=True)
        self.split_dir = split_dir
        self.feature_cols = feature_cols
        self.mean = mean
        self.std = std
        self.emp_mean = emp_mean

        self.task_dir = task_dir if task_dir is not None else split_dir.parent
        self.split = split if split is not None else split_dir.name

        self.cache_path = cache_path
        self._cache: Optional[Dict[int, Sample]] = None

        if self.cache_path is not None and self.cache_path.exists():
            self._cache = torch.load(self.cache_path)

    def __len__(self) -> int:
        return len(self.df)

    def _load_one(self, idx: int) -> Sample:
        row = self.df.iloc[idx]
        stay = str(row["stay"])
        y = float(row["y_true"])

        # 1) try split_dir/stay
        fp = self.split_dir / stay

        # 2) try adding .csv
        if not fp.exists():
            if not stay.endswith(".csv") and (self.split_dir / f"{stay}.csv").exists():
                fp = self.split_dir / f"{stay}.csv"

        # 3) cross-split resolve (val-in-train issue)
        if not fp.exists():
            resolved = resolve_timeseries_path(self.task_dir, self.split, stay)
            if resolved is None and not stay.endswith(".csv"):
                resolved = resolve_timeseries_path(self.task_dir, self.split, stay + ".csv")
            if resolved is None:
                raise FileNotFoundError(
                    f"Missing timeseries file for stay='{stay}' "
                    f"(split='{self.split}', split_dir='{self.split_dir}')"
                )
            fp = resolved

        ts = pd.read_csv(fp)

        X_raw, M = resample_to_hourly_grid(ts, self.feature_cols, T_STEPS, DT_HOURS)
        Dlt = compute_deltas(M, DT_HOURS)

        # z-score standardization (train stats)
        X = (X_raw - self.mean.reshape(1, -1)) / self.std.reshape(1, -1)

        # replace missing with 0 (mask M tells missingness)
        X = np.nan_to_num(X, nan=0.0).astype(np.float32)

        return Sample(
            x=torch.from_numpy(X),
            m=torch.from_numpy(M.astype(np.float32)),
            d=torch.from_numpy(Dlt.astype(np.float32)),
            y=torch.tensor([y], dtype=torch.float32),
        )

    def __getitem__(self, idx: int) -> Sample:
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        return self._load_one(idx)

    def save_cache(self):
        if self.cache_path is None:
            return
        cache: Dict[int, Sample] = {}
        for i in range(len(self)):
            cache[i] = self._load_one(i)
        torch.save(cache, self.cache_path)
        self._cache = cache


def collate_fn(batch: List[Sample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.stack([b.x for b in batch], dim=0)  # [B,T,D]
    m = torch.stack([b.m for b in batch], dim=0)  # [B,T,D]
    d = torch.stack([b.d for b in batch], dim=0)  # [B,T,D]
    y = torch.cat([b.y for b in batch], dim=0)    # [B]
    return x, m, d, y


# ----------------------------
# GRU-D
# ----------------------------
class GRUDCell(nn.Module):
    """
    GRU-D (Che et al.) cell.
    Inputs per time t:
      x_t: [B,D] value (zeros where missing)
      m_t: [B,D] mask
      d_t: [B,D] delta time since last observation (hours)
    Uses x_mean: [D] empirical mean for decay-to-mean.
    """
    def __init__(self, input_size: int, hidden_size: int, x_mean: np.ndarray, dropout: float = 0.0):
        super().__init__()
        self.D = input_size
        self.H = hidden_size

        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))  # [D]

        # feature-wise decay for x
        self.Wx = nn.Parameter(torch.empty(self.D))
        self.bx = nn.Parameter(torch.zeros(self.D))

        # hidden decay: delta [B,D] -> [B,H]
        self.Wh = nn.Linear(self.D, hidden_size, bias=True)

        # GRU gates
        self.z = nn.Linear(self.D + hidden_size, hidden_size)
        self.r = nn.Linear(self.D + hidden_size, hidden_size)
        self.h_tilde = nn.Linear(self.D + hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        nn.init.uniform_(self.Wx, a=-0.05, b=0.05)
        nn.init.xavier_uniform_(self.Wh.weight)
        nn.init.zeros_(self.Wh.bias)

    def forward(self, x_t, m_t, d_t, h_prev):
        gamma_x = torch.exp(-torch.relu(d_t * self.Wx + self.bx))  # [B,D]
        x_hat = m_t * x_t + (1.0 - m_t) * (gamma_x * x_t + (1.0 - gamma_x) * self.x_mean)  # [B,D]

        gamma_h = torch.exp(-torch.relu(self.Wh(d_t)))  # [B,H]
        h_prev = gamma_h * h_prev

        inp = torch.cat([x_hat, h_prev], dim=-1)
        z = torch.sigmoid(self.z(inp))
        r = torch.sigmoid(self.r(inp))

        inp_r = torch.cat([x_hat, r * h_prev], dim=-1)
        h_tilde = torch.tanh(self.h_tilde(inp_r))

        h = (1.0 - z) * h_prev + z * h_tilde
        h = self.dropout(h)
        return h


class GRUDModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, x_mean: np.ndarray, dropout: float = 0.0):
        super().__init__()
        self.cell = GRUDCell(input_size, hidden_size, x_mean=x_mean, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, m, d):
        B, T, D = x.shape
        h = torch.zeros((B, self.cell.H), device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], m[:, t], d[:, t], h)
        logits = self.classifier(h).squeeze(-1)
        return logits


# ----------------------------
# Train / Eval
# ----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    ys, ps = [], []
    for x, m, d, y in loader:
        x, m, d = x.to(DEVICE), m.to(DEVICE), d.to(DEVICE)
        logits = model(x, m, d)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        ys.append(y.numpy())
        ps.append(prob)

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)

    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    pr = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    return float(roc), float(pr)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for x, m, d, y in loader:
        x, m, d, y = x.to(DEVICE), m.to(DEVICE), d.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x, m, d)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * y.shape[0]
        n += y.shape[0]
    return total_loss / max(n, 1)


def main():
    feature_cols = load_feature_cols(FEATURE_JSON)
    print(f"Loaded {len(feature_cols)} feature columns from {FEATURE_JSON}")

    train_df = read_listfile(LISTFILES["train"])
    val_df = read_listfile(LISTFILES["val"])
    test_df = read_listfile(LISTFILES["test"])

    mean, std, emp_mean = compute_train_stats(train_df, SPLIT_DIR["train"], feature_cols)
    print(f"Computed train stats: mean/std for {len(feature_cols)} features")

    train_cache = CACHE_DIR / "train.pt"
    val_cache = CACHE_DIR / "val.pt"
    test_cache = CACHE_DIR / "test.pt"

    train_set = IHM48hDataset(
        train_df, SPLIT_DIR["train"], feature_cols, mean, std, emp_mean,
        cache_path=train_cache, task_dir=DATA_DIR, split="train"
    )
    val_set = IHM48hDataset(
        val_df, SPLIT_DIR["val"], feature_cols, mean, std, emp_mean,
        cache_path=val_cache, task_dir=DATA_DIR, split="val"
    )
    test_set = IHM48hDataset(
        test_df, SPLIT_DIR["test"], feature_cols, mean, std, emp_mean,
        cache_path=test_cache, task_dir=DATA_DIR, split="test"
    )

    if not train_cache.exists():
        print("Building train cache...")
        train_set.save_cache()
    if not val_cache.exists():
        print("Building val cache...")
        val_set.save_cache()
    if not test_cache.exists():
        print("Building test cache...")
        test_set.save_cache()

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

    y_train = train_df["y_true"].to_numpy()
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(DEVICE)
    print(f"Train positives={int(pos)} negatives={int(neg)} pos_weight={pos_weight.item():.3f}")

    model = GRUDModel(
        input_size=len(feature_cols),
        hidden_size=HIDDEN_SIZE,
        x_mean=emp_mean,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_pr = -1.0
    best_path = CACHE_DIR / "best_grud_48h.pt"

    print("\nTraining GRU-D (48h)...\n")
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        tr_roc, tr_pr = evaluate(model, train_loader)
        va_roc, va_pr = evaluate(model, val_loader)

        print(
            f"epoch {epoch:02d} | loss {loss:.4f} | "
            f"train ROC {tr_roc:.3f} PR {tr_pr:.3f} | "
            f"val ROC {va_roc:.3f} PR {va_pr:.3f}"
        )

        if va_pr > best_val_pr:
            best_val_pr = va_pr
            torch.save({"model": model.state_dict(), "feature_cols": feature_cols}, best_path)

    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])

    va_roc, va_pr = evaluate(model, val_loader)
    te_roc, te_pr = evaluate(model, test_loader)

    print("\nBest checkpoint:", best_path)
    print(f"Final (best) val  | ROC-AUC: {va_roc:.3f} | PR-AUC: {va_pr:.3f}")
    print(f"Final (best) test | ROC-AUC: {te_roc:.3f} | PR-AUC: {te_pr:.3f}\n")


if __name__ == "__main__":
    main()