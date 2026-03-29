from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# Config
# ============================================================
DATA_DIR = Path("data") / "in-hospital-mortality-cleaned"

# Listfiles still come from the benchmark folder
LISTFILE_BASE = Path("data") / "in-hospital-mortality"

LISTFILES = {
    "train": LISTFILE_BASE / "train_listfile.csv",
    "val": LISTFILE_BASE / "val_listfile.csv",
    "test": LISTFILE_BASE / "test_listfile.csv",
}

# Validation matrices live in train_matrices
MATRIX_DIR = {
    "train": DATA_DIR / "train_matrices",
    "val": DATA_DIR / "train_matrices",
    "test": DATA_DIR / "test_matrices",
}

# Use a fresh cache name since sample contents changed
CACHE_DIR = DATA_DIR / "cache_grud_cleaned_48h"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_HOURS = 48.0
DT_HOURS = 1.0
T_STEPS = int(MAX_HOURS / DT_HOURS)  # 48

BATCH_SIZE = 64
EPOCHS = 40
PATIENCE = 6
LR = 5e-4
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 256
DROPOUT = 0.2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Utilities
# ============================================================
def read_listfile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "stay" not in df.columns:
        df.rename(columns={df.columns[0]: "stay"}, inplace=True)
    if "y_true" not in df.columns:
        df.rename(columns={df.columns[1]: "y_true"}, inplace=True)

    df["stay"] = df["stay"].astype(str)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)

    return df[["stay", "y_true"]]


def resolve_matrix_paths(matrix_dir: Path, stay: str) -> Tuple[Path, Path]:
    stay = str(stay).replace(".csv", "").strip()

    x_path = matrix_dir / f"{stay}_data.npy"
    m_path = matrix_dir / f"{stay}_mask.npy"

    if not x_path.exists():
        raise FileNotFoundError(f"Missing data matrix: {x_path}")
    if not m_path.exists():
        raise FileNotFoundError(f"Missing mask matrix: {m_path}")

    return x_path, m_path


def infer_input_size(matrix_dir: Path) -> int:
    sample_file = next(matrix_dir.glob("*_data.npy"))
    x = np.load(sample_file)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix [T, D], got shape {x.shape} for {sample_file}")
    return int(x.shape[1])


def compute_train_stats_from_matrices(
    train_list: pd.DataFrame,
    matrix_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute train-only per-feature mean/std using observed values only.
    """
    sums = None
    sums_sq = None
    counts = None

    for stay in train_list["stay"].tolist():
        x_path, m_path = resolve_matrix_paths(matrix_dir, stay)
        x = np.load(x_path).astype(np.float32)   # [T, D]
        m = np.load(m_path).astype(np.float32)   # [T, D]

        if x.shape != m.shape:
            raise ValueError(f"Shape mismatch for stay={stay}: x{x.shape} vs m{m.shape}")

        if sums is None:
            d = x.shape[1]
            sums = np.zeros(d, dtype=np.float64)
            sums_sq = np.zeros(d, dtype=np.float64)
            counts = np.zeros(d, dtype=np.float64)

        observed = (m == 1.0)
        sums += (x * observed).sum(axis=0)
        sums_sq += ((x ** 2) * observed).sum(axis=0)
        counts += observed.sum(axis=0)

    mean = sums / np.maximum(counts, 1.0)
    var = sums_sq / np.maximum(counts, 1.0) - mean ** 2
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)

    mean[counts == 0] = 0.0
    std[counts == 0] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


def compute_empirical_mean_normalized(input_size: int) -> np.ndarray:
    """
    After z-score normalization, the empirical feature mean is approximately 0.
    GRU-D decays toward this normalized mean.
    """
    return np.zeros(input_size, dtype=np.float32)


def compute_deltas(mask: np.ndarray, dt_hours: float) -> np.ndarray:
    """
    mask: [T, D] with 1 if observed else 0
    delta[t, d] = time since last observation of feature d at time t
    """
    t_steps, d_feats = mask.shape
    delta = np.zeros((t_steps, d_feats), dtype=np.float32)

    delta[0] = (1.0 - mask[0]) * dt_hours

    for t in range(1, t_steps):
        delta[t] = dt_hours + (1.0 - mask[t]) * delta[t - 1]
        delta[t] = mask[t] * dt_hours + (1.0 - mask[t]) * delta[t]

    return delta


def compute_last_observed(x_norm_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    x_norm_obs: [T, D] normalized observed values, NaN where missing
    mask: [T, D]

    Returns forward-carried last observed normalized value.
    """
    t_steps, d_feats = x_norm_obs.shape
    x_last = np.zeros((t_steps, d_feats), dtype=np.float32)

    prev = np.zeros(d_feats, dtype=np.float32)
    for t in range(t_steps):
        current_obs = np.where(mask[t] == 1.0, x_norm_obs[t], prev)
        prev = current_obs.astype(np.float32)
        x_last[t] = prev

    return x_last


# ============================================================
# Dataset
# ============================================================
@dataclass
class Sample:
    x: torch.Tensor       # [T, D] normalized observed values, 0 where missing
    x_last: torch.Tensor  # [T, D] forward-carried last observed normalized value
    m: torch.Tensor       # [T, D]
    d: torch.Tensor       # [T, D]
    y: torch.Tensor       # [1]


class IHM48hDataset(Dataset):
    def __init__(
        self,
        listfile: pd.DataFrame,
        matrix_dir: Path,
        mean: np.ndarray,
        std: np.ndarray,
        emp_mean: np.ndarray,
        cache_path: Optional[Path] = None,
        split: Optional[str] = None,
    ):
        self.df = listfile.reset_index(drop=True)
        self.matrix_dir = matrix_dir
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.emp_mean = emp_mean.astype(np.float32)
        self.cache_path = cache_path
        self.split = split if split is not None else "train"
        self._cache: Optional[Dict[int, Sample]] = None

        if self.cache_path is not None and self.cache_path.exists():
            self._cache = torch.load(self.cache_path, weights_only=False)

    def __len__(self) -> int:
        return len(self.df)

    def _load_one(self, idx: int) -> Sample:
        row = self.df.iloc[idx]
        stay = str(row["stay"])
        y = float(row["y_true"])

        x_path, m_path = resolve_matrix_paths(self.matrix_dir, stay)

        x_raw = np.load(x_path).astype(np.float32)   # [T, D]
        m = np.load(m_path).astype(np.float32)       # [T, D]

        if x_raw.shape != m.shape:
            raise ValueError(f"Shape mismatch for stay={stay}: x{x_raw.shape} vs m{m.shape}")
        if x_raw.ndim != 2:
            raise ValueError(f"Expected x shape [T, D], got {x_raw.shape} for stay={stay}")
        if x_raw.shape[0] != T_STEPS:
            raise ValueError(f"Expected {T_STEPS} time steps, got {x_raw.shape[0]} for stay={stay}")

        # Normalize observed values only
        x_norm_obs = np.where(
            m == 1.0,
            (x_raw - self.mean.reshape(1, -1)) / self.std.reshape(1, -1),
            np.nan,
        ).astype(np.float32)

        # Proper GRU-D style decay source: last observed normalized value
        x_last = compute_last_observed(x_norm_obs, m)

        # Main input uses 0 where missing, mask tracks true missingness
        x = np.nan_to_num(x_norm_obs, nan=0.0).astype(np.float32)

        dlt = compute_deltas(m, DT_HOURS).astype(np.float32)

        return Sample(
            x=torch.from_numpy(x),
            x_last=torch.from_numpy(x_last),
            m=torch.from_numpy(m),
            d=torch.from_numpy(dlt),
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


def collate_fn(
    batch: List[Sample],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.stack([b.x for b in batch], dim=0)           # [B, T, D]
    x_last = torch.stack([b.x_last for b in batch], dim=0) # [B, T, D]
    m = torch.stack([b.m for b in batch], dim=0)           # [B, T, D]
    d = torch.stack([b.d for b in batch], dim=0)           # [B, T, D]
    y = torch.cat([b.y for b in batch], dim=0)             # [B]
    return x, x_last, m, d, y


# ============================================================
# GRU-D
# ============================================================
class GRUDCell(nn.Module):
    """
    Revised GRU-D cell for cleaned pipeline:
    - decays missing features from last observed value toward empirical mean
    - explicitly includes mask in gates
    """
    def __init__(self, input_size: int, hidden_size: int, x_mean: np.ndarray, dropout: float = 0.0):
        super().__init__()
        self.D = input_size
        self.H = hidden_size

        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32))  # [D]

        # feature-wise decay for x
        self.Wx = nn.Parameter(torch.empty(self.D))
        self.bx = nn.Parameter(torch.zeros(self.D))

        # hidden decay
        self.Wh = nn.Linear(self.D, hidden_size, bias=True)

        # Include mask in gates
        gate_in = self.D + self.D + hidden_size
        self.z = nn.Linear(gate_in, hidden_size)
        self.r = nn.Linear(gate_in, hidden_size)
        self.h_tilde = nn.Linear(gate_in, hidden_size)

        self.dropout = nn.Dropout(dropout)

        nn.init.uniform_(self.Wx, a=-0.05, b=0.05)
        nn.init.xavier_uniform_(self.Wh.weight)
        nn.init.zeros_(self.Wh.bias)

    def forward(self, x_t, x_last_t, m_t, d_t, h_prev):
        gamma_x = torch.exp(-torch.relu(d_t * self.Wx + self.bx))  # [B, D]

        # Decay missing values from last observed value toward empirical mean
        x_hat = m_t * x_t + (1.0 - m_t) * (gamma_x * x_last_t + (1.0 - gamma_x) * self.x_mean)

        gamma_h = torch.exp(-torch.relu(self.Wh(d_t)))  # [B, H]
        h_prev = gamma_h * h_prev

        inp = torch.cat([x_hat, m_t, h_prev], dim=-1)
        z = torch.sigmoid(self.z(inp))
        r = torch.sigmoid(self.r(inp))

        inp_r = torch.cat([x_hat, m_t, r * h_prev], dim=-1)
        h_tilde = torch.tanh(self.h_tilde(inp_r))

        h = (1.0 - z) * h_prev + z * h_tilde
        h = self.dropout(h)
        return h


class GRUDModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, x_mean: np.ndarray, dropout: float = 0.0):
        super().__init__()
        self.cell = GRUDCell(input_size, hidden_size, x_mean=x_mean, dropout=dropout)

        # Add summary missingness signal to classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, x_last, m, d):
        bsz, t_steps, d_feats = x.shape
        h = torch.zeros((bsz, self.cell.H), device=x.device)

        for t in range(t_steps):
            h = self.cell(x[:, t], x_last[:, t], m[:, t], d[:, t], h)

        m_summary = m.mean(dim=1)  # [B, D]
        logits = self.classifier(torch.cat([h, m_summary], dim=-1)).squeeze(-1)
        return logits


# ============================================================
# Train / Eval
# ============================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    ys, ps = [], []

    for x, x_last, m, d, y in loader:
        x = x.to(DEVICE)
        x_last = x_last.to(DEVICE)
        m = m.to(DEVICE)
        d = d.to(DEVICE)

        logits = model(x, x_last, m, d)
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

    for x, x_last, m, d, y in loader:
        x = x.to(DEVICE)
        x_last = x_last.to(DEVICE)
        m = m.to(DEVICE)
        d = d.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x, x_last, m, d)
        loss = criterion(logits, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * y.shape[0]
        n += y.shape[0]

    return total_loss / max(n, 1)


def main():
    print(f"Using device: {DEVICE}")
    print(f"Cleaned data dir: {DATA_DIR}")

    train_df = read_listfile(LISTFILES["train"])
    val_df = read_listfile(LISTFILES["val"])
    test_df = read_listfile(LISTFILES["test"])

    input_size = infer_input_size(MATRIX_DIR["train"])
    print(f"Input dimension for GRU-D: {input_size}")

    mean, std = compute_train_stats_from_matrices(train_df, MATRIX_DIR["train"])
    emp_mean = compute_empirical_mean_normalized(input_size)

    print(f"Computed train normalization stats for {input_size} features")

    train_cache = CACHE_DIR / "train.pt"
    val_cache = CACHE_DIR / "val.pt"
    test_cache = CACHE_DIR / "test.pt"

    train_set = IHM48hDataset(
        listfile=train_df,
        matrix_dir=MATRIX_DIR["train"],
        mean=mean,
        std=std,
        emp_mean=emp_mean,
        cache_path=train_cache,
        split="train",
    )
    val_set = IHM48hDataset(
        listfile=val_df,
        matrix_dir=MATRIX_DIR["val"],
        mean=mean,
        std=std,
        emp_mean=emp_mean,
        cache_path=val_cache,
        split="val",
    )
    test_set = IHM48hDataset(
        listfile=test_df,
        matrix_dir=MATRIX_DIR["test"],
        mean=mean,
        std=std,
        emp_mean=emp_mean,
        cache_path=test_cache,
        split="test",
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

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    y_train = train_df["y_true"].to_numpy()
    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32).to(DEVICE)
    print(f"Train positives={int(pos)} negatives={int(neg)} pos_weight={pos_weight.item():.3f}")

    model = GRUDModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        x_mean=emp_mean,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_pr = -1.0
    best_path = CACHE_DIR / "best_grud_cleaned_48h_v2.pt"
    no_improve = 0

    print("\nTraining revised GRU-D on cleaned 48h matrices...\n")
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion)
        tr_roc, tr_pr = evaluate(model, train_loader)
        va_roc, va_pr = evaluate(model, val_loader)

        scheduler.step(va_pr)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"epoch {epoch:02d} | lr {current_lr:.6f} | loss {loss:.4f} | "
            f"train ROC {tr_roc:.3f} PR {tr_pr:.3f} | "
            f"val ROC {va_roc:.3f} PR {va_pr:.3f}"
        )

        if va_pr > best_val_pr:
            best_val_pr = va_pr
            no_improve = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "input_size": input_size,
                    "emp_mean": torch.tensor(emp_mean, dtype=torch.float32),
                    "mean": torch.tensor(mean, dtype=torch.float32),
                    "std": torch.tensor(std, dtype=torch.float32),
                },
                best_path,
            )
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])

    va_roc, va_pr = evaluate(model, val_loader)
    te_roc, te_pr = evaluate(model, test_loader)

    print("\nBest checkpoint:", best_path)
    print(f"Final (best) val  | ROC-AUC: {va_roc:.3f} | PR-AUC: {va_pr:.3f}")
    print(f"Final (best) test | ROC-AUC: {te_roc:.3f} | PR-AUC: {te_pr:.3f}\n")


if __name__ == "__main__":
    main()
