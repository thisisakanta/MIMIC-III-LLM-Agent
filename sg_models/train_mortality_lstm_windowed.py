import os
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
TRAIN_MATRIX_DIR = "data/in-hospital-mortality-cleaned/train_matrices"
TEST_MATRIX_DIR = "data/in-hospital-mortality-cleaned/test_matrices"

TRAIN_LISTFILE = "data/in-hospital-mortality/train_listfile.csv"
VAL_LISTFILE = "data/in-hospital-mortality/val_listfile.csv"
TEST_LISTFILE = "data/in-hospital-mortality/test_listfile.csv"

BATCH_TRAIN = 32
BATCH_EVAL = 64
EPOCHS = 20
PATIENCE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN = 64
DROPOUT = 0.3
NUM_WORKERS = 0  # Mac-safe


# -----------------------------
# Helpers
# -----------------------------
def get_matrix_paths(matrix_dir: str, stay_name: str) -> tuple[str, str]:
    base = stay_name.replace(".csv", "")
    data_path = os.path.join(matrix_dir, base + "_data.npy")
    mask_path = os.path.join(matrix_dir, base + "_mask.npy")
    return data_path, mask_path


def compute_train_stats_from_observed(matrix_dir: str, listfile_path: str):
    """
    Compute per-feature mean/std using ONLY observed values from train.
    Missing entries are excluded using the mask.
    """
    lf = pd.read_csv(listfile_path)

    sum_x = None
    sum_x2 = None
    count_x = None

    for _, row in tqdm(lf.iterrows(), total=len(lf), desc="Computing train stats"):
        stay = row["stay"]
        data_path, mask_path = get_matrix_paths(matrix_dir, stay)

        if not (os.path.exists(data_path) and os.path.exists(mask_path)):
            continue

        X = np.load(data_path).astype(np.float32)   # (T, F)
        M = np.load(mask_path).astype(np.float32)   # (T, F), 1=observed

        if sum_x is None:
            F = X.shape[1]
            sum_x = np.zeros(F, dtype=np.float64)
            sum_x2 = np.zeros(F, dtype=np.float64)
            count_x = np.zeros(F, dtype=np.float64)

        # Only observed values contribute to stats
        sum_x += (X * M).sum(axis=0)
        sum_x2 += ((X ** 2) * M).sum(axis=0)
        count_x += M.sum(axis=0)

    count_x = np.maximum(count_x, 1.0)
    mean = sum_x / count_x
    var = (sum_x2 / count_x) - (mean ** 2)
    var = np.maximum(var, 1e-6)
    std = np.sqrt(var)

    return mean.astype(np.float32), std.astype(np.float32)


# -----------------------------
# Dataset
# -----------------------------
class MortalityNPYDataset(Dataset):
    def __init__(self, matrix_dir: str, listfile_path: str, mean: np.ndarray, std: np.ndarray):
        self.matrix_dir = matrix_dir
        self.listfile = pd.read_csv(listfile_path)
        self.mean = mean
        self.std = std

        self.samples = []
        for _, row in self.listfile.iterrows():
            stay = row["stay"]
            y = float(row["y_true"])

            data_path, mask_path = get_matrix_paths(matrix_dir, stay)
            if os.path.exists(data_path) and os.path.exists(mask_path):
                self.samples.append((data_path, mask_path, y))

        print(f"Loaded {len(self.samples)} samples from {matrix_dir} using {os.path.basename(listfile_path)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data_path, mask_path, y = self.samples[idx]

        X = np.load(data_path).astype(np.float32)   # (48, F)
        M = np.load(mask_path).astype(np.float32)   # (48, F)

        # Normalize with train-only stats
        X_norm = (X - self.mean[None, :]) / self.std[None, :]

        # Set missing entries back to 0 after normalization
        X_norm[M == 0] = 0.0

        # Optional clipping for stability
        X_norm = np.clip(X_norm, -5.0, 5.0)

        # Concatenate mask as extra channels
        X_final = np.concatenate([X_norm, M], axis=1)  # (48, 2F)

        length = X_final.shape[0]

        return (
            torch.tensor(X_final, dtype=torch.float32),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )


# -----------------------------
# Collate
# -----------------------------
def collate_fn(batch):
    xs, lens, ys = zip(*batch)

    lengths = torch.stack(lens)
    y = torch.stack(ys)

    T_max = max(x.shape[0] for x in xs)
    F = xs[0].shape[1]

    x_pad = torch.zeros(len(xs), T_max, F, dtype=torch.float32)
    for i, x in enumerate(xs):
        x_pad[i, : x.shape[0], :] = x

    return x_pad, lengths, y


# -----------------------------
# Model
# -----------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]  # (B, H)
        h_last = self.dropout(h_last)
        return self.head(h_last).squeeze(1)


# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []

    for x, lengths, y in loader:
        x = x.to(device)
        lengths = lengths.to(device)

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

    # Train-only stats
    mean, std = compute_train_stats_from_observed(TRAIN_MATRIX_DIR, TRAIN_LISTFILE)

    # Official benchmark splits
    train_ds = MortalityNPYDataset(TRAIN_MATRIX_DIR, TRAIN_LISTFILE, mean, std)
    val_ds = MortalityNPYDataset(TRAIN_MATRIX_DIR, VAL_LISTFILE, mean, std)
    test_ds = MortalityNPYDataset(TEST_MATRIX_DIR, TEST_LISTFILE, mean, std)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_EVAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    sample_x, _, _ = train_ds[0]
    input_dim = sample_x.shape[1]
    print("Input dimension:", input_dim)

    # pos_weight from official train only
    y_train = pd.read_csv(TRAIN_LISTFILE)["y_true"].astype(int).to_numpy()
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)
    print("pos_weight:", float(pos_weight.item()))

    model = LSTMClassifier(input_dim=input_dim, hidden_dim=HIDDEN, dropout=DROPOUT).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_pr = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []

        for x, lengths, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x = x.to(device)
            lengths = lengths.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            losses.append(loss.item())

        val_roc, val_pr = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: loss={np.mean(losses):.4f}  VAL_ROC={val_roc:.4f}  VAL_PR={val_pr:.4f}")

        if val_pr > best_val_pr:
            best_val_pr = val_pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_roc, test_pr = evaluate(model, test_loader, device)
    print("\nFINAL RESULTS")
    print("TEST ROC-AUC:", test_roc)
    print("TEST PR-AUC :", test_pr)


if __name__ == "__main__":
    main()
