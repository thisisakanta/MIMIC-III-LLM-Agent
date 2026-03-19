import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score

TASK_DIR = "data/in-hospital-mortality"


def read_timeseries(split: str, fname: str) -> pd.DataFrame:
    # val_listfile points to files in train/
    folder = "train" if split == "val" else split
    ts_path = os.path.join(TASK_DIR, folder, fname)
    df = pd.read_csv(ts_path)
    if "Hours" in df.columns:
        df = df.drop(columns=["Hours"])
    return df


def load_listfile(split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(TASK_DIR, f"{split}_listfile.csv"))


def infer_feature_cols_from_one_file() -> list[str]:
    lf = load_listfile("train")
    df = read_timeseries("train", lf.iloc[0]["stay"])
    return list(df.columns)


def encode_df_raw(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """
    Minimal encoding:
    - numeric columns -> float
    - string columns -> NaN (coerce)
    - NaNs -> 0 (so the model can run)
    """
    mats = []
    for c in cols:
        mats.append(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float32))
    x = np.stack(mats, axis=1)  # (T, F)
    x = np.nan_to_num(x, nan=0.0)
    return x


class MortalityDataset(Dataset):
    def __init__(self, split: str, cols: list[str], max_len: int = 200):
        self.split = split
        self.cols = cols
        self.max_len = max_len
        self.listfile = load_listfile(split)

    def __len__(self):
        return len(self.listfile)

    def __getitem__(self, idx):
        row = self.listfile.iloc[idx]
        fname = row["stay"]
        y = float(row["y_true"])

        df = read_timeseries(self.split, fname)
        x = encode_df_raw(df, self.cols)

        # truncate
        if x.shape[0] > self.max_len:
            x = x[: self.max_len, :]

        length = x.shape[0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(length, dtype=torch.long), torch.tensor(y, dtype=torch.float32)


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


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]  # (B, hidden_dim)
        logits = self.head(h_last).squeeze(1)
        return logits


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)
        probs = torch.sigmoid(logits)
        ys.append(y.cpu().numpy())
        ps.append(probs.cpu().numpy())
    y_true = np.concatenate(ys)
    p = np.concatenate(ps)
    return roc_auc_score(y_true, p), average_precision_score(y_true, p)


def main():
    device = torch.device("cpu")

    cols = infer_feature_cols_from_one_file()
    print("Num raw features:", len(cols))

    max_len = 200
    train_ds = MortalityDataset("train", cols, max_len=max_len)
    val_ds = MortalityDataset("val", cols, max_len=max_len)
    test_ds = MortalityDataset("test", cols, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # imbalance handling
    train_lf = load_listfile("train")
    y_train = train_lf["y_true"].astype(int).to_numpy()
    pos = y_train.sum()
    neg = len(y_train) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32).to(device)
    print("pos_weight:", float(pos_weight.item()))

    model = LSTMClassifier(input_dim=len(cols), hidden_dim=64, num_layers=1, dropout=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_pr = -1
    best_state = None

    for epoch in range(1, 6):
        model.train()
        losses = []
        for x, lengths, y in tqdm(train_loader, desc=f"Epoch {epoch}/5"):
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

    if best_state is not None:
        model.load_state_dict(best_state)

    test_roc, test_pr = evaluate(model, test_loader, device)
    print("TEST ROC-AUC:", test_roc)
    print("TEST PR-AUC :", test_pr)


if __name__ == "__main__":
    main()