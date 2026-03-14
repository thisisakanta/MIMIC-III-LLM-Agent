import argparse
import os
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import build_grid_records, summarize_split
from .metrics import print_metrics_binary
from .model import MultiBranchMortalityModel
from .preprocessing import GroupPreprocessor


class MortalityDataset(Dataset):
    def __init__(self, transformed: List[Dict]):
        self.items = transformed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x = self.items[idx]
        return (
            torch.from_numpy(x["neuro"]),
            torch.from_numpy(x["cardio"]),
            torch.from_numpy(x["resp"]),
            torch.from_numpy(x["meta"]),
            torch.from_numpy(x["static"]),
            torch.tensor(x["y"], dtype=torch.float32),
        )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str):
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(items: List[Dict], batch_size: int, shuffle: bool) -> DataLoader:
    ds = MortalityDataset(items)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    for neuro, cardio, resp, meta, static, y in loader:
        neuro = neuro.to(device)
        cardio = cardio.to(device)
        resp = resp.to(device)
        meta = meta.to(device)
        static = static.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(neuro, cardio, resp, meta, static)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_count += bs

    return total_loss / max(total_count, 1)


def evaluate(model, loader, criterion, device, verbose_metrics: int = 0):
    model.eval()
    total_loss = 0.0
    total_count = 0
    ys = []
    ps = []
    with torch.no_grad():
        for neuro, cardio, resp, meta, static, y in loader:
            neuro = neuro.to(device)
            cardio = cardio.to(device)
            resp = resp.to(device)
            meta = meta.to(device)
            static = static.to(device)
            y = y.to(device)

            logits = model(neuro, cardio, resp, meta, static)
            loss = criterion(logits, y)
            prob = torch.sigmoid(logits)

            bs = y.shape[0]
            total_loss += float(loss.item()) * bs
            total_count += bs

            ys.append(y.detach().cpu().numpy())
            ps.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.float32)
    y_prob = np.concatenate(ps) if ps else np.array([], dtype=np.float32)

    metrics_dict = {
        "acc": float("nan"),
        "prec0": float("nan"),
        "prec1": float("nan"),
        "rec0": float("nan"),
        "rec1": float("nan"),
        "auroc": float("nan"),
        "auprc": float("nan"),
        "minpse": float("nan"),
    }
    if y_true.size > 0:
        metrics_dict = print_metrics_binary(y_true, y_prob, verbose=verbose_metrics)

    return total_loss / max(total_count, 1), metrics_dict


def print_input_usage_demo(records, preprocessor, sample_idx: int = 0, print_rows: int = 5):
    if not records:
        print("No records available to print input usage demo.")
        return

    idx = max(0, min(sample_idx, len(records) - 1))
    rec = records[idx]
    transformed = preprocessor.transform(rec)

    print("\n=== INPUT USAGE DEMO ===")
    print(f"sample_index: {idx}")
    print(f"stay: {rec.name}")
    print(f"label (y): {rec.y}")

    cols_to_show = [
        "Capillary refill rate",
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale total",
        "Glascow coma scale verbal response",
        "Heart Rate",
        "Diastolic blood pressure",
        "Mean blood pressure",
        "Systolic blood pressure",
        "Oxygen saturation",
        "Respiratory rate",
        "Fraction inspired oxygen",
        "Glucose",
        "pH",
        "Temperature",
        "Height",
        "Weight",
    ]
    show_cols = [c for c in cols_to_show if c in rec.frame.columns]
    print("\nRaw hourly-grid input (first rows):")
    print(rec.frame[show_cols].head(print_rows).to_string())

    print("\nBranch tensors used by the model:")
    print(f"  neuro shape : {transformed['neuro'].shape}")
    print(f"  cardio shape: {transformed['cardio'].shape}")
    print(f"  resp shape  : {transformed['resp'].shape}")
    print(f"  meta shape  : {transformed['meta'].shape}")
    print(f"  static shape: {transformed['static'].shape}")

    print("\nNeuro first rows:")
    print(np.array2string(transformed["neuro"][:print_rows], precision=3, suppress_small=True))

    print("\nCardio first rows:")
    print(np.array2string(transformed["cardio"][:print_rows], precision=3, suppress_small=True))

    print("\nResp first rows:")
    print(np.array2string(transformed["resp"][:print_rows], precision=3, suppress_small=True))

    print("\nMeta first rows (value|mask|delta):")
    print(np.array2string(transformed["meta"][:print_rows], precision=3, suppress_small=True))

    print("\nStatic vector:")
    print(np.array2string(transformed["static"], precision=3, suppress_small=True))


def parse_args():
    parser = argparse.ArgumentParser(description="Incremental multi-branch mortality training")
    parser.add_argument("--data", type=str, required=True, help="Path to in-hospital-mortality folder")
    parser.add_argument("--stage", type=str, default="prepare", choices=["prepare", "train"])
    parser.add_argument("--output_dir", type=str, default="multibranch_output")
    parser.add_argument("--stats_file", type=str, default="preprocessor_state.json")

    parser.add_argument("--max_hours", type=float, default=48.0)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--max_patients", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip_test_eval", action="store_true", help="Skip final test split evaluation")
    parser.add_argument("--print_input_demo", action="store_true", help="Print one raw-to-processed sample flow")
    parser.add_argument("--demo_index", type=int, default=0, help="Sample index used for --print_input_demo")
    parser.add_argument("--demo_rows", type=int, default=5, help="Rows printed in --print_input_demo")
    return parser.parse_args()


def _load_and_transform(data_root, split, pre, max_hours, timestep, max_patients):
    records = build_grid_records(
        data_root=data_root,
        split=split,
        max_hours=max_hours,
        timestep=timestep,
        max_patients=max_patients,
    )
    transformed = [pre.transform(r) for r in records]
    return records, transformed


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    stats_path = os.path.join(args.output_dir, args.stats_file)

    print("Loading train split...")
    train_records = build_grid_records(
        data_root=args.data,
        split="train",
        max_hours=args.max_hours,
        timestep=args.timestep,
        max_patients=args.max_patients,
    )
    print("Train summary:", summarize_split(train_records))

    pre = GroupPreprocessor(max_hours=args.max_hours, timestep=args.timestep)

    if args.stage == "prepare":
        pre.fit(train_records)
        pre.save_state(stats_path)

        if args.print_input_demo:
            print_input_usage_demo(
                train_records,
                preprocessor=pre,
                sample_idx=args.demo_index,
                print_rows=args.demo_rows,
            )

        sample = pre.transform(train_records[0])
        print("Saved preprocessing state to", stats_path)
        print("Sample tensor shapes:")
        print("  neuro :", sample["neuro"].shape)
        print("  cardio:", sample["cardio"].shape)
        print("  resp  :", sample["resp"].shape)
        print("  meta  :", sample["meta"].shape)
        print("  static:", sample["static"].shape)
        return

    if os.path.exists(stats_path):
        pre.load_state(stats_path)
        print("Loaded preprocessing state:", stats_path)
    else:
        pre.fit(train_records)
        pre.save_state(stats_path)
        print("Fitted and saved preprocessing state:", stats_path)

    if args.print_input_demo:
        print_input_usage_demo(
            train_records,
            preprocessor=pre,
            sample_idx=args.demo_index,
            print_rows=args.demo_rows,
        )

    print("Transforming train/val splits...")
    _, train_items = _load_and_transform(
        args.data, "train", pre, args.max_hours, args.timestep, args.max_patients
    )
    _, val_items = _load_and_transform(
        args.data, "val", pre, args.max_hours, args.timestep, args.max_patients
    )

    train_loader = make_loader(train_items, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(val_items, batch_size=args.batch_size, shuffle=False)

    sample = train_items[0]
    model = MultiBranchMortalityModel(
        neuro_dim=sample["neuro"].shape[1],
        cardio_dim=sample["cardio"].shape[1],
        resp_dim=sample["resp"].shape[1],
        meta_dim=sample["meta"].shape[1],
        static_dim=sample["static"].shape[0],
        dropout=args.dropout,
    ).to(device)

    ys = np.array([x["y"] for x in train_items], dtype=np.float32)
    n_pos = int((ys == 1).sum())
    n_neg = int((ys == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auc = -1.0
    best_state = None

    print("Training on", device)
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, verbose_metrics=0)
        val_auc = float(val_metrics["auroc"])

        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.5f} | val_loss {val_loss:.5f} "
            f"| val_auroc {val_metrics['auroc']:.4f} | val_auprc {val_metrics['auprc']:.4f} "
            f"| val_acc {val_metrics['acc']:.4f}"
        )

        metric = val_auc if not np.isnan(val_auc) else -1.0
        if metric > best_auc:
            best_auc = metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Final validation metrics (best checkpoint):")
    _, _ = evaluate(model, val_loader, criterion, device, verbose_metrics=1)

    if not args.skip_test_eval:
        print("Transforming and evaluating test split...")
        _, test_items = _load_and_transform(
            args.data, "test", pre, args.max_hours, args.timestep, args.max_patients
        )
        if test_items:
            test_loader = make_loader(test_items, batch_size=args.batch_size, shuffle=False)
            print("Final test metrics:")
            _, _ = evaluate(model, test_loader, criterion, device, verbose_metrics=1)
        else:
            print("No test items available for evaluation.")

    ckpt = os.path.join(args.output_dir, "multibranch_model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "max_hours": args.max_hours,
                "timestep": args.timestep,
            },
        },
        ckpt,
    )
    print("Saved model checkpoint:", ckpt)


if __name__ == "__main__":
    main()
