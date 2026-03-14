# Multi-Branch ICU Mortality Pipeline (Incremental)

This package is an incremental implementation of a grouped architecture:

- Neurology (GCS) -> GRU -> embedding 64
- Cardiology (HR, BP) -> 1D CNN -> embedding 64
- Respiratory (SpO2, RR, FiO2) -> GRU + Attention -> embedding 64
- Metabolic (Glucose, pH, Temp) -> GRU-D encoder (value/mask/delta) -> embedding 32
- Static (Height, Weight) -> MLP -> embedding 16
- Concatenate all embeddings -> Fusion MLP -> mortality logit

## Data compatibility

Data loading follows the legacy style from `old_pipleline/readers.py`:

- `train_listfile.csv`, `val_listfile.csv`, `test_listfile.csv`
- Time-series CSV files under `train/` and `test/`

## Incremental workflow

1. Prepare preprocessing stats and verify tensors:

```powershell
python -m multibranch_pipeline.train_incremental --data in-hospital-mortality --stage prepare
```

Print one end-to-end input usage example (raw rows -> branch tensors):

```powershell
python -m multibranch_pipeline.train_incremental --data in-hospital-mortality --stage prepare --print_input_demo --demo_index 0 --demo_rows 6
```

2. Train first baseline end-to-end model:

```powershell
python -m multibranch_pipeline.train_incremental --data in-hospital-mortality --stage train --epochs 10
```

## What is implemented in this increment

- Group-specific preprocessing rules:
  - Neurology: category cleaning + encoding
  - Cardiology: outlier filtering + forward fill
  - Respiratory: unit normalization + smoothing
  - Metabolic: mask + time-gap features
  - Static: median imputation + normalization
- Multi-branch PyTorch network + fusion head
- Train/validation loop with legacy-style mortality metrics:
  - confusion matrix
  - accuracy, precision/recall (class-wise)
  - AUROC, AUPRC, min(+P, Se)
  - final test evaluation (unless `--skip_test_eval` is set)

Future increments can add richer augmentation and calibration.
