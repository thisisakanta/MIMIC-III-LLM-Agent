import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
from tqdm import tqdm


DATA_ROOT = "data"
CLEAN_DIR = "data/in-hospital-mortality-cleaned"
LIST_DIR = "data/in-hospital-mortality"

OUTPUT_DIR = "data/in-hospital-mortality-7subseq-features"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


# ----------------------------------------
# Define subsequence windows
# ----------------------------------------

SUBSEQS = {
    "full": (0, 48),
    "first10": (0, 5),
    "first25": (0, 12),
    "first50": (0, 24),
    "last50": (24, 48),
    "last25": (36, 48),
    "last10": (43, 48),
}


# ----------------------------------------
# Compute statistics for a sequence
# ----------------------------------------

def compute_stats(values):

    values = values.dropna()

    if len(values) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "skew": np.nan,
            "count": 0,
        }

    return {
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "skew": skew(values),
        "count": len(values),
    }


# ----------------------------------------
# Extract features for one patient
# ----------------------------------------
# 17 variables × 7 subsequences × 6 stats = 714
# 17 variables x 7 subsequences (missing flags) = 119
# 714 (stats)+ 119 (subsequence missing flags)= 833+ label = 834
def extract_patient_features(file_path):

    df = pd.read_csv(file_path)

    df = df.sort_values("Hours")

    variables = [c for c in df.columns if c != "Hours" and not c.endswith("_missing")]

    features = {}

    for var in variables:

        series = df[var]

        for name, (start, end) in SUBSEQS.items():

            subseq = series.iloc[start:end]

            stats = compute_stats(subseq)

            # Add statistical features (UNCHANGED)
            for stat_name, val in stats.items():
                feature_name = f"{var}_{name}_{stat_name}"
                features[feature_name] = val

            # Add missing indicator (NEW)
            missing_flag = int(subseq.dropna().shape[0] == 0)
            features[f"{var}_{name}_missing"] = missing_flag

    return features


# ----------------------------------------
# Process dataset split
# ----------------------------------------

def process_split(split):

    listfile_path = os.path.join(LIST_DIR, f"{split}_listfile.csv")

    listfile = pd.read_csv(listfile_path)

    features = []

    for _, row in tqdm(listfile.iterrows(), total=len(listfile)):

        stay = row["stay"]
        label = row["y_true"]

        if split == "val":
            ts_path = os.path.join(CLEAN_DIR, "train", stay)
        else:
            ts_path = os.path.join(CLEAN_DIR, split, stay)

        feat = extract_patient_features(ts_path)

        feat["label"] = label

        features.append(feat)

    df = pd.DataFrame(features)

    out_path = os.path.join(OUTPUT_DIR, f"{split}_features.csv")

    df.to_csv(out_path, index=False)

    print(f"Saved {out_path}")


# ----------------------------------------
# Run
# ----------------------------------------

if __name__ == "__main__":

    for split in ["train", "val", "test"]:
        process_split(split)

    print("Feature extraction complete.")
