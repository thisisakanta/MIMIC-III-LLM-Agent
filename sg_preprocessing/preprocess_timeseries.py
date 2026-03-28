import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

INPUT_DIR = "data/in-hospital-mortality"
OUTPUT_DIR = "data/in-hospital-mortality-cleaned"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

############################################################
# CLEANING FUNCTION MAP
############################################################

CLEAN_FNS = {
    "Capillary refill rate": "clean_crr",
    "Diastolic blood pressure": "clean_dbp",
    "Systolic blood pressure": "clean_sbp",
    "Fraction inspired oxygen": "clean_fio2",
    "Oxygen saturation": "clean_o2sat",
    "Glucose": "clean_lab",
    "pH": "clean_lab",
    "Temperature": "clean_temperature",
    "Weight": "clean_weight",
    "Height": "clean_height",
    "Glasgow coma scale eye opening": "clean_gcs_eye",
    "Glasgow coma scale motor response": "clean_gcs_motor",
    "Glasgow coma scale verbal response": "clean_gcs_verbal"
}

############################################################
# CLEANING FUNCTIONS
############################################################

def clean_sbp(series):
    def extract(v):
        if isinstance(v, str) and "/" in v:
            return float(v.split("/")[0])
        return v
    return pd.to_numeric(series.apply(extract), errors="coerce")


def clean_dbp(series):
    def extract(v):
        if isinstance(v, str) and "/" in v:
            return float(v.split("/")[1])
        return v
    return pd.to_numeric(series.apply(extract), errors="coerce")


def clean_crr(series):
    s = series.astype(str).str.strip()
    mapping = {
        "Normal <3 secs": 0,
        "Brisk": 0,
        "Abnormal >3 secs": 1,
        "Delayed": 1
    }
    mapped = s.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.combine_first(numeric)


def clean_fio2(series):
    v = pd.to_numeric(series, errors="coerce")
    idx = v > 1.0
    v.loc[idx] = v.loc[idx] / 100.0
    return v


def clean_lab(series):
    return pd.to_numeric(series, errors="coerce")


def clean_o2sat(series):
    v = pd.to_numeric(series, errors="coerce")
    idx = v <= 1.0
    v.loc[idx] = v.loc[idx] * 100.0
    return v


def clean_temperature(series):
    v = pd.to_numeric(series, errors="coerce")
    idx = v >= 79
    v.loc[idx] = (v.loc[idx] - 32) * 5.0 / 9.0
    return v


def clean_weight(series):
    v = pd.to_numeric(series, errors="coerce")
    idx = v > 250
    v.loc[idx] = v.loc[idx] * 0.453592
    return v


def clean_height(series):
    v = pd.to_numeric(series, errors="coerce")
    idx = (v > 0) & (v < 3)
    v.loc[idx] = v.loc[idx] * 100.0
    return v


def clean_gcs_eye(series):
    s = series.astype(str).str.strip()
    mapping = {
        "Spontaneously": 4,
        "To Speech": 3,
        "To Pain": 2,
        "None": 1
    }
    mapped = s.map(mapping)
    numeric_prefix = pd.to_numeric(s.str.extract(r"^(\d+)")[0], errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.combine_first(numeric_prefix).combine_first(numeric)


def clean_gcs_motor(series):
    s = series.astype(str).str.strip()
    mapping = {
        "Obeys Commands": 6,
        "Localizes Pain": 5,
        "Withdraws": 4,
        "Flexion": 3,
        "Extension": 2,
        "None": 1
    }
    mapped = s.map(mapping)
    numeric_prefix = pd.to_numeric(s.str.extract(r"^(\d+)")[0], errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.combine_first(numeric_prefix).combine_first(numeric)


def clean_gcs_verbal(series):
    s = series.astype(str).str.strip()
    mapping = {
        "Oriented": 5,
        "Confused": 4,
        "Inappropriate Words": 3,
        "Incomprehensible": 2,
        "None": 1
    }
    mapped = s.map(mapping)
    numeric_prefix = pd.to_numeric(s.str.extract(r"^(\d+)")[0], errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.combine_first(numeric_prefix).combine_first(numeric)

############################################################
# APPLY CLEANING
############################################################

def clean_dataframe(df):

    for col in df.columns:
        if col == "Hours":
            continue

        if col in CLEAN_FNS:
            df[col] = globals()[CLEAN_FNS[col]](df[col])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

############################################################
# LOAD VARIABLE RANGES & CLIPPING
############################################################

def load_variable_ranges(path):
    df = pd.read_csv(path)

    df = df.rename(columns={
        "LEVEL2": "VARIABLE",
        "OUTLIER LOW": "OUTLIER_LOW",
        "VALID LOW": "VALID_LOW",
        "VALID HIGH": "VALID_HIGH",
        "OUTLIER HIGH": "OUTLIER_HIGH"
    })

    df = df.set_index("VARIABLE")

    return df

def clip_variable(series, var_name, ranges):

    if var_name not in ranges.index:
        return series

    r = ranges.loc[var_name]

    v = series.copy()

    # remove extreme outliers
    v[v < r.OUTLIER_LOW] = np.nan
    v[v > r.OUTLIER_HIGH] = np.nan

    # clip to valid range
    v[v < r.VALID_LOW] = r.VALID_LOW
    v[v > r.VALID_HIGH] = r.VALID_HIGH

    return v

############################################################
# BINNING
############################################################

def bin_time_series(df):
    df["time_bin"] = df["Hours"].astype(int)
    df = df.groupby("time_bin").mean(numeric_only=True)

    full_index = pd.Index(range(48), name="time_bin")
    df = df.reindex(full_index)

    df["Hours"] = df.index
    return df.reset_index(drop=True)

############################################################
# TIME SINCE LAST OBS
############################################################

def add_time_since_last_obs(series):
    last_seen = -1
    result = []

    for i, val in enumerate(series):
        if not pd.isna(val):
            last_seen = i
            result.append(0)
        else:
            result.append(i - last_seen if last_seen != -1 else np.nan)

    return result

############################################################
# MATRIX + MASK
############################################################

def build_matrix_and_mask(df):

    channels = [c for c in df.columns if c != "Hours"]

    T = len(df)
    F = len(channels)

    data_matrix = np.zeros((T, F))
    mask_matrix = np.zeros((T, F))

    for i, col in enumerate(channels):
        values = df[col].values
        mask = ~pd.isna(values)

        data_matrix[:, i] = np.nan_to_num(values, nan=0.0)
        mask_matrix[:, i] = mask.astype(int)

    return data_matrix, mask_matrix

############################################################
# MAIN FILE PROCESSING
############################################################

def preprocess_file(file_path, ranges):

    df = pd.read_csv(file_path)

    df = clean_dataframe(df)
    
    df = df[df["Hours"] <= 48].copy()

    # CLIPPING ADDED HERE
    for col in df.columns:
        if col != "Hours":
            df[col] = clip_variable(df[col], col, ranges)

    df = df.sort_values("Hours")

    df = bin_time_series(df)

    df_before_fill = df.copy()

    for col in df.columns:
        if col == "Hours":
            continue

        df[col + "_missing"] = df[col].isna().astype(int)
        df[col + "_time_since"] = add_time_since_last_obs(df[col])
        df[col + "_delta"] = df[col].diff()

    data_matrix, mask_matrix = build_matrix_and_mask(df_before_fill)

    # controlled fill
    for col in df.columns:
        if col != "Hours":
            df[col] = df[col].ffill(limit=3)

    return df, data_matrix, mask_matrix

############################################################
# PROCESS SPLIT
############################################################

def process_split(split, ranges):

    input_dir = os.path.join(INPUT_DIR, split)
    output_dir = os.path.join(OUTPUT_DIR, split)
    matrix_dir = os.path.join(OUTPUT_DIR, f"{split}_matrices")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(matrix_dir).mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith("_timeseries.csv")]

    for f in tqdm(files):

        input_path = os.path.join(input_dir, f)
        output_path = os.path.join(output_dir, f)

        df, data_matrix, mask_matrix = preprocess_file(input_path, ranges)

        df.to_csv(output_path, index=False)

        np.save(os.path.join(matrix_dir, f.replace(".csv", "_data.npy")), data_matrix)
        np.save(os.path.join(matrix_dir, f.replace(".csv", "_mask.npy")), mask_matrix)

############################################################
# MAIN
############################################################

if __name__ == "__main__":

    ranges_path = os.path.join(
        "mimic3benchmark",
        "resources",
        "variable_ranges.csv"
    )

    ranges = load_variable_ranges(ranges_path)

    for split in ["train", "test"]:
        print(f"Processing {split}...")
        process_split(split, ranges)

    print("Preprocessing complete.")
