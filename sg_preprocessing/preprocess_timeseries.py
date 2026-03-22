import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

############################################################
# CONFIG
############################################################

MAX_HOURS = 48.0

# Map your variable names to cleaning functions
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
    "Glascow coma scale eye opening": "clean_gcs_eye",
    "Glascow coma scale motor response": "clean_gcs_motor",
    "Glascow coma scale verbal response": "clean_gcs_verbal"
}


############################################################
# Cleaning Functions (adapted for wide format)
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
    s = series.astype(str)
    out = pd.Series(np.nan, index=series.index)
    out[(s == "Normal <3 secs") | (s == "Brisk")] = 0
    out[(s == "Abnormal >3 secs") | (s == "Delayed")] = 1
    return out


def clean_fio2(series):
    v = pd.to_numeric(series, errors="coerce")
    idx = v > 1.0
    v.loc[idx] = v.loc[idx] / 100.0
    return v


def clean_lab(series):
    return pd.to_numeric(series, errors="coerce")


def clean_o2sat(series):
    v = pd.to_numeric(series, errors="coerce")
    idx = (v <= 1.0)
    v.loc[idx] = v.loc[idx] * 100.0
    return v


def clean_temperature(series):
    v = pd.to_numeric(series, errors="coerce")
    # assume values >= 79 are Fahrenheit
    idx = v >= 79
    v.loc[idx] = (v.loc[idx] - 32) * 5.0 / 9.0
    return v


def clean_weight(series):
    v = pd.to_numeric(series, errors="coerce")
    # assume > 250 likely lb
    idx = v > 250
    v.loc[idx] = v.loc[idx] * 0.453592
    return v


def clean_height(series):
    v = pd.to_numeric(series, errors="coerce")
    # assume < 3 likely meters, > 3 likely cm already
    idx = (v > 0) & (v < 3)
    v.loc[idx] = v.loc[idx] * 100.0
    return v

def clean_crr(series):
    s = series.astype(str).str.strip()

    mapping = {
        "Normal <3 secs": 0,
        "Brisk": 0,
        "Abnormal >3 secs": 1,
        "Delayed": 1
    }

    # First try direct mapping
    mapped = s.map(mapping)

    # Keep numeric if already numeric
    numeric = pd.to_numeric(series, errors="coerce")

    return mapped.combine_first(numeric)

def clean_gcs_eye(series):
    s = series.astype(str).str.strip()

    mapping = {
        "Spontaneously": 4,
        "To Speech": 3,
        "To Pain": 2,
        "None": 1
    }

    mapped = s.map(mapping)

    # Handle cases like "4 Spontaneously"
    numeric_prefix = s.str.extract(r"^(\d+)")[0]
    numeric_prefix = pd.to_numeric(numeric_prefix, errors="coerce")

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

    numeric_prefix = s.str.extract(r"^(\d+)")[0]
    numeric_prefix = pd.to_numeric(numeric_prefix, errors="coerce")

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

    numeric_prefix = s.str.extract(r"^(\d+)")[0]
    numeric_prefix = pd.to_numeric(numeric_prefix, errors="coerce")

    numeric = pd.to_numeric(series, errors="coerce")

    return mapped.combine_first(numeric_prefix).combine_first(numeric)



############################################################
# Range Clipping
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

    v[v < r.OUTLIER_LOW] = np.nan
    v[v > r.OUTLIER_HIGH] = np.nan
    v[v < r.VALID_LOW] = r.VALID_LOW
    v[v > r.VALID_HIGH] = r.VALID_HIGH

    return v


############################################################
# Main Cleaning Logic
############################################################

def clean_timeseries_file(input_path, output_path, ranges):

    df = pd.read_csv(input_path)

    # restrict to first 48 hours
    df = df[df["Hours"] <= MAX_HOURS].copy()

    for col in df.columns:
        if col == "Hours":
            continue

        # Apply cleaning if available
        if col in CLEAN_FNS:
            fn = globals()[CLEAN_FNS[col]]
            df[col] = fn(df[col])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Clip ranges
        df[col] = clip_variable(df[col], col, ranges)

        # Missing value handling (NEW)
        if col == "Hours":
            continue

        # Create missing indicator BEFORE imputation
        df[col + "_missing"] = df[col].isna().astype(int)

        # Forward fill then backward fill
        df[col] = df[col].ffill().bfill()
    
    df.to_csv(output_path, index=False)


############################################################
# Bulk Preprocessing
############################################################

def preprocess_split(input_dir, output_dir, ranges):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*_timeseries.csv"))

    for f in files:
        out_file = output_dir / f.name
        if out_file.exists():
            continue  # caching

        clean_timeseries_file(f, out_file, ranges)


def run_full_preprocessing(data_root):

    ranges_path = os.path.join(
        data_root,
        "mimic3benchmark/resources/variable_ranges.csv"
    )

    ranges = load_variable_ranges(ranges_path)

    for split in ["train", "test"]:
        input_dir = os.path.join(
            data_root,
            "data/in-hospital-mortality",
            split
        )
        output_dir = os.path.join(
            data_root,
            "data/in-hospital-mortality-cleaned",
            split
        )

        preprocess_split(input_dir, output_dir, ranges)

    print("Preprocessing complete.")

if __name__ == "__main__":
    run_full_preprocessing(".")
