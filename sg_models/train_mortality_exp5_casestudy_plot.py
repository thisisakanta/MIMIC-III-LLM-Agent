import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================
TRAIN_MATRIX_DIR = "data/in-hospital-mortality-cleaned/train_matrices"
TEST_MATRIX_DIR = "data/in-hospital-mortality-cleaned/test_matrices"

SAVE_DIR = "data/in-hospital-mortality-cleaned/experiment5_case_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Canonical variable order for 17 cleaned matrix columns
# Must match the matrix column order used in preprocessing
# ============================================================
VARIABLES_17 = [
    "Capillary refill rate",
    "Diastolic blood pressure",
    "Fraction inspired oxygen",
    "Glascow coma scale eye opening",
    "Glascow coma scale motor response",
    "Glascow coma scale total",
    "Glascow coma scale verbal response",
    "Glucose",
    "Heart Rate",
    "Height",
    "Mean blood pressure",
    "Oxygen saturation",
    "Respiratory rate",
    "Systolic blood pressure",
    "Temperature",
    "Weight",
    "pH",
]

VAR_TO_IDX = {v: i for i, v in enumerate(VARIABLES_17)}

# ============================================================
# Dynamic plotting config by case type
# These are chosen to tell the most clinically meaningful story
# for each confusion-matrix outcome
# ============================================================
PLOT_CONFIG = {
    "TP": [
        "Glascow coma scale total",
        "Heart Rate",
        "Mean blood pressure",
        "Respiratory rate",
        "Oxygen saturation",
        "Temperature",
    ],
    "TN": [
        "Glascow coma scale total",
        "Heart Rate",
        "Mean blood pressure",
        "Respiratory rate",
        "Oxygen saturation",
    ],
    "FP": [
        "Glascow coma scale total",
        "Heart Rate",
        "Respiratory rate",
        "Oxygen saturation",
        "Temperature",
    ],
    "FN": [
        "Glascow coma scale total",
        "Heart Rate",
        "Mean blood pressure",
        "Respiratory rate",
        "Oxygen saturation",
    ],
}

# Optional fallback if case_type is unknown
DEFAULT_PLOT_VARS = [
    "Glascow coma scale total",
    "Heart Rate",
    "Mean blood pressure",
    "Respiratory rate",
    "Oxygen saturation",
    "Temperature",
]

# ============================================================
# Helper: resolve matrix paths
# ============================================================
def get_matrix_paths(matrix_dir: str, stay_name: str):
    base = stay_name.replace(".csv", "")
    data_path = os.path.join(matrix_dir, base + "_data.npy")
    mask_path = os.path.join(matrix_dir, base + "_mask.npy")
    return data_path, mask_path

# ============================================================
# Helper: infer whether stay is from test or train/val
# For Experiment 5, usually you will use test set cases.
# Keep matrix_dir configurable so you can override manually.
# ============================================================
def resolve_case_matrix_paths(stay: str, matrix_dir: str = TEST_MATRIX_DIR):
    data_path, mask_path = get_matrix_paths(matrix_dir, stay)

    if not os.path.exists(data_path) or not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Could not find matrix files for stay={stay}\n"
            f"Looked for:\n  {data_path}\n  {mask_path}"
        )

    return data_path, mask_path

# ============================================================
# Helper: sanitize filename
# ============================================================
def safe_name(text: str) -> str:
    return (
        str(text)
        .replace(".csv", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )

# ============================================================
# Main plotting function
# ============================================================
def plot_case_timeseries(
    stay: str,
    case_type: str,
    y_true: int | None = None,
    pred_prob: float | None = None,
    matrix_dir: str = TEST_MATRIX_DIR,
    custom_plot_vars: list[str] | None = None,
    shade_last24h: bool = True,
    save_dir: str = SAVE_DIR,
):
    """
    Plot a dynamic time-series panel for one selected case.

    Parameters
    ----------
    stay : str
        Example: "56825_episode1_timeseries.csv"
    case_type : str
        One of TP, TN, FP, FN
    y_true : int | None
        Optional true label to include in title
    pred_prob : float | None
        Optional predicted probability to include in title
    matrix_dir : str
        Usually TEST_MATRIX_DIR
    custom_plot_vars : list[str] | None
        If provided, overrides default case-type variable config
    shade_last24h : bool
        Whether to shade hours 24-48
    save_dir : str
        Where to save the PNG
    """
    os.makedirs(save_dir, exist_ok=True)

    case_type = case_type.upper().strip()
    plot_vars = custom_plot_vars if custom_plot_vars is not None else PLOT_CONFIG.get(case_type, DEFAULT_PLOT_VARS)

    # Validate variables
    for var in plot_vars:
        if var not in VAR_TO_IDX:
            raise ValueError(f"Variable '{var}' not found in canonical VARIABLES_17")

    data_path, mask_path = resolve_case_matrix_paths(stay, matrix_dir=matrix_dir)

    X = np.load(data_path).astype(np.float32)   # shape (48, 17)
    M = np.load(mask_path).astype(np.float32)   # shape (48, 17)

    hours = np.arange(X.shape[0])

    fig, axes = plt.subplots(len(plot_vars), 1, figsize=(11, 2.2 * len(plot_vars)), sharex=True)

    if len(plot_vars) == 1:
        axes = [axes]

    for ax, var in zip(axes, plot_vars):
        idx = VAR_TO_IDX[var]

        values = X[:, idx].copy()
        observed = M[:, idx].astype(bool)

        # Show only observed values in the line
        values_plot = values.copy()
        values_plot[~observed] = np.nan

        ax.plot(hours, values_plot, marker="o", linewidth=1.6)
        ax.set_ylabel(var, fontsize=10)
        ax.grid(True, alpha=0.3)

        if shade_last24h:
            ax.axvspan(24, 48, color="gray", alpha=0.08)

        # Optional: mark missingness
        missing_hours = hours[~observed]
        if len(missing_hours) > 0:
            y_min, y_max = ax.get_ylim()
            y_marker = y_min + 0.03 * (y_max - y_min)
            ax.scatter(
                missing_hours,
                np.full_like(missing_hours, y_marker, dtype=float),
                marker="x",
                s=20,
                alpha=0.5,
            )

    axes[-1].set_xlabel("Hour within 48h window", fontsize=11)

    title_parts = [f"Case Study {case_type}", stay.replace(".csv", "")]
    if pred_prob is not None:
        title_parts.append(f"Pred Prob={pred_prob:.3f}")
    if y_true is not None:
        title_parts.append(f"True={int(y_true)}")

    fig.suptitle(" | ".join(title_parts), fontsize=14)
    plt.tight_layout()

    out_name = f"case_plot_{safe_name(case_type)}_{safe_name(stay)}.png"
    out_path = os.path.join(save_dir, out_name)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved case plot to:", out_path)
    print("Plotted variables:", plot_vars)

    return out_path

# ============================================================
# Optional batch helper from selected-cases CSV
# ============================================================
def plot_selected_cases_from_csv(
    csv_path: str,
    matrix_dir: str = TEST_MATRIX_DIR,
    save_dir: str = SAVE_DIR,
):
    """
    Expects CSV with columns at least:
      case_type, stay
    Optional columns:
      y_true, ensemble_prob
    """
    df = pd.read_csv(csv_path)

    required = {"case_type", "stay"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in selected-cases CSV: {missing}")

    output_paths = []

    for _, row in df.iterrows():
        out_path = plot_case_timeseries(
            stay=row["stay"],
            case_type=row["case_type"],
            y_true=row["y_true"] if "y_true" in df.columns else None,
            pred_prob=row["ensemble_prob"] if "ensemble_prob" in df.columns else None,
            matrix_dir=matrix_dir,
            save_dir=save_dir,
        )
        output_paths.append(out_path)

    return output_paths

# ============================================================
# EXAMPLE USAGE: one case at a time
# Uncomment and edit
# ============================================================
# plot_case_timeseries(
#     stay="56825_episode1_timeseries.csv",
#     case_type="TP",
#     y_true=1,
#     pred_prob=0.932,
#     matrix_dir=TEST_MATRIX_DIR,
# )

# ============================================================
# EXAMPLE USAGE: batch from CSV created in Colab
# Uncomment and edit
# ============================================================
if __name__ == "__main__":
    plot_selected_cases_from_csv(
        csv_path="data/in-hospital-mortality-cleaned/experiment5_selected_cases_ensemble_48h.csv",
        matrix_dir=TEST_MATRIX_DIR
    )