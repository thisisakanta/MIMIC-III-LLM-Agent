import pandas as pd

# ===== CHANGE PATHS =====
INPUT_CSV = "E:\\L4-T2\\tanmoy_da\\mimic3-benchmarks\\mimic3models\\in_hospital_mortality\\random_forest\\extracted_features\\01_val_raw_features.csv"
OUTPUT_CSV = "E:\\L4-T2\\tanmoy_da\\mimic3-benchmarks\\mimic3models\\in_hospital_mortality\\random_forest\\extracted_features\\llm_Val.csv"
# ========================

print("Loading aggregated CSV...")
df = pd.read_csv(INPUT_CSV)


base_columns = ["patient_name", "mortality_label"]

mean_columns = [
    col for col in df.columns
    if col.endswith("_first100%_mean")
]

print(f"Found {len(mean_columns)} first100% mean columns.")


def build_clinical_text(row):
    sentences = []

    for col in mean_columns:
        value = row[col]

        if pd.isna(value):
            continue

        feature_name = col.replace("_first100%_mean", "")
        sentences.append(f"{feature_name} {value:.3f}.")

    return "" + " ".join(sentences)+" These are the average values of the features over the first 48 hours of the ICU stay."

print("Building clinical_text column...")
df["clinical_text"] = df.apply(build_clinical_text, axis=1)

# Create final 3-column dataframe
final_df = df[["patient_name", "mortality_label", "clinical_text"]]

print("Saving new CSV...")
final_df.to_csv(OUTPUT_CSV, index=False)

print("Done!")
print(f"Saved as: {OUTPUT_CSV}")