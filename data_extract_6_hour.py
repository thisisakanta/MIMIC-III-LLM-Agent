import os
import json
import numpy as np
import pandas as pd
from readers import InHospitalMortalityReader

DATA_PATH = "E:\\L4-T2\\MIMIC_III_LLM_AGENT\\data"
OUTPUT_FILE = "E:\\L4-T2\\MIMIC_III_LLM_AGENT\\data\\in-hospital-mortalityllm_inhospital_6hour_val.csv"

TIME_WINDOW = 6
MAX_HOURS = 48

TARGET_CHANNELS = [
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
    "pH"
]


def convert_categorical(value, channel, channel_info):
    if value == "":
        return np.nan
    if len(channel_info[channel]['possible_values']) != 0:
        return channel_info[channel]['values'][value]
    return float(value)


def process_patient(timeseries, header, channel_info):
    df = pd.DataFrame(timeseries, columns=header)
    df["Hours"] = df["Hours"].astype(float)

    text_block = "ICU admission summary of First 48 hours.\nMeasurements averaged over 6-hour intervals:\n[0–6h, 6–12h, 12–18h, 18–24h, 24–30h, 30–36h, 36–42h, 42–48h]\n"

    for channel in TARGET_CHANNELS:
        if channel not in df.columns:
            continue

        numeric_values = []
        for idx, row in df.iterrows():
            val = convert_categorical(row[channel], channel, channel_info)
            numeric_values.append(val)

        df[channel] = numeric_values

        bin_means = []

        for start in range(0, MAX_HOURS, TIME_WINDOW):
            end = start + TIME_WINDOW
            window = df[(df["Hours"] >= start) & (df["Hours"] < end)]
            mean_val = window[channel].mean()
            bin_means.append(mean_val)

        # forward fill
        series = pd.Series(bin_means)
        series = series.fillna(method="ffill")
        bin_means = series.tolist()

        # format text
        text_block += f"{channel.lower()}: "
        text_block += ", ".join(
            [f"{v:.2f}" if not np.isnan(v) else "nan" for v in bin_means]
        )
        text_block += "\n\n"

    return text_block.strip()


def main():
    reader = InHospitalMortalityReader(
        dataset_dir=os.path.join(DATA_PATH, "train"),
        listfile=os.path.join(DATA_PATH, "train_listfile.csv"),
        period_length=48.0
    )

    with open("E:\\L4-T2\MIMIC_III_LLM_AGENT\\resources\\channel_info.json") as f:
        channel_info = json.load(f)

    rows = []
    i=0
    for _ in range(reader.get_number_of_examples()):
        example = reader.read_next()
        i=i+1
        print(i)

        patient_name = example["name"]
        mortality_label = example["y"]
        timeseries = example["X"]
        header = example["header"]

        clinical_text = process_patient(timeseries, header, channel_info)

        rows.append({
            "patient_name": patient_name,
            "mortality_label": mortality_label,
            "clinical_text": clinical_text
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()