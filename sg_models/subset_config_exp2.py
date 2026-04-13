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

SUBSET_VARS = {
    "hemodynamics": [
        "Capillary refill rate",
        "Diastolic blood pressure",
        "Mean blood pressure",
        "Systolic blood pressure",
        "Heart Rate",
    ],
    "respiratory_acidbase": [
        "Fraction inspired oxygen",
        "Oxygen saturation",
        "Respiratory rate",
        "pH",
    ],
    "neurologic": [
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale total",
        "Glascow coma scale verbal response",
    ],
    "metabolic_general": [
        "Glucose",
        "Temperature",
        "Height",
        "Weight",
    ],
}

def get_subset_indices(subset_name: str):
    vars_for_subset = SUBSET_VARS[subset_name]
    return [VARIABLES_17.index(v) for v in vars_for_subset]