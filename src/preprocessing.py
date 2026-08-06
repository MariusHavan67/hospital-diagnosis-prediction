"""
Preprocessing du dataset Diabetes 130-US hospitals pour la classification
de la catégorie diagnostique principale (diag_1_category).

Étapes :
    1. Chargement + dédoublonnage (on garde 1 encounter par patient pour
       éviter la fuite d'information entre train/test)
    2. Génération de la cible (diag_1 -> catégorie ICD-9)
    3. Nettoyage des valeurs manquantes ('?')
    4. Sélection des features et encodage
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from icd9_mapping import add_diagnosis_categories

RAW_PATH = "data/diabetic_data.csv"

# Colonnes qu'on retire d'office (identifiants, colonnes trop incomplètes,
# ou qui dérivent directement de la cible)
DROP_COLS = [
    "encounter_id", "patient_nbr", "weight", "payer_code",
    "diag_1", "diag_2", "diag_3",   # remplacées par les catégories
    "diag_2_category", "diag_3_category",  # évite la fuite d'info (co-diagnostics)
]

CATEGORICAL_COLS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty", "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin", "change",
    "diabetesMed", "readmitted",
]

NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]


def load_and_clean(path=RAW_PATH, one_encounter_per_patient=True, min_class_size=100):
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)

    if one_encounter_per_patient:
        # Garde la première visite de chaque patient pour éviter les fuites
        df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")

    df = add_diagnosis_categories(df, cols=("diag_1", "diag_2", "diag_3"))
    df = df[df["diag_1_category"] != "Missing"]

    # On retire les classes trop rares pour la validation croisée stratifiée
    counts = df["diag_1_category"].value_counts()
    valid_classes = counts[counts >= min_class_size].index
    df = df[df["diag_1_category"].isin(valid_classes)]

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # Colonnes quasi-constantes (peu de variance) souvent inutiles
    drop_low_variance = [
        "examide", "citoglipton", "glimepiride-pioglitazone",
        "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    df = df.drop(columns=[c for c in drop_low_variance if c in df.columns])

    return df


def encode_features(df, target_col="diag_1_category"):
    df = df.copy()
    y_raw = df.pop(target_col)

    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].fillna("Missing").astype(str)

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # XGBoost n'accepte pas [, ] ou < dans les noms de colonnes
    # (ex: age "[10-20)" généré par get_dummies)
    df.columns = [
        str(c).replace("[", "").replace("]", "").replace("<", "lt_")
        for c in df.columns
    ]

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)

    return df, y, target_encoder


def get_train_test_split(test_size=0.2, random_state=42):
    df = load_and_clean()
    X, y, target_encoder = encode_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, target_encoder


if __name__ == "__main__":
    df = load_and_clean()
    print("Shape après nettoyage:", df.shape)
    print(df["diag_1_category"].value_counts())
