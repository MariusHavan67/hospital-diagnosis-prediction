"""
Mapping des codes ICD-9 (diag_1, diag_2, diag_3) vers 9 grandes catégories
diagnostiques, en suivant le regroupement standard utilisé par Strack et al.
(2014) dans l'étude originale sur ce dataset.

Les codes ICD-9 numériques sont regroupés par plage ; les codes commençant
par 'V' (facteurs influençant l'état de santé) et 'E' (causes externes) sont
regroupés à part.
"""

import pandas as pd
import numpy as np


def map_icd9_to_category(code):
    """
    Convertit un code ICD-9 brut (string) en une catégorie diagnostique.

    Catégories retournées :
        - Circulatory, Respiratory, Digestive, Diabetes, Injury,
          Musculoskeletal, Genitourinary, Neoplasms, Other
    """
    if pd.isna(code) or code == "?":
        return "Missing"

    code = str(code).strip()

    # Codes V et E : regroupés dans "Other"
    if code.startswith("V") or code.startswith("E"):
        return "Other"

    try:
        value = float(code)
    except ValueError:
        return "Other"

    # Diabète : bloc 250.xx en priorité (avant la règle générale 240-279)
    if 250 <= value < 251:
        return "Diabetes"

    if 390 <= value <= 459 or value == 785:
        return "Circulatory"
    if 460 <= value <= 519 or value == 786:
        return "Respiratory"
    if 520 <= value <= 579 or value == 787:
        return "Digestive"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or value == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"

    return "Other"


def add_diagnosis_categories(df, cols=("diag_1", "diag_2", "diag_3")):
    """Ajoute les colonnes *_category pour chaque colonne de diagnostic."""
    df = df.copy()
    for col in cols:
        df[f"{col}_category"] = df[col].apply(map_icd9_to_category)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/diabetic_data.csv")
    df = add_diagnosis_categories(df)
    print(df["diag_1_category"].value_counts())
