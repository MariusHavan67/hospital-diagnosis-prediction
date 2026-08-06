"""
Analyse exploratoire du dataset Diabetes 130-US hospitals.
Génère les figures dans reports/figures/.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from icd9_mapping import add_diagnosis_categories

sns.set_theme(style="whitegrid")
FIG_DIR = "reports/figures"


def main():
    df = pd.read_csv("data/diabetic_data.csv")
    df = df.replace("?", np.nan)
    df = add_diagnosis_categories(df)

    # 1. Distribution de la cible
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df["diag_1_category"].value_counts().index
    sns.countplot(data=df, y="diag_1_category", order=order, ax=ax, color="#4C72B0")
    ax.set_title("Distribution des catégories diagnostiques (diag_1)")
    ax.set_xlabel("Nombre d'admissions")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_target_distribution.png", dpi=150)
    plt.close()

    # 2. Valeurs manquantes
    missing = df.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=missing.values * 100, y=missing.index, ax=ax, color="#DD8452")
    ax.set_title("Top 10 des colonnes avec valeurs manquantes")
    ax.set_xlabel("% de valeurs manquantes")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/02_missing_values.png", dpi=150)
    plt.close()

    # 3. Temps d'hospitalisation par catégorie
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="diag_1_category", y="time_in_hospital", ax=ax, order=order)
    ax.set_title("Durée de séjour par catégorie diagnostique")
    ax.set_xlabel("")
    ax.set_ylabel("Jours d'hospitalisation")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/03_los_by_category.png", dpi=150)
    plt.close()

    # 4. Âge vs catégorie diagnostique (heatmap de proportions)
    age_ct = pd.crosstab(df["age"], df["diag_1_category"], normalize="index")
    age_order = sorted(df["age"].dropna().unique(),
                        key=lambda x: int(x.strip("[)").split("-")[0]))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(age_ct.loc[age_order], cmap="YlOrRd", ax=ax, cbar_kws={"label": "Proportion"})
    ax.set_title("Proportion de catégories diagnostiques par tranche d'âge")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_age_vs_category.png", dpi=150)
    plt.close()

    print("Figures générées dans", FIG_DIR)
    print(df["diag_1_category"].value_counts())


if __name__ == "__main__":
    main()
