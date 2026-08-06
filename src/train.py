"""
Entraîne et compare 3 modèles de classification multi-classes pour prédire
la catégorie diagnostique principale (diag_1_category) :

    - Régression logistique multinomiale (baseline linéaire)
    - Random Forest
    - XGBoost

Le déséquilibre de classes est géré via class_weight='balanced' (RF, LogReg)
et sample_weight (XGBoost). La métrique principale est le F1-macro, plus
robuste à ce déséquilibre que l'accuracy.
"""

import json
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib

from preprocessing import get_train_test_split

RANDOM_STATE = 42


def evaluate(name, model, X_test, y_test, label_names):
    y_pred = model.predict(X_test)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    print(f"\n=== {name} ===")
    print(f"Accuracy   : {acc:.4f}")
    print(f"F1-macro   : {f1_macro:.4f}")
    return {"name": name, "accuracy": acc, "f1_macro": f1_macro, "report": report,
            "y_pred": y_pred}


def main():
    print("Chargement et préparation des données...")
    X_train, X_test, y_train, y_test, target_encoder = get_train_test_split()
    label_names = list(target_encoder.classes_)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Classes ({len(label_names)}): {label_names}")

    results = []

    # --- Modèle 1 : Régression logistique multinomiale ---
    t0 = time.time()
    logreg = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    logreg.fit(X_train, y_train)
    print(f"LogReg entraîné en {time.time()-t0:.1f}s")
    results.append(evaluate("Logistic Regression", logreg, X_test, y_test, label_names))

    # --- Modèle 2 : Random Forest ---
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=20, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print(f"Random Forest entraîné en {time.time()-t0:.1f}s")
    results.append(evaluate("Random Forest", rf, X_test, y_test, label_names))

    # --- Modèle 3 : XGBoost ---
    t0 = time.time()
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", num_class=len(label_names),
        eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"XGBoost entraîné en {time.time()-t0:.1f}s")
    results.append(evaluate("XGBoost", xgb, X_test, y_test, label_names))

    # --- Sauvegarde du meilleur modèle (F1-macro) ---
    best = max(results, key=lambda r: r["f1_macro"])
    print(f"\n>>> Meilleur modèle : {best['name']} (F1-macro = {best['f1_macro']:.4f})")

    models = {"Logistic Regression": logreg, "Random Forest": rf, "XGBoost": xgb}
    joblib.dump(models[best["name"]], "models/best_model.joblib")
    joblib.dump(target_encoder, "models/target_encoder.joblib")
    joblib.dump(list(X_train.columns), "models/feature_columns.joblib")

    # --- Résumé comparatif ---
    summary = pd.DataFrame([
        {"model": r["name"], "accuracy": r["accuracy"], "f1_macro": r["f1_macro"]}
        for r in results
    ])
    summary.to_csv("reports/model_comparison.csv", index=False)
    print("\nRésumé :\n", summary)

    # Matrice de confusion + rapport détaillé du meilleur modèle -> reports/
    cm = confusion_matrix(y_test, best["y_pred"])
    np.save("reports/confusion_matrix_best_model.npy", cm)
    with open("reports/classification_report_best_model.json", "w") as f:
        json.dump(best["report"], f, indent=2)

    return results, label_names


if __name__ == "__main__":
    main()
