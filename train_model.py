"""
Predictive Modeling Using Machine Learning
===========================================
Trains and compares Logistic Regression, Decision Tree, and Random Forest
classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset, then
evaluates each model with accuracy, a classification report, a confusion
matrix, and an ROC curve.

Dataset: sklearn's built-in `load_breast_cancer` (569 samples, 30 numeric
features, binary target: malignant=0 / benign=1). Using a built-in dataset
keeps the project fully reproducible with no external downloads.

Run:
    python src/train_model.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_STATE = 42


def load_data():
    """Load the dataset into a pandas DataFrame + feature/target arrays."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target  # 0 = malignant, 1 = benign
    return df, data.target_names


def build_models():
    """Return a dict of model name -> (needs_scaling, estimator)."""
    return {
        "Logistic Regression": (True, LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        "Decision Tree": (False, DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        "Random Forest": (False, RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    }


def evaluate_model(name, model, X_test, y_test, class_names, ax_cm, ax_roc):
    """Print metrics and draw confusion matrix + ROC curve for one model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'=' * 60}")
    print(f"Model: {name}")
    print(f"{'=' * 60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax_cm, colorbar=False)
    ax_cm.set_title(name)

    RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=ax_roc)

    return {"model": name, "accuracy": acc, "roc_auc": auc}


def main():
    df, class_names = load_data()
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = build_models()

    fig_cm, axes_cm = plt.subplots(1, len(models), figsize=(15, 4.5))
    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))

    results = []
    for (name, (needs_scaling, model)), ax_cm in zip(models.items(), axes_cm):
        Xtr = X_train_scaled if needs_scaling else X_train
        Xte = X_test_scaled if needs_scaling else X_test

        model.fit(Xtr, y_train)
        results.append(evaluate_model(name, model, Xte, y_test, class_names, ax_cm, ax_roc))

    fig_cm.suptitle("Confusion Matrices", y=1.05, fontsize=14)
    fig_cm.tight_layout()
    fig_cm.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax_roc.set_title("ROC Curves — Model Comparison")
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150, bbox_inches="tight")

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)

    print(f"\n{'=' * 60}")
    print("Summary (sorted by accuracy):")
    print(f"{'=' * 60}")
    print(results_df.to_string(index=False))
    print(f"\nPlots and results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
