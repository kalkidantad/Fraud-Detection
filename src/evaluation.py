"""Evaluation metrics & plots tuned for imbalanced fraud detection.

For imbalanced problems we prioritise **AUC-PR** (average precision) and
**F1** over accuracy, and always inspect the confusion matrix.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config


def evaluate(model, X_test, y_test, name: str = "model") -> dict:
    """Return a dict of headline metrics and print a classification report."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:  # fall back to decision_function
        y_score = model.decision_function(X_test)

    metrics = {
        "model": name,
        "auc_pr": average_precision_score(y_test, y_score),
        "roc_auc": roc_auc_score(y_test, y_score),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
    }

    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k:>10}: {v:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    return metrics


def plot_confusion(model, X_test, y_test, name: str = "model"):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["legit", "fraud"], yticklabels=["legit", "fraud"])
    ax.set(title=f"Confusion matrix — {name}", xlabel="predicted", ylabel="actual")
    path = config.FIGURES_DIR / f"confusion_{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved -> {path.relative_to(config.PROJECT_ROOT)}")
    return cm


def plot_pr_curve(models: dict, X_test, y_test, name: str = "pr_curve"):
    """Overlay precision-recall curves for several fitted models."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, model in models.items():
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_test)[:, 1]
        else:
            score = model.decision_function(X_test)
        prec, rec, _ = precision_recall_curve(y_test, score)
        ap = average_precision_score(y_test, score)
        ax.plot(rec, prec, label=f"{label} (AP={ap:.3f})")
    ax.set(title="Precision-Recall curves", xlabel="recall", ylabel="precision")
    ax.legend()
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved -> {path.relative_to(config.PROJECT_ROOT)}")
    return path


def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Tidy comparison table from a list of ``evaluate`` outputs."""
    return pd.DataFrame(rows).set_index("model").round(4)
