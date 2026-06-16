"""Task 3 — Model explainability for the selected XGBoost models.

This module turns the two serialized XGBoost pipelines (``models/*.joblib``)
into interpretable artefacts:

1. **Built-in feature importance** — XGBoost's gain-based importance, the
   model's own view of which features split the data most usefully.
2. **SHAP analysis** — game-theoretic Shapley values (via the exact
   ``TreeExplainer``) that explain *both* global behaviour (summary/beeswarm
   plots) and *individual* predictions (force plots).

Design notes
------------
* The saved models are imbalanced-learn ``Pipeline`` objects of the form
  ``preprocess (ColumnTransformer) -> clf (XGBClassifier)``. SHAP's
  ``TreeExplainer`` must run on the **transformed** feature matrix (post
  scaling + one-hot encoding), so we split the pipeline and feed the explainer
  the transformed design matrix together with the human-readable feature names
  produced by ``ColumnTransformer.get_feature_names_out``.
* We rebuild the **exact** stratified test split used in Task 2 (same
  ``random_state``) so that predictions — and therefore the true-positive /
  false-positive / false-negative examples we explain — are reproducible.
* SHAP is computed on the test split only (data the model never saw).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config, transform

# Headless-friendly backend so figures render when run as a script.
plt.switch_backend("Agg")


# --------------------------------------------------------------------------- #
# Pipeline helpers
# --------------------------------------------------------------------------- #
def load_model(path):
    """Load a serialized imbalanced-learn pipeline from ``models/``."""
    return joblib.load(path)


def get_classifier(model):
    """Return the fitted final estimator (the XGBoost classifier)."""
    return model.named_steps["clf"]


def _clean_names(names) -> list[str]:
    """Strip ColumnTransformer prefixes (``num__`` / ``cat__``) for readability."""
    cleaned = []
    for n in names:
        for prefix in ("num__", "cat__", "remainder__"):
            if n.startswith(prefix):
                n = n[len(prefix):]
                break
        cleaned.append(n)
    return cleaned


def transform_features(model, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted preprocessor and return a named DataFrame.

    The returned matrix is exactly what the XGBoost step sees, so SHAP values
    line up one-to-one with these columns.
    """
    pre = model.named_steps["preprocess"]
    X_trans = pre.transform(X)
    names = _clean_names(pre.get_feature_names_out())
    if hasattr(X_trans, "toarray"):  # densify if a sparse matrix slipped through
        X_trans = X_trans.toarray()
    return pd.DataFrame(X_trans, columns=names, index=X.index)


# --------------------------------------------------------------------------- #
# Built-in (gain-based) importance
# --------------------------------------------------------------------------- #
def builtin_importance(model, top_n: int | None = None) -> pd.DataFrame:
    """Gain-based feature importance straight from the trained XGBoost model.

    XGBoost stores importances by internal feature index (``f0``, ``f1`` …);
    we map those back to the transformed feature names so the table is
    readable.
    """
    clf = get_classifier(model)
    pre = model.named_steps["preprocess"]
    names = _clean_names(pre.get_feature_names_out())

    importances = clf.feature_importances_  # gain, normalised to sum to 1
    df = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return df.head(top_n) if top_n else df


def plot_builtin_importance(model, name: str, top_n: int = 10):
    """Horizontal bar chart of the top-N built-in importances; saved to figures."""
    imp = builtin_importance(model, top_n=top_n)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#2b8cbe")
    ax.set(
        title=f"Top {top_n} built-in feature importance — {name}",
        xlabel="XGBoost gain importance",
    )
    fig.tight_layout()
    path = config.FIGURES_DIR / f"importance_builtin_{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path.relative_to(config.PROJECT_ROOT)}")
    return imp


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def compute_shap(model, X_trans: pd.DataFrame):
    """Exact tree-SHAP values for the transformed design matrix.

    Returns a ``shap.Explanation`` whose ``.values`` are in log-odds (margin)
    space — the natural output space for an XGBoost classifier.
    """
    clf = get_classifier(model)
    explainer = shap.TreeExplainer(clf)
    explanation = explainer(X_trans, check_additivity=False)
    return explainer, explanation


def shap_importance(explanation, top_n: int | None = None) -> pd.DataFrame:
    """Global SHAP importance = mean(|SHAP value|) per feature."""
    vals = np.abs(explanation.values).mean(axis=0)
    df = (
        pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": vals})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return df.head(top_n) if top_n else df


def plot_shap_summary(explanation, name: str, max_display: int = 15):
    """Beeswarm summary plot (global feature importance + direction)."""
    fig = plt.figure()
    shap.summary_plot(
        explanation.values,
        explanation.data,
        feature_names=explanation.feature_names,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(f"SHAP summary (beeswarm) — {name}", y=1.02)
    path = config.FIGURES_DIR / f"shap_summary_{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path.relative_to(config.PROJECT_ROOT)}")


def plot_shap_bar(explanation, name: str, max_display: int = 15):
    """Bar plot of mean(|SHAP|) — directly comparable to built-in importance."""
    fig = plt.figure()
    shap.summary_plot(
        explanation.values,
        explanation.data,
        feature_names=explanation.feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(f"SHAP global importance (mean |SHAP|) — {name}", y=1.02)
    path = config.FIGURES_DIR / f"shap_bar_{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path.relative_to(config.PROJECT_ROOT)}")


# --------------------------------------------------------------------------- #
# Individual predictions: TP / FP / FN selection + force plots
# --------------------------------------------------------------------------- #
def classify_examples(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray):
    """Return positional indices for a confident TP, FP and FN.

    We pick the most *confident* mistake/hit in each bucket (extreme scores)
    so the force plots tell a clear story.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]

    picks: dict[str, int] = {}
    if len(tp):
        picks["true_positive"] = int(tp[np.argmax(y_score[tp])])      # most confident catch
    if len(fp):
        picks["false_positive"] = int(fp[np.argmax(y_score[fp])])     # most confident false alarm
    if len(fn):
        picks["false_negative"] = int(fn[np.argmin(y_score[fn])])     # most badly missed fraud
    return picks


def plot_force(explainer, explanation, idx: int, name: str, kind: str):
    """Save a SHAP force plot (matplotlib) for a single prediction."""
    fig = shap.force_plot(
        explainer.expected_value,
        explanation.values[idx, :],
        features=explanation.data[idx, :],
        feature_names=explanation.feature_names,
        matplotlib=True,
        show=False,
    )
    path = config.FIGURES_DIR / f"shap_force_{name}_{kind}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path.relative_to(config.PROJECT_ROOT)}")
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class ExplainResult:
    name: str
    builtin_top: pd.DataFrame
    shap_top: pd.DataFrame
    examples: dict = field(default_factory=dict)


def explain_dataset(df: pd.DataFrame, target: str, model_path, name: str,
                    sample_size: int = 5000) -> ExplainResult:
    """Full Task-3 explainability run for one dataset.

    Parameters
    ----------
    sample_size : cap the number of test rows used for the (expensive-to-plot)
        SHAP summary so the beeswarm stays legible and fast; TP/FP/FN selection
        still uses the full test set.
    """
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    model = load_model(model_path)

    # Reproduce the exact Task-2 test split.
    X_train, X_test, y_train, y_test = transform.stratified_split(df, target)

    # Predictions on the full test set (for confusion-bucket selection).
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    # 1) Built-in importance.
    builtin_top = plot_builtin_importance(model, name, top_n=10)
    print("\nBuilt-in importance (top 10):")
    print(builtin_top.to_string(index=False))

    # 2) SHAP on a sample of the test set.
    rng = np.random.RandomState(config.RANDOM_STATE)
    if len(X_test) > sample_size:
        sample_pos = rng.choice(len(X_test), size=sample_size, replace=False)
    else:
        sample_pos = np.arange(len(X_test))
    X_sample = X_test.iloc[sample_pos]
    X_sample_trans = transform_features(model, X_sample)

    explainer, explanation = compute_shap(model, X_sample_trans)
    plot_shap_summary(explanation, name)
    plot_shap_bar(explanation, name)
    shap_top = shap_importance(explanation, top_n=10)
    print("\nSHAP importance (top 10, mean |SHAP|):")
    print(shap_top.to_string(index=False))

    # 3) Force plots for TP / FP / FN (selected from the full test set, then
    #    explained individually so we always have a valid example).
    picks = classify_examples(y_test.to_numpy(), y_pred, y_score)
    examples = {}
    for kind, pos in picks.items():
        X_one = X_test.iloc[[pos]]
        X_one_trans = transform_features(model, X_one)
        expl_one_explainer, expl_one = compute_shap(model, X_one_trans)
        plot_force(expl_one_explainer, expl_one, 0, name, kind)
        examples[kind] = {
            "true_label": int(y_test.to_numpy()[pos]),
            "pred_label": int(y_pred[pos]),
            "fraud_probability": float(y_score[pos]),
        }
    print("\nExplained individual predictions:")
    for k, v in examples.items():
        print(f"  {k:>15}: prob={v['fraud_probability']:.4f} "
              f"(true={v['true_label']}, pred={v['pred_label']})")

    return ExplainResult(name=name, builtin_top=builtin_top,
                         shap_top=shap_top, examples=examples)


def main() -> None:
    """Generate every Task-3 artefact for both datasets."""
    fraud_path = config.PROCESSED_DIR / "fraud_processed.csv"
    cc_path = config.PROCESSED_DIR / "creditcard_processed.csv"

    if not fraud_path.exists() or not cc_path.exists():
        raise FileNotFoundError(
            "Processed data missing. Run `python -m src.pipeline` first."
        )

    fraud = pd.read_csv(fraud_path)
    cc = pd.read_csv(cc_path)

    explain_dataset(fraud, "class", config.PROJECT_ROOT / "models" /
                    "xgb_ecommerce.joblib", "ecommerce")
    explain_dataset(cc, "Class", config.PROJECT_ROOT / "models" /
                    "xgb_creditcard.joblib", "creditcard")


if __name__ == "__main__":
    main()
