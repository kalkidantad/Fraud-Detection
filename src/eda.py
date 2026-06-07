"""Reusable EDA plotting helpers. Figures are saved to ``reports/figures``."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid")


def _save(fig, name: str):
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved figure -> {path.relative_to(config.PROJECT_ROOT)}")
    return path


def class_balance(df: pd.DataFrame, target: str, name="class_balance.png"):
    """Bar plot of the target distribution and printed imbalance ratio."""
    counts = df[target].value_counts().sort_index()
    ratio = counts.min() / counts.max()
    print(f"Class counts:\n{counts}\nMinority/majority ratio: {ratio:.4f}")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    ax.set(title=f"Class distribution ({target})", xlabel=target, ylabel="count")
    return _save(fig, name)


def numeric_distribution(df: pd.DataFrame, col: str, target: str | None = None,
                         name: str | None = None, log: bool = False):
    """Histogram of a numeric column, optionally split by the target class."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if target is not None and target in df.columns:
        for cls, sub in df.groupby(target):
            sns.histplot(sub[col].dropna(), label=f"{target}={cls}", stat="density",
                         element="step", ax=ax, log_scale=log)
        ax.legend()
    else:
        sns.histplot(df[col].dropna(), ax=ax, log_scale=log)
    ax.set(title=f"Distribution of {col}", xlabel=col)
    return _save(fig, name or f"dist_{col}.png")


def fraud_rate_by_category(df: pd.DataFrame, col: str, target: str,
                           name: str | None = None, top: int = 15):
    """Bar plot of mean fraud rate per category level."""
    rate = (df.groupby(col)[target].mean().sort_values(ascending=False).head(top))
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=rate.values, y=rate.index.astype(str), ax=ax)
    ax.set(title=f"Fraud rate by {col}", xlabel="fraud rate", ylabel=col)
    return _save(fig, name or f"fraud_rate_{col}.png")


def correlation_heatmap(df: pd.DataFrame, name="correlation_heatmap.png"):
    """Heatmap of correlations among numeric features."""
    numeric = df.select_dtypes("number")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set(title="Numeric feature correlations")
    return _save(fig, name)
