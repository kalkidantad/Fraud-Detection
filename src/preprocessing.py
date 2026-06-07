"""Data-cleaning utilities: missing values, duplicates and dtype fixes."""
from __future__ import annotations

import pandas as pd


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and report the change.

    Returns a new, de-duplicated DataFrame. Missing-value handling is kept
    separate because the right strategy differs per column.
    """
    n_before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    n_after = len(out)
    if n_before != n_after:
        print(f"Removed {n_before - n_after} duplicate rows ({n_before} -> {n_after}).")
    return out


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing counts and percentages, sorted descending."""
    missing = df.isna().sum()
    summary = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_pct": (missing / len(df) * 100).round(3),
            "dtype": df.dtypes.astype(str),
        }
    )
    return summary.sort_values("missing_count", ascending=False)


def handle_missing_values(df: pd.DataFrame, drop_threshold: float = 0.5) -> pd.DataFrame:
    """Impute / drop missing values with sensible per-type defaults.

    - Columns with more than ``drop_threshold`` fraction missing are dropped.
    - Numeric columns are imputed with the median (robust to outliers).
    - Categorical/object columns are imputed with the mode.
    """
    out = df.copy()
    n = len(out)

    # Drop columns that are mostly empty.
    high_missing = [c for c in out.columns if out[c].isna().mean() > drop_threshold]
    if high_missing:
        print(f"Dropping high-missingness columns (> {drop_threshold:.0%}): {high_missing}")
        out = out.drop(columns=high_missing)

    for col in out.columns:
        if out[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())
        else:
            mode = out[col].mode(dropna=True)
            if not mode.empty:
                out[col] = out[col].fillna(mode.iloc[0])
    return out


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known categorical columns to the pandas ``category`` dtype."""
    out = df.copy()
    categorical_candidates = ["source", "browser", "sex", "country"]
    for col in categorical_candidates:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out
