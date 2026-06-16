"""Encoding and scaling helpers (categorical -> numeric, normalize features)."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def stratified_split(df: pd.DataFrame, target: str, test_size: float = 0.2):
    """Stratified train/test split returning (X_train, X_test, y_train, y_test)."""
    drop_cols = [c for c in (target,) if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE
    )


def split_feature_types(df: pd.DataFrame, target: str):
    """Return (numeric_cols, categorical_cols) excluding the target/IDs."""
    drop_cols = {target, "user_id", "device_id", "ip_address", "signup_time",
                 "purchase_time"}
    features = [c for c in df.columns if c not in drop_cols]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def build_preprocessor(numeric_cols, categorical_cols) -> ColumnTransformer:
    """ColumnTransformer: StandardScaler for numeric, OneHot for categorical.

    Fit this on the TRAIN split only to avoid leakage.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                              sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
