"""Feature engineering for the e-commerce fraud dataset.

Highlights required by the brief:
- ``time_since_signup``: hours between signup and purchase (a very strong
  fraud signal — fraudulent accounts often transact almost immediately).
- Time-based features: hour-of-day and day-of-week of the purchase.
- Velocity / frequency features: transaction counts per user and per device.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``time_since_signup`` (hours) plus hour/day-of-week features."""
    out = df.copy()
    if {"signup_time", "purchase_time"}.issubset(out.columns):
        delta = out["purchase_time"] - out["signup_time"]
        out["time_since_signup"] = delta.dt.total_seconds() / 3600.0
        # Guard against negative/implausible values from bad timestamps.
        out["time_since_signup"] = out["time_since_signup"].clip(lower=0)

    if "purchase_time" in out.columns:
        out["purchase_hour"] = out["purchase_time"].dt.hour
        out["purchase_dayofweek"] = out["purchase_time"].dt.dayofweek
    return out


def add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transaction-count features per user and per device (velocity)."""
    out = df.copy()
    if "user_id" in out.columns:
        out["user_txn_count"] = out.groupby("user_id")["user_id"].transform("count")
    if "device_id" in out.columns:
        out["device_txn_count"] = out.groupby("device_id")["device_id"].transform(
            "count"
        )
        # Shared devices are a classic fraud-ring indicator.
        out["device_shared"] = (out["device_txn_count"] > 1).astype(int)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full feature-engineering pipeline."""
    out = add_time_features(df)
    out = add_frequency_features(out)
    return out
