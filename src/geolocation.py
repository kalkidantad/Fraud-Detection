"""Map e-commerce IP addresses to countries via the IP-range lookup table.

The ``ip_address`` column in ``Fraud_Data.csv`` is a numeric (often float)
representation of an IPv4 address. ``IpAddress_to_Country.csv`` provides
``[lower_bound_ip_address, upper_bound_ip_address]`` ranges per country.

A naive cross-join is O(n*m) and far too slow. We sort the ranges by their
lower bound and use ``merge_asof`` (a backward search) to attach the candidate
range to each transaction in O((n+m) log m), then validate the upper bound.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def merge_ip_to_country(
    fraud_df: pd.DataFrame,
    ip_df: pd.DataFrame,
    ip_col: str = "ip_address",
) -> pd.DataFrame:
    """Return ``fraud_df`` with an added ``country`` column.

    IPs that fall outside every known range are labelled ``"Unknown"``.
    """
    df = fraud_df.copy()

    # Integer IP for range comparison; invalid values become NaN -> Unknown.
    df["ip_int"] = pd.to_numeric(df[ip_col], errors="coerce").astype("float64")

    ip = ip_df.copy()
    ip["lower_bound_ip_address"] = pd.to_numeric(
        ip["lower_bound_ip_address"], errors="coerce"
    ).astype("float64")
    ip["upper_bound_ip_address"] = pd.to_numeric(
        ip["upper_bound_ip_address"], errors="coerce"
    ).astype("float64")
    ip = ip.dropna(subset=["lower_bound_ip_address"]).sort_values(
        "lower_bound_ip_address"
    )

    # Preserve original order, sort by key for merge_asof, then restore.
    df = df.reset_index().rename(columns={"index": "_orig_order"})
    df_sorted = df.dropna(subset=["ip_int"]).sort_values("ip_int")

    merged = pd.merge_asof(
        df_sorted,
        ip,
        left_on="ip_int",
        right_on="lower_bound_ip_address",
        direction="backward",
    )

    # merge_asof only guarantees lower_bound <= ip; validate the upper bound.
    out_of_range = merged["ip_int"] > merged["upper_bound_ip_address"]
    merged.loc[out_of_range, "country"] = np.nan

    merged = merged[["_orig_order", "country"]]
    result = df.merge(merged, on="_orig_order", how="left")
    result["country"] = result["country"].fillna("Unknown")

    result = (
        result.sort_values("_orig_order")
        .drop(columns=["_orig_order", "ip_int"])
        .reset_index(drop=True)
    )
    return result
