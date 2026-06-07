"""Load the raw challenge datasets with light type parsing."""
from __future__ import annotations

import pandas as pd

from . import config


def load_fraud_data(path=config.FRAUD_DATA_PATH) -> pd.DataFrame:
    """Load e-commerce ``Fraud_Data.csv`` and parse the timestamp columns."""
    df = pd.read_csv(path)
    for col in ("signup_time", "purchase_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_ip_country(path=config.IP_COUNTRY_PATH) -> pd.DataFrame:
    """Load the IP-range to country lookup table."""
    df = pd.read_csv(path)
    # IP bounds can be floats in the source; cast to a nullable integer.
    for col in ("lower_bound_ip_address", "upper_bound_ip_address"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def load_creditcard(path=config.CREDITCARD_PATH) -> pd.DataFrame:
    """Load the bank ``creditcard.csv`` dataset."""
    return pd.read_csv(path)
