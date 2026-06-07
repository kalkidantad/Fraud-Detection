"""End-to-end Task 1 preprocessing pipeline.

Run as a script to produce cleaned, feature-engineered datasets in
``data/processed``:

    python -m src.pipeline
"""
from __future__ import annotations

import pandas as pd

from . import config, data_loader, feature_engineering, geolocation, preprocessing


def build_fraud_dataset() -> pd.DataFrame:
    """Clean -> geolocate -> feature-engineer the e-commerce dataset."""
    fraud = data_loader.load_fraud_data()
    ip = data_loader.load_ip_country()

    fraud = preprocessing.basic_clean(fraud)
    fraud = preprocessing.handle_missing_values(fraud)
    fraud = geolocation.merge_ip_to_country(fraud, ip)
    fraud = feature_engineering.build_features(fraud)
    fraud = preprocessing.fix_dtypes(fraud)
    return fraud


def build_creditcard_dataset() -> pd.DataFrame:
    """Clean the bank dataset (already numeric / PCA-transformed)."""
    cc = data_loader.load_creditcard()
    cc = preprocessing.basic_clean(cc)
    cc = preprocessing.handle_missing_values(cc)
    return cc


def main() -> None:
    fraud = build_fraud_dataset()
    out_fraud = config.PROCESSED_DIR / "fraud_processed.csv"
    fraud.to_csv(out_fraud, index=False)
    print(f"Wrote {out_fraud.relative_to(config.PROJECT_ROOT)}  shape={fraud.shape}")

    cc = build_creditcard_dataset()
    out_cc = config.PROCESSED_DIR / "creditcard_processed.csv"
    cc.to_csv(out_cc, index=False)
    print(f"Wrote {out_cc.relative_to(config.PROJECT_ROOT)}  shape={cc.shape}")


if __name__ == "__main__":
    main()
