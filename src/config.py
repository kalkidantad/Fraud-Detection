"""Central configuration: paths and shared constants."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

FRAUD_DATA_PATH = RAW_DIR / "Fraud_Data.csv"
IP_COUNTRY_PATH = RAW_DIR / "IpAddress_to_Country.csv"
CREDITCARD_PATH = RAW_DIR / "creditcard.csv"

RANDOM_STATE = 42

# Ensure output directories exist
for _d in (PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
