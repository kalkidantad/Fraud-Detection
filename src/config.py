"""Central configuration: paths and shared constants."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _resolve_data_path(filename: str) -> Path:
    """Prefer ``data/raw/``; fall back to ``data/`` (direct CSV drop)."""
    in_raw = RAW_DIR / filename
    if in_raw.exists():
        return in_raw
    return DATA_DIR / filename


FRAUD_DATA_PATH = _resolve_data_path("Fraud_Data.csv")
IP_COUNTRY_PATH = _resolve_data_path("IpAddress_to_Country.csv")
CREDITCARD_PATH = _resolve_data_path("creditcard.csv")

RANDOM_STATE = 42

# Ensure output directories exist
for _d in (PROCESSED_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
