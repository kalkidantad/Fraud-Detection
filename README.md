# Fraud-Detection

Improved detection of fraud cases for e-commerce and bank transactions
(10 Academy, Week 5 & 6 — Adey Innovations Inc.).

## Project structure

```
.
├── data/
│   ├── raw/          # put Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv here (git-ignored)
│   └── processed/    # generated, cleaned datasets
├── notebooks/
│   ├── 01_data_analysis_preprocessing.ipynb   # Task 1
│   └── 02_model_building.ipynb                # Task 2
├── models/           # serialized trained models (git-ignored)
├── reports/
│   ├── figures/      # generated EDA & evaluation figures
│   ├── interim_1_report.md
│   └── interim_2_report.md
├── src/              # reusable pipeline modules
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── geolocation.py
│   ├── feature_engineering.py
│   ├── transform.py
│   ├── eda.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── pipeline.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the three datasets in `data/raw/` (see `data/raw/README.md`).

## Task 1 — Data Analysis & Preprocessing

Run the notebook or the pipeline:

```bash
jupyter notebook notebooks/01_data_analysis_preprocessing.ipynb
# or
python -m src.pipeline
```

Covers cleaning, EDA, IP-to-country geolocation, feature engineering
(`time_since_signup`, velocity features), and the class-imbalance strategy.
See `reports/interim_1_report.md`.

## Task 2 — Model Building & Training

```bash
jupyter notebook notebooks/02_model_building.ipynb
```

Trains and compares **Logistic Regression** and **XGBoost** on both datasets
with SMOTE/`scale_pos_weight`, evaluated with AUC-PR, F1 and confusion
matrices. See `reports/interim_2_report.md`.
