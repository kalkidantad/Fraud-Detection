# Fraud-Detection

Improved detection of fraud cases for e-commerce and bank transactions
(10 Academy, Week 5 & 6 — Adey Innovations Inc.).

End-to-end project: data analysis & feature engineering → model building &
selection → model explainability with SHAP → business recommendations. The two
datasets are treated as **separate modeling problems** with independent
pipelines. Because both are highly imbalanced, the primary metrics are
**AUC-PR** and **F1** (not accuracy).

## Project structure

```
.
├── data/
│   ├── raw/          # put Fraud_Data.csv, IpAddress_to_Country.csv, creditcard.csv here (git-ignored)
│   └── processed/    # generated, cleaned datasets
├── notebooks/
│   ├── 01_data_analysis_preprocessing.ipynb   # Task 1
│   ├── 02_model_building.ipynb                 # Task 2
│   └── 03_model_explainability.ipynb           # Task 3 (SHAP)
├── models/           # serialized trained models (git-ignored)
├── reports/
│   ├── figures/      # generated EDA, evaluation & SHAP figures
│   ├── interim_1_report.md   # Task 1
│   ├── interim_2_report.md   # Task 2
│   ├── interim_3_report.md   # Task 3 (explainability + recommendations)
│   └── final_report.md       # end-to-end narrative report
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
│   ├── explainability.py     # Task 3: feature importance + SHAP
│   └── pipeline.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the three datasets in `data/raw/` (see `data/raw/README.md`). The loader
also accepts them dropped directly in `data/`.

## Quickstart (full pipeline)

```bash
python -m src.pipeline                                                       # Task 1 -> data/processed/
jupyter nbconvert --to notebook --execute notebooks/02_model_building.ipynb  # Task 2 -> models/
python -m src.explainability                                                 # Task 3 -> reports/figures/
```

## Task 1 — Data Analysis & Preprocessing

```bash
jupyter notebook notebooks/01_data_analysis_preprocessing.ipynb
# or
python -m src.pipeline
```

Cleaning, EDA, IP-to-country geolocation (`merge_asof`), feature engineering
(`time_since_signup`, velocity features), and the class-imbalance strategy.
See `reports/interim_1_report.md`.

## Task 2 — Model Building & Training

```bash
jupyter notebook notebooks/02_model_building.ipynb
```

Trains and compares **Logistic Regression** and **XGBoost** on both datasets
with SMOTE / `scale_pos_weight` (resampling on the training fold only),
evaluated with AUC-PR, F1, ROC-AUC and confusion matrices. **XGBoost** is
selected for both datasets and serialized to `models/`. See
`reports/interim_2_report.md`.

## Task 3 — Model Explainability (SHAP)

```bash
jupyter notebook notebooks/03_model_explainability.ipynb
# or
python -m src.explainability
```

Interprets the selected XGBoost models:

- **Built-in feature importance** (top 10 per dataset).
- **SHAP summary / beeswarm** for global importance and direction.
- **SHAP force plots** for a true positive, false positive and false negative.
- **Interpretation**: SHAP vs built-in importance, top-5 fraud drivers,
  surprising findings, and **business recommendations** tied to SHAP evidence.

See `reports/interim_3_report.md` and the end-to-end `reports/final_report.md`.

### Key findings

- **E-commerce:** `device_txn_count` (shared-device velocity) and
  `time_since_signup` (purchases moments after signup) dominate predictions.
- **Credit card:** anonymised components `V14`, `V4`, `V12`, `V10`, `V3` carry
  almost all the signal.
- Recommendations include step-up verification shortly after signup,
  rate-limiting shared devices, and a real-time anomaly score on the top card
  components.
