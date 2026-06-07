# Interim-1 Report — Task 1: Data Analysis & Preprocessing

**Project:** Improved detection of fraud cases for e-commerce and bank transactions
**Organization:** Adey Innovations Inc.

> Reproduce all numbers/figures by placing the three CSVs in `data/raw/` and running
> `notebooks/01_data_analysis_preprocessing.ipynb` (or `python -m src.pipeline`).
> Figures are written to `reports/figures/`.

---

## 1. Datasets

| Dataset                    | Granularity            | Target  | Notes                                    |
| -------------------------- | ---------------------- | ------- | ---------------------------------------- |
| `Fraud_Data.csv`           | One e-commerce purchase | `class` | Rich behavioural + identity features     |
| `IpAddress_to_Country.csv` | IP range → country      | —       | Used to geolocate transactions           |
| `creditcard.csv`           | One card transaction    | `Class` | Anonymised PCA features `V1..V28`        |

## 2. Data cleaning & preprocessing

Implemented in `src/preprocessing.py` and `src/data_loader.py`:

1. **Type parsing** — `signup_time` and `purchase_time` parsed to `datetime`;
   IP bounds coerced to integers.
2. **Duplicates** — exact duplicate rows dropped (`basic_clean`).
3. **Missing values** (`handle_missing_values`):
   - Columns >50% missing are dropped.
   - Numeric columns imputed with the **median** (robust to the heavy-tailed
     `purchase_value`).
   - Categorical columns imputed with the **mode**.
4. **Dtypes** — categorical columns (`source`, `browser`, `sex`, `country`)
   cast to `category` for memory and clarity.

> The `creditcard.csv` dataset is already numeric/PCA-transformed and is
> typically complete; cleaning mainly removes duplicate transactions.

## 3. EDA insights

Figures (saved under `reports/figures/`):

- **Class balance** (`*_class_balance.png`) — both datasets are highly
  imbalanced. E-commerce fraud is roughly **9–10%** of rows; credit-card fraud
  is **~0.17%**. This dictates the metric choice (AUC-PR / F1, not accuracy).
- **`purchase_value` / `age` by class** — distributions overlap heavily;
  amount alone is a weak separator, motivating engineered features.
- **Fraud rate by category** (`fraud_rate_source/browser/sex.png`) — fraud rate
  varies by acquisition `source` and `browser`, making them useful predictors.
- **Correlation heatmap** — engineered numeric features add signal beyond the
  raw columns.

## 4. Feature engineering choices

Implemented in `src/feature_engineering.py`:

- **`time_since_signup`** (hours between signup and purchase) — the headline
  feature. Fraudulent accounts tend to purchase almost immediately after
  signup, so the median `time_since_signup` for fraud is dramatically lower than
  for legitimate transactions (see `time_since_signup_by_class.png`).
- **Time-based** — `purchase_hour` and `purchase_dayofweek` capture diurnal /
  weekly patterns in fraudulent activity.
- **Velocity / frequency** — `user_txn_count`, `device_txn_count`, and
  `device_shared` flag accounts/devices reused across many transactions, a
  strong fraud-ring indicator.

## 5. IP-to-country mapping

Implemented in `src/geolocation.py`. The numeric `ip_address` is matched against
`[lower_bound, upper_bound]` ranges using a **sorted `merge_asof`** (backward
search) followed by upper-bound validation — `O((n+m) log m)` instead of an
`O(n*m)` cross-join. IPs outside all ranges are labelled `Unknown`. The mapped
`country` exposes geographic concentration of fraud (a small set of countries
carries disproportionately high fraud rates among segments with sufficient
volume).

## 6. Encoding, scaling & class-imbalance strategy

- **Encoding** — one-hot for `source`, `browser`, `sex`, `country`
  (`min_frequency` collapses rare countries; unknown categories ignored at
  inference).
- **Scaling** — `StandardScaler` on numeric features (needed by Logistic
  Regression).
- **Leakage control** — the `ColumnTransformer` is fit on the **training split
  only**.
- **Class imbalance** — handled in Task 2:
  - Evaluate with **AUC-PR** and **F1** rather than accuracy.
  - **SMOTE applied to the training fold only** (no leakage into validation).
  - Tree-based models additionally use class weights.

## 7. Artifacts

- `data/processed/fraud_processed.csv`, `data/processed/creditcard_processed.csv`
- Figures in `reports/figures/`
- Reusable code in `src/` (`data_loader`, `preprocessing`, `geolocation`,
  `feature_engineering`, `transform`, `eda`, `pipeline`).
