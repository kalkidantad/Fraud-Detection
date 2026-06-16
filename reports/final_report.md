# Fraud Detection for E-Commerce and Bank Transactions — Final Report

**Project:** Improved detection of fraud cases for e-commerce and bank transactions
**Organization:** Adey Innovations Inc.
**Author:** 10 Academy, Week 5 & 6 challenge

This report narrates the project end-to-end: data analysis and feature
engineering (Task 1), model building and selection (Task 2), and model
explainability with SHAP plus the resulting business recommendations (Task 3).
Every number and figure is reproducible from the code in `src/` and the three
notebooks in `notebooks/`.

---

## 1. Problem framing

Adey Innovations needs to flag fraudulent transactions across two very
different channels:

| Dataset | Rows | Fraud rate | Feature style |
| --- | ---: | ---: | --- |
| `Fraud_Data.csv` (e-commerce) | 151,112 | ~9.4% | Behavioural + identity + geo |
| `creditcard.csv` (bank) | 283,726 (deduped) | ~0.17% | Anonymised PCA `V1..V28` + `Amount`, `Time` |

These are **separate modeling problems** with different feature structures, so
we build **independent pipelines** for each and never mix their features.

Because both datasets are highly imbalanced, **accuracy is misleading** (a
model predicting "never fraud" scores 99.83% accuracy on the card data while
catching zero fraud). We therefore lead with **AUC-PR** (average precision) and
**F1**, supported by precision, recall and the confusion matrix.

---

## 2. Task 1 — Data analysis & feature engineering

### 2.1 Cleaning

- Parsed `signup_time` / `purchase_time` to `datetime`; coerced IP bounds to
  integers.
- Dropped exact duplicate rows.
- Imputed missing numeric values with the **median** (robust to the
  heavy-tailed `purchase_value`) and categoricals with the **mode**; dropped any
  column >50% missing.

### 2.2 IP → country geolocation

The numeric `ip_address` is matched to `[lower, upper]` ranges with a sorted
**`merge_asof`** (backward search) + upper-bound validation — `O((n+m) log m)`
instead of an `O(n·m)` cross-join. Out-of-range IPs become `Unknown`.

### 2.3 Engineered features (the heart of the e-commerce signal)

| Feature | Rationale |
| --- | --- |
| `time_since_signup` (hours) | Fraud accounts buy almost immediately after signup. |
| `purchase_hour`, `purchase_dayofweek` | Diurnal / weekly fraud patterns. |
| `user_txn_count`, `device_txn_count` | Velocity — reuse across transactions. |
| `device_shared` | Multiple accounts on one device = fraud-ring tell. |

### 2.4 EDA highlights

- Class-balance plots confirm severe imbalance in both datasets.
- `purchase_value` / `age` distributions overlap heavily by class — raw amount
  is a weak separator, justifying the engineered behavioural features.
- Fraud rate varies meaningfully by acquisition `source`, `browser` and
  `country`.

Figures: `reports/figures/*_class_balance.png`, `time_since_signup_by_class.png`,
`fraud_rate_*.png`, correlation heatmap.

---

## 3. Task 2 — Model building, evaluation & selection

### 3.1 Methodology

- **Stratified 80/20 split** preserves the fraud ratio in both folds.
- **Leakage control:** preprocessing (`StandardScaler` + `OneHotEncoder`) and
  **SMOTE** are wrapped in an imbalanced-learn `Pipeline`, so the scaler is fit
  and SMOTE resamples on the **training fold only** — never the test set.
- Two models per dataset: **Logistic Regression** (interpretable baseline,
  `class_weight="balanced"` + SMOTE) and **XGBoost** (ensemble,
  `scale_pos_weight = neg/pos`).

### 3.2 Results

**E-commerce (`Fraud_Data`)**

| Model | AUC-PR | ROC-AUC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6651 | 0.8375 | 0.6014 | 0.5194 | 0.7141 |
| **XGBoost** | **0.7087** | **0.8404** | **0.6174** | **0.5670** | 0.6777 |

**Credit card (`creditcard`)**

| Model | AUC-PR | ROC-AUC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6750 | 0.9626 | 0.1000 | 0.0530 | 0.8737 |
| **XGBoost** | **0.8287** | **0.9773** | **0.8671** | **0.9615** | 0.7895 |

### 3.3 Selection & justification

**XGBoost is selected for both datasets.** It wins on AUC-PR and F1
everywhere. The credit-card gap is decisive: Logistic Regression catches 87% of
fraud but at 5.3% precision (F1 = 0.10) — it would bury analysts in false
alarms — whereas XGBoost reaches **0.96 precision at 0.79 recall** (F1 = 0.87),
the only operationally usable model on the ~0.17% imbalance. ROC-AUC looks
high for everything on the card data, which is exactly why we lead with AUC-PR.

Fitted pipelines are serialized to `models/xgb_ecommerce.joblib` and
`models/xgb_creditcard.joblib`.

Figures: `reports/figures/pr_{ecommerce,creditcard}.png`,
`confusion_*_{ecommerce,creditcard}.png`.

---

## 4. Task 3 — Explainability with SHAP

We interpret the selected XGBoost models with XGBoost's built-in gain
importance and with **exact tree-SHAP** on the held-out test split (same
`random_state=42` split as Task 2). Because the model is a pipeline, SHAP runs
on the **transformed** matrix; force-plot feature values are in scaled units.

### 4.1 Global importance — e-commerce

![Built-in importance — e-commerce](figures/importance_builtin_ecommerce.png)

![SHAP summary — e-commerce](figures/shap_summary_ecommerce.png)

Top-5 SHAP drivers: `device_txn_count` (1.122), `time_since_signup` (0.673),
`purchase_value` (0.172), `age` (0.141), `purchase_hour` (0.128). High
`device_txn_count` and *short* `time_since_signup` both push hard toward fraud.

### 4.2 Global importance — credit card

![Built-in importance — credit card](figures/importance_builtin_creditcard.png)

![SHAP summary — credit card](figures/shap_summary_creditcard.png)

Top-5 SHAP drivers: `V14` (2.907), `V4` (1.999), `V12` (1.442), `V10` (0.973),
`V3` (0.960). Fraud concentrates in the extreme tails of a handful of PCA
components; `Amount` also reaches the SHAP top-10.

### 4.3 Local explanations (force plots)

E-commerce:

| True positive | False positive | False negative |
| --- | --- | --- |
| ![TP](figures/shap_force_ecommerce_true_positive.png) | ![FP](figures/shap_force_ecommerce_false_positive.png) | ![FN](figures/shap_force_ecommerce_false_negative.png) |

- **TP (prob 1.000):** short signup gap + high device count → confident catch.
- **FP (prob 0.9999):** a legitimate buyer on a shared device who bought
  quickly — the two strongest fraud features both fire.
- **FN (prob 0.0004):** real fraud the model missed; only signup-recency
  flagged it while the (non-shared) device and everything else looked normal.

Credit card:

| True positive | False positive | False negative |
| --- | --- | --- |
| ![TP](figures/shap_force_creditcard_true_positive.png) | ![FP](figures/shap_force_creditcard_false_positive.png) | ![FN](figures/shap_force_creditcard_false_negative.png) |

### 4.4 SHAP vs built-in importance

Both methods agree on the dominant features (`device_txn_count` /
`time_since_signup` for e-commerce; `V14`/`V12`/`V4`/`V10`/`V3` for cards).
They diverge below the top: gain importance over-credits sparse one-hot
categoricals (browsers, countries), while SHAP correctly elevates the
continuous behavioural features (`purchase_value`, `age`, `purchase_hour`) and
`Amount`. We trust SHAP's contribution-based ranking for decisions.

### 4.5 Surprising findings

- **The e-commerce model is essentially a velocity detector** —
  `device_txn_count` alone is ~56% of gain. Powerful but fragile: rotate
  devices and you evade it (exactly the false negative we found).
- **`time_since_signup` acts as a hard threshold, not a smooth trend** — short
  gaps form an isolated high-fraud cluster in the beeswarm.
- **Built-in importance can mislead** (high-cardinality bias), so business
  decisions should be driven by SHAP.

---

## 5. Business recommendations

1. **Step-up verification within the first hour after signup** — `time_since_signup`
   is the #2 driver with a distinct early-purchase fraud cluster.
2. **Rate-limit / review devices shared across many accounts** — `device_txn_count`
   is the #1 driver; pair with rule 1 to limit false positives on legitimate
   shared devices.
3. **Real-time anomaly score on `V14/V4/V12/V10/V3`** for card transactions,
   auto-holding extreme cases and routing borderline ones to manual review.
4. **Close the velocity blind spot** by adding IP-velocity, billing/shipping
   mismatch and email-domain-age features so detection survives device rotation.
5. **Treat `Amount` as an explicit risk input** for cards (top-10 by SHAP
   despite low gain rank).

Full evidence per recommendation is in `reports/interim_3_report.md`.

---

## 6. Reproducibility

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the 3 CSVs in data/raw/ (or data/)
python -m src.pipeline                                   # Task 1: processed data
jupyter nbconvert --to notebook --execute notebooks/02_model_building.ipynb   # Task 2: models
python -m src.explainability                             # Task 3: SHAP figures
```

- Reusable code: `src/` (`data_loader`, `preprocessing`, `geolocation`,
  `feature_engineering`, `transform`, `modeling`, `evaluation`, `explainability`,
  `pipeline`).
- Notebooks: `notebooks/01..03`.
- All randomness seeded via `config.RANDOM_STATE = 42`.
