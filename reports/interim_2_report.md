# Interim-2 Report — Task 2: Model Building & Training

**Project:** Improved detection of fraud cases for e-commerce and bank transactions
**Organization:** Adey Innovations Inc.

> Builds on Interim-1. Reproduce by running
> `notebooks/02_model_building.ipynb` after Task 1 has produced the processed
> datasets (or it rebuilds them automatically).

---

## 1. Approach

For **both** datasets (`Fraud_Data` and `creditcard`) we train and compare:

| Model                  | Role                | Imbalance handling                      |
| ---------------------- | ------------------- | --------------------------------------- |
| Logistic Regression    | Interpretable baseline | `class_weight="balanced"` + SMOTE     |
| XGBoost                | Ensemble (best)     | `scale_pos_weight = neg/pos`            |

## 2. Methodology

1. **Stratified train/test split** (80/20) preserves the fraud ratio in both
   folds (`transform.stratified_split`).
2. **Pipeline + leakage control** — preprocessing (scaling + one-hot encoding)
   and SMOTE are wrapped in an **imbalanced-learn `Pipeline`**, so the scaler is
   fit and SMOTE resamples on the **training fold only**. Validation/test data
   are never resampled (`src/modeling.py`).
3. **Metrics** — because the classes are highly imbalanced, accuracy is
   misleading. We report:
   - **AUC-PR (average precision)** — primary metric.
   - **F1**, **precision**, **recall**.
   - **ROC-AUC** and the **confusion matrix**.
   (`src/evaluation.py`)

## 3. Results

The numbers below come from running `notebooks/02_model_building.ipynb` on the
processed datasets (e-commerce: 151,112 rows, 9.36% fraud; credit card: 283,726
rows after de-duplication, ~0.17% fraud) with an 80/20 stratified split. Curves
and confusion matrices are saved to `reports/figures/` (`pr_ecommerce.png`,
`pr_creditcard.png`, `confusion_*.png`).

### 3.1 E-commerce (`Fraud_Data`)

| Model               | AUC-PR | ROC-AUC |    F1 | Precision | Recall |
| ------------------- | -----: | ------: | ----: | --------: | -----: |
| Logistic Regression | 0.6651 |  0.8375 | 0.6014 |    0.5194 | 0.7141 |
| **XGBoost**         | **0.7087** | **0.8404** | **0.6174** | **0.5670** | 0.6777 |

### 3.2 Credit card (`creditcard`)

| Model               | AUC-PR | ROC-AUC |    F1 | Precision | Recall |
| ------------------- | -----: | ------: | ----: | --------: | -----: |
| Logistic Regression | 0.6750 |  0.9626 | 0.1000 |    0.0530 | 0.8737 |
| **XGBoost**         | **0.8287** | **0.9773** | **0.8671** | **0.9615** | 0.7895 |

**Observations:**

- XGBoost beats Logistic Regression on AUC-PR and F1 on **both** datasets.
- On `creditcard` the gap is dramatic: Logistic Regression catches most fraud
  (87% recall) but at terrible precision (5.3%), giving an F1 of just 0.10 — it
  floods the analyst with false positives. XGBoost reaches 0.96 precision at
  0.79 recall (F1 = 0.87), the only model that is operationally usable on the
  extreme ~0.17% imbalance.
- ROC-AUC looks high for every model on `creditcard`, which is exactly why we
  lead with AUC-PR: it is the metric that actually separates the two models.

## 4. Model selection

**XGBoost is selected** as the production model for both datasets based on
AUC-PR / F1. Fitted pipelines are serialized to `models/` (`xgb_ecommerce.joblib`,
`xgb_creditcard.joblib`) for the explainability work in Task 3 (SHAP).

## 5. Reproducibility

- All randomness is seeded via `config.RANDOM_STATE`.
- Code: `src/modeling.py`, `src/evaluation.py`, `src/transform.py`.
- Notebook: `notebooks/02_model_building.ipynb`.
