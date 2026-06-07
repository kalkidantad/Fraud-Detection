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

Run the notebook to populate the exact figures; the comparison table is printed
by `evaluation.metrics_table(...)` and curves/confusion matrices are saved to
`reports/figures/` (`pr_ecommerce.png`, `pr_creditcard.png`,
`confusion_*.png`).

**Expected pattern:** XGBoost outperforms Logistic Regression on AUC-PR and F1
for both datasets, while Logistic Regression provides an interpretable baseline.
On `creditcard` (≈0.17% fraud) AUC-PR is the only metric that meaningfully
separates the models.

## 4. Model selection

**XGBoost is selected** as the production model for both datasets based on
AUC-PR / F1. Fitted pipelines are serialized to `models/` (`xgb_ecommerce.joblib`,
`xgb_creditcard.joblib`) for the explainability work in Task 3 (SHAP).

## 5. Reproducibility

- All randomness is seeded via `config.RANDOM_STATE`.
- Code: `src/modeling.py`, `src/evaluation.py`, `src/transform.py`.
- Notebook: `notebooks/02_model_building.ipynb`.
