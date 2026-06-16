# Interim-3 Report — Task 3: Model Explainability (SHAP)

**Project:** Improved detection of fraud cases for e-commerce and bank transactions
**Organization:** Adey Innovations Inc.

> Reproduce by running `notebooks/03_model_explainability.ipynb` (or
> `python -m src.explainability`) after Tasks 1–2 have produced the processed
> data and the `models/xgb_*.joblib` files. All figures referenced below live
> in `reports/figures/`.

---

## 1. Objective & approach

Task 2 selected **XGBoost** as the production model for *both* datasets (best
AUC-PR / F1). Task 3 opens that black box. We use two complementary lenses:

| Lens | What it answers | How |
| --- | --- | --- |
| **Built-in importance** | Which features did the trees split on most? | XGBoost gain importance |
| **SHAP** | How much, and in which direction, does each feature move each prediction? | exact `TreeExplainer` |

SHAP is computed on the **held-out test split** (never seen in training) using
the exact same stratified split as Task 2 (`random_state=42`), so the
true-positive / false-positive / false-negative cases we dissect are
reproducible. Because the model is a pipeline
(`StandardScaler` + `OneHotEncoder` → `XGBClassifier`), SHAP runs on the
**transformed** design matrix; feature values inside force plots are therefore
in *scaled* units (a negative `time_since_signup` = shorter-than-average gap).

---

## 2. E-commerce model (`Fraud_Data`)

### 2.1 Built-in feature importance (top 10)

`reports/figures/importance_builtin_ecommerce.png`

| Rank | Feature | Gain importance |
| ---: | --- | ---: |
| 1 | `device_txn_count` | 0.555 |
| 2 | `time_since_signup` | 0.038 |
| 3 | `source_Direct` | 0.017 |
| 4 | `device_shared` | 0.016 |
| 5 | `country_Korea Republic of` | 0.015 |
| 6 | `browser_Chrome` | 0.015 |
| 7 | `browser_Opera` | 0.015 |
| 8 | `country_United States` | 0.014 |
| 9 | `browser_IE` | 0.014 |
| 10 | `source_SEO` | 0.014 |

`device_txn_count` alone accounts for ~56% of total gain — the model leans
heavily on transaction velocity per device.

### 2.2 SHAP global importance (top 10, mean |SHAP|)

`reports/figures/shap_summary_ecommerce.png` · `reports/figures/shap_bar_ecommerce.png`

| Rank | Feature | mean |SHAP| |
| ---: | --- | ---: |
| 1 | `device_txn_count` | 1.122 |
| 2 | `time_since_signup` | 0.673 |
| 3 | `purchase_value` | 0.172 |
| 4 | `age` | 0.141 |
| 5 | `purchase_hour` | 0.128 |
| 6 | `purchase_dayofweek` | 0.083 |
| 7 | `source_Direct` | 0.040 |
| 8 | `browser_Chrome` | 0.034 |
| 9 | `sex_F` | 0.031 |
| 10 | `source_SEO` | 0.031 |

**Direction (from the beeswarm):**

- **`device_txn_count`** — high values (red) push strongly toward **fraud**.
  Many transactions on one device = classic fraud-ring behaviour.
- **`time_since_signup`** — *low* values (blue, short gap) form a distinct
  cluster at very high positive SHAP (≈ +8). Accounts that purchase within
  minutes of signup are the single clearest fraud signature.
- **`purchase_value`**, **`age`**, **`purchase_hour`** add secondary, more
  symmetric contributions.

### 2.3 Local explanations (force plots)

| Case | Fraud prob. | What the force plot shows |
| --- | ---: | --- |
| **True positive** (`shap_force_ecommerce_true_positive.png`) | 1.000 | Short `time_since_signup` and high `device_txn_count` both push hard toward fraud → f(x) ≈ 11.7. A textbook catch. |
| **False positive** (`shap_force_ecommerce_false_positive.png`) | 0.9999 | A legitimate buyer with a *shared* device (`device_txn_count` high) **and** a quick first purchase. The two strongest fraud features both fire, so the model over-commits → f(x) ≈ 9.2. |
| **False negative** (`shap_force_ecommerce_false_negative.png`) | 0.0004 | Real fraud the model missed. Only `time_since_signup` flagged it (red); the device was *not* shared and every other feature looked normal (blue), dragging f(x) to ≈ −7.8. |

The FP and FN are mirror images of the same lesson: the model is almost
**entirely** driven by velocity + signup-recency, so fraud that doesn't fit
that pattern slips through, and legitimate users who happen to match it get
flagged.

### 2.4 SHAP vs built-in importance

- **Agreement at the top:** both rank `device_txn_count` #1 and
  `time_since_signup` #2 — the two dominant drivers are robust to the method.
- **Disagreement below:** built-in importance promotes sparse one-hot columns
  (`source_Direct`, `country_*`, individual browsers), whereas SHAP elevates
  the **continuous behavioural** features `purchase_value`, `age`,
  `purchase_hour`. Gain importance is biased toward high-cardinality / many-split
  categoricals; SHAP measures *actual contribution to predictions* and is the
  more trustworthy ranking for decision-making.

---

## 3. Credit-card model (`creditcard`)

### 3.1 Built-in feature importance (top 10)

`reports/figures/importance_builtin_creditcard.png`

| Rank | Feature | Gain importance |
| ---: | --- | ---: |
| 1 | `V14` | 0.461 |
| 2 | `V12` | 0.128 |
| 3 | `V4` | 0.058 |
| 4 | `V10` | 0.050 |
| 5 | `V3` | 0.031 |
| 6 | `V8` | 0.031 |
| 7 | `V19` | 0.021 |
| 8 | `V17` | 0.020 |
| 9 | `V18` | 0.016 |
| 10 | `V20` | 0.015 |

### 3.2 SHAP global importance (top 10, mean |SHAP|)

`reports/figures/shap_summary_creditcard.png` · `reports/figures/shap_bar_creditcard.png`

| Rank | Feature | mean |SHAP| |
| ---: | --- | ---: |
| 1 | `V14` | 2.907 |
| 2 | `V4` | 1.999 |
| 3 | `V12` | 1.442 |
| 4 | `V10` | 0.973 |
| 5 | `V3` | 0.960 |
| 6 | `V11` | 0.934 |
| 7 | `V8` | 0.581 |
| 8 | `V19` | 0.551 |
| 9 | `Amount` | 0.429 |
| 10 | `V1` | 0.427 |

The features are anonymised PCA components, so we describe *behaviour* not
semantics: from the beeswarm, **low `V14` and low `V12`/`V10` values push
strongly toward fraud**, while `V4` shows the opposite polarity. These
components are the model's learned "anomaly axes".

### 3.3 Local explanations (force plots)

| Case | Fraud prob. | Notes |
| --- | ---: | --- |
| **True positive** (`shap_force_creditcard_true_positive.png`) | 1.000 | `V14`, `V12`, `V10`, `V4` all align toward fraud — a strongly anomalous transaction. |
| **False positive** (`shap_force_creditcard_false_positive.png`) | 1.000 | A legitimate transaction that lands deep in the anomalous region of the same components. |
| **False negative** (`shap_force_creditcard_false_negative.png`) | ≈ 0.000 | Fraud whose component values look statistically normal — no anomaly axis fired. |

### 3.4 SHAP vs built-in importance

Both methods agree the signal is concentrated in a handful of components
(`V14`, `V12`, `V4`, `V10`, `V3`). The main divergence: built-in gain ranks
`V12` second, but SHAP ranks `V4` second and surfaces `V11` and `Amount` into
the top 10. Again, SHAP's contribution-based view is preferred, and it usefully
flags that **transaction `Amount` does carry independent signal** even though
gain importance buries it.

---

## 4. Top 5 drivers of fraud (combined view)

**E-commerce**

1. `device_txn_count` — shared-device / velocity (dominant).
2. `time_since_signup` — purchases moments after account creation.
3. `purchase_value` — transaction amount.
4. `age` — customer age band.
5. `purchase_hour` — time-of-day pattern.

**Credit card**

1. `V14`  2. `V4`  3. `V12`  4. `V10`  5. `V3` — anonymised anomaly components.

## 5. Surprising / counter-intuitive findings

- **One feature dominates e-commerce.** `device_txn_count` carries ~56% of gain
  and the largest mean|SHAP|. The model is effectively a velocity detector;
  this is powerful but fragile (see the FP/FN above) and creates a single point
  of evasion — a fraudster using a fresh device per transaction defeats it.
- **`time_since_signup` is bimodal, not linear.** SHAP shows short gaps create
  an *isolated* high-fraud cluster rather than a smooth trend — evidence for a
  hard "minutes-after-signup" rule rather than a gentle continuous score.
- **Built-in importance over-credits one-hot categoricals.** Several
  country/browser dummies rank high on gain but contribute little SHAP — a
  reminder not to drive business decisions from gain importance alone.
- **`Amount` matters more than gain suggests** on the credit-card model
  (top-10 by SHAP, outside top-10 by some gain orderings).

## 6. Business recommendations

Each recommendation is tied to a specific SHAP insight.

1. **Step-up verification for purchases shortly after signup.**
   *SHAP evidence:* `time_since_signup` is the #2 driver, and low values form a
   dedicated high-fraud cluster (beeswarm + the TP/FP force plots). **Action:**
   require an extra verification step (OTP / 3-D Secure) for any purchase made
   within the first ~1 hour of account creation. Expected to catch a large share
   of fraud at minimal friction to genuine customers.

2. **Monitor and rate-limit devices shared across accounts.**
   *SHAP evidence:* `device_txn_count` is the #1 driver (~56% of gain, highest
   mean|SHAP|); high values push hard toward fraud. **Action:** when one
   `device_id` is linked to many accounts/transactions, throttle it and route to
   review — but pair it with rule #1 to avoid the false positives we saw when a
   *legitimate* shared device buys quickly.

3. **Deploy a real-time anomaly score on the top credit-card components.**
   *SHAP evidence:* `V14`, `V4`, `V12`, `V10`, `V3` dominate; fraud lives in the
   extreme tails of these axes. **Action:** compute these components in-stream and
   auto-decline / hold transactions whose combined SHAP push exceeds a
   threshold, sending borderline cases to manual review.

4. **Reduce the velocity blind spot (model-improvement action).**
   *SHAP evidence:* the false negative was missed because only signup-recency
   fired while the device looked clean. **Action:** add features that don't
   collapse to a single device signal — IP-velocity, billing/shipping mismatch,
   email-domain age — so detection degrades gracefully when fraudsters rotate
   devices.

5. **Use `Amount` (and amount-velocity) as an explicit risk input** for cards.
   *SHAP evidence:* `Amount` reaches the SHAP top-10 despite low gain rank.
   **Action:** combine amount with the anomaly score so unusually large
   transactions in anomalous regions are prioritised for review.

## 7. Reproducibility

- Code: `src/explainability.py` (importance + SHAP utilities), built on the
  Task-1/2 modules.
- Notebook: `notebooks/03_model_explainability.ipynb`.
- Figures: `reports/figures/importance_builtin_*.png`, `shap_summary_*.png`,
  `shap_bar_*.png`, `shap_force_*_{true_positive,false_positive,false_negative}.png`.
- Determinism: same `random_state=42` split as Task 2; SHAP sample seeded.
