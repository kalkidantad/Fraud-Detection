"""Model building for Task 2.

Two models per dataset:
- **Logistic Regression** — interpretable baseline (with ``class_weight`` to
  cope with imbalance).
- **XGBoost** — high-performance ensemble (with ``scale_pos_weight``).

Resampling (SMOTE) is wrapped inside an imbalanced-learn ``Pipeline`` so it is
applied to the training fold *only* during ``fit`` — never to validation/test
data. This prevents leakage.
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression

from . import config

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_XGB = False


def scale_pos_weight(y) -> float:
    """Ratio of negatives to positives, used to weight the positive class."""
    y = np.asarray(y)
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(neg / max(pos, 1))


def build_logreg_pipeline(preprocessor, use_smote: bool = True) -> ImbPipeline:
    """Preprocess -> (SMOTE) -> Logistic Regression."""
    steps = [("preprocess", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
    steps.append(
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
            ),
        )
    )
    return ImbPipeline(steps)


def build_xgb_pipeline(preprocessor, y_train, use_smote: bool = False) -> ImbPipeline:
    """Preprocess -> (SMOTE) -> XGBoost.

    By default XGBoost uses ``scale_pos_weight`` instead of SMOTE, which is
    usually more stable for tree ensembles on extreme imbalance.
    """
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed. `pip install xgboost`.")

    steps = [("preprocess", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
        spw = 1.0  # SMOTE already balances the classes
    else:
        spw = scale_pos_weight(y_train)

    steps.append(
        (
            "clf",
            XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=spw,
                eval_metric="aucpr",
                n_jobs=-1,
                random_state=config.RANDOM_STATE,
            ),
        )
    )
    return ImbPipeline(steps)
