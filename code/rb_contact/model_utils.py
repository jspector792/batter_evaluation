"""
model_utils.py
===============
Shared utilities for RB-Contact% Steps 1-2: feature assembly, model fitting
(GBM + logistic/spline), calibration, and evaluation metrics. Used by both
step1a (X_raw only) and step1b (X_hybrid, once OOF predictions exist).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import log_loss, brier_score_loss
from xgboost import XGBClassifier

CONTINUOUS_RAW = [
    'bat_speed', 'swing_length', 'swing_path_tilt', 'attack_angle', 'attack_direction',
    'release_speed', 'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z',
    'plate_x', 'plate_z', 'zone',
]
CATEGORICAL_RAW = ['pitch_type']
BINARY_RAW = ['same_hand']

X_RAW_COLS = CONTINUOUS_RAW + CATEGORICAL_RAW + BINARY_RAW
X_HYBRID_EXTRA = ['predicted_timing', 'predicted_offset']

# A small, curated set of domain-motivated interactions for the logistic/GAM
# comparison model -- NOT run through PolynomialFeatures on the full
# spline-expanded basis. That was tried first and produced ~5,900 columns
# (13 continuous features x ~7 spline basis functions each, all pairwise
# products) -- an ~12GB dense design matrix per fold that made the job crawl
# and balloon in memory. A handful of explicit, interpretable products is
# both cheaper and closer to what "explicit interaction terms" as an
# interpretable comparison point actually means.
CURATED_INTERACTIONS = [
    ('bat_speed', 'attack_angle'),
    ('plate_x', 'plate_z'),
    ('release_speed', 'plate_z'),
    ('pfx_x', 'pfx_z'),
    ('swing_length', 'bat_speed'),
]


def assemble_design(df: pd.DataFrame, extra_continuous: list = None,
                    pitch_type_categories: list = None) -> pd.DataFrame:
    """
    One-hot encode pitch_type, add curated interaction columns, keep everything
    else as-is. Returns a design matrix ready for either the GBM or the
    spline/logistic pipeline to consume.

    pitch_type_categories must be the FULL category list (fit on the whole
    swing population, not just this split) -- otherwise a rare pitch type
    (e.g. forkball 'FO') present in one fold/split but absent from another
    produces mismatched dummy columns between train and test (hit this in
    practice: XGBoost's inplace_predict raises "feature_names mismatch" when
    the test fold happens to be missing a category the train fold has).
    """
    extra_continuous = extra_continuous or []
    cols = CONTINUOUS_RAW + extra_continuous + BINARY_RAW
    out = df[cols].copy()
    for a, b in CURATED_INTERACTIONS:
        out[f'{a}_x_{b}'] = df[a] * df[b]
    pt = pd.Categorical(df['pitch_type'], categories=pitch_type_categories)
    dummies = pd.get_dummies(pt, prefix='pt', drop_first=False)
    return pd.concat([out.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


def interaction_cols() -> list:
    return [f'{a}_x_{b}' for a, b in CURATED_INTERACTIONS]


def make_gbm() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1,
        eval_metric='logloss', verbosity=0,
    )


def make_logistic_spline(continuous_cols: list, other_cols: list) -> Pipeline:
    """
    Logistic regression with additive natural-spline basis expansion on
    continuous predictors (smooth main effects, ~7 basis functions each --
    stands in for a GAM since pygam isn't installed here; see
    out/rb_contact/data_audit.md §6) plus a curated set of explicit
    interaction terms (CURATED_INTERACTIONS, added upstream in
    assemble_design) passed through unexpanded. No PolynomialFeatures blowup
    across the full spline basis -- see CURATED_INTERACTIONS docstring for
    why (~5,900-column design matrix, ~12GB/fold, crawled/ballooned memory).
    """
    pre = ColumnTransformer([
        ('spline', Pipeline([
            ('scale', StandardScaler()),
            ('spline', SplineTransformer(n_knots=5, degree=3, include_bias=False)),
        ]), continuous_cols),
        ('passthrough', 'passthrough', other_cols),
    ])
    return Pipeline([
        ('pre', pre),
        ('clf', LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')),
    ])


def calibrate(fitted_model, X_calib, y_calib, method='isotonic'):
    """Post-hoc calibration on a held-out split, per Step 2.3. `cv='prefit'`
    was removed in this sklearn version -- FrozenEstimator is the replacement
    for wrapping an already-fitted model so CalibratedClassifierCV doesn't
    refit it."""
    cal = CalibratedClassifierCV(FrozenEstimator(fitted_model), method=method)
    cal.fit(X_calib, y_calib)
    return cal


def evaluate(model, X_test, y_test, n_bins=10) -> dict:
    p = model.predict_proba(X_test)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    misclass = ((p >= 0.5).astype(int) != y_test).mean()
    frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=n_bins, strategy='quantile')
    return dict(
        log_loss=log_loss(y_test, p),
        brier=brier_score_loss(y_test, p),
        misclass_rate=misclass,
        n=len(y_test),
        calib_mean_pred=mean_pred.tolist(),
        calib_frac_pos=frac_pos.tolist(),
    )
