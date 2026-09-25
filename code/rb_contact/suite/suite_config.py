"""
suite_config.py
================
Shared configuration for the miss-distance model suite.

What this suite changes relative to the original rb_contact pipeline
--------------------------------------------------------------------
1. Swing population: bat_speed >= 50 mph, bunts excluded.
2. Stage-1 offset target is `miss_distance` (tracked, inches) with CONTACT
   ASSIGNED 0, replacing `barrel_distance_v2` (which spliced tracked
   miss_distance on whiffs with a launch-angle-derived proxy on contact).
3. Training season is 2025 in full -- spring training, regular season and
   postseason all included.
4. Evaluation season is 2026, scored by the 2025-trained models. No 2026 row
   ever enters a fit, so the evaluation cannot be contaminated by training.

Stage scope
-----------
The three variants are STAGE-1 only -- the two continuous regressors that
answer "given this pitch, what should the batter have done":

    timing  ->  int_y            (contact depth in inches)
    offset  ->  miss_distance    (0 on contact)

Stage 2 (the is_contact classifier) is unchanged and simply consumes whichever
variant's out-of-fold predictions. Bat-tracking variables are already in stage
2's raw feature set and stage 2 has no random effects, so variants 2 and 3
would be no-ops there.

The variants
------------
m1_merf_base      MERF (XGBoost fixed effects + per-batter random intercept),
                  the existing architecture and feature sets.
m2_merf_battrack  m1 + bat-tracking features (attack_angle, swing_path_tilt,
                  attack_direction) on BOTH regressors.
m3_rf_nore        Plain RandomForest. No random intercept, and the batter-mean
                  aggregate features are dropped from the timing model too --
                  those aggregates are batter identity by another name, so
                  leaving them in would defeat the point of the variant.
                  Feature set otherwise matches m2, so m2 -> m3 isolates the
                  random effect and m1 -> m2 isolates the bat-tracking block.

Zero-inflation note
-------------------
With contact assigned 0, ~78% of the offset target is exactly zero, and
`miss_distance == 0` is definitionally equivalent to `is_contact`. Both
regressors are still fit with a Gaussian/squared-error loss as specified.
Consequences to keep in mind when reading results: offset residuals are far
from normal, and `predicted_offset` is effectively a regression estimate of
P(contact), which makes the stage-2 hybrid model a two-level stack.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_ROOT = os.path.join(BASE_DIR, 'data')
# SUITE_OUT_DIR lets a smoke-test run write somewhere harmless instead of
# overwriting the real artifacts while a full run is in flight.
OUT_DIR   = os.environ.get('SUITE_OUT_DIR',
                           os.path.join(BASE_DIR, 'out', 'rb_contact_suite'))
PLOT_DIR  = os.path.join(OUT_DIR, 'plots')
MODEL_DIR = os.path.join(OUT_DIR, 'models')
for _d in (OUT_DIR, PLOT_DIR, MODEL_DIR):
    os.makedirs(_d, exist_ok=True)

TRAIN_SEASON = 2025
EVAL_SEASON  = 2026

N_FOLDS = 5
SEED    = 42

# ── Swing population ───────────────────────────────────────────────────────
MISS       = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
FOUL       = {'foul', 'foul_tip', 'foul_bunt', 'bunt_foul_tip'}
IN_PLAY    = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
BUNT_DESC  = {'missed_bunt', 'foul_bunt', 'bunt_foul_tip'}
ALL_SWINGS = MISS | FOUL | IN_PLAY
CONTACT    = FOUL | IN_PLAY

MIN_BAT_SPEED = 50.0

INTERCEPT_Y_COL = 'intercept_ball_minus_batter_pos_y_inches'

# ── Feature blocks ─────────────────────────────────────────────────────────
TIMING_BASE = ['release_speed_c', 'plate_x_bat_flip', 'plate_z',
               'pfx_x_bat_flip', 'pfx_z', 'same_hand']
OFFSET_BASE = ['release_speed_c', 'plate_x_bat_flip', 'plate_z', 'intercept_y']
BAT_TRACKING = ['attack_angle', 'swing_path_tilt', 'attack_direction']

TIMING_AGG = ['bat_speed', 'release_speed']   # batter-level means

XGB_TIMING = dict(n_estimators=400, max_depth=5, learning_rate=0.04,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                  reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
                  n_jobs=-1, verbosity=0)
XGB_OFFSET = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                  reg_alpha=0.1, reg_lambda=1.0, random_state=SEED,
                  n_jobs=-1, verbosity=0)

# n_jobs=2 not -1: the box has 4 cores and MERF/XGB already thread internally.
RF_PARAMS = dict(n_estimators=300, max_depth=18, min_samples_leaf=20,
                 max_features='sqrt', random_state=SEED, n_jobs=2)


def _variant(name, kind, timing_feats, offset_feats, timing_agg,
             offset_train_subset='all'):
    """
    offset_train_subset:
      'all'        fit the offset model on every swing, contact carrying the
                   assigned 0 (~78% of rows)
      'miss_only'  fit on whiffs only, where miss_distance is genuinely
                   tracked, then still PREDICT for every swing
    """
    return dict(
        name=name, kind=kind,
        timing=dict(name='timing', outcome_col='int_y', features=timing_feats,
                    batter_agg_features=timing_agg, xgb_params=XGB_TIMING,
                    max_iter=15, train_subset='all'),
        offset=dict(name='offset', outcome_col='miss_distance_t',
                    features=offset_feats, batter_agg_features=[],
                    xgb_params=XGB_OFFSET, max_iter=20,
                    train_subset=offset_train_subset),
    )


VARIANTS = {
    'm1_merf_base': _variant(
        'm1_merf_base', 'merf',
        TIMING_BASE, OFFSET_BASE, TIMING_AGG),
    'm2_merf_battrack': _variant(
        'm2_merf_battrack', 'merf',
        TIMING_BASE + BAT_TRACKING, OFFSET_BASE + BAT_TRACKING, TIMING_AGG),
    'm3_rf_nore': _variant(
        'm3_rf_nore', 'rf',
        TIMING_BASE + BAT_TRACKING, OFFSET_BASE + BAT_TRACKING, []),
    # Identical to m3 in every respect EXCEPT that the offset model is fit on
    # whiffs only. m3 -> m4 therefore isolates one thing: whether the ~78%
    # block of assigned zeros helps or hurts the offset model's learning.
    # It still predicts for all swings, so its predicted_offset answers "if
    # this swing missed, by how much" rather than "how far from the ball will
    # this swing end up" -- a different quantity, and notably one that is NOT
    # a restatement of P(contact).
    'm4_rf_missonly': _variant(
        'm4_rf_missonly', 'rf',
        TIMING_BASE + BAT_TRACKING, OFFSET_BASE + BAT_TRACKING, [],
        offset_train_subset='miss_only'),
}

VARIANT_ORDER = ['m1_merf_base', 'm2_merf_battrack', 'm3_rf_nore',
                 'm4_rf_missonly']
TARGETS = ['timing', 'offset']

# Random-effect handling when scoring the evaluation season.
#   carry    -- each batter keeps the random intercept fit on their 2025 swings;
#               a batter with no 2025 swings falls through to fixed effects
#   context  -- random intercept forced to 0 for everyone (league-average batter)
# For the RF variant the two are identical by construction; both are still
# written so downstream code can treat all variants uniformly.
RE_MODES = ['carry', 'context']
PLACEHOLDER_BATTER = '__CONTEXT_ONLY_PLACEHOLDER__'


def oof_path(variant: str) -> str:
    return os.path.join(OUT_DIR, f'stage1_oof_{variant}.parquet')


def eval_scored_path(variant: str) -> str:
    return os.path.join(OUT_DIR, f'stage1_eval{EVAL_SEASON}_{variant}.parquet')


def probs_path(variant: str, season: int) -> str:
    return os.path.join(OUT_DIR, f'stage2_probs_{variant}_{season}.parquet')
