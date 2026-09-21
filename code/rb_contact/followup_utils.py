"""
followup_utils.py
==================
Shared loaders and meta-metric estimators for the three post-RB-Contact%
follow-up analyses (Directions A/B/C).

Everything here reads *cached* artifacts from out/rb_contact/ -- no model
refitting. The out-of-fold quantities these analyses depend on already exist:

  oof_predictions.parquet   row_key, batter, pitcher, is_contact, description,
                            fold, predicted_timing (OOF int_y),
                            predicted_offset (OOF barrel_distance_v2),
                            p_contact (OOF hybrid GBM contact probability)
  oof_contact_probs.parquet row_key, batter, game_date, is_contact, p_contact

`load_swing_frame()` re-derives the swing population via
oof_timing_offset.prepare_data() (so the population, bunt exclusion, and the
barrel_distance_v2 construction stay byte-identical to the rest of the
pipeline) and left-joins the OOF columns onto it.

Two additions this module makes that the main pipeline doesn't have:

1. **Reconstructed ball/strike count.** data_audit.md §2 flagged that the
   processed parquets carry no `balls`/`strikes` columns. They are fully
   recoverable from the pitch sequence: every at-bat's pitches are present
   and contiguous (verified: 194,314 at-bats, 100% have pitch_number running
   1..N with no gaps), and `description` says what each pitch did. Walking
   each at-bat in pitch order reproduces the count *before* each pitch
   exactly. Direction C needs this for its count-dependence queries.

2. **Franks et al. meta-metrics** (discrimination / stability) generalized to
   continuous per-swing metrics, not just binomial rates. step5's
   `discrimination_metric` hardcodes the p(1-p)/n sampling variance; here the
   sampling variance is supplied by the caller so a rate and a mean-of-inches
   metric can be compared in the same table.
"""

import os
import sys
import glob
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oof_timing_offset import prepare_data, OUT_DIR, DATA_DIR, C  # noqa: E402

warnings.filterwarnings('ignore')

FOLLOWUP_DIR = os.path.join(OUT_DIR, 'followups')
os.makedirs(FOLLOWUP_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATTER_NAMES_CSV = os.path.join(BASE_DIR, 'data', 'batter_stats_2025.csv')

# `description` -> effect on the count, for count reconstruction.
BALL_DESC = {'ball', 'blocked_ball', 'automatic_ball', 'intent_ball', 'pitchout'}
STRIKE_DESC = {'called_strike', 'swinging_strike', 'swinging_strike_blocked',
               'missed_bunt', 'automatic_strike'}
# Fouls add a strike only with < 2 strikes -- except foul_bunt, which is a
# strikeout with 2 strikes, so it always increments.
FOUL_DESC = {'foul', 'foul_tip', 'bunt_foul_tip', 'foul_pitchout'}
ALWAYS_STRIKE_FOUL_DESC = {'foul_bunt'}

# Order that defines "chronological" for a batter's swings. game_date alone
# leaves same-day swings unordered (and a batter can have two games in a day),
# which would make Direction A's "first N% of the season" split depend on row
# order in the parquet.
CHRON_KEYS = ['game_date', 'game_pk', 'at_bat_number', 'pitch_number']


# ──────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────

def walk_counts(descriptions: np.ndarray, ab_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Walk pitches in order and return (balls, strikes) *entering* each pitch.

    `ab_id` must be an at-bat grouping key, and the arrays must already be
    sorted so each at-bat's pitches are contiguous and in pitch order. The
    count resets whenever `ab_id` changes.

    Pure function over arrays so it can be unit-tested without touching the
    parquet files (see tests/test_followups.py).
    """
    desc = np.asarray(descriptions)
    is_ball = np.isin(desc, list(BALL_DESC))
    is_strike = np.isin(desc, list(STRIKE_DESC))
    is_foul = np.isin(desc, list(FOUL_DESC))
    is_always_strike_foul = np.isin(desc, list(ALWAYS_STRIKE_FOUL_DESC))

    n = len(desc)
    balls = np.zeros(n, dtype=np.int8)
    strikes = np.zeros(n, dtype=np.int8)

    b = s = 0
    prev_ab = object()
    for i in range(n):
        if ab_id[i] != prev_ab:
            b = s = 0
            prev_ab = ab_id[i]
        balls[i], strikes[i] = b, s
        if is_ball[i]:
            b = min(b + 1, 3)
        elif is_strike[i] or is_always_strike_foul[i]:
            s = min(s + 1, 2)
        elif is_foul[i] and s < 2:
            s += 1
    return balls, strikes


def reconstruct_counts() -> pd.DataFrame:
    """
    Return row_key -> (balls, strikes, count_str) for every pitch in the
    season, where balls/strikes are the count *entering* that pitch.

    Reads the raw parquets directly (4 columns only) rather than taking
    prepare_data()'s output, because prepare_data() drops non-swing pitches --
    and the called strikes and balls it drops are exactly what moves the count.
    """
    cols = ['game_pk', 'at_bat_number', 'pitch_number', 'description']
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)
    df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number'], kind='mergesort')

    ab_id = (df['game_pk'].astype(np.int64) * 1000 +
             df['at_bat_number'].astype(np.int64)).values
    balls, strikes = walk_counts(df['description'].values, ab_id)
    df['balls'] = balls
    df['strikes'] = strikes
    df['count_str'] = df['balls'].astype(str) + '-' + df['strikes'].astype(str)
    df['row_key'] = (df['game_pk'].astype(str) + '_' +
                     df['at_bat_number'].astype(str) + '_' +
                     df['pitch_number'].astype(str))
    return df[['row_key', 'balls', 'strikes', 'count_str']]


def load_swing_frame(with_counts: bool = True) -> pd.DataFrame:
    """
    The swing population (bunts excluded, per data_audit.md §1) with every
    out-of-fold quantity the follow-ups need joined on.

    Columns added on top of prepare_data()'s output:
      predicted_timing   OOF predicted int_y      (inches)
      predicted_offset   OOF predicted barrel_distance_v2 (inches)
      p_contact          OOF P(contact) from the hybrid GBM
      timing_residual    int_y - predicted_timing
      offset_residual    barrel_distance_v2 - predicted_offset
                         (negative = closer to the ball than the model expected)
      offset_source      'whiff_tracked'  -> barrel_distance_v2 = real
                                             miss_distance + C
                         'contact_la_derived' -> |C*sin(launch_angle - 20deg)|,
                                             an outcome-derived proxy, NOT a
                                             tracked miss distance
      in_zone            zone <= 9
      balls/strikes/count_str  reconstructed count entering the pitch

    If out/rb_contact/oof_context_offset.parquet exists (built by
    dirB_context_offset_oof.py) two more columns appear:
      predicted_offset_context   OOF offset prediction with the batter random
                                 intercept suppressed -- the league-average
                                 batter's expected offset on that pitch
      offset_residual_context    barrel_distance_v2 - predicted_offset_context

    The distinction matters a lot. `offset_residual` is measured against a
    prediction that already contains the batter's own random intercept, so it
    is a deviation from *that batter's own norm* (right for Direction C's
    within-batter diagnostic, fatal for Direction B's skill stat -- see
    dirB_context_offset_oof.py's docstring). `offset_residual_context` is
    measured against the league-average batter, so it retains the batter
    effect and can carry skill.
    """
    df = prepare_data()

    oof_path = os.path.join(OUT_DIR, 'oof_predictions.parquet')
    if not os.path.exists(oof_path):
        raise FileNotFoundError(
            f'{oof_path} not found -- run oof_timing_offset.py (Step 1c) and '
            f'finalize_reports.py first; the follow-up analyses consume its output.')
    oof = pd.read_parquet(oof_path)
    keep = ['row_key', 'fold', 'predicted_timing', 'predicted_offset']
    if 'p_contact' in oof.columns:
        keep.append('p_contact')
    df = df.merge(oof[keep], on='row_key', how='left')

    if 'p_contact' not in df.columns or df['p_contact'].notna().sum() == 0:
        probs_path = os.path.join(OUT_DIR, 'oof_contact_probs.parquet')
        if not os.path.exists(probs_path):
            raise FileNotFoundError(
                f'No p_contact in {oof_path} and {probs_path} missing -- '
                f'run step5_rb_contact_pct.py to generate OOF contact probabilities.')
        df = df.drop(columns=[c for c in ['p_contact'] if c in df.columns])
        df = df.merge(pd.read_parquet(probs_path)[['row_key', 'p_contact']],
                      on='row_key', how='left')

    df['game_date'] = pd.to_datetime(df['game_date'])
    df['timing_residual'] = df['int_y'] - df['predicted_timing']
    df['offset_residual'] = df['barrel_distance_v2'] - df['predicted_offset']
    df['offset_source'] = np.where(df['is_contact'], 'contact_la_derived', 'whiff_tracked')
    df.loc[df['barrel_distance_v2'].isna(), 'offset_source'] = None
    df['in_zone'] = df['zone'] <= 9

    ctx_path = os.path.join(OUT_DIR, 'oof_context_offset.parquet')
    if os.path.exists(ctx_path):
        ctx = pd.read_parquet(ctx_path)[['row_key', 'predicted_offset_context']]
        df = df.merge(ctx, on='row_key', how='left')
        df['offset_residual_context'] = (df['barrel_distance_v2'] -
                                         df['predicted_offset_context'])

    if with_counts:
        df = df.merge(reconstruct_counts(), on='row_key', how='left')

    return df.sort_values(CHRON_KEYS, kind='mergesort').reset_index(drop=True)


def load_batter_names() -> pd.Series:
    """batter id (str) -> display name, from data/batter_stats_2025.csv."""
    if not os.path.exists(BATTER_NAMES_CSV):
        return pd.Series(dtype=object)
    names = pd.read_csv(BATTER_NAMES_CSV, usecols=['batter_id', 'batter_name'])
    names = names.dropna(subset=['batter_id']).drop_duplicates('batter_id')
    names['batter_id'] = names['batter_id'].astype(int).astype(str)
    return names.set_index('batter_id')['batter_name']


# ──────────────────────────────────────────────────────────────────────────
# Franks et al. meta-metrics
# ──────────────────────────────────────────────────────────────────────────

def discrimination(values: np.ndarray, sampling_var: np.ndarray) -> float:
    """
    Franks et al. discrimination: the fraction of observed between-player
    variance in a metric that reflects real skill differences rather than
    sampling noise.

        D = (Var_obs - E[sampling variance]) / Var_obs

    `sampling_var` is the estimated sampling variance of *each player's own*
    metric value (p(1-p)/n for a rate, s^2/n for a mean). Supplying it rather
    than hardcoding the binomial form is what lets a rate stat and a
    mean-of-inches stat share one table.

    Clipped at 0: a negative numerator means the observed spread is smaller
    than sampling noise alone predicts, i.e. no detectable skill signal.
    """
    values = np.asarray(values, dtype=float)
    sampling_var = np.asarray(sampling_var, dtype=float)
    ok = np.isfinite(values) & np.isfinite(sampling_var)
    values, sampling_var = values[ok], sampling_var[ok]
    total_var = values.var(ddof=1)
    if total_var <= 0:
        return np.nan
    return float(max(0.0, total_var - sampling_var.mean()) / total_var)


def discrimination_bootstrap(swing_df: pd.DataFrame, value_col: str,
                             batter_col: str = 'batter', n_boot: int = 200,
                             seed: int = 42) -> float:
    """
    Nonparametric check on `discrimination()`: resample each batter's swings
    with replacement, recompute that batter's mean, and use the spread of
    those bootstrap means as the noise variance. Makes no distributional
    assumption about the per-swing values, so it catches cases where the
    analytic s^2/n understates noise (heavy tails, within-batter clustering
    by game/pitcher).
    """
    rng = np.random.default_rng(seed)
    sub = swing_df[[batter_col, value_col]].dropna()
    groups = {b: g[value_col].to_numpy() for b, g in sub.groupby(batter_col)}
    observed = np.array([v.mean() for v in groups.values()])
    total_var = observed.var(ddof=1)
    if total_var <= 0:
        return np.nan

    noise = []
    for vals in groups.values():
        n = len(vals)
        draws = vals[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
        noise.append(draws.var(ddof=1))
    return float(max(0.0, total_var - np.mean(noise)) / total_var)


def stability_two_period(x1: np.ndarray, x2: np.ndarray,
                         var1: np.ndarray, var2: np.ndarray) -> dict:
    """
    Franks et al. stability from two measurement periods: what fraction of a
    player's true (noise-free) metric variance is persistent across periods
    versus genuinely changing over time?

    With observed x_pt = theta_p + delta_pt + e_pt   (theta = persistent skill,
    delta = real period-to-period change, e = sampling noise):

        Var(theta)  = Cov(x_p1, x_p2)                  [noise and delta are
                                                        independent across
                                                        periods, so only the
                                                        persistent part
                                                        survives the covariance]
        Var(delta)  = (Var(x_p1 - x_p2) - E[var1] - E[var2]) / 2
        stability   = Var(theta) / (Var(theta) + Var(delta))

    Returns the components too, since a negative Var(theta) or Var(delta)
    estimate (possible with small samples) is informative rather than an
    error -- it is reported, clipped at 0, and flagged.
    """
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    var1, var2 = np.asarray(var1, float), np.asarray(var2, float)
    ok = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(var1) & np.isfinite(var2)
    x1, x2, var1, var2 = x1[ok], x2[ok], var1[ok], var2[ok]

    var_persistent_raw = float(np.cov(x1, x2, ddof=1)[0, 1])
    var_diff = float(np.var(x1 - x2, ddof=1))
    var_change_raw = (var_diff - var1.mean() - var2.mean()) / 2.0

    var_persistent = max(0.0, var_persistent_raw)
    var_change = max(0.0, var_change_raw)
    denom = var_persistent + var_change
    stab = float(var_persistent / denom) if denom > 0 else np.nan
    return dict(
        stability=stab,
        var_persistent=var_persistent_raw,
        var_change=var_change_raw,
        clipped=bool(var_persistent_raw < 0 or var_change_raw < 0),
        n=int(len(x1)),
    )


# ──────────────────────────────────────────────────────────────────────────
# Per-batter aggregation helpers
# ──────────────────────────────────────────────────────────────────────────

def rate_stat(df: pd.DataFrame, flag_col: str, subset_mask: pd.Series = None,
              batter_col: str = 'batter') -> pd.DataFrame:
    """
    Per-batter rate of a boolean column, with its binomial sampling variance
    p(1-p)/n. `subset_mask` restricts the denominator (e.g. in-zone swings
    only for zone-contact%).
    """
    sub = df if subset_mask is None else df[subset_mask]
    g = sub.groupby(batter_col)[flag_col]
    out = pd.DataFrame({'value': g.mean(), 'n': g.size()})
    out['sampling_var'] = out['value'] * (1 - out['value']) / out['n']
    return out


def mean_stat(df: pd.DataFrame, value_col: str, batter_col: str = 'batter') -> pd.DataFrame:
    """
    Per-batter mean of a continuous column, with its sampling variance s^2/n.
    Batters with a single usable swing get NaN variance (no within estimate)
    and are dropped by downstream filters via the n threshold.
    """
    g = df.dropna(subset=[value_col]).groupby(batter_col)[value_col]
    out = pd.DataFrame({'value': g.mean(), 'n': g.size(), 'sd': g.std(ddof=1)})
    out['sampling_var'] = out['sd'] ** 2 / out['n']
    return out


def chronological_prefix(df: pd.DataFrame, fraction: float,
                         batter_col: str = 'batter') -> pd.DataFrame:
    """
    Each batter's first `fraction` of their own swings, in chronological
    order. At least one swing per batter (ceil), matching the reference
    paper's "first N% of the season" construction -- chronological rather
    than random, so nothing from later in the season leaks in.
    """
    df = df.sort_values(CHRON_KEYS, kind='mergesort')
    rank = df.groupby(batter_col).cumcount() + 1
    n = df.groupby(batter_col)[batter_col].transform('size')
    keep = np.ceil(n * fraction).astype(int).clip(lower=1)
    return df[rank <= keep]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


def md_table(df: pd.DataFrame, floatfmt: int = 4) -> str:
    """Minimal DataFrame -> GitHub markdown table (no tabulate dependency)."""
    d = df.copy()
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            # astype(float) first: rounding a float32 in place leaves values
            # like 0.51 printing as 0.5099999904632568.
            d[col] = d[col].astype(float).round(floatfmt)
    header = '| ' + ' | '.join(str(c) for c in d.columns) + ' |'
    sep = '|' + '|'.join(['---'] * len(d.columns)) + '|'
    body = '\n'.join('| ' + ' | '.join('' if pd.isna(v) else str(v) for v in row) + ' |'
                     for row in d.itertuples(index=False))
    return '\n'.join([header, sep, body])
