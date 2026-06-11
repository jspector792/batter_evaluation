"""
Shared helpers for the 2025 swing-quality analysis.

Timing is defined as the y-component of the bat-ball intercept relative to the
batter's position (in inches), centred on the population median across all
swings. Positive = pull / early (ball contacted further out in front), negative
= oppo / late (ball contacted closer to the catcher).

The pull/early-positive sign convention is empirically validated against hc_x
spray direction for both RHH and LHH (r ≈ ±0.47 with hc_x).
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import ttest_ind

# ── Description filters ───────────────────────────────────────────────────────
IN_PLAY = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
FOUL    = {'foul', 'foul_tip', 'foul_bunt'}
MISS    = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
CONTACT = IN_PLAY | FOUL
ALL_SWINGS = CONTACT | MISS
STRIKE_DESC = {'called_strike', 'swinging_strike', 'swinging_strike_blocked',
               'foul', 'foul_tip'}

# ── Pitch-type groups ────────────────────────────────────────────────────────
FASTBALL_TYPES = {"FF", "SI", "FC", "FT"}

# ── Statcast zone classification (catcher's view, 3×3 in-zone + shadow) ──────
INSIDE_Z  = {1, 4, 7, 11, 13}
MIDDLE_Z  = {2, 5, 8}
OUTSIDE_Z = {3, 6, 9, 12, 14}

# ── Consistent mode-colour palette across plots ──────────────────────────────
MODE_COLORS = {'oppo': '#1f77b4', 'oppo-peak': '#1f77b4',
               'pull': '#d62728', 'pull-peak': '#d62728'}

INTERCEPT_X = 'intercept_ball_minus_batter_pos_x_inches'
INTERCEPT_Y = 'intercept_ball_minus_batter_pos_y_inches'


# ── Timing column ────────────────────────────────────────────────────────────
def add_timing(df, center=None):
    """
    Add a 'timing' column = intercept_y - center, where center defaults to the
    population median of intercept_y across df.

    Returns the centre value used (so other dataframes can be centred on the
    same reference).
    """
    df['timing'] = df[INTERCEPT_Y]
    if center is None:
        center = df['timing'].median()
    df['timing'] = df['timing'] - center
    return center


# ── Per-hitter peak via count-weighted moving average + side-significance ───
def smoothed_peak(timing, dre, min_n=50, n_bins=18, window=3.0,
                  alpha=0.05, ma_window=5,
                  min_per_bin=None, min_window_total=None,
                  return_curve=False, **_legacy_kwargs):
    """
    Estimate a hitter's peak timing location using a **count-weighted moving
    average** over equal-width timing bins, with a one-sided Welch t-test
    gating whether the winning peak is significantly higher than the losing
    peak.

    Pipeline:
      1. Drop NaNs, require ≥ `min_n` swings.
      2. Bin timing into `n_bins` equal-width bins between p5 and p95.
      3. Drop bins with fewer than `min_per_bin` swings (auto-scales to
         data size if not specified). This is the key defence against single
         high-leverage swings.
      4. Centred moving average over `ma_window` consecutive bins, weighted
         by per-bin pitch counts. Windows whose total swings fall below
         `min_window_total` are skipped.
      5. Find peak on positive vs negative side from the moving-average curve.
      6. One-sided Welch t-test on raw delta_run_exp values within ±`window`
         inches of the winning vs losing peak. The peak is recorded only when
         it has higher mean than the loser and p < `alpha`.

    Returns (peak_loc, peak_val, p_value, significant); when `return_curve=True`
    also returns the smoothed (grid_x, grid_y) curve.
    """
    timing = np.asarray(timing, dtype=float)
    dre    = np.asarray(dre,    dtype=float)
    mask   = np.isfinite(timing) & np.isfinite(dre)
    timing, dre = timing[mask], dre[mask]
    n = len(timing)
    nan_result = (np.nan, np.nan, np.nan, False)
    if n < min_n:
        return (*nan_result, None) if return_curve else nan_result

    # Adaptive bin-count thresholds: floor at 3 swings/bin, 15 swings/window;
    # ceiling at 15 swings/bin once we have plenty of data.
    if min_per_bin is None:
        min_per_bin = int(max(3, min(15, n // (3 * n_bins))))
    if min_window_total is None:
        min_window_total = int(max(15, n // max(1, n_bins // 2)))

    gx, gy, bin_tbl = weighted_moving_average(
        timing, dre, n_bins=n_bins, window=ma_window,
        min_per_bin=min_per_bin, min_window_total=min_window_total,
        quantile_range=(0.05, 0.95))

    if len(gx) < 6:
        return (*nan_result, None) if return_curve else nan_result
    pos_mask = gx > 0
    neg_mask = gx < 0
    if pos_mask.sum() < 2 or neg_mask.sum() < 2:
        return (*nan_result, None) if return_curve else nan_result

    pk_p_idx = int(np.argmax(gy[pos_mask]))
    pk_n_idx = int(np.argmax(gy[neg_mask]))
    pk_p_loc = float(gx[pos_mask][pk_p_idx])
    pk_p_val = float(gy[pos_mask][pk_p_idx])
    pk_n_loc = float(gx[neg_mask][pk_n_idx])
    pk_n_val = float(gy[neg_mask][pk_n_idx])

    if pk_p_val >= pk_n_val:
        winner_loc, winner_val, loser_loc = pk_p_loc, pk_p_val, pk_n_loc
    else:
        winner_loc, winner_val, loser_loc = pk_n_loc, pk_n_val, pk_p_loc

    win_data = dre[(timing >= winner_loc - window) & (timing <= winner_loc + window)]
    los_data = dre[(timing >= loser_loc  - window) & (timing <= loser_loc  + window)]
    if len(win_data) < 5 or len(los_data) < 5:
        result = (winner_loc, winner_val, np.nan, False)
    else:
        try:
            t_stat, p_two = ttest_ind(win_data, los_data, equal_var=False)
            p = p_two / 2 if t_stat > 0 else 1 - p_two / 2
        except Exception:
            p = np.nan
        sig = bool(np.isfinite(p) and p < alpha and
                   np.mean(win_data) > np.mean(los_data))
        result = (winner_loc, winner_val, float(p), sig)

    if return_curve:
        return (*result, (gx, gy))
    return result


# ── Convenience: peak that returns NaN unless significant ────────────────────
def significant_peak(timing, dre, **kwargs):
    """Return peak location only if it passes the side-significance test."""
    loc, _, _, sig = smoothed_peak(timing, dre, **kwargs)
    return loc if sig else np.nan


# ── Binned LW curve helper ───────────────────────────────────────────────────
def binned_lw(timing, dre, n_bins=40, range_quantiles=(0.005, 0.995)):
    """Population LW vs timing: returns (x_means, y_means, n_per_bin)."""
    timing = np.asarray(timing, dtype=float)
    dre    = np.asarray(dre,    dtype=float)
    mask   = np.isfinite(timing) & np.isfinite(dre)
    timing, dre = timing[mask], dre[mask]
    if len(timing) < 10:
        return np.array([]), np.array([]), np.array([])
    edges = np.linspace(np.quantile(timing, range_quantiles[0]),
                        np.quantile(timing, range_quantiles[1]),
                        n_bins + 1)
    bins  = pd.cut(timing, bins=edges, include_lowest=True)
    g = (pd.DataFrame({'t': timing, 'y': dre, 'b': bins})
           .dropna(subset=['b'])
           .groupby('b', observed=True))
    x = g['t'].mean().values
    y = g['y'].mean().values
    n = g['y'].count().values
    return x, y, n


# ── Standard axis-label string ───────────────────────────────────────────────
TIMING_AXIS_LABEL = ("Timing (in)\n"
                     "← oppo/late (behind batter)        pull/early (in front) →")
TIMING_AXIS_LABEL_SHORT = "Timing (in, centred on median)"


# ── Bin-weighted moving average (robust to sparse bins) ──────────────────────
def weighted_moving_average(timing, y, n_bins=24, window=5,
                            min_per_bin=8, min_window_total=30,
                            quantile_range=(0.02, 0.98)):
    """
    Bin (timing, y), then compute a centred moving average over bins,
    weighting each bin by its pitch count.

    Bins with fewer than `min_per_bin` pitches are dropped before the moving
    average; windows whose total count is below `min_window_total` are also
    skipped. This makes the curve robust to single high-leverage bins (e.g.,
    a lone barrel at extreme timing).

    Returns (grid_x, grid_y, bin_table) where bin_table has columns
    [t_mid, y_mean, n] for the bins that survived the per-bin filter.
    """
    timing = np.asarray(timing, dtype=float)
    y      = np.asarray(y,      dtype=float)
    mask   = np.isfinite(timing) & np.isfinite(y)
    timing, y = timing[mask], y[mask]
    if len(timing) < min_window_total:
        return np.array([]), np.array([]), pd.DataFrame(columns=['t_mid','y_mean','n'])
    edges = np.linspace(np.quantile(timing, quantile_range[0]),
                        np.quantile(timing, quantile_range[1]),
                        n_bins + 1)
    bins = pd.cut(timing, bins=edges, include_lowest=True)
    g = (pd.DataFrame({'t': timing, 'y': y, 'b': bins})
           .dropna(subset=['b'])
           .groupby('b', observed=True)
           .agg(t_mid=('t', 'mean'), y_mean=('y', 'mean'), n=('y', 'count'))
           .reset_index(drop=True)
           .sort_values('t_mid')
           .reset_index(drop=True))
    g = g[g['n'] >= min_per_bin].reset_index(drop=True)
    if len(g) < max(window, 3):
        return np.array([]), np.array([]), g
    half = window // 2
    out_x, out_y = [], []
    n_arr = g['n'].values
    y_arr = g['y_mean'].values
    t_arr = g['t_mid'].values
    for i in range(len(g)):
        lo = max(0, i - half)
        hi = min(len(g), i + half + 1)
        ws = n_arr[lo:hi]
        ys = y_arr[lo:hi]
        if ws.sum() < min_window_total:
            continue
        out_x.append(t_arr[i])
        out_y.append(float(np.average(ys, weights=ws)))
    return np.array(out_x), np.array(out_y), g


# ── Categorical helpers (zone, matchup) ──────────────────────────────────────
def zone_group(z):
    """Return 'inside' / 'middle' / 'outside' for Statcast zone numbers."""
    if z in INSIDE_Z:  return 'inside'
    if z in MIDDLE_Z:  return 'middle'
    if z in OUTSIDE_Z: return 'outside'
    return None


def matchup_label(stand, p_throws):
    """Return 'same-hand' / 'diff-hand' from stand and p_throws columns."""
    return 'same-hand' if stand == p_throws else 'diff-hand'


def add_zone_and_matchup(df):
    """Add `zone_grp` (in/mid/out) and `matchup` (same/diff-hand) to df in-place."""
    if 'zone' in df.columns:
        df['zone_grp'] = df['zone'].map(zone_group)
    if {'stand', 'p_throws'}.issubset(df.columns):
        df['matchup'] = np.where(df['stand'] == df['p_throws'],
                                  'same-hand', 'diff-hand')
    return df


# ── Unified data loaders ─────────────────────────────────────────────────────
def load_pitches(data_dir, cols=None, pitch_types=None, descriptions=None,
                  parquet=None):
    """
    Load all CSV or parquet files in `data_dir` and optionally filter by
    `pitch_types` and/or `descriptions`. Auto-detects format from file
    extension unless `parquet` is set explicitly (True for parquet, False
    for csv).

    Returns the concatenated DataFrame after filtering (no timing column
    added — call `add_timing` separately so the centre is explicit).
    """
    csv_files     = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if parquet is None:
        parquet = bool(parquet_files) and not csv_files
    files = parquet_files if parquet else csv_files
    if not files:
        raise FileNotFoundError(f"No {'parquet' if parquet else 'csv'} files in {data_dir}")
    if parquet:
        if cols is not None:
            chunks = [pd.read_parquet(f, columns=list(cols)) for f in files]
        else:
            chunks = [pd.read_parquet(f) for f in files]
    else:
        if cols is not None:
            chunks = [pd.read_csv(f, usecols=list(cols)) for f in files]
        else:
            chunks = [pd.read_csv(f) for f in files]
    df = pd.concat(chunks, ignore_index=True)
    if pitch_types is not None and 'pitch_type' in df.columns:
        df = df[df['pitch_type'].isin(pitch_types)].copy()
    if descriptions is not None and 'description' in df.columns:
        df = df[df['description'].isin(descriptions)].copy()
    return df


def load_swings_with_timing(data_dir, pitch_types=FASTBALL_TYPES,
                             descriptions=ALL_SWINGS, extra_cols=()):
    """
    One-line loader for the most common use case: load swings, drop rows
    missing intercept_y, add the `timing` column centred on the population
    median. Returns (df, center).
    """
    base_cols = {'pitch_type', 'batter', 'stand', 'description',
                  INTERCEPT_Y, 'delta_run_exp'}
    cols = list(base_cols | set(extra_cols))
    df = load_pitches(data_dir, cols=cols, pitch_types=pitch_types,
                      descriptions=descriptions)
    df = df.dropna(subset=[INTERCEPT_Y])
    center = add_timing(df)
    return df, center


# ── Sorting / ranking helpers ────────────────────────────────────────────────
def sort_by_velocity(centers_df, col='release_speed'):
    """Return the cluster index order (slowest → fastest) given centres DataFrame."""
    return centers_df.sort_values(col).index.tolist()


# ── Unified LW-curve plotter (uses weighted_moving_average) ─────────────────
def plot_lw_curves(ax, subsets, palette=None, n_bins=24, ma_window=5,
                    min_per_bin=None, min_window_total=None,
                    show_bins=True, label_n=True,
                    timing_col='timing', y_col='delta_run_exp',
                    min_n_data=300):
    """
    Overlay one count-weighted moving-average LW curve per subset.

    Parameters
    ----------
    ax : matplotlib axis
    subsets : dict[label -> DataFrame] or dict[label -> (DataFrame, color, ls)]
    palette : list of colours used when subsets entries are bare DataFrames.
    n_bins, ma_window : passed to weighted_moving_average.
    min_per_bin, min_window_total : if None, scale with subset size.
    show_bins : if True, plot raw bin means as faint dots beneath the curve.
    label_n : add (n=…) to each subset's legend label.
    timing_col, y_col : DataFrame column names.
    min_n_data : skip a subset that has fewer than this many rows.

    Each subset entry may be either a DataFrame (palette is then used to assign
    colour) or a tuple (DataFrame, color, linestyle).
    """
    default_palette = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
                       '#9467bd', '#8c564b', '#17becf', '#e377c2']
    if palette is None:
        palette = default_palette
    pal_iter = iter(palette)
    for label, entry in subsets.items():
        if isinstance(entry, tuple):
            data, color, ls = entry
        else:
            data, color, ls = entry, next(pal_iter, default_palette[0]), '-'
        sub = data.dropna(subset=[timing_col, y_col])
        if len(sub) < min_n_data:
            continue
        mpb = (min_per_bin if min_per_bin is not None
               else max(8, len(sub) // 300))
        mwt = (min_window_total if min_window_total is not None
               else max(30, len(sub) // 60))
        gx, gy, bin_tbl = weighted_moving_average(
            sub[timing_col].values, sub[y_col].values,
            n_bins=n_bins, window=ma_window,
            min_per_bin=mpb, min_window_total=mwt)
        if len(gx) < 4:
            continue
        legend = f"{label}  (n={len(sub):,})" if label_n else label
        if show_bins and len(bin_tbl):
            ax.plot(bin_tbl['t_mid'], bin_tbl['y_mean'],
                    color=color, ls='', marker='o', ms=2.5, alpha=0.3, zorder=1)
        ax.plot(gx, gy, color=color, ls=ls, lw=2.2, label=legend, zorder=3)
    ax.axhline(0, color='black', lw=0.8, ls='--')
    ax.axvline(0, color='grey',  lw=0.8, ls=':')


# ── Peak from moving-average curve (no significance test) ────────────────────
def peak_from_moving_average(timing, dre, n_bins=32, ma_window=5,
                              min_per_bin=30, min_window_total=200):
    """
    Return (peak_loc, peak_val) — argmax of the weighted moving-average curve.
    Use this when you only need the population-level peak, not the per-hitter
    significance gate (use `smoothed_peak` for that).
    Returns (NaN, NaN) if too sparse.
    """
    gx, gy, _ = weighted_moving_average(
        timing, dre, n_bins=n_bins, window=ma_window,
        min_per_bin=min_per_bin, min_window_total=min_window_total)
    if len(gx) < 3:
        return np.nan, np.nan
    i = int(np.argmax(gy))
    return float(gx[i]), float(gy[i])


# ── GMM mode fitting (2-component, consistent oppo/pull ordering) ───────────
def fit_gmm_modes(peaks, n_components=2, random_state=42):
    """
    Fit a Gaussian mixture to `peaks` (Series or array) and return:
        gmm           — the fitted sklearn GaussianMixture
        gm_means      — array of component means (raw GMM order)
        oppo_idx      — index of the lower-mean component
        pull_idx      — index of the higher-mean component
        mode_map      — dict {peak_value → 'oppo' / 'pull'} for the input peaks
        mode_label    — function (peak_value) -> 'oppo' / 'pull' / NaN

    Only `peaks` with finite values are used to fit.
    """
    from sklearn.mixture import GaussianMixture
    arr = np.asarray(peaks).astype(float).ravel()
    arr = arr[np.isfinite(arr)]
    if len(arr) < 10:
        raise ValueError(f"Too few peaks ({len(arr)}) to fit a GMM.")
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(arr.reshape(-1, 1))
    gm_means = gmm.means_.flatten()
    oppo_idx = int(np.argmin(gm_means))
    pull_idx = int(np.argmax(gm_means))

    def mode_label(v):
        if not np.isfinite(v):
            return np.nan
        comp = int(np.argmax(gmm.predict_proba([[v]])[0]))
        return 'oppo' if comp == oppo_idx else 'pull'

    if isinstance(peaks, pd.Series):
        mode_map = {k: mode_label(v) for k, v in peaks.dropna().items()}
    else:
        mode_map = {i: mode_label(v) for i, v in enumerate(arr)}
    return gmm, gm_means, oppo_idx, pull_idx, mode_map, mode_label
