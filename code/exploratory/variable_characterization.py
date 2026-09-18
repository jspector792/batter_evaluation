"""
swing_diagnostics.py
====================
Diagnostic and exploratory plots for the swing mixed-effects models.

Section 1 – Group size distributions
    1a. Observations per batter/pitch_group cell — timing model data (all swings)
    1b. Observations per batter/pitch_group cell — tilt model data (misses only)
    1c. Observations per batter (unnested) — timing model data
    1d. Observations per batter (unnested) — tilt model data

Section 2 – Variable relationship plots (mean ± SD of outcome vs decile-binned predictor)
    For each x/y pair: one aggregate panel + one faceted panel (per pitch cluster)

    2.1  plate_z          → swing_path_tilt
    2.2  pitch_cluster    → swing_path_tilt   (categorical x, no binning needed)
    2.3  release_speed    → swing_path_tilt
    2.4  plate_x          → intercept_y
    2.5  plate_x_bat_flip → intercept_y
    2.6  release_speed    → intercept_y
    2.7  pitch_cluster    → intercept_y       (categorical x)

Requires: numpy, pandas, matplotlib, seaborn, scikit-learn
Run from the project root after running the clustering step.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "diagnostics")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
INTERCEPT_X = 'intercept_ball_minus_batter_pos_x_inches'
INTERCEPT_Y = 'intercept_ball_minus_batter_pos_y_inches'

IN_PLAY  = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
FOUL     = {'foul', 'foul_tip', 'foul_bunt'}
MISS     = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
CONTACT  = IN_PLAY | FOUL
ALL_SWINGS = CONTACT | MISS

N_CLUSTERS  = 10
N_DECILES   = 10
CLUSTER_COL = 'pitch_cluster'

# ── Plot style ────────────────────────────────────────────────────────────────
PALETTE  = sns.color_palette('tab10', N_CLUSTERS)
BLUE     = '#2563EB'
RED      = '#DC2626'
GRAY     = '#6B7280'
sns.set_theme(style='whitegrid', font_scale=1.05)


# ══════════════════════════════════════════════════════════════════════════════
# 0. DATA LOADING + CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_cluster():
    """Load all parquet files and attach k=10 pitch clusters."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files in {DATA_DIR}')

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f'Loaded {len(df):,} pitches')

    # Handedness-normalised horizontal break
    df['pfx_x_arm'] = (
        df['pfx_x'] * df['p_throws'].map({'R': 1, 'L': -1}).fillna(1)
    )

    # plate_x_bat_flip: positive = arm side of plate from batter's perspective
    if 'plate_x_bat_flip' not in df.columns:
        df['plate_x_bat_flip'] = (
            df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )

    # Drop rows missing clustering features
    feats = ['pfx_x_arm', 'pfx_z', 'release_speed']
    df = df.dropna(subset=feats + ['pitch_type', 'p_throws']).copy()

    # k=10 clustering (mirrors main script)
    X      = df[feats].values
    scaler = StandardScaler().fit(X)
    Xs     = scaler.transform(X)
    km     = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10).fit(Xs)
    df[CLUSTER_COL] = km.labels_.astype(str)

    # Annotate cluster labels with their mean velocity for readability
    centers = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_), columns=feats
    )
    label_map = {
        str(i): f'C{i} ({centers.loc[i,"release_speed"]:.0f}mph)'
        for i in range(N_CLUSTERS)
    }
    df['cluster_label'] = df[CLUSTER_COL].map(label_map)

    print(f'After clustering: {len(df):,} pitches, {N_CLUSTERS} clusters')
    return df, label_map


# ══════════════════════════════════════════════════════════════════════════════
# 1. GROUP SIZE DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _cell_counts(df, group_cols):
    """Return series of obs counts per group defined by group_cols."""
    return df.groupby(group_cols).size()


def _plot_count_distribution(counts_list, titles, suptitle, fname,
                              df_subsets=None, log_y=True, vlines=None):
    """
    Plot histograms of group-cell observation counts side by side.

    counts_list : list of pd.Series (one per panel)
    titles      : matching list of panel titles
    vlines      : list of x values to mark with vertical lines (e.g. [10, 30])
    """
    n = len(counts_list)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for idx, (ax, counts, title) in enumerate(zip(axes, counts_list, titles)):
        df_subset = df_subsets[idx] if df_subsets else None
        vals = counts.values
        bins = np.logspace(np.log10(max(vals.min(), 1)),
                           np.log10(vals.max()), 50)
        ax.hist(vals, bins=bins, color=BLUE, edgecolor='white', alpha=0.85)
        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')

        if vlines:
            for v in vlines:
                ax.axvline(v, color=RED, linewidth=1.2, linestyle='--',
                           label=f'n={v}')
            ax.legend(fontsize=9)

        median = np.median(vals)
        mean   = np.mean(vals)
        ax.axvline(median, color=GRAY, linewidth=1.5, linestyle=':')
        ax.text(median * 1.15, ax.get_ylim()[1] * 0.6,
                f'median={median:.0f}', color=GRAY, fontsize=9)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Observations per cell (log scale)')
        ax.set_ylabel('Number of cells (log scale)' if log_y
                      else 'Number of cells')

        # Stats annotation
        pct_lt10  = (vals < 10).mean() * 100
        pct_lt30  = (vals < 30).mean() * 100
        # Zero-obs cells: all possible batter×cluster combos minus observed ones
        n_batters  = df_subset['batter'].nunique() if df_subset is not None else None
        zero_cells = (n_batters * N_CLUSTERS - len(vals)) if n_batters else None
        zero_txt   = (f'zero-obs cells: {zero_cells:,}'
                      if zero_cells is not None else 'zero-obs cells: N/A')
        stats_txt = (f'n cells: {len(vals):,}\n'
                     f'{zero_txt}\n'
                     f'mean: {mean:.1f}\n'
                     f'median: {median:.0f}\n'
                     f'<10 obs: {pct_lt10:.1f}%\n'
                     f'<30 obs: {pct_lt30:.1f}%')
        ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.8))

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def plot_group_size_distributions(df):
    """Section 1: all four group-size panels."""

    print('\n' + '='*70)
    print('SECTION 1: GROUP SIZE DISTRIBUTIONS')
    print('='*70)

    # ── Data subsets ──────────────────────────────────────────────────────────
    timing_needed = [INTERCEPT_Y, 'release_speed', 'plate_x_bat_flip',
                     'batter', CLUSTER_COL, 'description']
    tilt_needed   = ['swing_path_tilt', 'plate_z',
                     'batter', CLUSTER_COL, 'description']

    df_timing = (df[df['description'].isin(ALL_SWINGS)]
                 .dropna(subset=timing_needed).copy())
    df_tilt   = (df[df['description'].isin(MISS)]
                 .dropna(subset=tilt_needed).copy())

    print(f'  Timing model rows (all swings): {len(df_timing):,}')
    print(f'  Tilt model rows   (misses only): {len(df_tilt):,}')

    # ── 1a + 1b: Nested batter/pitch_group counts ─────────────────────────────
    counts_timing_nested = _cell_counts(df_timing, ['batter', CLUSTER_COL])
    counts_tilt_nested   = _cell_counts(df_tilt,   ['batter', CLUSTER_COL])

    _plot_count_distribution(
        [counts_timing_nested, counts_tilt_nested],
        titles=[
            f'Timing model\n(all swings, nested batter×cluster)\n'
            f'{df_timing["batter"].nunique():,} batters × {N_CLUSTERS} clusters '
            f'= {len(counts_timing_nested):,} cells',
            f'Tilt model\n(misses only, nested batter×cluster)\n'
            f'{df_tilt["batter"].nunique():,} batters × {N_CLUSTERS} clusters '
            f'= {len(counts_tilt_nested):,} cells',
        ],
        suptitle='Group size distributions: batter × pitch_cluster cells\n'
                 '(original nested formulation)',
        fname='1ab_group_sizes_nested.png',
        df_subsets=[df_timing, df_tilt],
        vlines=[10, 30],
    )

    # ── 1c + 1d: Unnested batter-only counts ─────────────────────────────────
    counts_timing_batter = _cell_counts(df_timing, ['batter'])
    counts_tilt_batter   = _cell_counts(df_tilt,   ['batter'])

    _plot_count_distribution(
        [counts_timing_batter, counts_tilt_batter],
        titles=[
            f'Timing model\n(all swings, batter-only)\n'
            f'{len(counts_timing_batter):,} batters',
            f'Tilt model\n(misses only, batter-only)\n'
            f'{len(counts_tilt_batter):,} batters',
        ],
        suptitle='Group size distributions: batter-level (unnested)',
        fname='1cd_group_sizes_unnested.png',
        df_subsets=None,
        vlines=[10, 30],
    )

    # ── Print summary comparison ───────────────────────────────────────────────
    print('\n  Summary: median obs per cell')
    print(f'    Timing nested:   {np.median(counts_timing_nested.values):.0f}')
    print(f'    Tilt nested:     {np.median(counts_tilt_nested.values):.0f}')
    print(f'    Timing batter:   {np.median(counts_timing_batter.values):.0f}')
    print(f'    Tilt batter:     {np.median(counts_tilt_batter.values):.0f}')
    pct = lambda c: (c.values < 10).mean() * 100
    print(f'\n  Fraction of cells with <10 obs:')
    print(f'    Timing nested:   {pct(counts_timing_nested):.1f}%')
    print(f'    Tilt nested:     {pct(counts_tilt_nested):.1f}%')


# ══════════════════════════════════════════════════════════════════════════════
# 2. VARIABLE RELATIONSHIP PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _decile_bin(series, n=N_DECILES):
    """
    Assign each value to its decile (0–9).
    Returns (binned series, bin midpoint labels).
    """
    quantiles = np.linspace(0, 100, n + 1)
    edges     = np.percentile(series.dropna(), quantiles)
    # Ensure unique edges (handles constant-ish columns)
    edges = np.unique(edges)
    if len(edges) < 3:
        # fallback: equal-width
        edges = np.linspace(series.min(), series.max(), n + 1)
    labels = [f'{(edges[i]+edges[i+1])/2:.2f}' for i in range(len(edges)-1)]
    binned = pd.cut(series, bins=edges, labels=labels, include_lowest=True)
    return binned, labels


def _agg_mean_sd(df, x_col, y_col, is_categorical=False):
    """
    For continuous x: bin into deciles, then compute mean ± SD of y per bin.
    For categorical x: group directly.
    Returns DataFrame with columns: x_label, y_mean, y_sd, n.
    """
    df = df[[x_col, y_col]].dropna().copy()
    if is_categorical:
        grp = df.groupby(x_col)[y_col]
    else:
        df['_bin'], _ = _decile_bin(df[x_col])
        grp = df.groupby('_bin', observed=True)[y_col]

    result = grp.agg(y_mean='mean', y_sd='std', n='count').reset_index()
    result.columns = ['x_label', 'y_mean', 'y_sd', 'n']
    result['x_label'] = result['x_label'].astype(str)
    return result


def _plot_mean_sd_panel(ax, agg_df, color, title, xlabel, ylabel,
                        rotate_x=False):
    """Draw mean line + ±1 SD shaded band on ax."""
    x   = np.arange(len(agg_df))
    mu  = agg_df['y_mean'].values
    sd  = agg_df['y_sd'].values

    ax.plot(x, mu, color=color, linewidth=2, marker='o', markersize=4)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.18)

    ax.set_xticks(x)
    ax.set_xticklabels(agg_df['x_label'].values,
                       rotation=45 if rotate_x else 30,
                       ha='right', fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis='y', alpha=0.4)


def _relationship_plot(df, x_col, y_col, x_label, y_label,
                       is_categorical=False, fname_prefix='',
                       facet_col=None):
    """
    Build the two-panel figure:
      - Left:  aggregate (all clusters combined)
      - Right: faceted grid (one subplot per cluster), sorted by cluster label
    """
    if facet_col is None:
        facet_col = CLUSTER_COL

    clusters      = sorted(df[facet_col].dropna().unique(),
                           key=lambda c: str(c))
    n_clusters    = len(clusters)
    n_cols_facet  = 5
    n_rows_facet  = int(np.ceil(n_clusters / n_cols_facet))

    fig = plt.figure(figsize=(20, 6 + 4 * n_rows_facet))
    gs  = gridspec.GridSpec(
        1 + n_rows_facet, n_cols_facet,
        figure=fig,
        height_ratios=[2.5] + [1.8] * n_rows_facet,
        hspace=0.55, wspace=0.35,
    )

    # ── Aggregate panel (spans all columns in top row) ────────────────────────
    ax_agg = fig.add_subplot(gs[0, :])
    agg    = _agg_mean_sd(df, x_col, y_col, is_categorical)
    _plot_mean_sd_panel(
        ax_agg, agg, BLUE,
        title=f'{x_label} → {y_label}  [all pitches, n={len(df):,}]',
        xlabel=x_label, ylabel=y_label,
        rotate_x=is_categorical,
    )

    # ── Faceted panels (one per cluster) ─────────────────────────────────────
    for idx, cl in enumerate(clusters):
        row = 1 + idx // n_cols_facet
        col = idx % n_cols_facet
        ax  = fig.add_subplot(gs[row, col])

        sub = df[df[facet_col] == cl]
        if len(sub) < 20:
            ax.text(0.5, 0.5, 'insufficient\ndata',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color=GRAY)
            ax.set_title(str(cl), fontsize=9)
            continue

        sub_agg = _agg_mean_sd(sub, x_col, y_col, is_categorical)
        color   = PALETTE[idx % len(PALETTE)]
        # Use cluster_label if available, else fall back to the facet value
        if 'cluster_label' in sub.columns and facet_col == CLUSTER_COL:
            label = sub['cluster_label'].iloc[0]
        else:
            label = str(cl)
        _plot_mean_sd_panel(
            ax, sub_agg, color,
            title=f'{label}\n(n={len(sub):,})',
            xlabel=x_label, ylabel=y_label,
            rotate_x=is_categorical,
        )

    fig.suptitle(
        f'Relationship: {x_label} → {y_label}\n'
        f'Mean ± 1 SD  |  decile bins for continuous x',
        fontsize=13, fontweight='bold', y=1.01,
    )

    fname = os.path.join(OUT_DIR, f'{fname_prefix}_{x_col}_vs_{y_col}.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  → {fname}')


def plot_relationships(df):
    """Section 2: all seven x/y relationship plots."""

    print('\n' + '='*70)
    print('SECTION 2: VARIABLE RELATIONSHIPS')
    print('='*70)

    # Build the working dataset: need all relevant columns present
    needed = [
        'swing_path_tilt', INTERCEPT_Y, 'plate_z', 'plate_x',
        'plate_x_bat_flip', 'release_speed', CLUSTER_COL, 'cluster_label',
        'description', 'batter',
    ]
    df_work = df.dropna(subset=[c for c in needed if c in df.columns]).copy()
    df_work = df_work.rename(columns={INTERCEPT_Y: 'intercept_y'})
    print(f'  Working dataset: {len(df_work):,} rows')

    # ── 2.1  plate_z → swing_path_tilt ───────────────────────────────────────
    print('\n2.1  plate_z → swing_path_tilt')
    df_21 = df_work[df_work['description'].isin(MISS)].copy()
    print(f'     n={len(df_21):,} (misses only)')
    _relationship_plot(df_21, 'plate_z', 'swing_path_tilt',
                       'Pitch height (plate_z, ft)',
                       'Swing path tilt (°)',
                       fname_prefix='2.1')

    # ── 2.2  pitch_cluster → swing_path_tilt (categorical) ───────────────────
    print('\n2.2  pitch_cluster → swing_path_tilt')
    df_22 = df_work[df_work['description'].isin(MISS)].copy()
    _relationship_plot(df_22, CLUSTER_COL, 'swing_path_tilt',
                       'Pitch cluster',
                       'Swing path tilt (°)',
                       is_categorical=True,
                       fname_prefix='2.2')

    # ── 2.2b pitch_type label → swing_path_tilt (raw Statcast label) ─────────
    print('\n2.2b pitch_type → swing_path_tilt (raw label)')
    df_22b = df_work[df_work['description'].isin(MISS)].dropna(subset=['pitch_type']).copy()
    _relationship_plot(df_22b, 'pitch_type', 'swing_path_tilt',
                       'Pitch type (Statcast label)',
                       'Swing path tilt (°)',
                       is_categorical=True,
                       facet_col='pitch_type',
                       fname_prefix='2.2b')

    # ── 2.3  release_speed → swing_path_tilt ─────────────────────────────────
    print('\n2.3  release_speed → swing_path_tilt')
    df_23 = df_work[df_work['description'].isin(MISS)].copy()
    print(f'     n={len(df_23):,} (misses only)')
    _relationship_plot(df_23, 'release_speed', 'swing_path_tilt',
                       'Release speed (mph)',
                       'Swing path tilt (°)',
                       fname_prefix='2.3')

    # ── 2.4  plate_x → intercept_y ───────────────────────────────────────────
    print('\n2.4  plate_x → intercept_y')
    df_24 = df_work[df_work['description'].isin(ALL_SWINGS)].copy()
    print(f'     n={len(df_24):,} (all swings)')
    _relationship_plot(df_24, 'plate_x', 'intercept_y',
                       'Horizontal pitch location (plate_x, ft)',
                       'Intercept Y (inches)',
                       fname_prefix='2.4')

    # ── 2.5  plate_x_bat_flip → intercept_y ──────────────────────────────────
    print('\n2.5  plate_x_bat_flip → intercept_y')
    df_25 = df_work[df_work['description'].isin(ALL_SWINGS)].copy()
    _relationship_plot(df_25, 'plate_x_bat_flip', 'intercept_y',
                       'Horizontal location arm-side flipped (plate_x_bat_flip, ft)',
                       'Intercept Y (inches)',
                       fname_prefix='2.5')

    # ── 2.6  release_speed → intercept_y ─────────────────────────────────────
    print('\n2.6  release_speed → intercept_y')
    df_26 = df_work[df_work['description'].isin(ALL_SWINGS)].copy()
    _relationship_plot(df_26, 'release_speed', 'intercept_y',
                       'Release speed (mph)',
                       'Intercept Y (inches)',
                       fname_prefix='2.6')

    # ── 2.7  pitch_cluster → intercept_y (categorical) ───────────────────────
    print('\n2.7  pitch_cluster → intercept_y')
    df_27 = df_work[df_work['description'].isin(ALL_SWINGS)].copy()
    _relationship_plot(df_27, CLUSTER_COL, 'intercept_y',
                       'Pitch cluster',
                       'Intercept Y (inches)',
                       is_categorical=True,
                       fname_prefix='2.7')

    # ── 2.7b pitch_type label → intercept_y (raw Statcast label) ─────────────
    print('\n2.7b pitch_type → intercept_y (raw label)')
    df_27b = df_work[df_work['description'].isin(ALL_SWINGS)].dropna(subset=['pitch_type']).copy()
    _relationship_plot(df_27b, 'pitch_type', 'intercept_y',
                       'Pitch type (Statcast label)',
                       'Intercept Y (inches)',
                       is_categorical=True,
                       facet_col='pitch_type',
                       fname_prefix='2.7b')

    # ══════════════════════════════════════════════════════════════════════════
    # Contact-model plots (in-play only — EV and LA only exist on contact)
    # ══════════════════════════════════════════════════════════════════════════
    contact_needed = [
        'intercept_y', 'intercept_x', 'attack_angle', 'swing_path_tilt',
        'launch_speed', 'launch_angle', CLUSTER_COL, 'description',
    ]
    # intercept_x is INTERCEPT_X renamed; handle both column name forms
    if INTERCEPT_X in df_work.columns:
        df_work = df_work.rename(columns={INTERCEPT_X: 'intercept_x'})

    df_contact = (
        df_work[df_work['description'].isin(IN_PLAY)]
        .rename(columns={'launch_speed': 'exit_velo'})
        .dropna(subset=['intercept_y', 'attack_angle',
                        'exit_velo', 'launch_angle', CLUSTER_COL])
        .copy()
    )
    # plate_x_bat_flip analogue for intercept_x: flip sign by batter handedness
    if 'intercept_x' in df_contact.columns and 'stand' in df_contact.columns:
        df_contact['intercept_x_bat_flip'] = (
            df_contact['intercept_x']
            * df_contact['stand'].map({'R': -1, 'L': 1}).fillna(1)
        )
    elif 'intercept_x' in df_contact.columns:
        df_contact['intercept_x_bat_flip'] = df_contact['intercept_x']

    print(f'\nContact events for plots 2.8–2.15: {len(df_contact):,}')

    # ── 2.8  intercept_y → exit_velo ─────────────────────────────────────────
    print('\n2.8  intercept_y → exit_velo')
    _relationship_plot(df_contact, 'intercept_y', 'exit_velo',
                       'Intercept Y (inches)',
                       'Exit velocity (mph)',
                       fname_prefix='2.8')

    # ── 2.9  intercept_x_bat_flip → exit_velo ────────────────────────────────
    print('\n2.9  intercept_x_bat_flip → exit_velo')
    if 'intercept_x_bat_flip' in df_contact.columns:
        _relationship_plot(df_contact, 'intercept_x_bat_flip', 'exit_velo',
                           'Intercept X bat-flipped (inches)',
                           'Exit velocity (mph)',
                           fname_prefix='2.9')
    else:
        print('     SKIP — intercept_x not available')

    # ── 2.10 attack_angle → exit_velo ────────────────────────────────────────
    print('\n2.10 attack_angle → exit_velo')
    _relationship_plot(df_contact, 'attack_angle', 'exit_velo',
                       'Attack angle (°)',
                       'Exit velocity (mph)',
                       fname_prefix='2.10')

    # ── 2.11 pitch_cluster → exit_velo ───────────────────────────────────────
    print('\n2.11 pitch_cluster → exit_velo')
    _relationship_plot(df_contact, CLUSTER_COL, 'exit_velo',
                       'Pitch cluster',
                       'Exit velocity (mph)',
                       is_categorical=True,
                       fname_prefix='2.11')

    # ── 2.12 intercept_y → launch_angle ──────────────────────────────────────
    print('\n2.12 intercept_y → launch_angle')
    _relationship_plot(df_contact, 'intercept_y', 'launch_angle',
                       'Intercept Y (inches)',
                       'Launch angle (°)',
                       fname_prefix='2.12')

    # ── 2.13 swing_path_tilt → launch_angle ──────────────────────────────────
    print('\n2.13 swing_path_tilt → launch_angle')
    df_213 = df_contact.dropna(subset=['swing_path_tilt']).copy()
    _relationship_plot(df_213, 'swing_path_tilt', 'launch_angle',
                       'Swing path tilt (°)',
                       'Launch angle (°)',
                       fname_prefix='2.13')

    # ── 2.14 attack_angle → launch_angle ─────────────────────────────────────
    print('\n2.14 attack_angle → launch_angle')
    _relationship_plot(df_contact, 'attack_angle', 'launch_angle',
                       'Attack angle (°)',
                       'Launch angle (°)',
                       fname_prefix='2.14')

    # ── 2.15 pitch_cluster → launch_angle ────────────────────────────────────
    print('\n2.15 pitch_cluster → launch_angle')
    _relationship_plot(df_contact, CLUSTER_COL, 'launch_angle',
                       'Pitch cluster',
                       'Launch angle (°)',
                       is_categorical=True,
                       fname_prefix='2.15')

    # ── 2.16 plate_z → launch_angle ────────────────────────────────────
    print('\n2.15 pitch_cluster → launch_angle')
    _relationship_plot(df_contact, 'plate_z', 'launch_angle',
                       'Plate z',
                       'Launch angle (°)',
                       is_categorical=False,
                       fname_prefix='2.15')
    
    # ── 2.15 release_speed → launch_angle ────────────────────────────────────
    print('\n2.15 pitch_cluster → launch_angle')
    _relationship_plot(df_contact, 'release_speed', 'launch_angle',
                       'Release Speed',
                       'Launch angle (°)',
                       is_categorical=False,
                       fname_prefix='2.15')
    
    # ── 2.16 plate_z → exit velo ────────────────────────────────────
    print('\n2.15 pitch_cluster → exit_velo')
    _relationship_plot(df_contact, 'plate_z', 'exit_velo',
                       'Plate z',
                       'Exit velocity (mph)',
                       is_categorical=False,
                       fname_prefix='2.15')
    
    # ── 2.15 release_speed → exit velo ────────────────────────────────────
    print('\n2.15 pitch_cluster → exit_velo')
    _relationship_plot(df_contact, 'release_speed', 'exit_velo',
                       'Release Speed',
                       'Exit velocity (mph)',
                       is_categorical=False,
                       fname_prefix='2.15')
    
    


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df, label_map = load_and_cluster()

    print(f'\nCluster labels:')
    for k, v in sorted(label_map.items(), key=lambda x: int(x[0])):
        n = (df[CLUSTER_COL] == k).sum()
        print(f'  {v}: {n:,} pitches')

    plot_group_size_distributions(df)
    plot_relationships(df)

    print(f'\nAll plots saved to {OUT_DIR}')


if __name__ == '__main__':
    main()