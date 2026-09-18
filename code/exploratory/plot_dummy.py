"""
binned_residual_plots.py
========================
2x2 grid of binned residual plots, one panel per model:
  - barrel_swing_v3_fitted  (base + attack_angle + swing_path_tilt)
  - contact_v1_fitted       (signed barrel distance, contact only)
  - contact_v2_fitted       (launch angle, contact only)
  - contact_v3_fitted       (launch speed, contact only)

Each panel: 10 launch angle bins on the x-axis, residuals shown as
box plots with individual points overlaid as a strip plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "final_models_variants")
DIAG_DIR = os.path.join(BASE_DIR, "out", "exploratory", "diagnostics")
os.makedirs(DIAG_DIR, exist_ok=True)

# ── Model definitions ──────────────────────────────────────────────────────────
# (filename_stem, panel_title, outcome_col, residual_col)
MODELS = [
    ('barrel_swing_v3_fitted',
     'Barrel distance\n(swing V3: base + attack + tilt)',
     'barrel_distance_v2',
     'residual'),
    ('contact_v1_fitted',
     'Signed barrel distance\n(contact V1)',
     'barrel_distance_signed',
     'residual'),
    ('contact_v2_fitted',
     'Launch angle\n(contact V2)',
     'launch_angle',
     'residual'),
    ('contact_v3_fitted',
     'Launch speed\n(contact V3)',
     'launch_speed',
     'residual'),
]

N_BINS   = 10
MIN_BIN  = 10    # minimum observations per bin to display
ALPHA_STRIP = 0.25
STRIP_SIZE  = 2
BOX_WIDTH   = 0.55

sns.set_theme(style='whitegrid', font_scale=1.0)
BLUE  = '#2563EB'
RED   = '#DC2626'
GRAY  = '#6B7280'


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_fitted(stem: str) -> pd.DataFrame:
    path = os.path.join(OUT_DIR, f'{stem}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Fitted file not found: {path}')
    df = pd.read_csv(path)
    return df


def make_la_bins(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """
    Assign each row a launch angle bin label.
    Bins are equal-width across the observed LA range.
    Rows missing launch_angle are dropped.
    """
    if 'launch_angle' not in df.columns:
        raise ValueError('launch_angle column required for binning')

    df = df.dropna(subset=['launch_angle', 'residual']).copy()

    la_min = np.floor(df['launch_angle'].min())
    la_max = np.ceil(df['launch_angle'].max())
    edges  = np.linspace(la_min, la_max, n_bins + 1)

    df['la_bin'] = pd.cut(
        df['launch_angle'],
        bins    = edges,
        include_lowest = True,
    )

    # Label each bin by its midpoint for readability
    df['la_bin_mid'] = df['la_bin'].apply(
        lambda b: f'{(b.left + b.right) / 2:.0f}°' if pd.notna(b) else np.nan
    )

    # Ordered category so seaborn respects x-axis order
    bin_order = [
        f'{(edges[i] + edges[i+1]) / 2:.0f}°'
        for i in range(n_bins)
    ]
    df['la_bin_mid'] = pd.Categorical(
        df['la_bin_mid'], categories=bin_order, ordered=True)

    return df, bin_order


def panel(ax: plt.Axes, df: pd.DataFrame,
          title: str, outcome_col: str):
    """
    Draw one binned residual panel on ax.
    """
    df, bin_order = make_la_bins(df, N_BINS)

    # Drop bins with fewer than MIN_BIN observations
    counts = df.groupby('la_bin_mid', observed=True)['residual'].count()
    valid_bins = counts[counts >= MIN_BIN].index.tolist()
    df = df[df['la_bin_mid'].isin(valid_bins)]

    if df.empty:
        ax.text(0.5, 0.5, 'Insufficient data',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontweight='bold', fontsize=10)
        return

    # Strip plot (individual points, jittered)
    sns.stripplot(
        data    = df,
        x       = 'la_bin_mid',
        y       = 'residual',
        order   = [b for b in bin_order if b in valid_bins],
        ax      = ax,
        color   = BLUE,
        alpha   = ALPHA_STRIP,
        size    = STRIP_SIZE,
        jitter  = True,
        zorder  = 1,
    )

    # Box plot on top
    sns.boxplot(
        data       = df,
        x          = 'la_bin_mid',
        y          = 'residual',
        order      = [b for b in bin_order if b in valid_bins],
        ax         = ax,
        color      = 'white',
        width      = BOX_WIDTH,
        linewidth  = 1.2,
        fliersize  = 0,        # hide outlier dots — strip plot shows them
        boxprops   = dict(edgecolor=GRAY, linewidth=1.2),
        medianprops= dict(color=RED, linewidth=2.0),
        whiskerprops=dict(color=GRAY, linewidth=1.0),
        capprops   = dict(color=GRAY, linewidth=1.0),
        zorder     = 2,
    )

    ax.axhline(0, color=RED, linewidth=1.0, linestyle='--', zorder=3)

    # Annotate bin counts along the bottom
    bin_counts = df.groupby('la_bin_mid', observed=True)['residual'].count()
    valid_order = [b for b in bin_order if b in valid_bins]
    for i, bin_label in enumerate(valid_order):
        n = bin_counts.get(bin_label, 0)
        ax.text(i, ax.get_ylim()[0], f'n={n}',
                ha='center', va='bottom', fontsize=6.5, color=GRAY)

    ax.set_xlabel('Launch angle bin (midpoint)', fontsize=9)
    ax.set_ylabel('Residual', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', alpha=0.3)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'Binned residuals by launch angle ({N_BINS} bins)\n'
        'Red line = zero, red bar = median, boxes = IQR',
        fontsize=12, fontweight='bold', y=1.01,
    )

    for ax, (stem, title, outcome_col, resid_col) in zip(
            axes.flat, MODELS):

        print(f'Loading {stem}...')
        try:
            df = load_fitted(stem)
        except FileNotFoundError as e:
            print(f'  SKIPPED: {e}')
            ax.text(0.5, 0.5, f'File not found:\n{stem}.csv',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color=RED)
            ax.set_title(title, fontweight='bold', fontsize=10)
            continue

        # The fitted files all have a 'residual' column from the modelling
        # scripts. We need launch_angle for binning — it's available directly
        # in V2/V3 as the outcome column, and must be present in V1's fitted
        # file if it was saved alongside the outcome. If missing, try to use
        # the outcome column itself as a proxy for binning (V2 only).
        if 'launch_angle' not in df.columns:
            if outcome_col == 'launch_angle' and outcome_col in df.columns:
                df['launch_angle'] = df[outcome_col]
            else:
                print(f'  WARNING: launch_angle missing from {stem} — '
                      f'cannot bin. Skipping.')
                ax.text(0.5, 0.5, 'launch_angle not in file',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9, color=RED)
                ax.set_title(title, fontweight='bold', fontsize=10)
                continue

        print(f'  {len(df):,} rows, '
              f'residual range=[{df["residual"].min():.2f}, '
              f'{df["residual"].max():.2f}]')

        panel(ax, df, title, outcome_col)

    plt.tight_layout()
    out_path = os.path.join(DIAG_DIR, 'binned_residuals_2x2.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()