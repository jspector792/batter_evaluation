"""
la_vs_predictions.py
====================
2x2 grid of scatter plots: launch angle vs model predictions.
Same panel layout as the binned residual plots.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR  = os.path.join(BASE_DIR, 'out', 'exploratory', 'final_models_variants')
DIAG_DIR = os.path.join(BASE_DIR, 'out', 'exploratory', 'diagnostics')
os.makedirs(DIAG_DIR, exist_ok=True)

# ── Same four models as the binned residual plot ───────────────────────────────
# (filename_stem, panel_title, fitted_col)
MODELS = [
    ('barrel_swing_v3_fitted',
     'Barrel distance\n(swing V3: base + attack + tilt)',
     'fitted'),
    ('contact_v1_fitted',
     'Signed barrel distance\n(contact V1)',
     'fitted'),
    ('contact_v2_fitted',
     'Launch angle\n(contact V2)',
     'fitted'),
    ('contact_v3_fitted',
     'Launch speed\n(contact V3)',
     'fitted'),
]

sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE = '#2563EB'; RED = '#DC2626'; GRAY = '#6B7280'

ALPHA_POINTS = 0.08
POINT_SIZE   = 2


def panel(ax: plt.Axes, df: pd.DataFrame,
          title: str, fitted_col: str):

    df = df.dropna(subset=['launch_angle', fitted_col])
    if df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title, fontweight='bold', fontsize=10)
        return

    x = df['launch_angle'].values
    y = df[fitted_col].values

    ax.scatter(x, y, s=POINT_SIZE, alpha=ALPHA_POINTS,
               color=BLUE, rasterized=True)

    # Line of best fit + 95% CI
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line  = np.linspace(x.min(), x.max(), 300)
    y_line  = slope * x_line + intercept
    n       = len(x)
    x_bar   = x.mean()
    se_line = se * np.sqrt(1/n + (x_line - x_bar)**2
                           / np.sum((x - x_bar)**2))
    ci      = 1.96 * se_line

    ax.plot(x_line, y_line, color=RED, linewidth=1.5,
            label=f'r={r:.3f}, slope={slope:.3f}')
    ax.fill_between(x_line, y_line - ci, y_line + ci,
                    color=RED, alpha=0.15)

    ax.set_xlabel('Launch angle (degrees)', fontsize=9)
    ax.set_ylabel('Predicted value', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Launch angle vs model predictions\nwith line of best fit (95% CI)',
                 fontsize=12, fontweight='bold', y=1.01)

    for ax, (stem, title, fitted_col) in zip(axes.flat, MODELS):
        path = os.path.join(OUT_DIR, f'{stem}.csv')
        if not os.path.exists(path):
            print(f'  SKIPPED: {stem}.csv not found')
            ax.text(0.5, 0.5, f'File not found:\n{stem}.csv',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color=RED)
            ax.set_title(title, fontweight='bold', fontsize=10)
            continue

        df = pd.read_csv(path)

        # For V2 (launch angle model), fitted values ARE launch angle predictions
        # so label the y-axis more specifically
        if 'launch_angle' == stem.split('_')[1] if len(stem.split('_')) > 1 else '':
            fitted_col = fitted_col

        # If launch_angle is the outcome column, use it directly
        if 'launch_angle' not in df.columns and fitted_col in df.columns:
            print(f'  WARNING: launch_angle missing from {stem} — skipping')
            ax.set_title(title, fontweight='bold', fontsize=10)
            ax.text(0.5, 0.5, 'launch_angle not in file',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=9, color=RED)
            continue

        print(f'  {stem}: {len(df):,} rows')
        panel(ax, df, title, fitted_col)

    plt.tight_layout()
    out_path = os.path.join(DIAG_DIR, 'la_vs_predictions_2x2.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()