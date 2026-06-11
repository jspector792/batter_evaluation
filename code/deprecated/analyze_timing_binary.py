"""
Timing Binary Hypothesis Test — 2025 MLB Fastballs (all swings).

Timing proxy: intercept_y − population median (inches, pull/early-positive).
Per-hitter peaks use a smoothing-spline pipeline with a side-significance test
(see timing_utils.smoothed_peak). Non-significant peaks are treated as missing.

Analyses:
  1. Distribution of per-hitter peak timing (significant peaks only)
  2. Same, broken out by batter handedness
  3. Same, paired same-hand vs diff-hand matchup peaks per batter
  4. H1-style LW vs timing: batter handedness × pitcher matchup subsets
  5. H1-style LW vs timing: zone subsets (inside / middle / outside)
  6. Bimodality check on peak distribution
  7. Mean timing per hitter with std error bars
"""
import os, sys, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde, ttest_ind, ttest_rel, mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from timing_utils import (ALL_SWINGS, CONTACT, IN_PLAY,
                          INTERCEPT_X, INTERCEPT_Y, FASTBALL_TYPES,
                          INSIDE_Z, MIDDLE_Z, OUTSIDE_Z,
                          add_timing, add_zone_and_matchup, smoothed_peak,
                          binned_lw, weighted_moving_average,
                          plot_lw_curves as _tu_plot_lw_curves,
                          TIMING_AXIS_LABEL)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out")
MIN_PA   = 50    # minimum swings for per-hitter peak
MIN_PAIR = 30    # minimum per matchup-type for paired matchup analysis

COLS = [
    "pitch_type", "batter", "stand", "p_throws", "zone",
    INTERCEPT_X, INTERCEPT_Y,
    "delta_run_exp", "description",
]

# ── Load ─────────────────────────────────────────────────────────────────────
files  = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
chunks = [pd.read_csv(f, usecols=COLS) for f in files]
df = pd.concat(chunks, ignore_index=True)
df = df[df['pitch_type'].isin(FASTBALL_TYPES)].copy()
df = df[df['description'].isin(ALL_SWINGS)].copy()

df = df.dropna(subset=['delta_run_exp', 'stand', 'p_throws'])
df = df.dropna(subset=[INTERCEPT_Y, 'delta_run_exp'])
center = add_timing(df)
print(f"Swings with all fields: {len(df):,}")
print(f"  intercept_y centre (median, inches): {center:.2f}")
print(f"  timing: min={df['timing'].min():.1f}  max={df['timing'].max():.1f}  "
      f"median={df['timing'].median():.1f}")

# ── Derived columns ───────────────────────────────────────────────────────────
# matchup column added by add_zone_and_matchup() (called above)

# zone_label provided by timing_utils.zone_group
add_zone_and_matchup(df)

print(f"Stand:   {df['stand'].value_counts().to_dict()}")
print(f"Matchup: {df['matchup'].value_counts().to_dict()}")
print(f"Zone:    {df['zone_grp'].value_counts().to_dict()}")

# ── Peak-finding helper ───────────────────────────────────────────────────────
def peak_timing(data, min_n=MIN_PA, **kw):
    """Return per-hitter peak timing (in) — NaN unless statistically significant."""
    if len(data) < min_n:
        return np.nan
    loc, _, _, sig = smoothed_peak(data['timing'].values,
                                    data['delta_run_exp'].values,
                                    min_n=min_n, **kw)
    return loc if sig else np.nan

# ── LW-vs-timing curve plotter ───────────────────────────────────────────────
# Thin wrapper around timing_utils.plot_lw_curves that adds title + axis labels.
def plot_lw_curves(ax, subsets, title, show_bins=True):
    """Overlay smoothed LW-vs-timing curves with this script's standard
    formatting (title + axis labels + legend). Delegates the count-weighted
    moving-average computation to timing_utils.plot_lw_curves."""
    _tu_plot_lw_curves(ax, subsets, n_bins=24, ma_window=5,
                        show_bins=show_bins, min_n_data=300)
    ax.set_xlabel(TIMING_AXIS_LABEL, fontsize=9)
    ax.set_ylabel("Mean Δ Run Expectancy", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

# ═══════════════════════════════════════════════════════════════════════════════
# Per-hitter peak calculations
# ═══════════════════════════════════════════════════════════════════════════════
print("\nCalculating per-hitter peak timing (inches)...")

# Diagnostic: how many hitters have a peak at all vs how many are significant
def _peak_diag(data, min_n=MIN_PA):
    if len(data) < min_n:
        return pd.Series({'loc': np.nan, 'sig': False})
    loc, _, _, sig = smoothed_peak(data['timing'].values,
                                    data['delta_run_exp'].values,
                                    min_n=min_n)
    return pd.Series({'loc': loc, 'sig': sig})

_diag = df.groupby('batter').apply(_peak_diag)
_n_any = int(_diag['loc'].notna().sum())
_n_sig = int(_diag['sig'].sum())
print(f"  Hitters with a (any) spline peak:    {_n_any}")
print(f"  Hitters with a SIGNIFICANT peak:     {_n_sig}  "
      f"({(_n_sig/_n_any if _n_any else 0):.1%} of those)")

# Analysis 1 — all swings per batter (only significant peaks retained)
all_peaks = (df.groupby('batter')
               .apply(peak_timing)
               .dropna()
               .rename('peak'))
print(f"\nAll batters with significant peak (≥{MIN_PA} swings): n={len(all_peaks)}")
print(f"  mean={all_peaks.mean():.2f} in  median={all_peaks.median():.2f} in  "
      f"std={all_peaks.std():.2f} in")
print(f"  % peaking above 0 in (pull/early): {(all_peaks > 0).mean():.1%}")

# Analysis 2 — per batter, separated by batter handedness
stand_col = df.groupby('batter')['stand'].agg(lambda s: s.mode()[0])
all_peaks_df = pd.DataFrame({'peak': all_peaks, 'stand': stand_col})

l_peaks = all_peaks_df.loc[all_peaks_df['stand'] == 'L', 'peak']
r_peaks = all_peaks_df.loc[all_peaks_df['stand'] == 'R', 'peak']
print(f"\nLHH: n={len(l_peaks)}  mean={l_peaks.mean():.2f} in  median={l_peaks.median():.2f} in")
print(f"RHH: n={len(r_peaks)}  mean={r_peaks.mean():.2f} in  median={r_peaks.median():.2f} in")
t_lr, p_lr = ttest_ind(l_peaks.dropna(), r_peaks.dropna())
print(f"LHH vs RHH t-test: t={t_lr:.3f}  p={p_lr:.4f}")

# Analysis 3 — paired same-hand vs diff-hand peaks per batter
same_peaks = (df[df['matchup']=='same-hand']
              .groupby('batter')
              .apply(lambda g: peak_timing(g, min_n=MIN_PAIR))
              .dropna()
              .rename('same_peak'))
diff_peaks = (df[df['matchup']=='diff-hand']
              .groupby('batter')
              .apply(lambda g: peak_timing(g, min_n=MIN_PAIR))
              .dropna()
              .rename('diff_peak'))

paired = pd.DataFrame({'same_peak': same_peaks, 'diff_peak': diff_peaks}).dropna()
paired['delta_peak'] = paired['diff_peak'] - paired['same_peak']
paired['stand'] = stand_col.reindex(paired.index)
print(f"\nPaired matchup analysis (≥{MIN_PAIR} swings each, both peaks significant): n={len(paired)}")
print(f"  Same-hand mean={paired['same_peak'].mean():.2f} in  "
      f"Diff-hand mean={paired['diff_peak'].mean():.2f} in")
print(f"  Mean shift (diff − same) = {paired['delta_peak'].mean():.2f} in")
t_p, p_p = ttest_rel(paired['same_peak'], paired['diff_peak'])
print(f"  Paired t-test: t={t_p:.3f}  p={p_p:.4f}")
for hand in ['L','R']:
    sub = paired[paired['stand'] == hand]
    if len(sub) < 2:
        continue
    t_h, p_h = ttest_rel(sub['same_peak'], sub['diff_peak'])
    print(f"  {hand}HH: same mean={sub['same_peak'].mean():.2f} in  "
          f"diff mean={sub['diff_peak'].mean():.2f} in  "
          f"shift={sub['delta_peak'].mean():.2f} in  p={p_h:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Fig 1: All-batter peak distribution ──────────────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

ax = axes1[0]
ax.hist(all_peaks.values, bins=35, edgecolor='k', lw=0.4,
        color='steelblue', alpha=0.75, density=True, label='histogram')
kde = gaussian_kde(all_peaks.values)
xs  = np.linspace(all_peaks.min(), all_peaks.max(), 400)
ax.plot(xs, kde(xs), 'r-', lw=2.5, label='KDE')
ax.axvline(0,                   color='black', lw=1.2, ls='--', label='0 in (median)')
ax.axvline(all_peaks.median(),  color='green',  lw=1.5, ls=':',
           label=f"Median peak = {all_peaks.median():.1f} in")
ax.axvline(all_peaks.mean(),    color='orange', lw=1.5, ls=':',
           label=f"Mean peak = {all_peaks.mean():.1f} in")
ax.set_xlabel("Per-Hitter Peak Timing (in, centred on median)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"Distribution of Per-Hitter Peak Timing (significant only)\n"
             f"(n={len(all_peaks)} batters, ≥{MIN_PA} swings each)", fontsize=11)
ax.legend(fontsize=8)

ax = axes1[1]
sorted_peaks = np.sort(all_peaks.values)
cdf = np.arange(1, len(sorted_peaks)+1) / len(sorted_peaks)
ax.plot(sorted_peaks, cdf, color='steelblue', lw=2)
ax.axvline(0, color='black', lw=1.2, ls='--', label='0 in (median)')
ax.axhline(0.5, color='grey', lw=0.8, ls=':')
pct_up = (all_peaks > 0).mean()
ax.fill_betweenx([0,1], sorted_peaks.min(), 0, alpha=0.08, color='green',
                 label=f'Peak below 0 in (oppo/late): {1-pct_up:.1%}')
ax.fill_betweenx([0,1], 0, sorted_peaks.max(), alpha=0.08, color='red',
                 label=f'Peak above 0 in (pull/early): {pct_up:.1%}')
ax.set_xlabel("Peak Timing (in, centred on median)", fontsize=11)
ax.set_ylabel("Cumulative Fraction of Batters", fontsize=11)
ax.set_title("CDF of Per-Hitter Peak Timing", fontsize=11)
ax.legend(fontsize=8)

plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'timing_binary_1_all.png'), dpi=180, bbox_inches='tight')
plt.close(fig1)
print("\nSaved timing_binary_1_all.png")

# ── Fig 2: L vs R batter peak distributions + curve ──────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

ax = axes2[0]
for sub, label, color, ls in [
    (l_peaks, f'LHH (n={len(l_peaks)})', '#1f77b4', '-'),
    (r_peaks, f'RHH (n={len(r_peaks)})', '#d62728', '--'),
]:
    ax.hist(sub.values, bins=25, color=color, alpha=0.45, density=True,
            edgecolor='k', lw=0.3)
    kde_h = gaussian_kde(sub.dropna().values)
    xs = np.linspace(sub.min(), sub.max(), 300)
    ax.plot(xs, kde_h(xs), color=color, lw=2.5, ls=ls, label=label)
ax.axvline(0, color='black', lw=1.2, ls='--')
ax.set_xlabel("Peak Timing (in, centred on median)", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(f"Per-Hitter Peak: LHH vs RHH\n"
             f"t-test p={p_lr:.4f}  "
             f"LHH mean={l_peaks.mean():.1f} in  RHH mean={r_peaks.mean():.1f} in", fontsize=10)
ax.legend(fontsize=9)

plot_lw_curves(axes2[1], {
    'LHH': df[df['stand']=='L'],
    'RHH': df[df['stand']=='R'],
}, "Mean LW vs Timing\nLHH vs RHH")

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'timing_binary_2_handedness.png'), dpi=180, bbox_inches='tight')
plt.close(fig2)
print("Saved timing_binary_2_handedness.png")

# ── Fig 3: Paired same-hand vs diff-hand matchup peaks ───────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(20, 5))

ax = axes3[0]
for col, label, color, ls in [
    ('same_peak', f'same-hand (n={len(paired)})', '#ff7f0e', '-'),
    ('diff_peak',  f'diff-hand  (n={len(paired)})', '#9467bd', '--'),
]:
    vals = paired[col].values
    ax.hist(vals, bins=25, color=color, alpha=0.45, density=True, edgecolor='k', lw=0.3)
    kde_h = gaussian_kde(vals)
    xs = np.linspace(vals.min(), vals.max(), 300)
    ax.plot(xs, kde_h(xs), color=color, lw=2.5, ls=ls, label=label)
ax.axvline(0, color='black', lw=1.2, ls='--')
ax.set_xlabel("Peak Timing (in, centred on median)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title(f"Paired Same vs Diff Handedness Peaks\n"
             f"Mean shift (diff−same) = {paired['delta_peak'].mean():.2f} in  p={p_p:.4f}", fontsize=10)
ax.legend(fontsize=8)

ax = axes3[1]
for hand, color in [('L','#1f77b4'),('R','#d62728'),('all','grey')]:
    if hand == 'all':
        vals = paired['delta_peak'].values
        label = f'All (n={len(vals)})'
    else:
        vals = paired[paired['stand']==hand]['delta_peak'].values
        label = f'{hand}HH (n={len(vals)})'
    if len(vals) > 5:
        kde_h = gaussian_kde(vals)
        xs = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(xs, kde_h(xs), color=color, lw=2, label=label)
ax.axvline(0, color='black', lw=1.2, ls='--', label='No shift')
ax.set_xlabel("Shift in Peak Timing: Diff-Hand − Same-Hand (in)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("Do Batters Shift Peak Timing vs Different-Hand Pitchers?", fontsize=10)
ax.legend(fontsize=8)

plot_lw_curves(axes3[2], {
    'same-hand': df[df['matchup']=='same-hand'],
    'diff-hand':  df[df['matchup']=='diff-hand'],
}, "Mean LW vs Timing\nSame vs Diff Handedness Matchup")

plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'timing_binary_3_matchup.png'), dpi=180, bbox_inches='tight')
plt.close(fig3)
print("Saved timing_binary_3_matchup.png")

# ── Fig 4: H1-style 2×2 by batter hand × pitcher hand ───────────────────────
fig4, axes4 = plt.subplots(2, 2, figsize=(16, 10))

plot_lw_curves(axes4[0,0], {
    'LHH': df[df['stand']=='L'],
    'RHH': df[df['stand']=='R'],
}, "LW vs Timing: All LHH vs All RHH")

plot_lw_curves(axes4[0,1], {
    'same-hand': df[df['matchup']=='same-hand'],
    'diff-hand':  df[df['matchup']=='diff-hand'],
}, "LW vs Timing: Same vs Diff Handedness (all batters)")

plot_lw_curves(axes4[1,0], {
    'LHH vs LHP (same)': df[(df['stand']=='L') & (df['p_throws']=='L')],
    'LHH vs RHP (diff)': df[(df['stand']=='L') & (df['p_throws']=='R')],
}, "LHH: vs LHP (same-hand) vs vs RHP (diff-hand)")

plot_lw_curves(axes4[1,1], {
    'RHH vs RHP (same)': df[(df['stand']=='R') & (df['p_throws']=='R')],
    'RHH vs LHP (diff)': df[(df['stand']=='R') & (df['p_throws']=='L')],
}, "RHH: vs RHP (same-hand) vs vs LHP (diff-hand)")

fig4.suptitle("Analysis 4: H1-Style LW vs Timing — Handedness Breakdown",
              fontsize=13, y=1.01)
plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'timing_binary_4_h1_matchup.png'), dpi=180, bbox_inches='tight')
plt.close(fig4)
print("Saved timing_binary_4_h1_matchup.png")

# ── Fig 5: H1-style by zone ───────────────────────────────────────────────────
fig5, axes5 = plt.subplots(2, 2, figsize=(16, 10))

plot_lw_curves(axes5[0,0], {
    'inside':  df[df['zone_grp']=='inside'],
    'middle':  df[df['zone_grp']=='middle'],
    'outside': df[df['zone_grp']=='outside'],
}, "LW vs Timing by Pitch Zone (all batters)")

plot_lw_curves(axes5[0,1], {
    'RHH — inside':  df[(df['stand']=='R') & (df['zone_grp']=='inside')],
    'RHH — outside': df[(df['stand']=='R') & (df['zone_grp']=='outside')],
    'LHH — inside':  df[(df['stand']=='L') & (df['zone_grp']=='inside')],
    'LHH — outside': df[(df['stand']=='L') & (df['zone_grp']=='outside')],
}, "LW vs Timing: Zone × Batter Handedness")

plot_lw_curves(axes5[1,0], {
    'same-hand, inside':  df[(df['matchup']=='same-hand') & (df['zone_grp']=='inside')],
    'same-hand, outside': df[(df['matchup']=='same-hand') & (df['zone_grp']=='outside')],
    'diff-hand, inside':  df[(df['matchup']=='diff-hand') & (df['zone_grp']=='inside')],
    'diff-hand, outside': df[(df['matchup']=='diff-hand') & (df['zone_grp']=='outside')],
}, "LW vs Timing: Matchup × Zone")

plot_lw_curves(axes5[1,1], {
    'inside':  df[df['zone_grp']=='inside'],
    'middle':  df[df['zone_grp']=='middle'],
    'outside': df[df['zone_grp']=='outside'],
}, "LW vs Timing by Zone\n(replicated for reference)")

fig5.suptitle("Analysis 5: H1-Style LW vs Timing — Zone Location Breakdown",
              fontsize=13, y=1.01)
plt.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, 'timing_binary_5_zones.png'), dpi=180, bbox_inches='tight')
plt.close(fig5)
print("Saved timing_binary_5_zones.png")

# ── Fig 6: Bimodality check ───────────────────────────────────────────────────
fig6, axes6 = plt.subplots(1, 3, figsize=(20, 5))

ax = axes6[0]
for bw_scale, ls, alpha in [(0.3, '-', 1.0), (1.0, '--', 0.8), (2.5, ':', 0.7)]:
    bw = bw_scale / all_peaks.std()
    kde_bw = gaussian_kde(all_peaks.values, bw_method=bw)
    xs = np.linspace(all_peaks.min(), all_peaks.max(), 400)
    ax.plot(xs, kde_bw(xs), lw=2, ls=ls, label=f'BW factor={bw_scale} in')
ax.axvline(0, color='black', lw=1, ls='--', label='0 in (median)')
ax.set_xlabel("Peak Timing (in, centred on median)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title(f"Bimodality Check: KDE at Multiple Bandwidths\n"
             f"(n={len(all_peaks)} significant peaks)", fontsize=10)
ax.legend(fontsize=8)

ax = axes6[1]
sc = ax.scatter(
    all_peaks_df.dropna()['peak'].reindex(all_peaks_df.dropna().index),
    [df[df['batter']==b].shape[0] for b in all_peaks_df.dropna().index],
    c=all_peaks_df.dropna()['stand'].map({'L':'steelblue','R':'tomato'}),
    alpha=0.4, s=12
)
ax.axvline(0, color='black', lw=1, ls='--')
ax.set_xlabel("Peak Timing (in, centred on median)", fontsize=10)
ax.set_ylabel("N swings", fontsize=10)
ax.set_title("Peak Timing vs Sample Size\n(stability check)", fontsize=10)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='steelblue', label='LHH'), Patch(color='tomato', label='RHH')], fontsize=8)

ax = axes6[2]
for hand, color, marker in [('L','#1f77b4','o'),('R','#d62728','^')]:
    sub = paired[paired['stand']==hand]
    ax.scatter(sub['same_peak'], sub['diff_peak'], color=color, alpha=0.4,
               s=15, marker=marker, label=f'{hand}HH (n={len(sub)})')
lims = [min(paired['same_peak'].min(), paired['diff_peak'].min()),
        max(paired['same_peak'].max(), paired['diff_peak'].max())]
ax.plot(lims, lims, 'k--', lw=1, label='same peak (y=x)')
ax.set_xlabel("Peak Timing — Same-Hand PAs (in)", fontsize=10)
ax.set_ylabel("Peak Timing — Diff-Hand PAs (in)", fontsize=10)
ax.set_title(f"Do Batters Consistently Shift Peak Timing by Matchup?\n"
             f"r = {paired['same_peak'].corr(paired['diff_peak']):.3f}", fontsize=10)
ax.legend(fontsize=8)

fig6.suptitle("Additional: Bimodality Check & Timing Stability", fontsize=12, y=1.01)
plt.tight_layout()
fig6.savefig(os.path.join(OUT_DIR, 'timing_binary_6_bimodality.png'), dpi=180, bbox_inches='tight')
plt.close(fig6)
print("Saved timing_binary_6_bimodality.png")

# ── Fig 7: Mean attack angle per hitter with std error bars ──────────────────
hitter_stats = (
    df.groupby('batter')['timing']
    .agg(mean_timing='mean', std_timing='std', n='count')
    .dropna()
)
hitter_stats['stand'] = stand_col.reindex(hitter_stats.index)
hitter_stats['peak']  = all_peaks.reindex(hitter_stats.index)
hitter_stats = hitter_stats.dropna(subset=['stand'])
hitter_stats = hitter_stats[hitter_stats['n'] >= MIN_PA]

print(f"\nFig 7: {len(hitter_stats)} hitters with ≥{MIN_PA} swings")
print(f"  Mean timing: mean={hitter_stats['mean_timing'].mean():.2f} in  "
      f"std={hitter_stats['mean_timing'].std():.2f} in")
print(f"  Mean std (within-hitter spread): {hitter_stats['std_timing'].mean():.2f} in")

both = hitter_stats.dropna(subset=['peak', 'mean_timing'])
r_mean_peak = both['mean_timing'].corr(both['peak'])
print(f"  Correlation mean_timing vs peak_timing (significant only): "
      f"r={r_mean_peak:.3f}  (n={len(both)})")

fig7, ax7 = plt.subplots(figsize=(13, 7))

for hand, color, marker, zorder in [('L', 'steelblue', 'o', 3), ('R', 'tomato', '^', 2)]:
    sub = hitter_stats[hitter_stats['stand'] == hand]
    sub = sub.sort_values('mean_timing')
    ax7.errorbar(
        sub['mean_timing'], sub['n'],
        xerr=sub['std_timing'],
        fmt='none',
        ecolor=color, alpha=0.25, elinewidth=0.8, capsize=0, zorder=zorder,
    )
    ax7.scatter(
        sub['mean_timing'], sub['n'],
        c=color, alpha=0.6, s=18, marker=marker,
        label=f"{hand}HH (n={len(sub)}, mean={sub['mean_timing'].mean():.1f} in)",
        zorder=zorder + 1,
    )

ax7.axvline(0, color='black', lw=1.2, ls='--', label='0 in (median)')
ax7.set_xlabel(
    "Mean Timing per Hitter (in, centred on median)\n"
    "← oppo/late                           pull/early →",
    fontsize=11,
)
ax7.set_ylabel("N Swings", fontsize=11)
ax7.set_title(
    f"Mean Timing per Hitter (all swings, not just peaks)\n"
    f"Error bars = ±1 SD within hitter   "
    f"Corr(mean, peak) r={r_mean_peak:.3f}   n={len(hitter_stats)} hitters",
    fontsize=12,
)
ax7.legend(fontsize=9)

plt.tight_layout()
fig7.savefig(os.path.join(OUT_DIR, 'timing_binary_7_mean_timing.png'), dpi=180, bbox_inches='tight')
plt.close(fig7)
print("Saved timing_binary_7_mean_timing.png")

print("\nDone.")
