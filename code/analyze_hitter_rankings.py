"""
Rank hitters by fastball timing skill and by fastball xwOBA, then correlate
the two rankings.

Timing score (user-specified):
    z_timing = (μ_hitter − x_peak) / σ_hitter
where μ_hitter and σ_hitter are the hitter's mean and std of timing on
fastballs, and x_peak is the population peak timing distance (~+12 in).
Higher = better. The sign is reversed from the typical z-score because
the variance is coming from the hitters and lower average timing is worse.

Alternative ranking metrics (for comparison):
  A. Simple mean timing μ_hitter
  B. % of swings within ±5 in of x_peak (\"time on target\")
  C. (μ_hitter − x_peak) / σ_population  — normalises by population variance
      instead of the hitter's own variance, so it doesn't reward consistency

xwOBA score:
    z_xwoba = (μ_xwoba_hitter − μ_xwoba_population) / σ_xwoba_hitter
Computed on in-play fastballs only (since xwOBA requires launch metrics).

Output:
  - Per-hitter ranking table (CSV) with all four timing metrics + xwOBA
  - Spearman correlations between timing rankings and xwOBA ranking
  - Scatter plot: mean fastball timing vs mean fastball xwOBA, one point per hitter
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_utils import (ALL_SWINGS, IN_PLAY, INTERCEPT_Y, FASTBALL_TYPES,
                          add_timing, load_pitches, peak_from_moving_average,
                          TIMING_AXIS_LABEL)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out")
MIN_SWINGS_FOR_TIMING = 50    # minimum fastball swings for timing rank
MIN_IP_FOR_XWOBA      = 30    # minimum in-play balls for xwOBA rank
WINDOW                = 5.0   # ±5 in window for "time on target" metric

# ── Load ──────────────────────────────────────────────────────────────────────
COLS = ['pitch_type', 'description', 'batter', 'stand', INTERCEPT_Y,
        'delta_run_exp', 'estimated_woba_using_speedangle']
raw = load_pitches(DATA_DIR, cols=COLS, pitch_types=FASTBALL_TYPES)
raw = raw.dropna(subset=[INTERCEPT_Y])

# Timing data: all swings (in-play + foul + miss)
swings = raw[raw['description'].isin(ALL_SWINGS)].copy()
center = add_timing(swings)
print(f"All-swing fastballs: {len(swings):,}  (timing centre = {center:.2f} in)")

# Population peak: x_peak from count-weighted moving average
x_peak, _ = peak_from_moving_average(
    swings['timing'].values, swings['delta_run_exp'].values,
    n_bins=40, ma_window=5, min_per_bin=100, min_window_total=400)
print(f"Population peak timing distance: x_peak = {x_peak:+.2f} in")
sigma_pop = swings['timing'].std()
print(f"Population timing std:           σ_pop  = {sigma_pop:.2f} in")

# In-play subset for xwOBA
inplay = raw[raw['description'].isin(IN_PLAY)].copy()
inplay = inplay.dropna(subset=['estimated_woba_using_speedangle'])
add_timing(inplay, center=center)
pop_xwoba_mean = inplay['estimated_woba_using_speedangle'].mean()
print(f"Population mean xwOBA (in-play fastballs): {pop_xwoba_mean:.4f}")
print(f"In-play fastballs with xwOBA: {len(inplay):,}")

# ── Per-hitter aggregates ────────────────────────────────────────────────────
def percent_on_target(arr, x_peak, w=WINDOW):
    arr = np.asarray(arr)
    return float(((arr >= x_peak - w) & (arr <= x_peak + w)).mean())

agg_swings = (
    swings.groupby('batter')
          .agg(n_swings=('timing', 'count'),
               mu_timing=('timing', 'mean'),
               sigma_timing=('timing', 'std'),
               stand=('stand', lambda s: s.mode().iloc[0]))
)
# Look up batter names via pybaseball (best-effort)
try:
    from pybaseball import playerid_reverse_lookup
    ids = agg_swings.index.tolist()
    name_df = playerid_reverse_lookup(ids, key_type='mlbam')[
        ['key_mlbam', 'name_first', 'name_last']
    ].set_index('key_mlbam')
    name_df['name'] = (name_df['name_last'].str.title() + ', ' +
                       name_df['name_first'].str.title())
    agg_swings = agg_swings.join(name_df['name'])
except Exception as exc:
    print(f"  (name lookup failed: {exc!r} — using batter IDs as names)")
    agg_swings['name'] = agg_swings.index.astype(str)
agg_swings['name'] = agg_swings['name'].fillna(agg_swings.index.to_series().astype(str))
agg_swings['frac_on_target'] = (
    swings.groupby('batter')['timing']
          .apply(lambda v: percent_on_target(v, x_peak))
)
agg_swings = agg_swings[agg_swings['n_swings'] >= MIN_SWINGS_FOR_TIMING].copy()

agg_inplay = (
    inplay.groupby('batter')
          .agg(n_inplay=('estimated_woba_using_speedangle', 'count'),
               mu_xwoba=('estimated_woba_using_speedangle', 'mean'),
               sigma_xwoba=('estimated_woba_using_speedangle', 'std'))
)
agg_inplay = agg_inplay[agg_inplay['n_inplay'] >= MIN_IP_FOR_XWOBA].copy()

# Combine
H = agg_swings.join(agg_inplay, how='inner')
print(f"\nHitters with both fastball thresholds met: {len(H)}")

# ── Compute metrics ──────────────────────────────────────────────────────────
H['z_timing_user']  = (H['mu_timing'] - x_peak) / H['sigma_timing']
H['z_timing_popsd'] = (H['mu_timing'] - x_peak) / sigma_pop
H['z_xwoba']        = (H['mu_xwoba']  - pop_xwoba_mean) / H['sigma_xwoba']

# Ranks (higher = better in all metrics by construction)
for col, name in [
    ('z_timing_user',  'rank_z_timing_user'),
    ('mu_timing',      'rank_mu_timing'),
    ('frac_on_target', 'rank_frac_on_target'),
    ('z_timing_popsd', 'rank_z_timing_popsd'),
    ('mu_xwoba',       'rank_mu_xwoba'),
    ('z_xwoba',        'rank_z_xwoba'),
]:
    H[name] = H[col].rank(ascending=False, method='average')

# ── Correlations between timing rankings and xwOBA rankings ──────────────────
print("\n══ Correlations: timing metric vs xwOBA metric (Spearman ρ, Pearson r) ══")
combos = [
    ('z_timing_user',  'z_xwoba',  "z_timing (user) → z_xwoba"),
    ('z_timing_user',  'mu_xwoba', "z_timing (user) → mean xwOBA"),
    ('mu_timing',      'mu_xwoba', "mean timing     → mean xwOBA"),
    ('frac_on_target', 'mu_xwoba', "% on target     → mean xwOBA"),
    ('z_timing_popsd', 'mu_xwoba', "z (pop σ)       → mean xwOBA"),
    ('mu_timing',      'z_xwoba',  "mean timing     → z_xwoba"),
]
for a, b, label in combos:
    sub = H[[a, b]].dropna()
    rho, p_s = spearmanr(sub[a], sub[b])
    r, p_r   = pearsonr(sub[a], sub[b])
    print(f"  {label:<40}  Spearman ρ={rho:+.3f} (p={p_s:.4g})  "
          f"Pearson r={r:+.3f} (p={p_r:.4g})  n={len(sub)}")

# Inter-correlation between the four timing metrics
print("\n══ Inter-correlations among timing metrics (Spearman ρ) ══")
timing_metrics = ['z_timing_user', 'mu_timing', 'frac_on_target', 'z_timing_popsd']
corr = H[timing_metrics].corr(method='spearman')
print(corr.round(3))

# ── Output ranking table ─────────────────────────────────────────────────────
H_out = H.reset_index().rename(columns={'index': 'batter'})
cols_out = ['batter', 'name', 'stand', 'n_swings', 'n_inplay',
            'mu_timing', 'sigma_timing', 'frac_on_target',
            'z_timing_user', 'z_timing_popsd',
            'mu_xwoba', 'sigma_xwoba', 'z_xwoba',
            'rank_z_timing_user', 'rank_mu_timing', 'rank_frac_on_target',
            'rank_z_timing_popsd', 'rank_mu_xwoba', 'rank_z_xwoba']
H_out = H_out[cols_out].sort_values('rank_z_timing_user')
out_csv = os.path.join(OUT_DIR, 'hitter_rankings_fastball.csv')
H_out.to_csv(out_csv, index=False, float_format='%.4f')
print(f"\nFull ranking table saved: {out_csv}")
print(f"  n hitters ranked: {len(H_out)}")

# Show top/bottom 10 by user z_timing
print("\nTop 10 hitters by z_timing (user metric):")
print(H_out[['name','n_swings','mu_timing','sigma_timing','z_timing_user',
            'mu_xwoba','z_xwoba']].head(10).to_string(index=False))
print("\nBottom 10 hitters by z_timing (user metric):")
print(H_out[['name','n_swings','mu_timing','sigma_timing','z_timing_user',
            'mu_xwoba','z_xwoba']].tail(10).to_string(index=False))

# ── Plot 1: rank-correlation matrix ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(timing_metrics))); ax.set_yticks(range(len(timing_metrics)))
ax.set_xticklabels(['z (user)', 'μ timing', '% on target', 'z (pop σ)'], rotation=30, ha='right')
ax.set_yticklabels(['z (user)', 'μ timing', '% on target', 'z (pop σ)'])
for i in range(len(timing_metrics)):
    for j in range(len(timing_metrics)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center',
                color='white' if abs(corr.iloc[i,j]) > 0.6 else 'black', fontsize=10)
fig.colorbar(im, ax=ax, label='Spearman ρ')
ax.set_title("Inter-correlation of timing metrics (Spearman)", fontsize=11)

ax = axes[1]
labels = []
rhos   = []
for a, _, label in combos:
    sub = H[[a, 'mu_xwoba']].dropna()
    rho, _ = spearmanr(sub[a], sub['mu_xwoba'])
    labels.append(label.split(' →')[0].strip())
    rhos.append(rho)
colors = ['#1f77b4' if r >= 0 else '#d62728' for r in rhos]
ax.barh(range(len(labels)), rhos, color=colors, alpha=0.75)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel("Spearman ρ with mean xwOBA", fontsize=10)
ax.set_title("Which timing metric best predicts xwOBA?", fontsize=11)
for i, r in enumerate(rhos):
    ax.text(r + (0.005 if r >= 0 else -0.005), i, f"{r:+.3f}",
            va='center', ha='left' if r >= 0 else 'right', fontsize=9)

fig.suptitle("Hitter Ranking Metrics — Timing vs xwOBA (2025 Fastballs)",
             fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'hitter_rankings_metric_comparison.png'),
            dpi=160, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved → hitter_rankings_metric_comparison.png")

# ── Plot 2: PRIMARY scatter — mean timing vs mean xwOBA, one point per hitter
fig, ax = plt.subplots(figsize=(11, 8))
x_plot = H['mu_timing']
y_plot = H['mu_xwoba']
sizes  = (H['n_swings'] - H['n_swings'].min()) / (H['n_swings'].max() - H['n_swings'].min()) * 100 + 20

sc = ax.scatter(x_plot, y_plot, s=sizes, c=H['z_timing_user'],
                cmap='RdYlGn', alpha=0.75, edgecolors='k', linewidths=0.4)
fig.colorbar(sc, ax=ax, label='z_timing (user)', pad=0.02)

# Reference lines
ax.axvline(x_peak, color='black', lw=1.2, ls='--', alpha=0.7,
           label=f'Population peak = {x_peak:+.1f} in')
ax.axhline(pop_xwoba_mean, color='grey', lw=0.8, ls=':',
           label=f'Population mean xwOBA = {pop_xwoba_mean:.3f}')

# Trend line and stats
rho, p_s = spearmanr(x_plot, y_plot)
r, p_r   = pearsonr(x_plot, y_plot)
m, b = np.polyfit(x_plot, y_plot, 1)
xs = np.linspace(x_plot.min(), x_plot.max(), 100)
ax.plot(xs, m*xs + b, color='royalblue', lw=2, alpha=0.7,
        label=f'OLS:  ρ={rho:+.3f}  r={r:+.3f}')

# Annotate extremes
top5_t = H.nlargest(5, 'mu_timing')
bot5_t = H.nsmallest(5, 'mu_timing')
top5_x = H.nlargest(5, 'mu_xwoba')
for _, row in pd.concat([top5_t, bot5_t, top5_x]).drop_duplicates().iterrows():
    label_name = row['name'].split(',')[0] if isinstance(row['name'], str) else str(row.name)
    ax.annotate(label_name, (row['mu_timing'], row['mu_xwoba']),
                fontsize=7, alpha=0.75, xytext=(3, 3), textcoords='offset points')

ax.set_xlabel("Mean Fastball Timing (in, centred on median)\n"
              "← oppo/late                            pull/early →", fontsize=10)
ax.set_ylabel("Mean xwOBA on Fastballs (in-play)", fontsize=10)
ax.set_title(f"Per-Hitter Mean Fastball Timing vs Mean Fastball xwOBA\n"
             f"n={len(H)} hitters  ·  Spearman ρ={rho:+.3f}  Pearson r={r:+.3f}  "
             f"(p={'<0.0001' if min(p_s, p_r)<0.0001 else f'{min(p_s,p_r):.4f}'})  ·  "
             f"size ∝ n_swings",
             fontsize=11)
ax.legend(fontsize=9, loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'hitter_timing_vs_xwoba.png'),
            dpi=160, bbox_inches='tight')
plt.close(fig)
print(f"Saved → hitter_timing_vs_xwoba.png")

print("\nDone.")
