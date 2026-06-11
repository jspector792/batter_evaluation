"""
Systematic comparison of timing proxies for fastball swings.

Candidates:
  timing_y      = intercept_y - population median (inches)   [primary - linear distance]
  timing_angle  = atan2(intercept_y, intercept_x)  (deg)      [standard polar]
  timing_angle2 = atan2(intercept_x, intercept_y)  (deg)      [prior-convention variant]
  attack_dir    = attack_direction                   (deg)    [Statcast bat-tracking horizontal]

Questions:
  1. Sign convention: does positive = pull or oppo? Validated via hc_x (spray direction).
  2. Per-hitter predictive quality: which measure produces higher polynomial R^2 for
     (timing, delta_run_exp) curves?
  3. Intercorrelation: are they measuring the same thing or complementary things?
  4. Combination: does using multiple measures in a linear model outperform either alone?
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from numpy.polynomial import polynomial as P

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from timing_utils import (ALL_SWINGS, CONTACT, IN_PLAY, INTERCEPT_X, INTERCEPT_Y,
                          add_timing, smoothed_peak, binned_lw, TIMING_AXIS_LABEL, FASTBALL_TYPES, INSIDE_Z, MIDDLE_Z, OUTSIDE_Z, add_zone_and_matchup)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out")

COLS = [
    "pitch_type", "batter", "stand", "description",
    "delta_run_exp",
    INTERCEPT_X,
    INTERCEPT_Y,
    "attack_direction",
    "hc_x", "hc_y",
]
MIN_N = 50          # hitters qualifying for per-hitter analysis
POLY_DEG = 4        # polynomial degree for LW curves
TIMING_BINS = 40    # for population LW curves

# ── Load ──────────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
df = pd.concat([pd.read_csv(f, usecols=COLS) for f in files], ignore_index=True)
# Use ALL_SWINGS for the timing-side comparison (the primary measure should
# be defined consistently with the rest of the project).
df = df[df['pitch_type'].isin(FASTBALL_TYPES) & df['description'].isin(ALL_SWINGS)].copy()
df = df.dropna(subset=['delta_run_exp', 'stand', INTERCEPT_Y])

# Primary measure: linear y-distance, centred on population median.
TIMING_CENTER_Y = add_timing(df)         # adds df['timing'] = intercept_y - median
df = df.rename(columns={'timing': 'timing_y'})

ix = df[INTERCEPT_X]
iy = df[INTERCEPT_Y]
df['timing_angle']  = np.degrees(np.arctan2(iy, ix))   # atan2(y,x) - standard polar
df['timing_angle2'] = np.degrees(np.arctan2(ix, iy))   # atan2(x,y) - prior convention
df['attack_dir']    = df['attack_direction']

# centre the angle measures on their own medians for comparability
df['timing_angle']  -= df['timing_angle'].median()
df['timing_angle2'] -= df['timing_angle2'].median()

# drop rows missing all candidates
df = df.dropna(subset=['timing_angle', 'attack_dir'])
print(f"Swings with all timing measures: {len(df):,}")
print(f"  attack_dir available: {df['attack_dir'].notna().sum():,}")
print(f"  hc_x available (in-play only): {df['hc_x'].notna().sum():,}")

# ── Helper: population binned LW ──────────────────────────────────────────────
def pop_lw(x, y, n=TIMING_BINS):
    edges = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), n+1)
    bins  = pd.cut(x, bins=edges)
    agg   = pd.DataFrame({'x': x, 'y': y, 'b': bins}).groupby('b', observed=True)
    return agg['x'].mean().values, agg['y'].mean().values, agg['y'].count().values

# ── Helper: per-hitter polynomial R² ─────────────────────────────────────────
def hitter_poly_r2(sub, xcol):
    x = sub[xcol].values
    y = sub['delta_run_exp'].values
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < MIN_N:
        return np.nan
    xc = x - x.mean()
    coeffs = np.polyfit(xc, y, POLY_DEG)
    yhat = np.polyval(coeffs, xc)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

# ── Per-hitter R² for all three measures ────────────────────────────────────
hitters = df.groupby('batter').filter(lambda g: len(g) >= MIN_N)['batter'].unique()
print(f"\nHitters with ≥{MIN_N} contact fastballs: {len(hitters)}")

rows = []
for b in hitters:
    sub = df[df['batter'] == b]
    r2_ty  = hitter_poly_r2(sub, 'timing_y')
    r2_ta  = hitter_poly_r2(sub, 'timing_angle')
    r2_ta2 = hitter_poly_r2(sub, 'timing_angle2')
    r2_ad  = hitter_poly_r2(sub, 'attack_dir')

    # combination: linear model with timing_y and attack_dir
    sub2 = sub.dropna(subset=['timing_y', 'attack_dir', 'delta_run_exp'])
    r2_combo = np.nan
    if len(sub2) >= MIN_N:
        X = sub2[['timing_y', 'attack_dir']].values
        X = X - X.mean(axis=0)
        y = sub2['delta_run_exp'].values
        reg = LinearRegression().fit(X, y)
        yhat = reg.predict(X)
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2_combo = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    stand = sub['stand'].iloc[0]
    n     = len(sub)
    rows.append({'batter': b, 'stand': stand, 'n': n,
                 'r2_ty': r2_ty, 'r2_ta': r2_ta, 'r2_ta2': r2_ta2,
                 'r2_ad': r2_ad, 'r2_combo': r2_combo})

per_hitter = pd.DataFrame(rows)

for col, label in [('r2_ty','intercept_y (in)  [primary]'),
                   ('r2_ta','timing_angle (atan2 y,x)'),
                   ('r2_ta2','timing_angle2 (atan2 x,y)'),
                   ('r2_ad','attack_direction'),
                   ('r2_combo','combo (intercept_y + attack_dir)')]:
    vals = per_hitter[col].dropna()
    print(f"  {label:40s}  median R^2 = {vals.median():.4f}  "
          f"mean = {vals.mean():.4f}  n = {len(vals)}")

# ── Population-level correlation with hc_x (spray direction) ─────────────────
print("\n--- Spray direction correlation (in-play only, hc_x available) ---")
inplay = df.dropna(subset=['hc_x'])
for hand in ['R','L']:
    sub = inplay[inplay['stand']==hand]
    if len(sub) < 100:
        continue
    for col, label in [('timing_y','intercept_y (in)'),
                       ('timing_angle','atan2(y,x)'),
                       ('timing_angle2','atan2(x,y)'),
                       ('attack_dir','attack_direction')]:
        mask = sub[col].notna()
        r, p = pearsonr(sub.loc[mask, col], sub.loc[mask, 'hc_x'])
        print(f"  {hand}HH  {label:20s}  r(hc_x) = {r:+.3f}  p={'<0.0001' if p<0.0001 else f'{p:.4f}'}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Sign convention — hexbin of each measure vs hc_x, by handedness
# ═══════════════════════════════════════════════════════════════════════════
inplay = df.dropna(subset=['hc_x'])

fig, axes = plt.subplots(2, 4, figsize=(24, 11))
measures = [
    ('timing_y',      'intercept_y centred (in)  [primary]'),
    ('timing_angle',  'Timing Angle atan2(y,x) (deg)'),
    ('timing_angle2', 'Timing Angle atan2(x,y) (deg)'),
    ('attack_dir',    'Attack Direction (deg)'),
]
for row, (hand, hand_label) in enumerate([('R','RHH'),('L','LHH')]):
    sub = inplay[inplay['stand']==hand]
    for col_idx, (col, xlabel) in enumerate(measures):
        ax = axes[row, col_idx]
        vals = sub[[col, 'hc_x']].dropna()
        xp = vals[col].clip(*np.percentile(vals[col], [0.5, 99.5]))
        yp = vals['hc_x']
        r, _ = pearsonr(vals[col], vals['hc_x'])
        hb = ax.hexbin(xp, yp, gridsize=50, cmap='YlOrRd', mincnt=3, linewidths=0.1)
        fig.colorbar(hb, ax=ax, pad=0.01).set_label("Count", fontsize=7)
        ax.axvline(0, color='k', lw=1, ls='--', alpha=0.6)
        ax.axhline(np.median(yp), color='royalblue', lw=1, ls=':', alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("hc_x (spray direction; + = first base side)" if col_idx==0 else "", fontsize=8)
        ax.set_title(f"{hand_label}  ·  r = {r:+.3f}   n = {len(vals):,}", fontsize=10)
        if col_idx == 0:
            ax.text(0.02, 0.97, f'{hand_label}', transform=ax.transAxes,
                    fontsize=11, fontweight='bold', va='top')

fig.suptitle("Sign Convention Validation: Timing Measure vs Spray Direction (hc_x)\n"
             "High hc_x ~ first-base side; Low hc_x ~ third-base side  ·  "
             "Primary measure (intercept_y) shown first",
             fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'timing_comparison_1_spray.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("\nSaved → timing_comparison_1_spray.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Population LW curves side by side (4 panels)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(26, 6))

for ax, (col, label, unit, color) in zip(axes, [
    ('timing_y',      'intercept_y (centred)',     'in',  '#9467bd'),
    ('timing_angle',  'Timing Angle atan2(y,x)',   'deg', '#1f77b4'),
    ('timing_angle2', 'Timing Angle atan2(x,y)',   'deg', '#ff7f0e'),
    ('attack_dir',    'Attack Direction',          'deg', '#2ca02c'),
]):
    sub = df.dropna(subset=[col, 'delta_run_exp'])
    x = sub[col].values
    y = sub['delta_run_exp'].values
    xm, ym, nm = pop_lw(x, y)
    ax.plot(xm, ym, color=color, lw=2.5, label='Binned mean')
    ax.fill_between(xm, ym, alpha=0.15, color=color)
    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.axvline(xm[np.argmax(ym)], color='red', lw=1.2, ls='--', alpha=0.7,
               label=f'Pop peak = {xm[np.argmax(ym)]:.1f} {unit}')
    r_pop, _ = pearsonr(x, y)
    ax.set_xlabel(f"{label} ({unit})", fontsize=10)
    ax.set_ylabel("Mean Delta Run Expectancy", fontsize=10)
    ax.set_title(f"{label}\nr(linear) = {r_pop:+.3f}   n = {len(sub):,}", fontsize=11)
    ax.legend(fontsize=9)

fig.suptitle("Population LW Curves by Timing Proxy  ·  2025 MLB Fastball Swings  ·  "
             "Primary measure (intercept_y) shown first",
             fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'timing_comparison_2_pop_lw.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → timing_comparison_2_pop_lw.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Per-hitter polynomial R² — distribution + direct scatter
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Histograms
for ax, col, label, color in [
    (axes[0], 'r2_ty',  'intercept_y (in) [primary]', '#9467bd'),
    (axes[0], 'r2_ta',  'atan2(y,x)',                 '#1f77b4'),
    (axes[0], 'r2_ta2', 'atan2(x,y)',                  '#ff7f0e'),
    (axes[0], 'r2_ad',  'attack_direction',             '#2ca02c'),
    (axes[0], 'r2_combo','combo (intercept_y + attack_dir)', '#8c564b'),
]:
    vals = per_hitter[col].dropna()
    ax.hist(vals, bins=30, alpha=0.45, color=color, label=f"{label} (med={vals.median():.3f})")

axes[0].axvline(0, color='k', lw=1, ls='--', alpha=0.5)
axes[0].set_xlabel("Per-hitter polynomial R^2", fontsize=10)
axes[0].set_ylabel("Count", fontsize=10)
axes[0].set_title("Distribution of Per-Hitter R^2\n(degree-4 poly fit on timing proxy -> delta_run_exp)", fontsize=10)
axes[0].legend(fontsize=8)

# Scatter: primary (intercept_y) vs angle atan2(y,x) R^2
comp = per_hitter.dropna(subset=['r2_ty', 'r2_ta'])
ax = axes[1]
sc = ax.scatter(comp['r2_ty'], comp['r2_ta'],
                c=comp['n'], cmap='viridis', s=30, alpha=0.6, edgecolors='none')
fig.colorbar(sc, ax=ax, pad=0.02).set_label("n swings", fontsize=8)
mn = min(comp['r2_ty'].min(), comp['r2_ta'].min()) - 0.01
mx = max(comp['r2_ty'].max(), comp['r2_ta'].max()) + 0.01
ax.plot([mn, mx], [mn, mx], 'k--', lw=1, alpha=0.5, label='y = x')
diff = comp['r2_ty'] - comp['r2_ta']
n_better_ty   = (diff > 0).sum()
n_better_ang  = (diff < 0).sum()
ax.set_xlabel("R^2  intercept_y (in)", fontsize=10)
ax.set_ylabel("R^2  atan2(y,x) (deg)", fontsize=10)
ax.set_title(f"Per-hitter R^2: intercept_y vs atan2(y,x)\n"
             f"intercept_y better: {n_better_ty}  |  atan2 better: {n_better_ang}", fontsize=10)
ax.legend(fontsize=9)

# Scatter: best single vs combo
comp2 = per_hitter.dropna(subset=['r2_ty', 'r2_ta', 'r2_ad', 'r2_combo'])
best_single = comp2[['r2_ty','r2_ta','r2_ad']].max(axis=1)
ax = axes[2]
ax.scatter(best_single, comp2['r2_combo'], s=30, alpha=0.5, color='purple', edgecolors='none')
mn2 = min(best_single.min(), comp2['r2_combo'].min()) - 0.01
mx2 = max(best_single.max(), comp2['r2_combo'].max()) + 0.01
ax.plot([mn2, mx2], [mn2, mx2], 'k--', lw=1, alpha=0.5, label='y = x')
n_combo_better = (comp2['r2_combo'] > best_single).sum()
ax.set_xlabel("Best single-measure R^2", fontsize=10)
ax.set_ylabel("Combo (intercept_y + attack_dir) R^2", fontsize=10)
ax.set_title(f"Does combining measures help?\n"
             f"Combo beats best single: {n_combo_better}/{len(comp2)} hitters", fontsize=10)
ax.legend(fontsize=9)

fig.suptitle(f"Per-Hitter Timing Proxy Quality  ·  {len(per_hitter)} Qualifying Hitters  ·  Degree-4 Polynomial",
             fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'timing_comparison_3_per_hitter.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → timing_comparison_3_per_hitter.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Intercorrelation among primary timing measures
# ═══════════════════════════════════════════════════════════════════════════
comp_df = df.dropna(subset=['timing_y', 'timing_angle', 'attack_dir', 'delta_run_exp'])
r_ty_ang, _   = pearsonr(comp_df['timing_y'],     comp_df['timing_angle'])
r_ty_ad,  _   = pearsonr(comp_df['timing_y'],     comp_df['attack_dir'])
r_ang_ad, _   = pearsonr(comp_df['timing_angle'], comp_df['attack_dir'])

fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# Panel A: intercept_y vs atan2(y,x)
ax = axes[0]
x = comp_df['timing_y'].clip(*np.percentile(comp_df['timing_y'], [0.5, 99.5]))
y = comp_df['timing_angle'].clip(*np.percentile(comp_df['timing_angle'], [0.5, 99.5]))
hb = ax.hexbin(x, y, gridsize=60, cmap='Purples', mincnt=3, linewidths=0.1)
fig.colorbar(hb, ax=ax, pad=0.02).set_label("Count", fontsize=8)
ax.set_xlabel("intercept_y centred (in)", fontsize=10)
ax.set_ylabel("Timing Angle atan2(y,x) (deg)", fontsize=10)
ax.set_title(f"intercept_y vs atan2(y,x)\nr = {r_ty_ang:+.3f}   n = {len(comp_df):,}", fontsize=11)

# Panel B: intercept_y vs attack_dir
ax = axes[1]
x = comp_df['timing_y'].clip(*np.percentile(comp_df['timing_y'], [0.5, 99.5]))
y = comp_df['attack_dir'].clip(*np.percentile(comp_df['attack_dir'], [0.5, 99.5]))
hb = ax.hexbin(x, y, gridsize=60, cmap='Blues', mincnt=3, linewidths=0.1)
fig.colorbar(hb, ax=ax, pad=0.02).set_label("Count", fontsize=8)
ax.set_xlabel("intercept_y centred (in)", fontsize=10)
ax.set_ylabel("Attack Direction (deg)", fontsize=10)
ax.set_title(f"intercept_y vs attack_direction\nr = {r_ty_ad:+.3f}", fontsize=11)

# Panel C: residual of attack_dir after intercept_y vs LW
coeffs = np.polyfit(comp_df['timing_y'], comp_df['attack_dir'], 1)
resid = comp_df['attack_dir'] - np.polyval(coeffs, comp_df['timing_y'])
r_resid, p_resid = pearsonr(resid, comp_df['delta_run_exp'])

ax = axes[2]
x2 = resid.clip(*np.percentile(resid, [0.5, 99.5]))
y2 = comp_df['delta_run_exp']
hb2 = ax.hexbin(x2, y2, gridsize=50, cmap='YlOrRd', mincnt=5, linewidths=0.1)
fig.colorbar(hb2, ax=ax, pad=0.02).set_label("Count", fontsize=8)
edges = np.linspace(x2.quantile(0.005), x2.quantile(0.995), 41)
bins  = pd.cut(x2, bins=edges)
bm    = pd.DataFrame({'x': x2, 'y': y2, 'b': bins}).groupby('b', observed=True)
ax.plot(bm['x'].mean().values, bm['y'].mean().values, color='royalblue', lw=2.5, label='Binned mean')
ax.axvline(0, color='k', lw=1, ls='--', alpha=0.6)
ax.axhline(0, color='grey', lw=0.8, ls=':')
ax.set_xlabel("attack_direction residual after controlling for intercept_y", fontsize=9)
ax.set_ylabel("Delta Run Expectancy", fontsize=10)
ax.set_title(f"Unique variance in attack_direction -> run value\n"
             f"r(resid, delta_run_exp) = {r_resid:+.3f}  "
             f"{'<0.0001' if p_resid < 0.0001 else f'p={p_resid:.4f}'}", fontsize=10)
ax.legend(fontsize=9)

fig.suptitle("Intercorrelation Structure: intercept_y (primary) vs Angles vs Attack Direction\n"
             "Right panel: does attack_direction contain unique timing information beyond intercept_y?",
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'timing_comparison_4_intercorr.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → timing_comparison_4_intercorr.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: LW curves by handedness for primary measure vs alternates
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(22, 11))
for row, (hand, hand_label) in enumerate([('R','RHH'),('L','LHH')]):
    sub = df[df['stand']==hand].dropna(subset=['delta_run_exp'])
    for col_idx, (col, label, unit, color) in enumerate([
        ('timing_y',     'intercept_y (centred)',     'in',  '#9467bd'),
        ('timing_angle', 'Timing Angle atan2(y,x)',   'deg', '#1f77b4'),
        ('attack_dir',   'Attack Direction',          'deg', '#2ca02c'),
    ]):
        ax = axes[row, col_idx]
        vals = sub.dropna(subset=[col])
        x, y = vals[col].values, vals['delta_run_exp'].values
        xm, ym, nm = pop_lw(x, y)
        r_lin, _ = pearsonr(x, y)
        ax.bar(xm, ym, width=(xm[1]-xm[0])*0.8, color=color, alpha=0.3)
        ax.plot(xm, ym, color=color, lw=2.5, label='Binned mean')
        ax.axhline(0, color='grey', lw=0.8, ls=':')
        ax.axvline(xm[np.argmax(ym)], color='red', lw=1.2, ls='--', alpha=0.7,
                   label=f'Peak = {xm[np.argmax(ym)]:.1f} {unit}')
        ax.set_xlabel(f"{label} ({unit})", fontsize=10)
        ax.set_ylabel("Mean Delta Run Expectancy", fontsize=10)
        ax.set_title(f"{hand_label}  ·  {label}\n"
                     f"r(linear) = {r_lin:+.3f}   n = {len(vals):,}", fontsize=10)
        ax.legend(fontsize=9)

fig.suptitle("Population LW Curves by Handedness x Timing Proxy",
             fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'timing_comparison_5_by_hand.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → timing_comparison_5_by_hand.png")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n== Summary ==")
print(f"{'Measure':<40} {'Median per-hitter R^2':>22} {'Mean per-hitter R^2':>20}")
for col, label in [('r2_ty','intercept_y (in) [primary]'),
                   ('r2_ta','atan2(y,x)'), ('r2_ta2','atan2(x,y)'),
                   ('r2_ad','attack_direction'),
                   ('r2_combo','combo (intercept_y + attack_dir)')]:
    vals = per_hitter[col].dropna()
    print(f"  {label:<38} {vals.median():>22.4f} {vals.mean():>20.4f}")

print(f"\nMeasure intercorrelations:")
print(f"  r(intercept_y, atan2(y,x))   = {r_ty_ang:+.3f}")
print(f"  r(intercept_y, attack_dir)   = {r_ty_ad:+.3f}")
print(f"  r(atan2(y,x),  attack_dir)   = {r_ang_ad:+.3f}")
print(f"Unique attack_dir signal (after intercept_y): r(resid, delta_run_exp) = {r_resid:+.3f}")
print("\nDone.")
