"""
Compare swing and pitch attributes at the two population LW peaks among
in-play balls with positive delta_run_exp.

Positive peak ~ +12.7 in (pull/early timing)
Negative peak ~ -12.0 in (oppo/late timing trough)

Three outputs:
  1. Profile plots: mean of each variable across all timing bins (positive ERV in-play only),
     with both peak bins highlighted - shows which variables co-vary with timing.

  2. Ranked difference bar chart: standardised (peak+ - peak-) for every variable,
     showing which attributes most strongly separate high-value early-timing contact
     from high-value late-timing contact.

  3. LW-curve stratification: for the top candidate "break-out" variables, split the
     FULL in-play dataset at the median and plot LW-vs-timing curves for each half.
     If curves diverge (different peak locations / shapes) the variable should be used
     to condition the timing analysis; if they are parallel it is safe to pool.
"""

import os, sys, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, ttest_ind

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from timing_utils import (ALL_SWINGS, CONTACT, IN_PLAY, INTERCEPT_X, INTERCEPT_Y,
                          add_timing, smoothed_peak, binned_lw, TIMING_AXIS_LABEL, FASTBALL_TYPES, INSIDE_Z, MIDDLE_Z, OUTSIDE_Z, add_zone_and_matchup)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "deprecated")

COLS = [
    "pitch_type", "batter", "stand", "description",
    "delta_run_exp",
    INTERCEPT_X,
    INTERCEPT_Y,
    # swing attributes
    "swing_path_tilt", "attack_angle", "attack_direction",
    "bat_speed", "swing_length",
    # launch outcomes
    "launch_angle", "launch_speed", "launch_speed_angle",
    "estimated_woba_using_speedangle",
    # pitch attributes
    "release_speed", "release_spin_rate",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z",
    "zone",
]

PEAK_POS =  12.7   # pull/early peak (in, post-centering)
PEAK_NEG = -12.0   # oppo/late trough (in, post-centering)
N_BINS   = 40


# ── Load all in-play fastballs ────────────────────────────────────────────────
# User-specified: positive-ERV in-play balls. Keep the IN_PLAY filter.
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
df = pd.concat([pd.read_csv(f, usecols=COLS) for f in files], ignore_index=True)
df = df[df['pitch_type'].isin(FASTBALL_TYPES) & df['description'].isin(IN_PLAY)].copy()
df = df.dropna(subset=[INTERCEPT_Y, 'delta_run_exp'])
add_timing(df)

df['zone_grp_num'] = df['zone'].map(
    lambda z: 0 if z in INSIDE_Z else (1 if z in MIDDLE_Z else (2 if z in OUTSIDE_Z else np.nan))
)

print(f"In-play fastballs: {len(df):,}")
print(f"  timing range: [{df['timing'].min():.1f}, {df['timing'].max():.1f}] in  "
      f"median={df['timing'].median():.2f} in")

# ── Bin all in-play by timing (for the stratification LW curves later) ────────
bin_edges = np.linspace(df['timing'].quantile(0.005),
                        df['timing'].quantile(0.995), N_BINS + 1)
df['timing_bin']     = pd.cut(df['timing'], bins=bin_edges)
bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# ── Positive-ERV subset for peak comparison ───────────────────────────────────
pos = df[df['delta_run_exp'] > 0].copy()
print(f"\nPositive-ERV in-play fastballs: {len(pos):,} "
      f"({len(pos)/len(df):.1%} of in-play)")

# Find which timing bins contain the two peak angles
def find_bin_idx(target, mids):
    return int(np.argmin(np.abs(mids - target)))

idx_pos = find_bin_idx(PEAK_POS, bin_mids)
idx_neg = find_bin_idx(PEAK_NEG, bin_mids)
print(f"\nPositive peak bin: mid={bin_mids[idx_pos]:.1f} in  "
      f"(target {PEAK_POS:+.1f} in)")
print(f"Negative peak bin: mid={bin_mids[idx_neg]:.1f} in  "
      f"(target {PEAK_NEG:+.1f} in)")

# Assign bin index to positive-ERV subset
pos['bin_idx'] = pos['timing_bin'].apply(
    lambda b: np.searchsorted(bin_edges[1:], b.mid) if hasattr(b, 'mid') else np.nan
).astype(float)

# ── Variable definitions ──────────────────────────────────────────────────────
SWING_VARS = [
    ('swing_path_tilt',              'Swing Path Tilt (deg)'),
    ('attack_angle',                 'Attack Angle (deg)'),
    ('bat_speed',                    'Bat Speed (mph)'),
    ('swing_length',                 'Swing Length (ft)'),
    ('attack_direction',             'Attack Direction (deg)'),
]
PITCH_VARS = [
    ('release_speed',                'Release Speed (mph)'),
    ('release_spin_rate',            'Spin Rate (rpm)'),
    ('pfx_x',                        'Horizontal Break pfx_x (in)'),
    ('pfx_z',                        'Vertical Break pfx_z (in)'),
    ('plate_x',                      'Plate Position X (ft)'),
    ('plate_z',                      'Plate Position Z (ft)'),
    ('zone_grp_num',                 'Zone (0=in, 1=mid, 2=out)'),
]
LAUNCH_VARS = [
    ('launch_angle',                 'Launch Angle (deg)'),
    ('launch_speed',                 'Exit Velocity (mph)'),
    ('launch_speed_angle',           'Launch Speed-Angle (barrel quality 1-6)'),
    ('estimated_woba_using_speedangle', 'xwOBA'),
]
ALL_VARS = SWING_VARS + PITCH_VARS + LAUNCH_VARS

# ── Compute per-bin means (positive ERV only) ─────────────────────────────────
bin_stats = {}
for col, _ in ALL_VARS:
    grp = pos.dropna(subset=[col]).groupby('timing_bin', observed=True)[col].mean()
    bin_stats[col] = grp

# Check n per bin for the two peaks
n_pos_peak = pos[pos['bin_idx'] == idx_pos].shape[0]
n_neg_peak = pos[pos['bin_idx'] == idx_neg].shape[0]
print(f"\nn in positive-peak bin ({bin_mids[idx_pos]:+.1f} in): {n_pos_peak:,}")
print(f"n in negative-peak bin ({bin_mids[idx_neg]:+.1f} in): {n_neg_peak:,}")

# ── Two-bin comparison: means and t-tests ────────────────────────────────────
peak_pos_data = pos[pos['bin_idx'] == idx_pos]
peak_neg_data = pos[pos['bin_idx'] == idx_neg]

print(f"\n{'Variable':<42} {'Peak+':>9} {'Peak-':>9} {'Delta':>9} {'p':>8}")
print("─" * 82)
diffs = []
for col, label in ALL_VARS:
    v_pos = peak_pos_data[col].dropna().values
    v_neg = peak_neg_data[col].dropna().values
    if len(v_pos) < 10 or len(v_neg) < 10:
        continue
    mu_pos, mu_neg = v_pos.mean(), v_neg.mean()
    _, p = ttest_ind(v_pos, v_neg, equal_var=False)
    pooled_std = np.sqrt((v_pos.std()**2 + v_neg.std()**2) / 2)
    cohen_d = (mu_pos - mu_neg) / pooled_std if pooled_std > 0 else 0
    pstr = '<.0001' if p < 0.0001 else f'{p:.4f}'
    print(f"  {label:<40} {mu_pos:>9.3f} {mu_neg:>9.3f} {mu_pos-mu_neg:>+9.3f} {pstr:>8}")
    diffs.append({'col': col, 'label': label, 'delta': mu_pos - mu_neg,
                  'cohen_d': cohen_d, 'p': p,
                  'mu_pos': mu_pos, 'mu_neg': mu_neg,
                  'std_pos': v_pos.std(), 'std_neg': v_neg.std()})

diffs_df = pd.DataFrame(diffs).sort_values('cohen_d', key=abs, ascending=False)

# ── Pearson r: each variable vs timing angle (positive ERV only) ──────────────
print(f"\n{'Variable':<42} {'r(var, timing)':>16}  {'p':>8}")
print("─" * 70)
r_timing = []
for col, label in ALL_VARS:
    sub = pos.dropna(subset=[col, 'timing'])
    if len(sub) < 100:
        continue
    r, p = pearsonr(sub['timing'], sub[col])
    pstr = '<.0001' if p < 0.0001 else f'{p:.4f}'
    print(f"  {label:<40} {r:>+16.3f}  {pstr:>8}")
    r_timing.append({'col': col, 'label': label, 'r': r, 'p': p})
r_df = pd.DataFrame(r_timing).sort_values('r', key=abs, ascending=False)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Profile plots — mean variable value across timing bins (pos ERV)
# ════════════════════════════════════════════════════════════════════════════
ncols = 4
nrows = int(np.ceil(len(ALL_VARS) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 3.8))
axes = axes.flatten()

for i, (col, label) in enumerate(ALL_VARS):
    ax = axes[i]
    grp = pos.dropna(subset=[col]).groupby('timing_bin', observed=True)
    mids = [b.mid for b in grp[col].mean().index]
    means = grp[col].mean().values

    ax.plot(mids, means, color='steelblue', lw=2)
    ax.axvline(0, color='grey', lw=0.8, ls=':', alpha=0.6)

    # Highlight the two peak bins
    ax.axvline(bin_mids[idx_pos], color='#d62728', lw=1.8, ls='--', alpha=0.85,
               label=f'Pull peak ({bin_mids[idx_pos]:+.1f} in)')
    ax.axvline(bin_mids[idx_neg], color='#1f77b4', lw=1.8, ls='--', alpha=0.85,
               label=f'Oppo peak ({bin_mids[idx_neg]:+.1f} in)')

    # Mark mean values at the two peaks
    val_p = peak_pos_data[col].mean()
    val_n = peak_neg_data[col].mean()
    if not np.isnan(val_p):
        ax.scatter([bin_mids[idx_pos]], [val_p], c='#d62728', s=60, zorder=5)
    if not np.isnan(val_n):
        ax.scatter([bin_mids[idx_neg]], [val_n], c='#1f77b4', s=60, zorder=5)

    _sub = pos.dropna(subset=[col, 'timing'])
    r, p = pearsonr(_sub['timing'], _sub[col]) if len(_sub) > 30 else (np.nan, np.nan)

    ax.set_xlabel("Timing (in)", fontsize=8)
    ax.set_title(f"{label}\nr(timing) = {r:+.3f}", fontsize=8)
    if i == 0:
        ax.legend(fontsize=6)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Mean Variable Value vs Timing  ·  Positive-ERV In-Play Fastballs Only\n"
             "Red dashed = pull/early peak  ·  Blue dashed = oppo/late peak",
             fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'peak_comparison_1_profiles.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("\nSaved → peak_comparison_1_profiles.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Ranked bar chart — standardised difference at the two peaks
# ════════════════════════════════════════════════════════════════════════════
diffs_sorted = diffs_df.sort_values('cohen_d', ascending=True)
colors = ['#d62728' if d > 0 else '#1f77b4' for d in diffs_sorted['cohen_d']]

fig, ax = plt.subplots(figsize=(10, max(6, len(diffs_sorted) * 0.45)))
bars = ax.barh(range(len(diffs_sorted)), diffs_sorted['cohen_d'], color=colors, alpha=0.75,
               edgecolor='white', linewidth=0.5)
ax.axvline(0, color='black', lw=1.2)
ax.set_yticks(range(len(diffs_sorted)))
ax.set_yticklabels(diffs_sorted['label'], fontsize=9)
ax.set_xlabel("Cohen's d  (positive = higher at pull/early peak)", fontsize=10)
ax.set_title(f"What Separates High-Value Contact at Pull Peak ({bin_mids[idx_pos]:+.1f} in) "
             f"vs Oppo Peak ({bin_mids[idx_neg]:+.1f} in)?\n"
             f"Positive-ERV in-play fastballs only  ·  "
             f"n = {n_pos_peak:,} (pull) vs {n_neg_peak:,} (oppo)", fontsize=10)

# Add significance markers
for j, (_, row) in enumerate(diffs_sorted.iterrows()):
    sig = '***' if row['p'] < 0.001 else ('**' if row['p'] < 0.01 else ('*' if row['p'] < 0.05 else ''))
    if sig:
        x_off = row['cohen_d'] + (0.02 if row['cohen_d'] >= 0 else -0.02)
        ax.text(x_off, j, sig, va='center', ha='left' if row['cohen_d'] >= 0 else 'right',
                fontsize=8, color='black')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'peak_comparison_2_ranked.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → peak_comparison_2_ranked.png")

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3: LW-curve stratification — split by top candidate variables
# Tests whether a variable modifies the timing → run value relationship
# ════════════════════════════════════════════════════════════════════════════
# Pick top swing/pitch variables by |r(timing, variable)| from positive-ERV data,
# but evaluate stratification on the FULL in-play dataset
top_candidates = (r_df[r_df['col'].isin([c for c, _ in SWING_VARS + PITCH_VARS])]
                  .sort_values('r', key=abs, ascending=False)
                  .head(6))

print(f"\nTop stratification candidates (by |r| with timing, positive ERV):")
print(top_candidates[['label','r','p']].to_string(index=False))

def lw_curve(data, n=N_BINS):
    sub = data.dropna(subset=['timing', 'delta_run_exp'])
    edges = np.linspace(sub['timing'].quantile(0.005), sub['timing'].quantile(0.995), n + 1)
    bins  = pd.cut(sub['timing'], bins=edges)
    agg   = pd.DataFrame({'t': sub['timing'], 'lw': sub['delta_run_exp'], 'b': bins})
    g     = agg.groupby('b', observed=True)
    return g['t'].mean().values, g['lw'].mean().values

ncols2 = 3
nrows2 = int(np.ceil(len(top_candidates) / ncols2))
fig, axes = plt.subplots(nrows2, ncols2, figsize=(21, nrows2 * 5))
axes = axes.flatten()

for i, (_, row) in enumerate(top_candidates.iterrows()):
    col, label = row['col'], row['label']
    ax = axes[i]

    sub = df.dropna(subset=[col])
    median_val = sub[col].median()
    lo  = sub[sub[col] <= median_val]
    hi  = sub[sub[col] >  median_val]

    t_lo, lw_lo = lw_curve(lo)
    t_hi, lw_hi = lw_curve(hi)

    # find peaks
    def safe_peak(t, lw):
        mask = np.isfinite(lw)
        if mask.sum() == 0:
            return np.nan
        return t[mask][np.argmax(lw[mask])]

    pk_lo = safe_peak(t_lo, lw_lo)
    pk_hi = safe_peak(t_hi, lw_hi)

    ax.plot(t_lo, lw_lo, color='#1f77b4', lw=2.5,
            label=f'Low {label.split("(")[0].strip()} (<={median_val:.1f})  peak={pk_lo:+.1f} in')
    ax.plot(t_hi, lw_hi, color='#d62728', lw=2.5,
            label=f'High {label.split("(")[0].strip()} (>{median_val:.1f})  peak={pk_hi:+.1f} in')

    ax.axhline(0, color='grey', lw=0.8, ls=':')
    ax.axvline(0, color='grey', lw=0.8, ls='--', alpha=0.5)
    if not np.isnan(pk_lo):
        ax.axvline(pk_lo, color='#1f77b4', lw=1, ls=':', alpha=0.6)
    if not np.isnan(pk_hi):
        ax.axvline(pk_hi, color='#d62728', lw=1, ls=':', alpha=0.6)

    ax.set_xlabel(TIMING_AXIS_LABEL, fontsize=9)
    ax.set_ylabel("Mean Delta Run Expectancy", fontsize=9)
    ax.set_title(f"LW by {label}\nPeak shift: {pk_hi - pk_lo if not (np.isnan(pk_lo) or np.isnan(pk_hi)) else 'n/a':+.1f} in",
                 fontsize=10)
    ax.legend(fontsize=8)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Does This Variable Modify the Timing → Run Value Relationship?\n"
             "Median-split on each variable  ·  All in-play fastballs  ·  "
             "Large peak shift → variable should be used to condition the timing analysis",
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'peak_comparison_3_stratification.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → peak_comparison_3_stratification.png")

print("\nDone.")
