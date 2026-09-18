"""
Replicate the timing analyses (handedness × matchup, pitch zone, intra-cluster
speed quartiles) for each of the 10 pitch clusters from
analyze_pitch_clustering.py.

Timing is centred on the GLOBAL median across all 2025 swings, so cluster-level
peak timings are comparable on the same axis.

Output figures:
  pitch_cluster_timing_zones.png       — 2×5 grid, LW by inside/middle/outside
  pitch_cluster_timing_handedness.png  — 2×5 grid, LW by batter × pitcher hand
  pitch_cluster_timing_speed.png       — 2×5 grid, intra-cluster speed quartiles
  pitch_cluster_timing_summary.png     — peak timing per cluster + velocity scatter
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timing_utils import (ALL_SWINGS, INTERCEPT_Y, INSIDE_Z, MIDDLE_Z, OUTSIDE_Z,
                          load_pitches, add_timing, add_zone_and_matchup,
                          plot_lw_curves, peak_from_moving_average,
                          sort_by_velocity, TIMING_AXIS_LABEL_SHORT)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "misc")
K        = 10

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_pitches(DATA_DIR, descriptions=ALL_SWINGS)
print(f"Total swings loaded: {len(df):,}")

df = df.dropna(subset=['pitch_type', 'pfx_x', 'pfx_z', 'release_speed',
                        'p_throws', 'stand', INTERCEPT_Y, 'delta_run_exp', 'zone'])
print(f"After dropna on essentials: {len(df):,}")

GLOBAL_CENTER = add_timing(df)
print(f"Global timing centre: {GLOBAL_CENTER:.2f} in")

add_zone_and_matchup(df)
df['pfx_x_arm'] = df['pfx_x'] * df['p_throws'].map({'R': 1, 'L': -1}).fillna(1)

# ── Cluster ──────────────────────────────────────────────────────────────────
FEATS = ['pfx_x_arm', 'pfx_z', 'release_speed']
scaler = StandardScaler().fit(df[FEATS].values)
Xs = scaler.transform(df[FEATS].values)
km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(Xs)
df['cluster'] = km.labels_
centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=FEATS)
print("\nCluster centres:")
print(centers.round(2).to_string())

cl_order = sort_by_velocity(centers, 'release_speed')

# Human-readable cluster labels (primary pitch_type + velocity)
common_pt = df['pitch_type'].value_counts()[lambda s: s >= 1000].index
df['pitch_type_clean'] = df['pitch_type'].where(df['pitch_type'].isin(common_pt), 'OTHER')
cluster_labels = {}
for cl in cl_order:
    top = df[df['cluster'] == cl]['pitch_type_clean'].value_counts().head(2)
    primary = top.index[0]
    secondary = (f" + {top.index[1]}"
                 if len(top) > 1 and top.iloc[1] / df[df['cluster'] == cl].shape[0] > 0.15
                 else "")
    c = centers.iloc[cl]
    cluster_labels[cl] = f"C{cl}: {primary}{secondary} ({c['release_speed']:.0f} mph)"

print("\nCluster labels:")
for cl in cl_order:
    print(f"  {cluster_labels[cl]}  n={int((df['cluster']==cl).sum()):,}")

# ── Cluster-grid helper using the unified plot_lw_curves ─────────────────────
def make_cluster_grid(color_map_fn, fig_path, suptitle):
    fig, axes = plt.subplots(2, 5, figsize=(24, 11), sharex=True, sharey=False)
    axes = axes.flatten()
    for i, cl in enumerate(cl_order):
        ax = axes[i]
        sub = df[df['cluster'] == cl]
        subsets = color_map_fn(sub)
        if not subsets:
            ax.set_axis_off()
            continue
        plot_lw_curves(ax, subsets, min_per_bin=15, min_window_total=80,
                       min_n_data=200)
        ax.set_title(cluster_labels[cl], fontsize=9)
        ax.legend(fontsize=7, loc='lower left')
        if i % 5 == 0:
            ax.set_ylabel("Mean Δ Run Exp", fontsize=8)
        if i >= 5:
            ax.set_xlabel("Timing (in)\n← oppo/late      pull/early →", fontsize=8)
    fig.suptitle(suptitle, fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved → {os.path.basename(fig_path)}")

# Lens 1 — zone
make_cluster_grid(
    lambda sub: {
        'inside':  (sub[sub['zone_grp']=='inside'],  '#d62728', '-'),
        'middle':  (sub[sub['zone_grp']=='middle'],  '#2ca02c', '-'),
        'outside': (sub[sub['zone_grp']=='outside'], '#1f77b4', '-'),
    },
    os.path.join(OUT_DIR, 'pitch_cluster_timing_zones.png'),
    "LW vs Timing by Pitch Zone — within each pitch cluster",
)

# Lens 2 — handedness × matchup
make_cluster_grid(
    lambda sub: {
        'LHH vs LHP': (sub[(sub['stand']=='L') & (sub['p_throws']=='L')], '#1f77b4', '-'),
        'LHH vs RHP': (sub[(sub['stand']=='L') & (sub['p_throws']=='R')], '#1f77b4', '--'),
        'RHH vs RHP': (sub[(sub['stand']=='R') & (sub['p_throws']=='R')], '#d62728', '-'),
        'RHH vs LHP': (sub[(sub['stand']=='R') & (sub['p_throws']=='L')], '#d62728', '--'),
    },
    os.path.join(OUT_DIR, 'pitch_cluster_timing_handedness.png'),
    "LW vs Timing by Batter Handedness × Pitcher Hand — within each pitch cluster",
)

# Lens 3 — intra-cluster speed quartiles
def speed_quartiles_map(sub):
    if len(sub) < 200:
        return {}
    q = sub['release_speed'].quantile([0.25, 0.5, 0.75]).values
    return {
        f"Q1 ≤{q[0]:.0f}": (sub[sub['release_speed'] <  q[0]], '#0571b0', '-'),
        f"Q2 ≤{q[1]:.0f}": (sub[(sub['release_speed'] >= q[0]) & (sub['release_speed'] < q[1])], '#92c5de', '-'),
        f"Q3 ≤{q[2]:.0f}": (sub[(sub['release_speed'] >= q[1]) & (sub['release_speed'] < q[2])], '#f4a582', '-'),
        f"Q4 >{q[2]:.0f}": (sub[sub['release_speed'] >= q[2]], '#ca0020', '-'),
    }

make_cluster_grid(
    speed_quartiles_map,
    os.path.join(OUT_DIR, 'pitch_cluster_timing_speed.png'),
    "LW vs Timing by intra-cluster speed quartile — within each pitch cluster\n"
    "(Speed is one of the clustering features, so within-cluster spread is narrow.)",
)

# ── Summary figure: peak timing per cluster ──────────────────────────────────
print("\nComputing cluster-level peak timings (from count-weighted moving avg)…")
peaks = []
for cl in cl_order:
    sub = df[df['cluster'] == cl]
    peak_loc, peak_val = peak_from_moving_average(
        sub['timing'].values, sub['delta_run_exp'].values,
        n_bins=32, ma_window=5, min_per_bin=30, min_window_total=200)
    peaks.append((cl, peak_loc, peak_val, len(sub)))
peak_df = pd.DataFrame(peaks, columns=['cluster','peak_loc','peak_val','n'])
peak_df['label'] = peak_df['cluster'].map(cluster_labels)
peak_df['velocity'] = peak_df['cluster'].map(lambda c: centers.iloc[c]['release_speed'])
peak_df = peak_df.sort_values('velocity')
print(peak_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(20, 7))

# Panel A: peak timing by cluster, ordered by velocity
ax = axes[0]
ax.barh(range(len(peak_df)), peak_df['peak_loc'],
        color=['#d62728' if v >= 90 else '#ff7f0e' if v >= 83 else '#1f77b4'
               for v in peak_df['velocity']], alpha=0.75, edgecolor='white')
ax.set_yticks(range(len(peak_df)))
ax.set_yticklabels(peak_df['label'], fontsize=8)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel("Population peak timing (in, centred on global median)\n"
              "← later peak     earlier peak →", fontsize=10)
ax.set_title("Population peak timing per cluster\n"
             "(red = fastballs, orange = mid-velocity, blue = breaking/off-speed)",
             fontsize=11)
for i, row in enumerate(peak_df.itertuples()):
    if np.isfinite(row.peak_loc):
        ax.text(row.peak_loc + (0.3 if row.peak_loc >= 0 else -0.3), i,
                f"{row.peak_loc:+.1f} in",
                va='center', ha='left' if row.peak_loc >= 0 else 'right',
                fontsize=8)

# Panel B: peak timing vs cluster velocity (scatter)
ax = axes[1]
plot_df = peak_df.dropna(subset=['peak_loc'])
sc = ax.scatter(plot_df['velocity'], plot_df['peak_loc'],
                s=plot_df['n']/plot_df['n'].max()*400+50,
                c=plot_df['velocity'], cmap='RdYlBu_r', edgecolors='k', linewidths=0.5)
for _, row in plot_df.iterrows():
    ax.annotate(row['label'].split(':')[0], (row['velocity'], row['peak_loc']),
                fontsize=8, xytext=(4, 4), textcoords='offset points')
ax.axhline(0, color='grey', lw=0.6, ls=':')
ax.set_xlabel("Cluster mean velocity (mph)", fontsize=10)
ax.set_ylabel("Population peak timing (in)", fontsize=10)
ax.set_title("Does faster velocity push the optimal timing later?\n"
             "Point size ∝ n pitches per cluster", fontsize=11)
plt.colorbar(sc, ax=ax, label='velocity (mph)', pad=0.02)

fig.suptitle("Cluster-Level Population Peak Timing — All 2025 Pitches by KMeans Cluster",
             fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'pitch_cluster_timing_summary.png'),
            dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → pitch_cluster_timing_summary.png")

print("\nDone.")
