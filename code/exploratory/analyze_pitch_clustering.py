"""
Pitch-type label vs (h_break, v_break, velocity) clustering.

Goal: decide whether to bin pitches by their `pitch_type` label or by an
unsupervised cluster in 3D physics space (horizontal break, vertical break,
release speed). The user wants a coarse categorisation that can identify
mislabeled pitches (\"this is labeled a curveball but clusters with sliders
because of its break and velocity\").

Handedness-normalised h-break: pfx_x is multiplied by sign(p_throws) so that
positive = arm-side break for both RHP and LHP. (A RHP slider breaks
glove-side; a LHP slider breaks the opposite raw direction but the same
arm/glove-side direction.)

Approach:
  1. Try k ∈ {8, 10, 12} kmeans clusters in standardised (pfx_x_arm, pfx_z, release_speed) space.
  2. For each k, build the pitch_type × cluster contingency table and a
     normalised cross-classification matrix.
  3. Visualise: pfx_x_arm vs pfx_z scatter coloured by pitch_type, then by cluster.
     Also velocity vs pfx_z by cluster.
  4. Find the largest "mislabelled" groups — pitches whose pitch_type is a
     small fraction of their cluster's content (and vice versa).
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "misc")

FEATS = ['pfx_x_arm', 'pfx_z', 'release_speed']
K_VALUES = [8, 10, 12]

# ── Load ──────────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet")))
if not files:
    raise SystemExit("No parquet files in data/all_pitches_2025/. Run "
                     "code/download_all_pitches_2025.py first.")
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"Total pitches loaded: {len(df):,}")

df = df.dropna(subset=['pitch_type', 'pfx_x', 'pfx_z', 'release_speed', 'p_throws'])
print(f"After dropna on essential cols: {len(df):,}")

# Handedness-normalised horizontal break (positive = arm-side)
df['pfx_x_arm'] = df['pfx_x'] * df['p_throws'].map({'R': 1, 'L': -1}).fillna(1)

# Keep pitch types with ≥1000 pitches; lump rare types
pt_counts = df['pitch_type'].value_counts()
print(f"\nPitch type counts:\n{pt_counts}")
common_pt = pt_counts[pt_counts >= 1000].index.tolist()
df['pitch_type_clean'] = df['pitch_type'].where(df['pitch_type'].isin(common_pt), 'OTHER')

# ── Cluster ──────────────────────────────────────────────────────────────────
X = df[FEATS].values
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)

cluster_results = {}
for k in K_VALUES:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    df[f'cluster_k{k}'] = km.labels_
    # Cluster centres in original units
    centers = scaler.inverse_transform(km.cluster_centers_)
    cluster_results[k] = {'model': km, 'centers': pd.DataFrame(centers, columns=FEATS)}
    # External agreement vs pitch_type
    ari = adjusted_rand_score(df['pitch_type_clean'], km.labels_)
    nmi = normalized_mutual_info_score(df['pitch_type_clean'], km.labels_)
    print(f"\nk={k}: ARI={ari:.3f}  NMI={nmi:.3f}")
    print(cluster_results[k]['centers'].round(2).to_string())

# Use k=10 as the primary clustering for plots & contingency
K_PRIM = 10
df['cluster'] = df[f'cluster_k{K_PRIM}']

# Order pitch types by mean velocity (slowest → fastest) for nicer matrices
pt_order = (df.groupby('pitch_type_clean')['release_speed']
              .mean().sort_values().index.tolist())
# Order clusters by their centre velocity
cl_order = (cluster_results[K_PRIM]['centers']
            .sort_values('release_speed').index.tolist())

# ── Contingency matrix ───────────────────────────────────────────────────────
cont = (pd.crosstab(df['pitch_type_clean'], df['cluster'])
          .loc[pt_order, cl_order])
print("\n══ Contingency: pitch_type × cluster (raw counts) ══")
print(cont.to_string())

# Row-normalised (% of each pitch type per cluster)
cont_row = cont.div(cont.sum(axis=1), axis=0) * 100
# Col-normalised (% of each cluster per pitch type)
cont_col = cont.div(cont.sum(axis=0), axis=1) * 100

# Find "mislabelled" pitches: pitches whose pitch_type is < 20% of their cluster
def mislabel_share(row):
    return cont_col.loc[row['pitch_type_clean'], row['cluster']]
print("\nPicking mislabel candidates (slowest)…")
df['cluster_share_of_my_type'] = df.apply(mislabel_share, axis=1)

# ── Plot 1: Scatter pfx_x_arm vs pfx_z, two colourings ───────────────────────
sample = df.sample(min(40000, len(df)), random_state=0)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Panel A: coloured by pitch_type
ax = axes[0]
palette_pt = sns.color_palette('tab20', n_colors=len(pt_order))
pt_color = dict(zip(pt_order, palette_pt))
for pt in pt_order:
    s = sample[sample['pitch_type_clean'] == pt]
    if len(s) == 0: continue
    ax.scatter(s['pfx_x_arm'], s['pfx_z'], c=[pt_color[pt]], s=3, alpha=0.4, label=pt)
ax.set_xlabel("Horizontal break, arm-side normalised (pfx_x × sign(p_throws), ft)", fontsize=10)
ax.set_ylabel("Vertical break (pfx_z, ft)", fontsize=10)
ax.set_title(f"Coloured by pitch_type label  (n={len(sample):,})", fontsize=11)
ax.legend(fontsize=7, loc='upper left', ncol=2, markerscale=3)
ax.axhline(0, color='grey', lw=0.5, ls=':')
ax.axvline(0, color='grey', lw=0.5, ls=':')

# Panel B: coloured by cluster
ax = axes[1]
palette_cl = sns.color_palette('tab10', n_colors=K_PRIM)
cl_color = dict(zip(cl_order, palette_cl))
for cl in cl_order:
    s = sample[sample['cluster'] == cl]
    if len(s) == 0: continue
    ax.scatter(s['pfx_x_arm'], s['pfx_z'], c=[cl_color[cl]], s=3, alpha=0.4,
               label=f"C{cl}  (n={(df['cluster']==cl).sum():,})")
ax.set_xlabel("Horizontal break, arm-side normalised (ft)", fontsize=10)
ax.set_ylabel("Vertical break (ft)", fontsize=10)
ax.set_title(f"Coloured by KMeans cluster (k={K_PRIM})  (n={len(sample):,})", fontsize=11)
ax.legend(fontsize=7, loc='upper left', ncol=2, markerscale=3)
# Overlay cluster centres
for cl in cl_order:
    c = cluster_results[K_PRIM]['centers'].iloc[cl]
    ax.scatter(c['pfx_x_arm'], c['pfx_z'], c='black', marker='X', s=120, edgecolors='white', linewidths=1.5)
    ax.text(c['pfx_x_arm'], c['pfx_z']+0.05, f"C{cl}  {c['release_speed']:.0f}mph",
            fontsize=8, ha='center', va='bottom', fontweight='bold')
ax.axhline(0, color='grey', lw=0.5, ls=':')
ax.axvline(0, color='grey', lw=0.5, ls=':')

fig.suptitle("Pitch shape space — label vs unsupervised cluster", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'pitch_cluster_1_break_space.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → pitch_cluster_1_break_space.png")

# ── Plot 2: velocity vs vertical break by cluster ────────────────────────────
fig, ax = plt.subplots(figsize=(11, 8))
for cl in cl_order:
    s = sample[sample['cluster'] == cl]
    ax.scatter(s['release_speed'], s['pfx_z'], c=[cl_color[cl]], s=3, alpha=0.4,
               label=f"C{cl}")
for cl in cl_order:
    c = cluster_results[K_PRIM]['centers'].iloc[cl]
    ax.scatter(c['release_speed'], c['pfx_z'], c='black', marker='X', s=120,
               edgecolors='white', linewidths=1.5, zorder=5)
    ax.text(c['release_speed'], c['pfx_z']+0.04, f"C{cl}", fontsize=9, ha='center')
ax.set_xlabel("Release speed (mph)", fontsize=10)
ax.set_ylabel("Vertical break pfx_z (ft)", fontsize=10)
ax.set_title(f"Cluster structure in velocity × vertical-break space (k={K_PRIM})", fontsize=11)
ax.legend(fontsize=8, markerscale=3, loc='upper left')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'pitch_cluster_2_velo_vbreak.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → pitch_cluster_2_velo_vbreak.png")

# ── Plot 3: contingency heatmap (% of each pitch type) ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 0.45 * len(pt_order) + 2))
sns.heatmap(cont_row, ax=axes[0], annot=True, fmt='.0f', cmap='YlOrRd', cbar_kws={'label':'%'},
            xticklabels=[f'C{c}' for c in cl_order], yticklabels=pt_order)
axes[0].set_title(f"Row %: of all pitches labelled X, how many fell in cluster Y?\n"
                  f"(rows sum to 100%)", fontsize=11)
axes[0].set_xlabel("Cluster")
axes[0].set_ylabel("pitch_type")

sns.heatmap(cont_col, ax=axes[1], annot=True, fmt='.0f', cmap='YlGnBu', cbar_kws={'label':'%'},
            xticklabels=[f'C{c}' for c in cl_order], yticklabels=pt_order)
axes[1].set_title(f"Column %: of all pitches in cluster Y, how many were labelled X?\n"
                  f"(columns sum to 100%)", fontsize=11)
axes[1].set_xlabel("Cluster")
axes[1].set_ylabel("pitch_type")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'pitch_cluster_3_contingency.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → pitch_cluster_3_contingency.png")

# ── Plot 4: ARI / NMI vs k ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ks = list(K_VALUES)
aris = []
nmis = []
for k in ks:
    ari = adjusted_rand_score(df['pitch_type_clean'], df[f'cluster_k{k}'])
    nmi = normalized_mutual_info_score(df['pitch_type_clean'], df[f'cluster_k{k}'])
    aris.append(ari); nmis.append(nmi)
ax.plot(ks, aris, 'o-', label='ARI (adjusted Rand)', color='#1f77b4')
ax.plot(ks, nmis, 's-', label='NMI (normalised MI)', color='#d62728')
ax.set_xlabel('k (number of clusters)')
ax.set_ylabel('Score (1 = perfect agreement with pitch_type)')
ax.set_title('Agreement between unsupervised clusters and Statcast pitch_type label')
ax.legend()
ax.set_xticks(ks)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'pitch_cluster_4_ari_nmi.png'), dpi=160, bbox_inches='tight')
plt.close(fig)
print("Saved → pitch_cluster_4_ari_nmi.png")

# ── Examples of "mislabelled" pitches ────────────────────────────────────────
# Find clusters where a minority pitch_type contributes ≥10% of the cluster
print("\n══ \"Mislabelled\" examples (largest minority pitch_type in each cluster) ══")
for cl in cl_order:
    col = cont_col[cl].sort_values(ascending=False)
    primary = col.index[0]
    minorities = col[col.index != primary]
    minor_pt   = minorities.idxmax() if len(minorities) else None
    minor_pct  = minorities.max() if len(minorities) else 0.0
    c = cluster_results[K_PRIM]['centers'].iloc[cl]
    print(f"  C{cl} (h={c['pfx_x_arm']:+.2f} ft, v={c['pfx_z']:+.2f} ft, "
          f"{c['release_speed']:.1f} mph): primary={primary} ({col[primary]:.0f}%) "
          f"largest minority={minor_pt} ({minor_pct:.0f}%)")

print("\nDone.")
