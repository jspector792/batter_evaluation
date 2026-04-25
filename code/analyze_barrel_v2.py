"""
Barrel Placement Follow-up v2 — 2025 MLB Fastballs

Q1: How much of |angle_diff| signal is already in launch_angle alone?
Q2: Swap attack_angle for swing_path_tilt → does the hex plot change?
Q3: Does swing_path_tilt add explanatory power to the barrel metric?
General: What combination of swing-mechanics variables best quantifies
         barrel placement, and how should we think about it as a batter skill?

Note on tilt_diff = swing_path_tilt - launch_angle:
  swing_path_tilt is measured 40 ms BEFORE contact (pre-outcome intent proxy),
  making tilt_diff a potentially better "barrel placement" signal than
  attack_angle - launch_angle, which is measured AT contact after ball-bat
  interaction has already begun.
"""
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

FASTBALL_TYPES = {"FF", "SI", "FC", "FT"}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out")

COLS = [
    "pitch_type", "batter",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "attack_angle", "swing_path_tilt",
    "bat_speed", "launch_angle", "launch_speed", "launch_speed_angle",
    "delta_run_exp",
]

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
chunks = [pd.read_csv(f, usecols=COLS) for f in files]
df = pd.concat(chunks, ignore_index=True)
df = df[df['pitch_type'].isin(FASTBALL_TYPES)].copy()
print(f"Fastball pitches: {len(df):,}")

IX = 'intercept_ball_minus_batter_pos_x_inches'
IY = 'intercept_ball_minus_batter_pos_y_inches'

df['timing_angle'] = np.degrees(np.arctan2(df[IX], df[IY]))
df['timing_angle'] -= df['timing_angle'].median()

bs_counts = df.groupby('batter')['bat_speed'].count()
valid_b = bs_counts[bs_counts >= 5].index
bs_max = df[df['batter'].isin(valid_b)].groupby('batter')['bat_speed'].max()
df = df.join(bs_max.rename('batter_max_bs'), on='batter')
df['bat_speed_rel'] = df['bat_speed'] / df['batter_max_bs']

# angle_diff: attack angle vs outcome (at contact)
df['angle_diff']     = df['attack_angle'] - df['launch_angle']
df['angle_diff_abs'] = df['angle_diff'].abs()

# tilt_diff: PRE-contact path vs outcome (better intent proxy)
df['tilt_diff']      = df['swing_path_tilt'] - df['launch_angle']
df['tilt_diff_abs']  = df['tilt_diff'].abs()

# tilt-to-attack gap: how much did the bat angle change during contact?
df['tilt_attack_gap'] = df['swing_path_tilt'] - df['attack_angle']

contact = df.dropna(subset=[IX, IY, 'timing_angle', 'delta_run_exp',
                              'launch_angle', 'launch_speed',
                              'bat_speed_rel', 'angle_diff', 'tilt_diff']).copy()
print(f"Contact swings with all fields: {len(contact):,}")

# ── timing baseline LW (20-bin approach, same as main script) ───────────────
contact['tbin'] = pd.cut(contact['timing_angle'], bins=20)
tbase = (contact.groupby('tbin', observed=True)['delta_run_exp']
                .mean().rename('lw_base'))
contact = contact.join(tbase, on='tbin').dropna(subset=['lw_base'])
contact['lw_resid'] = contact['delta_run_exp'] - contact['lw_base']
print(f"After timing baseline: {len(contact):,}")

# bat-speed quartile labels for heatmaps
contact['bs_q'] = pd.qcut(contact['bat_speed_rel'], 4,
                           labels=['Q1\n(slow)', 'Q2', 'Q3', 'Q4\n(fast)'])

# ── helpers ─────────────────────────────────────────────────────────────────
def ols_r2(xcols, ycol, d):
    sub = d[xcols + [ycol]].dropna()
    X, y = sub[xcols].values, sub[ycol].values
    return r2_score(y, LinearRegression().fit(X, y).predict(X)), len(sub)

def hex_panel(ax, fig, xcol, zcol='lw_resid', ycol='bat_speed_rel',
              xlabel=None, title=None, vmin=None, vmax=None, gridsize=40):
    sub = contact.dropna(subset=[xcol, ycol, zcol])
    norm = plt.Normalize(
        vmin=(sub[zcol].quantile(0.05) if vmin is None else vmin),
        vmax=(sub[zcol].quantile(0.95) if vmax is None else vmax)
    )
    hb = ax.hexbin(sub[xcol], sub[ycol], C=sub[zcol],
                   reduce_C_function=np.mean, gridsize=gridsize,
                   cmap='RdYlGn', norm=norm, linewidths=0.15, mincnt=15)
    fig.colorbar(hb, ax=ax, label=zcol, fraction=0.046, pad=0.04)
    ax.set_xlabel(xlabel or xcol, fontsize=9)
    ax.set_ylabel('bat_speed_rel', fontsize=9)
    if title: ax.set_title(title, fontsize=9)

def heatmap_panel(ax, xcol, xq_labels, title=None):
    contact['_xq'] = pd.qcut(contact[xcol], 4, labels=xq_labels)
    piv   = contact.pivot_table('lw_resid', index='bs_q', columns='_xq', aggfunc='mean')
    piv_n = contact.pivot_table('lw_resid', index='bs_q', columns='_xq', aggfunc='count')
    annot = [[f"{piv.iloc[i,j]:.3f}\nn={int(piv_n.iloc[i,j]):,}"
              for j in range(piv.shape[1])] for i in range(piv.shape[0])]
    sns.heatmap(piv, ax=ax, cmap='RdYlGn', center=0, annot=annot, fmt='',
                linewidths=0.5, cbar_kws={'label': 'Mean LW Residual'})
    ax.set_xlabel(f'{xcol} quartile', fontsize=9)
    ax.set_ylabel('bat_speed_rel quartile', fontsize=9)
    if title: ax.set_title(title, fontsize=9)

# ============================================================
# Printed Q&A
# ============================================================
print("\n" + "="*60)
print("Q1: How much of |angle_diff| is captured by launch_angle?")
print("="*60)
for xcols, label in [
    (['angle_diff_abs'],                    '|angle_diff| alone'),
    (['launch_angle'],                      'launch_angle alone'),
    (['angle_diff_abs', 'launch_angle'],    '|angle_diff| + launch_angle'),
    (['angle_diff'],                        'signed angle_diff alone'),
    (['angle_diff', 'launch_angle'],        'signed angle_diff + launch_angle'),
]:
    r2, n = ols_r2(xcols, 'lw_resid', contact)
    print(f"  {label:40s}: R²={r2:.4f}  n={n:,}")

print("\n  Pearson r between |angle_diff| and launch_angle:")
sub = contact[['angle_diff_abs','launch_angle']].dropna()
r, _ = pearsonr(sub['angle_diff_abs'], sub['launch_angle'])
print(f"    r={r:.4f}")

print("\n" + "="*60)
print("Q2/Q3: tilt_diff as barrel metric")
print("="*60)
for xcols, label in [
    (['tilt_diff_abs'],                              '|tilt_diff| alone'),
    (['tilt_diff'],                                  'signed tilt_diff alone'),
    (['angle_diff_abs', 'tilt_diff_abs'],            '|angle_diff| + |tilt_diff|'),
    (['bat_speed_rel', 'tilt_diff_abs'],             'bat_speed_rel + |tilt_diff|'),
    (['bat_speed_rel', 'angle_diff_abs'],            'bat_speed_rel + |angle_diff| (baseline)'),
    (['bat_speed_rel', 'angle_diff_abs',
      'tilt_diff_abs'],                              'bs_rel + |angle_diff| + |tilt_diff|'),
    (['bat_speed_rel', 'angle_diff',
      'tilt_diff'],                                  'bs_rel + signed angle_diff + signed tilt_diff'),
    (['bat_speed_rel', 'angle_diff_abs',
      'tilt_attack_gap'],                            'bs_rel + |angle_diff| + tilt-attack gap'),
]:
    r2, n = ols_r2(xcols, 'lw_resid', contact)
    print(f"  {label:50s}: R²={r2:.4f}  n={n:,}")

print("\n" + "="*60)
print("General: Full R² sweep for barrel placement candidates")
print("="*60)
all_models = [
    (['bat_speed_rel'],                                      'bat_speed_rel'),
    (['launch_angle'],                                       'launch_angle'),
    (['angle_diff_abs'],                                     '|angle_diff|'),
    (['tilt_diff_abs'],                                      '|tilt_diff|'),
    (['angle_diff'],                                         'signed angle_diff'),
    (['tilt_diff'],                                          'signed tilt_diff'),
    (['tilt_attack_gap'],                                    'tilt-attack gap'),
    (['bat_speed_rel', 'launch_angle'],                      'bs_rel + LA'),
    (['bat_speed_rel', 'angle_diff_abs'],                    'bs_rel + |angle_diff|'),
    (['bat_speed_rel', 'tilt_diff_abs'],                     'bs_rel + |tilt_diff|'),
    (['bat_speed_rel', 'angle_diff'],                        'bs_rel + signed diff'),
    (['bat_speed_rel', 'tilt_diff'],                         'bs_rel + signed tilt'),
    (['angle_diff_abs', 'tilt_diff_abs'],                    '|angle_diff| + |tilt_diff|'),
    (['bat_speed_rel', 'angle_diff_abs', 'tilt_diff_abs'],   'bs_rel + both abs'),
    (['bat_speed_rel', 'angle_diff', 'tilt_diff'],           'bs_rel + both signed'),
    (['bat_speed_rel', 'angle_diff_abs', 'launch_angle'],    'bs_rel + |diff| + LA'),
    (['bat_speed_rel', 'angle_diff', 'tilt_diff',
      'launch_angle'],                                       'bs_rel + signed + LA'),
]
r2_all = []
for xcols, label in all_models:
    r2, n = ols_r2(xcols, 'lw_resid', contact)
    r2_all.append(r2)
    print(f"  {label:45s}: R²={r2:.4f}")

# ============================================================
# Figure 1: Q1 — launch_angle vs |angle_diff|
# ============================================================
r2_ad, _ = ols_r2(['angle_diff_abs'], 'lw_resid', contact)
r2_la, _ = ols_r2(['launch_angle'], 'lw_resid', contact)
r2_both, _ = ols_r2(['angle_diff_abs','launch_angle'], 'lw_resid', contact)
vmin = contact['lw_resid'].quantile(0.05)
vmax = contact['lw_resid'].quantile(0.95)

# Also compute lw_resid after partialling out launch_angle
la_sub = contact.dropna(subset=['launch_angle','lw_resid'])
la_pred = LinearRegression().fit(la_sub[['launch_angle']], la_sub['lw_resid'])
contact.loc[la_sub.index, 'lw_resid_la_removed'] = (
    la_sub['lw_resid'] - la_pred.predict(la_sub[['launch_angle']])
)

fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6))

hex_panel(axes1[0], fig1, 'angle_diff_abs',
          xlabel="|attack_angle − launch_angle|  (°)",
          title=f"|angle_diff| → LW residual\nR²={r2_ad:.4f}",
          vmin=vmin, vmax=vmax)

hex_panel(axes1[1], fig1, 'launch_angle',
          xlabel="launch_angle  (°)",
          title=f"launch_angle → LW residual\nR²={r2_la:.4f}",
          vmin=vmin, vmax=vmax)

# Panel 3: angle_diff after removing launch_angle's contribution
vr = contact['lw_resid_la_removed'].quantile([0.05,0.95]).values
hex_panel(axes1[2], fig1, 'angle_diff_abs',
          zcol='lw_resid_la_removed',
          xlabel="|attack_angle − launch_angle|  (°)",
          title=f"|angle_diff| → LW residual with LA partialled out\n"
                f"Combined R²={r2_both:.4f}  (Δ over LA = {r2_both-r2_la:.4f})",
          vmin=vr[0], vmax=vr[1])

fig1.suptitle("Q1: Does launch_angle capture what |angle_diff| does?", fontsize=13, y=1.01)
plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'barrel_v2_q1_la_vs_diff.png'), dpi=180, bbox_inches='tight')
plt.close(fig1)
print("\nSaved barrel_v2_q1_la_vs_diff.png")

# ============================================================
# Figure 2: Q2/Q3 — 2×2 angle_diff vs tilt_diff
# ============================================================
r2_td,  _ = ols_r2(['tilt_diff'],     'lw_resid', contact)
r2_tda, _ = ols_r2(['tilt_diff_abs'], 'lw_resid', contact)
vmin2 = contact['lw_resid'].quantile(0.05)
vmax2 = contact['lw_resid'].quantile(0.95)

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

hex_panel(axes2[0,0], fig2, 'angle_diff_abs',
          xlabel="|attack_angle − launch_angle|  (°)",
          title=f"|angle_diff|  (at-contact mismatch)\nR²={r2_ad:.4f}",
          vmin=vmin2, vmax=vmax2)
hex_panel(axes2[0,1], fig2, 'tilt_diff_abs',
          xlabel="|swing_path_tilt − launch_angle|  (°)",
          title=f"|tilt_diff|  (pre-contact intent vs outcome)\nR²={r2_tda:.4f}",
          vmin=vmin2, vmax=vmax2)
hex_panel(axes2[1,0], fig2, 'angle_diff',
          xlabel="attack_angle − launch_angle  (°)  [+ = hit under]",
          title=f"signed angle_diff\nR²={r2_ad:.4f}",
          vmin=vmin2, vmax=vmax2)
hex_panel(axes2[1,1], fig2, 'tilt_diff',
          xlabel="swing_path_tilt − launch_angle  (°)  [+ = path steeper than exit]",
          title=f"signed tilt_diff\nR²={r2_td:.4f}",
          vmin=vmin2, vmax=vmax2)

fig2.suptitle("Q2/Q3: angle_diff (at contact) vs tilt_diff (pre-contact intent)",
              fontsize=13, y=1.01)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'barrel_v2_q2_tilt_hex.png'), dpi=180, bbox_inches='tight')
plt.close(fig2)
print("Saved barrel_v2_q2_tilt_hex.png")

# ============================================================
# Figure 3: R² sweep bar chart
# ============================================================
fig3, ax3 = plt.subplots(figsize=(13, 8))
n_feats = [len(m[0]) for m in all_models]
max_f = max(n_feats)
pal = plt.cm.plasma(np.linspace(0.15, 0.85, max_f))
bar_cols = [pal[n-1] for n in n_feats]
labels3  = [m[1] for m in all_models]
ybars = ax3.barh(labels3[::-1], r2_all[::-1],
                 color=bar_cols[::-1], alpha=0.85, edgecolor='k', lw=0.4)
for bar, v in zip(ybars, r2_all[::-1]):
    ax3.text(v + 0.0003, bar.get_y()+bar.get_height()/2,
             f'{v:.4f}', va='center', fontsize=8)
ax3.set_xlabel("R² predicting LW residual (timing-conditioned)", fontsize=11)
ax3.set_title("Barrel Placement — Explanatory Power Sweep\n2025 MLB Fastballs", fontsize=12)
ax3.axvline(0, color='black', lw=0.8)
from matplotlib.patches import Patch
ax3.legend(handles=[Patch(color=pal[i-1], label=f'{i} feature{"s" if i>1 else ""}')
                    for i in range(1, max_f+1)], fontsize=8, loc='lower right')
plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'barrel_v2_r2_sweep.png'), dpi=180, bbox_inches='tight')
plt.close(fig3)
print("Saved barrel_v2_r2_sweep.png")

# ============================================================
# Figure 4: Proposed barrel placement metric
# General: bat_speed_rel × f(signed tilt_diff) — using pre-contact path
# Rationale: swing_path_tilt is measured before ball-bat interaction,
# so it reflects the batter's barrel positioning intent more cleanly
# than attack_angle (which is partially determined by ball-bat physics).
# The signed tilt_diff shows where on the ball the path was aimed relative
# to where the ball actually went (small positive = slightly under = optimal).
#
# We fit a degree-2 polynomial of (bat_speed_rel, tilt_diff) to predict
# lw_resid. The polynomial captures the non-linear "sweet spot" in tilt_diff.
# ============================================================
metric_cols = ['bat_speed_rel', 'tilt_diff', 'angle_diff']
h4 = contact.dropna(subset=metric_cols + ['lw_resid', 'launch_speed_angle']).copy()

pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
    ('lr', LinearRegression()),
])
pipe.fit(h4[metric_cols], h4['lw_resid'])
h4['barrel_score'] = pipe.predict(h4[metric_cols])
r2_bs = r2_score(h4['lw_resid'], h4['barrel_score'])
r_lsa, _ = pearsonr(h4['barrel_score'], h4['launch_speed_angle'])
r_lsp, _ = pearsonr(h4['barrel_score'], h4['launch_speed'].reindex(h4.index))

# per-batter aggregate: does this identify skilled barrel-placers?
batter_stats = (h4.groupby('batter')
                  .agg(mean_barrel=('barrel_score','mean'),
                       mean_lsa   =('launch_speed_angle','mean'),
                       n          =('barrel_score','count'))
                  .query('n >= 20').reset_index())
r_batter, _ = pearsonr(batter_stats['mean_barrel'], batter_stats['mean_lsa'])

print(f"\nProposed barrel score (poly-2 of bs_rel, tilt_diff, angle_diff):")
print(f"  R²(lw_resid):            {r2_bs:.4f}")
print(f"  r(launch_speed_angle):   {r_lsa:.4f}")
print(f"  r(mean per-batter LSA):  {r_batter:.4f}  (n={len(batter_stats):,} batters ≥20 swings)")

fig4, axes4 = plt.subplots(1, 3, figsize=(20, 6))

# Panel 1: barrel_score surface on (tilt_diff, bat_speed_rel)
ax = axes4[0]
hb = ax.hexbin(h4['tilt_diff'], h4['bat_speed_rel'], C=h4['barrel_score'],
               reduce_C_function=np.mean, gridsize=45,
               cmap='RdYlGn', linewidths=0.15, mincnt=15)
fig4.colorbar(hb, ax=ax, label='Barrel Score', fraction=0.046, pad=0.04)
ax.set_xlabel("signed tilt_diff  (swing_path_tilt − launch_angle)  (°)", fontsize=9)
ax.set_ylabel("bat_speed_rel", fontsize=9)
ax.set_title("Barrel Score surface\n(poly-2 of bs_rel + tilt_diff + angle_diff)", fontsize=9)

# Panel 2: barrel_score by launch_speed_angle category
ax = axes4[1]
lsa_map = {1:'1-weak',2:'2-topped',3:'3-under',4:'4-flare',5:'5-solid',6:'6-barrel'}
h4['lsa_label'] = h4['launch_speed_angle'].map(lsa_map)
order = list(lsa_map.values())
sns.boxplot(data=h4, x='lsa_label', y='barrel_score', order=order,
            palette='RdYlGn', linewidth=0.8, ax=ax)
ax.set_xlabel("launch_speed_angle category", fontsize=9)
ax.set_ylabel("Barrel Score", fontsize=9)
ax.set_title(f"Barrel Score by Contact Quality\nr = {r_lsa:.3f}", fontsize=9)
ax.tick_params(axis='x', rotation=20)

# Panel 3: per-batter mean barrel score vs mean launch_speed_angle
ax = axes4[2]
ax.hexbin(batter_stats['mean_barrel'], batter_stats['mean_lsa'],
          gridsize=30, cmap='Blues', mincnt=1, linewidths=0.2)
m, b = np.polyfit(batter_stats['mean_barrel'], batter_stats['mean_lsa'], 1)
xl = np.linspace(batter_stats['mean_barrel'].min(), batter_stats['mean_barrel'].max(), 100)
ax.plot(xl, m*xl+b, 'r-', lw=2)
ax.set_xlabel("Batter Mean Barrel Score", fontsize=9)
ax.set_ylabel("Batter Mean launch_speed_angle", fontsize=9)
ax.set_title(f"Per-batter: barrel skill stability\nr = {r_batter:.3f}  "
             f"(n={len(batter_stats):,} batters)", fontsize=9)

fig4.suptitle("Proposed Barrel Placement Metric — Validation", fontsize=13, y=1.01)
plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'barrel_v2_proposed_metric.png'), dpi=180, bbox_inches='tight')
plt.close(fig4)
print("Saved barrel_v2_proposed_metric.png")

print("\nDone.")
