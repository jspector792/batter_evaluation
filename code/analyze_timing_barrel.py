"""
Timing and Barrel Placement Analysis — 2025 MLB Fastballs
Answers Q1-Q6 and tests H1/H2 from the experiment design.
"""
import os, glob, pickle, warnings, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

FASTBALL_TYPES = {"FF", "SI", "FC", "FT"}
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fastballs_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# Tee stdout to file
class Tee:
    def __init__(self, *files): self.files = files
    def write(self, s):
        for f in self.files: f.write(s); f.flush()
    def flush(self):
        for f in self.files: f.flush()

log = open(os.path.join(OUT_DIR, 'analysis_summary.txt'), 'w')
sys.stdout = Tee(sys.__stdout__, log)

# ------------------------------------------------------------------ load ----
COLS = [
    "pitch_type", "pitch_name", "batter",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
    "attack_angle", "attack_direction", "swing_path_tilt",
    "swing_length", "bat_speed",
    "launch_angle", "launch_speed", "launch_speed_angle",
    "delta_run_exp",
]

files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"Loading {len(files)} team files...")
chunks = [pd.read_csv(f, usecols=COLS) for f in files]
df = pd.concat(chunks, ignore_index=True)
df = df[df['pitch_type'].isin(FASTBALL_TYPES)].copy()
print(f"Fastball pitches: {len(df):,}")
print(df['pitch_type'].value_counts().to_string())

# -------------------------------------------------------- pitch type dict ----
pt_dict = (df.dropna(subset=['pitch_type','pitch_name'])
             .groupby('pitch_type')['pitch_name']
             .agg(lambda s: s.mode()[0])
             .to_dict())
print(f"\nPitch type → name: {pt_dict}")
with open(os.path.join(OUT_DIR, 'pitch_type_names.pkl'), 'wb') as fh:
    pickle.dump(pt_dict, fh)

# ---------------------------------------------------- feature engineering ----
IX = 'intercept_ball_minus_batter_pos_x_inches'
IY = 'intercept_ball_minus_batter_pos_y_inches'

df['timing_angle_raw'] = np.degrees(np.arctan2(df[IX], df[IY]))
timing_median = df['timing_angle_raw'].median()
df['timing_angle'] = df['timing_angle_raw'] - timing_median
print(f"\nTiming angle raw median = {timing_median:.2f}°  "
      f"(std = {df['timing_angle_raw'].std():.2f}°)")

# Relative bat speed: bat_speed / batter's max bat_speed (min 5 pitch-events)
bs_counts = df.groupby('batter')['bat_speed'].count()
valid_b = bs_counts[bs_counts >= 5].index
bs_max = df[df['batter'].isin(valid_b)].groupby('batter')['bat_speed'].max()
df = df.join(bs_max.rename('batter_max_bs'), on='batter')
df['bat_speed_rel'] = df['bat_speed'] / df['batter_max_bs']

df['angle_diff']     = df['attack_angle'] - df['launch_angle']   # signed
df['angle_diff_abs'] = df['angle_diff'].abs()

# ---------------------------------------------------------------- subsets ----
swings  = df.dropna(subset=[IX, IY, 'timing_angle', 'delta_run_exp']).copy()
contact = swings.dropna(subset=['launch_angle', 'launch_speed']).copy()
print(f"\nSwings (intercept present):  {len(swings):,}")
print(f"Contact (launch metrics):    {len(contact):,}")

# ---------------------------------------------------------------- helpers ----
def corr_pair(xa, ya, lx, ly):
    m = np.isfinite(xa) & np.isfinite(ya)
    r,  rp = pearsonr(xa[m], ya[m])
    rh, sp = spearmanr(xa[m], ya[m])
    print(f"  {lx}  ↔  {ly}")
    print(f"    Pearson r={r:+.4f} (p={rp:.2e})  Spearman ρ={rh:+.4f} (p={sp:.2e})  n={m.sum():,}")
    return r, rh

def ols_r2(xcols, ycol, df_in):
    d = df_in[xcols + [ycol]].dropna()
    X, y = d[xcols].values, d[ycol].values
    return r2_score(y, LinearRegression().fit(X, y).predict(X)), len(d)

# ================================================================ Q1 ========
print("\n" + "="*60)
print("Q1: timing_angle  ↔  attack_direction")
print("="*60)
q1 = swings.dropna(subset=['timing_angle','attack_direction'])
r_q1, rho_q1 = corr_pair(q1['timing_angle'].values, q1['attack_direction'].values,
                          'timing_angle', 'attack_direction')

# ================================================================ Q2 ========
print("\n" + "="*60)
print("Q2: timing_angle  ↔  swing_length")
print("="*60)
q2 = swings.dropna(subset=['timing_angle','swing_length'])
r_q2, rho_q2 = corr_pair(q2['timing_angle'].values, q2['swing_length'].values,
                          'timing_angle', 'swing_length')

# ================================================================ Q3 ========
print("\n" + "="*60)
print("Q3: swing_path_tilt  ↔  attack_direction  (redundancy check)")
print("="*60)
q3 = swings.dropna(subset=['swing_path_tilt','attack_direction'])
r_q3, rho_q3 = corr_pair(q3['swing_path_tilt'].values, q3['attack_direction'].values,
                          'swing_path_tilt', 'attack_direction')

print("\n  Also: swing_path_tilt ↔ attack_angle")
q3b = swings.dropna(subset=['swing_path_tilt','attack_angle'])
r_q3b, rho_q3b = corr_pair(q3b['swing_path_tilt'].values, q3b['attack_angle'].values,
                            'swing_path_tilt', 'attack_angle')

print("\n  What does swing_path_tilt add for predicting launch_angle?")
for cols, label in [
    (['attack_direction'],                    'attack_direction only'),
    (['swing_path_tilt'],                     'swing_path_tilt only'),
    (['attack_direction','swing_path_tilt'],  'both'),
]:
    r2, n = ols_r2(cols, 'launch_angle', contact)
    print(f"    R²(launch_angle ~ {label:35s}) = {r2:.4f}  (n={n:,})")

# ================================================================ Q4 ========
print("\n" + "="*60)
print("Q4: Predictors of launch_speed_angle")
print("="*60)
q4 = contact.dropna(subset=['launch_speed_angle'])
pred_cols = [
    ('attack_angle',    'attack_angle'),
    ('launch_angle',    'launch_angle'),
    ('launch_speed',    'launch_speed'),
    ('bat_speed',       'bat_speed'),
    ('bat_speed_rel',   'bat_speed_rel'),
    ('swing_path_tilt', 'swing_path_tilt'),
    ('attack_direction','attack_direction'),
    ('swing_length',    'swing_length'),
    ('|atk-launch|',    'angle_diff_abs'),
    ('atk-launch',      'angle_diff'),
    ('timing_angle',    'timing_angle'),
]
q4_spearman = {}
print(f"  Spearman ρ  (n up to {len(q4):,}):")
for label, col in pred_cols:
    sub = q4[['launch_speed_angle', col]].dropna()
    rh, p = spearmanr(sub['launch_speed_angle'], sub[col])
    q4_spearman[label] = rh
    print(f"    {label:30s}: ρ={rh:+.4f}  (p={p:.2e}  n={len(sub):,})")

print("\n  OLS R² for launch_speed_angle:")
for cols, label in [
    (['attack_angle'],                                           'attack_angle'),
    (['launch_angle'],                                           'launch_angle'),
    (['launch_speed'],                                           'launch_speed'),
    (['attack_angle','launch_angle'],                           'atk + launch angles'),
    (['attack_angle','launch_angle','launch_speed'],             'angles + speed'),
    (['attack_angle','launch_angle','launch_speed','bat_speed_rel','angle_diff_abs'], 'full'),
]:
    r2, n = ols_r2(cols, 'launch_speed_angle', q4)
    print(f"    {label:45s}: R²={r2:.4f}  (n={n:,})")

# ================================================================ Q5 ========
print("\n" + "="*60)
print("Q5: (attack_angle - launch_angle) as predictor of launch_speed")
print("="*60)
q5 = contact.dropna(subset=['angle_diff','launch_speed'])
corr_pair(q5['angle_diff'].values,     q5['launch_speed'].values, 'angle_diff (signed)', 'launch_speed')
corr_pair(q5['angle_diff_abs'].values, q5['launch_speed'].values, '|angle_diff|',        'launch_speed')
r2_d,  _ = ols_r2(['angle_diff'],     'launch_speed', q5)
r2_da, _ = ols_r2(['angle_diff_abs'], 'launch_speed', q5)
print(f"\n  OLS R²(~ angle_diff):    {r2_d:.4f}")
print(f"  OLS R²(~ |angle_diff|):  {r2_da:.4f}")

# ================================================================ Q6 ========
print("\n" + "="*60)
print("Q6: Adding bat_speed to explain launch_speed")
print("="*60)
models_q6 = [
    # --- baseline singles ---
    (['angle_diff_abs'],                                                  '|angle_diff| only'),
    (['bat_speed_rel'],                                                   'bat_speed_rel only'),
    # --- two-variable core ---
    (['angle_diff_abs', 'bat_speed_rel'],                                '|diff| + bs_rel'),
    # --- add timing proxy: timing_angle ---
    (['angle_diff_abs', 'bat_speed_rel', 'timing_angle'],               '|diff|+bs_rel+timing_angle'),
    # --- add timing proxy: attack_direction (substitute) ---
    (['angle_diff_abs', 'bat_speed_rel', 'attack_direction'],           '|diff|+bs_rel+atk_dir'),
    # --- full models ---
    (['angle_diff_abs', 'bat_speed_rel', 'timing_angle',
      'attack_angle', 'launch_angle'],                                   'full (w/ timing_angle)'),
    (['angle_diff_abs', 'bat_speed_rel', 'attack_direction',
      'attack_angle', 'launch_angle'],                                   'full (w/ atk_dir)'),
]
r2_vals_q6 = []
for cols, label in models_q6:
    r2, n = ols_r2(cols, 'launch_speed', contact)
    r2_vals_q6.append(r2)
    print(f"  {label:50s}: R²={r2:.4f}  (n={n:,})")

print("\n  → timing_angle vs attack_direction as 3rd variable (Δ over |diff|+bs_rel):")
r2_base, _ = ols_r2(['angle_diff_abs','bat_speed_rel'], 'launch_speed', contact)
r2_ta,   _ = ols_r2(['angle_diff_abs','bat_speed_rel','timing_angle'], 'launch_speed', contact)
r2_ad,   _ = ols_r2(['angle_diff_abs','bat_speed_rel','attack_direction'], 'launch_speed', contact)
print(f"    base R²={r2_base:.4f}  +timing_angle ΔR²={r2_ta-r2_base:.4f}  "
      f"+attack_direction ΔR²={r2_ad-r2_base:.4f}")

# ============================================================= H1 ==========
print("\n" + "="*60)
print("H1: Mean LW vs timing angle (all swings)")
print("="*60)
h1 = swings.dropna(subset=['timing_angle','delta_run_exp'])
bins_h1 = np.linspace(h1['timing_angle'].quantile(0.005),
                       h1['timing_angle'].quantile(0.995), 41)
h1['bin'] = pd.cut(h1['timing_angle'], bins=bins_h1)
lw_bins = (h1.groupby('bin', observed=True)
             .agg(mean_lw=('delta_run_exp','mean'),
                  sem_lw =('delta_run_exp','sem'),
                  mid    =('timing_angle','median'),
                  n      =('delta_run_exp','count'))
             .dropna().reset_index())
peak = lw_bins.loc[lw_bins['mean_lw'].idxmax()]
print(f"  Peak mean LW = {peak['mean_lw']:.4f}  at timing_angle = {peak['mid']:.2f}°")
print(f"  (0 = median timing; negative = earlier, positive = later)")

fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.fill_between(lw_bins['mid'],
                 lw_bins['mean_lw'] - 1.96*lw_bins['sem_lw'],
                 lw_bins['mean_lw'] + 1.96*lw_bins['sem_lw'],
                 alpha=0.25, color='royalblue')
ax1.plot(lw_bins['mid'], lw_bins['mean_lw'], color='royalblue', lw=2, marker='o', ms=4)
ax1.axhline(0,  color='black', lw=0.8, ls='--')
ax1.axvline(0,  color='red',   lw=1,   ls=':',  label='Median timing (0°)')
ax1.axvline(peak['mid'], color='green', lw=1.5, ls='--',
            label=f"Peak LW @ {peak['mid']:.1f}°")
ax1.set_xlabel("Timing Angle  (°, centered on median; negative=earlier)", fontsize=12)
ax1.set_ylabel("Mean Δ Run Expectancy (LW)", fontsize=12)
ax1.set_title("H1: Mean LW vs Swing Timing Angle — 2025 MLB Fastballs", fontsize=13)
ax1.legend()
plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'h1_lw_vs_timing.png'), dpi=180, bbox_inches='tight')
plt.close(fig1)
print("  Saved h1_lw_vs_timing.png")

# ============================================================= H2 ==========
print("\n" + "="*60)
print("H2: Barrel placement skill conditioned on timing")
print("="*60)
h2 = contact.dropna(subset=['timing_angle','delta_run_exp',
                              'bat_speed_rel','angle_diff_abs']).copy()
# Timing baseline: bin mean LW from contact swings
h2['tbin'] = pd.cut(h2['timing_angle'], bins=20)
tbase = h2.groupby('tbin', observed=True)['delta_run_exp'].mean().rename('lw_base')
h2 = h2.join(tbase, on='tbin')
h2 = h2.dropna(subset=['lw_base'])
h2['lw_resid'] = h2['delta_run_exp'] - h2['lw_base']

r_bsr,  p_bsr  = pearsonr(h2['bat_speed_rel'],  h2['lw_resid'])
r_adif, p_adif = pearsonr(h2['angle_diff_abs'], h2['lw_resid'])
r_comb,  _ = pearsonr(h2['bat_speed_rel'] / (1 + h2['angle_diff_abs']), h2['lw_resid'])
print(f"  bat_speed_rel  ↔ LW residual: r={r_bsr:.4f}  (p={p_bsr:.2e})")
print(f"  |angle_diff|   ↔ LW residual: r={r_adif:.4f}  (p={p_adif:.2e})")
print(f"  composite barrel_skill ↔ LW residual: r={r_comb:.4f}")

r2_h2_1, _ = ols_r2(['bat_speed_rel'],              'lw_resid', h2)
r2_h2_2, _ = ols_r2(['angle_diff_abs'],             'lw_resid', h2)
r2_h2_3, _ = ols_r2(['bat_speed_rel','angle_diff_abs'], 'lw_resid', h2)
print(f"\n  OLS R²(lw_resid ~ bat_speed_rel):                {r2_h2_1:.4f}")
print(f"  OLS R²(lw_resid ~ |angle_diff|):                 {r2_h2_2:.4f}")
print(f"  OLS R²(lw_resid ~ bat_speed_rel + |angle_diff|): {r2_h2_3:.4f}")

# ================================================================ FIGURES ===

# --- Fig 2: Q1/Q2/Q3 scatter (4 panels) ------------------------------------
fig2, axes2 = plt.subplots(1, 4, figsize=(24, 5))
plot_specs = [
    ('timing_angle',    'attack_direction',  q1,
     f"Q1: Timing Angle vs Attack Direction\nPearson r={r_q1:.3f}  Spearman ρ={rho_q1:.3f}"),
    ('timing_angle',    'swing_length',      q2,
     f"Q2: Timing Angle vs Swing Length\nPearson r={r_q2:.3f}  Spearman ρ={rho_q2:.3f}"),
    ('attack_direction','swing_path_tilt',   q3,
     f"Q3a: Attack Direction vs Swing Path Tilt\nPearson r={r_q3:.3f}  Spearman ρ={rho_q3:.3f}"),
    ('attack_angle',    'swing_path_tilt',   q3b,
     f"Q3b: Attack Angle vs Swing Path Tilt\nPearson r={r_q3b:.3f}  Spearman ρ={rho_q3b:.3f}"),
]
for ax, (xc, yc, src, title) in zip(axes2, plot_specs):
    sub = src.dropna(subset=[xc, yc])
    ax.hexbin(sub[xc], sub[yc], gridsize=45, cmap='Blues', mincnt=2, linewidths=0.1)
    m, b = np.polyfit(sub[xc], sub[yc], 1)
    xl = np.linspace(sub[xc].quantile(0.01), sub[xc].quantile(0.99), 100)
    ax.plot(xl, m*xl+b, 'r-', lw=2, label='OLS fit')
    ax.set_xlabel(xc, fontsize=10); ax.set_ylabel(yc, fontsize=10)
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'q1_q3_correlations.png'), dpi=180, bbox_inches='tight')
plt.close(fig2)
print("\nSaved q1_q3_correlations.png")

# --- Fig 3: Q4 bar + Q5/Q6 R² bar ------------------------------------------
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))

ax = axes3[0]
s4 = sorted(q4_spearman.items(), key=lambda x: abs(x[1]), reverse=True)
names4 = [s[0] for s in s4]
vals4  = [s[1] for s in s4]
cols4  = ['#2ca02c' if v > 0 else '#d62728' for v in vals4]
ax.barh(names4[::-1], vals4[::-1], color=cols4[::-1], alpha=0.8, edgecolor='k', lw=0.5)
ax.axvline(0, color='black', lw=1)
for i, (n, v) in enumerate(zip(names4[::-1], vals4[::-1])):
    ax.text(v + (0.01 if v>=0 else -0.01), i, f'{v:.3f}',
            va='center', ha='left' if v>=0 else 'right', fontsize=8)
ax.set_xlabel("Spearman ρ", fontsize=11)
ax.set_title("Q4: Predictors of launch_speed_angle\n(Spearman ρ)", fontsize=12)

ax = axes3[1]
# Colors: blue=baseline, orange=core, green=+timing_angle, red=+atk_dir, purple=full
bar_colors = ['#1f77b4','#1f77b4', '#ff7f0e', '#2ca02c','#d62728', '#9467bd','#8c564b']
bars = ax.bar(range(len(models_q6)), r2_vals_q6, color=bar_colors, alpha=0.8,
              edgecolor='k', lw=0.5)
for bar, v in zip(bars, r2_vals_q6):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.001, f'{v:.3f}',
            ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(range(len(models_q6)))
ax.set_xticklabels([m[1] for m in models_q6], rotation=40, ha='right', fontsize=7.5)
ax.set_ylabel("R² (predicting launch_speed)", fontsize=11)
ax.set_title("Q5/Q6: Explaining Launch Speed\n(in-sample OLS R²)", fontsize=12)
ax.set_ylim(0, min(1.0, max(r2_vals_q6)*1.25))
# Legend for color groups
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color='#1f77b4', label='baseline singles'),
    Patch(color='#ff7f0e', label='|diff|+bs_rel core'),
    Patch(color='#2ca02c', label='+timing_angle'),
    Patch(color='#d62728', label='+attack_direction'),
    Patch(color='#9467bd', label='full (timing)'),
    Patch(color='#8c564b', label='full (atk_dir)'),
], fontsize=7, loc='upper left')

plt.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'q4_q6_launch_speed.png'), dpi=180, bbox_inches='tight')
plt.close(fig3)
print("Saved q4_q6_launch_speed.png")

# --- Fig 4: H2 barrel placement (2×2: row 0 = |diff|, row 1 = signed diff) --
r_diff_signed, p_diff_signed = pearsonr(h2['angle_diff'], h2['lw_resid'])
print(f"\n  signed angle_diff ↔ LW residual: r={r_diff_signed:.4f}  (p={p_diff_signed:.2e})")

fig4, axes4 = plt.subplots(2, 2, figsize=(16, 12))

def _hex_row(ax_hex, ax_heat, xcol, xlabel, xtick_labels, r_bs, r_x):
    # hexbin
    hb = ax_hex.hexbin(h2[xcol], h2['bat_speed_rel'], C=h2['lw_resid'],
                       reduce_C_function=np.mean, gridsize=40,
                       cmap='RdYlGn', linewidths=0.2, mincnt=10)
    cb = fig4.colorbar(hb, ax=ax_hex)
    cb.set_label('Mean LW Residual', fontsize=9)
    ax_hex.set_xlabel(xlabel, fontsize=10)
    ax_hex.set_ylabel("Relative Bat Speed (fraction of batter max)", fontsize=10)
    ax_hex.set_title(f"bat_speed_rel r={r_bs:.3f}   {xcol} r={r_x:.3f}", fontsize=10)
    # heatmap
    h2['_xq'] = pd.qcut(h2[xcol], 4, labels=xtick_labels)
    piv   = h2.pivot_table('lw_resid', index='bs_q', columns='_xq', aggfunc='mean')
    piv_n = h2.pivot_table('lw_resid', index='bs_q', columns='_xq', aggfunc='count')
    annot = [[f"{piv.iloc[i,j]:.3f}\nn={int(piv_n.iloc[i,j]):,}"
              for j in range(piv.shape[1])] for i in range(piv.shape[0])]
    sns.heatmap(piv, ax=ax_heat, cmap='RdYlGn', center=0, annot=annot, fmt='',
                linewidths=0.5, cbar_kws={'label':'Mean LW Residual'})
    ax_heat.set_xlabel(f"{xcol} Quartile", fontsize=10)
    ax_heat.set_ylabel("Relative Bat Speed Quartile", fontsize=10)

# Bat speed quartiles (shared across rows)
h2['bs_q'] = pd.qcut(h2['bat_speed_rel'], 4,
                      labels=['Q1\n(slow)', 'Q2', 'Q3', 'Q4\n(fast)'])

# Row 0: |angle_diff|
_hex_row(axes4[0,0], axes4[0,1],
         'angle_diff_abs', "|Attack − Launch Angle|  (°)",
         ['Q1\n(match)', 'Q2', 'Q3', 'Q4\n(miss)'],
         r_bsr, r_adif)
axes4[0,0].set_title(f"Row 1 — |angle_diff|\n"
                     f"bat_speed_rel r={r_bsr:.3f}   |angle_diff| r={r_adif:.3f}", fontsize=10)
axes4[0,1].set_title("Mean LW Residual: Bat Speed × |Angle Match| Quartiles", fontsize=10)

# Row 1: signed angle_diff  (positive = attack steeper than launch → hit under)
_hex_row(axes4[1,0], axes4[1,1],
         'angle_diff', "Attack − Launch Angle  (°)  [+ = hit under, − = hit on top]",
         ['Q1\n(−, on top)', 'Q2', 'Q3', 'Q4\n(+, under)'],
         r_bsr, r_diff_signed)
axes4[1,0].set_title(f"Row 2 — signed angle_diff\n"
                     f"bat_speed_rel r={r_bsr:.3f}   angle_diff r={r_diff_signed:.3f}", fontsize=10)
axes4[1,1].set_title("Mean LW Residual: Bat Speed × Signed Angle Diff Quartiles", fontsize=10)

plt.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'h2_barrel_placement.png'), dpi=180, bbox_inches='tight')
plt.close(fig4)
print("Saved h2_barrel_placement.png")

log.close()
sys.stdout = sys.__stdout__
print("Done. Summary → out/analysis_summary.txt")
