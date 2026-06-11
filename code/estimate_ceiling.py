"""
estimate_r2_ceiling_v2.py
=========================
Estimate theoretical R² ceiling for LA and EV using two methods:
  1. Binning: fixed-grid partitioning (interpretable, prone to curse of dimensionality)
  2. Regression tree: adaptive partitioning (robust to high dimensions)

Tests all predictor subsets of size ≥3 for both outcomes.

Runtime: ~5-10 minutes for 466 LA combos + 968 EV combos
"""

import os, glob, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "r2_ceiling")
os.makedirs(OUT_DIR, exist_ok=True)

INTERCEPT_X = 'intercept_ball_minus_batter_pos_x_inches'
INTERCEPT_Y = 'intercept_ball_minus_batter_pos_y_inches'
IN_PLAY = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
FASTBALL_TYPES = ['FF', 'SI', 'FC', 'FT']

sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE = '#2563EB'; GREEN = '#16A34A'; RED = '#DC2626'


def prepare_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    rs_mean = df['release_speed'].mean()
    df['release_speed_c'] = df['release_speed'] - rs_mean
    df['pfx_x_adj'] = df['pfx_x'].abs()
    
    fb_stats = (df[df['pitch_type'].isin(FASTBALL_TYPES)]
                .groupby('pitcher').agg(fb_pfx_z=('pfx_z', 'mean'),
                                         fb_speed=('release_speed', 'mean'),
                                         fb_count=('pitch_type', 'count')))
    fb_stats = fb_stats[fb_stats['fb_count'] >= 20]
    df = df.merge(fb_stats[['fb_pfx_z', 'fb_speed']], left_on='pitcher',
                  right_index=True, how='left')
    df['vmov_diff'] = df['pfx_z'] - df['fb_pfx_z']
    df['velo_diff'] = df['release_speed'] - df['fb_speed']
    
    df = df.rename(columns={INTERCEPT_X: 'intercept_x',
                            INTERCEPT_Y: 'intercept_y',
                            'launch_speed': 'exit_velo',
                            'swing_path_tilt': 'swing_path_int'})
    return df


def ceiling_binning(df, outcome, predictors, bin_widths, min_bin_size=10):
    """Fixed-grid binning method."""
    df = df.copy()
    bin_cols = []
    for pred in predictors:
        width = bin_widths.get(pred, df[pred].std() * 0.2)
        bins = np.arange(df[pred].min(), df[pred].max() + width, width)
        df[f'_{pred}_bin'] = pd.cut(df[pred], bins=bins, labels=False,
                                     include_lowest=True)
        bin_cols.append(f'_{pred}_bin')
    
    df['_bin'] = df[bin_cols].astype(str).agg('_'.join, axis=1)
    bin_counts = df['_bin'].value_counts()
    valid_bins = bin_counts[bin_counts >= min_bin_size].index
    df_valid = df[df['_bin'].isin(valid_bins)]
    
    within_var = df_valid.groupby('_bin')[outcome].var().mean()
    total_var = df[outcome].var()
    r2 = 1 - within_var / total_var
    
    return r2, len(valid_bins), len(df_valid)


def ceiling_tree(df, outcome, predictors, max_leaves=100, min_samples_leaf=10):
    """Regression tree adaptive partitioning method."""
    df_sub = df[predictors + [outcome]].dropna()
    X, y = df_sub[predictors].values, df_sub[outcome].values
    
    tree = DecisionTreeRegressor(max_leaf_nodes=max_leaves,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=42).fit(X, y)
    
    leaf_ids = tree.apply(X)
    df_sub['_leaf'] = leaf_ids
    leaf_stats = df_sub.groupby('_leaf')[outcome].agg(['var', 'count'])
    
    within_var_weighted = (leaf_stats['var'] * leaf_stats['count']).sum() / leaf_stats['count'].sum()
    total_var = df_sub[outcome].var()
    r2 = 1 - within_var_weighted / total_var
    
    return r2, len(leaf_stats)


def main():
    df = prepare_data()
    
    print('R² CEILING ESTIMATION — ALL PREDICTOR SUBSETS')
    print('='*70)
    
    # ── LA ────────────────────────────────────────────────────────────────────
    la_preds = ['attack_angle', 'pfx_z', 'plate_z', 'intercept_y',
                'swing_path_int', 'plate_x', 'pfx_x_adj', 'vmov_diff', 'velo_diff']
    df_la = df[df['description'].isin(IN_PLAY)].dropna(subset=['launch_angle'] + la_preds)
    
    la_widths = {'attack_angle': 3.0, 'pfx_z': 2.0, 'plate_z': 0.15,
                 'intercept_y': 3.0, 'swing_path_int': 3.0, 'plate_x': 0.15,
                 'pfx_x_adj': 2.0, 'vmov_diff': 2.0, 'velo_diff': 2.0}
    
    n_la = sum(1 for k in range(3, 10) for _ in itertools.combinations(la_preds, k))
    print(f'\nLA: {n_la} combinations, {len(df_la):,} obs')
    
    la_recs = []
    for size in range(3, len(la_preds) + 1):
        for i, combo in enumerate(itertools.combinations(la_preds, size)):
            if i % 20 == 0: print(f'  n={size}: {i}', end='\r')
            r2_bin, n_bins, n_obs = ceiling_binning(df_la, 'launch_angle', list(combo), la_widths, 10)
            r2_tree, n_leaves = ceiling_tree(df_la, 'launch_angle', list(combo), 100, 10)
            la_recs.append({'n_preds': size, 'predictors': ', '.join(combo),
                            'r2_bin': r2_bin, 'r2_tree': r2_tree,
                            'n_bins': n_bins, 'n_leaves': n_leaves})
    
    la_df = pd.DataFrame(la_recs).sort_values('r2_tree', ascending=False)
    la_df.to_csv(os.path.join(OUT_DIR, 'la_ceiling.csv'), index=False)
    print(f'\n→ la_ceiling.csv: {(la_df["r2_bin"]>0).sum()} pos bin, {(la_df["r2_tree"]>0).sum()} pos tree')
    
    # ── EV ────────────────────────────────────────────────────────────────────
    has_bs = 'bat_speed' in df.columns
    ev_preds = (['bat_speed'] if has_bs else []) + \
               ['attack_angle', 'intercept_y', 'intercept_x', 'swing_path_int',
                'release_speed_c', 'plate_x', 'pfx_x_adj', 'vmov_diff', 'velo_diff']
    df_ev = df[df['description'].isin(IN_PLAY)].dropna(subset=['exit_velo'] + ev_preds)
    
    ev_widths = {'bat_speed': 2.0, 'attack_angle': 3.0, 'intercept_y': 3.0,
                 'intercept_x': 3.0, 'swing_path_int': 3.0, 'release_speed_c': 0.3,
                 'plate_x': 0.15, 'pfx_x_adj': 2.0, 'vmov_diff': 2.0, 'velo_diff': 2.0}
    
    n_ev = sum(1 for k in range(3, len(ev_preds)+1) for _ in itertools.combinations(ev_preds, k))
    print(f'\nEV: {n_ev} combinations, {len(df_ev):,} obs')
    
    ev_recs = []
    for size in range(3, len(ev_preds) + 1):
        for i, combo in enumerate(itertools.combinations(ev_preds, size)):
            if i % 20 == 0: print(f'  n={size}: {i}', end='\r')
            r2_bin, n_bins, n_obs = ceiling_binning(df_ev, 'exit_velo', list(combo), ev_widths, 10)
            r2_tree, n_leaves = ceiling_tree(df_ev, 'exit_velo', list(combo), 100, 10)
            ev_recs.append({'n_preds': size, 'predictors': ', '.join(combo),
                            'r2_bin': r2_bin, 'r2_tree': r2_tree,
                            'n_bins': n_bins, 'n_leaves': n_leaves})
    
    ev_df = pd.DataFrame(ev_recs).sort_values('r2_tree', ascending=False)
    ev_df.to_csv(os.path.join(OUT_DIR, 'ev_ceiling.csv'), index=False)
    print(f'\n→ ev_ceiling.csv: {(ev_df["r2_bin"]>0).sum()} pos bin, {(ev_df["r2_tree"]>0).sum()} pos tree')
    
    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('TOP 10 (tree ceiling)')
    print('='*70)
    print('\nLA:')
    print(la_df[['n_preds', 'r2_tree', 'r2_bin', 'n_leaves', 'predictors']].head(10).to_string(index=False))
    print('\nEV:')
    print(ev_df[['n_preds', 'r2_tree', 'r2_bin', 'n_leaves', 'predictors']].head(10).to_string(index=False))
    
    # ── PLOT ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    for row, (recs, title, obs) in enumerate([(la_df, 'Launch Angle', 0.20),
                                                (ev_df, 'Exit Velocity', 0.25)]):
        # Tree ceiling vs n_preds
        ax = axes[row, 0]
        ax.scatter(recs['n_preds'], recs['r2_tree'], alpha=0.3, s=15, color=BLUE)
        top = recs.iloc[0]
        ax.scatter(top['n_preds'], top['r2_tree'], s=200, color=RED, marker='*',
                   edgecolor='white', linewidth=2, label=f'Best: {top["r2_tree"]:.3f}')
        ax.axhline(obs, color=GREEN, linewidth=1.5, linestyle='--',
                   label=f'Observed: {obs:.2f}')
        ax.set_xlabel('N predictors'); ax.set_ylabel('R² ceiling (tree)')
        ax.set_title(title, fontweight='bold'); ax.legend(); ax.grid(alpha=0.3)
        
        # Bin vs tree
        ax = axes[row, 1]
        valid = recs['r2_bin'] > 0
        ax.scatter(recs.loc[valid, 'r2_bin'], recs.loc[valid, 'r2_tree'],
                   alpha=0.3, s=15, color=BLUE)
        lim = [0, max(recs['r2_tree'].max(), recs.loc[valid, 'r2_bin'].max())]
        ax.plot(lim, lim, color=RED, linewidth=1.5, linestyle='--')
        ax.set_xlabel('Bin ceiling'); ax.set_ylabel('Tree ceiling')
        ax.set_title(f'{title} — methods ({valid.sum()} pos bin)', fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'ceiling_comparison.png'), dpi=130, bbox_inches='tight')
    plt.close()
    print(f'\n→ ceiling_comparison.png')


if __name__ == '__main__':
    main()