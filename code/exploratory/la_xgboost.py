"""
la_xgboost_comparison.py
========================
Replicate the Medium article's XGBoost approach to LA prediction.

Tests whether the 0.77 R² claim is achievable with:
  - Three separate XGBoost models (one per pitch family)
  - Pitch characteristics as predictors (per their variable lists)
  - Hyperparameter tuning via optuna or grid search

Compares to the OLS hierarchical approach to isolate whether XGBoost's
nonlinear fitting explains the R² gap.

Requires: pip install xgboost optuna
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "la_xgboost")
os.makedirs(OUT_DIR, exist_ok=True)

IN_PLAY = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}

PITCH_FAMILY = {
    'FF': 'fastball', 'SI': 'fastball', 'FC': 'fastball', 'FT': 'fastball',
    'CU': 'breaking', 'SL': 'breaking', 'ST': 'breaking',
    'SV': 'breaking', 'KC': 'breaking',
    'CH': 'offspeed', 'FS': 'offspeed', 'FO': 'offspeed',
    'EP': 'offspeed', 'KN': 'offspeed', 'SC': 'offspeed',
}
FASTBALL_TYPES = ['FF', 'SI', 'FC', 'FT']

sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE = '#2563EB'; GREEN = '#16A34A'; RED = '#DC2626'


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREP
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    df    = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    df = df.dropna(subset=['pitch_type', 'pitcher']).copy()
    df['pitch_family'] = df['pitch_type'].map(PITCH_FAMILY).fillna('other')
    df['pfx_x_adj']    = df['pfx_x'].abs()
    
    # Pitcher fastball baselines
    fb_stats = (df[df['pitch_type'].isin(FASTBALL_TYPES)]
                .groupby('pitcher')
                .agg(fb_pfx_z=('pfx_z', 'mean'),
                     fb_speed=('release_speed', 'mean'),
                     fb_count=('pitch_type', 'count')))
    fb_stats = fb_stats[fb_stats['fb_count'] >= 20]
    
    df = df.merge(fb_stats[['fb_pfx_z', 'fb_speed']],
                  left_on='pitcher', right_index=True, how='left')
    df['vmov_diff'] = df['pfx_z'] - df['fb_pfx_z']
    df['velo_diff'] = df['release_speed'] - df['fb_speed']
    
    print(f'Loaded {len(df):,} pitches')
    print(f'{df[df["description"].isin(IN_PLAY)].shape[0]:,} in-play')
    print(f'Pitch family counts:\n{df["pitch_family"].value_counts()}')
    
    return df


# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST MODELS (one per family)
# ══════════════════════════════════════════════════════════════════════════════

def train_xgb_family(fam_df, family_name, predictors, tune=False):
    """
    Train XGBoost on one pitch family.
    If tune=True, run optuna hyperparameter search; else use defaults.
    """
    needed = ['launch_angle'] + predictors
    fam_df = fam_df.dropna(subset=needed).copy()
    
    if len(fam_df) < 500:
        print(f'  {family_name}: insufficient data ({len(fam_df)}) — skip')
        return None
    
    X = fam_df[predictors].values
    y = fam_df['launch_angle'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    if tune:
        print(f'  {family_name}: tuning hyperparameters (this may take a few minutes)...')
        import optuna
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            }
            xgb = XGBRegressor(**params, random_state=42, n_jobs=-1)
            xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                    verbose=False)
            return r2_score(y_test, xgb.predict(X_test))
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        best_params = study.best_params
        print(f'    Best params: {best_params}')
    else:
        best_params = {
            'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 200,
            'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
        }
    
    xgb = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred_train = xgb.predict(X_train)
    y_pred_test  = xgb.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test  = r2_score(y_test,  y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    # Feature importance
    importance = pd.Series(xgb.feature_importances_, index=predictors).sort_values(ascending=False)
    
    print(f'  {family_name}: n={len(fam_df):,}  '
          f'R² train={r2_train:.4f}  test={r2_test:.4f}  MAE={mae_test:.2f}°')
    print(f'    Feature importance: ' +
          ', '.join([f'{k}={v:.3f}' for k, v in importance.head(3).items()]))
    
    # Fit on full data for final model
    xgb_full = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    xgb_full.fit(X, y, verbose=False)
    
    return {
        'model': xgb_full,
        'predictors': predictors,
        'r2_train': r2_train,
        'r2_test': r2_test,
        'mae_test': mae_test,
        'importance': importance,
        'n': len(fam_df),
    }


def run_xgb_hierarchical(df, tune=False):
    """
    Three XGBoost models, one per pitch family.
    Returns fitted values for all in-play contact aligned with df index.
    """
    print('\n' + '='*70)
    print('XGBOOST HIERARCHICAL LA MODEL')
    print('='*70)
    
    sub = df[df['description'].isin(IN_PLAY)].copy()
    
    family_specs = {
        'fastball': ['pfx_z', 'plate_z', 'plate_x', 'pfx_x_adj'],
        'offspeed': ['plate_z', 'pfx_z', 'plate_x', 'vmov_diff',
                     'release_speed', 'velo_diff', 'pfx_x_adj'],
        'breaking': ['plate_z', 'pfx_z', 'plate_x', 'release_speed',
                     'vmov_diff', 'pfx_x_adj'],
    }
    
    results = {}
    all_fitted = pd.Series(index=sub.index, dtype=float)
    
    for family, predictors in family_specs.items():
        fam_df = sub[sub['pitch_family'] == family]
        r = train_xgb_family(fam_df, family, predictors, tune=tune)
        if r:
            results[family] = r
            # Predict on full family data
            X_full = fam_df[predictors].dropna()
            fitted = r['model'].predict(X_full.values)
            all_fitted.loc[X_full.index] = fitted
    
    # Combined R² across all families
    valid = all_fitted.dropna().index
    r2_combined = r2_score(sub.loc[valid, 'launch_angle'],
                           all_fitted.loc[valid])
    mae_combined = mean_absolute_error(sub.loc[valid, 'launch_angle'],
                                        all_fitted.loc[valid])
    
    print(f'\n  COMBINED XGBoost: n={len(valid):,}  R²={r2_combined:.4f}  MAE={mae_combined:.2f}°')
    
    # Compare to OLS
    print('\n  Comparison to OLS (same predictors):')
    ols_fitted = pd.Series(index=sub.index, dtype=float)
    for family, predictors in family_specs.items():
        fam_df = sub[sub['pitch_family'] == family].dropna(
            subset=['launch_angle'] + predictors)
        if len(fam_df) < 100:
            continue
        formula = f'launch_angle ~ {" + ".join(predictors)}'
        ols = smf.ols(formula, data=fam_df).fit()
        ols_fitted.loc[fam_df.index] = ols.fittedvalues
        print(f'    {family}: OLS R²={ols.rsquared:.4f}  XGB R²={results[family]["r2_test"]:.4f}')
    
    ols_valid = ols_fitted.dropna().index
    r2_ols = r2_score(sub.loc[ols_valid, 'launch_angle'], ols_fitted.loc[ols_valid])
    print(f'    Combined OLS: R²={r2_ols:.4f}  vs  XGB: R²={r2_combined:.4f}')
    print(f'    XGB improvement: +{r2_combined - r2_ols:.4f}')
    
    # Save results
    pd.DataFrame({
        'family': list(results.keys()),
        'r2_train': [r['r2_train'] for r in results.values()],
        'r2_test':  [r['r2_test'] for r in results.values()],
        'mae_test': [r['mae_test'] for r in results.values()],
        'n': [r['n'] for r in results.values()],
    }).to_csv(os.path.join(OUT_DIR, 'xgb_family_comparison.csv'), index=False)
    
    # Feature importance plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (family, r) in zip(axes, results.items()):
        imp = r['importance'].head(10)
        ax.barh(range(len(imp)), imp.values, color=BLUE, edgecolor='white', alpha=0.85)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp.index, fontsize=9)
        ax.set_xlabel('Importance')
        ax.set_title(f'{family.capitalize()}\nR² test={r["r2_test"]:.3f}',
                     fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    fig.suptitle('XGBoost feature importance by pitch family',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'xgb_importance.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')
    
    return results, all_fitted, r2_combined


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_xgb_vs_ols(df, xgb_fitted, ols_fitted):
    """Side-by-side pred vs actual for XGB and OLS."""
    valid_xgb = xgb_fitted.dropna().index
    valid_ols = ols_fitted.dropna().index
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    for ax, fitted, label, color in [
        (axes[0], xgb_fitted.loc[valid_xgb], 'XGBoost', GREEN),
        (axes[1], ols_fitted.loc[valid_ols], 'OLS',     BLUE),
    ]:
        actual = df.loc[fitted.index, 'launch_angle']
        lim = [min(actual.min(), fitted.min()), max(actual.max(), fitted.max())]
        ax.scatter(actual, fitted, alpha=0.1, s=3, color=color, rasterized=True)
        ax.plot(lim, lim, color=RED, linewidth=1.5, linestyle='--', label='y=x')
        corr = np.corrcoef(actual, fitted)[0, 1]
        r2   = r2_score(actual, fitted)
        ax.set_xlabel('Actual LA'); ax.set_ylabel('Predicted LA')
        ax.set_title(f'{label}\nR²={r2:.3f}, r={corr:.3f}', fontweight='bold')
        ax.legend(); ax.grid(alpha=0.3)
    
    fig.suptitle('Predicted vs Actual: XGBoost vs OLS (hierarchical by pitch family)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'xgb_vs_ols.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df = prepare_data()
    
    # XGBoost hierarchical
    # Set tune=True to run hyperparameter search (slow, ~5-10 min per family)
    xgb_results, xgb_fitted, xgb_r2 = run_xgb_hierarchical(df, tune=False)
    
    # OLS for comparison (recompute to align with XGB indices)
    print('\n' + '='*70)
    print('OLS COMPARISON (same data)')
    print('='*70)
    
    sub = df[df['description'].isin(IN_PLAY)].copy()
    family_specs = {
        'fastball': ['pfx_z', 'plate_z', 'plate_x', 'pfx_x_adj'],
        'offspeed': ['plate_z', 'pfx_z', 'plate_x', 'vmov_diff',
                     'release_speed', 'velo_diff', 'pfx_x_adj'],
        'breaking': ['plate_z', 'pfx_z', 'plate_x', 'release_speed',
                     'vmov_diff', 'pfx_x_adj'],
    }
    
    ols_fitted = pd.Series(index=sub.index, dtype=float)
    for family, predictors in family_specs.items():
        fam_df = sub[sub['pitch_family'] == family].dropna(
            subset=['launch_angle'] + predictors)
        if len(fam_df) < 100:
            continue
        formula = f'launch_angle ~ {" + ".join(predictors)}'
        ols = smf.ols(formula, data=fam_df).fit()
        ols_fitted.loc[fam_df.index] = ols.fittedvalues
        print(f'  {family}: OLS R²={ols.rsquared:.4f}')
    
    ols_valid = ols_fitted.dropna().index
    ols_r2 = r2_score(sub.loc[ols_valid, 'launch_angle'], ols_fitted.loc[ols_valid])
    print(f'  Combined OLS: R²={ols_r2:.4f}')
    
    # Comparative plot
    plot_xgb_vs_ols(sub, xgb_fitted, ols_fitted)
    
    print(f'\n{"="*70}')
    print('SUMMARY')
    print('='*70)
    print(f'XGBoost R²: {xgb_r2:.4f}')
    print(f'OLS R²:     {ols_r2:.4f}')
    print(f'Difference: {xgb_r2 - ols_r2:.4f}')
    
    if xgb_r2 < 0.3:
        print('\n⚠ WARNING: XGBoost R² is much lower than Medium article claim (0.77).')
        print('  Possible explanations:')
        print('  - Missing critical variables (rel_height, others not documented)')
        print('  - Sample filtering differences (they may use different quality filters)')
        print('  - Data year/version differences')
        print('  - The 0.77 claim may be on a validation set with leakage')
    
    print(f'\nAll outputs in {OUT_DIR}')


if __name__ == '__main__':
    main()