"""
swing_models_via_r.py
=====================
All four swing evaluation models using lme4 via subprocess (no rpy2 dependency).

Usage
-----
  python swing_models_via_r.py [timing] [tilt] [la] [ev]
  
  Runs only the specified models. If no arguments, runs all four.
  Examples:
    python swing_models_via_r.py timing tilt    # only timing and tilt
    python swing_models_via_r.py la ev          # only contact models
    python swing_models_via_r.py timing         # only timing

Models
------
1. TIMING:  Original nested formulation from BPS article
            int_y ~ 1 + (release_speed_c + plate_x_bat_flip | batter:pitch_type)
            (release_speed_c substitutes for deprecated tf)

2. TILT:    swing_path_int ~ 1 + (plate_z | batter:pitch_type)

3. LA:      OLS (random effects added negligible variance)
            Base: launch_angle ~ attack_angle + pfx_z + plate_z + intercept_y + 
                                 swing_path_tilt + C(pitch_type)
            + interactions: attack_angle:pfx_z + attack_angle:plate_z

4. EV:      OLS (random effects added negligible variance)
            Base: exit_velo ~ bat_speed + release_speed_c + attack_angle +
                              intercept_y + intercept_x + swing_path_tilt + C(pitch_type)
            + interaction: bat_speed:attack_angle (if bat_speed available)

Pitch family: fastball {FF,SI,FC,FT}, breaking {CU,SL,ST,SV,KC}, offspeed {CH,FS,FO,EP,KN,SC}

Requires: R with lme4 installed
"""

import os, glob, warnings, itertools, subprocess, tempfile, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from pybaseball import chadwick_register

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "final_models")
os.makedirs(OUT_DIR, exist_ok=True)

INTERCEPT_X = 'intercept_ball_minus_batter_pos_x_inches'
INTERCEPT_Y = 'intercept_ball_minus_batter_pos_y_inches'
MISS        = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
FOUL        = {'foul', 'foul_tip', 'foul_bunt'}
IN_PLAY     = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
ALL_SWINGS  = IN_PLAY | FOUL | MISS

# description values used for stat calculations
CONTACT_EVENTS = {'hit_into_play', 'foul', 'swinging_strike'}
HIT_EVENTS     = {'single', 'double', 'triple', 'home_run'}
BABIP_EXCLUDE  = {}       # HR excluded from BABIP numerator # removing this I think it should count for us even though the stat doesnt include HRs
BABIP_DENOM_EXCLUDE = {}  # HR excluded from BABIP denominator

W_EV = 0.50
W_LA = 0.50

PITCH_FAMILY = {
    'FF': 'fastball', 'SI': 'fastball', 'FC': 'fastball', 'FT': 'fastball',
    'CU': 'breaking', 'SL': 'breaking', 'ST': 'breaking',
    'SV': 'breaking', 'KC': 'breaking',
    'CH': 'offspeed', 'FS': 'offspeed', 'FO': 'offspeed',
    'EP': 'offspeed', 'KN': 'offspeed', 'SC': 'offspeed',
}

sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE = '#2563EB'; GREEN = '#16A34A'; RED = '#DC2626'; GRAY = '#6B7280'; DARK = '#1F2937'


# ══════════════════════════════════════════════════════════════════════════════
# NAME CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def convert_names(ids, register=None):
    if register is None:
        register = chadwick_register()
    if type(ids) == int or type(ids) == float:
        names = (
            f"{register[register.key_mlbam==ids].name_first.values[0]} "
            f"{register[register.key_mlbam==ids].name_last.values[0]}"
        )
    else:
        skipped = []
        names = []
        for i in ids:
            try:
                names.append(
                    f"{register[register.key_mlbam==i].name_first.values[0]} "
                    f"{register[register.key_mlbam==i].name_last.values[0]}"
                )
            except IndexError:
                names.append('None')
                skipped.append(i)
        print(f"{len(skipped)}/{len(ids)} skipped")
    return names


# ══════════════════════════════════════════════════════════════════════════════
# SEASON STATS
# ══════════════════════════════════════════════════════════════════════════════

def compute_season_stats(df):
    """
    Compute per-batter season stats from raw pitch/swing data.

    Columns used
    ------------
    description : swing/contact classification per pitch
    events      : play outcome (single, double, field_out, etc.)
    launch_speed_angle : 6 = barrel

    Stats returned (one row per batter)
    ------------------------------------
    pa            : plate appearances (proxy: pitches where description signals PA end)
    batting_avg   : H / AB  (AB = PA - walks/HBP; approximated as balls_in_play + strikeouts)
    swing_rate    : swings / pitches seen
    contact_rate  : contacts (in-play + foul) / swings
    barrel_rate   : barrels / balls in play
    babip         : (H - HR) / (AB - K - HR + SF)  — approximated from available columns
    """
    print('\nComputing season stats...')

    # ── pitches seen per batter (all rows, not just swings) ──────────────────
    pitches_seen = df.groupby('batter').size().rename('pitches_seen')

    # ── swing events ─────────────────────────────────────────────────────────
    swing_mask   = df['description'].isin(ALL_SWINGS)
    swings       = df[swing_mask].groupby('batter').size().rename('n_swings')

    # ── contact: in-play + foul (anything that isn't a miss) ─────────────────
    contact_mask = df['description'].isin(IN_PLAY | FOUL)
    contacts     = df[contact_mask].groupby('batter').size().rename('n_contacts')

    # ── balls in play (excluding fouls) ──────────────────────────────────────
    bip_mask = df['description'].isin(IN_PLAY)
    bip      = df[bip_mask].groupby('batter').size().rename('n_bip')

    # ── hits: rows where events is a hit type ────────────────────────────────
    hit_mask = df['events'].isin(HIT_EVENTS)
    hits     = df[hit_mask].groupby('batter').size().rename('n_hits')

    # ── home runs ────────────────────────────────────────────────────────────
    hr_mask = df['events'] == 'home_run'
    hrs     = df[hr_mask].groupby('batter').size().rename('n_hr')

    # ── strikeouts: swinging_strike that ends AB (events == 'strikeout') ─────
    k_mask = df['events'] == 'strikeout'
    ks     = df[k_mask].groupby('batter').size().rename('n_k')

    # ── barrels: launch_speed_angle == 6, among balls in play ────────────────
    barrel_mask = bip_mask & (df['launch_speed_angle'] == 6)
    barrels     = df[barrel_mask].groupby('batter').size().rename('n_barrels')

    # ── assemble ─────────────────────────────────────────────────────────────
    stats = (
        pd.concat([pitches_seen, swings, contacts, bip, hits, hrs, ks, barrels], axis=1)
        .fillna(0)
        .astype(int)
    )

    # swing rate
    stats['swing_rate'] = stats['n_swings'] / stats['pitches_seen']

    # contact rate  (contacts / swings; 0 if no swings)
    stats['contact_rate'] = np.where(
        stats['n_swings'] > 0,
        stats['n_contacts'] / stats['n_swings'],
        np.nan
    )

    # batting average: H / (BIP + K)  — AB proxy
    stats['ab_proxy'] = stats['n_bip'] + stats['n_k']
    stats['batting_avg'] = np.where(
        stats['ab_proxy'] > 0,
        stats['n_hits'] / stats['ab_proxy'],
        np.nan
    )

    # barrel rate: barrels / BIP
    stats['barrel_rate'] = np.where(
        stats['n_bip'] > 0,
        stats['n_barrels'] / stats['n_bip'],
        np.nan
    )

    # BABIP: (H - HR) / (AB - K - HR)
    babip_num   = stats['n_hits'] - stats['n_hr']
    babip_denom = stats['ab_proxy'] - stats['n_k'] - stats['n_hr']
    stats['babip'] = np.where(
        babip_denom > 0,
        babip_num / babip_denom,
        np.nan
    )

    # keep only the derived stats (drop intermediate counts)
    stat_cols = ['batting_avg', 'swing_rate', 'contact_rate', 'barrel_rate', 'babip','ab_proxy']
    stats = stats[stat_cols].reset_index()
    stats['batter'] = stats['batter'].astype(str)

    print(f'  Season stats computed for {len(stats):,} batters')
    print(stats[stat_cols].describe().round(3).to_string())
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# 0. DATA PREP
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    df    = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f'Loaded {len(df):,} pitches')

    if 'plate_x_bat_flip' not in df.columns:
        df['plate_x_bat_flip'] = df['plate_x'] * df['stand'].map({'R': -1, 'L': 1}).fillna(1)

    df = df.dropna(subset=['pitch_type']).copy()

    # Standardize release_speed and plate_x_bat_flip for numerical stability
    rs_mean = df['release_speed'].mean()
    rs_std  = df['release_speed'].std()
    df['release_speed_c'] = (df['release_speed'] - rs_mean) / rs_std

    px_mean = df['plate_x_bat_flip'].mean()
    px_std  = df['plate_x_bat_flip'].std()
    df['plate_x_bat_flip'] = (df['plate_x_bat_flip'] - px_mean) / px_std

    print(f'release_speed standardized (mean={rs_mean:.2f}, std={rs_std:.2f})')
    print(f'plate_x_bat_flip standardized (mean={px_mean:.4f}, std={px_std:.4f})')

    df['pitch_family'] = df['pitch_type'].map(PITCH_FAMILY).fillna('other')

    for col in ['batter', 'pitch_type', 'pitch_family']:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ── attach batter names ───────────────────────────────────────────────────
    print('Loading Chadwick register for name lookup...')
    try:
        register    = chadwick_register()
        batter_ids  = df['batter'].unique().tolist()
        # chadwick uses int keys; coerce safely
        batter_ids_int = []
        for b in batter_ids:
            try:
                batter_ids_int.append(int(b))
            except (ValueError, TypeError):
                batter_ids_int.append(None)

        valid_pairs = [(bid_str, bid_int)
                       for bid_str, bid_int in zip(batter_ids, batter_ids_int)
                       if bid_int is not None]

        id_strs  = [p[0] for p in valid_pairs]
        id_ints  = [p[1] for p in valid_pairs]
        name_list = convert_names(id_ints, register=register)

        name_map = dict(zip(id_strs, name_list))
        df['batter_name'] = df['batter'].map(name_map).fillna('Unknown')
        print(f'  Names resolved for {sum(n != "None" for n in name_list):,} / {len(name_list):,} batters')
    except Exception as e:
        print(f'  Name lookup failed ({e}); batter_name set to batter id')
        df['batter_name'] = df['batter']

    return df


# ══════════════════════════════════════════════════════════════════════════════
# R INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def run_lmer_via_r(df, outcome, formula, label, out_prefix):
    """
    Write df to CSV, generate R script, run lmer(), read results back.
    
    Returns dict: {label, formula, fitted, cond_r2, aic, singular, coefs}
    """
    # Write data
    data_path = os.path.join(OUT_DIR, f'_tmp_{label}_data.csv')
    df.to_csv(data_path, index=False)
    
    # Generate R script
    r_script = f'''
library(lme4)
library(performance)

df <- read.csv("{data_path}")

# Ensure grouping factors are factors
for (col in c("batter", "pitch_cluster", "pitch_type", "pitch_family")) {{
  if (col %in% names(df)) {{
    df[[col]] <- as.factor(df[[col]])
  }}
}}

# Fit model
cat("Fitting: {formula}\\n")
model <- lmer({formula}, data=df, REML=TRUE, control=lmerControl(optimizer="bobyqa"))

# Check singularity
is_singular <- isSingular(model)
cat("Singular:", is_singular, "\\n")

# Fitted values
fitted_vals <- fitted(model)

# R² (using performance package for proper mixed-model R²)
r2_vals <- tryCatch(
  r2(model),
  error = function(e) list(R2_marginal=NA, R2_conditional=NA)
)

# Handle both list and atomic vector returns
if (is.list(r2_vals)) {{
  marginal_r2 <- r2_vals$R2_marginal
  conditional_r2 <- r2_vals$R2_conditional
}} else {{
  # Fallback: compute manually
  var_fixed <- var(predict(model, re.form=NA))
  var_total <- var(df[["{outcome}"]])
  marginal_r2 <- var_fixed / var_total
  conditional_r2 <- 1 - var(residuals(model)) / var_total
}}

# AIC
aic_val <- AIC(model)

# Fixed effects summary
fe <- summary(model)$coefficients
cat("\\nFixed effects:\\n")
print(fe)

cat("\\nVariance components:\\n")
print(VarCorr(model))

cat("\\nModel summary:\\n")
cat("Marginal R²:", marginal_r2, "\\n")
cat("Conditional R²:", conditional_r2, "\\n")
cat("AIC:", aic_val, "\\n")

# Write outputs
write.csv(data.frame(fitted=fitted_vals), "{out_prefix}_fitted.csv", row.names=FALSE)
write.csv(fe, "{out_prefix}_coefs.csv", row.names=TRUE)
write.table(data.frame(marginal_r2=marginal_r2, conditional_r2=conditional_r2, 
                       aic=aic_val, singular=is_singular),
            "{out_prefix}_summary.csv", row.names=FALSE, sep=",")

# Full model summary to text file
sink("{out_prefix}_summary.txt")
print(summary(model))
sink()
'''
    
    r_path = os.path.join(OUT_DIR, f'_tmp_{label}_script.R')
    with open(r_path, 'w') as f:
        f.write(r_script)
    
    # Run R script
    print(f'  Running R: {label}')
    try:
        result = subprocess.run(['Rscript', r_path],
                                capture_output=True, text=True,
                                timeout=300, cwd=OUT_DIR)
        print(result.stdout)
        if result.returncode != 0:
            print(f'  R stderr: {result.stderr}')
            return None
    except Exception as e:
        print(f'  R execution failed: {e}')
        return None
    
    # Read results
    try:
        fitted_df = pd.read_csv(f'{out_prefix}_fitted.csv')
        coefs_df  = pd.read_csv(f'{out_prefix}_coefs.csv', index_col=0)
        summary   = pd.read_csv(f'{out_prefix}_summary.csv')
        
        fitted      = pd.Series(fitted_df['fitted'].values, index=df.index)
        cond_r2     = float(summary['conditional_r2'].iloc[0])
        marginal_r2 = float(summary['marginal_r2'].iloc[0])
        aic         = float(summary['aic'].iloc[0])
        singular    = bool(summary['singular'].iloc[0])
        
        # Clean up temp files
        for ext in ['_data.csv', '_script.R', '_fitted.csv', '_coefs.csv', '_summary.csv']:
            try: os.remove(os.path.join(OUT_DIR, f'_tmp_{label}{ext}'))
            except: pass
        
        return {
            'label': label, 'formula': formula, 'fitted': fitted,
            'cond_r2': cond_r2, 'marginal_r2': marginal_r2,
            'aic': aic, 'singular': singular, 'coefs': coefs_df,
        }
    except Exception as e:
        print(f'  Failed to read R outputs: {e}')
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SHAPLEY R²
# ══════════════════════════════════════════════════════════════════════════════

def compute_shapley_r2(df, outcome, predictor_pool):
    def ols_r2(preds):
        if not preds: return 0.0
        terms = [f'C({p})' if df[p].dtype == object else p for p in preds]
        try: return smf.ols(f'{outcome} ~ {" + ".join(terms)}', data=df).fit().rsquared
        except: return np.nan
    
    cache = {frozenset(c): ols_r2(list(c))
             for s in range(len(predictor_pool) + 1)
             for c in itertools.combinations(predictor_pool, s)}
    
    shapley = {}
    for p in predictor_pool:
        others = [q for q in predictor_pool if q != p]
        contrib = [cache.get(frozenset(s) | {p}, 0) - cache.get(frozenset(s), 0)
                   for sz in range(len(others) + 1)
                   for s in itertools.combinations(others, sz)]
        shapley[p] = np.mean([c for c in contrib if not np.isnan(c)])
    
    return pd.Series(shapley).sort_values(ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def summary_plots(result, df, outcome, fname_prefix):
    fitted = result['fitted']
    actual = df[outcome]
    resid  = actual - fitted
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'{result["label"]}', fontsize=13, fontweight='bold')
    
    # Resid vs fitted
    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.15, s=4, color=BLUE, rasterized=True)
    ax.axhline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Fitted'); ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Fitted', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Pred vs actual
    ax = axes[0, 1]
    lim = [min(actual.min(), fitted.min()), max(actual.max(), fitted.max())]
    ax.scatter(actual, fitted, alpha=0.15, s=4, color=BLUE, rasterized=True)
    ax.plot(lim, lim, color=RED, linewidth=1.2, linestyle='--')
    corr = np.corrcoef(actual, fitted)[0, 1]
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(f'Pred vs Actual (r={corr:.3f})', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Fixed-effect coefficients
    ax = axes[1, 0]
    coefs = result['coefs']
    preds = [i for i in coefs.index if i != '(Intercept)']
    if preds:
        y_pos = np.arange(len(preds))
        estimates = coefs.loc[preds, 'Estimate']
        se        = coefs.loc[preds, 'Std. Error']
        ci_lo, ci_hi = estimates - 1.96*se, estimates + 1.96*se
        ax.barh(y_pos, estimates, color=GREEN, alpha=0.7, height=0.5)
        ax.errorbar(estimates, y_pos, xerr=[estimates-ci_lo, ci_hi-estimates],
                    fmt='none', color=DARK, linewidth=1.5, capsize=4)
        ax.axvline(0, color=RED, linewidth=1, linestyle='--')
        ax.set_yticks(y_pos); ax.set_yticklabels(preds, fontsize=9)
        ax.set_xlabel('Coefficient'); ax.set_title('Fixed effects (95% CI)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Residual distribution
    ax = axes[1, 1]
    ax.hist(resid.clip(resid.quantile(0.005), resid.quantile(0.995)),
            bins=80, color=BLUE, edgecolor='white', alpha=0.85)
    ax.axvline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Residual'); ax.set_ylabel('Count')
    ax.set_title(f'Residuals (std={resid.std():.2f})', fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'{fname_prefix}_summary.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def comparison_plot(records, title, fname):
    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(max(10, len(df)*1.2), 5))
    x = np.arange(len(df))
    bars = ax.bar(x, df['cond_r2'], color=BLUE, edgecolor='white', alpha=0.85)
    for bar, v, sing in zip(bars, df['cond_r2'], df['singular']):
        label = f'{v:.4f}' + (' ⚠' if sing else '')
        ax.text(bar.get_x()+bar.get_width()/2, v+0.005, label, ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(df['label'], rotation=30, ha='right')
    ax.set_ylabel('Conditional R²'); ax.set_title(title, fontweight='bold')
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# 1. TIMING MODEL
# ══════════════════════════════════════════════════════════════════════════════

def run_timing_model(df):
    print('\n' + '='*70)
    print('TIMING MODEL (BPS original + platoon variant)')
    print('='*70)
    
    needed = [INTERCEPT_Y, 'release_speed_c', 'plate_x_bat_flip',
              'batter', 'pitch_type', 'stand', 'p_throws', 'description']
    sub = (df[df['description'].isin(ALL_SWINGS)]
           .dropna(subset=needed).rename(columns={INTERCEPT_Y: 'int_y'}).copy())
    
    sub['platoon'] = sub['stand'] + '_vs_' + sub['p_throws']
    
    print(f'n={len(sub):,}')
    print(f'Platoon:\n{sub["platoon"].value_counts().to_string()}')
    
    records = []
    results = {}
    
    # Base (original BPS)
    formula_base = 'int_y ~ 1 + (release_speed_c + plate_x_bat_flip | batter:pitch_type)'
    r_base = run_lmer_via_r(sub, 'int_y', formula_base, 'timing_base',
                            os.path.join(OUT_DIR, '_r_timing_base'))
    if r_base:
        results['base'] = (r_base, sub)
        records.append({'label': 'base', 'cond_r2': r_base['cond_r2'],
                        'aic': r_base['aic'], 'singular': r_base['singular']})
    
    if len(records) > 1:
        print(f'\nComparison:')
        print(pd.DataFrame(records).to_string(index=False))
    
    shapley = compute_shapley_r2(sub, 'int_y', ['release_speed_c', 'plate_x_bat_flip'])
    print(f'\nShapley R²:\n{shapley.to_string()}')
    
    best_key = min(results, key=lambda k: results[k][0]['aic'])
    summary_plots(results[best_key][0], sub, 'int_y', f'timing_{best_key}')
    
    scores = (sub.assign(resid=(sub['int_y'] - results[best_key][0]['fitted']).abs())
              .groupby('batter')['resid']
              .agg(mean='mean', std='std')
              .assign(timing_score=lambda x: 1.0 / x['mean']))
    scores.reset_index().to_csv(os.path.join(OUT_DIR, 'timing_scores.csv'), index=False)
    
    return results, shapley, scores


# ══════════════════════════════════════════════════════════════════════════════
# 2. TILT MODEL
# ══════════════════════════════════════════════════════════════════════════════

def run_tilt_model(df):
    print('\n' + '='*70)
    print('TILT MODEL')
    print('='*70)
    
    needed = ['swing_path_tilt', 'plate_z', 'batter',
              'pitch_type', 'description']
    sub = (df[df['description'].isin(MISS)]
           .dropna(subset=needed)
           .rename(columns={'swing_path_tilt': 'swing_path_int'}).copy())
    print(f'n={len(sub):,}')
    
    formula = 'swing_path_int ~ 1 + (plate_z | batter:pitch_type)'
    label   = 'tilt_pitch_type'
    out_pfx = os.path.join(OUT_DIR, f'_r_{label}')
    
    r = run_lmer_via_r(sub, 'swing_path_int', formula, label, out_pfx)
    if not r:
        return {}, pd.Series()
    
    results = {label: (r, sub)}
    
    shapley = compute_shapley_r2(sub, 'swing_path_int', ['plate_z'])
    print(f'\nShapley R²:\n{shapley.to_string()}')
    
    summary_plots(r, sub, 'swing_path_int', label)
    
    scores = (sub.assign(resid=(sub['swing_path_int'] - r['fitted']).abs())
              .groupby('batter')['resid']
              .agg(mean='mean', std='std')
              .assign(tilt_score=lambda x: 1.0 / x['mean']))
    scores.reset_index().to_csv(os.path.join(OUT_DIR, 'tilt_scores.csv'), index=False)
    
    return results, shapley


# ══════════════════════════════════════════════════════════════════════════════
# 3. LA MODEL
# ══════════════════════════════════════════════════════════════════════════════

def run_la_model(df):
    print('\n' + '='*70)
    print('LAUNCH ANGLE MODEL')
    print('='*70)
    
    needed = ['launch_angle', INTERCEPT_Y, 'attack_angle', 'pfx_z',
              'plate_z', 'swing_path_tilt', 'batter', 'pitch_type', 'description']
    sub = (df[df['description'].isin(IN_PLAY)]
           .dropna(subset=needed)
           .rename(columns={INTERCEPT_Y: 'intercept_y'}).copy())
    print(f'n={len(sub):,}')
    
    print('\nRandom effects added negligible variance (marginal ≈ conditional).')
    print('Using OLS with and without interaction terms.')
    
    base = 'launch_angle ~ attack_angle + pfx_z + plate_z + intercept_y + swing_path_tilt + C(pitch_type)'
    with_int = f'{base} + attack_angle:pfx_z + attack_angle:plate_z'
    
    records = []
    results = {}
    
    for label, formula in [('LA_base', base), ('LA_interactions', with_int)]:
        print(f'\n  [{label}]')
        ols = smf.ols(formula, data=sub).fit()
        print(f'  R²={ols.rsquared:.4f}  AIC={ols.aic:.1f}')
        
        fitted = pd.Series(ols.fittedvalues, index=sub.index)
        results[label] = {'label': label, 'formula': formula,
                          'fitted': fitted, 'cond_r2': ols.rsquared,
                          'marginal_r2': ols.rsquared, 'aic': ols.aic,
                          'singular': False,
                          'coefs': ols.params.to_frame('Estimate').assign(**{
                              'Std. Error': ols.bse.values})}
        records.append({'label': label, 'cond_r2': ols.rsquared,
                        'aic': ols.aic, 'singular': False})
    
    shapley = compute_shapley_r2(sub, 'launch_angle',
                                  ['attack_angle', 'pfx_z', 'plate_z',
                                   'intercept_y', 'swing_path_tilt'])
    print(f'\nShapley R²:\n{shapley.to_string()}')
    
    comparison_plot(records, 'LA model variants', 'la_comparison.png')
    pd.DataFrame(records).to_csv(os.path.join(OUT_DIR, 'la_comparison.csv'), index=False)
    
    best_key = max(results, key=lambda k: results[k]['cond_r2'])
    summary_plots(results[best_key], sub, 'launch_angle', f'la_{best_key}')
    
    return results, shapley


# ══════════════════════════════════════════════════════════════════════════════
# 4. EV MODEL
# ══════════════════════════════════════════════════════════════════════════════

def run_ev_model(df):
    print('\n' + '='*70)
    print('EXIT VELOCITY MODEL')
    print('='*70)
    
    needed = ['launch_speed', INTERCEPT_X, INTERCEPT_Y, 'attack_angle',
              'swing_path_tilt', 'release_speed_c', 'pitch_type', 'description']
    has_bs = 'bat_speed' in df.columns and df['bat_speed'].notna().sum() > 1000
    if has_bs: needed.append('bat_speed')
    
    sub = (df[df['description'].isin(IN_PLAY)]
           .dropna(subset=needed)
           .rename(columns={INTERCEPT_X: 'intercept_x',
                            INTERCEPT_Y: 'intercept_y',
                            'launch_speed': 'exit_velo'}).copy())
    print(f'n={len(sub):,}, bat_speed={has_bs}')
    
    print('\nRandom effects added negligible variance (marginal ≈ conditional).')
    print('Using OLS with and without interaction terms.')
    
    bs_term = 'bat_speed + ' if has_bs else ''
    base    = f'exit_velo ~ {bs_term}release_speed_c + attack_angle + intercept_y + intercept_x + swing_path_tilt + C(pitch_type)'
    
    if has_bs:
        with_int = f'{base} + bat_speed:attack_angle'
    else:
        with_int = base
    
    records = []
    results = {}
    
    for label, formula in [('EV_base', base),
                           ('EV_interactions', with_int) if has_bs else (None, None)]:
        if label is None:
            continue
        print(f'\n  [{label}]')
        ols = smf.ols(formula, data=sub).fit()
        print(f'  R²={ols.rsquared:.4f}  AIC={ols.aic:.1f}')
        
        fitted = pd.Series(ols.fittedvalues, index=sub.index)
        results[label] = {'label': label, 'formula': formula,
                          'fitted': fitted, 'cond_r2': ols.rsquared,
                          'marginal_r2': ols.rsquared, 'aic': ols.aic,
                          'singular': False,
                          'coefs': ols.params.to_frame('Estimate').assign(**{
                              'Std. Error': ols.bse.values})}
        records.append({'label': label, 'cond_r2': ols.rsquared,
                        'aic': ols.aic, 'singular': False})
    
    pred_pool = (['bat_speed'] if has_bs else []) + \
                ['release_speed_c', 'attack_angle', 'intercept_y',
                 'intercept_x', 'swing_path_tilt']
    shapley = compute_shapley_r2(sub, 'exit_velo', pred_pool)
    print(f'\nShapley R²:\n{shapley.to_string()}')
    
    comparison_plot(records, 'EV model variants', 'ev_comparison.png')
    pd.DataFrame(records).to_csv(os.path.join(OUT_DIR, 'ev_comparison.csv'), index=False)
    
    best_key = max(results, key=lambda k: results[k]['cond_r2'])
    summary_plots(results[best_key], sub, 'exit_velo', f'ev_{best_key}')
    
    return results, shapley


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONTACT SCORES
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_scores(timing_res, tilt_res, la_res, ev_res, df):
    # Best models
    best_timing = max(timing_res.values(), key=lambda x: x[0]['cond_r2'])[0]
    best_tilt   = tilt_res['tilt_pitch_type'][0]
    best_la     = max(la_res.values(),     key=lambda x: x['cond_r2'])
    best_ev     = max(ev_res.values(),     key=lambda x: x['cond_r2'])
    
    # Read saved scores
    timing_scores = pd.read_csv(os.path.join(OUT_DIR, 'timing_scores.csv'))
    tilt_scores   = pd.read_csv(os.path.join(OUT_DIR, 'tilt_scores.csv'))
    
    timing_scores['batter'] = timing_scores['batter'].astype(str)
    tilt_scores['batter']   = tilt_scores['batter'].astype(str)
    
    # Contact score from EV + LA residuals
    ev_df = (df[df['description'].isin(IN_PLAY)]
             .dropna(subset=['launch_speed', 'batter'])
             .rename(columns={'launch_speed': 'exit_velo'}))
    la_df = (df[df['description'].isin(IN_PLAY)]
             .dropna(subset=['launch_angle', 'batter']))
    
    ev_df = ev_df.loc[best_ev['fitted'].index].copy()
    la_df = la_df.loc[best_la['fitted'].index].copy()
    
    ev_df['batter'] = ev_df['batter'].astype(str)
    la_df['batter'] = la_df['batter'].astype(str)
    
    ev_resid = ev_df['exit_velo'] - best_ev['fitted']
    la_resid = (la_df['launch_angle'] - best_la['fitted']).abs()
    
    ev_z = ev_resid / ev_resid.std()
    la_z = la_resid / la_resid.std()
    
    ev_scores = pd.DataFrame({'batter': ev_df['batter'], 'z_ev': ev_z}).groupby('batter')['z_ev'].mean()
    la_scores = pd.DataFrame({'batter': la_df['batter'], 'z_la': la_z}).groupby('batter')['z_la'].mean()
    
    contact = pd.DataFrame({
        'ev_z': ev_scores, 'la_z': la_scores,
        'contact_score': W_EV * ev_scores - W_LA * la_scores,
    }).reset_index()

    # ── batter name lookup ────────────────────────────────────────────────────
    # Build a batter_id → batter_name map from df (already resolved in prepare_data)
    name_map = (
        df[['batter', 'batter_name']]
        .drop_duplicates(subset='batter')
        .set_index('batter')['batter_name']
    )

    # ── season stats ──────────────────────────────────────────────────────────
    season_stats = compute_season_stats(df)

    # ── assemble final table ──────────────────────────────────────────────────
    all_scores = (
        timing_scores[['batter', 'timing_score']]
        .merge(tilt_scores[['batter', 'tilt_score']],   on='batter', how='outer')
        .merge(contact[['batter', 'contact_score']],    on='batter', how='outer')
        .merge(season_stats,                             on='batter', how='left')
    )

    # Add name column right after batter id
    all_scores.insert(1, 'batter_name', all_scores['batter'].map(name_map).fillna('Unknown'))

    path = os.path.join(OUT_DIR, 'all_batter_scores.csv')
    all_scores.to_csv(path, index=False)
    print(f'\n→ All scores: {path}')
    print(all_scores.describe().round(3).to_string())
    
    return all_scores


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    valid_models = {'timing', 'tilt', 'la', 'ev'}
    requested    = set(sys.argv[1:]) if len(sys.argv) > 1 else valid_models
    invalid      = requested - valid_models
    
    if invalid:
        print(f'Unknown models: {invalid}. Valid options: {valid_models}')
        sys.exit(1)
    
    if not requested:
        requested = valid_models
    
    print(f'Running models: {", ".join(sorted(requested))}')
    
    df = prepare_data()
    
    results = {}
    
    if 'timing' in requested:
        timing_res, timing_shap, timing_scores = run_timing_model(df)
        results['timing'] = timing_res
    else:
        print('\nSkipping timing model')
        timing_res = {}
    
    if 'tilt' in requested:
        tilt_res, tilt_shap = run_tilt_model(df)
        results['tilt'] = tilt_res
    else:
        print('\nSkipping tilt model')
        tilt_res = {}
    
    if 'la' in requested:
        la_res, la_shap = run_la_model(df)
        results['la'] = la_res
    else:
        print('\nSkipping LA model')
        la_res = {}
    
    if 'ev' in requested:
        ev_res, ev_shap = run_ev_model(df)
        results['ev'] = ev_res
    else:
        print('\nSkipping EV model')
        ev_res = {}
    
    if requested == valid_models:
        all_scores = compute_all_scores(timing_res, tilt_res, la_res, ev_res, df)
    else:
        print('\nSkipping combined scores (not all models run)')
    
    print(f'\nAll outputs in {OUT_DIR}')


if __name__ == '__main__':
    main()