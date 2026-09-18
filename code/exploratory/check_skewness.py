"""
check_outcome_skewness.py
=========================
Fit skew-normal distributions to all four swing model outcomes and
test whether skewness (α) is systematically non-zero.

Outcomes tested:
  - int_y (timing model)
  - swing_path_tilt (tilt model)  
  - launch_angle (LA model)
  - exit_velo (EV model)

For each:
  - Fit skew-normal via MLE on the full outcome distribution
  - Report α (skewness), test statistic, visual comparison to normal
  - Fit per-batter and report distribution of α across batters
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "skewness_check")
os.makedirs(OUT_DIR, exist_ok=True)

INTERCEPT_Y = 'intercept_ball_minus_batter_pos_y_inches'
MISS        = {'swinging_strike', 'swinging_strike_blocked', 'missed_bunt'}
FOUL        = {'foul', 'foul_tip', 'foul_bunt'}
IN_PLAY     = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
ALL_SWINGS  = IN_PLAY | FOUL | MISS

sns.set_theme(style='whitegrid', font_scale=1.1)
BLUE = '#2563EB'; GREEN = '#16A34A'; RED = '#DC2626'; GRAY = '#6B7280'


def load_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    df    = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df


def analyze_skewness(data, name, description):
    """
    Fit skew-normal to data, test α ≠ 0, plot comparison to normal.
    Returns dict of results.
    """
    data = data.dropna()
    
    # Fit skew-normal
    alpha, loc, scale = stats.skewnorm.fit(data)
    fitted_dist = stats.skewnorm(alpha, loc=loc, scale=scale)
    
    # Fit normal for comparison
    mu, sigma = stats.norm.fit(data)
    
    # Empirical skewness (moment-based)
    empirical_skew = stats.skew(data)
    
    print(f'\n{name} ({description}):')
    print(f'  n={len(data):,}')
    print(f'  Empirical: mean={data.mean():.3f}, std={data.std():.3f}, skew={empirical_skew:.3f}')
    print(f'  Skew-normal fit: α={alpha:.3f}, ξ={loc:.3f}, ω={scale:.3f}')
    print(f'  Normal fit:      μ={mu:.3f}, σ={sigma:.3f}')
    print(f'  |α| interpretation: {"STRONG" if abs(alpha) > 3 else "moderate" if abs(alpha) > 1 else "weak"} skewness')
    
    # Likelihood ratio test: skew-normal vs normal
    # LLR = 2 * (loglik_skewnorm - loglik_normal)
    # df = 1 (one extra parameter: alpha)
    ll_skew   = np.sum(stats.skewnorm.logpdf(data, alpha, loc=loc, scale=scale))
    ll_normal = np.sum(stats.norm.logpdf(data, mu, sigma))
    lr_stat   = 2 * (ll_skew - ll_normal)
    p_val     = stats.chi2.sf(lr_stat, df=1)
    
    print(f'  Likelihood ratio test (skew-normal vs normal):')
    print(f'    LR statistic={lr_stat:.1f}, p={p_val:.4e}')
    print(f'    → Skew-normal {"significantly better" if p_val < 0.001 else "not meaningfully better"}')
    
    return {
        'outcome': name,
        'n': len(data),
        'alpha': alpha,
        'empirical_skew': empirical_skew,
        'lr_stat': lr_stat,
        'p_value': p_val,
        'loc': loc,
        'scale': scale,
        'mu_normal': mu,
        'sigma_normal': sigma,
    }


def plot_comparison(data, result, name):
    """Four-panel diagnostic: histogram + fitted PDFs + Q-Q plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{name}  —  Skewness diagnostic', fontsize=14, fontweight='bold')
    
    # Panel 1: histogram with fitted PDFs overlaid
    ax = axes[0, 0]
    data_clip = data.clip(data.quantile(0.01), data.quantile(0.99))
    ax.hist(data_clip, bins=60, density=True, color=BLUE,
            edgecolor='white', alpha=0.6, label='Data')
    
    x = np.linspace(data_clip.min(), data_clip.max(), 200)
    pdf_skew   = stats.skewnorm.pdf(x, result['alpha'],
                                     loc=result['loc'], scale=result['scale'])
    pdf_normal = stats.norm.pdf(x, result['mu_normal'], result['sigma_normal'])
    
    ax.plot(x, pdf_skew, color=GREEN, linewidth=2.5,
            label=f'Skew-normal (α={result["alpha"]:.2f})')
    ax.plot(x, pdf_normal, color=RED, linewidth=2, linestyle='--',
            label=f'Normal')
    
    ax.set_xlabel(name); ax.set_ylabel('Density')
    ax.set_title('Histogram with fitted distributions', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 2: Q-Q plot vs normal
    ax = axes[0, 1]
    stats.probplot(data, dist='norm', plot=ax)
    ax.set_title('Q-Q vs Normal', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Panel 3: Q-Q plot vs fitted skew-normal
    ax = axes[1, 0]
    # Generate quantiles from fitted skew-normal
    sorted_data = np.sort(data)
    theoretical = stats.skewnorm.ppf(np.linspace(0.01, 0.99, len(sorted_data)),
                                      result['alpha'], loc=result['loc'],
                                      scale=result['scale'])
    ax.scatter(theoretical, sorted_data, alpha=0.3, s=4, color=BLUE, rasterized=True)
    lim = [min(theoretical.min(), sorted_data.min()),
           max(theoretical.max(), sorted_data.max())]
    ax.plot(lim, lim, color=RED, linewidth=1.5, linestyle='--')
    ax.set_xlabel('Theoretical quantiles (skew-normal)')
    ax.set_ylabel('Sample quantiles')
    ax.set_title('Q-Q vs Skew-Normal', fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Panel 4: summary stats table
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f'''
Outcome: {name}
n = {result["n"]:,}

Empirical moments:
  mean  = {data.mean():.3f}
  std   = {data.std():.3f}
  skew  = {result["empirical_skew"]:.3f}

Skew-normal fit:
  α (shape)    = {result["alpha"]:.3f}
  ξ (location) = {result["loc"]:.3f}
  ω (scale)    = {result["scale"]:.3f}

Normal fit:
  μ = {result["mu_normal"]:.3f}
  σ = {result["sigma_normal"]:.3f}

Likelihood ratio test:
  LR statistic = {result["lr_stat"]:.1f}
  p-value = {result["p_value"]:.2e}
  
Conclusion: {"|α| > 3" if abs(result["alpha"]) > 3 else "|α| < 3"}
  → {"STRONG" if abs(result["alpha"]) > 3 else "Weak"} skewness
'''
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f'skewness_{name.lower().replace(" ", "_")}.png')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df = load_data()
    
    print('='*70)
    print('OUTCOME SKEWNESS ANALYSIS')
    print('='*70)
    
    outcomes = [
        (df[df['description'].isin(ALL_SWINGS)][INTERCEPT_Y].rename('int_y'),
         'int_y', 'Timing model (all swings)'),
        
        (df[df['description'].isin(MISS)]['swing_path_tilt'],
         'swing_path_tilt', 'Tilt model (misses only)'),
        
        (df[df['description'].isin(IN_PLAY)]['launch_angle'],
         'launch_angle', 'LA model (in-play only)'),
        
        (df[df['description'].isin(IN_PLAY)]['launch_speed'].rename('exit_velo'),
         'exit_velo', 'EV model (in-play only)'),
    ]
    
    results = []
    for data, name, desc in outcomes:
        r = analyze_skewness(data, name, desc)
        results.append(r)
        plot_comparison(data, r, name)
    
    # Summary comparison table
    print('\n' + '='*70)
    print('SUMMARY COMPARISON')
    print('='*70)
    
    summary = pd.DataFrame(results)[['outcome', 'n', 'empirical_skew',
                                      'alpha', 'lr_stat', 'p_value']]
    summary['abs_alpha'] = summary['alpha'].abs()
    summary['strength'] = summary['abs_alpha'].apply(
        lambda a: 'STRONG' if a > 3 else 'moderate' if a > 1 else 'weak')
    
    print(summary.to_string(index=False))
    
    path = os.path.join(OUT_DIR, 'skewness_summary.csv')
    summary.to_csv(path, index=False)
    print(f'\n→ {path}')
    
    print('\nInterpretation:')
    print('  |α| > 3:  strong skewness → skew-normal model warranted')
    print('  |α| 1-3:  moderate skewness → minor improvement from skew-normal')
    print('  |α| < 1:  weak skewness → normal distribution adequate')
    
    print(f'\nAll plots in {OUT_DIR}')


if __name__ == '__main__':
    main()