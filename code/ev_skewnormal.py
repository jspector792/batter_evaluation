"""
ev_skew_brms.py
===============
Skew-normal multilevel model for exit velocity using brms (via subprocess).

Inspired by the "Swinging Fast and Slow" specification:
  exit_velo_i ~ SkewNormal(mu_i, sigma, alpha_i)

  mu_i    = mu0
            + gamma_pitcher[p_i]                         # random intercept: pitcher
            + gamma_batter[b_i]                          # random intercept: batter
            + (beta_bs  + gamma_bs_batter[b_i])  * bat_speed_i
            + (beta_ix  + gamma_ix_batter[b_i])  * intercept_x_i
            + (beta_iy  + gamma_iy_batter[b_i])  * intercept_y_i
            + (beta_aa  + gamma_aa_batter[b_i])  * attack_angle_i

  alpha_i = alpha0 + nu_batter[b_i]                     # skewness: random intercept per batter

Random effects:
  gamma_pitcher[p]  ~ N(0, sigma_p^2)
  [gamma_batter, gamma_bs_batter,
   gamma_ix_batter, gamma_iy_batter,
   gamma_aa_batter][b]  ~ MVN(0, Sigma)   (LKJ prior on correlations)
  nu_batter[b]      ~ N(0, tau_b^2)       (batter-level skewness shift)

Priors (weakly informative, matching paper):
  sigma, sigma_p, diag(Sigma), tau_b  ~ half-t(3, 0, scale)
  correlations in Sigma               ~ LKJ(1)
  fixed effects                       ~ normal(0, broad_scale)

Usage
-----
  python ev_skew_brms.py [--chains N] [--iter N] [--warmup N] [--seed N]
                         [--pitch-cluster {family|pitch_type}]  # default: pitch_type
                         [--sample-n N]   # subsample rows for quick tests
                         [--dry-run]      # write R script but don't execute

Outputs (written to OUT_DIR):
  ev_skew_brms_summary.txt     brms model summary
  ev_skew_brms_fitted.csv      posterior mean fitted values + residuals
  ev_skew_brms_diagnostics.png trace / Rhat / neff plots
  ev_skew_brms_effects.png     fixed-effect posterior intervals
  ev_skew_brms_model.rds       saved brms model object

Requires: R with brms, posterior, bayesplot, ggplot2 installed.
"""

import argparse, os, glob, sys, subprocess, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ── Paths (same conventions as swing_models_via_r.py) ────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "final_models")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Column aliases ────────────────────────────────────────────────────────────
INTERCEPT_X = 'intercept_ball_minus_batter_pos_x_inches'
INTERCEPT_Y = 'intercept_ball_minus_batter_pos_y_inches'
IN_PLAY     = {'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}

PITCH_FAMILY = {
    'FF': 'fastball', 'SI': 'fastball', 'FC': 'fastball', 'FT': 'fastball',
    'CU': 'breaking', 'SL': 'breaking', 'ST': 'breaking',
    'SV': 'breaking', 'KC': 'breaking',
    'CH': 'offspeed', 'FS': 'offspeed', 'FO': 'offspeed',
    'EP': 'offspeed', 'KN': 'offspeed', 'SC': 'offspeed',
}

sns.set_theme(style='whitegrid', font_scale=1.05)
BLUE = '#2563EB'; RED = '#DC2626'; GREEN = '#16A34A'; DARK = '#1F2937'


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA PREP
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data(pitch_cluster_type: str = 'family', sample_n: int = None) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.parquet')))
    if not files:
        raise FileNotFoundError(f'No parquet files found in {DATA_DIR}')

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f'Loaded {len(df):,} total pitches')

    # ── Keep only balls in play ───────────────────────────────────────────────
    df = df[df['description'].isin(IN_PLAY)].copy()
    print(f'After filtering to in-play: {len(df):,}')

    # ── Required columns ─────────────────────────────────────────────────────
    needed = ['launch_speed', INTERCEPT_X, INTERCEPT_Y,
              'attack_angle', 'bat_speed',
              'batter', 'pitcher', 'pitch_type']
    df = df.dropna(subset=needed).copy()
    print(f'After dropping NA on required columns: {len(df):,}')

    # ── Rename to short model names ───────────────────────────────────────────
    df = df.rename(columns={
        'launch_speed': 'exit_velo',
        INTERCEPT_X:   'intercept_x',
        INTERCEPT_Y:   'intercept_y',
    })

    # ── Pitch cluster ─────────────────────────────────────────────────────────
    if pitch_cluster_type == 'family':
        df['pitch_cluster'] = df['pitch_type'].map(PITCH_FAMILY).fillna('other')
    else:
        df['pitch_cluster'] = df['pitch_type']

    # ── Batter × pitch_cluster grouping key ──────────────────────────────────
    df['batter_cluster'] = df['batter'].astype(str) + '_' + df['pitch_cluster'].astype(str)

    # ── Standardise continuous predictors (numerical stability) ──────────────
    scale_cols = ['bat_speed', 'intercept_x', 'intercept_y', 'attack_angle']
    for col in scale_cols:
        mu, sd = df[col].mean(), df[col].std()
        df[f'{col}_s'] = (df[col] - mu) / sd
        print(f'  {col}: mean={mu:.3f}, sd={sd:.3f}  → {col}_s')

    # ── String factors ────────────────────────────────────────────────────────
    for col in ['batter', 'pitcher', 'pitch_cluster', 'batter_cluster']:
        df[col] = df[col].astype(str)

    # ── Optional subsample ────────────────────────────────────────────────────
    if sample_n and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42).copy()
        print(f'Subsampled to {len(df):,} rows')

    print(f'\nFinal modelling dataset: {len(df):,} rows')
    print(f'Unique batters:          {df["batter"].nunique()}')
    print(f'Unique pitchers:         {df["pitcher"].nunique()}')
    print(f'Pitch clusters:          {sorted(df["pitch_cluster"].unique())}')

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD R SCRIPT
# ══════════════════════════════════════════════════════════════════════════════

def build_r_script(data_path: str, out_prefix: str,
                   chains: int, iter_: int, warmup: int, seed: int) -> str:
    """
    Return a self-contained R script string that:
      1. Reads the CSV written by Python.
      2. Fits the skew-normal multilevel model with brms.
      3. Writes summary, fitted values, and diagnostic plots to out_prefix*.

    Model specification
    -------------------
    mu formula  (bf mu ~ ...):
        Fixed:   bat_speed_s + intercept_x_s + intercept_y_s + attack_angle_s
        Random:  (1 + bat_speed_s + intercept_x_s + intercept_y_s + attack_angle_s | batter)
                     → batter-level random intercept + correlated random slopes
                     → Sigma estimated with LKJ(1) prior on correlations
                 (1 | pitcher)
                     → pitcher-level random intercept

    alpha formula (bf alpha ~ ...):
        (1 | batter)
                     → batter-level random skewness intercept (nu_b in the paper)

    Family: skew_normal()

    Priors (weakly informative):
        fixed betas           ~ normal(0, 10)
        intercept             ~ normal(0, 20)
        sd hyperparameters    ~ student_t(3, 0, 2.5)   [half-t, default brms]
        correlations          ~ lkj(1)
        sigma                 ~ student_t(3, 0, 5)
    """

    r = f"""
# ── ev_skew_brms auto-generated R script ─────────────────────────────────────
library(brms)
library(posterior)
library(bayesplot)
library(ggplot2)

set.seed({seed})

# ── 1. Load data ──────────────────────────────────────────────────────────────
cat("Reading data...\\n")
df <- read.csv("{data_path}", stringsAsFactors = FALSE)

# Ensure grouping variables are character/factor
for (col in c("batter", "pitcher", "pitch_cluster", "batter_cluster")) {{
  if (col %in% names(df)) df[[col]] <- as.factor(df[[col]])
}}

cat(sprintf("n = %d rows, %d batters, %d pitchers\\n",
            nrow(df),
            nlevels(df$batter),
            nlevels(df$pitcher)))

# ── 2. Model formula ──────────────────────────────────────────────────────────
# mu submodel: fixed slopes + correlated batter random slopes + pitcher intercept
# alpha submodel: batter random intercept on skewness

bf_mu <- bf(
  exit_velo ~
    bat_speed_s + intercept_x_s + intercept_y_s + attack_angle_s +
    (1 + bat_speed_s + intercept_x_s + intercept_y_s + attack_angle_s | batter) +
    (1 | pitcher),
  # skewness alpha: separate linear predictor
  alpha ~ 1 + (1 | batter)
)

# ── 3. Priors ─────────────────────────────────────────────────────────────────
priors <- c(
  # Fixed effects on mu
  prior(normal(0, 10),  class = b),
  prior(normal(0, 20),  class = Intercept),

  # SD hyperpriors (half-t3): batter random effects
  prior(student_t(3, 0, 2.5), class = sd),

  # Residual sigma
  prior(student_t(3, 0, 5), class = sigma),

  # LKJ prior on correlation matrix for batter random slopes
  prior(lkj(1), class = cor),

  # Fixed intercept on alpha (log-scale skewness)
  prior(normal(0, 2), class = Intercept, dpar = alpha),

  # SD hyperprior for batter-level skewness random intercepts
  prior(student_t(3, 0, 1), class = sd, dpar = alpha)
)

# ── 4. Fit ────────────────────────────────────────────────────────────────────
cat("Fitting skew-normal multilevel model with brms...\\n")
cat(sprintf("Chains=%d  iter=%d  warmup=%d\\n", {chains}, {iter_}, {warmup}))

fit <- brm(
  formula   = bf_mu,
  data      = df,
  family    = skew_normal(),
  prior     = priors,
  chains    = {chains},
  iter      = {iter_},
  warmup    = {warmup},
  cores     = {chains},
  seed      = {seed},
  backend   = "cmdstanr",           # falls back to rstan if cmdstanr absent
  control   = list(adapt_delta = 0.90, max_treedepth = 12),
  silent    = 0
)

cat("\\nModel fitted.\\n")

# ── 5. Save model object ──────────────────────────────────────────────────────
saveRDS(fit, "{out_prefix}_model.rds")
cat("Saved model RDS\\n")

# ── 6. Summary ────────────────────────────────────────────────────────────────
sink("{out_prefix}_summary.txt")
cat("=== brms model summary ===\\n\\n")
print(summary(fit))
cat("\\n=== LOO-CV (approximate) ===\\n")
tryCatch({{
  loo_result <- loo(fit)
  print(loo_result)
}}, error = function(e) cat("LOO failed:", conditionMessage(e), "\\n"))
sink()
cat("Written summary\\n")

# ── 7. Posterior predictive / fitted values ───────────────────────────────────
cat("Extracting fitted values...\\n")
fitted_mat <- fitted(fit, summary = TRUE)   # rows = obs, cols = Estimate, Est.Error, Q2.5, Q97.5
fitted_df  <- as.data.frame(fitted_mat)
colnames(fitted_df) <- c("fitted_mean", "fitted_se", "fitted_q2.5", "fitted_q97.5")
fitted_df$exit_velo  <- df$exit_velo
fitted_df$residual   <- fitted_df$exit_velo - fitted_df$fitted_mean
fitted_df$batter     <- as.character(df$batter)
fitted_df$pitcher    <- as.character(df$pitcher)
fitted_df$pitch_cluster <- as.character(df$pitch_cluster)

write.csv(fitted_df, "{out_prefix}_fitted.csv", row.names = FALSE)
cat("Written fitted values\\n")

# ── 8. Convergence diagnostics ────────────────────────────────────────────────
cat("Generating diagnostic plots...\\n")
draws <- as_draws_array(fit)

# Rhat and ESS table for fixed effects + key variance parameters
diag_df <- data.frame(
  parameter = variables(draws)
)
diag_df$rhat <- sapply(diag_df$parameter, function(p) {{
  tryCatch(rhat(fit, pars = p)[[1]], error = function(e) NA)
}})
diag_df$neff_ratio <- tryCatch(
  neff_ratio(fit),
  error = function(e) rep(NA, nrow(diag_df))
)
write.csv(diag_df, "{out_prefix}_diagnostics.csv", row.names = FALSE)

# Trace plots (fixed effects only for readability)
fe_pars <- grep("^b_", variables(draws), value = TRUE)
if (length(fe_pars) > 0) {{
  p_trace <- mcmc_trace(draws, pars = fe_pars[seq_len(min(8, length(fe_pars)))]) +
    ggtitle("Trace plots – fixed effects (mu)") +
    theme_minimal()
  ggsave("{out_prefix}_trace.png", p_trace, width = 12, height = 8, dpi = 130)
  cat("Written trace plot\\n")
}}

# ── 9. Fixed-effect posterior intervals ──────────────────────────────────────
cat("Generating fixed-effects plot...\\n")
p_intervals <- mcmc_intervals(
  draws,
  pars  = fe_pars,
  prob  = 0.80,
  prob_outer = 0.95
) +
  ggtitle("Fixed effects: 80% and 95% posterior intervals") +
  theme_minimal()
ggsave("{out_prefix}_fe_intervals.png", p_intervals, width = 9, height = 6, dpi = 130)
cat("Written fixed-effects interval plot\\n")

# ── 10. Quick R² (Bayesian) ───────────────────────────────────────────────────
tryCatch({{
  r2_result <- bayes_R2(fit)
  cat("\\nBayesian R² (posterior median and 95% CI):\\n")
  print(r2_result)
  r2_df <- as.data.frame(r2_result)
  write.csv(r2_df, "{out_prefix}_r2.csv", row.names = FALSE)
}}, error = function(e) cat("bayes_R2 failed:", conditionMessage(e), "\\n"))

cat("\\n=== Done. All outputs written to {out_prefix}* ===\\n")
"""
    return r


# ══════════════════════════════════════════════════════════════════════════════
# 3. POST-HOC PYTHON DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def python_diagnostic_plots(fitted_csv: str, out_prefix: str):
    """
    Re-read the fitted-values CSV and generate diagnostic plots in Python
    (mirrors the summary_plots() style from swing_models_via_r.py).
    """
    fitted_df = pd.read_csv(fitted_csv)
    actual    = fitted_df['exit_velo']
    fitted    = fitted_df['fitted_mean']
    resid     = fitted_df['residual']

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('Exit Velocity – Skew-Normal Multilevel Model (brms)',
                 fontsize=13, fontweight='bold')

    # Residuals vs fitted
    ax = axes[0, 0]
    ax.scatter(fitted, resid, alpha=0.12, s=4, color=BLUE, rasterized=True)
    ax.axhline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Posterior mean fitted value'); ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Fitted', fontweight='bold')
    ax.grid(alpha=0.3)

    # Predicted vs actual
    ax = axes[0, 1]
    lim = [min(actual.min(), fitted.min()), max(actual.max(), fitted.max())]
    ax.scatter(actual, fitted, alpha=0.12, s=4, color=BLUE, rasterized=True)
    ax.plot(lim, lim, color=RED, linewidth=1.2, linestyle='--')
    corr = np.corrcoef(actual, fitted)[0, 1]
    ax.set_xlabel('Actual exit velocity'); ax.set_ylabel('Predicted')
    ax.set_title(f'Pred vs Actual  (r = {corr:.3f})', fontweight='bold')
    ax.grid(alpha=0.3)

    # Residual distribution
    ax = axes[1, 0]
    lo, hi = resid.quantile(0.005), resid.quantile(0.995)
    ax.hist(resid.clip(lo, hi), bins=80,
            color=BLUE, edgecolor='white', alpha=0.85)
    ax.axvline(0, color=RED, linewidth=1.2, linestyle='--')
    ax.set_xlabel('Residual'); ax.set_ylabel('Count')
    ax.set_title(f'Residuals  (σ = {resid.std():.2f})', fontweight='bold')
    ax.grid(alpha=0.3)

    # Uncertainty ribbon: sorted fitted value vs 95% CI width
    ax = axes[1, 1]
    ci_width = fitted_df['fitted_q97.5'] - fitted_df['fitted_q2.5']
    order = fitted.argsort().values
    ax.plot(range(len(order)), ci_width.iloc[order].values,
            color=GREEN, linewidth=0.6, alpha=0.7)
    ax.set_xlabel('Observation (sorted by fitted value)')
    ax.set_ylabel('95% posterior CI width (mph)')
    ax.set_title('Posterior uncertainty per observation', fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = f'{out_prefix}_python_diagnostics.png'
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def batter_scores(fitted_csv: str, out_prefix: str):
    """
    Compute a per-batter exit-velocity score: mean residual (over-/under-performance
    relative to the model's expectation given bat speed, intercepts, attack angle,
    and batter identity).  Positive = beats expectation; negative = falls short.
    """
    fitted_df = pd.read_csv(fitted_csv)
    scores = (fitted_df
              .groupby('batter')
              .agg(
                  n=('residual', 'size'),
                  ev_above_expected=('residual', 'mean'),
                  ev_above_expected_sd=('residual', 'std'),
                  fitted_mean_ev=('fitted_mean', 'mean'),
                  actual_mean_ev=('exit_velo', 'mean'),
              )
              .reset_index()
              .sort_values('ev_above_expected', ascending=False))

    path = f'{out_prefix}_batter_scores.csv'
    scores.to_csv(path, index=False)
    print(f'  → {path}')
    print(scores.head(10).to_string(index=False))
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Skew-normal brms EV model')
    parser.add_argument('--chains',         type=int, default=4)
    parser.add_argument('--iter',           type=int, default=4000)
    parser.add_argument('--warmup',         type=int, default=2000)
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--pitch-cluster',  choices=['family', 'pitch_type'],
                        default='pitch_type',
                        help='Group pitch types into 3 families or keep raw pitch_type (default)')
    parser.add_argument('--sample-n',       type=int, default=None,
                        help='Subsample N rows for quick testing')
    parser.add_argument('--dry-run',        action='store_true',
                        help='Write R script but do not execute')
    args = parser.parse_args()

    if args.warmup >= args.iter:
        parser.error('--warmup must be less than --iter')

    # ── Data ──────────────────────────────────────────────────────────────────
    df = prepare_data(pitch_cluster_type=args.pitch_cluster,
                      sample_n=args.sample_n)

    # ── Write CSV for R ───────────────────────────────────────────────────────
    data_path  = os.path.join(OUT_DIR, '_ev_brms_data.csv')
    out_prefix = os.path.join(OUT_DIR, 'ev_skew_brms')

    cols_to_write = [
        'exit_velo',
        'bat_speed_s', 'intercept_x_s', 'intercept_y_s', 'attack_angle_s',
        'batter', 'pitcher', 'pitch_cluster', 'batter_cluster',
    ]
    df[cols_to_write].to_csv(data_path, index=False)
    print(f'\nData written to {data_path}')

    # ── Build R script ────────────────────────────────────────────────────────
    r_script = build_r_script(
        data_path  = data_path,
        out_prefix = out_prefix,
        chains     = args.chains,
        iter_      = args.iter,
        warmup     = args.warmup,
        seed       = args.seed,
    )

    r_path = os.path.join(OUT_DIR, '_ev_brms_script.R')
    with open(r_path, 'w') as f:
        f.write(r_script)
    print(f'R script written to {r_path}')

    if args.dry_run:
        print('\n--dry-run set: skipping R execution.')
        print(f'To run manually:  Rscript {r_path}')
        return

    # ── Run R ─────────────────────────────────────────────────────────────────
    print(f'\nRunning R  (chains={args.chains}, iter={args.iter}, warmup={args.warmup}) …')
    print('This will take several minutes. Progress appears below.\n')

    try:
        result = subprocess.run(
            ['Rscript', r_path],
            capture_output=False,       # let stdout/stderr stream to terminal
            text=True,
            cwd=OUT_DIR,
            # No hard timeout: large MCMC runs can take a long time.
            # Add timeout=N if you want a ceiling.
        )
    except FileNotFoundError:
        print('ERROR: Rscript not found. Make sure R is installed and on PATH.')
        sys.exit(1)

    if result.returncode != 0:
        print(f'\nR script exited with code {result.returncode}.')
        sys.exit(result.returncode)

    # ── Post-hoc Python diagnostics ───────────────────────────────────────────
    fitted_csv = f'{out_prefix}_fitted.csv'
    if os.path.exists(fitted_csv):
        print('\nGenerating Python diagnostic plots …')
        python_diagnostic_plots(fitted_csv, out_prefix)
        print('Computing per-batter scores …')
        batter_scores(fitted_csv, out_prefix)
    else:
        print(f'\nWARNING: fitted values not found at {fitted_csv}. '
              'Check R output above for errors.')

    # ── Cleanup temp files ────────────────────────────────────────────────────
    for tmp in [data_path, r_path]:
        try:
            os.remove(tmp)
        except OSError:
            pass

    print(f'\nAll outputs written to {OUT_DIR}')
    print('Key files:')
    for suffix in ['_summary.txt', '_fitted.csv', '_r2.csv',
                   '_diagnostics.csv', '_trace.png', '_fe_intervals.png',
                   '_python_diagnostics.png', '_batter_scores.csv', '_model.rds']:
        p = f'{out_prefix}{suffix}'
        exists = '✓' if os.path.exists(p) else '✗ (missing)'
        print(f'  {exists}  {p}')


if __name__ == '__main__':
    main()