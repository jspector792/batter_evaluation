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

import argparse, os, glob, sys, subprocess, warnings, signal, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ── Paths (same conventions as swing_models_via_r.py) ────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "all_pitches_2025")
OUT_DIR  = os.path.join(BASE_DIR, "out", "exploratory", "final_models_variants")
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

    # ── Drop bunts and physically implausible swings ──────────────────────────
    # attack_angle/bat_speed have a long tail of |z| > 9 driven mostly by bunts
    # (near-zero bat speed makes "attack angle" numerically meaningless) and a
    # residue of apparent mistracked swings. This is a real data-quality issue
    # (~0.7% of rows) but NOT the cause of the cmdstanr "-inf"/hang failure --
    # that turned out to be a backend bug (see build_r_script() docstring) --
    # it's still worth filtering since these rows aren't representative swings.
    is_bunt = df['events'].astype(str).str.contains('bunt', na=False)
    plausible = (
        (~is_bunt)
        & df['attack_angle'].between(-40, 60)
        & (df['bat_speed'] >= 40)
        & (df['launch_speed'] >= 20)
    )
    n_before = len(df)
    df = df[plausible].copy()
    print(f'After dropping bunts/implausible swings: {len(df):,} '
          f'(removed {n_before - len(df):,})')

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
                   chains: int, iter_: int, warmup: int, seed: int,
                   backend: str = 'rstan', refresh: int = 10,
                   max_treedepth: int = 8) -> str:
    """
    Return a self-contained R script string that:
      1. Reads the CSV written by Python.
      2. Fits the skew-normal multilevel model with brms.
      3. Writes summary, fitted values, and diagnostic plots to out_prefix*.

    Model specification
    -------------------
    mu formula  (bf mu ~ ...):
        Fixed:   bat_speed_s + intercept_x_s + intercept_y_s + attack_angle_s
        Random:  (1 + bat_speed_s + intercept_x_s + intercept_y_s + attack_angle_s || batter)
                     → batter-level random intercept + UNCORRELATED random slopes
                     → the `||` syntax drops the LKJ-correlated covariance matrix;
                       a full run at 118k rows / 776 batters with the correlated
                       (`|`) version got the NUTS sampler stuck at iteration 1 for
                       53+ minutes across all 4 chains (degenerate LKJ correlation
                       matrix, `-inf` skew-normal location proposals) and never
                       advanced — the 5x5 batter-level covariance was unidentifiable
                       at this scale. Independent slopes keep the same substantive
                       random effects without estimating their correlations.
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
        sigma                 ~ student_t(3, 0, 5)

    Backend: rstan, not cmdstanr
    ----------------------------
    A full diagnostic investigation (see out/exploratory/final_models_variants/
    or the session that added this comment) isolated the "-inf"/"nan" location-
    parameter and "Scale vector is 0/inf" exceptions that were hanging every
    run of this model. Ruled out, in order:
      - Data outliers (attack_angle/bat_speed z-scores of 9-10+, mostly from
        bunts and mistracked swings) -- cleaning them barely changed anything
      - The per-batter random alpha (skewness) submodel specifically -- a
        model without it got stuck exactly the same way
      - Wide/diffuse Stan initial values -- `init = 0.1` (tight init) failed
        identically
      - The skew_normal family / its already-extreme marginal alpha=-6.78
        (see out/exploratory/skewness_check/skewness_summary.csv) -- a plain
        gaussian() fit on a reflected+log-transformed (near-symmetric, skew
        -0.11) exit_velo, with the exact same random-intercept structure,
        failed identically
      - Having two crossed high-cardinality group factors (776 batters AND
        927 pitchers) -- single-factor models (batter-only OR pitcher-only)
        failed the same way
    The one thing that fixed it: switching `backend` from "cmdstanr" to
    "rstan" made the exceptions disappear completely on the exact same
    formula/data/priors that cmdstanr couldn't get through. This points to a
    numerical bug/regression specific to this cmdstanr/CmdStan build (CmdStan
    2.39.0) in the "_glm" sufficient-statistics code path brms generates for
    multilevel models, not a modeling or data problem. rstan is slower per
    gradient eval for a model this size (~0.07 sec/eval per rstan's own
    "adjust your expectations" notice), so budget real time, but it actually
    makes progress instead of hanging.

    No random slopes -- they're the real cost driver, not skew_normal
    -----------------------------------------------------------------
    Follow-up timing tests (after the rstan fix) isolated a second, separate
    issue: even under rstan, the correlated-vs-uncorrelated random SLOPES on
    batter (bat_speed/intercept_x/intercept_y/attack_angle, 5 random effects
    per batter) are ~6-7x more expensive per iteration than plain random
    INTERCEPTS (measured: ~5.4 sec/iter for batter+pitcher intercepts alone
    vs ~36 sec/iter once slopes were added back, same data/chains/backend).
    Adding the per-batter alpha submodel and skew_normal on top of
    intercepts-only measured ~23 sec/iter -- still leaves random slopes as
    the single biggest lever on runtime. At 1500 iterations x 4 chains, slopes
    would take ~9-10 hours; intercepts-only fits inside a few hours. So this
    formula drops the random slopes and keeps batter/pitcher random
    intercepts + the per-batter alpha submodel, which is still the
    substantive ask (per-batter EV skill under a skewed likelihood) just
    without batter-specific sensitivity to bat speed / pitch location.
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
# mu submodel: fixed slopes + batter/pitcher random INTERCEPTS ONLY
# (random slopes on batter measured ~6-7x slower per iteration than
#  intercepts alone -- see docstring above -- and would blow the runtime
#  budget; dropping them keeps the per-batter EV-skill estimate, just
#  without batter-specific sensitivity to bat speed / pitch location)
# alpha submodel: batter random intercept on skewness

bf_mu <- bf(
  exit_velo ~
    bat_speed_s + intercept_x_s + intercept_y_s + attack_angle_s +
    (1 | batter) + (1 | pitcher),
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
  backend   = "{backend}",          # rstan by default -- cmdstanr 2.39.0 hangs on
                                      # this model with "-inf"/"Scale vector" exceptions
                                      # even on the simplest possible random-intercept
                                      # spec; see docstring above for the full diagnosis
  control   = list(adapt_delta = 0.90, max_treedepth = {max_treedepth}),
                    # capped at 8 (down from 12) to bound worst-case per-iteration
                    # cost -- the 4-chain production run made real progress (up to
                    # 30/350 iterations) but then hit a long stretch of very slow
                    # iterations, most likely deep NUTS trees during the
                    # least-adapted early part of warmup; a lower ceiling trades
                    # some exploration efficiency for predictable runtime
  silent    = 0,
  refresh   = {refresh}   # default brms/Stan refresh (iter/10) can leave 50+ min
                           # between progress prints for a model this size -- that
                           # made genuinely-progressing runs indistinguishable from
                           # hung ones under the stall watchdog. Print far more often.
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
# 4. WATCHDOG SUBPROCESS RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_with_watchdog(cmd: list, cwd: str, log_path: str,
                      stall_minutes: int, max_minutes: int,
                      poll_seconds: int = 30) -> int:
    """
    Run `cmd`, streaming its output to both the terminal and `log_path`, while
    watching for two failure modes we've hit before with brms/Stan:
      1. A genuinely stuck sampler (e.g. a degenerate LKJ correlation matrix)
         that burns CPU forever without ever printing further progress --
         killed if `log_path` hasn't grown in `stall_minutes`.
      2. A model that's technically progressing but far slower than
         acceptable -- killed at the `max_minutes` hard ceiling regardless.

    Runs the child in its own process group so killing it also reaps the
    actual Stan sampler processes (brms/cmdstanr spawn these as grandchildren
    of Rscript, which a plain subprocess.kill() would leave running).
    """
    start = time.time()
    with open(log_path, 'w') as log_f:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=log_f, stderr=subprocess.STDOUT,
                                start_new_session=True)

        last_size = -1
        last_growth = start
        tail_printed = 0

        def _kill(reason: str):
            print(f'\nWATCHDOG: {reason} -- killing process tree.')
            # Belt and suspenders: killpg handles the common case, but
            # cmdstanr/processx puts each chain's compiled model binary in
            # its OWN session (a real incident: killing Rscript's process
            # group once left 3 cmdstan chain binaries running for 90+
            # minutes afterward, unnoticed, competing for CPU with the next
            # run). Enumerate descendants via psutil BEFORE killing anything
            # -- once the parent dies, orphaned children get reparented to
            # launchd/init and a post-hoc tree walk from the original pid
            # would miss them.
            descendants = []
            try:
                import psutil
                descendants = psutil.Process(proc.pid).children(recursive=True)
            except (ImportError, psutil.NoSuchProcess):
                pass
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            for child in descendants:
                try:
                    child.kill()
                except Exception:
                    pass
            if descendants:
                print(f'WATCHDOG: also killed {len(descendants)} '
                      f'descendant process(es): {[c.pid for c in descendants]}')
            proc.wait()

        while True:
            try:
                proc.wait(timeout=poll_seconds)
                break   # process finished on its own
            except subprocess.TimeoutExpired:
                pass

            elapsed_min = (time.time() - start) / 60
            size = os.path.getsize(log_path)
            if size != last_size:
                last_size = size
                last_growth = time.time()
                # Echo newly written lines to the terminal
                with open(log_path) as f:
                    f.seek(tail_printed)
                    new_text = f.read()
                    tail_printed += len(new_text)
                if new_text:
                    print(new_text, end='')

            stalled_min = (time.time() - last_growth) / 60

            if stalled_min >= stall_minutes:
                _kill(f'no new output for {stalled_min:.1f} min '
                      f'(limit {stall_minutes}) -- sampler is almost certainly stuck, '
                      f'not just slow')
                return 1
            if elapsed_min >= max_minutes:
                _kill(f'hit the {max_minutes}-minute hard ceiling '
                      f'({elapsed_min:.1f} min elapsed)')
                return 1

    return proc.returncode


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
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
    parser.add_argument('--stall-minutes',  type=int, default=20,
                        help='Kill the run if Stan produces no new console output for '
                             'this many minutes (default 20) -- catches a sampler stuck '
                             'at iteration 1 (e.g. a degenerate LKJ correlation matrix) '
                             'instead of letting it burn CPU forever.')
    parser.add_argument('--max-minutes',    type=int, default=180,
                        help='Hard ceiling on total runtime regardless of progress '
                             '(default 180).')
    parser.add_argument('--backend',        choices=['rstan', 'cmdstanr'],
                        default='rstan',
                        help='Stan backend (default rstan). cmdstanr 2.39.0 hangs on '
                             'this model with "-inf"/"Scale vector" exceptions even on '
                             'the simplest random-intercept spec -- see module docstring '
                             'in build_r_script(). Only pass --backend cmdstanr to '
                             're-check whether a future CmdStan release fixed it.')
    parser.add_argument('--refresh',        type=int, default=10,
                        help='Print Stan progress every N iterations (default 10). '
                             'The brms/Stan default (iter/10) can leave 50+ minutes '
                             'between prints for a model this size, which made '
                             'progressing runs indistinguishable from hung ones.')
    parser.add_argument('--max-treedepth',  type=int, default=8,
                        help='NUTS max treedepth (default 8, down from Stan default '
                             '10/brms default 12). Bounds worst-case per-iteration '
                             'cost -- a production run made real progress but then '
                             'hit a long stretch of very slow iterations, likely deep '
                             'NUTS trees during the least-adapted early warmup.')
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
        backend    = args.backend,
        refresh    = args.refresh,
        max_treedepth = args.max_treedepth,
    )

    r_path = os.path.join(OUT_DIR, '_ev_brms_script.R')
    with open(r_path, 'w') as f:
        f.write(r_script)
    print(f'R script written to {r_path}')

    if args.dry_run:
        print('\n--dry-run set: skipping R execution.')
        print(f'To run manually:  Rscript {r_path}')
        return

    # ── Run R (with stall/timeout watchdog) ──────────────────────────────────────
    print(f'\nRunning R  (chains={args.chains}, iter={args.iter}, warmup={args.warmup}) …')
    print(f'Watchdog: kill if no new output for {args.stall_minutes} min, '
          f'or total runtime exceeds {args.max_minutes} min.\n')

    run_log_path = f'{out_prefix}_run.log'
    returncode = run_with_watchdog(
        ['Rscript', r_path], cwd=OUT_DIR, log_path=run_log_path,
        stall_minutes=args.stall_minutes, max_minutes=args.max_minutes,
    )

    if returncode != 0:
        print(f'\nR script did not finish successfully (code {returncode}). '
              f'Full output in {run_log_path}')
        sys.exit(returncode)

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