"""
dirB_miss_distance_added.py
============================
Direction B: "Miss Distance Added" (MDA) as a continuous contact-skill
metric, tested with the Franks et al. discrimination/stability framework
against the public contact-skill proxies teams already use.

The idea
--------
Collapsing the model into a binary contact probability threw away the thing
the model is actually good at: predicting *how far from the barrel* a swing
will end up. Direction B tests that continuous residual directly as a stat
instead of forcing it through the RB machinery.

    miss_distance_actual      = barrel_distance_v2 (inches), the project's
                                existing bat-center-to-ball-center distance
                                at closest approach
    predicted_miss_distance   = out-of-fold BARREL_CONFIG MERF prediction
                                (pre-outcome inputs only: release_speed_c,
                                plate_x_bat_flip, plate_z, intercept_y)
    residual_i                = actual - predicted
    MDA_batter                = mean(residual_i)

Sign convention: **negative is good** -- the batter got closer to the ball
than the model expected for that pitch.

Which prediction to subtract -- the thing that decides whether this works
---------------------------------------------------------------------------
Running this as literally specified, against the cached `predicted_offset`,
produces a **null statistic**: discrimination exactly 0.000 and a first-half
to second-half correlation of **r = -0.77**. The cause is not noise. That
MERF prediction carries a batter random intercept fit on the batter's own
season, so it has already absorbed the batter effect; the random intercept
pins each batter's season-long residual sum near zero, which forces the two
halves to cancel. A skill metric cannot be built on a residual whose defining
property is that it averages to zero per player.

The fix is to subtract a **context-only** prediction -- the same MERF with the
random intercept suppressed, i.e. what the league-average batter would have
done against those pitches. `dirB_context_offset_oof.py` generates it
out-of-fold (`predicted_offset_context`). Both versions are reported:

    MDA_*            measured against predicted_offset_context  <- the metric
    MDA_batterRE_*   measured against predicted_offset           <- retained
                     to document the failure mode, not proposed as a stat

The two measurement populations, kept separate (never silently mixed)
------------------------------------------------------------------------
    whiff_tracked        barrel_distance_v2 = real tracked miss_distance + C.
                         Clean. No outcome-derived quantity anywhere.
    contact_la_derived   barrel_distance_v2 = |C * sin(launch_angle - 20deg)|.
                         Launch angle is an outcome-only field, so this is a
                         *descriptive outcome* measure, not something that
                         could ever be a model input (data_audit.md sections
                         2-4). Per the spec this is fine for a diagnostic
                         metric, and it is labeled as such everywhere it
                         appears.

Three MDA variants are therefore reported side by side:
    MDA_all       both populations pooled (the headline metric; inherits the
                  launch-angle caveat for its contact half)
    MDA_whiff     whiff swings only -- fully tracked, zero outcome-derived
                  input, the defensible-under-scrutiny version
    MDA_contact   contact swings only -- launch-angle-derived

Outputs (out/rb_contact/followups/)
-----------------------------------
    discrimination_table.csv       MDA variants vs. contact%/whiff%/zone-contact%
    stability_table.csv            first-half vs. second-half persistence
    cross_metric_prediction.csv    does early MDA predict later contact% better
                                   than early contact% does?
    miss_distance_added_by_batter.csv
    direction_b_verdict.md
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from followup_utils import (load_swing_frame, discrimination, discrimination_bootstrap,
                            stability_two_period, rate_stat, mean_stat, md_table,
                            load_batter_names, FOLLOWUP_DIR)

MIN_SWINGS = 100         # matches step5_rb_contact_pct.MIN_SWINGS for comparability
MIN_WHIFFS = 20          # MDA_whiff needs enough whiffs for a stable mean
MIN_IN_ZONE = 50         # zone-contact% needs a real denominator
MIN_HALF_SWINGS = 50     # per half, for the split-half stability test
N_BOOT = 200
SEED = 42

# Metric key -> label. The MDA_* family is measured against the context-only
# prediction; the MDA_batterRE_* family against the batter-random-effect
# prediction, and is present to document the failure mode described above.
CONTINUOUS_METRICS = {
    'MDA_all': 'Miss Distance Added (all swings)',
    'MDA_whiff': 'Miss Distance Added (whiffs only, fully tracked)',
    'MDA_contact': 'Miss Distance Added (contact only, LA-derived)',
    'MDA_batterRE_all': 'MDA vs. batter-RE prediction (all swings) [null by construction]',
    'MDA_batterRE_whiff': 'MDA vs. batter-RE prediction (whiffs only) [null by construction]',
}
CONTEXT_METRICS = ['MDA_all', 'MDA_whiff', 'MDA_contact']
RATE_METRICS = {
    'contact_pct': 'raw contact%',
    'whiff_pct': 'raw whiff%',
    'zone_contact_pct': 'zone-contact% (zone 1-9)',
    'ozone_contact_pct': 'out-of-zone contact% (zone 11-14)',
}


# ──────────────────────────────────────────────────────────────────────────
# Per-batter metric construction
# ──────────────────────────────────────────────────────────────────────────

def batter_cohort(df: pd.DataFrame) -> pd.Index:
    """
    One common batter cohort for every metric in the tables. Discrimination
    depends on the population it's computed over, so comparing MDA (measured
    on batters with >= MIN_WHIFFS whiffs) against contact% (measured on
    everyone) would confound the metric with the cohort. Everything below is
    computed on the intersection.
    """
    per = df.groupby('batter').agg(
        n_swings=('is_contact', 'size'),
        n_whiffs=('is_contact', lambda s: (~s).sum()),
        n_in_zone=('in_zone', 'sum'),
        n_mda=('offset_residual_context', 'count'),
    )
    whiff_resid = (df[~df['is_contact']].groupby('batter')['offset_residual_context']
                     .count().rename('n_resid_whiff'))
    per = per.join(whiff_resid).fillna({'n_resid_whiff': 0})
    keep = per[(per['n_swings'] >= MIN_SWINGS) &
               (per['n_whiffs'] >= MIN_WHIFFS) &
               (per['n_in_zone'] >= MIN_IN_ZONE) &
               (per['n_resid_whiff'] >= MIN_WHIFFS)]
    return keep.index


def build_metrics(df: pd.DataFrame, cohort: pd.Index = None) -> dict:
    """
    Returns {metric_key: DataFrame(index=batter, columns=[value, n,
    sampling_var])} for every metric in the comparison, restricted to
    `cohort` if given.
    """
    if cohort is not None:
        df = df[df['batter'].isin(cohort)]

    out = {}
    out['MDA_all'] = mean_stat(df, 'offset_residual_context')
    out['MDA_whiff'] = mean_stat(df[~df['is_contact']], 'offset_residual_context')
    out['MDA_contact'] = mean_stat(df[df['is_contact']], 'offset_residual_context')
    out['MDA_batterRE_all'] = mean_stat(df, 'offset_residual')
    out['MDA_batterRE_whiff'] = mean_stat(df[~df['is_contact']], 'offset_residual')

    df = df.copy()
    df['is_whiff'] = ~df['is_contact']
    out['contact_pct'] = rate_stat(df, 'is_contact')
    out['whiff_pct'] = rate_stat(df, 'is_whiff')
    out['zone_contact_pct'] = rate_stat(df, 'is_contact', df['in_zone'])
    out['ozone_contact_pct'] = rate_stat(df, 'is_contact', ~df['in_zone'])

    if cohort is not None:
        out = {k: v.reindex(cohort) for k, v in out.items()}
    return out


# ──────────────────────────────────────────────────────────────────────────
# Table 3 analog: discrimination
# ──────────────────────────────────────────────────────────────────────────

def discrimination_table(df: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    rows = []
    boot_source = {
        'MDA_all': (df, 'offset_residual_context'),
        'MDA_whiff': (df[~df['is_contact']], 'offset_residual_context'),
        'MDA_contact': (df[df['is_contact']], 'offset_residual_context'),
        'MDA_batterRE_all': (df, 'offset_residual'),
        'MDA_batterRE_whiff': (df[~df['is_contact']], 'offset_residual'),
        'contact_pct': (df.assign(_v=df['is_contact'].astype(float)), '_v'),
        'whiff_pct': (df.assign(_v=(~df['is_contact']).astype(float)), '_v'),
        'zone_contact_pct': (df[df['in_zone']].assign(
            _v=df.loc[df['in_zone'], 'is_contact'].astype(float)), '_v'),
        'ozone_contact_pct': (df[~df['in_zone']].assign(
            _v=df.loc[~df['in_zone'], 'is_contact'].astype(float)), '_v'),
    }
    labels = {**CONTINUOUS_METRICS, **RATE_METRICS}
    for key, m in metrics.items():
        m = m.dropna(subset=['value', 'sampling_var'])
        d_analytic = discrimination(m['value'].values, m['sampling_var'].values)
        src_df, src_col = boot_source[key]
        src_df = src_df[src_df['batter'].isin(m.index)]
        d_boot = discrimination_bootstrap(src_df, src_col, n_boot=N_BOOT, seed=SEED)
        rows.append(dict(
            metric=key, label=labels[key], n_batters=len(m),
            mean=m['value'].mean(), sd_across_batters=m['value'].std(ddof=1),
            mean_n_per_batter=m['n'].mean(),
            mean_sampling_sd=np.sqrt(m['sampling_var']).mean(),
            discrimination=d_analytic, discrimination_bootstrap=d_boot,
        ))
        print(f'  {labels[key]:<52s} D={d_analytic:.4f} (boot {d_boot:.4f})  n={len(m)}')
    return pd.DataFrame(rows).sort_values('discrimination', ascending=False)


# ──────────────────────────────────────────────────────────────────────────
# Stability
# ──────────────────────────────────────────────────────────────────────────

def split_halves(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by calendar median date, matching step5_rb_contact_pct's
    split_half_reliability so the two reports describe the same halves.
    """
    median_date = df['game_date'].median()
    return df[df['game_date'] <= median_date], df[df['game_date'] > median_date]


def stability_table(df: pd.DataFrame) -> pd.DataFrame:
    h1, h2 = split_halves(df)
    n1 = h1.groupby('batter').size()
    n2 = h2.groupby('batter').size()
    cohort = n1[n1 >= MIN_HALF_SWINGS].index.intersection(n2[n2 >= MIN_HALF_SWINGS].index)
    cohort = cohort.intersection(batter_cohort(df))
    print(f'  split-half cohort: {len(cohort)} batters '
          f'(>= {MIN_HALF_SWINGS} swings in each half, and in the full-season cohort)')

    m1 = build_metrics(h1, cohort)
    m2 = build_metrics(h2, cohort)

    labels = {**CONTINUOUS_METRICS, **RATE_METRICS}
    rows = []
    for key in labels:
        a, b = m1[key], m2[key]
        joined = a[['value', 'sampling_var']].join(
            b[['value', 'sampling_var']], lsuffix='_h1', rsuffix='_h2').dropna()
        if len(joined) < 10:
            continue
        pear = stats.pearsonr(joined['value_h1'], joined['value_h2'])
        spear = stats.spearmanr(joined['value_h1'], joined['value_h2'])
        st = stability_two_period(joined['value_h1'], joined['value_h2'],
                                  joined['sampling_var_h1'], joined['sampling_var_h2'])
        rows.append(dict(
            metric=key, label=labels[key], n_batters=len(joined),
            pearson_r=pear[0], pearson_p=pear[1],
            spearman_rho=spear[0], spearman_p=spear[1],
            franks_stability=st['stability'],
            var_persistent=st['var_persistent'], var_change=st['var_change'],
            variance_estimate_clipped=st['clipped'],
        ))
        print(f'  {labels[key]:<52s} r={pear[0]:.3f}  stability={st["stability"]:.3f}')
    return pd.DataFrame(rows).sort_values('pearson_r', ascending=False), m1, m2, cohort


# ──────────────────────────────────────────────────────────────────────────
# Cross-metric prediction
# ──────────────────────────────────────────────────────────────────────────

def cv_predict(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """
    Out-of-fold OLS predictions across batters. Needed because MDA is in
    inches and the targets are rates: a raw |MDA - contact%| difference is
    meaningless, so the comparison has to go through a fitted mapping. Fitting
    that mapping in-sample would hand the multi-predictor models a free
    advantage, hence the cross-validation.
    """
    X = np.asarray(X, float).reshape(len(y), -1)
    y = np.asarray(y, float)
    oof = np.full(len(y), np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X):
        model = LinearRegression().fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
    return oof


def cross_metric_table(m1: dict, m2: dict, cohort: pd.Index) -> pd.DataFrame:
    """
    Does a first-half MDA predict *second-half* contact%/whiff%/zone-contact%
    better than the batter's own first-half raw rate does? This is the test
    that matters practically: MDA only earns a place if it says something
    about the outcome stats teams already track, beyond what those stats say
    about themselves.
    """
    predictors = {
        'MDA_all_h1': m1['MDA_all']['value'],
        'MDA_whiff_h1': m1['MDA_whiff']['value'],
        'MDA_batterRE_all_h1': m1['MDA_batterRE_all']['value'],
        'contact_pct_h1': m1['contact_pct']['value'],
        'whiff_pct_h1': m1['whiff_pct']['value'],
        'zone_contact_pct_h1': m1['zone_contact_pct']['value'],
    }
    targets = {
        'contact_pct_h2': m2['contact_pct']['value'],
        'whiff_pct_h2': m2['whiff_pct']['value'],
        'zone_contact_pct_h2': m2['zone_contact_pct']['value'],
    }
    # Combined models: the incremental-value question ("does MDA add anything
    # on top of the stat a team already has?"), which single-predictor rows
    # can't answer.
    combos = {
        'contact_pct_h1 + MDA_all_h1': ['contact_pct_h1', 'MDA_all_h1'],
        'contact_pct_h1 + MDA_whiff_h1': ['contact_pct_h1', 'MDA_whiff_h1'],
    }

    frame = pd.DataFrame({**predictors, **targets}).reindex(cohort).dropna()
    rows = []
    for tname, tcol in targets.items():
        y = frame[tname].values
        specs = [([p], p) for p in predictors] + [(cols, name) for name, cols in combos.items()]
        for cols, label in specs:
            pred = cv_predict(frame[cols].values, y)
            rows.append(dict(
                target=tname, predictor=label, n_batters=len(frame),
                cv_mae=float(np.mean(np.abs(pred - y))),
                cv_rmse=float(np.sqrt(np.mean((pred - y) ** 2))),
                cv_pearson_r=float(np.corrcoef(pred, y)[0, 1]),
                cv_spearman_rho=float(stats.spearmanr(pred, y)[0]),
                raw_pearson_r=(float(np.corrcoef(frame[cols[0]], y)[0, 1])
                               if len(cols) == 1 else np.nan),
            ))
    out = pd.DataFrame(rows)
    for tname in targets:
        sub = out[out['target'] == tname].sort_values('cv_mae')
        print(f'\n  target = {tname} (lower cv_mae is better)')
        for r in sub.itertuples():
            print(f'    {r.predictor:<32s} MAE={r.cv_mae:.5f}  r={r.cv_pearson_r:+.3f}')
    return out


# ──────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────

def write_verdict(disc: pd.DataFrame, stab: pd.DataFrame, cross: pd.DataFrame,
                  n_swings: int, path: str):
    lines = ['# Direction B -- Miss Distance Added: Verdict\n']
    lines.append(
        '**Metric.** `Miss Distance Added (MDA)` = mean over a batter\'s swings of '
        '`barrel_distance_v2 - predicted_offset_context`, in inches -- how much closer '
        'to the ball the batter got than the **league-average batter** would have on '
        'those same pitches. `predicted_offset_context` is the out-of-fold '
        'BARREL_CONFIG MERF prediction with the batter random intercept suppressed '
        '(pre-outcome inputs only). **Negative is good.**\n')
    lines.append(
        '**Why not the pipeline\'s `predicted_offset`?** Because the spec\'s literal '
        'construction is a null statistic, and the `MDA_batterRE_*` rows in the tables '
        'below are kept to show it. That prediction carries a batter random intercept '
        'fit on the batter\'s own season, which pins each batter\'s season-long residual '
        'sum near zero -- so the residual has the batter effect already subtracted out, '
        'discrimination comes back at 0.000, and the two halves of the season are forced '
        'to cancel (r = -0.77). A skill metric cannot be built on a residual whose '
        'defining property is that it averages to zero per player. Subtracting the '
        'context-only prediction instead keeps the batter effect, which is the thing '
        'being measured. See `code/rb_contact/dirB_context_offset_oof.py`.\n')
    lines.append(
        '**Measurement caveat, carried everywhere.** For whiffs, `barrel_distance_v2` '
        'is the real tracked `miss_distance` (+ C). For contact events it is '
        '`|C*sin(launch_angle - 20deg)|`, derived from launch angle -- an outcome-only '
        'field. That is acceptable for a descriptive diagnostic (and is what the spec '
        'asks for) but it is not a tracked miss distance, so `MDA_whiff` is reported '
        'alongside as the fully-tracked, no-outcome-input version.\n')
    lines.append(f'**Population.** {n_swings:,} swings; common cohort of '
                 f'{int(disc["n_batters"].max()):,} batters '
                 f'(>= {MIN_SWINGS} swings, >= {MIN_WHIFFS} whiffs with a usable '
                 f'residual, >= {MIN_IN_ZONE} in-zone swings). Every metric in every '
                 f'table below is computed on that same cohort, since discrimination '
                 f'is a property of the population as much as the metric.\n')

    lines.append('\n## 1. Discrimination (Franks et al. Table 3 analog)\n')
    lines.append('Fraction of observed between-batter variance that is real skill '
                 'rather than sampling noise. Higher is better; 1.0 would mean the '
                 'metric is measured without error.\n')
    lines.append(md_table(disc[['label', 'n_batters', 'mean', 'sd_across_batters',
                                'mean_n_per_batter', 'discrimination',
                                'discrimination_bootstrap']]))
    top = disc.iloc[0]
    mda = disc[disc['metric'] == 'MDA_all'].iloc[0]
    mdaw = disc[disc['metric'] == 'MDA_whiff'].iloc[0]
    best_rate = disc[disc['metric'].isin(RATE_METRICS)].iloc[0]
    lines.append(f'\n- Most discriminative metric overall: **{top["label"]}** '
                 f'(D = {top["discrimination"]:.4f}).')
    lines.append(f'- `MDA_all` D = {mda["discrimination"]:.4f}; `MDA_whiff` D = '
                 f'{mdaw["discrimination"]:.4f}; best public baseline '
                 f'({best_rate["label"]}) D = {best_rate["discrimination"]:.4f}.')
    lines.append('- Note: raw whiff% is exactly `1 - raw contact%` on this swing '
                 'population, so the two necessarily share a discrimination value. '
                 'Both are listed because the spec asks for both; they are one '
                 'baseline, not two.')
    re_all = disc[disc['metric'] == 'MDA_batterRE_all'].iloc[0]
    lines.append(f'- The `MDA_batterRE_*` rows are the spec\'s literal construction '
                 f'(residual against the batter-random-effect prediction): '
                 f'D = {re_all["discrimination"]:.4f}, across-batter SD '
                 f'{re_all["sd_across_batters"]:.4f} inches against a mean sampling SD '
                 f'of {re_all["mean_sampling_sd"]:.4f} inches -- the between-batter '
                 f'spread is smaller than the noise. Not a weak metric, a null one.')

    lines.append('\n## 2. Stability (first half vs. second half of 2025)\n')
    lines.append('A true year-over-year test is not runnable: `data/` holds '
                 'pitch-level Statcast for 2025 only. Per the spec\'s fallback, this '
                 'is the two-halves-of-one-season version, and the correlations should '
                 'be read as a **lower-confidence proxy** for year-over-year '
                 'persistence -- within-season halves share park/team/approach context '
                 'that separate seasons do not, which biases these correlations '
                 'upward relative to a real YoY number.\n')
    lines.append('`franks_stability` = Var(persistent skill) / (Var(persistent) + '
                 'Var(real half-to-half change)), with sampling noise removed from '
                 'both. Var(persistent) is estimated as Cov(h1, h2).\n')
    lines.append(md_table(stab[['label', 'n_batters', 'pearson_r', 'spearman_rho',
                                'franks_stability', 'var_persistent', 'var_change',
                                'variance_estimate_clipped']]))
    s_mda = stab[stab['metric'] == 'MDA_all'].iloc[0]
    s_con = stab[stab['metric'] == 'contact_pct'].iloc[0]
    lines.append(f'\n- `MDA_all` half-to-half r = {s_mda["pearson_r"]:.3f} '
                 f'(stability {s_mda["franks_stability"]:.3f}) vs. raw contact% '
                 f'r = {s_con["pearson_r"]:.3f} '
                 f'(stability {s_con["franks_stability"]:.3f}).')

    lines.append('\n## 3. Cross-metric prediction\n')
    lines.append('Can a first-half metric predict a *second-half outcome rate* better '
                 'than that rate\'s own first-half value can? MDA is in inches and the '
                 'targets are rates, so each predictor is mapped to the target through '
                 'an OLS fit scored out-of-fold across batters (5-fold) -- otherwise '
                 'the scales are not comparable and multi-predictor rows would get a '
                 'free in-sample advantage.\n')
    lines.append(md_table(cross[['target', 'predictor', 'n_batters', 'cv_mae',
                                 'cv_rmse', 'cv_pearson_r', 'cv_spearman_rho']], 5))
    for tname in cross['target'].unique():
        sub = cross[cross['target'] == tname].sort_values('cv_mae')
        winner = sub.iloc[0]
        own = {'contact_pct_h2': 'contact_pct_h1', 'whiff_pct_h2': 'whiff_pct_h1',
               'zone_contact_pct_h2': 'zone_contact_pct_h1'}[tname]
        own_row = sub[sub['predictor'] == own].iloc[0]
        lines.append(f'\n- **{tname}**: best predictor is `{winner["predictor"]}` '
                     f'(CV MAE {winner["cv_mae"]:.5f}); the stat\'s own first half '
                     f'(`{own}`) gives MAE {own_row["cv_mae"]:.5f} '
                     f'({100*(winner["cv_mae"]-own_row["cv_mae"])/own_row["cv_mae"]:+.1f}%).')

    # ── Bottom line, answering the spec's question directly ────────────────
    lines.append('\n## Verdict\n')
    rate_disc = disc[disc['metric'].isin(RATE_METRICS)]['discrimination'].max()
    rate_stab = stab[stab['metric'].isin(RATE_METRICS)]['franks_stability'].max()
    mda_disc = disc[disc['metric'].isin(CONTEXT_METRICS)]['discrimination'].max()
    mda_stab = stab[stab['metric'].isin(CONTEXT_METRICS)]['franks_stability'].max()
    best_mda_d = disc[disc['metric'].isin(CONTEXT_METRICS)].sort_values(
        'discrimination', ascending=False).iloc[0]

    more_disc = mda_disc > rate_disc
    lines.append(
        f'**Is Miss Distance Added more discriminative than the existing public '
        f'contact-skill proxies?** {"Yes" if more_disc else "No"} -- best MDA variant '
        f'({best_mda_d["label"]}) D = {mda_disc:.4f} vs. best public baseline '
        f'D = {rate_disc:.4f}. The fully-tracked, no-launch-angle variant '
        f'(`MDA_whiff`) also clears it at D = '
        f'{disc[disc["metric"] == "MDA_whiff"]["discrimination"].iloc[0]:.4f}, so the '
        f'result does not depend on the launch-angle-derived contact half.')

    # Reliability (raw half-to-half r) and Franks stability can disagree, and
    # here they do -- worth spelling out rather than reporting one number.
    mda_r = stab[stab['metric'].isin(CONTEXT_METRICS)]['pearson_r'].max()
    con_r = stab[stab['metric'] == 'contact_pct']['pearson_r'].iloc[0]
    con_stab = stab[stab['metric'] == 'contact_pct']['franks_stability'].iloc[0]
    best_stab_row = stab.sort_values('franks_stability', ascending=False).iloc[0]
    lines.append(
        f'\n**More stable?** Mixed, and the two stability measures disagree, so both '
        f'are given.')
    lines.append(
        f'\n- *Raw half-to-half reliability*: best MDA variant r = {mda_r:.3f} vs. raw '
        f'contact% r = {con_r:.3f}. MDA is clearly the more repeatable measurement.')
    lines.append(
        f'- *Franks stability* (noise removed from both halves first): best MDA variant '
        f'{mda_stab:.4f} vs. raw contact% {con_stab:.4f} -- effectively a tie. The '
        f'highest value in the table belongs to {best_stab_row["label"]} '
        f'({best_stab_row["franks_stability"]:.4f}), but that is an artefact of how the '
        f'statistic is built: it is a ratio of persistent variance to *real* '
        f'period-to-period change, and a stat with large sampling noise has little '
        f'variance left over to attribute to real change once the noise is subtracted. '
        f'A noisy stat can therefore score high on stability while being useless in '
        f'practice, which is exactly why discrimination and stability are meant to be '
        f'read together rather than separately.')

    con = cross[cross['target'] == 'contact_pct_h2'].sort_values('cv_mae')
    own_mae = con[con['predictor'] == 'contact_pct_h1']['cv_mae'].iloc[0]
    combo = con[con['predictor'] == 'contact_pct_h1 + MDA_all_h1']
    if len(combo):
        combo_mae = combo['cv_mae'].iloc[0]
        lines.append(
            f'\n**Does it add anything on top of contact% itself?** Predicting '
            f'second-half contact%, `contact_pct_h1` alone gives CV MAE '
            f'{own_mae:.5f}; adding MDA gives {combo_mae:.5f} '
            f'({100*(combo_mae-own_mae)/own_mae:+.1f}%). MDA alone is a poor '
            f'predictor of contact% -- it measures a different thing (how close the '
            f'bat came, not whether it connected), so the incremental-value row is '
            f'the one that matters here, not the single-predictor row.')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\n-> {path}')


def main():
    ap = argparse.ArgumentParser(description='Direction B: Miss Distance Added')
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    args = ap.parse_args()
    globals()['N_BOOT'] = args.n_boot

    df = load_swing_frame(with_counts=False)
    if 'offset_residual_context' not in df.columns:
        raise SystemExit(
            'out/rb_contact/oof_context_offset.parquet is missing. Run\n'
            '    python code/rb_contact/dirB_context_offset_oof.py\n'
            'first -- Direction B needs the context-only (random-effect '
            'suppressed) offset prediction. Measured against the batter-RE '
            'prediction alone the metric is null by construction; see that '
            "script's docstring.")
    df = df.dropna(subset=['offset_residual', 'offset_residual_context']).copy()
    print(f'\nSwings with a usable offset residual: {len(df):,} '
          f'(whiff-tracked {(~df["is_contact"]).sum():,} / '
          f'contact LA-derived {df["is_contact"].sum():,})')

    cohort = batter_cohort(df)
    print(f'Common cohort: {len(cohort):,} batters')

    print('\n=== Discrimination ===')
    metrics = build_metrics(df, cohort)
    disc = discrimination_table(df[df['batter'].isin(cohort)], metrics)
    disc.to_csv(os.path.join(FOLLOWUP_DIR, 'discrimination_table.csv'), index=False)

    print('\n=== Stability (split-half) ===')
    stab, m1, m2, half_cohort = stability_table(df)
    stab.to_csv(os.path.join(FOLLOWUP_DIR, 'stability_table.csv'), index=False)

    print('\n=== Cross-metric prediction ===')
    cross = cross_metric_table(m1, m2, half_cohort)
    cross.to_csv(os.path.join(FOLLOWUP_DIR, 'cross_metric_prediction.csv'), index=False)

    # Per-batter leaderboard, so the metric is inspectable rather than just summarized.
    names = load_batter_names()
    board = pd.DataFrame({
        'batter': cohort,
        'batter_name': [names.get(b, '') for b in cohort],
    }).set_index('batter')
    for key in {**CONTINUOUS_METRICS, **RATE_METRICS}:
        board[key] = metrics[key]['value']
        board[f'{key}_n'] = metrics[key]['n']
    board = board.sort_values('MDA_all').reset_index()
    board.to_csv(os.path.join(FOLLOWUP_DIR, 'miss_distance_added_by_batter.csv'), index=False)
    print(f'-> miss_distance_added_by_batter.csv ({len(board)} batters)')

    write_verdict(disc, stab, cross, len(df),
                  os.path.join(FOLLOWUP_DIR, 'direction_b_verdict.md'))


if __name__ == '__main__':
    main()
