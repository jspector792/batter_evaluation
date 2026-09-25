"""
run_change_detection.py
========================
Direction 3: detect within-season shifts in a batter's residual pattern, and
check them against externally verifiable events.

Rolling series (step 1)
-----------------------
Two windows are built and compared for noisiness:
    swing-count  trailing 100 swings
    calendar     trailing 14 days
"Noisier" is measured as the sd of first differences of the rolling series,
normalised by the sd of the series itself -- a series that jitters a lot
relative to its own spread is harder to read change points off.

Change-point detection (step 2)
-------------------------------
CUSUM with a PER-BATTER PERMUTATION-CALIBRATED threshold, rather than a
fixed constant. For each batter the swing order is shuffled many times and
the max CUSUM statistic recorded; the 95th percentile of that null is the
threshold. This fixes the per-batter false-positive rate at ~5% by
construction, which is what makes the population-level false-positive count
in step 4 interpretable instead of an artifact of an arbitrary cutoff.
Detection is recursive (binary segmentation) with a minimum segment length.

Ground truth (step 3)
---------------------
Injured-list transactions from the MLB statsapi, which are objective and
dated. Both directions are used as candidate events:
    placement   going on the IL   (decline may precede it)
    activation  coming off the IL (mechanics may differ after)

Publicly reported mechanical changes and beat-coverage slump narratives are
NOT used. They would have to be assembled by hand for the current season and
scored subjectively; IL transactions are verifiable and unambiguous, so the
hit rate here is against that objective subset only. This narrows what the
check can claim -- an undetected mechanical tweak is not counted as a miss.

Why the random baseline matters (step 4)
-----------------------------------------
With a +/-14 day window and a ~180 day season, a change point placed at
random already has a sizeable chance of landing near some event. The
observed hit rate is therefore reported against a matched random baseline:
the same number of change points per batter, placed uniformly at random over
that batter's own active date range. Without that comparison a hit rate is
uninterpretable.

Outputs (out/exp3_change_detection/)
-------------------------------------
  il_events.csv              ground-truth events pulled from statsapi
  change_points.csv          every detected change point
  changepoint_summary.csv    hit rate, baseline, false-positive rate
  plots/case_<name>.png      rolling series + change points + event dates
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(BASE_DIR, 'code', 'exp_common'))
from base_table import load as load_base  # noqa: E402
from hitters import batter_names  # noqa: E402

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', font_scale=1.0)
BLUE, RED, GREEN, GRAY, PURPLE = ('#2563EB', '#DC2626', '#16A34A', '#6B7280',
                                  '#7C3AED')

OUT_DIR = os.path.join(BASE_DIR, 'out', 'exp3_change_detection')
PLOT_DIR = os.path.join(OUT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

SEASON = 2026
MIN_SWINGS = 300
SWING_WINDOW = 100
DAY_WINDOW = 14
MIN_SEG = 60          # swings; no change point inside this of a boundary
N_PERM = 300
MATCH_DAYS = 14
SEED = 42

METRICS = {'timing': 'timing_residual_context',
           'offset': 'offset_residual_context'}


# ──────────────────────────────────────────────────────────────────────────
# Ground truth
# ──────────────────────────────────────────────────────────────────────────

def fetch_il_events(season: int) -> pd.DataFrame:
    """IL placements and activations for the season, from the MLB statsapi."""
    rows = []
    for m in range(3, 12):
        # Month end via Period, not a hardcoded 31 -- the 30-day months
        # silently 400'd and dropped April, June and September entirely.
        per = pd.Period(f'{season}-{m:02d}', freq='M')
        start = str(per.start_time.date())
        end = str(per.end_time.date())
        try:
            r = requests.get('https://statsapi.mlb.com/api/v1/transactions',
                             params={'startDate': start, 'endDate': end,
                                     'sportId': 1}, timeout=90)
            r.raise_for_status()
            tx = r.json().get('transactions', [])
        except Exception as e:
            print(f'  {start}: fetch failed ({e})')
            continue
        for t in tx:
            desc = (t.get('description') or '').lower()
            if 'injured list' not in desc:
                continue
            pid = (t.get('person') or {}).get('id')
            if pid is None:
                continue
            if 'activated' in desc or 'reinstated' in desc:
                kind = 'activation'
            elif 'placed' in desc or 'transferred' in desc:
                kind = 'placement'
            else:
                continue
            rows.append(dict(batter=str(pid),
                             name=(t.get('person') or {}).get('fullName'),
                             date=pd.to_datetime(t.get('date')), kind=kind,
                             description=t.get('description')))
        time.sleep(0.3)
    df = pd.DataFrame(rows).drop_duplicates(['batter', 'date', 'kind'])
    print(f'IL events fetched: {len(df):,} '
          f'({(df["kind"] == "placement").sum()} placements, '
          f'{(df["kind"] == "activation").sum()} activations)')
    return df


# ──────────────────────────────────────────────────────────────────────────
# Rolling + detection
# ──────────────────────────────────────────────────────────────────────────

def rolling_series(d, col):
    s = d.dropna(subset=[col]).sort_values('game_date')
    if len(s) < MIN_SWINGS:
        return None
    out = pd.DataFrame({'game_date': s['game_date'].values,
                        'v': s[col].to_numpy(float)})
    out['roll_swings'] = out['v'].rolling(SWING_WINDOW, min_periods=30).mean()
    t = out.set_index('game_date')['v']
    out['roll_days'] = (t.rolling(f'{DAY_WINDOW}D', min_periods=20)
                        .mean().to_numpy())
    return out


def noisiness(series):
    x = pd.Series(series).dropna().to_numpy()
    if len(x) < 30 or x.std(ddof=1) == 0:
        return np.nan
    return float(np.std(np.diff(x), ddof=1) / np.std(x, ddof=1))


def _cusum_stat(v):
    """Max |cumulative deviation from the mean|, and where it occurs."""
    c = np.cumsum(v - v.mean())
    i = int(np.argmax(np.abs(c)))
    return float(abs(c[i])), i


def detect(v, rng, n_perm=N_PERM, min_seg=MIN_SEG, depth=0, offset=0):
    """Binary segmentation on CUSUM with a permutation-calibrated threshold."""
    n = len(v)
    if n < 2 * min_seg or depth > 3:
        return []
    stat, idx = _cusum_stat(v)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = _cusum_stat(rng.permutation(v))[0]
    thresh = float(np.quantile(null, 0.95))
    if stat <= thresh or idx < min_seg or idx > n - min_seg:
        return []
    cps = [dict(index=offset + idx, stat=stat, threshold=thresh,
                ratio=stat / thresh if thresh > 0 else np.nan)]
    cps += detect(v[:idx], rng, n_perm, min_seg, depth + 1, offset)
    cps += detect(v[idx:], rng, n_perm, min_seg, depth + 1, offset + idx)
    return cps


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Direction 3: change detection')
    ap.add_argument('--season', type=int, default=SEASON)
    ap.add_argument('--n-cases', type=int, default=6)
    ap.add_argument('--refresh-il', action='store_true')
    args = ap.parse_args()

    df = load_base(args.season)
    names = batter_names(args.season)

    il_path = os.path.join(OUT_DIR, 'il_events.csv')
    if args.refresh_il or not os.path.exists(il_path):
        il = fetch_il_events(args.season)
        il.to_csv(il_path, index=False)
    else:
        il = pd.read_csv(il_path, parse_dates=['date'])
        il['batter'] = il['batter'].astype(str)
    print(f'IL events: {len(il):,} rows, {il["batter"].nunique()} players')

    rng = np.random.default_rng(SEED)
    counts = df.groupby('batter').size()
    cohort = counts[counts >= MIN_SWINGS].index
    print(f'cohort: {len(cohort)} batters with >= {MIN_SWINGS} swings')

    cp_rows, noise_rows, series_cache = [], [], {}
    for bid in cohort:
        d = df[df['batter'] == bid]
        for metric, col in METRICS.items():
            s = rolling_series(d, col)
            if s is None:
                continue
            noise_rows.append(dict(batter=bid, metric=metric,
                                   n=len(s),
                                   noisiness_swings=noisiness(s['roll_swings']),
                                   noisiness_days=noisiness(s['roll_days'])))
            cps = detect(s['v'].to_numpy(), rng)
            for c in cps:
                cp_rows.append(dict(batter=bid, name=names.get(bid, bid),
                                    metric=metric,
                                    date=s['game_date'].iloc[c['index']],
                                    **c))
            series_cache[(bid, metric)] = s

    cps = pd.DataFrame(cp_rows)
    noise = pd.DataFrame(noise_rows)
    cps.to_csv(os.path.join(OUT_DIR, 'change_points.csv'), index=False)

    print(f'\nchange points detected: {len(cps):,} across '
          f'{cps["batter"].nunique() if len(cps) else 0} batters')
    print(f'median per batter-metric: '
          f'{cps.groupby(["batter", "metric"]).size().median() if len(cps) else 0}')
    print('\n=== Window noisiness (lower = smoother) ===')
    print(noise[['noisiness_swings', 'noisiness_days']].describe()
          .loc[['mean', '50%']].to_string(float_format=lambda v: f'{v:,.4f}'))

    # ── Hit rate vs a matched random baseline ─────────────────────────────
    il_c = il[il['batter'].isin(set(cohort))]
    span = df.groupby('batter')['game_date'].agg(['min', 'max'])
    res = []
    for metric in METRICS:
        m_cps = cps[cps['metric'] == metric] if len(cps) else cps
        ev = il_c.merge(span, left_on='batter', right_index=True, how='inner')
        ev = ev[(ev['date'] >= ev['min']) & (ev['date'] <= ev['max'])]
        if ev.empty or m_cps.empty:
            continue

        def hit_rate(cp_table):
            hits = 0
            for _, e in ev.iterrows():
                c = cp_table[cp_table['batter'] == e['batter']]
                if c.empty:
                    continue
                if (abs((pd.to_datetime(c['date']) - e['date']).dt.days)
                        <= MATCH_DAYS).any():
                    hits += 1
            return hits / len(ev)

        obs = hit_rate(m_cps)
        # Matched random baseline: same count per batter, uniform over that
        # batter's own active window.
        base = []
        for _ in range(60):
            fake = []
            for bid, g in m_cps.groupby('batter'):
                lo, hi = span.loc[bid, 'min'], span.loc[bid, 'max']
                days = max((hi - lo).days, 1)
                for _ in range(len(g)):
                    fake.append(dict(batter=bid,
                                     date=lo + pd.Timedelta(
                                         days=int(rng.integers(0, days)))))
            base.append(hit_rate(pd.DataFrame(fake)))
        base_mean = float(np.mean(base))

        # False positives: change points with no event within the window.
        fp = 0
        for _, c in m_cps.iterrows():
            e = ev[ev['batter'] == c['batter']]
            if e.empty or not (abs((e['date'] - c['date']).dt.days)
                               <= MATCH_DAYS).any():
                fp += 1
        res.append(dict(metric=metric, n_events=len(ev),
                        n_change_points=len(m_cps),
                        hit_rate=obs, random_baseline=base_mean,
                        lift=obs - base_mean,
                        false_positive_rate=fp / len(m_cps)))
    summ = pd.DataFrame(res)
    summ.to_csv(os.path.join(OUT_DIR, 'changepoint_summary.csv'), index=False)

    pd.set_option('display.width', 210)
    print('\n=== Ground-truth check (IL placements + activations) ===\n')
    print(summ.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))

    # ── Case plots: batters with both an IL event and a change point ──────
    cases = (il_c[il_c['batter'].isin(set(cps['batter']))]['batter']
             .value_counts().head(args.n_cases).index if len(cps) else [])
    for bid in cases:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True,
                                 constrained_layout=True)
        nm = names.get(bid, bid)
        for ax, (metric, col) in zip(axes, METRICS.items()):
            s = series_cache.get((bid, metric))
            if s is None:
                ax.set_visible(False)
                continue
            ax.plot(s['game_date'], s['roll_swings'], color=BLUE, lw=1.6,
                    label=f'{SWING_WINDOW}-swing rolling mean')
            ax.plot(s['game_date'], s['roll_days'], color=GREEN, lw=1.2,
                    alpha=0.8, label=f'{DAY_WINDOW}-day rolling mean')
            for _, c in cps[(cps['batter'] == bid) &
                            (cps['metric'] == metric)].iterrows():
                ax.axvline(c['date'], color=RED, ls='-', lw=1.8, alpha=0.8)
            for _, e in il_c[il_c['batter'] == bid].iterrows():
                ax.axvline(e['date'], color=PURPLE, ls='--', lw=1.8)
            ax.set_ylabel(f'{metric} residual')
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc='upper left')
        axes[0].set_title(f'{nm} — red = detected change point, '
                          f'purple dashed = IL transaction', fontsize=12)
        fig.savefig(os.path.join(PLOT_DIR,
                                 f'case_{str(nm).replace(" ", "_")}.png'),
                    dpi=135, bbox_inches='tight')
        plt.close(fig)
    print(f'\n-> {OUT_DIR}  ({len(list(cases))} case plots)')


if __name__ == '__main__':
    main()
