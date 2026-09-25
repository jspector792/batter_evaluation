"""
run_suite.py
=============
Orchestrator for the miss-distance model suite. Runs the four stages in order
and then writes a consolidated report.

    1. suite_stage1.py       the three stage-1 regressor variants
    2. suite_stage2.py       contact classifier per variant + raw baseline
    3. suite_diagnostics.py  hex predicted-vs-actual grids + fit metrics
    4. suite_eval.py         Directions A / B / C on the 2026 season
    5. this module           suite_report.md

Each stage is a normal script and can be run on its own; this just chains them
and stops at the first failure so a broken stage does not silently propagate
half-written artifacts into the next one.

Usage
-----
  python run_suite.py                     # everything
  python run_suite.py --from diagnostics  # resume partway
  python run_suite.py --report-only       # rebuild the report from artifacts
"""

import os
import sys
import argparse
import subprocess

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite_config import (OUT_DIR, PLOT_DIR, VARIANT_ORDER, TRAIN_SEASON,
                          EVAL_SEASON, VARIANTS, MIN_BAT_SPEED)

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ['stage1', 'stage2', 'diagnostics', 'eval']
SCRIPT = {'stage1': 'suite_stage1.py', 'stage2': 'suite_stage2.py',
          'diagnostics': 'suite_diagnostics.py', 'eval': 'suite_eval.py'}


def run_stage(stage: str, extra: list[str]) -> None:
    cmd = [sys.executable, '-u', os.path.join(HERE, SCRIPT[stage])] + extra
    print(f'\n{"#" * 72}\n# {stage}: {" ".join(cmd[1:])}\n{"#" * 72}',
          flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        raise SystemExit(f'{stage} failed with exit code {r.returncode}')


def _read(name: str) -> pd.DataFrame | None:
    path = os.path.join(OUT_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def _md(df: pd.DataFrame, floats: int = 4) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].astype(float).round(floats)
    head = '| ' + ' | '.join(str(c) for c in d.columns) + ' |'
    sep = '|' + '|'.join(['---'] * len(d.columns)) + '|'
    body = '\n'.join('| ' + ' | '.join('' if pd.isna(v) else str(v)
                                       for v in row) + ' |'
                     for row in d.itertuples(index=False))
    return '\n'.join([head, sep, body])


def write_report() -> str:
    L = []
    L.append('# Miss-distance model suite\n')
    L.append(f'Training season **{TRAIN_SEASON}** (spring training, regular '
             f'season and postseason), evaluation season **{EVAL_SEASON}**. '
             f'No {EVAL_SEASON} row takes part in any fit.\n')
    L.append(f'Swing population: bunts excluded, `bat_speed >= '
             f'{MIN_BAT_SPEED:g}` mph. Stage-1 offset target is tracked '
             f'`miss_distance` in inches with **contact assigned 0**, '
             f'replacing `barrel_distance_v2`.\n')

    L.append('\n## Variants\n')
    rows = []
    for v in VARIANT_ORDER:
        cfg = VARIANTS[v]
        rows.append(dict(
            variant=v, kind=cfg['kind'],
            timing_features=', '.join(cfg['timing']['features']),
            offset_features=', '.join(cfg['offset']['features']),
            batter_aggregates=', '.join(cfg['timing']['batter_agg_features'])
            or '(none)'))
    L.append(_md(pd.DataFrame(rows)))

    L.append('\n\n## Stage-1 fit quality\n')
    L.append('`all_swings` includes the assigned zeros; `whiffs_only` is the '
             'population where miss distance is genuinely tracked and is the '
             'honest read for the offset model.\n')
    met = _read('stage1_fit_metrics.csv')
    if met is not None:
        for target in sorted(met['target'].unique()):
            for pop in sorted(met[met['target'] == target]['population'].unique()):
                sub = met[(met['target'] == target) &
                          (met['population'] == pop)]
                L.append(f'\n**{target} / {pop}**\n')
                L.append(_md(sub[['variant', 'context', 'n', 'r2', 'rmse',
                                  'mae', 'spearman', 'bias']]))
                L.append('')
    else:
        L.append('_stage1_fit_metrics.csv not found._')

    L.append('\n## Direction B — discrimination and stability\n')
    L.append('Discrimination is the share of between-batter variance that is '
             'signal rather than sampling noise. MDA is negative-is-good.\n')
    disc = _read('dirB_discrimination.csv')
    if disc is not None:
        L.append(_md(disc))
    stab = _read('dirB_stability.csv')
    if stab is not None:
        L.append('\n')
        L.append(_md(stab))

    L.append('\n\n## Direction A — stabilization\n')
    a = _read('dirA_stabilization.csv')
    if a is not None:
        # One representative slice: the honest disjoint design, RE suppressed.
        sub = a[(a['design'] == 'holdout') & (a['mode'] == 'context')]
        if not sub.empty:
            piv = sub.pivot_table(index=['variant', 'sample_fraction'],
                                  columns='estimator', values='rmse')
            L.append('RMSE against the held-out remainder of '
                     f'{EVAL_SEASON}, random effects suppressed:\n')
            L.append(_md(piv.reset_index()))
        yoy = a[a['design'].str.startswith('yoy')]
        if not yoy.empty:
            piv = (yoy[yoy['mode'] == 'context']
                   .pivot_table(index=['variant', 'sample_fraction'],
                                columns='estimator', values='rmse'))
            L.append(f'\n\nYear-over-year ({TRAIN_SEASON} early sample -> full '
                     f'{EVAL_SEASON} contact%):\n')
            L.append(_md(piv.reset_index()))

    L.append('\n\n## Reading these results\n')
    L.append(
        '**1. The m2 timing model is not a timing expectation.** Adding the\n'
        'bat-tracking block lifts timing R^2 from ~0.55 (m1) to ~0.93 (m2),\n'
        'but that gain is geometric restatement, not prediction.\n'
        '`attack_direction` correlates with `int_y` at r = -0.913 and explains\n'
        'R^2 = 0.834 of it on its own; all three bat-tracking variables\n'
        'together give linear R^2 = 0.852, against 0.417 for the entire\n'
        'pitch-context block. `attack_direction` is the bat\'s horizontal\n'
        'heading at the bat-ball intercept and `int_y` is the depth of that\n'
        'same intercept -- they are two coordinates of one tracked event, so\n'
        'regressing one on the other mostly recovers swing geometry.\n'
        '\n'
        'The practical consequence: for m2, `predicted_timing` ~ actual\n'
        '`int_y`, so `timing_residual` collapses toward zero (mean 0.057 vs\n'
        'm1\'s 0.194) and the Direction C timing diagnostic loses its meaning\n'
        'for that variant. If the goal is "how early/late was this batter\n'
        'against this pitch", keep bat tracking OUT of the timing model.\n')
    L.append(
        '\n**2. The offset gain from bat tracking IS real.** On whiffs only --\n'
        'the population where miss distance is actually tracked rather than\n'
        'assigned -- m2 reaches R^2 0.75 vs m1 0.70 and m3 0.70. Nothing in\n'
        'the bat-tracking block is definitionally tied to miss distance, so\n'
        'this is a genuine improvement rather than an identity.\n')
    L.append(
        '\n**3. Zero-inflation biases the offset model on whiffs.** The\n'
        '`all_swings` R^2 (~0.72-0.79) is inflated: 78% of the target is the\n'
        'assigned zero, and predicting zero for contact is easy. The\n'
        'whiffs-only rows are the honest read. The positive bias there\n'
        '(+0.42 to +0.82 inches) is the model being pulled toward the zero\n'
        'mass, so it systematically UNDER-predicts how far a real whiff\n'
        'missed by.\n')
    L.append(
        '\n**4. The random effect matters far more for timing than for\n'
        'offset.** Scoring 2026, the carry-minus-context spread is sd 2.90 in\n'
        '(m1) and 2.67 in (m2) on timing, but only 0.23 in and 0.16 in on\n'
        'miss distance. For m3 it is exactly 0.0 by construction, which makes\n'
        'm3 the reference line for how much of any advantage is batter\n'
        'identity rather than pitch-context modelling.\n')
    L.append(
        '\n**5. `raw_contact_pct` and `raw_whiff_pct` necessarily share a\n'
        'discrimination value** (0.886) -- whiff% = 1 - contact%, so the\n'
        'observed and sampling variances are identical. The duplicate row is\n'
        'a reference point, not a finding.\n')

    L.append('\n\n## Figures\n')
    for f in sorted(os.listdir(PLOT_DIR)) if os.path.isdir(PLOT_DIR) else []:
        L.append(f'- `plots/{f}`')

    report = '\n'.join(L) + '\n'
    path = os.path.join(OUT_DIR, 'suite_report.md')
    with open(path, 'w') as fh:
        fh.write(report)
    print(f'\n-> {path}')
    return path


def main():
    ap = argparse.ArgumentParser(description='Run the miss-distance suite')
    ap.add_argument('--from', dest='start', choices=STAGES, default='stage1')
    ap.add_argument('--only', nargs='*', choices=STAGES)
    ap.add_argument('--variants', nargs='*', default=None)
    ap.add_argument('--report-only', action='store_true')
    ap.add_argument('--skip-existing', action='store_true')
    args = ap.parse_args()

    if not args.report_only:
        stages = args.only or STAGES[STAGES.index(args.start):]
        for stage in stages:
            extra = []
            if args.variants:
                extra += ['--variants'] + args.variants
            if stage == 'stage1' and args.skip_existing:
                extra += ['--skip-existing']
            run_stage(stage, extra)

    write_report()


if __name__ == '__main__':
    main()
