"""
test_followups.py
==================
Unit tests for the Direction A/B/C follow-up analyses. Pure-function tests
only -- no parquet reads, no model fits -- so the suite stays fast and
runnable without the full dataset. Integration-shaped checks that need the
real artifacts live in test_pipeline_outputs.py.

Run with the rest of the suite:  python code/rb_contact/tests/run_all.py
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from followup_utils import (walk_counts, chronological_prefix, discrimination,
                            discrimination_bootstrap, stability_two_period,
                            rate_stat, mean_stat, rmse, md_table)


class TestCountReconstruction(unittest.TestCase):
    """The count is derived, not read from a column (data_audit.md section 2
    flagged balls/strikes as absent), so it needs its own correctness tests."""

    def test_count_resets_each_at_bat(self):
        desc = np.array(['ball', 'ball', 'called_strike', 'ball', 'called_strike'])
        ab = np.array([1, 1, 1, 2, 2])
        b, s = walk_counts(desc, ab)
        self.assertEqual(list(b), [0, 1, 2, 0, 1])
        self.assertEqual(list(s), [0, 0, 0, 0, 0])

    def test_foul_does_not_add_third_strike(self):
        # 0-2 then three fouls: the count must stay 0-2 the whole way.
        desc = np.array(['called_strike', 'swinging_strike', 'foul', 'foul', 'foul'])
        ab = np.array([1] * 5)
        b, s = walk_counts(desc, ab)
        self.assertEqual(list(s), [0, 1, 2, 2, 2])
        self.assertEqual(list(b), [0, 0, 0, 0, 0])

    def test_foul_bunt_does_add_third_strike(self):
        # A foul bunt with two strikes IS a strikeout, unlike a normal foul.
        desc = np.array(['called_strike', 'called_strike', 'foul_bunt'])
        ab = np.array([1, 1, 1])
        b, s = walk_counts(desc, ab)
        self.assertEqual(list(s), [0, 1, 2])

    def test_blocked_and_automatic_variants_count(self):
        desc = np.array(['blocked_ball', 'automatic_ball', 'pitchout',
                         'swinging_strike_blocked', 'automatic_strike'])
        ab = np.array([1] * 5)
        b, s = walk_counts(desc, ab)
        self.assertEqual(list(b), [0, 1, 2, 3, 3])
        self.assertEqual(list(s), [0, 0, 0, 0, 1])

    def test_count_never_exceeds_3_2(self):
        rng = np.random.default_rng(0)
        desc = rng.choice(['ball', 'called_strike', 'foul', 'swinging_strike'], size=500)
        ab = np.repeat(np.arange(50), 10)
        b, s = walk_counts(desc, ab)
        self.assertTrue((b <= 3).all() and (b >= 0).all())
        self.assertTrue((s <= 2).all() and (s >= 0).all())

    def test_every_at_bat_starts_0_0(self):
        rng = np.random.default_rng(1)
        desc = rng.choice(['ball', 'called_strike', 'foul'], size=300)
        ab = np.repeat(np.arange(30), 10)
        b, s = walk_counts(desc, ab)
        firsts = np.r_[True, ab[1:] != ab[:-1]]
        self.assertTrue((b[firsts] == 0).all() and (s[firsts] == 0).all())


class TestChronologicalPrefix(unittest.TestCase):
    """Direction A's early sample must be chronological and must not leak
    anything from later in the season."""

    def _frame(self, n_per_batter=(100, 40)):
        rows = []
        for bi, n in enumerate(n_per_batter):
            for i in range(n):
                rows.append(dict(
                    batter=f'b{bi}',
                    game_date=pd.Timestamp('2025-04-01') + pd.Timedelta(days=i),
                    game_pk=1000 + i, at_bat_number=1, pitch_number=1, seq=i))
        return pd.DataFrame(rows).sample(frac=1, random_state=0)  # shuffled input

    def test_prefix_size_is_ceil(self):
        df = self._frame((100, 40))
        out = chronological_prefix(df, 0.10)
        sizes = out.groupby('batter').size().to_dict()
        self.assertEqual(sizes['b0'], 10)
        self.assertEqual(sizes['b1'], 4)

    def test_prefix_rounds_up_and_keeps_at_least_one(self):
        df = self._frame((7,))
        self.assertEqual(len(chronological_prefix(df, 0.05)), 1)
        self.assertEqual(len(chronological_prefix(df, 0.20)), 2)  # ceil(1.4)

    def test_prefix_contains_only_earliest_swings(self):
        """The defining leakage guard: nothing after the cutoff may appear."""
        df = self._frame((100, 40))
        for frac in [0.05, 0.25, 0.5]:
            out = chronological_prefix(df, frac)
            for b, g in out.groupby('batter'):
                k = len(g)
                self.assertEqual(sorted(g['seq']), list(range(k)),
                                 f'batter {b} at frac {frac} did not get the first {k} swings')

    def test_prefix_is_order_invariant(self):
        df = self._frame((100, 40))
        a = set(chronological_prefix(df, 0.3)['seq'].astype(str) +
                chronological_prefix(df, 0.3)['batter'])
        shuffled = df.sample(frac=1, random_state=99)
        b = set(chronological_prefix(shuffled, 0.3)['seq'].astype(str) +
                chronological_prefix(shuffled, 0.3)['batter'])
        self.assertEqual(a, b)

    def test_prefix_is_nested_across_fractions(self):
        df = self._frame((100,))
        small = set(chronological_prefix(df, 0.10)['seq'])
        large = set(chronological_prefix(df, 0.30)['seq'])
        self.assertTrue(small.issubset(large))


class TestDiscrimination(unittest.TestCase):
    """Franks et al. discrimination: 0 when the spread is pure noise, near 1
    when the metric is measured essentially without error."""

    def test_pure_noise_gives_zero(self):
        # All batters share one true value -- every observed difference is
        # noise, so D should be 0. It is estimated as
        # (Var_obs - mean sampling var)/Var_obs, and Var_obs over k batters
        # has relative SD ~ sqrt(2/(k-1)); at k=400 that is ~7%, so D bounces
        # around 0 by ~0.07 for reasons that have nothing to do with the
        # estimator. k=5000 puts the Monte Carlo error at ~2%, which is what
        # makes the assertion below decisive rather than seed-dependent.
        rng = np.random.default_rng(0)
        n_per, n_batters = 100, 5000
        values, svars = [], []
        for _ in range(n_batters):
            draws = rng.normal(0.0, 1.0, n_per)
            values.append(draws.mean())
            svars.append(draws.var(ddof=1) / n_per)
        self.assertLess(abs(discrimination(np.array(values), np.array(svars))), 0.05)

    def test_no_sampling_noise_gives_one(self):
        rng = np.random.default_rng(0)
        values = rng.normal(0, 1, 300)
        self.assertAlmostEqual(discrimination(values, np.zeros(300)), 1.0, places=6)

    def test_known_signal_noise_split_recovered(self):
        # True skill var = 1.0, sampling var = 1.0 -> discrimination should be 0.5.
        rng = np.random.default_rng(7)
        true = rng.normal(0, 1.0, 4000)
        observed = true + rng.normal(0, 1.0, 4000)
        d = discrimination(observed, np.full(4000, 1.0))
        self.assertGreater(d, 0.45)
        self.assertLess(d, 0.55)

    def test_clipped_at_zero_not_negative(self):
        # Sampling variance overstated well past the observed spread.
        self.assertEqual(discrimination(np.array([0.1, 0.2, 0.3]), np.full(3, 100.0)), 0.0)

    def test_bootstrap_agrees_with_analytic(self):
        rng = np.random.default_rng(3)
        rows = []
        for b in range(120):
            mu = rng.normal(0, 1.0)
            for v in rng.normal(mu, 1.0, 80):
                rows.append((f'b{b}', v))
        df = pd.DataFrame(rows, columns=['batter', 'v'])
        m = mean_stat(df, 'v')
        analytic = discrimination(m['value'].values, m['sampling_var'].values)
        boot = discrimination_bootstrap(df, 'v', n_boot=200, seed=1)
        self.assertLess(abs(analytic - boot), 0.05)


class TestStability(unittest.TestCase):

    def test_perfectly_persistent_metric(self):
        """No real change between periods and no sampling noise -> stability 1."""
        rng = np.random.default_rng(0)
        theta = rng.normal(0, 1, 500)
        out = stability_two_period(theta, theta.copy(), np.zeros(500), np.zeros(500))
        self.assertAlmostEqual(out['stability'], 1.0, places=6)

    def test_pure_change_metric(self):
        """Independent draws each period -> nothing persistent -> stability ~0."""
        rng = np.random.default_rng(0)
        x1, x2 = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
        out = stability_two_period(x1, x2, np.zeros(5000), np.zeros(5000))
        self.assertLess(out['stability'], 0.1)

    def test_sampling_noise_excluded_from_change(self):
        """Noise-only differences must not be charged as real change."""
        rng = np.random.default_rng(5)
        theta = rng.normal(0, 1.0, 6000)
        x1 = theta + rng.normal(0, 0.5, 6000)
        x2 = theta + rng.normal(0, 0.5, 6000)
        out = stability_two_period(x1, x2, np.full(6000, 0.25), np.full(6000, 0.25))
        self.assertGreater(out['stability'], 0.9)

    def test_clipping_is_reported(self):
        rng = np.random.default_rng(0)
        x1, x2 = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
        out = stability_two_period(x1, x2, np.full(200, 50.0), np.full(200, 50.0))
        self.assertTrue(out['clipped'])


class TestAggregationHelpers(unittest.TestCase):

    def test_rate_stat_binomial_variance(self):
        df = pd.DataFrame({'batter': ['a'] * 10 + ['b'] * 20,
                           'hit': [True] * 5 + [False] * 5 + [True] * 15 + [False] * 5})
        out = rate_stat(df, 'hit')
        self.assertAlmostEqual(out.loc['a', 'value'], 0.5)
        self.assertAlmostEqual(out.loc['a', 'sampling_var'], 0.5 * 0.5 / 10)
        self.assertAlmostEqual(out.loc['b', 'value'], 0.75)
        self.assertAlmostEqual(out.loc['b', 'sampling_var'], 0.75 * 0.25 / 20)

    def test_rate_stat_subset_mask_changes_denominator(self):
        df = pd.DataFrame({'batter': ['a'] * 4,
                           'hit': [True, True, False, False],
                           'in_zone': [True, True, True, False]})
        out = rate_stat(df, 'hit', df['in_zone'])
        self.assertEqual(out.loc['a', 'n'], 3)
        self.assertAlmostEqual(out.loc['a', 'value'], 2 / 3)

    def test_mean_stat_sampling_variance(self):
        df = pd.DataFrame({'batter': ['a'] * 5, 'v': [1.0, 2.0, 3.0, 4.0, 5.0]})
        out = mean_stat(df, 'v')
        self.assertAlmostEqual(out.loc['a', 'value'], 3.0)
        self.assertAlmostEqual(out.loc['a', 'sampling_var'], np.var([1, 2, 3, 4, 5], ddof=1) / 5)

    def test_mean_stat_ignores_nulls_in_denominator(self):
        df = pd.DataFrame({'batter': ['a'] * 5, 'v': [1.0, 2.0, np.nan, np.nan, 3.0]})
        out = mean_stat(df, 'v')
        self.assertEqual(out.loc['a', 'n'], 3)
        self.assertAlmostEqual(out.loc['a', 'value'], 2.0)

    def test_rmse_ignores_non_finite_pairs(self):
        a = np.array([1.0, 2.0, np.nan])
        b = np.array([1.0, 4.0, 10.0])
        self.assertAlmostEqual(rmse(a, b), np.sqrt((0 + 4) / 2))

    def test_md_table_rounds_float32(self):
        """float32 rounded in place prints as 0.5099999904632568."""
        df = pd.DataFrame({'x': np.array([0.51, 1.25], dtype=np.float32)})
        self.assertIn('0.51', md_table(df, 2))
        self.assertNotIn('0.5099999', md_table(df, 2))


class TestMDAFailureMode(unittest.TestCase):
    """
    Guards the finding that motivated dirB_context_offset_oof.py: a residual
    taken against a prediction that already contains a per-batter intercept
    is self-cancelling across any within-batter split, so it cannot support a
    skill stat. If a future refactor quietly points Direction B back at
    `offset_residual`, this reproduces the symptom it should have caught.
    """

    def test_residual_against_own_batter_mean_anticorrelates_across_halves(self):
        rng = np.random.default_rng(0)
        rows = []
        for b in range(300):
            skill = rng.normal(0, 1.0)
            vals = skill + rng.normal(0, 1.0, 200)
            # A "prediction" carrying the batter's own fitted intercept.
            pred = np.full(200, vals.mean())
            for i, (v, p) in enumerate(zip(vals, pred)):
                rows.append((f'b{b}', 'h1' if i < 100 else 'h2', v - p))
        df = pd.DataFrame(rows, columns=['batter', 'half', 'resid'])
        piv = df.pivot_table(index='batter', columns='half', values='resid', aggfunc='mean')
        r = np.corrcoef(piv['h1'], piv['h2'])[0, 1]
        self.assertLess(r, -0.9, 'per-batter-centred residual should anticorrelate ~-1')

        m = mean_stat(df, 'resid')
        self.assertAlmostEqual(discrimination(m['value'].values, m['sampling_var'].values),
                               0.0, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
