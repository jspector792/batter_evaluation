"""
Testing checklist item: "Unit test the Beta-Binomial shrinkage function
against a known closed-form case (e.g., a batter with prior-mean p_i for all
swings should shrink to exactly the prior)."
"""
import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shrinkage import fit_beta_prior_moments, shrink_beta_binomial


class TestShrinkage(unittest.TestCase):

    def test_prior_mean_input_shrinks_to_itself(self):
        """A batter whose observed rate exactly equals the prior mean should
        shrink to exactly that same value, regardless of swing count."""
        alpha, beta = 8.0, 22.0
        prior_mean = alpha / (alpha + beta)
        for n in [1, 10, 100, 10_000]:
            shrunk = shrink_beta_binomial(np.array([prior_mean]), np.array([n]), alpha, beta)
            self.assertAlmostEqual(shrunk[0], prior_mean, places=10,
                                   msg=f'failed at n={n}')

    def test_zero_swings_gives_pure_prior(self):
        """With zero swings, the shrunk estimate must equal the prior mean
        regardless of the (meaningless, since n=0) input rate."""
        alpha, beta = 8.0, 22.0
        prior_mean = alpha / (alpha + beta)
        for p_hat in [0.0, 0.3, 0.65, 1.0]:
            shrunk = shrink_beta_binomial(np.array([p_hat]), np.array([0.0]), alpha, beta)
            self.assertAlmostEqual(shrunk[0], prior_mean, places=10,
                                   msg=f'failed at p_hat={p_hat}')

    def test_large_n_converges_to_raw_rate(self):
        """As swing count grows, the shrunk estimate should approach the raw
        observed rate (the prior's influence vanishes)."""
        alpha, beta = 8.0, 22.0
        p_hat = 0.55
        shrunk_small = shrink_beta_binomial(np.array([p_hat]), np.array([10.0]), alpha, beta)[0]
        shrunk_large = shrink_beta_binomial(np.array([p_hat]), np.array([100_000.0]), alpha, beta)[0]
        self.assertLess(abs(shrunk_large - p_hat), abs(shrunk_small - p_hat))
        self.assertAlmostEqual(shrunk_large, p_hat, places=3)

    def test_shrinkage_direction(self):
        """A rate above the prior mean should shrink downward; below should
        shrink upward -- shrinkage always moves toward the prior mean."""
        alpha, beta = 8.0, 22.0
        prior_mean = alpha / (alpha + beta)

        high = shrink_beta_binomial(np.array([0.9]), np.array([50.0]), alpha, beta)[0]
        self.assertLess(high, 0.9)
        self.assertGreater(high, prior_mean)

        low = shrink_beta_binomial(np.array([0.05]), np.array([50.0]), alpha, beta)[0]
        self.assertGreater(low, 0.05)
        self.assertLess(low, prior_mean)

    def test_fit_beta_prior_moments_recovers_known_params(self):
        """Simulate batter rates from a known Beta(alpha, beta) and confirm
        method-of-moments recovers approximately the same parameters."""
        rng = np.random.default_rng(42)
        true_alpha, true_beta = 8.0, 22.0
        rates = rng.beta(true_alpha, true_beta, size=20_000)
        alpha_hat, beta_hat = fit_beta_prior_moments(rates)
        self.assertAlmostEqual(alpha_hat, true_alpha, delta=0.5)
        self.assertAlmostEqual(beta_hat, true_beta, delta=1.5)

    def test_fit_beta_prior_moments_rejects_inconsistent_variance(self):
        """Variance >= mean*(1-mean) is impossible for a Beta distribution --
        must raise rather than silently return nonsense (e.g. negative
        alpha/beta)."""
        degenerate = np.array([0.0, 1.0] * 100)  # var = 0.25 = mean*(1-mean) at mean=0.5
        with self.assertRaises(ValueError):
            fit_beta_prior_moments(degenerate)


if __name__ == '__main__':
    unittest.main()
