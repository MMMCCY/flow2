"""Small required correctness fixtures; no old tests or experiment runners."""
import unittest

import numpy as np

from metrics import average_precision, fixed_top5


class RankingChecks(unittest.TestCase):
    def test_constant_scores_equal_prevalence(self):
        ap, curve = average_precision([1, 0, 0, 1, 0], np.zeros(5))
        self.assertEqual(ap, 2/5)
        self.assertEqual(curve["selected_count"].tolist(), [5])

    def test_tied_positives_enter_together(self):
        # A negative alone, then two positives tied: AP=(2/2)*(2/3).
        ap, _ = average_precision([0, 1, 1, 0], [.9, .5, .5, .1])
        self.assertAlmostEqual(ap, 2/3)
        other, _ = average_precision([0, 0, 1, 1], [.1, .9, .5, .5])
        self.assertEqual(ap, other)

    def test_documented_sklearn_example(self):
        ap, _ = average_precision([0, 0, 1, 1], [.1, .4, .35, .8])
        self.assertAlmostEqual(ap, 5/6)

    def test_exhaustive_small_binary_tie_orders(self):
        # Independent integration over threshold-defined sets, including ties.
        from itertools import product
        scores=np.array([2., 1., 1., 0.])
        for truth in product((False, True), repeat=4):
            if not any(truth):
                continue
            truth=np.array(truth)
            expected, previous=0.,0.
            for threshold in [2.,1.,0.]:
                selected=scores >= threshold
                recall=float(truth[selected].sum()/truth.sum())
                precision=float(truth[selected].mean())
                expected+=(recall-previous)*precision
                previous=recall
            actual,_=average_precision(truth,scores)
            self.assertAlmostEqual(actual,expected)

    def test_top5_includes_ties_without_truth(self):
        score=np.r_[np.zeros(90),np.ones(10)]
        selected, threshold, requested=fixed_top5(score)
        self.assertEqual(requested,5)
        self.assertEqual(int(selected.sum()),10)
        self.assertEqual(threshold,1.)
        selected, _, _=fixed_top5(np.ones(20))
        self.assertTrue(selected.all())


if __name__ == "__main__":
    unittest.main()
