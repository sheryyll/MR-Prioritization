import numpy as np
from mrrank.evaluation_module import KILL_THRESHOLD


def test_kill_threshold_is_strict_greater_than():
    """Boundary: exactly 5pp difference must NOT count as killed."""
    clean_rate = 0.10
    mutant_rate_exact = 0.15  # exactly +5pp
    mutant_rate_over = 0.1501  # just over
    assert not ((mutant_rate_exact - clean_rate) > KILL_THRESHOLD)
    assert (mutant_rate_over - clean_rate) > KILL_THRESHOLD


def test_fdr_computation_from_synthetic_matrix():
    km = np.array([
        [True, True, False, False],
        [False, False, False, False],
        [True, True, True, True],
    ])
    fdr = km.sum(axis=1) / km.shape[1]
    assert fdr[0] == 0.5
    assert fdr[1] == 0.0
    assert fdr[2] == 1.0