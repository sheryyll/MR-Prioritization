import numpy as np
from mrrank.metrics import compute_apfd, compute_fd_at_k


def _toy_matrix():
    # 2 MRs x 3 mutants
    return np.array([[True, False, True], [False, True, False]])


def test_apfd_best_case_first_mr_kills_everything():
    km = np.array([[True, True, True], [False, False, False]])
    idx = {"A": 0, "B": 1}
    apfd = compute_apfd(["A", "B"], km, idx)
    assert apfd > 0.9


def test_apfd_worst_case_last_mr_kills_everything():
    km = np.array([[False, False, False], [True, True, True]])
    idx = {"A": 0, "B": 1}
    apfd = compute_apfd(["A", "B"], km, idx)
    assert apfd < 0.6


def test_fd_at_k_full_k_equals_total_detectable_fraction():
    km = _toy_matrix()
    idx = {"A": 0, "B": 1}
    assert compute_fd_at_k(["A", "B"], km, idx, k=2) == 1.0


def test_fd_at_k_monotonically_nondecreasing():
    km = _toy_matrix()
    idx = {"A": 0, "B": 1}
    fd1 = compute_fd_at_k(["A", "B"], km, idx, k=1)
    fd2 = compute_fd_at_k(["A", "B"], km, idx, k=2)
    assert fd2 >= fd1