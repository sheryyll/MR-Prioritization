import numpy as np
from mrrank.greedy import compute_norm_cost, get_kill_sets, greedy_rank, jaccard


def test_greedy_prefers_diverse_second_pick():
    km = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0]], bool)
    ids = ["A", "B", "C"]
    ks = get_kill_sets(km, ids)
    fdr = {i: km[k].mean() for k, i in enumerate(ids)}
    nc = compute_norm_cost({"A": 1.0, "B": 1.0, "C": 1.0})
    r = greedy_rank(ids, fdr, nc, ks)
    assert [x["mr_id"] for x in r][:2] == ["A", "C"]   # B duplicates A, so C wins round 2
    assert jaccard(set(), set()) == 0.0
