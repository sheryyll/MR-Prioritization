import numpy as np
import pytest
from mrrank.matrix_utils import merge_results


def _r(ids, km, muts=("a", "b")):
    return {"kill_matrix": np.array(km, bool), "mr_ids": ids, "mutant_ids": list(muts),
            "clean_violation_rates": {i: 0.0 for i in ids}, "fdr_per_mr": {}}


def test_merge():
    m = merge_results(_r(["M1"], [[1, 0]]), _r(["U1"], [[1, 1]]))
    assert m["mr_ids"] == ["M1", "U1"] and m["kill_matrix"].shape == (2, 2)
    assert m["fdr_per_mr"] == {"M1": 0.5, "U1": 1.0}


def test_merge_rejects_mismatch():
    with pytest.raises(ValueError):
        merge_results(_r(["M1"], [[1, 0]]), _r(["U1"], [[1, 1]], muts=("a", "c")))
    with pytest.raises(ValueError):
        merge_results(_r(["M1"], [[1, 0]]), _r(["M1"], [[1, 1]]))
