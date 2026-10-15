r"""
Characterization test: the refactored pipeline must reproduce the committed
Model A kill matrix EXACTLY. Needs torch + checkpoints + mutants (slow: full 20x80).

Before touching anything else, save the reference:
    mkdir tests\fixtures
    copy outputs\kill_matrix.npy tests\fixtures\kill_matrix_ref.npy
    copy outputs\kill_matrix_metadata.json tests\fixtures\kill_matrix_metadata_ref.json
"""
import json
from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).parent / "fixtures"
from mrrank import config

pytestmark = pytest.mark.skipif(
    not ((FIX / "kill_matrix_ref.npy").exists()
         and Path(config.MODEL_A_CHECKPOINT).exists()
         and any(Path(config.MUTANTS_DIR).glob("*.pth"))),
    reason="needs reference fixtures, Model A checkpoint and mutants",
)

def test_full_matrix_identical():
    from mrrank import evaluation_module as ev
    res = ev.build_kill_matrix(verbose=False)
    ref = np.load(FIX / "kill_matrix_ref.npy")
    meta = json.load(open(FIX / "kill_matrix_metadata_ref.json"))
    assert res["mr_ids"] == meta["mr_ids"]
    assert res["mutant_ids"] == meta["mutant_ids"]
    assert np.array_equal(res["kill_matrix"], ref)
    assert res["clean_violation_rates"] == pytest.approx(meta["clean_violation_rates"])


def test_partial_equals_full_rows():
    """Evaluating a subset of MRs gives the same rows as the full run."""
    from mrrank import evaluation_module as ev, mr_engine
    ref = np.load(FIX / "kill_matrix_ref.npy")
    ids = [mr.id for mr in mr_engine.ALL_MRS]
    subset = [mr_engine.MR_BY_ID[i] for i in ("MR03", "MR15")]
    res = ev.build_kill_matrix(verbose=False, mrs=subset)
    for k, mr in enumerate(subset):
        assert np.array_equal(res["kill_matrix"][k], ref[ids.index(mr.id)])
