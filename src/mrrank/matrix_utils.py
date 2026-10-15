"""
Torch-free helpers for kill-matrix results: load the cached matrix and merge
a partial matrix (new MRs x the same mutants) into it. Safe for the lean app.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_cached_results(outputs_dir: Path) -> dict:
    """Load kill_matrix.npy + kill_matrix_metadata.json written by save_results."""
    outputs_dir = Path(outputs_dir)
    km = np.load(outputs_dir / "kill_matrix.npy").astype(bool)
    with open(outputs_dir / "kill_matrix_metadata.json") as f:
        meta = json.load(f)
    assert km.shape == (len(meta["mr_ids"]), len(meta["mutant_ids"])), "matrix/metadata shape mismatch"
    return {
        "kill_matrix": km,
        "mr_ids": list(meta["mr_ids"]),
        "mutant_ids": list(meta["mutant_ids"]),
        "clean_violation_rates": dict(meta["clean_violation_rates"]),
        "fdr_per_mr": dict(meta["fdr_per_mr"]),
    }


def merge_results(base: dict, extra: dict) -> dict:
    """
    Stack `extra` (new MRs x same mutants, same order) under `base`.
    Raises ValueError on mutant mismatch or MR id collision.
    """
    if base["mutant_ids"] != extra["mutant_ids"]:
        raise ValueError("mutant ids/order differ between base and extra matrices")
    dup = set(base["mr_ids"]) & set(extra["mr_ids"])
    if dup:
        raise ValueError(f"MR ids already present: {sorted(dup)}")

    km = np.vstack([base["kill_matrix"], extra["kill_matrix"]]).astype(bool)
    mr_ids = base["mr_ids"] + extra["mr_ids"]
    n_mut = len(base["mutant_ids"])
    return {
        "kill_matrix": km,
        "mr_ids": mr_ids,
        "mutant_ids": list(base["mutant_ids"]),
        "clean_violation_rates": {**base["clean_violation_rates"], **extra["clean_violation_rates"]},
        "fdr_per_mr": {mid: float(km[i].sum() / n_mut) for i, mid in enumerate(mr_ids)},
    }
