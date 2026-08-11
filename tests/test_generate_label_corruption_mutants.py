"""
Fast, CPU-only tests for label-corruption mutant generation logic --
NOT the actual training loop (that requires GPU and is verified manually
on Kaggle). Tests the deterministic seeding and manifest-building pieces
that CAN be verified quickly.
"""

from mrrank.generate_label_corruption_mutants import _deterministic_run_seed


def test_deterministic_run_seed_reproducible():
    s1 = _deterministic_run_seed(0.10, 1, global_seed=42)
    s2 = _deterministic_run_seed(0.10, 1, global_seed=42)
    assert s1 == s2


def test_deterministic_run_seed_varies_by_run_index():
    s1 = _deterministic_run_seed(0.10, 1, global_seed=42)
    s2 = _deterministic_run_seed(0.10, 2, global_seed=42)
    assert s1 != s2


def test_deterministic_run_seed_varies_by_corruption_pct():
    s1 = _deterministic_run_seed(0.10, 1, global_seed=42)
    s2 = _deterministic_run_seed(0.20, 1, global_seed=42)
    assert s1 != s2


def test_deterministic_run_seed_is_valid_seed_range():
    seed = _deterministic_run_seed(0.10, 1, global_seed=42)
    assert 0 <= seed < 2**31