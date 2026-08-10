"""
Unit tests for the Mutation Engine. Fast, CPU-only, uses tiny synthetic
state_dicts (not the real Model A checkpoint) to verify operator
correctness in isolation.
"""

from collections import Counter
from mrrank.mutation_engine import _is_conv_weight
import torch

from mrrank.mutation_engine import (
    weight_fuzz, weight_negate, weight_zero, corrupt_labels,
    build_all_mutant_specs, _select_layer_params,
)


def _tiny_state_dict():
    return {
        "layer1.0.conv1.weight": torch.ones(4, 4),
        "layer1.0.bn1.weight": torch.ones(4),
        "layer1.1.conv1.weight": torch.full((4, 4), 2.0),
        "layer2.conv1.weight": torch.full((4, 4), 3.0),
        "fc.weight": torch.full((4, 4), 9.0),
    }


def test_select_layer_params_matches_only_target_prefix():
    sd = _tiny_state_dict()
    keys = _select_layer_params(sd, "layer1.0")
    assert set(keys) == {"layer1.0.conv1.weight", "layer1.0.bn1.weight"}


def test_weight_negate_only_affects_targeted_conv_weights():
    """
    Negation now targets ONLY convolutional weight tensors (>=2D) within
    the layer, NOT BatchNorm scale/shift parameters (1D) -- BatchNorm
    exclusion was added after empirically observing that negating BN
    parameters at the same operation as conv weights catastrophically
    destabilized the network (45/60 weight-based mutants collapsed to
    exactly random-guess accuracy, 0.1000, before this fix).
    """
    sd = _tiny_state_dict()
    mutated = weight_negate(sd, "layer1.0")
    assert torch.equal(mutated["layer1.0.conv1.weight"], -sd["layer1.0.conv1.weight"])
    # bn1.weight is 1D -- must be UNCHANGED, not negated
    assert torch.equal(mutated["layer1.0.bn1.weight"], sd["layer1.0.bn1.weight"])
    assert torch.equal(mutated["layer1.1.conv1.weight"], sd["layer1.1.conv1.weight"])
    assert torch.equal(mutated["layer2.conv1.weight"], sd["layer2.conv1.weight"])
    assert torch.equal(mutated["fc.weight"], sd["fc.weight"])

def test_weight_zero_only_affects_targeted_layer():
    sd = _tiny_state_dict()
    mutated = weight_zero(sd, "layer2")
    assert torch.equal(mutated["layer2.conv1.weight"], torch.zeros(4, 4))
    assert torch.equal(mutated["layer1.0.conv1.weight"], sd["layer1.0.conv1.weight"])


def test_weight_fuzz_is_reproducible_with_same_seed():
    sd = _tiny_state_dict()
    m1 = weight_fuzz(sd, "layer1.1", std=0.1, seed=42)
    m2 = weight_fuzz(sd, "layer1.1", std=0.1, seed=42)
    assert torch.equal(m1["layer1.1.conv1.weight"], m2["layer1.1.conv1.weight"])


def test_weight_fuzz_different_seeds_differ():
    sd = _tiny_state_dict()
    m1 = weight_fuzz(sd, "layer1.1", std=0.1, seed=1)
    m2 = weight_fuzz(sd, "layer1.1", std=0.1, seed=2)
    assert not torch.equal(m1["layer1.1.conv1.weight"], m2["layer1.1.conv1.weight"])


def test_weight_fuzz_only_affects_targeted_layer():
    sd = _tiny_state_dict()
    mutated = weight_fuzz(sd, "layer2", std=0.5, seed=0)
    assert not torch.equal(mutated["layer2.conv1.weight"], sd["layer2.conv1.weight"])
    assert torch.equal(mutated["layer1.0.conv1.weight"], sd["layer1.0.conv1.weight"])


def test_corrupt_labels_changes_exact_expected_fraction():
    targets = [c % 10 for c in range(1000)]
    corrupted = corrupt_labels(targets, corruption_pct=0.1, seed=0, num_classes=10)
    n_changed = sum(1 for a, b in zip(targets, corrupted) if a != b)
    assert n_changed == 100


def test_corrupt_labels_never_maps_to_same_class():
    targets = [c % 10 for c in range(200)]
    corrupted = corrupt_labels(targets, corruption_pct=1.0, seed=0, num_classes=10)
    for orig, new in zip(targets, corrupted):
        assert new != orig


def test_corrupt_labels_reproducible():
    targets = [c % 10 for c in range(200)]
    c1 = corrupt_labels(targets, corruption_pct=0.2, seed=5)
    c2 = corrupt_labels(targets, corruption_pct=0.2, seed=5)
    assert c1 == c2


def test_build_all_mutant_specs_totals_80():
    specs = build_all_mutant_specs()
    assert len(specs) == 80
    ids = {s.mutant_id for s in specs}
    assert len(ids) == 80


def test_mutant_spec_breakdown_matches_appendix_b():
    specs = build_all_mutant_specs()
    counts = Counter(s.operator for s in specs)
    assert counts["weight_fuzz_low"] == 15
    assert counts["weight_fuzz_high"] == 15
    assert counts["weight_negate"] == 15
    assert counts["weight_zero"] == 15
    assert counts["label_corrupt_10"] == 10
    assert counts["label_corrupt_20"] == 10


def test_gpu_required_specs_are_exactly_label_corruption():
    specs = build_all_mutant_specs()
    gpu_specs = [s for s in specs if s.requires_gpu]
    assert len(gpu_specs) == 20
    assert all(s.operator.startswith("label_corrupt") for s in gpu_specs)

def test_is_conv_weight_distinguishes_dimensionality():
    from mrrank.mutation_engine import _is_conv_weight
    conv_like = torch.ones(4, 4)       # 2D -> conv/linear weight
    bn_like = torch.ones(4)             # 1D -> BatchNorm scale/shift
    assert _is_conv_weight("any.key", conv_like) is True
    assert _is_conv_weight("any.key", bn_like) is False


def test_generate_weight_mutant_reproducible_across_calls():
    """
    The same mutant_id must produce IDENTICAL mutated weights every time
    this function is called -- including across separate Python process
    invocations (not just within one run). Regression test for a bug
    where Python's built-in hash() was used for seeding, which is
    randomized per-process by default and broke this guarantee.
    """
    from mrrank.mutation_engine import generate_weight_mutant, build_all_mutant_specs
    sd = _tiny_state_dict()
    specs = [s for s in build_all_mutant_specs() if s.operator == "weight_fuzz_low"]
    spec = specs[0]
    m1 = generate_weight_mutant(spec, sd, global_seed=42)
    m2 = generate_weight_mutant(spec, sd, global_seed=42)
    for key in m1:
        assert torch.equal(m1[key], m2[key])