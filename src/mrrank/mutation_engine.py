"""
Mutation Engine for MR-Rank.

Implements the 4 mutation operators (report Appendix B) that generate 80
mutant models from Model A:
    - Weight Fuzzing (low std=0.05, high std=0.20): 15 + 15 = 30 mutants
    - Weight Negation: 15 mutants
    - Weight Zeroing: 15 mutants
    - Label Corruption (10%, 20%): 10 + 10 = 20 mutants

See config.py's MUTATION_TARGET_LAYERS docstring for the layer-mapping
rationale, and this module's own docstring notes below for the "3 runs"
ambiguity resolution for deterministic operators.

RESOLUTION OF "3 RUNS" AMBIGUITY (Weight Negation / Weight Zeroing): these
operators are fully deterministic -- negating or zeroing the same tensors
gives an identical result every time, unlike Weight Fuzzing where each run
uses a genuinely different noise seed. To preserve the report's exact
mutant counts (15 negation + 15 zeroing) without fabricating artificial
variation, each of the 3 "runs" per (operator, layer) pair produces the
SAME mutated weights, saved as 3 separate files. This is intentionally
reframed as a reproducibility check: Phase 5's Kill Matrix should show
identical kill-vectors across these 3 replicate mutants, confirming the
mutation and evaluation pipeline is deterministic end-to-end -- a
legitimate validation signal, not wasted computation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from mrrank import config, data_module
from mrrank.model_module import ModelWrapper


# ---------------------------------------------------------------------------
# Layer targeting
# ---------------------------------------------------------------------------
def _select_layer_params(state_dict: dict, layer_prefix: str) -> list[str]:
    """
    Return all state_dict keys belonging to the given layer/sub-module
    prefix (e.g. "layer1.0" matches "layer1.0.conv1.weight",
    "layer1.0.bn1.weight", etc., but NOT "layer1.1.*").
    """
    prefix = layer_prefix + "."
    keys = [k for k in state_dict.keys() if k.startswith(prefix)]
    if not keys:
        raise ValueError(
            f"No parameters found for layer prefix '{layer_prefix}'. Check "
            f"MUTATION_TARGET_LAYERS matches the model's actual state_dict "
            f"key structure (expected torchvision ResNet-18 naming, e.g. "
            f"'layer1.0.conv1.weight')."
        )
    return keys


# ---------------------------------------------------------------------------
# Mutation operators (operate on a state_dict, return a NEW mutated dict)
# ---------------------------------------------------------------------------
def weight_fuzz(state_dict: dict, layer_prefix: str, std: float, seed: int) -> dict:
    """
    Add Gaussian noise (std=`std`) to every FLOATING-POINT tensor in the
    targeted layer (skips integer buffers like BatchNorm's
    num_batches_tracked, which aren't meaningful to perturb with noise).
    """
    mutated = copy.deepcopy(state_dict)
    keys = _select_layer_params(mutated, layer_prefix)
    generator = torch.Generator().manual_seed(seed)
    for key in keys:
        tensor = mutated[key]
        if tensor.is_floating_point():
            noise = torch.randn(tensor.shape, generator=generator) * std
            mutated[key] = tensor + noise
    return mutated


def weight_negate(state_dict: dict, layer_prefix: str) -> dict:
    """Reverse the sign of every floating-point tensor in the targeted layer."""
    mutated = copy.deepcopy(state_dict)
    keys = _select_layer_params(mutated, layer_prefix)
    for key in keys:
        if mutated[key].is_floating_point():
            mutated[key] = -mutated[key]
    return mutated


def weight_zero(state_dict: dict, layer_prefix: str) -> dict:
    """
    Set every tensor (including integer buffers) in the targeted layer to
    zero -- a full "this layer contributes nothing" mutation.
    """
    mutated = copy.deepcopy(state_dict)
    keys = _select_layer_params(mutated, layer_prefix)
    for key in keys:
        mutated[key] = torch.zeros_like(mutated[key])
    return mutated


def corrupt_labels(targets: list, corruption_pct: float, seed: int, num_classes: int = 10) -> list:
    """
    Randomly relabel `corruption_pct` fraction of training labels to a
    DIFFERENT class (never the original), using a local seeded RandomState
    for reproducibility (per Reproducibility Constraint).
    """
    rng = np.random.RandomState(seed)
    targets = list(targets)
    n_corrupt = int(len(targets) * corruption_pct)
    indices = rng.choice(len(targets), size=n_corrupt, replace=False)
    for idx in indices:
        original = targets[idx]
        choices = [c for c in range(num_classes) if c != original]
        targets[idx] = int(rng.choice(choices))
    return targets


# ---------------------------------------------------------------------------
# Mutant specification registry (metadata for all 80 mutants)
# ---------------------------------------------------------------------------
@dataclass
class MutantSpec:
    mutant_id: str
    operator: str
    layer: str | None
    run_index: int
    strength: float | None
    requires_gpu: bool


def build_all_mutant_specs() -> list[MutantSpec]:
    """Enumerate metadata for all 80 mutants, per Appendix B's exact counts."""
    specs = []
    layers = config.MUTATION_TARGET_LAYERS

    for layer in layers:
        for run in range(1, 4):
            specs.append(MutantSpec(f"WF_low_{layer}_run{run}", "weight_fuzz_low",
                                     layer, run, config.WEIGHT_FUZZ_LOW_STD, False))
    for layer in layers:
        for run in range(1, 4):
            specs.append(MutantSpec(f"WF_high_{layer}_run{run}", "weight_fuzz_high",
                                     layer, run, config.WEIGHT_FUZZ_HIGH_STD, False))
    for layer in layers:
        for run in range(1, 4):
            specs.append(MutantSpec(f"WN_{layer}_run{run}", "weight_negate",
                                     layer, run, None, False))
    for layer in layers:
        for run in range(1, 4):
            specs.append(MutantSpec(f"WZ_{layer}_run{run}", "weight_zero",
                                     layer, run, None, False))
    for run in range(1, config.LABEL_CORRUPTION_RUNS_PER_PCT + 1):
        specs.append(MutantSpec(f"LC_10pct_run{run}", "label_corrupt_10",
                                 None, run, 0.10, True))
    for run in range(1, config.LABEL_CORRUPTION_RUNS_PER_PCT + 1):
        specs.append(MutantSpec(f"LC_20pct_run{run}", "label_corrupt_20",
                                 None, run, 0.20, True))

    assert len(specs) == 80, f"Expected 80 mutant specs, got {len(specs)}"
    return specs


# ---------------------------------------------------------------------------
# Generation + verification (weight-based mutants -- CPU, local)
# ---------------------------------------------------------------------------
def generate_weight_mutant(spec: MutantSpec, base_state_dict: dict, global_seed: int) -> dict:
    """Apply the mutation described by `spec` to a copy of `base_state_dict`."""
    if spec.operator in ("weight_fuzz_low", "weight_fuzz_high"):
        # Each run needs a genuinely different seed -- derived deterministically
        # from the global seed + mutant id, so it's reproducible but distinct
        # per mutant.
        run_seed = (global_seed + abs(hash(spec.mutant_id))) % (2**31)
        return weight_fuzz(base_state_dict, spec.layer, spec.strength, run_seed)
    elif spec.operator == "weight_negate":
        return weight_negate(base_state_dict, spec.layer)
    elif spec.operator == "weight_zero":
        return weight_zero(base_state_dict, spec.layer)
    else:
        raise ValueError(
            f"generate_weight_mutant() does not handle operator '{spec.operator}' "
            f"-- label-corruption mutants require full retraining; see "
            f"generate_label_corruption_mutants.py."
        )


def verify_mutant_accuracy(state_dict: dict, images: np.ndarray, labels: np.ndarray) -> float:
    """
    Load `state_dict` into a fresh ModelWrapper and measure accuracy on the
    given (raw uint8) images/labels -- typically the 500-image eval subset,
    for fast verification without needing the full 10,000-image test set.
    """
    wrapper = ModelWrapper(device="cpu")
    wrapper.model.load_state_dict(state_dict)
    batch = data_module.normalize_batch(images)
    preds = wrapper.predict_batch(batch)
    correct = (preds.numpy() == labels).sum()
    return correct / len(labels)