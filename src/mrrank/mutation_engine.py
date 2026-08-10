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
import hashlib
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

def _deterministic_seed(text: str, salt: int) -> int:
    """
    Deterministic string->int hash, stable across process runs (unlike
    Python's built-in hash(), which is randomized per-interpreter-session
    by default for security reasons -- PYTHONHASHSEED). Required so that
    the SAME mutant_id always produces the SAME seed, and therefore the
    SAME mutated weights, across every invocation of this script -- per
    the project's Reproducibility Constraint.
    """
    digest = hashlib.md5(f"{text}_{salt}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % (2**31)

# ---------------------------------------------------------------------------
# Mutation operators (operate on a state_dict, return a NEW mutated dict)
# ---------------------------------------------------------------------------
def _is_conv_weight(key: str, tensor: torch.Tensor) -> bool:
    """
    True for convolutional/linear weight tensors (the actual learned
    feature-extraction parameters), False for BatchNorm scale/shift
    parameters and any 1-D tensors. BatchNorm parameters are deliberately
    excluded from Fuzz/Negate: mutating them at the same scale as conv
    weights was found to catastrophically destabilize the entire network
    (empirically: 45/60 mutants collapsed to exactly random-guess accuracy,
    0.1000, regardless of which layer or fuzz strength was targeted) --
    BatchNorm's scale/shift operate on normalized activations and are far
    more sensitive to perturbation than raw convolutional weights.
    """
    return tensor.dim() >= 2  # conv/linear weights are >=2D; BN params are 1D



def weight_fuzz(state_dict: dict, layer_prefix: str, std: float, seed: int) -> dict:
    """
    Add Gaussian noise (std=`std`) to convolutional weight tensors ONLY
    in the targeted layer -- BatchNorm scale/shift parameters are
    excluded (see _is_conv_weight docstring).
    """
    mutated = copy.deepcopy(state_dict)
    keys = _select_layer_params(mutated, layer_prefix)
    generator = torch.Generator().manual_seed(seed)
    for key in keys:
        tensor = mutated[key]
        if tensor.is_floating_point() and _is_conv_weight(key, tensor):
            noise = torch.randn(tensor.shape, generator=generator) * std
            mutated[key] = tensor + noise
    return mutated


def weight_negate(state_dict: dict, layer_prefix: str) -> dict:
    """Reverse the sign of convolutional weight tensors ONLY in the
    targeted layer -- BatchNorm parameters excluded (see weight_fuzz)."""
    mutated = copy.deepcopy(state_dict)
    keys = _select_layer_params(mutated, layer_prefix)
    for key in keys:
        if mutated[key].is_floating_point() and _is_conv_weight(key, mutated[key]):
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
            std = config.WEIGHT_FUZZ_STD_BY_LAYER[layer]["low"]
            specs.append(MutantSpec(f"WF_low_{layer}_run{run}", "weight_fuzz_low",
                                     layer, run, std, False))
    for layer in layers:
        for run in range(1, 4):
            std = config.WEIGHT_FUZZ_STD_BY_LAYER[layer]["high"]
            specs.append(MutantSpec(f"WF_high_{layer}_run{run}", "weight_fuzz_high",
                                     layer, run, std, False))
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
import hashlib

def _deterministic_seed(text: str, salt: int) -> int:
    """
    Deterministic string->int hash, stable across process runs (unlike
    Python's built-in hash(), which is randomized per-interpreter-session
    by default for security reasons -- PYTHONHASHSEED). Required so that
    the SAME mutant_id always produces the SAME seed, and therefore the
    SAME mutated weights, across every invocation of this script -- per
    the project's Reproducibility Constraint.
    """
    digest = hashlib.md5(f"{text}_{salt}".encode()).digest()
    return int.from_bytes(digest[:4], "little") % (2**31)

def generate_weight_mutant(spec: MutantSpec, base_state_dict: dict, global_seed: int) -> dict:
    """Apply the mutation described by `spec` to a copy of `base_state_dict`."""
    if spec.operator in ("weight_fuzz_low", "weight_fuzz_high"):
        run_seed = _deterministic_seed(spec.mutant_id, global_seed)
        tier = "low" if spec.operator == "weight_fuzz_low" else "high"
        std = config.WEIGHT_FUZZ_STD_BY_LAYER[spec.layer][tier]
        return weight_fuzz(base_state_dict, spec.layer, std, run_seed)
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