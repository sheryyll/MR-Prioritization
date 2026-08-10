"""
Per-layer binary-search calibration for Weight Fuzzing std.

Weight Fuzz applies additive Gaussian noise to convolutional weights. A
fixed std applied uniformly across all 5 target layers was found
empirically to be wildly miscalibrated by depth: layer1 barely degrades
at std=0.05/0.20, while layer2-4 collapse to random-guess accuracy
(~0.10) at the SAME std. This is a known property of deep networks --
later layers' representations are more load-bearing for the final
prediction, with less downstream capacity to recover from perturbation.

This script finds, per layer, a std value that lands accuracy near the
CENTER of the target band [0.40, 0.70] (targeting ~0.55, not an edge --
same margin-safety lesson learned from MR calibration, where edge-exact
values proved fragile across runs).

Negation and Zeroing have NO continuous strength parameter -- they are
not calibrated here. Where they collapse the network below 0.40 (observed
for layer2/3/4 in this project), they are accepted as legitimate
"equivalent mutants" per standard mutation-testing terminology, not
force-fitted into range.

Usage:
    python -m mrrank.calibrate_mutants
"""

from __future__ import annotations

from mrrank import config, data_module
from mrrank.model_module import ModelWrapper
from mrrank.mutation_engine import weight_fuzz, verify_mutant_accuracy


TARGET_CENTER = 0.55
TARGET_MIN = config.MUTANT_ACC_MIN
TARGET_MAX = config.MUTANT_ACC_MAX


def calibrate_layer_std(
    layer: str, base_state_dict: dict, images, labels,
    low: float = 0.001, high: float = 0.20,
    target_center: float = 0.55,
    precision: float = 0.0005, max_iterations: int = 20,
) -> dict:
    """
    Binary search for a std value producing accuracy close to
    `target_center` for the given layer.
    """
    def measure(std: float) -> float:
        mutated = weight_fuzz(base_state_dict, layer, std=std, seed=config.SEED)
        return verify_mutant_accuracy(mutated, images, labels)

    trace = []
    acc_low = measure(low)
    trace.append((low, acc_low))
    if acc_low < config.MUTANT_ACC_MIN:
        return {
            "layer": layer, "best_std": None, "achieved_acc": None, "trace": trace,
            "note": f"Even the smallest tested std ({low}) already drops accuracy "
                    f"to {acc_low:.4f}, below the target band floor ({config.MUTANT_ACC_MIN}).",
        }

    lo, hi = low, high
    best_std, best_acc = low, acc_low

    for _ in range(max_iterations):
        if abs(hi - lo) < precision:
            break
        mid = (lo + hi) / 2
        acc = measure(mid)
        trace.append((mid, acc))

        if acc > target_center:
            lo = mid
            if config.MUTANT_ACC_MIN <= acc <= config.MUTANT_ACC_MAX:
                best_std, best_acc = mid, acc
        else:
            hi = mid
            if config.MUTANT_ACC_MIN <= acc <= config.MUTANT_ACC_MAX and abs(acc - target_center) < abs(best_acc - target_center):
                best_std, best_acc = mid, acc

    return {"layer": layer, "best_std": best_std, "achieved_acc": best_acc,
            "trace": trace, "note": None}

def main():
    print("Loading Model A checkpoint...")
    wrapper = ModelWrapper.from_checkpoint(config.MODEL_A_CHECKPOINT, device="cpu")
    base_state_dict = wrapper.model.state_dict()

    print("Loading fixed 500-image evaluation subset...")
    images, labels = data_module.get_eval_subset_raw()

    print(f"\n{'Layer':<12}{'Tier':<8}{'Best std':<14}{'Achieved Acc':<16}{'Iterations'}")
    print("-" * 65)

    results = {}
    for layer in config.MUTATION_TARGET_LAYERS:
        for tier, center in [("low", 0.55), ("high", 0.45)]:
            result = calibrate_layer_std(layer, base_state_dict, images, labels, target_center=center)
            results[f"{layer}_{tier}"] = result
            if result["best_std"] is not None:
                print(f"{layer:<12}{tier:<8}{result['best_std']:<14.5f}{result['achieved_acc']:<16.4f}"
                      f"{len(result['trace'])}")
            else:
                print(f"{layer:<12}{tier:<8}{'NEEDS REVIEW':<14}{'--':<16}{len(result['trace'])}")
                print(f"    NOTE: {result['note']}")

    return results


if __name__ == "__main__":
    main()
