"""
Generates all 60 weight-based mutants (Weight Fuzz low/high, Negation,
Zeroing) from the real, trained Model A checkpoint. Runs entirely on CPU
-- no GPU required. Typically completes in a few minutes.

Usage:
    python -m mrrank.generate_weight_mutants
"""

from __future__ import annotations

import json

import torch

from mrrank import config, data_module
from mrrank.model_module import ModelWrapper
from mrrank.mutation_engine import (
    build_all_mutant_specs, generate_weight_mutant, verify_mutant_accuracy,
)


def main():
    config.MUTANTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Model A checkpoint...")
    base_state_dict = torch.load(config.MODEL_A_CHECKPOINT, map_location="cpu")

    print("Loading fixed 500-image evaluation subset...")
    images, labels = data_module.get_eval_subset_raw()

    print("Measuring Model A's own (clean) accuracy for accuracy-drop reference...")
    model_a_wrapper = ModelWrapper.from_checkpoint(config.MODEL_A_CHECKPOINT, device="cpu")
    model_a_acc = verify_mutant_accuracy(base_state_dict, images, labels)
    print(f"Model A clean accuracy (500-image eval subset): {model_a_acc:.4f}\n")

    specs = [s for s in build_all_mutant_specs() if not s.requires_gpu]
    print(f"Generating {len(specs)} weight-based mutants...\n")

    manifest = []
    n_in_band = 0

    print(f"{'Mutant ID':<28}{'Accuracy':<12}{'Acc Drop':<12}{'In [0.40,0.70]'}")
    print("-" * 70)

    for spec in specs:
        mutated_state_dict = generate_weight_mutant(spec, base_state_dict, config.SEED)
        acc = verify_mutant_accuracy(mutated_state_dict, images, labels)
        acc_drop = model_a_acc - acc
        in_band = bool(config.MUTANT_ACC_MIN <= acc <= config.MUTANT_ACC_MAX)
        if in_band:
            n_in_band += 1

        status = "YES" if in_band else "no"
        print(f"{spec.mutant_id:<28}{acc:<12.4f}{acc_drop:<12.4f}{status}")

        path = config.MUTANTS_DIR / f"{spec.mutant_id}.pth"
        torch.save(mutated_state_dict, path)

        is_equivalent = bool(
            spec.operator in ("weight_negate", "weight_zero")
            and acc < config.MUTANT_ACC_MIN
        )
        manifest.append({
            "mutant_id": spec.mutant_id,
            "operator": spec.operator,
            "layer": spec.layer,
            "run_index": spec.run_index,
            "strength": spec.strength,
            "accuracy": acc,
            "accuracy_drop_from_model_a": acc_drop,
            "in_target_band": bool(in_band),
            "is_equivalent_mutant": is_equivalent,
            "checkpoint_path": str(path),
        })

    print("-" * 70)
    n_equivalent = sum(1 for m in manifest if m["is_equivalent_mutant"])
    n_out_of_band_stochastic = len(specs) - n_in_band - n_equivalent
    print(f"\n{n_in_band} in target band, {n_equivalent} equivalent mutants "
          f"(deterministic operator collapsed the network), "
          f"{n_out_of_band_stochastic} out-of-band (Fuzz variance).")
    if n_in_band < len(specs):
        print(
            f"NOTE: {len(specs) - n_in_band} mutants fell outside the target "
            f"band. Reported here for review, not silently discarded -- see "
            f"manifest for full details before deciding whether any need "
            f"strength adjustment."
        )

    manifest_path = config.OUTPUTS_DIR / "weight_mutant_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest to {manifest_path}")


if __name__ == "__main__":
    main()