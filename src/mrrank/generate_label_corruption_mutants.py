"""
Generates the 20 label-corruption mutants (10x 10%-corrupted, 10x
20%-corrupted CIFAR-10 training labels), each requiring a full retrain
from scratch. Runs on GPU (Kaggle) -- CPU is technically possible but
impractically slow for 20 full training runs.

Epoch count is capped well below Model A/B's schedule (config.py:
LABEL_CORRUPTION_EPOCHS, default 20) since these mutants only need to
land in [MUTANT_ACC_MIN, MUTANT_ACC_MAX] (0.40-0.70), not converge to
high accuracy -- label corruption alone suppresses accuracy without
needing many epochs, and label noise makes prolonged training
increasingly prone to overfitting the corrupted labels themselves rather
than staying in a genuinely "degraded but functional" state.

Usage (Kaggle):
    python -m mrrank.generate_label_corruption_mutants
    python -m mrrank.generate_label_corruption_mutants --epochs 15  # override cap
"""

from __future__ import annotations

import argparse
import json

import torch

from mrrank import config, data_module
from mrrank.config import set_seed
from mrrank.model_module import ModelWrapper, TrainConfig
from mrrank.mutation_engine import corrupt_labels, verify_mutant_accuracy


def _deterministic_run_seed(corruption_pct: float, run_index: int, global_seed: int) -> int:
    """Distinct, reproducible seed per (corruption_pct, run_index) pair."""
    import hashlib
    text = f"label_corrupt_{corruption_pct}_{run_index}_{global_seed}"
    digest = hashlib.md5(text.encode()).digest()
    return int.from_bytes(digest[:4], "little") % (2**31)


def train_one_label_corruption_mutant(
    corruption_pct: float, run_index: int, epochs: int, num_workers: int,
) -> tuple[dict, float, float]:
    """
    Trains one label-corruption mutant from scratch. Returns
    (state_dict, eval_subset_accuracy, full_test_set_accuracy).
    """
    run_seed = _deterministic_run_seed(corruption_pct, run_index, config.SEED)
    set_seed(run_seed)

    # Build a corrupted training dataset. We corrupt the labels of the
    # normalized (augmented) train dataset directly -- torchvision's
    # CIFAR10 stores labels in .targets, mutable in place.
    train_dataset = data_module.load_normalized_cifar10(train=True, augment=True)
    train_dataset.targets = corrupt_labels(
        train_dataset.targets, corruption_pct=corruption_pct, seed=run_seed
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=num_workers,
    )
    test_loader = data_module.get_test_loader(
        batch_size=config.TRAIN_BATCH_SIZE, num_workers=num_workers
    )

    wrapper = ModelWrapper()
    train_cfg = TrainConfig(
        epochs=epochs,
        batch_size=config.TRAIN_BATCH_SIZE,
        enable_early_stopping=False,  # fixed epoch count, no target-band logic here
    )
    wrapper.fit(train_loader, test_loader, train_cfg, verbose=True)

    state_dict = wrapper.model.state_dict()

    # Report accuracy on BOTH the fixed 500-image eval subset (consistent
    # with weight-mutant verification, and what Phase 5's Kill Matrix will
    # use) AND the full 10,000-image test set (more stable estimate, useful
    # for sanity-checking / reporting).
    eval_images, eval_labels = data_module.get_eval_subset_raw()
    eval_subset_acc = verify_mutant_accuracy(state_dict, eval_images, eval_labels)
    full_test_acc = wrapper.evaluate(test_loader)

    return state_dict, eval_subset_acc, full_test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    config.MUTANTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    n_in_band = 0

    print("Generating 20 label-corruption mutants (calibrated per-tier)...\n")
    print(f"{'Mutant ID':<24}{'EvalSubAcc':<12}{'FullTestAcc':<14}{'In [0.40,0.70]'}")
    print("-" * 65)

    for tier_name, tier_cfg in config.LABEL_CORRUPTION_CONFIG.items():
        pct = tier_cfg["pct"]
        epochs = tier_cfg["epochs"]
        pct_label = f"{int(pct*100)}pct"

        for run in range(1, config.LABEL_CORRUPTION_RUNS_PER_PCT + 1):
            mutant_id = f"LC_{pct_label}_run{run}"
            state_dict, eval_acc, full_acc = train_one_label_corruption_mutant(
                corruption_pct=pct, run_index=run,
                epochs=epochs, num_workers=args.num_workers,
            )

            in_band = bool(config.MUTANT_ACC_MIN <= eval_acc <= config.MUTANT_ACC_MAX)
            if in_band:
                n_in_band += 1
            status = "YES" if in_band else "no"
            print(f"{mutant_id:<24}{eval_acc:<12.4f}{full_acc:<14.4f}{status}")

            path = config.MUTANTS_DIR / f"{mutant_id}.pth"
            torch.save(state_dict, path)

            manifest.append({
                "mutant_id": mutant_id,
                "operator": f"label_corrupt_{pct_label}",
                "layer": None,
                "run_index": run,
                "strength": pct,
                "epochs": epochs,
                "accuracy": eval_acc,
                "full_test_set_accuracy": full_acc,
                "in_target_band": in_band,
                "is_equivalent_mutant": False,
                "checkpoint_path": str(path),
            })

    print("-" * 65)
    print(f"\n{n_in_band} of 20 label-corruption mutants landed in target band "
          f"[{config.MUTANT_ACC_MIN}, {config.MUTANT_ACC_MAX}].")

    manifest_path = config.OUTPUTS_DIR / "label_corruption_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()