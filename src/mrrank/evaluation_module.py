"""
Evaluation Module for MR-Rank: builds the 20x80 Kill Matrix.

Kill rule (report Sec 4.3): killed = (mutant_violation_rate - clean_violation_rate) > 0.05

Clean violation rate per MR is computed ONCE against Model A and reused
across all 80 mutants (it doesn't depend on the mutant), avoiding 80x
redundant computation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from mrrank import config, data_module, mr_engine
from mrrank.model_module import ModelWrapper

KILL_THRESHOLD = 0.05


def compute_clean_violation_rates(model_wrapper, images: np.ndarray, mrs=None) -> dict[str, float]:
    """Violation rate of each MR against the CLEAN model. Computed once.
    `mrs` defaults to the canonical 20."""
    mrs = mr_engine.ALL_MRS if mrs is None else mrs
    return {mr.id: mr_engine.compute_violation_rate(mr, model_wrapper, images) for mr in mrs}


def list_mutant_paths() -> list:
    """Sorted paths of all mutant checkpoints in mutants/."""
    return sorted(config.MUTANTS_DIR.glob("*.pth"))


def iter_mutant_state_dicts(paths=None):
    """Yield (mutant_id, state_dict) ONE AT A TIME so only one ~45 MB state
    dict is in memory (the old loader held all 80, ~3.6 GB)."""
    for p in (list_mutant_paths() if paths is None else paths):
        yield p.stem, torch.load(p, map_location="cpu")


def evaluate_mrs(mrs, mutant_paths, clean_rates: dict, images: np.ndarray,
                 verbose: bool = True) -> dict:
    """
    Partial evaluation: len(mrs) x len(mutant_paths) kill matrix.
    `clean_rates` must contain an entry for every MR in `mrs`.
    Row order follows `mrs`, column order follows `mutant_paths`.
    """
    mr_ids = [mr.id for mr in mrs]
    mutant_ids = [Path(p).stem for p in mutant_paths]
    kill_matrix = np.zeros((len(mrs), len(mutant_ids)), dtype=bool)
    mutant_violation_rates = {mid: {} for mid in mutant_ids}
    mutant_wrapper = ModelWrapper(device="cpu")

    if verbose:
        print(f"\nBuilding {len(mrs)}x{len(mutant_ids)} Kill Matrix "
              f"({len(mrs) * len(mutant_ids)} evaluations)...\n")

    start = time.time()
    for j, (mutant_id, state_dict) in enumerate(iter_mutant_state_dicts(mutant_paths)):
        mutant_wrapper.model.load_state_dict(state_dict)
        del state_dict  # free before next checkpoint is loaded
        for i, mr in enumerate(mrs):
            mutant_rate = mr_engine.compute_violation_rate(mr, mutant_wrapper, images)
            mutant_violation_rates[mutant_id][mr.id] = mutant_rate
            kill_matrix[i, j] = (mutant_rate - clean_rates[mr.id]) > KILL_THRESHOLD

        if verbose and (j + 1) % 10 == 0:
            elapsed = time.time() - start
            eta = elapsed / (j + 1) * (len(mutant_ids) - j - 1)
            print(f"  {j + 1}/{len(mutant_ids)} mutants processed "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    fdr_per_mr = {
        mr_id: float(kill_matrix[i].sum() / len(mutant_ids))
        for i, mr_id in enumerate(mr_ids)
    }
    return {
        "kill_matrix": kill_matrix,
        "mr_ids": mr_ids,
        "mutant_ids": mutant_ids,
        "mutant_violation_rates": mutant_violation_rates,
        "fdr_per_mr": fdr_per_mr,
    }


def build_kill_matrix(model_a_path=None, verbose: bool = True, mrs=None) -> dict:
    """
    Builds the Kill Matrix. Default (mrs=None) is the full canonical 20x80
    matrix, identical to the pre-refactor behaviour. Pass `mrs` to evaluate
    only those MRs (e.g. user-defined) against all 80 mutants.
    Returns dict with: kill_matrix (np.ndarray bool), mr_ids, mutant_ids,
    clean_violation_rates, mutant_violation_rates, fdr_per_mr.
    """
    model_a_path = model_a_path or config.MODEL_A_CHECKPOINT
    full_run = mrs is None
    mrs = mr_engine.ALL_MRS if full_run else mrs

    if verbose:
        print("Loading Model A checkpoint...")
    clean_wrapper = ModelWrapper.from_checkpoint(model_a_path, device="cpu")

    if verbose:
        print("Loading fixed 500-image evaluation subset...")
    images, _ = data_module.get_eval_subset_raw()

    if verbose:
        print(f"Computing clean-model violation rates for {len(mrs)} MRs...")
    clean_rates = compute_clean_violation_rates(clean_wrapper, images, mrs)

    mutant_paths = list_mutant_paths()
    if full_run:
        assert len(mutant_paths) == 80, f"Expected 80 mutants, found {len(mutant_paths)}"

    results = evaluate_mrs(mrs, mutant_paths, clean_rates, images, verbose)
    results["clean_violation_rates"] = clean_rates
    return results


def save_results(results: dict) -> None:
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(config.OUTPUTS_DIR / "kill_matrix.npy", results["kill_matrix"])

    killed_by = {}
    for j, mutant_id in enumerate(results["mutant_ids"]):
        killed_by[mutant_id] = [
            results["mr_ids"][i] for i in range(len(results["mr_ids"]))
            if results["kill_matrix"][i, j]
        ]

    metadata = {
        "mr_ids": results["mr_ids"],
        "mutant_ids": results["mutant_ids"],
        "clean_violation_rates": results["clean_violation_rates"],
        "fdr_per_mr": results["fdr_per_mr"],
        "killed_by_per_mutant": killed_by,
        "kill_threshold": KILL_THRESHOLD,
    }
    with open(config.OUTPUTS_DIR / "kill_matrix_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    results = build_kill_matrix()
    save_results(results)

    print("\n--- FDR per MR ---")
    for mr_id, fdr in sorted(results["fdr_per_mr"].items(), key=lambda x: -x[1]):
        print(f"{mr_id}: {fdr:.3f}")

    n_dead = sum(1 for kb in results["mutant_ids"]
                 if len(json.load(open(config.OUTPUTS_DIR / "kill_matrix_metadata.json"))
                        ["killed_by_per_mutant"][kb]) == 0)
    print(f"\nMutants killed by ZERO MRs (equivalent, undetectable): {n_dead}")
    print(f"Saved to {config.OUTPUTS_DIR / 'kill_matrix.npy'} and kill_matrix_metadata.json")