"""Phase 9: APFD, FD@k, Wilcoxon signed-rank test."""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import wilcoxon

from mrrank import config


def compute_apfd(ordering: list[str], kill_matrix: np.ndarray, mr_id_to_idx: dict) -> float:
    n = len(ordering)
    m = kill_matrix.shape[1]
    tf_sum = 0
    for mutant_idx in range(m):
        first_kill_rank = None
        for rank, mr_id in enumerate(ordering, start=1):
            if kill_matrix[mr_id_to_idx[mr_id], mutant_idx]:
                first_kill_rank = rank
                break
        if first_kill_rank is None:
            first_kill_rank = n  # never killed -> worst-case rank
        tf_sum += first_kill_rank
    return 1 - (tf_sum / (n * m)) + 1 / (2 * n)


def compute_fd_at_k(ordering: list[str], kill_matrix: np.ndarray, mr_id_to_idx: dict, k: int) -> float:
    top_k = ordering[:k]
    detected = set()
    for mr_id in top_k:
        idx = mr_id_to_idx[mr_id]
        detected |= set(np.where(kill_matrix[idx])[0].tolist())
    all_detectable = set(np.where(kill_matrix.any(axis=0))[0].tolist())
    return len(detected) / len(all_detectable) if all_detectable else 0.0


def random_baseline_apfds(mr_ids: list[str], kill_matrix: np.ndarray, mr_id_to_idx: dict, n_trials: int = 30) -> list[float]:
    apfds = []
    for trial_seed in range(n_trials):
        rng = np.random.RandomState(trial_seed)
        ordering = list(mr_ids)
        rng.shuffle(ordering)
        apfds.append(compute_apfd(ordering, kill_matrix, mr_id_to_idx))
    return apfds


def fdr_only_ordering(mr_ids: list[str], fdr: dict) -> list[str]:
    return sorted(mr_ids, key=lambda mr: -fdr[mr])


def main():
    km = np.load(config.OUTPUTS_DIR / "kill_matrix.npy")
    meta = json.load(open(config.OUTPUTS_DIR / "kill_matrix_metadata.json"))
    mr_ids = meta["mr_ids"]
    mr_id_to_idx = {mr: i for i, mr in enumerate(mr_ids)}
    fdr = meta["fdr_per_mr"]

    greedy_ranking = json.load(open(config.OUTPUTS_DIR / "greedy_ranking.json"))
    greedy_order = [r["mr_id"] for r in greedy_ranking]

    # Model B predicted ordering (Phase 8) -- evaluated against MODEL A's kill
    # matrix as a proxy comparison, since Model B has no mutants of its own.
    model_b = json.load(open(config.OUTPUTS_DIR / "model_b_predicted_ranking.json"))
    meta_order = [r["mr_id"] for r in model_b["predicted_ranking"]]

    fdr_order = fdr_only_ordering(mr_ids, fdr)

    greedy_apfd = compute_apfd(greedy_order, km, mr_id_to_idx)
    meta_apfd = compute_apfd(meta_order, km, mr_id_to_idx)
    fdr_apfd = compute_apfd(fdr_order, km, mr_id_to_idx)
    random_apfds = random_baseline_apfds(mr_ids, km, mr_id_to_idx, n_trials=30)
    random_apfd_mean = float(np.mean(random_apfds))

    print(f"{'Ordering':<25}{'APFD'}")
    print("-" * 40)
    print(f"{'Random (mean of 30)':<25}{random_apfd_mean:.4f}")
    print(f"{'FDR-only':<25}{fdr_apfd:.4f}")
    print(f"{'Greedy (Model A)':<25}{greedy_apfd:.4f}")
    print(f"{'Meta-Classifier (Model B)':<25}{meta_apfd:.4f}")

    # FD@k curves, k=1..20
    print(f"\n{'k':<5}{'Greedy':<10}{'Random(mean)':<15}{'FDR-only'}")
    fd_at_k_results = {"greedy": [], "random": [], "fdr_only": []}
    for k in range(1, 21):
        g = compute_fd_at_k(greedy_order, km, mr_id_to_idx, k)
        r_vals = [compute_fd_at_k(list(np.random.RandomState(s).permutation(mr_ids)), km, mr_id_to_idx, k) for s in range(30)]
        r = float(np.mean(r_vals))
        f = compute_fd_at_k(fdr_order, km, mr_id_to_idx, k)
        fd_at_k_results["greedy"].append(g)
        fd_at_k_results["random"].append(r)
        fd_at_k_results["fdr_only"].append(f)
        print(f"{k:<5}{g:<10.3f}{r:<15.3f}{f:.3f}")

    # Wilcoxon: greedy's single APFD vs random's 30-trial distribution.
    # Greedy/meta-classifier are deterministic given a fixed Kill Matrix, so
    # the single value is repeated to pair against the 30 random trials.
    greedy_repeated = [greedy_apfd] * 30
    stat, pval = wilcoxon(greedy_repeated, random_apfds)
    print(f"\nWilcoxon (Greedy vs Random, 30 trials): statistic={stat:.4f}, p={pval:.6f}")
    print(f"Significant at p<0.05: {pval < 0.05}")

    results = {
        "apfd": {"random_mean": random_apfd_mean, "random_all_trials": random_apfds,
                  "fdr_only": fdr_apfd, "greedy": greedy_apfd, "meta_classifier": meta_apfd},
        "fd_at_k": fd_at_k_results,
        "wilcoxon": {"statistic": float(stat), "p_value": float(pval), "significant": bool(pval < 0.05)},
    }
    with open(config.OUTPUTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {config.OUTPUTS_DIR / 'evaluation_metrics.json'}")


if __name__ == "__main__":
    main()