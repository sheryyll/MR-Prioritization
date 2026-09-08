"""
Ranking Module: Greedy MR prioritization (report Eq 4.1-4.4).

Priority(MR_i) = alpha*FDR(MR_i) + beta*(1-NormCost(MR_i)) + gamma*Diversity(MR_i, S)
alpha=0.5, beta=0.2, gamma=0.3
"""

from __future__ import annotations

import json

import numpy as np

from mrrank import config

ALPHA, BETA, GAMMA = 0.5, 0.2, 0.3


def load_kill_matrix():
    km = np.load(config.OUTPUTS_DIR / "kill_matrix.npy")
    meta = json.load(open(config.OUTPUTS_DIR / "kill_matrix_metadata.json"))
    return km, meta


def get_kill_sets(kill_matrix: np.ndarray, mr_ids: list[str]) -> dict[str, set]:
    return {
        mr_ids[i]: set(np.where(kill_matrix[i])[0].tolist())
        for i in range(len(mr_ids))
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def compute_norm_cost(cost_by_mr: dict[str, float]) -> dict[str, float]:
    max_cost = max(cost_by_mr.values())
    return {mr: (c / max_cost if max_cost else 0.0) for mr, c in cost_by_mr.items()}


def diversity(mr_id: str, selected: list[str], kill_sets: dict[str, set]) -> float:
    if not selected:
        return 1.0
    max_sim = max(jaccard(kill_sets[mr_id], kill_sets[s]) for s in selected)
    return 1 - max_sim


def greedy_rank(mr_ids: list[str], fdr: dict[str, float], norm_cost: dict[str, float],
                 kill_sets: dict[str, set]) -> list[dict]:
    remaining = list(mr_ids)
    selected: list[str] = []
    ranking = []

    while remaining:
        scores = {}
        for mr in remaining:
            div = diversity(mr, selected, kill_sets)
            scores[mr] = ALPHA * fdr[mr] + BETA * (1 - norm_cost[mr]) + GAMMA * div

        best = max(scores, key=scores.get)
        ranking.append({
            "rank": len(ranking) + 1,
            "mr_id": best,
            "priority_score": scores[best],
            "fdr": fdr[best],
            "norm_cost": norm_cost[best],
            "diversity": diversity(best, selected, kill_sets),
        })
        selected.append(best)
        remaining.remove(best)

    return ranking


def main():
    km, meta = load_kill_matrix()
    mr_ids = meta["mr_ids"]
    fdr = meta["fdr_per_mr"]
    kill_sets = get_kill_sets(km, mr_ids)

    # Cost from mr_validation.json (cost_ms per MR)
    validation = json.load(open(config.OUTPUTS_DIR / "mr_validation.json"))
    cost_by_mr = {mr: validation[mr]["cost_ms"] for mr in mr_ids}
    norm_cost = compute_norm_cost(cost_by_mr)

    ranking = greedy_rank(mr_ids, fdr, norm_cost, kill_sets)

    print(f"{'Rank':<6}{'MR':<8}{'Priority':<12}{'FDR':<8}{'NormCost':<10}{'Diversity'}")
    print("-" * 55)
    for r in ranking:
        print(f"{r['rank']:<6}{r['mr_id']:<8}{r['priority_score']:<12.4f}"
              f"{r['fdr']:<8.4f}{r['norm_cost']:<10.4f}{r['diversity']:.4f}")

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUTS_DIR / "greedy_ranking.json", "w") as f:
        json.dump(ranking, f, indent=2)
    print(f"\nSaved to {config.OUTPUTS_DIR / 'greedy_ranking.json'}")


if __name__ == "__main__":
    main()