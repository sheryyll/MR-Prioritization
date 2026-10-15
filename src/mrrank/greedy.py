"""
Greedy MR prioritization (report Eq 4.1-4.4). Pure numpy/stdlib: NO torch,
sklearn or pandas, so the lean web app and the CLI can import it cheaply.

Priority(MR_i) = alpha*FDR(MR_i) + beta*(1-NormCost(MR_i)) + gamma*Diversity(MR_i, S)
alpha=0.5, beta=0.2, gamma=0.3

Moved verbatim from ranking_module.py; ranking_module re-exports these names.
"""
from __future__ import annotations

import numpy as np

ALPHA, BETA, GAMMA = 0.5, 0.2, 0.3


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
