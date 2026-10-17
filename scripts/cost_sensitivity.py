"""
Does the Greedy ranking depend on HOW cost is measured?

measure_mr_cost_ms times transform + model inference on all images, and the
inference part is identical for every MR, so stored NormCost values are nearly
constant. This script re-ranks the canonical 20 MRs from the EXISTING kill
matrix under three cost definitions and compares the orderings:

  stored      cost_ms from outputs/mr_validation.json (what the report used)
  transform   transform-only time (no model inference), min of N repeats
  none        cost ignored (all NormCost = 0, so the cost term is constant)

Run from the repo root:
    python scripts\\cost_sensitivity.py
Needs torch (data_module import) and the usual outputs/. Changes nothing.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from mrrank import config
from mrrank.greedy import compute_norm_cost, get_kill_sets, greedy_rank
from mrrank.matrix_utils import load_cached_results


def transform_cost_ms(mr, images, repeats: int = 15) -> float:
    """Transform-only wall time over all images; minimum of `repeats` (load noise is one-sided)."""
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        for img in images:
            mr.transform(img)
        best = min(best, (time.perf_counter() - t) * 1000)
    return best


def fd_at_k(kill_matrix: np.ndarray, mr_ids: list, order: list, k: int) -> float:
    """Fraction of detectable mutants killed by the first k MRs of `order`."""
    row = {m: i for i, m in enumerate(mr_ids)}
    detectable = kill_matrix.any(axis=0)
    if not detectable.sum():
        return 0.0
    first_k = kill_matrix[[row[m] for m in order[:k]]].any(axis=0)
    return float((first_k & detectable).sum() / detectable.sum())


def rank_with_cost(base: dict, cost_ms: dict) -> list:
    mr_ids = base["mr_ids"]
    norm = compute_norm_cost({m: cost_ms[m] for m in mr_ids})
    kill_sets = get_kill_sets(base["kill_matrix"], mr_ids)
    return [r["mr_id"] for r in greedy_rank(mr_ids, base["fdr_per_mr"], norm, kill_sets)]


def compare(base: dict, orders: dict, topk: int = 5, ks=(1, 2, 3, 4, 5)) -> dict:
    out = {"top": {n: o[:topk] for n, o in orders.items()}, "fd_at_k": {}, "kendall_tau": {}}
    for n, o in orders.items():
        out["fd_at_k"][n] = [round(fd_at_k(base["kill_matrix"], base["mr_ids"], o, k), 3) for k in ks]
    try:
        from scipy.stats import kendalltau
        ref = orders["stored"]
        pos_ref = {m: i for i, m in enumerate(ref)}
        for n, o in orders.items():
            if n == "stored":
                continue
            tau, _ = kendalltau([pos_ref[m] for m in o], list(range(len(o))))
            out["kendall_tau"][f"stored_vs_{n}"] = round(float(tau), 3)
    except ImportError:
        pass
    out["top5_overlap"] = {n: len(set(o[:5]) & set(orders["stored"][:5])) for n, o in orders.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args()

    from mrrank import data_module, mr_engine

    base = load_cached_results(config.OUTPUTS_DIR)
    validation = json.loads((config.OUTPUTS_DIR / "mr_validation.json").read_text(encoding="utf-8"))
    stored = {m: float(validation[m]["cost_ms"]) for m in base["mr_ids"]}

    images, _ = data_module.get_eval_subset_raw()
    transform = {m: transform_cost_ms(mr_engine.MR_BY_ID[m], images, args.repeats) for m in base["mr_ids"]}
    none = {m: 0.0 for m in base["mr_ids"]}

    ns, nt = compute_norm_cost(stored), compute_norm_cost(transform)
    print(f"{'MR':<6}{'stored ms':>11}{'NormCost':>10}{'transform ms':>14}{'NormCost':>10}")
    for m in base["mr_ids"]:
        print(f"{m:<6}{stored[m]:>11.1f}{ns[m]:>10.3f}{transform[m]:>14.2f}{nt[m]:>10.3f}")
    print(f"\nstored cost spread: min/max = {min(stored.values())/max(stored.values()):.3f}; "
          f"transform-only spread: {min(transform.values())/max(transform.values()):.3f}")

    orders = {"stored": rank_with_cost(base, stored),
              "transform": rank_with_cost(base, transform),
              "none": rank_with_cost(base, none)}
    result = compare(base, orders)

    print("\nTop 5 by cost definition:")
    for n, o in result["top"].items():
        print(f"  {n:<10}{o}   top-5 overlap with stored: {result['top5_overlap'][n]}/5")
    print("\nFD@k for k=1..5:")
    for n, v in result["fd_at_k"].items():
        print(f"  {n:<10}{v}")
    print("\nKendall tau vs stored ordering (1.0 = identical order):", result["kendall_tau"])

    out = config.OUTPUTS_DIR.parent / "results" / "cost_sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stored_ms": stored, "transform_ms": transform,
                               "orders": orders, **result}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
