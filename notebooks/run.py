"""
MR-Rank CLI: rank user-defined MRs against the fixed Model A / 80-mutant setup.

    python -m mrrank.run --mrs examples\\my_mrs.json --validate-only
    python -m mrrank.run --mrs examples\\my_mrs.json --out results\\run1
    python -m mrrank.run --mrs examples\\my_mrs.json --out results\\demo --fast

Pipeline: validate specs -> clean-model false-positive check (reject MRs above
5%) -> evaluate accepted MRs against the mutants -> merge with the cached
canonical 20x80 matrix -> re-measure cost for every MR in this run ->
Greedy ranking (FDR + cost + diversity).

Nothing in outputs/ is overwritten; results go to --out.
Top-level imports are torch-free; torch is only imported inside run_pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from mrrank import config
from mrrank.greedy import compute_norm_cost, get_kill_sets, greedy_rank
from mrrank.matrix_utils import load_cached_results, merge_results
from mrrank.mr_spec import MRSpecError, build_mrs_from_specs

CLEAN_RATE_LIMIT = 0.05      # report: MR must fire on <5% of clean-model images
FAST_MUTANTS = 20
FAST_IMAGES = 100
SEC_PER_PAIR_500 = 3.3       # rough: measured on the dev laptop at 500 images
MAX_SPEC_FILE_BYTES = 1_000_000
COST_REPEATS = 5            # median of N timings: wall-clock cost is noisy


# ----------------------------------------------------------------- helpers --

def load_specs(path) -> list:
    p = Path(path)
    if p.stat().st_size > MAX_SPEC_FILE_BYTES:
        raise MRSpecError("spec file too large")
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("mrs")
    if not isinstance(data, list):
        raise MRSpecError('spec file must be a JSON list of MR specs, or {"mrs": [...]}')
    return data


def pick_evenly(items: list, n: int) -> list:
    """Deterministic evenly spaced subset (keeps family spread when sorted)."""
    if n >= len(items):
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx.tolist()]


def stratified_indices(labels, n_total: int) -> list:
    """First n_total//n_classes images of each class (deterministic)."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    per = max(1, n_total // len(classes))
    idx: list = []
    for c in classes:
        idx.extend(np.where(labels == c)[0][:per].tolist())
    return sorted(idx)


def subset_columns(base: dict, mutant_ids: list) -> dict:
    """Restrict a cached result to the given mutants (same order as given)."""
    pos = {m: i for i, m in enumerate(base["mutant_ids"])}
    missing = [m for m in mutant_ids if m not in pos]
    if missing:
        raise ValueError(f"mutants not in cached matrix: {missing[:3]}")
    km = base["kill_matrix"][:, [pos[m] for m in mutant_ids]]
    return {
        **base,
        "kill_matrix": km,
        "mutant_ids": list(mutant_ids),
        "fdr_per_mr": {mid: float(km[i].mean()) for i, mid in enumerate(base["mr_ids"])},
    }


def rank_merged(merged: dict, cost_ms: dict, user_ids: set) -> list:
    mr_ids = merged["mr_ids"]
    norm_cost = compute_norm_cost({m: cost_ms[m] for m in mr_ids})
    kill_sets = get_kill_sets(merged["kill_matrix"], mr_ids)
    ranking = greedy_rank(mr_ids, merged["fdr_per_mr"], norm_cost, kill_sets)
    for r in ranking:
        mid = r["mr_id"]
        r["source"] = "user" if mid in user_ids else "canonical"
        r["cost_ms"] = cost_ms[mid]
        r["clean_violation_rate"] = merged["clean_violation_rates"].get(mid)
    return ranking


def write_outputs(out_dir: Path, ranking: list, merged: dict, record: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ranking.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    (out_dir / "run_config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    cols = ["rank", "mr_id", "source", "priority_score", "fdr", "norm_cost",
            "diversity", "cost_ms", "clean_violation_rate"]
    with open(out_dir / "ranking.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ranking:
            w.writerow({c: r.get(c) for c in cols})

    with open(out_dir / "kill_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mr_id"] + merged["mutant_ids"])
        for i, mid in enumerate(merged["mr_ids"]):
            w.writerow([mid] + merged["kill_matrix"][i].astype(int).tolist())


def print_ranking(ranking: list) -> None:
    print(f"\n{'Rank':<6}{'MR':<10}{'Source':<11}{'Priority':<10}{'FDR':<8}"
          f"{'NormCost':<10}{'Diversity'}")
    print("-" * 64)
    for r in ranking:
        print(f"{r['rank']:<6}{r['mr_id']:<10}{r['source']:<11}{r['priority_score']:<10.4f}"
              f"{r['fdr']:<8.3f}{r['norm_cost']:<10.3f}{r['diversity']:.3f}")


# ---------------------------------------------------------------- pipeline --

def run_pipeline(mrs, specs, out_dir: Path, n_mutants, n_images, keep_invalid: bool) -> int:
    from mrrank import data_module, mr_engine                  # cv2/numpy
    from mrrank import evaluation_module as ev                  # torch
    from mrrank.model_module import ModelWrapper

    t0 = time.time()
    base = load_cached_results(config.OUTPUTS_DIR)

    print("Loading Model A and the evaluation images...")
    wrapper = ModelWrapper.from_checkpoint(config.MODEL_A_CHECKPOINT, device="cpu")
    images, labels = data_module.get_eval_subset_raw()
    full_images = images
    if n_images and n_images < len(images):
        images = images[stratified_indices(labels, n_images)]

    mutant_paths = ev.list_mutant_paths()
    if n_mutants and n_mutants < len(mutant_paths):
        mutant_paths = pick_evenly(mutant_paths, n_mutants)
    mutant_ids = [p.stem for p in mutant_paths]
    approximate = len(images) < 500 or len(mutant_ids) < len(base["mutant_ids"])

    # Gate on ALL images (a 100-image rate is too noisy to accept/reject on);
    # the kill rule below uses the clean rate on the same images as the mutants.
    print(f"Checking {len(mrs)} MR(s) against the clean model on "
          f"{len(full_images)} images (limit {CLEAN_RATE_LIMIT:.0%})...")
    gate = ev.compute_clean_violation_rates(wrapper, full_images, mrs)
    clean = gate if images is full_images else \
        ev.compute_clean_violation_rates(wrapper, images, mrs)
    accepted = [m for m in mrs if keep_invalid or gate[m.id] <= CLEAN_RATE_LIMIT]
    rejected = {m.id: gate[m.id] for m in mrs if m not in accepted}
    for m in mrs:
        flag = "REJECTED (too noisy on the clean model)" if m.id in rejected else "ok"
        print(f"  {m.id:<10} clean violation rate {gate[m.id]:.3f}  {flag}")

    record = {
        "mode": "approximate" if approximate else "full",
        "n_images": int(len(images)), "n_mutants": len(mutant_ids),
        "seed": config.SEED, "kill_threshold": ev.KILL_THRESHOLD,
        "clean_rate_limit": CLEAN_RATE_LIMIT, "specs": specs,
        "clean_violation_rates": clean, "gate_violation_rates": gate,
        "rejected": rejected,
        "mutant_ids": mutant_ids,
    }

    if not accepted:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print("\nNo MRs passed the clean-model check; nothing to rank.")
        return 1

    pairs = len(accepted) * len(mutant_ids)
    est = pairs * SEC_PER_PAIR_500 * (len(images) / 500)
    est_txt = f"{est:.0f} s" if est < 120 else f"{est / 60:.0f} min"
    print(f"\nEvaluating {len(accepted)} MR(s) x {len(mutant_ids)} mutants = {pairs} pairs "
          f"(rough estimate {est_txt} on a laptop CPU)...")
    extra = ev.evaluate_mrs(accepted, mutant_paths, clean, images)
    extra["clean_violation_rates"] = {m.id: clean[m.id] for m in accepted}

    merged = merge_results(subset_columns(base, mutant_ids), extra)

    # Re-measure cost for ALL MRs in this run so they share hardware and images.
    print(f"Measuring cost for every MR on this machine (median of {COST_REPEATS} runs)...")
    cost_ms = {mr.id: statistics.median(
                   mr_engine.measure_mr_cost_ms(mr, wrapper, images) for _ in range(COST_REPEATS))
               for mr in list(mr_engine.ALL_MRS) + accepted}

    ranking = rank_merged(merged, cost_ms, {m.id for m in accepted})
    record["cost_ms"] = cost_ms
    record["elapsed_sec"] = round(time.time() - t0, 1)
    write_outputs(out_dir, ranking, merged, record)

    print_ranking(ranking)
    if approximate:
        print("\nNOTE: approximate run (reduced mutants/images). Use it for a quick look; "
              "run without --fast for the full evaluation.")
    print(f"\nSaved ranking.csv, ranking.json, kill_matrix.csv, run_config.json to {out_dir}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mrrank.run",
                                 description="Rank user-defined MRs with the MR-Rank pipeline.")
    ap.add_argument("--mrs", required=True, help="JSON file with MR specs")
    ap.add_argument("--out", default="results/run", help="output folder")
    ap.add_argument("--fast", action="store_true",
                    help=f"use {FAST_MUTANTS} mutants and {FAST_IMAGES} images (approximate)")
    ap.add_argument("--mutants", type=int, help="use only N evenly spaced mutants")
    ap.add_argument("--images", type=int, help="use only N stratified evaluation images")
    ap.add_argument("--keep-invalid", action="store_true",
                    help="evaluate MRs even if they fail the clean-model check")
    ap.add_argument("--validate-only", action="store_true",
                    help="check the spec file and exit (no torch needed)")
    args = ap.parse_args(argv)

    try:
        specs = load_specs(args.mrs)
        mrs = build_mrs_from_specs(specs)
    except (MRSpecError, ValueError, OSError) as e:
        print(f"Invalid MR file: {e}", file=sys.stderr)
        return 2

    print(f"{len(mrs)} MR spec(s) valid: {', '.join(m.id for m in mrs)}")
    if args.validate_only:
        return 0

    n_mutants = args.mutants or (FAST_MUTANTS if args.fast else None)
    n_images = args.images or (FAST_IMAGES if args.fast else None)
    return run_pipeline(mrs, specs, Path(args.out), n_mutants, n_images, args.keep_invalid)


if __name__ == "__main__":
    sys.exit(main())
