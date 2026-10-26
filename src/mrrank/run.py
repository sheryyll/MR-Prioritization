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

from mrrank import config, mr_autocal
from mrrank.greedy import compute_norm_cost, get_kill_sets, greedy_rank
from mrrank.matrix_utils import load_cached_results, merge_results
from mrrank.mr_spec import MRSpecError, build_mrs_from_specs

CLEAN_RATE_LIMIT = 0.05      # report: MR must fire on <5% of clean-model images
FAST_MUTANTS = 20
FAST_IMAGES = 100
SEC_PER_PAIR_500 = 3.3       # rough: measured on the dev laptop at 500 images
MAX_SPEC_FILE_BYTES = 1_000_000
COST_REPEATS = 5            # timings per MR; the MINIMUM is used (load noise is one-sided)
CALIBRATION_MRS = 5         # canonical MRs re-timed to scale user-MR costs
STABILITY_TRIALS = 200      # bootstrap trials for the rank-stability analysis
STABILITY_COST_NOISE = 0.02 # +-0.02 on NormCost: about the run-to-run spread observed
TOPK = 5


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


def interleaved_min(measure, mrs, repeats: int) -> dict:
    """Minimum timing per MR, measured round-robin (all MRs, then again, ...)
    so CPU drift hits every MR equally instead of whichever is timed last."""
    best = {m.id: float("inf") for m in mrs}
    for _ in range(repeats):
        for m in mrs:
            best[m.id] = min(best[m.id], measure(m))
    return best


def calibrated_costs(stored: dict, ref_now: dict, new_now: dict) -> dict:
    """Stored canonical costs + user-MR timings rescaled to the same conditions.
    ratio = median(now/stored) over the re-timed reference MRs."""
    ratios = [ref_now[m] / stored[m] for m in ref_now if stored.get(m)]
    if not ratios:
        raise ValueError("no usable reference MRs for cost calibration")
    ratio = statistics.median(ratios)
    out = dict(stored)
    out.update({m: c / ratio for m, c in new_now.items()})
    return out


def build_scope(base_subset: dict, extra: dict, scope: str) -> dict:
    """scope='mine': rank only the user's MRs. scope='all': stack them under the study's
    canonical MRs (same mutants) so they compete with, and are diversified against, the 20."""
    if scope == "all":
        return merge_results(base_subset, extra)
    if scope != "mine":
        raise ValueError(f"unknown scope {scope!r}")
    return {k: extra[k] for k in ("kill_matrix", "mr_ids", "mutant_ids",
                                  "clean_violation_rates", "fdr_per_mr")}


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



def rank_stability(merged: dict, cost_ms: dict, trials: int = STABILITY_TRIALS,
                   cost_noise: float = STABILITY_COST_NOISE, seed=None) -> dict:
    """How much does each MR's Greedy rank move? Each trial resamples the mutants
    (bootstrap, with replacement) and jitters NormCost by +-cost_noise, then re-runs
    Greedy. Returns per MR: median rank, 10th/90th percentile rank, P(top-k)."""
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    mr_ids, km = merged["mr_ids"], merged["kill_matrix"]
    n_mut = km.shape[1]
    base_norm = compute_norm_cost({m: cost_ms[m] for m in mr_ids})
    ranks = {m: [] for m in mr_ids}
    for _ in range(trials):
        kb = km[:, rng.integers(0, n_mut, n_mut)]
        fdr = {m: float(kb[i].mean()) for i, m in enumerate(mr_ids)}
        norm = {m: float(np.clip(base_norm[m] + rng.uniform(-cost_noise, cost_noise), 0.0, 1.0))
                for m in mr_ids}
        for r in greedy_rank(mr_ids, fdr, norm, get_kill_sets(kb, mr_ids)):
            ranks[r["mr_id"]].append(r["rank"])
    out = {}
    for m, rs in ranks.items():
        a = np.asarray(rs)
        out[m] = {"rank_median": float(np.median(a)),
                  "rank_p10": int(np.floor(np.percentile(a, 10))),
                  "rank_p90": int(np.ceil(np.percentile(a, 90))),
                  "top5_prob": float((a <= TOPK).mean())}
    return out


def add_stability(ranking: list, stability: dict) -> None:
    for r in ranking:
        r.update(stability[r["mr_id"]])


def load_run(run_dir: Path):
    """Rebuild (merged, cost_ms, user_ids) from a finished run's saved files."""
    run_dir = Path(run_dir)
    rows = list(csv.reader(open(run_dir / "kill_matrix.csv", encoding="utf-8")))
    mutant_ids, mr_ids = rows[0][1:], [r[0] for r in rows[1:]]
    km = np.array([[int(x) for x in r[1:]] for r in rows[1:]], dtype=bool)
    info = {r["mr_id"]: r for r in json.loads((run_dir / "ranking.json").read_text(encoding="utf-8"))}
    merged = {"kill_matrix": km, "mr_ids": mr_ids, "mutant_ids": mutant_ids,
              "clean_violation_rates": {m: info[m].get("clean_violation_rate") for m in mr_ids},
              "fdr_per_mr": {m: float(km[i].mean()) for i, m in enumerate(mr_ids)}}
    cost_ms = {m: float(info[m]["cost_ms"]) for m in mr_ids}
    user_ids = {m for m in mr_ids if info[m].get("source") == "user"}
    return merged, cost_ms, user_ids


def rerank(run_dir: Path) -> int:
    """Recompute ranking + stability from a finished run (no evaluation needed)."""
    merged, cost_ms, user_ids = load_run(run_dir)
    ranking = rank_merged(merged, cost_ms, user_ids)
    add_stability(ranking, rank_stability(merged, cost_ms))
    write_ranking(Path(run_dir), ranking)
    print_ranking(ranking)
    print(f"\nUpdated ranking.csv and ranking.json in {run_dir}")
    return 0

RANK_COLS = ["rank", "mr_id", "source", "priority_score", "fdr", "norm_cost", "diversity",
             "cost_ms", "clean_violation_rate", "rank_median", "rank_p10", "rank_p90", "top5_prob"]


def write_ranking(out_dir: Path, ranking: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ranking.json").write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    cols = [c for c in RANK_COLS if c in ranking[0]]
    with open(out_dir / "ranking.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ranking:
            w.writerow({c: r.get(c) for c in cols})


def write_outputs(out_dir: Path, ranking: list, merged: dict, record: dict) -> None:
    write_ranking(out_dir, ranking)
    (out_dir / "run_config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    with open(out_dir / "kill_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mr_id"] + merged["mutant_ids"])
        for i, mid in enumerate(merged["mr_ids"]):
            w.writerow([mid] + merged["kill_matrix"][i].astype(int).tolist())


def print_ranking(ranking: list) -> None:
    has = "rank_p10" in ranking[0]
    head = (f"\n{'Rank':<6}{'MR':<10}{'Source':<11}{'Priority':<10}{'FDR':<8}"
            f"{'NormCost':<10}{'Diversity':<11}")
    print(head + (f"{'Rank range':<12}{'P(top5)'}" if has else ""))
    print("-" * (76 if has else 64))
    for r in ranking:
        line = (f"{r['rank']:<6}{r['mr_id']:<10}{r['source']:<11}{r['priority_score']:<10.4f}"
                f"{r['fdr']:<8.3f}{r['norm_cost']:<10.3f}{r['diversity']:<11.3f}")
        if has:
            line += f"{str(r['rank_p10']) + '-' + str(r['rank_p90']):<12}{r['top5_prob']:.2f}"
        print(line)


# ---------------------------------------------------------------- pipeline --

def run_pipeline(mrs, specs, out_dir: Path, n_mutants, n_images, keep_invalid: bool,
                 scope: str = "mine", auto_jobs=()) -> int:
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

    # Auto-calibration: find the largest magnitude under the clean-model limit for each
    # parameter the user set to "auto" (measured on ALL images, so the gate below agrees).
    calibration = None
    if auto_jobs:
        print(f"Auto-calibrating {len(auto_jobs)} MR(s): largest magnitude with a clean "
              f"violation rate under {mr_autocal.SEARCH_THRESHOLD:.1%}...")

        def rate_fn(cand):
            mr = build_mrs_from_specs([cand], max_mrs=1)[0]
            return ev.compute_clean_violation_rates(wrapper, full_images, [mr])[mr.id]

        final_specs, calibration = mr_autocal.resolve(specs, list(auto_jobs), rate_fn,
                                                      log=lambda t: print(t, flush=True))
        for mid, rep in calibration.items():
            if rep["value"] is None:
                print(f"  {mid}: could not be calibrated - {rep['note']}")
            else:
                print(f"  {mid}: {rep['param']} = {rep['value']} "
                      f"(clean violation rate {rep['rate']:.1%})"
                      + (f"; {rep['note']}" if rep["note"] else ""))
        specs = [sp for sp in final_specs if sp is not None]
        if not specs:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "run_config.json").write_text(
                json.dumps({"calibration": calibration, "specs": []}, indent=2), encoding="utf-8")
            print("\nNo MR could be calibrated; nothing to rank.")
            return 1
        mrs = build_mrs_from_specs(specs)

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
        "seed": config.SEED, "scope": scope, "kill_threshold": ev.KILL_THRESHOLD,
        "clean_rate_limit": CLEAN_RATE_LIMIT, "specs": specs,
        "clean_violation_rates": clean, "gate_violation_rates": gate,
        "rejected": rejected,
        "mutant_ids": mutant_ids,
    }
    record["calibration"] = calibration

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
    # Report the 500-image gate rate (comparable with the canonical MRs' cached rates);
    # the kill decisions above already used the same-image `clean` rates.
    extra["clean_violation_rates"] = {m.id: gate[m.id] for m in accepted}

    merged = build_scope(subset_columns(base, mutant_ids), extra, scope)

    # Canonical costs come from mr_validation.json (fixed, as in the report).
    # User-MR timings are rescaled by re-timing a few canonical MRs right now.
    print(f"Measuring cost of the new MR(s) (min of {COST_REPEATS} runs, "
          f"calibrated on {CALIBRATION_MRS} canonical MRs)...")
    validation = json.loads((config.OUTPUTS_DIR / "mr_validation.json").read_text(encoding="utf-8"))
    stored = {mid: float(validation[mid]["cost_ms"]) for mid in base["mr_ids"]}

    refs = pick_evenly(list(mr_engine.ALL_MRS), CALIBRATION_MRS)
    times = interleaved_min(lambda mr: mr_engine.measure_mr_cost_ms(mr, wrapper, images),
                            refs + accepted, COST_REPEATS)
    cost_ms = calibrated_costs(stored,
                               {mr.id: times[mr.id] for mr in refs},
                               {mr.id: times[mr.id] for mr in accepted})
    record["cost_source"] = "mr_validation.json (canonical) + calibrated timing (user MRs)"

    ranking = rank_merged(merged, cost_ms, {m.id for m in accepted})
    add_stability(ranking, rank_stability(merged, cost_ms))
    record["stability"] = {"trials": STABILITY_TRIALS, "cost_noise": STABILITY_COST_NOISE,
                           "method": "bootstrap over mutants + NormCost jitter"}
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
    ap.add_argument("--mrs", help="JSON file with MR specs")
    ap.add_argument("--rerank", metavar="RUN_DIR",
                    help="recompute ranking + rank stability from a finished run folder")
    ap.add_argument("--out", default="results/run", help="output folder")
    ap.add_argument("--fast", action="store_true",
                    help=f"use {FAST_MUTANTS} mutants and {FAST_IMAGES} images (approximate)")
    ap.add_argument("--mutants", type=int, help="use only N evenly spaced mutants")
    ap.add_argument("--images", type=int, help="use only N stratified evaluation images")
    ap.add_argument("--scope", choices=["mine", "all"], default="mine",
                    help="mine: rank only your MRs (default); all: rank them together with the 20 study MRs")
    ap.add_argument("--keep-invalid", action="store_true",
                    help="evaluate MRs even if they fail the clean-model check")
    ap.add_argument("--validate-only", action="store_true",
                    help="check the spec file and exit (no torch needed)")
    args = ap.parse_args(argv)
    if args.rerank:
        return rerank(Path(args.rerank))
    if not args.mrs:
        ap.error("--mrs is required unless --rerank is given")

    try:
        specs = load_specs(args.mrs)
        placeholder, auto_jobs = mr_autocal.prepare(specs)
        mrs = build_mrs_from_specs(placeholder)      # validates everything except the auto value
    except (MRSpecError, ValueError, OSError) as e:
        print(f"Invalid MR file: {e}", file=sys.stderr)
        return 2

    print(f"{len(mrs)} MR spec(s) valid: {', '.join(m.id for m in mrs)}"
          + (f" ({len(auto_jobs)} to be auto-calibrated)" if auto_jobs else ""))
    if args.validate_only:
        return 0

    n_mutants = args.mutants or (FAST_MUTANTS if args.fast else None)
    n_images = args.images or (FAST_IMAGES if args.fast else None)
    return run_pipeline(mrs, specs, Path(args.out), n_mutants, n_images, args.keep_invalid,
                        args.scope, auto_jobs)


if __name__ == "__main__":
    sys.exit(main())
