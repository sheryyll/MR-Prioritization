"""
Auto-calibration of user MRs (torch-free; the model-dependent part is a callback).

A user writes "auto" for ONE numeric parameter of a simple MR:

    {"id": "U01", "type": "rotate", "params": {"angle": "auto"}}

and the tool searches for the LARGEST magnitude whose clean-model violation rate stays
under SEARCH_THRESHOLD (a little below the 5% validity limit). Larger magnitudes detect
more mutants, so the largest valid one is the most useful valid MR of that kind.

Method: bisection between the mildest and most aggressive allowed magnitudes, assuming
the violation rate grows with magnitude (same assumption as calibrate_mrs.py). The result
is specific to Model A and the evaluation images it was measured on.
"""
from __future__ import annotations

import copy
from typing import Callable

AUTO = "auto"
SEARCH_THRESHOLD = 0.04          # target; the validity limit itself stays at 5%
MAX_ITERATIONS = 9

# (type, param) -> (mildest magnitude, most aggressive allowed magnitude, integer?)
SEARCH: dict[tuple[str, str], tuple[float, float, bool]] = {
    ("rotate", "angle"): (0.2, 45.0, False),
    ("translation", "dx"): (0.25, 8.0, False),
    ("translation", "dy"): (0.25, 8.0, False),
    ("shear", "shear_factor"): (0.01, 0.5, False),
    ("scale", "factor"): (0.99, 0.5, False),           # zoom-out, as in the study
    ("noise", "sigma"): (0.001, 0.2, False),
    ("salt_pepper", "amount"): (0.0005, 0.1, False),
    ("sharpen", "amount"): (0.05, 3.0, False),
    ("brightness", "delta"): (1.0, 100.0, False),      # brightening direction
    ("contrast", "factor"): (1.01, 2.0, False),        # increasing direction
    ("saturation", "factor"): (1.01, 2.0, False),
    ("posterize", "bits"): (8, 1, True),
    ("jpeg", "quality"): (100, 1, True),
}


class AutoSpecError(ValueError):
    """Invalid use of "auto" in an MR spec."""


def find_auto(spec: dict):
    """Name of the parameter set to "auto" (at most one, simple types only), else None."""
    params = spec.get("params") if isinstance(spec, dict) else None
    if not isinstance(params, dict):
        return None
    autos = [k for k, v in params.items() if isinstance(v, str) and v.strip().lower() == AUTO]
    if not autos:
        return None
    if len(autos) > 1:
        raise AutoSpecError(f"{spec.get('id')}: only one parameter can be \"auto\"")
    t = spec.get("type")
    if (t, autos[0]) not in SEARCH:
        ok = sorted(p for (ty, p) in SEARCH if ty == t)
        raise AutoSpecError(
            f"{spec.get('id')}: \"auto\" is not supported for {t}.{autos[0]}"
            + (f" (supported: {', '.join(ok)})" if ok else " (composites cannot be auto-calibrated)"))
    return autos[0]


def _rnd(value: float, integer: bool):
    return int(round(value)) if integer else round(float(value), 4)


def prepare(specs: list):
    """Return (placeholder_specs, jobs). Placeholders carry the mildest magnitude so the
    normal validator can check everything else; jobs = [(index, param), ...]."""
    if not isinstance(specs, list):
        raise AutoSpecError("specs must be a list")
    placeholder, jobs = copy.deepcopy(specs), []
    for i, spec in enumerate(placeholder):
        p = find_auto(spec)
        if p is not None:
            mild, _aggr, integer = SEARCH[(spec["type"], p)]
            spec["params"][p] = _rnd(mild, integer)
            jobs.append((i, p))
    return placeholder, jobs


def bisect_max_valid(measure: Callable[[float], float], mild: float, aggressive: float,
                     threshold: float = SEARCH_THRESHOLD, integer: bool = False,
                     max_iter: int = MAX_ITERATIONS) -> dict:
    """Largest magnitude (towards `aggressive`) whose rate is below `threshold`."""
    trace: list = []

    def m(v):
        r = measure(v)
        trace.append((v, r))
        return r

    mild, aggressive = _rnd(mild, integer), _rnd(aggressive, integer)
    r_mild = m(mild)
    if r_mild >= threshold:
        return {"value": None, "rate": None, "trace": trace,
                "note": f"even the mildest allowed magnitude ({mild}) fires on {r_mild:.1%} of the "
                        f"clean model's predictions (target < {threshold:.1%})"}
    r_aggr = m(aggressive)
    if r_aggr < threshold:
        return {"value": aggressive, "rate": r_aggr, "trace": trace,
                "note": "limited by the maximum allowed magnitude"}
    lo, hi, best, best_r = mild, aggressive, mild, r_mild
    for _ in range(max_iter):
        mid = _rnd((lo + hi) / 2, integer)
        if mid == lo or mid == hi:
            break
        r = m(mid)
        if r < threshold:
            lo, best, best_r = mid, mid, r
        else:
            hi = mid
    return {"value": best, "rate": best_r, "trace": trace, "note": None}


def resolve(specs: list, jobs: list, rate_fn: Callable[[dict], float],
            threshold: float = SEARCH_THRESHOLD, log: Callable[[str], None] = lambda s: None):
    """Calibrate each job. rate_fn(candidate_spec) -> clean violation rate.
    Returns (final_specs with None for MRs that could not be calibrated, report)."""
    final, report = copy.deepcopy(specs), {}
    for idx, param in jobs:
        spec = specs[idx]
        mild, aggr, integer = SEARCH[(spec["type"], param)]

        def measure(v, spec=spec, param=param):
            cand = copy.deepcopy(spec)
            cand["params"][param] = v
            r = rate_fn(cand)
            log(f"    {spec['id']}: {param}={v} -> {r:.1%}")
            return r

        res = bisect_max_valid(measure, mild, aggr, threshold, integer)
        report[spec["id"]] = {"param": param, **res}
        if res["value"] is None:
            final[idx] = None
        else:
            final[idx]["params"][param] = res["value"]
    return final, report
