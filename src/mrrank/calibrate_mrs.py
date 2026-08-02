"""
Automated MR calibration via binary search.

For each MR with a continuous, single tunable magnitude parameter, finds
the LARGEST magnitude value whose measured false-positive violation rate
on the given model is still under the target threshold (default 5%) --
i.e., the strongest version of the transform that still passes. This
replaces manual grid-sweeping (sweep_mr_params.py) with a principled
search that converges on the actual boundary rather than sampling a fixed
list of candidate values and potentially straddling the true threshold
without ever landing near it.

Search strategy: binary search over the magnitude range [low, high],
assuming violation rate is monotonically non-decreasing in magnitude
(true for all MRs calibrated so far: noise sigma, contrast/brightness
delta or factor distance-from-identity, salt-and-pepper amount). Stops
when the search interval is narrower than `precision`, or after
`max_iterations` steps (whichever comes first).

Usage:
    python -m mrrank.calibrate_mrs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mrrank import config, data_module, mr_engine
from mrrank.model_module import ModelWrapper
from mrrank.mr_engine import (
    MetamorphicRelation, _gaussian_noise, _contrast, _brightness, _salt_pepper, _saturation,
)


@dataclass
class CalibrationTarget:
    name: str
    param_name: str
    low: float          # magnitude with LOWEST violation rate (safest/mildest)
    high: float          # magnitude with HIGHEST violation rate (Appendix A's original, or current failing value)
    build_fn: Callable[[float], Callable]
    threshold: float = 0.05
    precision: float = 0.0005   # stop once search interval is narrower than this
    max_iterations: int = 20
    is_integer: bool = False    # True for kernel sizes etc. that must stay integer/odd
    min_meaningful_magnitude: float | None = None
    # If the binary search converges to a value LESS disruptive than this
    # (i.e., closer to `low`/identity than this floor), the result is
    # rejected as a degenerate "softened into uselessness" calibration,
    # even if it technically passes the violation-rate threshold. This
    # prevents e.g. "1 degree rotation passes" from being accepted as a
    # legitimate calibrated MR.

def binary_search_calibrate(target: CalibrationTarget, wrapper, images) -> dict:
    """
    Returns dict with: best_passing_value (or None if even `low` fails),
    rate_at_best, iterations_run, and a full trace of (value, rate) tried.
    """
    def measure(value: float) -> float:
        mr = MetamorphicRelation(
            id="CAL", name=target.name, category="x", mr_type="x",
            magnitude=value, is_composite=False, fn=target.build_fn(value),
        )
        return mr_engine.compute_violation_rate(mr, wrapper, images)

    trace = []

    rate_low = measure(target.low)
    trace.append((target.low, rate_low))
    if rate_low >= target.threshold:
        # Even the mildest candidate fails -- no value in range passes.
        return {
            "best_passing_value": None,
            "rate_at_best": None,
            "trace": trace,
            "note": f"Even the lowest tested magnitude ({target.low}) has "
                    f"violation rate {rate_low*100:.2f}% >= threshold. "
                    f"No passing value exists in the given search range.",
        }

    lo, hi = target.low, target.high
    best_passing_value, best_rate = lo, rate_low

    for _ in range(target.max_iterations):
        if abs(hi - lo) < target.precision:
            break
        mid = (lo + hi) / 2
        if target.is_integer:
            mid = round(mid)
            if mid % 2 == 0:  # keep kernel-size-style params odd
                mid += 1
            if mid == best_passing_value or mid >= hi:
                break

        rate = measure(mid)
        trace.append((mid, rate))

        if rate < target.threshold:
            # mid passes -- it's our new best (closer to `high`, i.e.
            # stronger transform), search the stronger half
            best_passing_value, best_rate = mid, rate
            lo = mid
        else:
            # mid fails -- search the milder half
            hi = mid
    if (target.min_meaningful_magnitude is not None
            and abs(best_passing_value - target.low) < abs(target.min_meaningful_magnitude - target.low)):
        return {
            "best_passing_value": None,
            "rate_at_best": None,
            "trace": trace,
            "note": f"Binary search converged to {best_passing_value}, which is "
                    f"less disruptive than the minimum meaningful magnitude "
                    f"({target.min_meaningful_magnitude}) -- rejected as a "
                    f"degenerate (near-identity) calibration, not a genuine pass.",
        }

    return {
        "best_passing_value": best_passing_value,
        "rate_at_best": best_rate,
        "trace": trace,
        "note": None,
    }


def build_default_targets(global_seed: int) -> list[CalibrationTarget]:
    return [
        CalibrationTarget(
            "MR06 Gaussian Noise (low)", "sigma", low=0.002, high=0.02,
            build_fn=lambda v: (lambda img: _gaussian_noise(
                img, sigma_uint8=v * 255, mr_id="MR06", global_seed=global_seed)),
        ),
        CalibrationTarget(
            "MR07 Gaussian Noise (high)", "sigma", low=0.01, high=0.10,
            build_fn=lambda v: (lambda img: _gaussian_noise(
                img, sigma_uint8=v * 255, mr_id="MR07", global_seed=global_seed)),
        ),
        CalibrationTarget(
            "MR08 Brightness Increase", "delta", low=1, high=30,
            build_fn=lambda v: (lambda img: _brightness(img, delta=v)),
        ),
        CalibrationTarget(
            "MR09 Brightness Decrease", "delta", low=-1, high=-30,
            build_fn=lambda v: (lambda img: _brightness(img, delta=v)),
        ),
        CalibrationTarget(
            "MR10 Contrast Increase", "factor", low=1.02, high=1.5,
            build_fn=lambda v: (lambda img: _contrast(img, factor=v)),
        ),
        CalibrationTarget(
            "MR11 Contrast Decrease", "factor", low=0.98, high=0.5,
            build_fn=lambda v: (lambda img: _contrast(img, factor=v)),
        ),
        CalibrationTarget(
            "MR16 Salt & Pepper", "amount", low=0.0005, high=0.02,
            build_fn=lambda v: (lambda img: _salt_pepper(
                img, amount=v, mr_id="MR16", global_seed=global_seed)),
        ),
        CalibrationTarget(
            "MR17 Saturation", "factor", low=1.02, high=1.5,
            build_fn=lambda v: (lambda img: _saturation(img, factor=v)),
        ),
        CalibrationTarget(
            "MR03 Rotate 15", "angle", low=1, high=15,
            build_fn=lambda v: (lambda img: mr_engine._rotate(img, angle=v)),
            min_meaningful_magnitude=5,  # below 5 degrees, rotation is
            # visually negligible on a 32x32 image; not a meaningful test
        ),
        CalibrationTarget(
            "MR14 Center Crop", "fraction", low=0.99, high=0.80,
            build_fn=lambda v: (lambda img: mr_engine._center_crop_resize(img, fraction=v)),
        ),
        CalibrationTarget(
            "MR18 JPEG Compression", "quality", low=99, high=50,
            build_fn=lambda v: (lambda img: mr_engine._jpeg_compress(img, quality=int(v))),
            precision=1,
            min_meaningful_magnitude=85,  # quality > 85 is near-lossless,
            # not a meaningful compression-artifact test
        ),
    ]


def main():
    print("Loading Model A checkpoint...")
    wrapper = ModelWrapper.from_checkpoint(config.MODEL_A_CHECKPOINT, device="cpu")
    images, _ = data_module.get_eval_subset_raw()

    targets = build_default_targets(config.SEED)

    print(f"\n{'MR':<28}{'Param':<10}{'Best Value':<14}{'Rate at Best':<16}{'Iterations'}")
    print("-" * 85)

    results = {}
    for target in targets:
        result = binary_search_calibrate(target, wrapper, images)
        results[target.name] = result
        if result["best_passing_value"] is not None:
            print(f"{target.name:<28}{target.param_name:<10}"
                  f"{result['best_passing_value']:<14.4f}"
                  f"{result['rate_at_best']*100:<15.2f}%"
                  f"{len(result['trace'])}")
        else:
            print(f"{target.name:<28}{target.param_name:<10}"
                  f"{'NO PASS':<14}{'--':<16}{len(result['trace'])}")
            print(f"    NOTE: {result['note']}")

    return results


if __name__ == "__main__":
    main()