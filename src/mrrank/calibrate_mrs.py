"""
Automated MR calibration via binary search.

For each MR with a continuous, single tunable magnitude parameter, finds
a magnitude value whose measured false-positive violation rate on the
given model has genuine margin below the target threshold (search targets
`search_threshold`, e.g. 3.5%, leaving headroom below the 5% reporting
cutoff -- prevents values that flip pass/fail on minor measurement
variance across runs).

Usage:
    python -m mrrank.calibrate_mrs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mrrank import config, data_module, mr_engine
from mrrank.model_module import ModelWrapper
from mrrank.mr_engine import (
    MetamorphicRelation, _gaussian_noise, _contrast, _brightness,
    _salt_pepper, _saturation, _shear, _translate, _scale, _rotate,
)


@dataclass
class CalibrationTarget:
    name: str
    param_name: str
    low: float           # magnitude with LOWEST violation rate (mildest)
    high: float           # magnitude with HIGHEST violation rate (most aggressive)
    build_fn: Callable[[float], Callable]
    threshold: float = 0.05
    search_threshold: float = 0.035
    precision: float = 0.0005
    max_iterations: int = 20
    is_integer: bool = False
    min_meaningful_magnitude: float | None = None


def binary_search_calibrate(target: CalibrationTarget, wrapper, images) -> dict:
    def measure(value: float) -> float:
        mr = MetamorphicRelation(
            id="CAL", name=target.name, category="x", mr_type="x",
            magnitude=value, is_composite=False, fn=target.build_fn(value),
        )
        return mr_engine.compute_violation_rate(mr, wrapper, images)

    trace = []

    rate_low = measure(target.low)
    trace.append((target.low, rate_low))
    if rate_low >= target.search_threshold:
        return {
            "best_passing_value": None,
            "rate_at_best": None,
            "trace": trace,
            "note": f"Even the lowest tested magnitude ({target.low}) has "
                    f"violation rate {rate_low*100:.2f}% >= search target "
                    f"{target.search_threshold*100:.1f}%. No passing value "
                    f"with adequate margin exists in the given search range.",
        }

    lo, hi = target.low, target.high
    best_passing_value, best_rate = lo, rate_low

    for _ in range(target.max_iterations):
        if abs(hi - lo) < target.precision:
            break
        mid = (lo + hi) / 2
        if target.is_integer:
            mid = round(mid)
            if mid == best_passing_value or mid == hi:
                break

        rate = measure(mid)
        trace.append((mid, rate))

        if rate < target.search_threshold:
            best_passing_value, best_rate = mid, rate
            lo = mid
        else:
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
            "MR02 Shear", "shear_factor", low=0.02, high=0.30,
            build_fn=lambda v: (lambda img: _shear(img, shear_factor=v)),
        ),
        CalibrationTarget(
            "MR03 Rotate (fine)", "angle", low=0.2, high=3.0,
            build_fn=lambda v: (lambda img: _rotate(img, angle=v)),
            min_meaningful_magnitude=1.0,  # below 1 degree, rotation is
            # visually negligible on a 32x32 image
        ),
        CalibrationTarget(
            "MR04 Translate Horizontal", "dx", low=0.5, high=6,
            build_fn=lambda v: (lambda img: _translate(img, dx=v, dy=0)),
        ),
        CalibrationTarget(
            "MR05 Translate Vertical", "dy", low=0.5, high=6,
            build_fn=lambda v: (lambda img: _translate(img, dx=0, dy=v)),
        ),
        CalibrationTarget(
            "MR14 Translate Diagonal", "dx=dy", low=0.5, high=6,
            build_fn=lambda v: (lambda img: _translate(img, dx=v, dy=v)),
        ),
        CalibrationTarget(
            "MR15 Scale Down", "factor", low=0.99, high=0.80,
            build_fn=lambda v: (lambda img: _scale(img, factor=v)),
            min_meaningful_magnitude=0.97,  # above 0.97, zoom-out is
            # visually negligible
        ),
        CalibrationTarget(
            "MR19 Flip+Noise (noise component)", "sigma", low=0.002, high=0.02,
            build_fn=lambda v: (lambda img: _gaussian_noise(
                mr_engine._hflip(img), sigma_uint8=v * 255, mr_id="MR19", global_seed=global_seed)),
            # NOTE: this measures violation rate of the FULL composite
            # (flip + noise), so it will NOT go below MR01's own ~10%
            # floor no matter how small sigma gets -- included for
            # completeness/documentation, not because a pass is expected.
        ),
    ]


def main():
    print("Loading Model A checkpoint...")
    wrapper = ModelWrapper.from_checkpoint(config.MODEL_A_CHECKPOINT, device="cpu")
    images, _ = data_module.get_eval_subset_raw()

    targets = build_default_targets(config.SEED)

    print(f"\n{'MR':<32}{'Param':<12}{'Best Value':<14}{'Rate at Best':<16}{'Iterations'}")
    print("-" * 90)

    results = {}
    for target in targets:
        result = binary_search_calibrate(target, wrapper, images)
        results[target.name] = result
        if result["best_passing_value"] is not None:
            print(f"{target.name:<32}{target.param_name:<12}"
                  f"{result['best_passing_value']:<14.4f}"
                  f"{result['rate_at_best']*100:<15.2f}%"
                  f"{len(result['trace'])}")
        else:
            print(f"{target.name:<32}{target.param_name:<12}"
                  f"{'NO PASS':<14}{'--':<16}{len(result['trace'])}")
            print(f"    NOTE: {result['note']}")

    return results


if __name__ == "__main__":
    main()