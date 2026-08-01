"""
Phase 3 validation entry point: runs all 20 MRs against the clean, real
Model A checkpoint, confirms each has a false-positive violation rate
under 5% (report Section 4.3, MR Engine), and persists results +
per-MR cost data to outputs/mr_validation.json (consumed later by
Phase 6's NormCost calculation).

Usage:
    python -m mrrank.validate_mrs
"""

from __future__ import annotations

import json

from mrrank import config, data_module, mr_engine
from mrrank.model_module import ModelWrapper


def main():
    print("Loading Model A checkpoint...")
    wrapper = ModelWrapper.from_checkpoint(config.MODEL_A_CHECKPOINT, device="cpu")

    print("Loading fixed 500-image evaluation subset...")
    images, _labels = data_module.get_eval_subset_raw()

    print(f"Validating all {len(mr_engine.ALL_MRS)} MRs against clean Model A...\n")
    results = mr_engine.validate_all_mrs(wrapper, images)

    print(f"{'MR ID':<8}{'Name':<26}{'Violation Rate':<18}{'Cost (ms)':<12}{'Pass (<5%)'}")
    print("-" * 80)
    n_failed = 0
    for mr_id, r in results.items():
        status = "PASS" if r["passes_threshold"] else "FAIL"
        if not r["passes_threshold"]:
            n_failed += 1
        print(f"{mr_id:<8}{r['name']:<26}{r['violation_rate']*100:>6.2f}%{'':<10}"
              f"{r['cost_ms']:>8.1f}{'':<4}{status}")

    print("-" * 80)
    print(f"{n_failed} of {len(results)} MRs exceeded the 5% false-positive threshold.")
    if n_failed > 0:
        print(
            "WARNING: some MRs exceed the report's <5% false-positive target. "
            "These are reported, not silently modified -- review whether this "
            "is expected given the report's exact Appendix A parameters before "
            "proceeding to Phase 4."
        )

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUTS_DIR / "mr_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()