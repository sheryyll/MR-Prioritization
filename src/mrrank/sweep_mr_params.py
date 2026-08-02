"""
One-off calibration utility: sweeps candidate magnitude values for
borderline MRs against the real Model A checkpoint, to pick empirically
justified parameter values rather than guessing single adjustments.

Not part of the permanent pipeline -- used to generate the calibration
decisions that get hardcoded into mr_engine.py's ALL_MRS definitions.

Usage:
    python -m mrrank.sweep_mr_params
"""

from __future__ import annotations

import numpy as np

from mrrank import config, data_module, mr_engine
from mrrank.model_module import ModelWrapper
from mrrank.mr_engine import (
    _gaussian_noise, _contrast, _brightness, _saturation, MetamorphicRelation,
)


def sweep(name, param_name, candidates, build_fn, wrapper, images):
    print(f"\n{name} -- sweeping {param_name}:")
    print(f"{'Value':<12}{'Violation Rate':<18}")
    print("-" * 30)
    for val in candidates:
        mr = MetamorphicRelation(
            id="SWEEP", name=name, category="x", mr_type="x",
            magnitude=val, is_composite=False, fn=build_fn(val),
        )
        rate = mr_engine.compute_violation_rate(mr, wrapper, images)
        marker = " <-- PASS" if rate < 0.05 else ""
        print(f"{val:<12}{rate*100:>6.2f}%{'':<10}{marker}")


def main():
    wrapper = ModelWrapper.from_checkpoint(config.MODEL_A_CHECKPOINT, device="cpu")
    images, _ = data_module.get_eval_subset_raw()

    sweep(
        "MR06 Gaussian Noise (low)", "sigma",
        [0.020, 0.015, 0.012, 0.010, 0.008, 0.006],
        lambda v: (lambda img: _gaussian_noise(img, sigma_uint8=v * 255, mr_id="MR06", global_seed=config.SEED)),
        wrapper, images,
    )

    sweep(
        "MR10 Contrast Increase", "factor",
        [1.50, 1.35, 1.25, 1.20, 1.15, 1.10],
        lambda v: (lambda img: _contrast(img, factor=v)),
        wrapper, images,
    )

    sweep(
        "MR03 Rotate", "angle",
        [15, 10, 8, 6, 5],
        lambda v: (lambda img: mr_engine._rotate(img, angle=v)),
        wrapper, images,
    )

    sweep(
        "MR14 Center Crop", "fraction",
        [0.90, 0.93, 0.95, 0.97],
        lambda v: (lambda img: mr_engine._center_crop_resize(img, fraction=v)),
        wrapper, images,
    )
    sweep(
        "MR04 Rotate", "angle",
        [30, 20, 15, 12, 10],
        lambda v: (lambda img: mr_engine._rotate(img, angle=v)),
        wrapper, images,
    )

    sweep(
        "MR05 Rotate", "angle",
        [45, 30, 20, 15],
        lambda v: (lambda img: mr_engine._rotate(img, angle=v)),
        wrapper, images,
    )

    sweep(
        "MR07 Gaussian Noise (high)", "sigma",
        [0.10, 0.06, 0.04, 0.03, 0.025],
        lambda v: (lambda img: _gaussian_noise(img, sigma_uint8=v * 255, mr_id="MR07", global_seed=config.SEED)),
        wrapper, images,
    )

    sweep(
        "MR11 Contrast Decrease", "factor",
        [0.50, 0.65, 0.75, 0.80, 0.85],
        lambda v: (lambda img: _contrast(img, factor=v)),
        wrapper, images,
    )

    sweep(
        "MR12 Gaussian Blur (light)", "ksize",
        [3, 3],  # kernel size must stay odd; 3 is already the minimum meaningful blur
        lambda v: (lambda img: mr_engine._blur(img, ksize=v)),
        wrapper, images,
    )

    sweep(
        "MR13 Gaussian Blur (heavy)", "ksize",
        [7, 5, 3],
        lambda v: (lambda img: mr_engine._blur(img, ksize=v)),
        wrapper, images,
    )

    sweep(
        "MR15 Center Crop", "fraction",
        [0.80, 0.85, 0.90, 0.93],
        lambda v: (lambda img: mr_engine._center_crop_resize(img, fraction=v)),
        wrapper, images,
    )

    sweep(
        "MR16 Salt & Pepper", "amount",
        [0.003, 0.002, 0.0015, 0.001],
        lambda v: (lambda img: mr_engine._salt_pepper(img, amount=v, mr_id="MR16", global_seed=config.SEED)),
        wrapper, images,
    )

    sweep(
        "MR18 JPEG Compression", "quality",
        [50, 70, 85, 92],
        lambda v: (lambda img: mr_engine._jpeg_compress(img, quality=v)),
        wrapper, images,
    )
    sweep(
        "MR08 Brightness Increase", "delta",
        [30, 25, 20, 15],
        lambda v: (lambda img: mr_engine._brightness(img, delta=v)),
        wrapper, images,
    )


if __name__ == "__main__":
    main()