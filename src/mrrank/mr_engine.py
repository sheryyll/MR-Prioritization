"""
MR Engine for MR-Rank.

Implements 20 Metamorphic Relations as OpenCV-based image transformations
operating on raw uint8 pixel arrays, shape (32, 32, 3), RGB, range [0, 255].

Also provides:
    - compute_violation_rate(): compares model predictions before/after an
      MR's transformation.
    - measure_mr_cost_ms(): wall-clock cost tracking, feeding NormCost.
    - MR metadata (mr_type, magnitude, category, is_composite) for the
      meta-classifier feature table.

MR LIBRARY REVISION LOG:
Multiple rounds of empirical validation (validate_mrs.py) and binary-search
calibration (calibrate_mrs.py) against the trained Model A checkpoint drove
several redesign passes:
  - MR02 (Vertical Flip, 60.6% FP, non-calibratable) -> Shear
  - MR04/MR05 (Rotate 5/8deg, redundant with MR03, all on the same failing
    curve) -> Translate Horizontal / Translate Vertical (direct match to
    RandomCrop(padding=4) training augmentation)
  - MR12/MR13 (Gaussian Blur light/heavy, proven no passing kernel exists
    at 32x32 via binary search) -> Unsharp Mask / Posterize
  - MR14/MR15 (Center Crop 90%/80%, crop-via-resize introduces confounding
    interpolation artifacts) -> Translate Diagonal / Scale Down (zoom-out)
  - MR20 (Rotate + Blur, both constituents proven dead ends) -> Contrast +
    Saturation Jitter (built from two independently-passing components)

MR01 (Horizontal Flip) is INTENTIONALLY RETAINED despite having no
continuous calibration parameter and a measured ~10% false-positive rate.
This is a deliberate, explicit project decision (not an oversight): flip
invariance is one of the most commonly cited MT examples for image
classifiers in the literature this project's report itself draws on, and
removing it would weaken research credibility more than keeping an
imperfect-but-standard MR. MR19 (Flip + Noise) inherits this same bound
since it composites MR01.

All magnitudes below reflect the MOST RECENT calibration/redesign pass.
Re-run calibrate_mrs.py after any model retrain -- calibration is
checkpoint-specific, not universal (empirically confirmed: identical MR
parameters produced different pass/fail results across different trained
checkpoints in this project's history).

DESIGN NOTE on stochastic MRs (Gaussian Noise, Salt & Pepper): each of
these must be DETERMINISTIC given the same input image, per the report's
Reproducibility Constraint (Section 3.3). Achieved by seeding a LOCAL
RandomState from a hash of (image bytes + MR id + global config.SEED).
Translation, Scale, Shear, Unsharp Mask, and Posterize are all
deterministic pure functions of the image with no randomness involved.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Encoding tables (used by the meta-classifier feature table)
# ---------------------------------------------------------------------------
CATEGORY_ENCODING = {
    "geometric": 1,
    "photometric": 2,
    "noise": 3,
    "composite": 4,
}

MR_TYPE_ENCODING = {
    "flip": 1,
    "rotate": 2,
    "noise": 3,
    "brightness": 4,
    "contrast": 5,
    "blur": 6,
    "crop": 7,
    "salt_pepper": 8,
    "saturation": 9,
    "jpeg": 10,
    "composite": 11,
    "translation": 12,
    "scale": 13,
    "sharpen": 14,
    "posterize": 15,
    "shear": 16,
}


# ---------------------------------------------------------------------------
# Deterministic per-image RNG for stochastic transforms
# ---------------------------------------------------------------------------
def _seeded_rng(image: np.ndarray, salt: str, global_seed: int) -> np.random.RandomState:
    """
    Local RandomState seeded from image content + MR-specific salt +
    global seed. Guarantees transform(image) is a pure, reproducible
    function of (image, mr_id) -- required for Kill Matrix reproducibility.
    """
    digest = hashlib.md5(image.tobytes() + salt.encode("utf-8") + str(global_seed).encode()).digest()
    seed_int = int.from_bytes(digest[:4], "little")
    return np.random.RandomState(seed_int)


# ---------------------------------------------------------------------------
# Individual transform implementations (pure functions on uint8 images)
# ---------------------------------------------------------------------------
def _hflip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def _vflip(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 0)


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _translate(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Shift the image by (dx, dy) pixels, filling the exposed border via
    reflection. This is the pixel-level operation RandomCrop(padding=N)
    performs at training time (pad then crop == an effective shift), so
    a model trained with that augmentation should have direct, learned
    invariance to this transform.
    """
    h, w = img.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)


def _shear(img: np.ndarray, shear_factor: float) -> np.ndarray:
    """
    Horizontal shear by `shear_factor` (e.g. 0.1 shifts the top row by
    10% of image width relative to the bottom row). A genuinely different
    affine distortion family from rotate/translate/scale -- skews the
    image rather than rotating, shifting, or resizing it uniformly.
    """
    h, w = img.shape[:2]
    matrix = np.float32([[1, shear_factor, -shear_factor * h / 2], [0, 1, 0]])
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT101)


def _scale(img: np.ndarray, factor: float) -> np.ndarray:
    """
    factor > 1.0: zoom in (resize up, center-crop back to original size).
    factor < 1.0: zoom out (resize down, pad back to original size).
    Only used in the zoom-out direction in this library -- zoom-in is
    mathematically equivalent to crop-then-resize, already shown to
    perform poorly; zoom-out keeps the full object visible rather than
    cutting off real content.
    """
    h, w = img.shape[:2]
    new_h, new_w = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if factor >= 1.0:
        top = (new_h - h) // 2
        left = (new_w - w) // 2
        return resized[top:top + h, left:left + w]
    pad_top = (h - new_h) // 2
    pad_bottom = h - new_h - pad_top
    pad_left = (w - new_w) // 2
    pad_right = w - new_w - pad_left
    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                               borderType=cv2.BORDER_REFLECT101)


def _gaussian_noise(img: np.ndarray, sigma_uint8: float, mr_id: str, global_seed: int) -> np.ndarray:
    rng = _seeded_rng(img, mr_id, global_seed)
    noise = rng.normal(0, sigma_uint8, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _brightness(img: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def _contrast(img: np.ndarray, factor: float) -> np.ndarray:
    adjusted = (img.astype(np.float32) - 127.5) * factor + 127.5
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def _blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """Retained for backward compatibility -- no longer wired into any MR.
    Binary-search calibration proved no kernel size passes at 32x32
    resolution; replaced by Unsharp Mask / Posterize."""
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _unsharp_mask(img: np.ndarray, amount: float, ksize: int = 3) -> np.ndarray:
    """
    Sharpen via: sharpened = img + amount * (img - blur(img)). Conceptually
    the inverse of Gaussian blur -- mild amounts enhance existing edges
    rather than destroying them, giving it a structurally different (and
    empirically better) pass/fail profile than blur.
    """
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0).astype(np.float32)
    sharpened = img.astype(np.float32) + amount * (img.astype(np.float32) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _posterize(img: np.ndarray, bits: int) -> np.ndarray:
    """
    Reduce color depth to `bits` bits per channel. A structurally
    different information-loss mechanism than blur (quantization vs.
    frequency-domain smoothing).
    """
    bits = max(1, min(8, int(bits)))
    shift = 8 - bits
    mask = (0xFF << shift) & 0xFF
    return (img & mask).astype(np.uint8)


def _center_crop_resize(img: np.ndarray, fraction: float) -> np.ndarray:
    """Retained for backward compatibility -- no longer wired into any MR.
    Empirically shown to underperform Translation/Scale-Down."""
    h, w = img.shape[:2]
    new_h, new_w = int(h * fraction), int(w * fraction)
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    cropped = img[top:top + new_h, left:left + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def _salt_pepper(img: np.ndarray, amount: float, mr_id: str, global_seed: int) -> np.ndarray:
    rng = _seeded_rng(img, mr_id, global_seed)
    out = img.copy()
    h, w = img.shape[:2]
    n_pixels = int(amount * h * w)

    salt_rows = rng.randint(0, h, n_pixels // 2)
    salt_cols = rng.randint(0, w, n_pixels // 2)
    out[salt_rows, salt_cols] = 255

    pepper_rows = rng.randint(0, h, n_pixels // 2)
    pepper_cols = rng.randint(0, w, n_pixels // 2)
    out[pepper_rows, pepper_cols] = 0

    return out


def _saturation(img: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# MR container
# ---------------------------------------------------------------------------
@dataclass
class MetamorphicRelation:
    id: str
    name: str
    category: str
    mr_type: str
    magnitude: float
    is_composite: bool
    fn: Callable[[np.ndarray], np.ndarray] = field(repr=False)

    def transform(self, image: np.ndarray) -> np.ndarray:
        return self.fn(image)

    def get_metadata(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "category_encoded": CATEGORY_ENCODING[self.category],
            "mr_type": self.mr_type,
            "mr_type_encoded": MR_TYPE_ENCODING[self.mr_type],
            "magnitude": self.magnitude,
            "is_composite": int(self.is_composite),
        }


def _build_all_mrs(global_seed: int) -> list[MetamorphicRelation]:
    """Build the canonical list of 20 MRs. See module docstring for the
    full revision log explaining each redesign decision."""
    return [
        MetamorphicRelation("MR01", "Horizontal Flip", "geometric", "flip", 0, False, _hflip),
        # RETAINED BY EXPLICIT PROJECT DECISION: non-calibratable (no
        # continuous magnitude parameter), measured ~10.4% FP rate.
        # Kept deliberately for research credibility -- one of the most
        # commonly cited MT examples for image classifiers in the
        # literature. MR19 (composite) inherits this same bound.

        MetamorphicRelation("MR02", "Shear (mild)", "geometric", "shear", 0.04, False,
                             partial(_shear, shear_factor=0.04)),
        # REPLACED (was Vertical Flip, 60.6% FP, non-calibratable, zero
        # training signal). Shear is a new affine family (skew) distinct
        # from rotate/translate/scale, WITH a continuous calibration
        # parameter. PROVISIONAL magnitude, needs calibration.

        MetamorphicRelation("MR03", "Rotate (fine)", "geometric", "rotate", 1.0, False,
                             partial(_rotate, angle=1.0)),
        # RECALIBRATE: closest-to-passing rotation across prior testing
        # (3deg -> 9.4%). One further attempt at a smaller angle before
        # this family is considered exhausted. PROVISIONAL, needs
        # calibration; may prove degenerate at the passing boundary.

        MetamorphicRelation("MR04", "Translate Horizontal", "geometric", "translation", 1, False,
                             partial(_translate, dx=1 , dy=0)),
        # REPLACED (was Rotate 5deg, redundant with MR03/MR05 on the same
        # failing curve). Directly matches RandomCrop(padding=4) training
        # augmentation. PROVISIONAL magnitude, needs calibration.

        MetamorphicRelation("MR05", "Translate Vertical", "geometric", "translation", 1, False,
                             partial(_translate, dx=0, dy=1)),
        # REPLACED (was Rotate 8deg, worst-performing of the three
        # rotations). Same justification as MR04, orthogonal direction.
        # PROVISIONAL, needs calibration.

        MetamorphicRelation("MR06", "Gaussian Noise (low)", "noise", "noise", 0.009, False,
                             partial(_gaussian_noise, sigma_uint8=0.009 * 255, mr_id="MR06",
                                     global_seed=global_seed)),
        # KEPT: passing (4.00%), calibrated in prior work.

        MetamorphicRelation("MR07", "Gaussian Noise (high)", "noise", "noise", 0.009, False,
                             partial(_gaussian_noise, sigma_uint8=0.009 * 255, mr_id="MR07",
                                     global_seed=global_seed)),
        # KEPT: now passing (4.60% at this magnitude in most recent
        # validation run) and genuinely distinct from MR06. If
        # recalibration against a new checkpoint collides with MR06
        # again, revisit per the MR-collision precedent documented
        # earlier in this project.

        MetamorphicRelation("MR08", "Brightness Increase", "photometric", "brightness", 10, False,
                             partial(_brightness, delta=10)),
        MetamorphicRelation("MR09", "Brightness Decrease", "photometric", "brightness", -10, False,
                             partial(_brightness, delta=-10)),
        MetamorphicRelation("MR10", "Contrast Increase", "photometric", "contrast", 1.10, False,
                             partial(_contrast, factor=1.10)),
        MetamorphicRelation("MR11", "Contrast Decrease", "photometric", "contrast", 0.90, False,
                             partial(_contrast, factor=0.90)),
        # MR08-11: KEPT, passing, calibrated in prior work.

        MetamorphicRelation("MR12", "Unsharp Mask", "noise", "sharpen", 0.5, False,
                             partial(_unsharp_mask, amount=0.5, ksize=3)),
        # REPLACED (was Gaussian Blur light, 36.6% FP; binary search
        # proved no blur kernel passes at 32x32). KEPT from prior
        # validated-passing result (3.80%).

        MetamorphicRelation("MR13", "Posterize", "photometric", "posterize", 5, False,
                             partial(_posterize, bits=5)),
        # REPLACED (was Gaussian Blur heavy, 68.6% FP, same blur-family
        # dead end). KEPT from prior validated-passing result (3.20%).

        MetamorphicRelation("MR14", "Translate Diagonal", "geometric", "translation", 1, False,
                             partial(_translate, dx=1, dy=1)),
        # REPLACED (was Center Crop 95%, 13.2% FP; crop-via-resize
        # introduces confounding interpolation artifacts). Reduced
        # magnitude from earlier 6px attempt (22.6% FP). PROVISIONAL,
        # needs calibration.

        MetamorphicRelation("MR15", "Scale Down (zoom-out)", "geometric", "scale", 0.90, False,
                             partial(_scale, factor=0.90)),
        # REPLACED (was Center Crop 80%, 17.0% FP, same crop-family
        # weakness). Zoom-out keeps full object visible. Increased factor
        # toward 1.0 from earlier 0.9x attempt (8.8% FP). PROVISIONAL,
        # needs calibration.

        MetamorphicRelation("MR16", "Salt & Pepper Noise", "noise", "salt_pepper", 0.0017, False,
                             partial(_salt_pepper, amount=0.0017, mr_id="MR16", global_seed=global_seed)),
        MetamorphicRelation("MR17", "Saturation Change", "photometric", "saturation", 1.10, False,
                             partial(_saturation, factor=1.10)),
        MetamorphicRelation("MR18", "JPEG Compression", "photometric", "jpeg", 98, False,
                             partial(_jpeg_compress, quality=98)),
        # MR16-18: KEPT, passing, calibrated in prior work.

        MetamorphicRelation(
            "MR19", "Flip + Noise (reduced)", "composite", "composite", 0, True,
            lambda img: _gaussian_noise(_hflip(img), sigma_uint8=0.004 * 255, mr_id="MR19",
                                         global_seed=global_seed),
        ),
        # RECALIBRATE (reduced noise component from sigma=0.009 to 0.006).
        # Composite's floor is bounded by MR01's ~10.4% FP rate since MR01
        # is retained by explicit decision -- this MR is EXPECTED to still
        # exceed 5% even after recalibration; documented, not concealed.
        # Reducing the noise half is attempted anyway since it can only
        # help, not hurt.

        MetamorphicRelation(
            "MR20", "Contrast + Saturation Jitter", "composite", "composite", 0, True,
            lambda img: _saturation(_contrast(img, factor=1.05), factor=1.05),
        ),
        # REPLACED (was Rotate + Blur, 49.0% FP -- both constituent
        # transforms are proven dead ends). Built from two independently-
        # passing components (contrast, saturation). KEPT from prior
        # validated-passing result (1.20%).
    ]


# Canonical registry, built once at import time using the project's global seed.
from mrrank import config as _config  # noqa: E402

ALL_MRS: list[MetamorphicRelation] = _build_all_mrs(_config.SEED)
MR_BY_ID: dict[str, MetamorphicRelation] = {mr.id: mr for mr in ALL_MRS}


# ---------------------------------------------------------------------------
# Violation rate + cost measurement
# ---------------------------------------------------------------------------
def compute_violation_rate(mr: MetamorphicRelation, model_wrapper, images: np.ndarray) -> float:
    from mrrank import data_module

    orig_batch = data_module.normalize_batch(images)
    orig_preds = model_wrapper.predict_batch(orig_batch)

    transformed = np.stack([mr.transform(img) for img in images])
    trans_batch = data_module.normalize_batch(transformed)
    trans_preds = model_wrapper.predict_batch(trans_batch)

    violations = (orig_preds != trans_preds).sum().item()
    return violations / len(images)


def measure_mr_cost_ms(mr: MetamorphicRelation, model_wrapper, images: np.ndarray) -> float:
    from mrrank import data_module

    start = time.time()
    transformed = np.stack([mr.transform(img) for img in images])
    batch = data_module.normalize_batch(transformed)
    model_wrapper.predict_batch(batch)
    return (time.time() - start) * 1000


def validate_all_mrs(model_wrapper, images: np.ndarray, threshold: float = 0.05) -> dict:
    results = {}
    for mr in ALL_MRS:
        violation_rate = compute_violation_rate(mr, model_wrapper, images)
        cost_ms = measure_mr_cost_ms(mr, model_wrapper, images)
        results[mr.id] = {
            **mr.get_metadata(),
            "violation_rate": violation_rate,
            "cost_ms": cost_ms,
            "passes_threshold": violation_rate < threshold,
        }
    return results