"""
MR Engine for MR-Rank.

Implements all 20 Metamorphic Relations (report Appendix A) as OpenCV-based
image transformations operating on raw uint8 pixel arrays, shape (32, 32, 3),
RGB, range [0, 255] -- matching what data_module.get_eval_subset_raw()
returns.

Also provides:
    - compute_violation_rate(): compares model predictions before/after an
      MR's transformation (used both for Phase 3's validation step and
      Phase 5's Kill Matrix construction).
    - measure_mr_cost_ms(): wall-clock cost tracking, feeding Phase 6's
      NormCost term.
    - MR metadata (mr_type, magnitude, category, is_composite) needed by
      Phase 7's meta-classifier feature table.

DESIGN NOTE on stochastic MRs (Gaussian Noise, Salt & Pepper): each of
these must be DETERMINISTIC given the same input image, per the report's
Reproducibility Constraint (Section 3.3) -- the same (MR, image) pair must
always transform identically, since Kill Matrix entries need to be
reproducible across runs. This is achieved by seeding a LOCAL RandomState
from a hash of (image bytes + MR id + global config.SEED), rather than
using any shared/global random state.

DESIGN NOTE on Gaussian noise scale: report Appendix A specifies
sigma=0.02 (MR06) and sigma=0.1 (MR07) without stating a scale convention.
This implementation treats these as normalized [0,1]-scale sigmas, i.e.
sigma_uint8 = sigma * 255 (~5.1 and ~25.5 respectively) -- the common
convention in ML image-augmentation literature. If experimental results
suggest this is too aggressive/weak (i.e. violates the <5% false-positive
target even after this choice), revisit this scale convention rather than
changing Appendix A's stated sigma values.
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
# Encoding tables (used later by Phase 7's meta-classifier feature table)
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
}


# ---------------------------------------------------------------------------
# Deterministic per-image RNG for stochastic transforms
# ---------------------------------------------------------------------------
def _seeded_rng(image: np.ndarray, salt: str, global_seed: int) -> np.random.RandomState:
    """
    Build a local RandomState seeded deterministically from the image's
    own content, an MR-specific salt string, and the project's global
    seed. Guarantees transform(image) is a pure, reproducible function of
    (image, mr_id) -- required for Kill Matrix reproducibility.
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
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _center_crop_resize(img: np.ndarray, fraction: float) -> np.ndarray:
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
    category: str       # geometric | photometric | noise | composite
    mr_type: str         # key into MR_TYPE_ENCODING
    magnitude: float     # numeric parameter (0 if not applicable)
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
    """Build the canonical list of all 20 MRs, per report Appendix A."""
    return [
        MetamorphicRelation("MR01", "Horizontal Flip", "geometric", "flip", 0, False, _hflip),
        MetamorphicRelation("MR02", "Vertical Flip", "geometric", "flip", 0, False, _vflip),
        MetamorphicRelation("MR03", "Rotate 15deg", "geometric", "rotate", 15, False,
                             partial(_rotate, angle=15)),
        MetamorphicRelation("MR04", "Rotate 30deg", "geometric", "rotate", 30, False,
                             partial(_rotate, angle=30)),
        MetamorphicRelation("MR05", "Rotate 45deg", "geometric", "rotate", 45, False,
                             partial(_rotate, angle=45)),
        MetamorphicRelation("MR06", "Gaussian Noise (low)", "noise", "noise", 0.02, False,
                             partial(_gaussian_noise, sigma_uint8=0.02 * 255, mr_id="MR06",
                                     global_seed=global_seed)),
        MetamorphicRelation("MR07", "Gaussian Noise (high)", "noise", "noise", 0.1, False,
                             partial(_gaussian_noise, sigma_uint8=0.1 * 255, mr_id="MR07",
                                     global_seed=global_seed)),
        MetamorphicRelation("MR08", "Brightness Increase", "photometric", "brightness", 30, False,
                             partial(_brightness, delta=30)),
        MetamorphicRelation("MR09", "Brightness Decrease", "photometric", "brightness", -30, False,
                             partial(_brightness, delta=-30)),
        MetamorphicRelation("MR10", "Contrast Increase", "photometric", "contrast", 1.5, False,
                             partial(_contrast, factor=1.5)),
        MetamorphicRelation("MR11", "Contrast Decrease", "photometric", "contrast", 0.5, False,
                             partial(_contrast, factor=0.5)),
        MetamorphicRelation("MR12", "Gaussian Blur (light)", "noise", "blur", 3, False,
                             partial(_blur, ksize=3)),
        MetamorphicRelation("MR13", "Gaussian Blur (heavy)", "noise", "blur", 7, False,
                             partial(_blur, ksize=7)),
        MetamorphicRelation("MR14", "Center Crop 90%", "geometric", "crop", 0.9, False,
                             partial(_center_crop_resize, fraction=0.9)),
        MetamorphicRelation("MR15", "Center Crop 80%", "geometric", "crop", 0.8, False,
                             partial(_center_crop_resize, fraction=0.8)),
        MetamorphicRelation("MR16", "Salt & Pepper Noise", "noise", "salt_pepper", 0.02, False,
                             partial(_salt_pepper, amount=0.02, mr_id="MR16", global_seed=global_seed)),
        MetamorphicRelation("MR17", "Saturation Change", "photometric", "saturation", 1.5, False,
                             partial(_saturation, factor=1.5)),
        MetamorphicRelation("MR18", "JPEG Compression", "photometric", "jpeg", 50, False,
                             partial(_jpeg_compress, quality=50)),
        MetamorphicRelation(
            "MR19", "Flip + Noise", "composite", "composite", 0, True,
            lambda img: _gaussian_noise(_hflip(img), sigma_uint8=0.02 * 255, mr_id="MR19",
                                         global_seed=global_seed),
        ),
        MetamorphicRelation(
            "MR20", "Rotate + Blur", "composite", "composite", 0, True,
            lambda img: _blur(_rotate(img, 15), ksize=3),
        ),
    ]


# Canonical registry, built once at import time using the project's global seed.
from mrrank import config as _config  # noqa: E402 (deliberate: avoid circular import issues)

ALL_MRS: list[MetamorphicRelation] = _build_all_mrs(_config.SEED)
MR_BY_ID: dict[str, MetamorphicRelation] = {mr.id: mr for mr in ALL_MRS}


# ---------------------------------------------------------------------------
# Violation rate + cost measurement (used in Phase 3 validation and Phase 5
# Kill Matrix construction)
# ---------------------------------------------------------------------------
def compute_violation_rate(mr: MetamorphicRelation, model_wrapper, images: np.ndarray) -> float:
    """
    Fraction of images where the model's prediction changes after applying
    `mr`'s transformation. images: (N, 32, 32, 3) uint8 array.
    """
    from mrrank import data_module

    orig_batch = data_module.normalize_batch(images)
    orig_preds = model_wrapper.predict_batch(orig_batch)

    transformed = np.stack([mr.transform(img) for img in images])
    trans_batch = data_module.normalize_batch(transformed)
    trans_preds = model_wrapper.predict_batch(trans_batch)

    violations = (orig_preds != trans_preds).sum().item()
    return violations / len(images)


def measure_mr_cost_ms(mr: MetamorphicRelation, model_wrapper, images: np.ndarray) -> float:
    """
    Wall-clock milliseconds to transform + run model inference on all
    given images. Feeds Phase 6's NormCost(MR_i) term.
    """
    from mrrank import data_module

    start = time.time()
    transformed = np.stack([mr.transform(img) for img in images])
    batch = data_module.normalize_batch(transformed)
    model_wrapper.predict_batch(batch)
    return (time.time() - start) * 1000


def validate_all_mrs(model_wrapper, images: np.ndarray, threshold: float = 0.05) -> dict:
    """
    Run every MR in ALL_MRS against the given (clean) model, computing
    violation rate and execution cost for each. Returns a dict keyed by
    MR id.
    """
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