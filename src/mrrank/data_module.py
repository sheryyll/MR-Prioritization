"""
Data Module for MR-Rank.

Responsibilities (per Phase-1 report, Section 4.3 "Data Module"):
    - Acquire and cache the CIFAR-10 dataset via torchvision.
    - Apply per-channel mean/std normalization matching the report's spec.
    - Extract and persist a fixed, stratified 500-image evaluation subset
      (50 images per class), drawn ONLY from the CIFAR-10 TEST split, so
      no training image ever appears in the evaluation subset.
    - Expose both:
        (a) normalized tensors, for model training/inference (Phase 2), and
        (b) raw uint8 pixel-space images, for the MR Engine (Phase 3), which
            applies transformations (flip, blur, noise, ...) via OpenCV and
            needs plain pixel arrays, not normalized tensors.

TRAINING AUGMENTATION (added after Phase 3 validation revealed Model A was
catastrophically sensitive to nearly every MR -- including a plain
horizontal flip -- because the original training pipeline used ZERO
augmentation. Standard practice for CIFAR-10 ResNets (used in the original
ResNet paper's own CIFAR experiments) is RandomCrop(32, padding=4) +
RandomHorizontalFlip() during training. This teaches the model basic
positional/flip invariance, which is a prerequisite for the report's MR
validation step (<5% false-positive rate on a "clean, well-trained
model") to be meaningful at all. This augmentation is applied ONLY to the
training loader -- the eval/test pipeline and the MR Engine's
normalize_image/normalize_batch functions remain deterministic and
augmentation-free, since introducing randomness there would break Kill
Matrix reproducibility.

Note on DataLoader num_workers: defaults to 0 in this module. On Windows,
num_workers > 0 spawns worker subprocesses via multiprocessing, which can
hang or error when invoked from contexts without a proper
`if __name__ == "__main__":` guard (e.g. some IDE/test runners). This is
safe-but-slower locally; when training on Kaggle (Linux) in Phase 2, pass
a higher num_workers explicitly to speed up data loading.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mrrank import config


def _build_transform() -> transforms.Compose:
    """
    Deterministic ToTensor + normalize pipeline (report Sec. 4.3). Used for
    the TEST split and for normalize_image/normalize_batch (the MR Engine
    bridge) -- must stay augmentation-free so Kill Matrix entries remain
    reproducible.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.CIFAR10_MEAN, std=config.CIFAR10_STD),
    ])


def _build_train_transform() -> transforms.Compose:
    """
    Training-only augmentation pipeline: standard CIFAR-10 ResNet recipe
    (RandomCrop with reflection padding + RandomHorizontalFlip), followed
    by the same ToTensor + normalize as the eval pipeline. This is what
    teaches the model basic invariances -- without it, the model has no
    pressure to generalize across flips/shifts and becomes extremely
    brittle to the MR Engine's transformations (observed empirically in
    Phase 3: 19/20 MRs exceeded the 5% false-positive threshold, including
    a plain horizontal flip at 16.2%, before this fix was applied).
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.CIFAR10_MEAN, std=config.CIFAR10_STD),
    ])


def load_raw_cifar10(train: bool) -> datasets.CIFAR10:
    """
    Load CIFAR-10 with NO transform applied.

    torchvision's CIFAR10 object stores images internally as a numpy array
    at `.data` (shape (N, 32, 32, 3), dtype uint8, RGB) and labels as a
    python list at `.targets`. We use this raw form for two purposes:
      1. Building the stratified evaluation subset (below).
      2. Feeding raw pixel-space images into the MR Engine (Phase 3).

    Downloads to config.DATA_DIR on first call (~170 MB); reuses the local
    cache on every subsequent call.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return datasets.CIFAR10(
        root=str(config.DATA_DIR),
        train=train,
        download=True,
        transform=None,
    )


def load_normalized_cifar10(train: bool, augment: bool = False) -> datasets.CIFAR10:
    """
    Same dataset, but with a transform applied for training/inference.

    augment=True applies the training augmentation pipeline (RandomCrop +
    RandomHorizontalFlip + normalize) -- only meaningful/intended when
    train=True. augment=False (default) applies the deterministic
    eval-only pipeline regardless of the train flag.
    """
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    transform = _build_train_transform() if augment else _build_transform()
    return datasets.CIFAR10(
        root=str(config.DATA_DIR),
        train=train,
        download=True,
        transform=transform,
    )


def get_train_loader(batch_size: int = 128, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """
    DataLoader over the full 50,000-image CIFAR-10 training split, WITH
    augmentation (RandomCrop + RandomHorizontalFlip) applied on top of
    normalization -- see module docstring for rationale.
    """
    dataset = load_normalized_cifar10(train=True, augment=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_test_loader(batch_size: int = 128, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    """
    DataLoader over the full 10,000-image CIFAR-10 test split, normalized
    only -- NO augmentation (test-time evaluation must be deterministic).
    """
    dataset = load_normalized_cifar10(train=False, augment=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def create_stratified_indices(
    targets,
    n_per_class: int,
    n_classes: int,
    seed: int,
) -> np.ndarray:
    """
    Deterministically select `n_per_class` indices for each of `n_classes`
    classes.

    Uses a LOCAL numpy RandomState seeded with `seed` (not the global numpy
    random state), so this function's output never depends on what other
    random calls happened earlier in the program -- critical for the
    report's Reproducibility Constraint (Section 3.3).
    """
    targets = np.asarray(targets)
    rng = np.random.RandomState(seed)
    selected = []
    for class_id in range(n_classes):
        class_indices = np.where(targets == class_id)[0]
        if len(class_indices) < n_per_class:
            raise ValueError(
                f"Class {class_id} has only {len(class_indices)} samples, "
                f"need {n_per_class}."
            )
        chosen = rng.choice(class_indices, size=n_per_class, replace=False)
        selected.append(chosen)
    indices = np.concatenate(selected)
    rng.shuffle(indices)  # avoid indices being grouped by class in file order
    return indices.astype(np.int64)


def get_eval_subset_indices(force_rebuild: bool = False) -> np.ndarray:
    """
    Return the fixed 500-image stratified evaluation subset indices
    (indices into the CIFAR-10 TEST split -- report Section 3.3 explicitly
    forbids any training image from appearing here).

    Persisted to outputs/eval_subset_indices.npy so the exact same 500
    images are used across every MR validation, mutant evaluation, and
    Kill Matrix entry for the rest of the project. If the file already
    exists, it is loaded rather than regenerated (unless force_rebuild=True).
    Even a forced rebuild is deterministic, since it re-derives from the
    fixed SEED.
    """
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    index_path = config.OUTPUTS_DIR / "eval_subset_indices.npy"

    if index_path.exists() and not force_rebuild:
        return np.load(index_path)

    test_dataset = load_raw_cifar10(train=False)
    indices = create_stratified_indices(
        targets=test_dataset.targets,
        n_per_class=config.EVAL_IMAGES_PER_CLASS,
        n_classes=config.NUM_CLASSES,
        seed=config.SEED,
    )
    np.save(index_path, indices)
    return indices


def get_eval_subset_raw(force_rebuild: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the fixed evaluation subset as raw pixel-space data:
        images: uint8 numpy array, shape (500, 32, 32, 3), RGB, range [0, 255]
        labels: int64 numpy array, shape (500,), values in [0, 9]

    This is the primary interface the MR Engine (Phase 3) will use -- MRs
    are implemented with OpenCV and operate directly on uint8 pixel arrays,
    not normalized tensors.
    """
    indices = get_eval_subset_indices(force_rebuild=force_rebuild)
    test_dataset = load_raw_cifar10(train=False)
    images = test_dataset.data[indices]              # (500, 32, 32, 3) uint8
    labels = np.asarray(test_dataset.targets)[indices]  # (500,)
    return images, labels


def normalize_image(image_uint8: np.ndarray) -> torch.Tensor:
    """
    Convert a single raw uint8 image (H, W, 3) into a normalized CHW float
    tensor ready for model inference. This is the bridge between the MR
    Engine's OpenCV output (Phase 3) and the Model Module's forward pass
    (Phase 2). Deliberately uses the deterministic (non-augmented)
    transform -- augmentation here would corrupt Kill Matrix reproducibility.
    """
    transform = _build_transform()
    return transform(image_uint8)


def normalize_batch(images_uint8: np.ndarray) -> torch.Tensor:
    """
    Convert a batch of raw uint8 images (N, H, W, 3) into a normalized
    (N, C, H, W) float tensor batch, ready for model inference.
    """
    tensors = [normalize_image(img) for img in images_uint8]
    return torch.stack(tensors, dim=0)