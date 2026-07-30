"""
Unit tests for the Data Module.

Run with:  pytest tests/test_data_module.py -v

Note: the first run downloads CIFAR-10 (~170 MB) to ./data/ if not already
present -- this can take a minute or two depending on connection speed.
Subsequent runs reuse the cached copy and run in a few seconds.
"""

import numpy as np

from mrrank import config
from mrrank import data_module


def test_eval_subset_size():
    images, labels = data_module.get_eval_subset_raw()
    assert images.shape == (config.EVAL_SUBSET_SIZE, 32, 32, 3)
    assert labels.shape == (config.EVAL_SUBSET_SIZE,)
    assert images.dtype == np.uint8


def test_eval_subset_stratification():
    """Exactly 50 images per class, all 10 classes represented."""
    _, labels = data_module.get_eval_subset_raw()
    counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    assert len(counts) == config.NUM_CLASSES
    assert all(c == config.EVAL_IMAGES_PER_CLASS for c in counts)


def test_eval_subset_no_duplicates():
    indices = data_module.get_eval_subset_indices()
    assert len(indices) == len(set(indices.tolist()))


def test_eval_subset_reproducibility():
    """Calling twice (without force_rebuild) must return identical indices."""
    indices_1 = data_module.get_eval_subset_indices()
    indices_2 = data_module.get_eval_subset_indices()
    np.testing.assert_array_equal(indices_1, indices_2)


def test_eval_subset_rebuild_is_deterministic():
    """Even a forced rebuild must give the same indices, since the seed is fixed."""
    indices_1 = data_module.get_eval_subset_indices(force_rebuild=True)
    indices_2 = data_module.get_eval_subset_indices(force_rebuild=True)
    np.testing.assert_array_equal(indices_1, indices_2)


def test_normalize_image_shape_and_type():
    images, _ = data_module.get_eval_subset_raw()
    tensor = data_module.normalize_image(images[0])
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype.is_floating_point


def test_normalize_batch():
    images, _ = data_module.get_eval_subset_raw()
    batch = images[:8]
    tensor_batch = data_module.normalize_batch(batch)
    assert tensor_batch.shape == (8, 3, 32, 32)


def test_train_test_loader_shapes():
    train_loader = data_module.get_train_loader(batch_size=16)
    test_loader = data_module.get_test_loader(batch_size=16)

    train_batch, train_labels = next(iter(train_loader))
    test_batch, test_labels = next(iter(test_loader))

    assert train_batch.shape == (16, 3, 32, 32)
    assert test_batch.shape == (16, 3, 32, 32)
    assert train_labels.shape == (16,)
    assert test_labels.shape == (16,)


def test_full_dataset_sizes():
    train_dataset = data_module.load_raw_cifar10(train=True)
    test_dataset = data_module.load_raw_cifar10(train=False)
    assert len(train_dataset) == 50_000
    assert len(test_dataset) == 10_000