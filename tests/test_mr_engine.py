"""
Unit tests for the MR Engine. Fast, deterministic, no model/GPU needed.
"""

import numpy as np
import pytest

from mrrank import mr_engine


@pytest.fixture
def sample_image():
    rng = np.random.RandomState(0)
    return rng.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)


def test_all_20_mrs_registered():
    assert len(mr_engine.ALL_MRS) == 20
    ids = {mr.id for mr in mr_engine.ALL_MRS}
    expected = {f"MR{i:02d}" for i in range(1, 21)}
    assert ids == expected


@pytest.mark.parametrize("mr_id", [f"MR{i:02d}" for i in range(1, 21)])
def test_output_shape_and_dtype(mr_id, sample_image):
    mr = mr_engine.MR_BY_ID[mr_id]
    out = mr.transform(sample_image)
    assert out.shape == (32, 32, 3), f"{mr_id} produced shape {out.shape}"
    assert out.dtype == np.uint8, f"{mr_id} produced dtype {out.dtype}"


@pytest.mark.parametrize("mr_id", [f"MR{i:02d}" for i in range(1, 21)])
def test_determinism(mr_id, sample_image):
    """Same image transformed twice by the same MR must give identical output."""
    mr = mr_engine.MR_BY_ID[mr_id]
    out1 = mr.transform(sample_image)
    out2 = mr.transform(sample_image)
    np.testing.assert_array_equal(out1, out2)


def test_hflip_vflip_are_actual_flips(sample_image):
    hflip = mr_engine.MR_BY_ID["MR01"].transform(sample_image)
    np.testing.assert_array_equal(hflip, sample_image[:, ::-1, :])

    vflip = mr_engine.MR_BY_ID["MR02"].transform(sample_image)
    np.testing.assert_array_equal(vflip, sample_image[::-1, :, :])


def test_brightness_increase_and_decrease(sample_image):
    bright = mr_engine.MR_BY_ID["MR08"].transform(sample_image)
    dark = mr_engine.MR_BY_ID["MR09"].transform(sample_image)
    # On average, brightness-increase should raise mean pixel value and
    # brightness-decrease should lower it (allowing for clipping at 0/255).
    assert bright.astype(float).mean() >= sample_image.astype(float).mean()
    assert dark.astype(float).mean() <= sample_image.astype(float).mean()


def test_composite_mrs_flagged_correctly():
    mr19 = mr_engine.MR_BY_ID["MR19"]
    mr20 = mr_engine.MR_BY_ID["MR20"]
    assert mr19.is_composite is True
    assert mr20.is_composite is True
    assert mr19.category == "composite"
    assert mr20.category == "composite"

    non_composite_ids = [mr.id for mr in mr_engine.ALL_MRS if not mr.is_composite]
    assert len(non_composite_ids) == 18


def test_metadata_encoding_consistency():
    for mr in mr_engine.ALL_MRS:
        meta = mr.get_metadata()
        assert meta["category_encoded"] == mr_engine.CATEGORY_ENCODING[mr.category]
        assert meta["mr_type_encoded"] == mr_engine.MR_TYPE_ENCODING[mr.mr_type]


def test_gaussian_noise_actually_changes_image(sample_image):
    noisy_low = mr_engine.MR_BY_ID["MR06"].transform(sample_image)
    noisy_high = mr_engine.MR_BY_ID["MR07"].transform(sample_image)
    assert not np.array_equal(noisy_low, sample_image)
    assert not np.array_equal(noisy_high, sample_image)
    # Higher sigma should produce (on average, across many pixels) a larger
    # absolute deviation from the original than the lower sigma.
    diff_low = np.abs(noisy_low.astype(int) - sample_image.astype(int)).mean()
    diff_high = np.abs(noisy_high.astype(int) - sample_image.astype(int)).mean()
    assert diff_high > diff_low


def test_crop_resize_returns_original_size(sample_image):
    crop90 = mr_engine.MR_BY_ID["MR14"].transform(sample_image)
    crop80 = mr_engine.MR_BY_ID["MR15"].transform(sample_image)
    assert crop90.shape == (32, 32, 3)
    assert crop80.shape == (32, 32, 3)


def test_different_images_get_different_noise(sample_image):
    """Sanity check that the seeded RNG is genuinely image-dependent, not a
    fixed pattern applied to every image regardless of content."""
    rng = np.random.RandomState(1)
    other_image = rng.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)

    noise_a = mr_engine.MR_BY_ID["MR06"].transform(sample_image)
    noise_b = mr_engine.MR_BY_ID["MR06"].transform(other_image)

    diff_a = noise_a.astype(int) - sample_image.astype(int)
    diff_b = noise_b.astype(int) - other_image.astype(int)
    assert not np.array_equal(diff_a, diff_b)