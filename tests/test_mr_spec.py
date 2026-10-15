"""Torch-free: spec-built MRs must reproduce the canonical MRs bit-for-bit."""
import numpy as np
import pytest

from mrrank import mr_engine
from mrrank.mr_spec import MRSpecError, build_mr_from_spec, build_mrs_from_specs

# Spec equivalents of every canonical MR (same ids so seeded noise matches).
CANONICAL_SPECS = [
    {"id": "MR01", "type": "flip"},
    {"id": "MR02", "type": "shear", "params": {"shear_factor": 0.04}},
    {"id": "MR03", "type": "rotate", "params": {"angle": 1.0}},
    {"id": "MR04", "type": "translation", "params": {"dx": 1, "dy": 0}},
    {"id": "MR05", "type": "translation", "params": {"dx": 0, "dy": 1}},
    {"id": "MR06", "type": "noise", "params": {"sigma": 0.009}},
    {"id": "MR07", "type": "noise", "params": {"sigma": 0.009}},
    {"id": "MR08", "type": "brightness", "params": {"delta": 10}},
    {"id": "MR09", "type": "brightness", "params": {"delta": -10}},
    {"id": "MR10", "type": "contrast", "params": {"factor": 1.10}},
    {"id": "MR11", "type": "contrast", "params": {"factor": 0.90}},
    {"id": "MR12", "type": "sharpen", "params": {"amount": 0.5, "ksize": 3}},
    {"id": "MR13", "type": "posterize", "params": {"bits": 5}},
    {"id": "MR14", "type": "translation", "params": {"dx": 1, "dy": 1}},
    {"id": "MR15", "type": "scale", "params": {"factor": 0.90}},
    {"id": "MR16", "type": "salt_pepper", "params": {"amount": 0.0017}},
    {"id": "MR17", "type": "saturation", "params": {"factor": 1.10}},
    {"id": "MR18", "type": "jpeg", "params": {"quality": 98}},
    {"id": "MR19", "type": "composite", "steps": [
        {"type": "flip"}, {"type": "noise", "params": {"sigma": 0.004}}]},
    {"id": "MR20", "type": "composite", "steps": [
        {"type": "contrast", "params": {"factor": 1.05}},
        {"type": "saturation", "params": {"factor": 1.05}}]},
]


@pytest.fixture(scope="module")
def images():
    return np.random.RandomState(0).randint(0, 256, (40, 32, 32, 3), dtype=np.uint8)


@pytest.mark.parametrize("spec", CANONICAL_SPECS, ids=lambda s: s["id"])
def test_spec_matches_canonical(spec, images):
    ref = mr_engine.MR_BY_ID[spec["id"]]
    mr = build_mr_from_spec(spec)
    for img in images:
        assert np.array_equal(mr.transform(img), ref.transform(img))
    m, r = mr.get_metadata(), ref.get_metadata()
    for k in ("category", "mr_type", "is_composite"):
        assert m[k] == r[k]


@pytest.mark.parametrize("bad", [
    {"id": "X1", "type": "os.system"},
    {"id": "X1", "type": "rotate"},                               # missing param
    {"id": "X1", "type": "rotate", "params": {"angle": 999}},     # out of range
    {"id": "X1", "type": "rotate", "params": {"angle": float("nan")}},
    {"id": "X1", "type": "rotate", "params": {"angle": True}},
    {"id": "X1", "type": "rotate", "params": {"angle": 1, "evil": 2}},
    {"id": "../x", "type": "flip"},
    {"id": "X1", "type": "composite", "steps": [{"type": "flip"}]},
    {"id": "X1", "type": "composite", "steps": [{"type": "flip"}, {"type": "composite"}]},
    {"id": "MR01", "type": "flip"},                               # collides with canonical
])
def test_invalid_specs_rejected(bad):
    with pytest.raises(MRSpecError):
        build_mrs_from_specs([bad])


def test_duplicate_ids_rejected():
    with pytest.raises(MRSpecError):
        build_mrs_from_specs([{"id": "U1", "type": "flip"}, {"id": "U1", "type": "flip"}])
