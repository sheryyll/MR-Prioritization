"""Torch-free tests for MR auto-calibration."""
import pytest

from mrrank import mr_autocal as ac


def test_bisect_finds_largest_value_under_threshold():
    res = ac.bisect_max_valid(lambda v: v / 100, mild=0.2, aggressive=45.0, threshold=0.04)
    assert 3.8 < res["value"] < 4.0 and res["rate"] < 0.04 and res["note"] is None


def test_bisect_mild_end_already_invalid():
    res = ac.bisect_max_valid(lambda v: 0.2, mild=0.2, aggressive=45.0, threshold=0.04)
    assert res["value"] is None and "mildest" in res["note"]


def test_bisect_capped_by_allowed_range():
    res = ac.bisect_max_valid(lambda v: 0.0, mild=0.2, aggressive=45.0, threshold=0.04)
    assert res["value"] == 45.0 and "maximum allowed" in res["note"]


def test_bisect_integer_decreasing_parameter():
    # bits 8 (mild) -> 1 (aggressive); rate grows as bits fall; valid while bits >= 5
    res = ac.bisect_max_valid(lambda b: (8 - b) * 0.01, mild=8, aggressive=1, threshold=0.04, integer=True)
    assert res["value"] == 5 and isinstance(res["value"], int)


def test_prepare_replaces_auto_with_mildest_and_lists_jobs():
    specs = [{"id": "U1", "type": "rotate", "params": {"angle": "auto"}},
             {"id": "U2", "type": "flip"},
             {"id": "U3", "type": "translation", "params": {"dx": "AUTO", "dy": 1}}]
    placeholder, jobs = ac.prepare(specs)
    assert jobs == [(0, "angle"), (2, "dx")]
    assert placeholder[0]["params"]["angle"] == 0.2 and placeholder[2]["params"] == {"dx": 0.25, "dy": 1}
    assert specs[0]["params"]["angle"] == "auto"          # input untouched


@pytest.mark.parametrize("bad", [
    {"id": "X", "type": "composite", "steps": [{"type": "flip"}, {"type": "flip"}],
     "params": {"angle": "auto"}},
    {"id": "X", "type": "rotate", "params": {"angle": "auto", "evil": "auto"}},
    {"id": "X", "type": "flip", "params": {"angle": "auto"}},
    {"id": "X", "type": "sharpen", "params": {"ksize": "auto", "amount": 1}},
])
def test_prepare_rejects_unsupported_auto(bad):
    with pytest.raises(ac.AutoSpecError):
        ac.prepare([bad])


def test_resolve_sets_values_and_drops_uncalibratable():
    specs = [{"id": "U1", "type": "rotate", "params": {"angle": "auto"}},
             {"id": "U2", "type": "brightness", "params": {"delta": "auto"}}]
    _, jobs = ac.prepare(specs)

    def rate(c):   # rotate: angle/100 ; brightness: always invalid
        return c["params"]["angle"] / 100 if c["type"] == "rotate" else 0.5

    logs = []
    final, report = ac.resolve(specs, jobs, rate, log=logs.append)
    assert 3.8 < final[0]["params"]["angle"] < 4.0
    assert final[1] is None and report["U2"]["value"] is None and report["U1"]["rate"] < 0.04
    assert logs and "U1" in logs[0]
