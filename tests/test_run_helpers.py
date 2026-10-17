"""Torch-free tests for the pure logic in mrrank.run."""
import json

import numpy as np
import pytest

from mrrank import run


def _base():
    km = np.array([[1, 0, 1, 0], [0, 1, 0, 0]], bool)
    return {"kill_matrix": km, "mr_ids": ["M1", "M2"], "mutant_ids": ["a", "b", "c", "d"],
            "clean_violation_rates": {"M1": 0.0, "M2": 0.0}, "fdr_per_mr": {"M1": .5, "M2": .25}}


def test_pick_evenly():
    items = list(range(80))
    sub = run.pick_evenly(items, 20)
    assert len(sub) == 20 and len(set(sub)) == 20 and sub[0] == 0 and sub[-1] == 79
    assert run.pick_evenly(items, 200) == items


def test_stratified_indices_balanced():
    labels = np.repeat(np.arange(10), 50)
    idx = run.stratified_indices(labels, 100)
    assert len(idx) == 100
    assert all((labels[idx] == c).sum() == 10 for c in range(10))


def test_subset_columns_recomputes_fdr():
    sub = run.subset_columns(_base(), ["a", "c"])
    assert sub["mutant_ids"] == ["a", "c"]
    assert sub["kill_matrix"].tolist() == [[True, True], [False, False]]
    assert sub["fdr_per_mr"] == {"M1": 1.0, "M2": 0.0}
    with pytest.raises(ValueError):
        run.subset_columns(_base(), ["zzz"])


def test_rank_merged_marks_sources():
    base = _base()
    merged = {**base, "mr_ids": ["M1", "M2", "U1"],
              "kill_matrix": np.vstack([base["kill_matrix"], [[1, 1, 1, 1]]]),
              "fdr_per_mr": {"M1": .5, "M2": .25, "U1": 1.0},
              "clean_violation_rates": {"M1": 0, "M2": 0, "U1": 0.01}}
    ranking = run.rank_merged(merged, {"M1": 1.0, "M2": 2.0, "U1": 1.5}, {"U1"})
    assert [r["rank"] for r in ranking] == [1, 2, 3]
    assert ranking[0]["mr_id"] == "U1" and ranking[0]["source"] == "user"
    assert {r["source"] for r in ranking[1:]} == {"canonical"}


def test_load_specs_formats(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps([{"id": "U1", "type": "flip"}]))
    assert run.load_specs(p)[0]["id"] == "U1"
    p.write_text(json.dumps({"mrs": [{"id": "U1", "type": "flip"}]}))
    assert len(run.load_specs(p)) == 1
    p.write_text(json.dumps({"nope": 1}))
    with pytest.raises(run.MRSpecError):
        run.load_specs(p)


def test_main_validate_only(tmp_path, capsys):
    good = tmp_path / "good.json"
    good.write_text(json.dumps([{"id": "U1", "type": "rotate", "params": {"angle": 2.0}}]))
    assert run.main(["--mrs", str(good), "--validate-only"]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "U1", "type": "os.system"}]))
    assert run.main(["--mrs", str(bad), "--validate-only"]) == 2
    assert run.main(["--mrs", str(tmp_path / "missing.json"), "--validate-only"]) == 2


def test_calibrated_costs_rescales_user_only():
    out = run.calibrated_costs({"A": 1.0, "B": 2.0}, {"A": 2.0, "B": 4.0}, {"U": 6.0})
    assert out == {"A": 1.0, "B": 2.0, "U": 3.0}
    with pytest.raises(ValueError):
        run.calibrated_costs({"A": 1.0}, {}, {"U": 1.0})
