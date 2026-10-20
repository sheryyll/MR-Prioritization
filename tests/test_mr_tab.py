"""Smoke tests for the Streamlit 'Run my MRs' tab (lightweight, no torch needed)."""
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAB = str(ROOT / "mr_tab.py")


def test_tab_renders_with_example_specs():
    at = AppTest.from_file(TAB, default_timeout=60).run()
    assert not at.exception
    assert any("valid" in s.value for s in at.success)


def test_tab_rejects_bad_specs():
    at = AppTest.from_file(TAB, default_timeout=60).run()
    at.text_area(key="mr_specs_text").set_value('[{"id": "U1", "type": "os.system"}]').run()
    assert not at.exception
    assert any("Invalid MR spec" in e.value for e in at.error)
    at.text_area(key="mr_specs_text").set_value("[{bad").run()
    assert any("Not valid JSON" in e.value for e in at.error)


def test_annotate_ranking_flags_zero_fdr_and_ties():
    import sys
    sys.path.insert(0, str(ROOT))
    import mr_tab
    df = pd.DataFrame({"mr_id": ["A", "B", "C"], "priority_score": [0.50, 0.495, 0.30],
                       "fdr": [0.6, 0.0, 0.1]})
    notes = mr_tab.annotate_ranking(df)["note"].tolist()
    assert "near-tie" in notes[0] and "detects no mutants" in notes[1] and notes[2] == ""


def test_annotate_flags_clean_rate_above_limit():
    import sys
    sys.path.insert(0, str(ROOT))
    import mr_tab
    df = pd.DataFrame({"mr_id": ["A", "B"], "priority_score": [0.9, 0.2], "fdr": [0.5, 0.5],
                       "clean_violation_rate": [0.088, 0.02]})
    notes = mr_tab.annotate_ranking(df)["note"].tolist()
    assert "clean rate above 5%" in notes[0] and notes[1] == ""


def test_annotate_flags_unstable_rank():
    import sys
    sys.path.insert(0, str(ROOT))
    import mr_tab
    df = pd.DataFrame({"mr_id": ["A", "B"], "priority_score": [0.9, 0.2], "fdr": [0.5, 0.5],
                       "rank_p10": [1, 4], "rank_p90": [1, 15]})
    notes = mr_tab.annotate_ranking(df)["note"].tolist()
    assert notes[0] == "" and "rank unstable" in notes[1]


def test_light_mode_shows_example_ranking(monkeypatch):
    monkeypatch.setenv("MRTAB_LIGHT", "1")
    at = AppTest.from_file(TAB, default_timeout=60).run()
    assert not at.exception
    assert any("Lightweight mode" in i.value for i in at.info)


def test_time_estimate_scales_with_mrs_and_mode():
    import sys
    sys.path.insert(0, str(ROOT))
    import mr_tab
    two, eighteen = mr_tab.estimate_seconds(2, True), mr_tab.estimate_seconds(18, True)
    assert 30 < two < 120                      # "about a minute" for 2 MRs
    assert 300 < eighteen < 600                # 5-10 minutes for 18 MRs in fast mode
    assert mr_tab.estimate_seconds(18, False) > 3600
    assert mr_tab.format_duration(45) == "45 seconds" and mr_tab.format_duration(300) == "5 minutes"


def test_run_cli_streams_and_returns_output():
    import sys
    sys.path.insert(0, str(ROOT))
    import mr_tab

    class Ph:
        def __init__(self): self.seen = []
        def code(self, text): self.seen.append(text)

    ph = Ph()
    rc, log = mr_tab._run_cli([sys.executable, "-u", "-c", "print('a'); print('b')"], None, ph)
    assert rc == 0 and log.splitlines() == ["a", "b"] and ph.seen
