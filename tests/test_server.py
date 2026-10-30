"""Server tests. A stub `mrrank.run` stands in for the real CLI, so no torch is needed."""
import importlib
import json
import sys
import textwrap
import time

import pytest
from fastapi.testclient import TestClient

STUB = textwrap.dedent('''
    import argparse, json, pathlib, sys, time
    p = argparse.ArgumentParser()
    p.add_argument("--mrs"); p.add_argument("--out"); p.add_argument("--fast", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    a = p.parse_args()
    specs = json.loads(pathlib.Path(a.mrs).read_text())
    if any("id" not in s for s in specs):
        print("MR spec invalid: missing id", file=sys.stderr); sys.exit(2)
    if a.validate_only:
        print(f"{len(specs)} MR spec(s) valid"); sys.exit(0)
    if any(s["id"] == "BOOM" for s in specs):
        print("crash"); sys.exit(1)
    time.sleep(0.2)
    out = pathlib.Path(a.out); out.mkdir(parents=True)
    (out / "ranking.json").write_text(json.dumps([{"rank": 1, "mr": specs[0]["id"]}]))
    (out / "kill_matrix.csv").write_text("mr,m1\\n" + specs[0]["id"] + ",1\\n")
    print("done")
''')


@pytest.fixture()
def client(tmp_path, monkeypatch):
    pkg = tmp_path / "stub" / "mrrank"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "run.py").write_text(STUB)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "stub"))
    monkeypatch.setenv("MRRANK_TOKEN", "secret")
    monkeypatch.setenv("MRRANK_REPO", str(tmp_path))
    monkeypatch.setenv("MRRANK_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setenv("MRRANK_MAX_MRS", "3")
    monkeypatch.setenv("MRRANK_MAX_QUEUE", "2")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "server"))
    sys.modules.pop("app", None)
    mod = importlib.import_module("app")
    yield TestClient(mod.app), mod
    sys.path.pop(0)


H = {"Authorization": "Bearer secret"}


def wait(c, job_id, want, timeout=15):
    end = time.time() + timeout
    while time.time() < end:
        st = c.get(f"/jobs/{job_id}", headers=H).json()
        if st["status"] in want:
            return st
        time.sleep(0.1)
    raise AssertionError(f"timeout, last={st}")


def test_health_is_open(client):
    c, _ = client
    assert c.get("/health").json()["ok"] is True


def test_auth_required(client):
    c, _ = client
    assert c.post("/jobs", json={"mrs": [{"id": "A"}]}).status_code == 401
    assert c.post("/jobs", json={"mrs": [{"id": "A"}]},
                  headers={"Authorization": "Bearer nope"}).status_code == 401


def test_happy_path(client):
    c, _ = client
    r = c.post("/jobs", json={"mrs": [{"id": "A"}]}, headers=H)
    assert r.status_code == 202
    jid = r.json()["job_id"]
    wait(c, jid, {"done"})
    res = c.get(f"/jobs/{jid}/result", headers=H).json()
    assert res["ranking"][0]["mr"] == "A"
    assert "kill_matrix_csv" in res


def test_invalid_spec_rejected(client):
    c, _ = client
    r = c.post("/jobs", json={"mrs": [{"name": "no id"}]}, headers=H)
    assert r.status_code == 422 and "missing id" in r.json()["detail"]


def test_too_many_mrs(client):
    c, _ = client
    r = c.post("/jobs", json={"mrs": [{"id": str(i)} for i in range(4)]}, headers=H)
    assert r.status_code == 422


def test_failed_job_reports_error(client):
    c, _ = client
    jid = c.post("/jobs", json={"mrs": [{"id": "BOOM"}]}, headers=H).json()["job_id"]
    st = wait(c, jid, {"failed"})
    assert "code 1" in st["error"]
    assert c.get(f"/jobs/{jid}/result", headers=H).status_code == 409


def test_bad_job_id_and_path_traversal(client):
    c, _ = client
    assert c.get("/jobs/..%2F..%2Fetc", headers=H).status_code == 404
    assert c.get("/jobs/" + "0" * 32, headers=H).status_code == 404


def test_queue_limit(client):
    c, _ = client
    codes = [c.post("/jobs", json={"mrs": [{"id": "A"}]}, headers=H).status_code
             for _ in range(4)]
    assert 429 in codes


def test_cleanup_removes_old_finished_jobs(client):
    c, mod = client
    jid = c.post("/jobs", json={"mrs": [{"id": "A"}]}, headers=H).json()["job_id"]
    wait(c, jid, {"done"})
    import os
    old = time.time() - 10 * 24 * 3600
    os.utime(mod.JOBS_DIR / jid, (old, old))
    c.post("/jobs", json={"mrs": [{"id": "A"}]}, headers=H)  # triggers cleanup
    assert not (mod.JOBS_DIR / jid).exists()
