"""MR-Rank job server.

Accepts MR specs, runs `python -m mrrank.run ... --fast` as a background
subprocess (one at a time), and exposes status and results.

Environment variables
  MRRANK_TOKEN        required. Clients send "Authorization: Bearer <token>".
  MRRANK_REPO         repo root used as the working directory (default: parent of server/)
  MRRANK_PYTHON       interpreter used for the run (default: the one running this server)
  MRRANK_MODULE       module to run (default: mrrank.run)
  MRRANK_JOBS_DIR     where job folders live (default: <repo>/results/jobs)
  MRRANK_MAX_MRS      max MRs per job (default 5)
  MRRANK_MAX_QUEUE    max waiting + running jobs (default 6)
  MRRANK_TTL_HOURS    delete job folders older than this (default 24)
  MRRANK_TIMEOUT_S    kill a run after this many seconds (default 1800)
"""
from __future__ import annotations

import hmac
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

REPO = Path(os.environ.get("MRRANK_REPO", Path(__file__).resolve().parent.parent))
PYTHON = os.environ.get("MRRANK_PYTHON", sys.executable)
MODULE = os.environ.get("MRRANK_MODULE", "mrrank.run")
JOBS_DIR = Path(os.environ.get("MRRANK_JOBS_DIR", REPO / "results" / "jobs"))
MAX_MRS = int(os.environ.get("MRRANK_MAX_MRS", "5"))
MAX_QUEUE = int(os.environ.get("MRRANK_MAX_QUEUE", "6"))
TTL_S = float(os.environ.get("MRRANK_TTL_HOURS", "24")) * 3600
TIMEOUT_S = int(os.environ.get("MRRANK_TIMEOUT_S", "1800"))

JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
LOG_TAIL_CHARS = 4000

app = FastAPI(title="MR-Rank server")
_queue: "queue.Queue[str]" = queue.Queue()
_lock = threading.Lock()
_active = 0  # queued + running
_worker_started = False


class JobRequest(BaseModel):
    mrs: list[dict[str, Any]] = Field(..., min_length=1)


def _token() -> str:
    tok = os.environ.get("MRRANK_TOKEN", "")
    if not tok:
        raise HTTPException(503, "Server has no MRRANK_TOKEN configured")
    return tok


def require_auth(authorization: str | None = Header(default=None)) -> None:
    expected = _token()
    given = ""
    if authorization and authorization.lower().startswith("bearer "):
        given = authorization[7:].strip()
    if not hmac.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(401, "Invalid or missing token")


def _job_dir(job_id: str) -> Path:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(404, "Unknown job")
    d = JOBS_DIR / job_id
    if not d.is_dir():
        raise HTTPException(404, "Unknown job")
    return d


def _write_status(d: Path, **fields: Any) -> None:
    path = d / "status.json"
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(fields)
    tmp = d / "status.json.tmp"
    tmp.write_text(json.dumps(cur))
    tmp.replace(path)


def _read_status(d: Path) -> dict[str, Any]:
    try:
        return json.loads((d / "status.json").read_text())
    except (OSError, ValueError):
        return {"status": "failed", "error": "status file unreadable"}


def _cleanup_old_jobs() -> None:
    if not JOBS_DIR.exists():
        return
    now = time.time()
    for d in JOBS_DIR.iterdir():
        try:
            if d.is_dir() and JOB_ID_RE.match(d.name) and now - d.stat().st_mtime > TTL_S:
                st = _read_status(d).get("status")
                if st not in ("queued", "running"):
                    shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def _run_cli(args: list[str], cwd: Path, log: Path | None, timeout: int):
    cmd = [PYTHON, "-m", MODULE, *args]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    if log is None:
        return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True,
                              text=True, timeout=timeout)
    with open(log, "w", encoding="utf-8") as fh:
        return subprocess.run(cmd, cwd=cwd, env=env, stdout=fh,
                              stderr=subprocess.STDOUT, timeout=timeout)


def _worker() -> None:
    global _active
    while True:
        job_id = _queue.get()
        d = JOBS_DIR / job_id
        try:
            _write_status(d, status="running", started=time.time())
            proc = _run_cli(
                ["--mrs", str(d / "mrs.json"), "--out", str(d / "out"), "--fast"],
                REPO, d / "job.log", TIMEOUT_S,
            )
            ok = proc.returncode == 0 and (d / "out" / "ranking.json").exists()
            _write_status(d, status="done" if ok else "failed", finished=time.time(),
                          error=None if ok else f"CLI exited with code {proc.returncode}")
        except subprocess.TimeoutExpired:
            _write_status(d, status="failed", finished=time.time(),
                          error=f"Timed out after {TIMEOUT_S}s")
        except Exception as exc:  # keep the worker alive whatever happens
            _write_status(d, status="failed", finished=time.time(), error=str(exc))
        finally:
            with _lock:
                _active -= 1
            _queue.task_done()


def _ensure_worker() -> None:
    global _worker_started
    with _lock:
        if not _worker_started:
            threading.Thread(target=_worker, daemon=True).start()
            _worker_started = True


def _run_summary(d: Path) -> dict[str, Any] | None:
    """Calibration results, rejected MRs and run mode from the CLI's run_config.json."""
    try:
        cfg = json.loads((d / "out" / "run_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    cal = cfg.get("calibration")
    if isinstance(cal, dict):
        cal = {k: {f: (v or {}).get(f) for f in ("param", "value", "rate", "note")}
               for k, v in cal.items()}
    return {"calibration": cal, "rejected": cfg.get("rejected") or {}, "mode": cfg.get("mode")}


def _log_tail(d: Path) -> str:
    try:
        return (d / "job.log").read_text(encoding="utf-8", errors="replace")[-LOG_TAIL_CHARS:]
    except OSError:
        return ""


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "active_jobs": _active, "max_mrs": MAX_MRS}


@app.post("/jobs", status_code=202, dependencies=[Depends(require_auth)])
def submit(req: JobRequest) -> dict[str, Any]:
    global _active
    if len(req.mrs) > MAX_MRS:
        raise HTTPException(422, f"At most {MAX_MRS} MRs per job")
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_old_jobs()
    with _lock:
        if _active >= MAX_QUEUE:
            raise HTTPException(429, "Server is busy, try again later")

    job_id = uuid.uuid4().hex
    d = JOBS_DIR / job_id
    d.mkdir()
    spec = d / "mrs.json"
    spec.write_text(json.dumps(req.mrs), encoding="utf-8")

    # Reuse the CLI's own validation so the server never accepts what the CLI rejects.
    try:
        chk = _run_cli(["--mrs", str(spec), "--validate-only"], REPO, None, 120)
    except subprocess.TimeoutExpired:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(500, "Validation timed out")
    if chk.returncode != 0:
        shutil.rmtree(d, ignore_errors=True)
        msg = (chk.stderr or chk.stdout).strip()[-1000:]
        raise HTTPException(422, f"Invalid MR spec: {msg}")

    _write_status(d, status="queued", created=time.time(), n_mrs=len(req.mrs))
    with _lock:
        _active += 1
    _ensure_worker()
    _queue.put(job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}", dependencies=[Depends(require_auth)])
def status(job_id: str) -> dict[str, Any]:
    d = _job_dir(job_id)
    st = _read_status(d)
    st["job_id"] = job_id
    st["log_tail"] = _log_tail(d)
    if st.get("status") in ("done", "failed"):
        st["summary"] = _run_summary(d)
    return st


@app.get("/jobs/{job_id}/result", dependencies=[Depends(require_auth)])
def result(job_id: str) -> dict[str, Any]:
    d = _job_dir(job_id)
    st = _read_status(d)
    if st.get("status") != "done":
        raise HTTPException(409, f"Job is {st.get('status')}")
    out = d / "out"
    payload: dict[str, Any] = {
        "job_id": job_id,
        "ranking": json.loads((out / "ranking.json").read_text(encoding="utf-8")),
        "log_tail": _log_tail(d),
        "summary": _run_summary(d),
    }
    km = out / "kill_matrix.csv"
    if km.exists():
        payload["kill_matrix_csv"] = km.read_text(encoding="utf-8")
    return payload
