"""
Optional "Run on the MR-Rank server" mode for the Run-my-MRs tab.

The public app never imports PyTorch. When MRRANK_SERVER_URL and MRRANK_SERVER_TOKEN are set
(environment variables locally, or Streamlit secrets on Community Cloud) and the server's
/health endpoint answers, the tab offers a button that sends the MR specs to the server,
polls the job, and shows the ranking. If anything is missing or the server is offline,
render() returns False and the tab falls back to its normal behaviour.

Secrets example (.streamlit/secrets.toml, never commit this file):
    MRRANK_SERVER_URL = "https://something.trycloudflare.com"
    MRRANK_SERVER_TOKEN = "the same string as the server's MRRANK_TOKEN"
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests
import streamlit as st

HEALTH_TTL_S = 15
POLL_EVERY_S = 3
MAX_WAIT_S = 45 * 60          # longer than the server's own 30-minute run limit
REQ_TIMEOUT_S = 15
LOG_LINES_SHOWN = 14


class ServerError(Exception):
    def __init__(self, msg: str, fatal: bool = False):
        super().__init__(msg)
        self.fatal = fatal


# ---------------------------------------------------------------- plain client --

def _setting(name: str):
    val = os.environ.get(name)
    if not val:
        try:
            val = st.secrets.get(name)
        except Exception:        # no secrets file at all
            val = None
    return str(val).strip() if val else None


def get_config():
    url, token = _setting("MRRANK_SERVER_URL"), _setting("MRRANK_SERVER_TOKEN")
    if not url or not token:
        return None
    return url.rstrip("/"), token


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _detail(resp: requests.Response) -> str:
    try:
        d = resp.json().get("detail")
    except ValueError:
        d = None
    if isinstance(d, list):
        d = "; ".join(str(x.get("msg", x)) if isinstance(x, dict) else str(x) for x in d)
    return str(d or resp.text[:200] or resp.reason)


def health(url: str):
    try:
        r = requests.get(f"{url}/health", timeout=4)
        j = r.json() if r.ok else None
    except (requests.RequestException, ValueError):
        return None
    return j if isinstance(j, dict) and j.get("ok") else None


def submit(url: str, token: str, specs: list) -> str:
    try:
        r = requests.post(f"{url}/jobs", json={"mrs": specs}, headers=_auth(token),
                          timeout=REQ_TIMEOUT_S)
    except requests.RequestException:
        raise ServerError("Could not reach the MR-Rank server.")
    if r.status_code == 202:
        return r.json()["job_id"]
    if r.status_code == 401:
        raise ServerError("The server rejected the access token.")
    if r.status_code == 429:
        raise ServerError("The server is busy. Try again in a minute.")
    raise ServerError(_detail(r))


def poll(url: str, token: str, job_id: str) -> dict:
    try:
        r = requests.get(f"{url}/jobs/{job_id}", headers=_auth(token), timeout=REQ_TIMEOUT_S)
    except requests.RequestException:
        raise ServerError("no answer from the server")
    if r.status_code == 404:
        raise ServerError("The job no longer exists on the server (it may have restarted).",
                          fatal=True)
    if not r.ok:
        raise ServerError(_detail(r))
    return r.json()


def fetch_result(url: str, token: str, job_id: str) -> dict:
    try:
        r = requests.get(f"{url}/jobs/{job_id}/result", headers=_auth(token),
                         timeout=REQ_TIMEOUT_S)
    except requests.RequestException:
        raise ServerError("no answer from the server")
    if not r.ok:
        raise ServerError(_detail(r), fatal=True)
    return r.json()


# ------------------------------------------------------------------- UI pieces --

@st.cache_data(ttl=HEALTH_TTL_S, show_spinner=False)
def _health_cached(url: str):
    return health(url)


def _finish(**state) -> None:
    for k, v in state.items():
        st.session_state[k] = v
    st.session_state["mr_srv_job"] = None
    st.rerun()


def _panel(url: str, token: str) -> None:
    job = st.session_state.get("mr_srv_job")
    if not job:
        return
    waited = time.time() - st.session_state.get("mr_srv_t0", time.time())
    if waited > MAX_WAIT_S:
        _finish(mr_srv_err="The server took too long. Try again, or run the command on your own machine.")
    try:
        s = poll(url, token, job)
    except ServerError as e:
        if e.fatal:
            _finish(mr_srv_err=str(e))
        st.warning(f"Lost contact with the server ({e}). Retrying...")
        return
    status = s.get("status")
    if status in ("queued", "running"):
        label = "Waiting in the queue" if status == "queued" else "Running the MR-Rank pipeline"
        st.info(f"{label}... {int(waited)} s elapsed")
        tail = (s.get("log_tail") or "").splitlines()[-LOG_LINES_SHOWN:]
        if tail:
            st.code("\n".join(tail))
    elif status == "done":
        try:
            res = fetch_result(url, token, job)
        except ServerError as e:
            _finish(mr_srv_err=f"The job finished but its result could not be fetched: {e}")
        _finish(mr_run_ranking=pd.DataFrame(res["ranking"]), mr_run_log=res.get("log_tail", ""),
                mr_srv_err=None)
    else:
        _finish(mr_srv_err=f"The run failed: {s.get('error') or 'unknown error'}",
                mr_run_ranking=None, mr_run_log=s.get("log_tail", ""))


def render(specs, mrs, show_ranking) -> bool:
    """Draw the server section. Returns False (and draws nothing) when no usable server exists."""
    cfg = get_config()
    if cfg is None:
        return False
    url, token = cfg
    info = _health_cached(url)
    if info is None:
        return False
    max_mrs = int(info.get("max_mrs", 5))
    busy = bool(st.session_state.get("mr_srv_job"))

    st.markdown("**Run on the MR-Rank server.** Your MRs are evaluated on the project's own "
                "machine, which holds the Model A checkpoint and the 80 mutants. It runs a fast "
                f"preview (20 mutants, 100 images), usually a few minutes. Up to {max_mrs} MRs per run.")
    too_many = specs is not None and len(specs) > max_mrs
    if too_many:
        st.warning(f"This server accepts at most {max_mrs} MRs per run; you have {len(specs)}.")
    if st.button("Run on the MR-Rank server", type="primary",
                 disabled=(mrs is None or too_many or busy), key="mr_srv_btn"):
        try:
            job = submit(url, token, specs)
        except ServerError as e:
            st.session_state["mr_srv_err"] = str(e)
        else:
            st.session_state.update(mr_srv_job=job, mr_srv_t0=time.time(), mr_srv_err=None,
                                    mr_run_ranking=None, mr_run_log="")

    st.fragment(run_every=POLL_EVERY_S if st.session_state.get("mr_srv_job") else None)(_panel)(url, token)

    if st.session_state.get("mr_srv_err"):
        st.error(st.session_state["mr_srv_err"])
    if st.session_state.get("mr_run_log") and not busy:
        with st.expander("Run log (shows any MR that could not be calibrated or was rejected)"):
            st.code(st.session_state["mr_run_log"])
    ranking = st.session_state.get("mr_run_ranking")
    if ranking is not None:
        show_ranking(ranking, "Your ranking (server, fast preview)")
    return True
