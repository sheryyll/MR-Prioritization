"""
"Run my MRs" tab for the MR-Prioritization Streamlit app.

Two modes, chosen automatically:
  * Lightweight (public deployment, no PyTorch): validate MR specs, preview the
    transformations on an uploaded image, and view a ranking (uploaded CSV or
    the bundled example).
  * Full (laptop with PyTorch + Model A checkpoint + mutants): also run the
    evaluation through the CLI (python -m mrrank.run, in a subprocess) and
    show the resulting ranking.

PyTorch is never imported here. Standalone test:  streamlit run mr_tab.py
Inside app.py:  import mr_tab  ...  with tab_run: mr_tab.render()
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from ui_theme import plot_chart

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from mrrank.mr_spec import MAX_COMPOSITE_STEPS, MRSpecError, build_mrs_from_specs, _TYPES
    from mrrank.mr_autocal import SEARCH as AUTO_SEARCH, prepare as prepare_specs
    SPEC_IMPORT_ERROR = None
except Exception as exc:  # e.g. OpenCV missing in this deployment
    SPEC_IMPORT_ERROR = exc

EXAMPLE_RANKING = ROOT / "app_artifacts" / "example_user_ranking.csv"
EXAMPLE_SPECS = [
    {"id": "U02", "name": "Brighten +15", "type": "brightness", "params": {"delta": 15}},
    {"id": "U01", "name": "Rotate (auto magnitude)", "type": "rotate", "params": {"angle": "auto"}},
    {"id": "U03", "name": "Mild contrast then JPEG", "type": "composite",
     "steps": [{"type": "contrast", "params": {"factor": 1.03}},
               {"type": "jpeg", "params": {"quality": 98}}]},
]
TIE_GAP = 0.01
CLEAN_LIMIT = 0.05
UNSTABLE_SPAN = 6   # 10th-90th percentile rank range wider than this = unstable
MAX_JSON_BYTES = 200_000
MAX_IMAGE_BYTES = 5_000_000
# Rough timing model (seconds, laptop CPU) used only for the on-screen estimate.
SEC_PER_PAIR_500 = 3.3      # one MR x one mutant on 500 images
COST_REPEATS, CALIBRATION_MRS = 5, 5
FIXED_OVERHEAD_S = 15


# ------------------------------------------------------------ pure helpers --

def annotate_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'note' column: MRs that detect no mutants, near-tied neighbours, and
    MRs that fire on more than 5% of the clean model's predictions."""
    df = df.reset_index(drop=True).copy()
    scores = df["priority_score"].astype(float).tolist()
    notes = []
    for i in range(len(df)):
        n = []
        if float(df.loc[i, "fdr"]) == 0.0:
            n.append("detects no mutants")
        if i + 1 < len(df) and abs(scores[i] - scores[i + 1]) < TIE_GAP:
            n.append("near-tie with next")
        if {"rank_p10", "rank_p90"} <= set(df.columns):
            if float(df.loc[i, "rank_p90"]) - float(df.loc[i, "rank_p10"]) > UNSTABLE_SPAN:
                n.append("rank unstable")
        if "clean_violation_rate" in df.columns:
            cv = df.loc[i, "clean_violation_rate"]
            if pd.notna(cv) and float(cv) > CLEAN_LIMIT:
                n.append("clean rate above 5%")
        notes.append("; ".join(n))
    df["note"] = notes
    return df


def parse_specs(text: str) -> list:
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("mrs")
    if not isinstance(data, list):
        raise ValueError('The JSON must be a list of MR specs, or {"mrs": [...]}.')
    return data


def count_auto(specs) -> int:
    return sum(1 for s in (specs or []) if any(isinstance(v, str) and v.strip().lower() == "auto"
                                                for v in (s.get("params") or {}).values()))


def estimate_seconds(n_mrs: int, fast: bool, n_auto: int = 0) -> float:
    """Upper-bound estimate: assumes every MR passes the clean-model check."""
    n_mut, n_img = (20, 100) if fast else (80, 500)
    scale = n_img / 500
    pairs = n_mrs * n_mut * SEC_PER_PAIR_500 * scale
    gate = n_mrs * SEC_PER_PAIR_500                                   # clean check, 500 images
    subset_clean = n_mrs * SEC_PER_PAIR_500 * scale if fast else 0.0
    cost = (CALIBRATION_MRS + n_mrs) * COST_REPEATS * SEC_PER_PAIR_500 * scale
    calib = n_auto * 10 * SEC_PER_PAIR_500                              # ~10 clean-rate measurements each
    return FIXED_OVERHEAD_S + pairs + gate + subset_clean + cost + calib


def format_duration(sec: float) -> str:
    return f"{sec:.0f} seconds" if sec < 90 else (f"{sec / 60:.0f} minutes" if sec < 7200
                                                    else f"{sec / 3600:.1f} hours")


def _run_cli(cmd, env, placeholder, timeout: int = 7200):
    """Run the CLI, streaming its output into `placeholder`. Always cleans up the child."""
    lines: list = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, env=env, cwd=ROOT)
    start = time.time()
    try:
        for line in proc.stdout:
            lines.append(line)
            placeholder.code("".join(lines[-14:]))
            if time.time() - start > timeout:
                lines.append("Timed out.\n")
                proc.kill()
                break
        proc.wait(timeout=30)
    finally:                      # also runs when the user presses Stop in Streamlit
        if proc.poll() is None:
            proc.kill()
    return proc.returncode, "".join(lines)


# ---------------------------------------------------------------- sections --

def show_ranking(df: pd.DataFrame, title: str = "Ranking") -> None:
    import plotly.express as px

    df = annotate_ranking(df)
    if len(df) <= 5:
        df = df.drop(columns=["top5_prob"], errors="ignore")
    st.subheader(title)
    top = ", ".join(df["mr_id"].head(5))
    st.markdown(f"**Top 5:** {top}")
    st.caption("Priority = 0.5·FDR + 0.2·(1−cost) + 0.3·diversity. Scores closer than "
               f"{TIE_GAP} are near-ties, so their exact order is not meaningful. "
               "An MR that detects no mutants still gets the full diversity bonus "
               "(its kill set overlaps nothing); it is flagged in the notes. "
               "rank_p10/rank_p90 give the 10th-90th percentile rank when mutants are "
               "resampled and cost is jittered; top5_prob is how often the MR lands in the top 5.")

    def highlight(row):
        return ["background-color: rgba(60,130,220,0.18)"] * len(row) if row.get("source") == "user" \
            else [""] * len(row)

    show_cols = [c for c in ["rank", "mr_id", "source", "priority_score", "fdr", "norm_cost",
                             "diversity", "rank_p10", "rank_p90", "top5_prob", "clean_violation_rate",
                             "note"] if c in df.columns]
    fmt = {c: "{:.3f}" for c in ["priority_score", "fdr", "norm_cost", "diversity",
                                 "clean_violation_rate", "top5_prob"] if c in show_cols}
    st.dataframe(df[show_cols].style.apply(highlight, axis=1).format(fmt),
                 width="stretch", hide_index=True)

    fig = px.bar(df, x="mr_id", y="priority_score",
                 color="source" if "source" in df.columns else None,
                 color_discrete_map={"user": "#3c82dc", "canonical": "#9aa5b1"})
    fig.update_layout(xaxis_title="MR (ranked order)", yaxis_title="Priority score",
                      xaxis={"categoryorder": "array", "categoryarray": df["mr_id"].tolist()})
    plot_chart(fig, width="stretch")
    st.download_button("Download ranking.csv", df.drop(columns=["note"]).to_csv(index=False),
                       file_name="ranking.csv", mime="text/csv")


def _specs_input():
    st.subheader("1. Define your MRs")
    up = st.file_uploader("Upload an MR spec file (JSON), or edit the text below",
                          type=["json"], key="mr_spec_upload")
    if "mr_specs_text" not in st.session_state:
        st.session_state["mr_specs_text"] = json.dumps(EXAMPLE_SPECS, indent=2)
    if up is not None and st.session_state.get("_mr_last_upload") != (up.name, up.size):
        st.session_state["_mr_last_upload"] = (up.name, up.size)
        if up.size > MAX_JSON_BYTES:
            st.error("That file is too large for an MR spec file.")
        else:
            st.session_state["mr_specs_text"] = up.getvalue().decode("utf-8", errors="replace")

    text = st.text_area("MR specs (JSON)", key="mr_specs_text", height=280)

    with st.expander("Allowed MR types and parameter ranges"):
        rows = [{"type": name,
                 "parameters (allowed range)": ", ".join(f"{p} [{lo}, {hi}]"
                                                         for p, (lo, hi, _d) in t["params"].items()) or "none"}
                for name, t in _TYPES.items()]
        rows.append({"type": "composite",
                     "parameters (allowed range)": f"steps: 2-{MAX_COMPOSITE_STEPS} of the types above"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        auto_rows = ", ".join(f"{t}.{p}" for (t, p) in AUTO_SEARCH)
        st.caption('**Auto magnitude:** set one numeric parameter to `"auto"` and the tool finds the '
                   "largest magnitude that stays under the clean-model limit (previews show it at its "
                   f"mildest). Supported: {auto_rows}.")
        st.caption("Each MR needs a unique id (letters, digits, - or _; not MR01-MR20). "
                   "An MR is only usable if it fires on under 5% of the clean model's "
                   "predictions; that check runs at evaluation time. Larger magnitudes "
                   "detect more mutants but are more likely to fail it.")

    try:
        specs = parse_specs(text)
        placeholder, _jobs = prepare_specs(specs)
        mrs = build_mrs_from_specs(placeholder)   # 'auto' values are shown at their mildest here
    except json.JSONDecodeError as e:
        st.error(f"Not valid JSON: {e}")
        return None, None
    except (MRSpecError, ValueError) as e:
        st.error(f"Invalid MR spec: {e}")
        return None, None
    st.success(f"{len(mrs)} MR spec(s) valid: " + ", ".join(m.id for m in mrs))
    st.download_button("Download specs as JSON", json.dumps(specs, indent=2),
                       file_name="my_mrs.json", mime="application/json")
    return specs, mrs


def _to_view(arr: np.ndarray, size: int = 160):
    from PIL import Image
    return Image.fromarray(arr).resize((size, size), Image.NEAREST)


def _preview(mrs) -> None:
    from PIL import Image, ImageOps

    st.subheader("2. Preview the transformations")
    f = st.file_uploader("Upload an image (the model works on 32×32 images, like CIFAR-10)",
                         type=["png", "jpg", "jpeg"], key="mr_img")
    if f is None:
        st.caption("Upload any image to see what each MR does to it.")
        return
    if f.size > MAX_IMAGE_BYTES:
        st.error("Image too large (limit 5 MB).")
        return
    try:
        orig = Image.open(f).convert("RGB")
        small = ImageOps.fit(orig, (32, 32), method=Image.LANCZOS)   # centre-crop to square, then shrink
        base = np.array(small, dtype=np.uint8)
    except Exception as e:
        st.error(f"Could not read that image: {e}")
        return

    show_diff = st.checkbox("Show amplified difference (×10) instead of the transformed image", value=False)
    shown = orig.copy()
    shown.thumbnail((160, 160))
    items = [("Uploaded image", shown), ("Model input (32×32)", _to_view(base))]
    for mr in mrs:
        out = np.asarray(mr.transform(base), dtype=np.uint8)
        if show_diff:
            d = np.abs(out.astype(np.int16) - base.astype(np.int16)) * 10
            out = np.clip(d, 0, 255).astype(np.uint8)
        items.append((f"{mr.id}: {mr.name}", _to_view(out)))
    for start in range(0, len(items), 4):
        cols = st.columns(4)
        for col, (cap, im) in zip(cols, items[start:start + 4]):
            col.image(im, caption=cap, width=160)
    st.caption("The model only ever sees the 32×32 version, so every tile after the first is "
               "that tiny image enlarged (hence the blocky pixels). Valid MRs are deliberately "
               "subtle: the prediction should not change. Tick the box to see where pixels changed.")


def can_run_locally():
    if os.environ.get("MRTAB_LIGHT") == "1":   # preview the public (no-torch) experience
        return False, "Lightweight mode is forced here (MRTAB_LIGHT=1)."
    if importlib.util.find_spec("torch") is None:
        return False, "PyTorch is not installed in this deployment."
    try:
        from mrrank import config
    except Exception as e:
        return False, f"The mrrank package could not be imported ({e})."
    if not Path(config.MODEL_A_CHECKPOINT).exists():
        return False, "The Model A checkpoint is not available here."
    if not any(Path(config.MUTANTS_DIR).glob("*.pth")):
        return False, "The mutant checkpoints are not available here."
    if not (Path(config.OUTPUTS_DIR) / "kill_matrix.npy").exists():
        return False, "The cached kill matrix is not available here."
    return True, ""


def _evaluation(specs, mrs) -> bool:
    """Draw step 3. Returns True when the server section was drawn (it already shows its own
    results, so the caller then skips the example-ranking viewer)."""
    st.subheader("3. Evaluate and rank")
    import mr_server
    if os.environ.get("MRRANK_PREFER_SERVER") == "1" and mr_server.render(specs, mrs, show_ranking):
        return True
    ok, why = can_run_locally()
    if ok:
        mode = st.radio("Mode", ["Fast preview (20 mutants, 100 images)",
                                 "Full evaluation (80 mutants, 500 images)"], key="mr_run_mode")
        fast = mode.startswith("Fast")
        scope_label = st.radio("Rank", ["My MRs only", "My MRs together with the 20 study MRs"],
                               key="mr_scope", horizontal=True,
                               help="'My MRs only' answers: which of MY MRs should I run first? "
                                    "'Together' shows how they compare with, and add coverage on top "
                                    "of, the study's MRs.")
        scope = "mine" if scope_label.startswith("My MRs only") else "all"
        if mrs:
            est = estimate_seconds(len(mrs), fast, count_auto(specs))
            st.caption(f"Estimated time for {len(mrs)} MR(s): up to about {format_duration(est)} "
                       "(MRs that fail the clean-model check are skipped; slower on a busy laptop).")
            if est > 600:
                st.warning("That is a long run. Use fewer MRs or Fast mode for a quick look.")
        if st.button("Run evaluation", type="primary", disabled=mrs is None):
            with tempfile.TemporaryDirectory() as td:
                spec_path, out = Path(td) / "mrs.json", Path(td) / "out"
                spec_path.write_text(json.dumps(specs), encoding="utf-8")
                cmd = [sys.executable, "-m", "mrrank.run", "--mrs", str(spec_path), "--out", str(out)]
                cmd += ["--scope", scope]
                if fast:
                    cmd.append("--fast")
                env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                       "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}
                live = st.empty()
                _rc, log = _run_cli(cmd, env, live)
                live.empty()
                st.session_state["mr_run_log"] = log
                csv, cfg = out / "ranking.csv", out / "run_config.json"
                st.session_state["mr_run_ranking"] = pd.read_csv(csv) if csv.exists() else None
                st.session_state["mr_run_cfg"] = (json.loads(cfg.read_text(encoding="utf-8"))
                                                  if cfg.exists() else None)
        cfg = st.session_state.get("mr_run_cfg")
        if cfg and cfg.get("rejected"):
            st.warning("Rejected before evaluation (fire on more than 5% of the clean model's "
                       "predictions, so their 'violations' are not reliable bug signals): "
                       + ", ".join(f"{k} ({v:.1%})" for k, v in cfg["rejected"].items())
                       + ". Try a smaller magnitude.")
        if cfg and cfg.get("calibration"):
            rows = [{"MR": k, "parameter": v["param"], "calibrated value": v["value"],
                     "clean violation rate": (f"{v['rate']:.1%}" if v["rate"] is not None else "-"),
                     "note": v["note"] or ""} for k, v in cfg["calibration"].items()]
            st.markdown("**Auto-calibrated magnitudes** (largest value under the clean-model limit, "
                        "specific to Model A and its evaluation images)")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        if cfg and cfg.get("mode") == "approximate":
            st.info("Approximate run (reduced mutants/images): use it as a preview.")
        if st.session_state.get("mr_run_log"):
            with st.expander("Run log"):
                st.code(st.session_state["mr_run_log"])
        ranking = st.session_state.get("mr_run_ranking")
        if ranking is not None:
            show_ranking(ranking, "Your ranking")
        return False
    if mr_server.render(specs, mrs, show_ranking):
        return True
    st.info(f"Running the evaluation needs PyTorch, the Model A checkpoint and the mutants. "
            f"{why} Run it where they are available (for example your own machine):")
    st.code("python -m mrrank.run --mrs my_mrs.json --out results\\my_run --fast", language="bash")
    st.caption("Drop `--fast` for the full 80-mutant, 500-image evaluation.")
    return False


def _results_viewer() -> None:
    st.subheader("View a ranking")
    up = st.file_uploader("Upload a ranking.csv produced by the command above",
                          type=["csv"], key="mr_rank_upload")
    df, title = None, ""
    if up is not None and up.size <= MAX_JSON_BYTES:
        try:
            df = pd.read_csv(up)
            need = {"mr_id", "priority_score", "fdr"}
            if not need <= set(df.columns):
                st.error(f"That CSV is missing columns: {sorted(need - set(df.columns))}")
                df = None
            else:
                title = "Your ranking"
        except Exception as e:
            st.error(f"Could not read that CSV: {e}")
    elif EXAMPLE_RANKING.exists():
        df, title = pd.read_csv(EXAMPLE_RANKING), "Example result (full run: U02 and U03 against all 80 mutants)"
    if df is not None:
        show_ranking(df, title)


def render() -> None:
    st.header("Run my MRs")
    st.markdown("Define your own metamorphic relations, preview what they do to an image, and "
                "rank them with the MR-Rank pipeline against the Model A / 80-mutant setup. "
                "Results apply to the CIFAR-10 ResNet-18 used in this study.")
    if SPEC_IMPORT_ERROR is not None:
        st.warning(f"Custom MR definition is unavailable in this deployment ({SPEC_IMPORT_ERROR}). "
                   "You can still view a ranking below.")
        _results_viewer()
        return
    st.markdown('<div id="define-mrs" class="section-anchor"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        specs, mrs = _specs_input()
    if mrs:
        with st.container(border=True):
            _preview(mrs)
    st.markdown('<div id="evaluate-mrs" class="section-anchor"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        server_used = _evaluation(specs, mrs)
    ok, _ = can_run_locally()
    if not ok and not server_used:
        _results_viewer()


if __name__ == "__main__":
    st.set_page_config(page_title="Run my MRs", layout="wide")
    render()
