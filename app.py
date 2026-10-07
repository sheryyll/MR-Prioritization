"""
app.py — MR-Rank Streamlit Community Cloud entry-point.

Tabs:
  1. Rankings         — Greedy + Meta-classifier tables & charts
  2. Evaluation       — APFD table, FD@k curves, kill-matrix heatmap
  3. Rank a new model — Upload fingerprint JSON → ranked MR list
  4. About            — Project summary, version, known issues

Hard constraints:
  * NO torch / torchvision import anywhere in this file.
  * All data loaded from app_artifacts/ with st.cache_data / st.cache_resource.
  * No training, mutation generation, or kill-matrix rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS = Path(__file__).resolve().parent / "app_artifacts"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MR-Rank | Metamorphic Relation Prioritization",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Refined typography & spacing */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 50%, #0f1117 100%);
    }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { color: #e0e7ff; letter-spacing: -0.5px; }
    h2 { color: #c7d2fe; font-size: 1.3rem; }
    h3 { color: #a5b4fc; font-size: 1.1rem; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    [data-testid="stMetricLabel"] { color: #a5b4fc !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"] { color: #e0e7ff !important; }
    [data-testid="stMetricDelta"] { font-size: 0.72rem; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(15, 17, 23, 0.6);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.25) !important;
        color: #e0e7ff !important;
    }

    /* Dataframes */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Info/warning boxes */
    .stAlert { border-radius: 10px; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(99, 102, 241, 0.35);
        border-radius: 12px;
        padding: 1rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1117; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.title("🧪 MR-Rank")
    st.caption("Metamorphic Relation Prioritization for ResNet-18 on CIFAR-10 · 20 MRs × 80 Mutants")
with col_badge:
    st.markdown(
        "<div style='text-align:right;padding-top:1rem'>"
        "<span style='background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.4);"
        "border-radius:20px;padding:4px 12px;font-size:0.75rem;color:#a5b4fc'>v1.0.0</span>"
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_greedy() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "greedy_ranking.csv")


@st.cache_data(show_spinner=False)
def load_meta_clf_ranking() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "metaclassifier_ranking.csv")


@st.cache_data(show_spinner=False)
def load_mr_library() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "mr_library.csv")


@st.cache_data(show_spinner=False)
def load_mr_scores() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "mr_scores.csv")


@st.cache_data(show_spinner=False)
def load_kill_matrix() -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(ARTIFACTS / "kill_matrix.csv", index_col=0)
    return df, list(df.index), list(df.columns)


@st.cache_data(show_spinner=False)
def load_eval() -> dict:
    return json.loads((ARTIFACTS / "evaluation_summary.json").read_text())


@st.cache_data(show_spinner=False)
def load_feature_columns() -> list[str]:
    return json.loads((ARTIFACTS / "feature_columns.json").read_text())


@st.cache_resource(show_spinner=False)
def load_classifier():
    bundle = joblib.load(ARTIFACTS / "meta_classifier.joblib")
    return bundle["clf"]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_rank, tab_eval, tab_infer, tab_about = st.tabs([
    "📊 Rankings",
    "📈 Evaluation",
    "🔮 Rank a new model",
    "ℹ️ About",
])

# ============================================================ Tab 1: Rankings
with tab_rank:
    greedy_df = load_greedy()
    meta_df = load_meta_clf_ranking()
    mr_lib = load_mr_library()
    mr_scores = load_mr_scores()

    st.markdown("### Greedy Ranking — Model A")
    st.caption(
        "Priority = **0.5 × FDR** + **0.2 × (1 − NormCost)** + **0.3 × Diversity**  "
        "(see report Eq. 4.1-4.4)"
    )

    # Metrics
    km_df, mr_ids, mutant_ids = load_kill_matrix()
    km_bool = km_df.values.astype(bool)
    n_detectable = int(km_bool.any(axis=0).sum())
    eval_data = load_eval()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MRs", len(mr_ids))
    m2.metric("Mutants", len(mutant_ids))
    m3.metric("Detectable mutants", n_detectable)
    greedy_apfd = eval_data["apfd"]["greedy"]
    rand_apfd = eval_data["apfd"]["random_mean"]
    m4.metric("APFD (Greedy)", f"{greedy_apfd:.4f}", delta=f"{greedy_apfd - rand_apfd:+.4f} vs random")

    st.divider()

    # Greedy bar chart
    greedy_display = greedy_df.merge(mr_lib[["mr_id", "name", "category"]], on="mr_id", how="left")
    greedy_display["label"] = greedy_display["mr_id"] + ": " + greedy_display["name"].fillna("")

    fig_greedy = px.bar(
        greedy_display,
        x="mr_id",
        y="priority_score",
        color="category",
        hover_data=["name", "fdr", "norm_cost", "diversity_at_selection"],
        color_discrete_sequence=px.colors.qualitative.Vivid,
        labels={"priority_score": "Priority Score", "mr_id": ""},
        category_orders={"mr_id": greedy_df["mr_id"].tolist()},
    )
    fig_greedy.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=320,
        xaxis=dict(tickfont=dict(size=11)),
        legend_title="Category",
    )
    st.plotly_chart(fig_greedy, use_container_width=True)

    with st.expander("Full Greedy Ranking Table", expanded=False):
        display_cols = ["rank", "mr_id", "priority_score", "fdr", "norm_cost", "diversity_at_selection"]
        st.dataframe(
            greedy_display[display_cols].set_index("rank").round(4),
            use_container_width=True,
        )

    st.divider()
    st.markdown("### Meta-Classifier Ranking — Model B")
    st.caption(
        "Gradient Boosting classifier predicts kill probability per (MR, mutant) pair. "
        "Ranks are averaged over all mutant profiles."
    )

    meta_display = meta_df.merge(mr_lib[["mr_id", "name", "category"]], on="mr_id", how="left")
    fig_meta = px.bar(
        meta_display,
        x="mr_id",
        y="predicted_kill_prob",
        color="category",
        hover_data=["name"],
        color_discrete_sequence=px.colors.qualitative.Vivid,
        labels={"predicted_kill_prob": "Predicted Kill Probability", "mr_id": ""},
        category_orders={"mr_id": meta_df["mr_id"].tolist()},
    )
    fig_meta.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=320,
        xaxis=dict(tickfont=dict(size=11)),
        legend_title="Category",
    )
    st.plotly_chart(fig_meta, use_container_width=True)

    with st.expander("Full Meta-Classifier Ranking Table", expanded=False):
        st.dataframe(
            meta_display[["rank", "mr_id", "name", "category", "predicted_kill_prob"]]
            .set_index("rank").round(4),
            use_container_width=True,
        )

    # Side-by-side rank comparison
    st.divider()
    st.markdown("### Side-by-Side Rank Comparison")
    merged = greedy_df[["mr_id", "rank"]].rename(columns={"rank": "rank_greedy"}).merge(
        meta_df[["mr_id", "rank"]].rename(columns={"rank": "rank_meta"}), on="mr_id"
    )
    merged["delta"] = merged["rank_greedy"] - merged["rank_meta"]
    merged = merged.merge(mr_lib[["mr_id", "name", "category"]], on="mr_id", how="left")

    fig_cmp = go.Figure()
    for _, row in merged.iterrows():
        color = "#818cf8" if row["delta"] > 0 else "#f87171" if row["delta"] < 0 else "#94a3b8"
        fig_cmp.add_trace(go.Scatter(
            x=["Greedy", "Meta-Clf"],
            y=[row["rank_greedy"], row["rank_meta"]],
            mode="lines+markers+text",
            line=dict(color=color, width=1.5),
            text=[row["mr_id"], row["mr_id"]],
            textposition=["middle left", "middle right"],
            textfont=dict(size=10, color=color),
            showlegend=False,
            hovertemplate=f"<b>{row['mr_id']}</b> — {row.get('name', '')}<br>"
                          f"Greedy rank: {row['rank_greedy']}<br>"
                          f"Meta-clf rank: {row['rank_meta']}<br>"
                          f"Delta: {row['delta']:+d}<extra></extra>",
        ))
    fig_cmp.update_yaxes(autorange="reversed", title="Rank", showgrid=False, tickmode="linear")
    fig_cmp.update_xaxes(showgrid=False)
    fig_cmp.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=550,
        margin=dict(l=80, r=80, t=20, b=20),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)
    spearman = merged[["rank_greedy", "rank_meta"]].corr(method="spearman").iloc[0, 1]
    st.metric("Spearman rank correlation (Greedy ↔ Meta-Clf)", f"{spearman:.4f}")


# ========================================================= Tab 2: Evaluation
with tab_eval:
    eval_data = load_eval()
    km_df, mr_ids, mutant_ids = load_kill_matrix()
    km_bool = km_df.values.astype(bool)
    n_det = int(km_bool.any(axis=0).sum())

    st.markdown("### APFD Comparison")
    st.caption(
        "**Average Percentage of Faults Detected** (higher is better). "
        "Evaluated on all 80 mutants and on the subset detectable by at least one MR."
    )

    apfd = eval_data["apfd"]
    apfd_det = eval_data["apfd_detectable_only"]

    apfd_table = pd.DataFrame({
        "Ordering": ["Random (mean of 30)", "FDR-only", "Greedy (Model A)", "Meta-Classifier (Model B)"],
        "APFD (all 80 mutants)": [
            apfd["random_mean"], apfd["fdr_only"], apfd["greedy"], apfd["meta_classifier"]
        ],
        f"APFD (detectable {n_det})": [
            apfd_det["random_mean"], apfd_det["fdr_only"], apfd_det["greedy"], apfd_det["meta_classifier"]
        ],
    })
    st.dataframe(apfd_table.set_index("Ordering").round(4), use_container_width=True)

    wilcoxon = eval_data.get("wilcoxon", {})
    pval = wilcoxon.get("p_value")
    if pval is not None:
        sig_str = "✓ significant" if wilcoxon.get("significant") else "✗ not significant"
        st.caption(f"Wilcoxon (Greedy vs 30 random trials): p = {pval:.2e} — {sig_str} at α = 0.05")

    st.divider()
    st.markdown("### FD@k — Fault Detection Curve")
    st.caption("Fraction of detectable mutants found by using only the top-k MRs.")

    fd_k = eval_data["fd_at_k"]
    k_vals = list(range(1, len(fd_k["greedy"]) + 1))
    fd_df = pd.DataFrame({
        "k": k_vals,
        "Greedy": fd_k["greedy"],
        "FDR-only": fd_k["fdr_only"],
        "Meta-Classifier": fd_k["meta_classifier"],
        "Random (mean)": fd_k["random"],
    })

    fig_fdk = px.line(
        fd_df.melt("k", var_name="Ordering", value_name="FD@k"),
        x="k", y="FD@k", color="Ordering", markers=True,
        color_discrete_map={
            "Greedy": "#818cf8",
            "FDR-only": "#34d399",
            "Meta-Classifier": "#f472b6",
            "Random (mean)": "#94a3b8",
        },
    )
    fig_fdk.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        height=380,
        xaxis_title="MR budget k",
        yaxis_title="Fraction of detectable mutants found",
        xaxis=dict(dtick=1),
        yaxis=dict(range=[0, 1.05]),
    )
    st.plotly_chart(fig_fdk, use_container_width=True)

    st.divider()
    st.markdown("### Kill Matrix Heatmap")
    st.caption("Rows = MRs (ordered by Greedy rank), Columns = Mutants. Blue = killed.")

    greedy_df = load_greedy()
    greedy_order = greedy_df["mr_id"].tolist()
    mr_id_to_idx = {mr: i for i, mr in enumerate(mr_ids)}
    z = km_bool[[mr_id_to_idx[m] for m in greedy_order]].astype(int)

    by_rank = st.checkbox("Order rows by Greedy rank", value=True)
    row_order = greedy_order if by_rank else mr_ids
    z_display = km_bool[[mr_id_to_idx[m] for m in row_order]].astype(int)

    fig_hm = go.Figure(go.Heatmap(
        z=z_display,
        x=mutant_ids,
        y=row_order,
        showscale=False,
        colorscale=[[0, "rgba(30,30,50,0.4)"], [1, "#818cf8"]],
        hoverongaps=False,
        hovertemplate="MR: %{y}<br>Mutant: %{x}<br>Kill: %{z}<extra></extra>",
    ))
    fig_hm.update_yaxes(autorange="reversed")
    fig_hm.update_xaxes(showticklabels=False, title="Mutants (80)")
    fig_hm.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=480,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # FDR bar chart
    st.divider()
    st.markdown("### Fault Detection Rate per MR")
    mr_scores_df = load_mr_scores()
    mr_lib = load_mr_library()
    fdr_display = mr_scores_df.merge(mr_lib[["mr_id", "name", "category"]], on="mr_id", how="left")
    fdr_display_sorted = fdr_display.sort_values("fdr", ascending=False)

    fig_fdr = px.bar(
        fdr_display_sorted,
        x="mr_id", y="fdr", color="category",
        hover_data=["name", "cost_ms"],
        color_discrete_sequence=px.colors.qualitative.Vivid,
        labels={"fdr": "FDR", "mr_id": ""},
        category_orders={"mr_id": fdr_display_sorted["mr_id"].tolist()},
    )
    fig_fdr.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_fdr, use_container_width=True)


# ===================================================== Tab 3: Rank a New Model
with tab_infer:
    st.markdown("### Rank MRs for a New Model")
    st.caption(
        "Provide your model's **fingerprint** (3 scalar statistics). The meta-classifier will "
        "predict kill probabilities and rank all 20 MRs for your model."
    )

    feature_cols = load_feature_columns()
    mr_lib = load_mr_library()
    clf = load_classifier()

    # Fingerprint format info
    with st.expander("How to generate a fingerprint", expanded=False):
        st.markdown("""
Run the standalone script on your ResNet-18 checkpoint:

```bash
python scripts/compute_fingerprint.py \\
    --checkpoint path/to/your_model.pth \\
    --out my_fingerprint.json
```

This produces a JSON file with these fields:

| Field | Description | Typical range |
|-------|-------------|---------------|
| `avg_weight_magnitude` | Mean of |all weights| | 0.02 – 0.15 |
| `weight_std` | Std of all weights | 0.03 – 0.30 |
| `test_accuracy` | Top-1 accuracy on the 500-image eval subset | 0.40 – 0.95 |

Then upload the file below, or paste the values manually.
""")

    input_mode = st.radio("Input method", ["Upload JSON file", "Enter manually"], horizontal=True)

    avg_weight_magnitude: float | None = None
    weight_std: float | None = None
    test_accuracy: float | None = None
    valid = False

    if input_mode == "Upload JSON file":
        uploaded = st.file_uploader(
            "Upload fingerprint JSON",
            type=["json"],
            help="JSON file produced by scripts/compute_fingerprint.py",
        )
        if uploaded is not None:
            try:
                fp = json.loads(uploaded.read())
                avg_weight_magnitude = float(fp["avg_weight_magnitude"])
                weight_std = float(fp["weight_std"])
                test_accuracy = float(fp["test_accuracy"])
                valid = True

                c1, c2, c3 = st.columns(3)
                c1.metric("avg_weight_magnitude", f"{avg_weight_magnitude:.6f}")
                c2.metric("weight_std", f"{weight_std:.6f}")
                c3.metric("test_accuracy", f"{test_accuracy:.4f}")

            except (KeyError, ValueError, json.JSONDecodeError) as e:
                st.error(
                    f"Invalid fingerprint file: {e}. "
                    "Expected keys: avg_weight_magnitude, weight_std, test_accuracy."
                )

    else:  # Manual entry
        st.markdown("Enter the three values below:")
        c1, c2, c3 = st.columns(3)
        avg_weight_magnitude = c1.number_input(
            "avg_weight_magnitude",
            min_value=0.0, max_value=10.0, value=0.05, step=0.001, format="%.6f",
            help="Mean absolute value of all model weights. Typical range: 0.02 – 0.15",
        )
        weight_std = c2.number_input(
            "weight_std",
            min_value=0.0, max_value=10.0, value=0.08, step=0.001, format="%.6f",
            help="Standard deviation of all model weights. Typical range: 0.03 – 0.30",
        )
        test_accuracy = c3.number_input(
            "test_accuracy",
            min_value=0.0, max_value=1.0, value=0.87, step=0.001, format="%.4f",
            help="Top-1 accuracy on the 500-image CIFAR-10 eval subset (fraction, not %).",
        )
        valid = True

    # Validate
    if valid:
        warnings = []
        if not (0.0 <= avg_weight_magnitude <= 2.0):
            warnings.append(f"avg_weight_magnitude={avg_weight_magnitude:.4f} is outside the expected range [0, 2].")
        if not (0.0 <= weight_std <= 2.0):
            warnings.append(f"weight_std={weight_std:.4f} is outside the expected range [0, 2].")
        if not (0.0 <= test_accuracy <= 1.0):
            warnings.append(f"test_accuracy={test_accuracy:.4f} must be in [0, 1].")
        for w in warnings:
            st.warning(w)

    if valid and not warnings:
        st.divider()
        run_btn = st.button("🔮 Rank MRs for this model", type="primary", use_container_width=True)
        if run_btn:
            with st.spinner("Running meta-classifier inference..."):
                model_features = {
                    "avg_weight_magnitude": avg_weight_magnitude,
                    "weight_std": weight_std,
                    "test_accuracy": test_accuracy,
                }

                # Rebuild rows: MR metadata × mutant profiles
                # Load from mr_library and mr_scores
                mr_lib_loaded = load_mr_library()
                mr_scores_loaded = load_mr_scores()
                km_df_l, _, mutant_ids_l = load_kill_matrix()

                def parse_mutant_local(mid: str) -> dict:
                    op_enc = {"weight_fuzz_low": 1, "weight_fuzz_high": 1,
                              "weight_negate": 2, "weight_zero": 3,
                              "label_corrupt_40": 4, "label_corrupt_55": 4}
                    layer_enc = {"layer1.0": 1, "layer1.1": 1, "layer2": 2,
                                 "layer3": 3, "layer4": 4}
                    if mid.startswith("LC_"):
                        pct = float(mid.split("_")[1].replace("pct", "")) / 100
                        return {"mutation_type_encoded": 4, "mutation_layer_encoded": 0,
                                "mutation_strength": pct}
                    parts = mid.split("_")
                    op_code = parts[0]
                    if op_code == "WF":
                        op_str = f"weight_fuzz_{parts[1]}"
                        layer_key = parts[2]
                    elif op_code == "WN":
                        op_str = "weight_negate"
                        layer_key = parts[1]
                    else:  # WZ
                        op_str = "weight_zero"
                        layer_key = parts[1]
                    return {
                        "mutation_type_encoded": op_enc.get(op_str, 0),
                        "mutation_layer_encoded": layer_enc.get(layer_key, 0),
                        "mutation_strength": 0.0,
                    }

                mutant_profiles_local = pd.DataFrame([parse_mutant_local(m) for m in mutant_ids_l])

                # MR metadata (from mr_library and mr_scores)
                mr_info = mr_lib_loaded.merge(mr_scores_loaded, on="mr_id", how="left")

                rows = []
                for _, mr_row in mr_info.iterrows():
                    base = {
                        "_mr": mr_row["mr_id"],
                        "mr_type_encoded": int(
                            pd.Series({
                                "flip": 1, "rotate": 2, "noise": 3, "brightness": 4,
                                "contrast": 5, "blur": 6, "crop": 7, "salt_pepper": 8,
                                "saturation": 9, "jpeg": 10, "composite": 11,
                                "translation": 12, "scale": 13, "sharpen": 14,
                                "posterize": 15, "shear": 16,
                            }).get(mr_row["mr_type"], 0)
                        ),
                        "mr_magnitude": float(mr_row["magnitude"]),
                        "mr_cost_ms": float(mr_row["cost_ms"]),
                        "mr_is_composite": int(mr_row["is_composite"]),
                        "mr_category_encoded": int(
                            pd.Series({
                                "geometric": 1, "photometric": 2,
                                "noise": 3, "composite": 4,
                            }).get(mr_row["category"], 0)
                        ),
                        "model_avg_weight_magnitude": avg_weight_magnitude,
                        "model_weight_std": weight_std,
                        "model_test_accuracy": test_accuracy,
                    }
                    for mu in mutant_profiles_local.to_dict("records"):
                        rows.append({**base, **mu})

                X = pd.DataFrame(rows)
                X["p"] = clf.predict_proba(X[feature_cols])[:, 1]
                mean_p = X.groupby("_mr")["p"].mean().sort_values(ascending=False)

                result_rows = [
                    {"rank": i + 1, "mr_id": mr_id, "predicted_kill_prob": round(float(p), 4)}
                    for i, (mr_id, p) in enumerate(mean_p.items())
                ]
                result_df = pd.DataFrame(result_rows)
                result_df = result_df.merge(mr_lib_loaded[["mr_id", "name", "category"]], on="mr_id", how="left")

            st.success("Inference complete!")
            st.markdown("#### Predicted MR Ranking for Your Model")

            fig_infer = px.bar(
                result_df,
                x="mr_id", y="predicted_kill_prob",
                color="category",
                hover_data=["name"],
                color_discrete_sequence=px.colors.qualitative.Vivid,
                labels={"predicted_kill_prob": "Predicted Kill Probability", "mr_id": ""},
                category_orders={"mr_id": result_df["mr_id"].tolist()},
            )
            fig_infer.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_infer, use_container_width=True)
            st.dataframe(
                result_df[["rank", "mr_id", "name", "category", "predicted_kill_prob"]]
                .set_index("rank"),
                use_container_width=True,
            )

            st.download_button(
                "⬇ Download ranking as CSV",
                data=result_df.to_csv(index=False),
                file_name="mr_ranking_custom_model.csv",
                mime="text/csv",
            )


# ============================================================== Tab 4: About
with tab_about:
    st.markdown("## About MR-Rank")
    st.markdown("""
**MR-Rank** is a scoring framework for prioritizing Metamorphic Relations (MRs) when testing
deep learning image classifiers. It answers the question:

> *Given a limited testing budget, which MRs should you run first to detect the most model faults?*

### System Overview

| Component | Detail |
|-----------|--------|
| Model under test | ResNet-18 adapted for CIFAR-10 (32×32, modified stem) |
| Evaluation subset | 500 images — stratified, 50 per class, fixed across all experiments |
| MR library | 20 MRs across 4 categories: geometric, photometric, noise, composite |
| Mutant pool | 80 mutants (4 types × 5 layers × 3 runs + 20 label-corruption variants) |
| Kill matrix | 20 × 80 binary matrix — entry (i,j) = 1 if MR_i detects mutant_j |
| Prioritization | Greedy algorithm: Priority = 0.5·FDR + 0.2·(1−NormCost) + 0.3·Diversity |
| Generalisation | Gradient Boosting meta-classifier predicts kill probabilities for a new model |

### MR Categories

- **Geometric**: Horizontal Flip, Shear, Rotate, Translate H/V/Diagonal, Scale Down
- **Photometric**: Brightness ±, Contrast ±, Posterize, Saturation, JPEG Compression
- **Noise**: Gaussian Noise (low/high), Unsharp Mask, Salt & Pepper
- **Composite**: Flip + Noise, Contrast + Saturation Jitter

### Meta-Classifier Features

The classifier uses **11 features** per (MR, mutant) row:
`mr_type_encoded`, `mr_magnitude`, `mr_cost_ms`, `mr_is_composite`, `mr_category_encoded`,
`mutation_type_encoded`, `mutation_layer_encoded`, `mutation_strength`,
`model_avg_weight_magnitude`, `model_weight_std`, `model_test_accuracy`.

---
""")

    col_ver, col_date = st.columns(2)
    col_ver.metric("App version", "1.0.0")
    col_date.metric("Last updated", "2026-10-07")

    st.markdown("---")
    st.markdown("### ⚠️ Known Issues / Limitations")
    st.info("""
**Edit this section as needed before sharing the app.**

1. **MR15 (Scale Down) and MR12 (Unsharp Mask) dominate** — together they cover every detectable
   mutant, so APFD cannot meaningfully separate Greedy from FDR-only ordering on this dataset.

2. **In-sample evaluation** — rankings are evaluated on the same kill matrix they were built from.
   Independent validation against new mutant families is future work.

3. **Model B fingerprint uses zeros by default** — if `scripts/compute_fingerprint.py` was not run
   before `scripts/export_artifacts.py`, the meta-classifier ranking for Model B will be based
   on `avg_weight_magnitude=0, weight_std=0, test_accuracy=0`. Run compute_fingerprint.py on
   `model_B.pth` and re-run export_artifacts.py to get accurate Model B results.

4. **Wilcoxon test is simplified** — a single deterministic APFD is compared against 30 random
   trials by repeating it 30 times. Treat the p-value as an indicative statistic, not a rigorous
   statistical test.

5. **App may sleep after inactivity** — Streamlit Community Cloud free-tier apps sleep after
   ~20 minutes of no traffic. The first wake-up request may take 30–60 seconds.
""")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#475569;font-size:0.8rem'>"
        "MR-Rank · ResNet-18 × CIFAR-10 · "
        "Deployed on Streamlit Community Cloud"
        "</div>",
        unsafe_allow_html=True,
    )
