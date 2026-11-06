"""
app.py — MR-Rank Streamlit Community Cloud entry-point.

Tabs:
  1. Rankings         — Greedy + Meta-classifier tables & charts
  2. Evaluation       — APFD table, FD@k curves, kill-matrix heatmap
  3. Run my MRs      — Define, preview, and evaluate custom MRs
  4. About            — Project summary and experimental setup

Hard constraints:
  * NO torch / torchvision import anywhere in this file.
  * All data loaded from app_artifacts/ with st.cache_data / st.cache_resource.
  * No training, mutation generation, or kill-matrix rebuild.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import mr_tab  # noqa: E402
from matrix_view import render_matrix
from ui_theme import apply_theme, section_header, plot_chart, CATEGORY_COLORS

ARTIFACTS = Path(__file__).resolve().parent / "app_artifacts"

st.set_page_config(
    page_title="MR-Rank | Metamorphic Relation Prioritization",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

apply_theme()

st.markdown('<div id="page-top"></div>', unsafe_allow_html=True)
st.markdown("### MR-Rank")
with st.container(key="main_navigation"):
    page = st.radio(
        "Main navigation", ["Rankings", "Evaluation", "Run my MRs", "About"],
        horizontal=True, label_visibility="collapsed", key="workspace_page",
    )
st.caption("ResNet-18 · CIFAR-10 · 20 relations · 80 mutants")
st.caption("Appearance: choose Light or Dark from the top-right ⋮ menu.")


def jump_links(items):
    links = "".join(f'<a href="#{target}" target="_self">{label}</a>' for target, label in items)
    st.markdown(f'<nav class="section-links" aria-label="On this page"><span>Jump to</span>{links}</nav>',
                unsafe_allow_html=True)


def anchor(name):
    st.markdown(f'<div id="{name}" class="section-anchor"></div>', unsafe_allow_html=True)


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


# Metrics
km_df, mr_ids, mutant_ids = load_kill_matrix()
km_bool = km_df.values.astype(bool)
n_detectable = int(km_bool.any(axis=0).sum())
eval_data = load_eval()

#  Tab 4: Run my MRs
if page == "Run my MRs":
    section_header("03 / MR WORKBENCH", "Custom relations", "Work through the existing custom-relation workflow below.")
    jump_links([("define-mrs", "Define MRs"), ("evaluate-mrs", "Evaluate & rank")])
    mr_tab.render()

if page == "Rankings":
    greedy_df = load_greedy()
    meta_df = load_meta_clf_ranking()
    mr_lib = load_mr_library()
    mr_scores = load_mr_scores()

    section_header("01 / RANKINGS", "Prioritization methods",
                   "Inspect each ordering, then compare how individual relations move between methods.")
    jump_links([("greedy-ranking", "Greedy ranking"), ("meta-ranking", "Meta-classifier"),
                ("rank-comparison", "Compare rankings")])
    anchor("greedy-ranking")
    with st.container():
        st.markdown("### Greedy Ranking — Model A")
        st.caption(
            "Priority = **0.5 × FDR** + **0.2 × (1 − NormCost)** + **0.3 × Diversity**  "
            "(see report Eq. 4.1-4.4)"
        )


        # Greedy bar chart
        greedy_display = greedy_df.merge(mr_lib[["mr_id", "name", "category"]], on="mr_id", how="left")
        greedy_display["label"] = greedy_display["mr_id"] + ": " + greedy_display["name"].fillna("")

        fig_greedy = px.bar(
            greedy_display,
            y="label",
            x="priority_score",
            orientation="h",
            color="category",
            hover_data=["name", "fdr", "norm_cost", "diversity_at_selection"],
            color_discrete_map=CATEGORY_COLORS,
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
        fig_greedy.update_layout(height=520, xaxis_title="Priority score", yaxis_title=None)
        fig_greedy.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=greedy_display["label"].tolist())
        plot_chart(fig_greedy, use_container_width=True)

        with st.expander("Full Greedy Ranking Table", expanded=False):
            display_cols = ["rank", "mr_id", "priority_score", "fdr", "norm_cost", "diversity_at_selection"]
            st.dataframe(
                greedy_display[display_cols].set_index("rank").round(4),
                use_container_width=True,
            )

    st.divider()
    anchor("meta-ranking")
    with st.container():
        st.markdown("### Meta-Classifier Ranking — Model B")
        st.caption(
            "Gradient Boosting classifier predicts kill probability per (MR, mutant) pair. "
            "Ranks are averaged over all mutant profiles."
        )

        meta_display = meta_df.merge(mr_lib[["mr_id", "name", "category"]], on="mr_id", how="left")
        meta_display["label"] = meta_display["mr_id"] + ": " + meta_display["name"].fillna("")
        fig_meta = px.bar(
            meta_display,
            y="label",
            x="predicted_kill_prob",
            orientation="h",
            color="category",
            hover_data=["name"],
            color_discrete_map=CATEGORY_COLORS,
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
        fig_meta.update_layout(height=520, xaxis_title="Predicted kill probability", yaxis_title=None)
        fig_meta.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=meta_display["label"].tolist())
        plot_chart(fig_meta, use_container_width=True)

        with st.expander("Full Meta-Classifier Ranking Table", expanded=False):
            st.dataframe(
                meta_display[["rank", "mr_id", "name", "category", "predicted_kill_prob"]]
                .set_index("rank").round(4),
                use_container_width=True,
            )

    # Side-by-side rank comparison
    st.divider()
    anchor("rank-comparison")
    st.markdown("### Side-by-Side Rank Comparison")
    merged = greedy_df[["mr_id", "rank"]].rename(columns={"rank": "rank_greedy"}).merge(
        meta_df[["mr_id", "rank"]].rename(columns={"rank": "rank_meta"}), on="mr_id"
    )
    merged["delta"] = merged["rank_greedy"] - merged["rank_meta"]
    merged = merged.merge(mr_lib[["mr_id", "name", "category"]], on="mr_id", how="left")

    fig_cmp = go.Figure()
    for _, row in merged.iterrows():
        color = "#D4A65A" if row["delta"] > 0 else "#BC8276" if row["delta"] < 0 else "#9A9DA2"
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
        height=460,
        margin=dict(l=80, r=80, t=20, b=20),
    )
    plot_chart(fig_cmp, use_container_width=True)
    spearman = merged[["rank_greedy", "rank_meta"]].corr(method="spearman").iloc[0, 1]
    st.metric("Spearman rank correlation (Greedy ↔ Meta-Clf)", f"{spearman:.4f}")


if page == "Evaluation":
    jump_links([("apfd", "Performance summary"), ("detection-curve", "Detection curve"),
                ("kill-matrix", "Kill matrix"), ("fdr", "Detection rate per MR")])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MRs", len(mr_ids))
    m2.metric("Mutants", len(mutant_ids))
    m3.metric("Detectable mutants", n_detectable)
    greedy_apfd = eval_data["apfd"]["greedy"]
    rand_apfd = eval_data["apfd"]["random_mean"]
    m4.metric("APFD (Greedy)", f"{greedy_apfd:.4f}", delta=f"{greedy_apfd - rand_apfd:+.4f} vs random")

    section_header("02 / EVALUATION", "Detection performance", "Compare detection performance, inspect fault coverage, and explore the contribution of each relation.")
    eval_data = load_eval()
    km_df, mr_ids, mutant_ids = load_kill_matrix()
    km_bool = km_df.values.astype(bool)
    n_det = int(km_bool.any(axis=0).sum())

    anchor("apfd")
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
    anchor("detection-curve")
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
            "Greedy": "#D4A65A",
            "FDR-only": "#8CABA0",
            "Meta-Classifier": "#A7A0BA",
            "Random (mean)": "#9A9DA2",
        },
    )
    fig_fdk.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        height=340,
        xaxis_title="MR budget k",
        yaxis_title="Fraction of detectable mutants found",
        xaxis=dict(dtick=1),
        yaxis=dict(range=[0, 1.05]),
    )
    plot_chart(fig_fdk, use_container_width=True)

    st.divider()
    anchor("kill-matrix")
    render_matrix(km_df, load_mr_library(), load_greedy()["mr_id"].tolist())

    # FDR bar chart
    st.divider()
    anchor("fdr")
    st.markdown("### Fault Detection Rate per MR")
    mr_scores_df = load_mr_scores()
    mr_lib = load_mr_library()
    fdr_display = mr_scores_df.merge(mr_lib[["mr_id", "name", "category"]], on="mr_id", how="left")
    fdr_display_sorted = fdr_display.sort_values("fdr", ascending=False)

    fig_fdr = px.bar(
        fdr_display_sorted,
        x="mr_id", y="fdr", color="category",
        hover_data=["name", "cost_ms"],
        color_discrete_map=CATEGORY_COLORS,
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
    plot_chart(fig_fdr, use_container_width=True)


if page == "About":
    section_header("04 / RESEARCH NOTES", "Study documentation", "The method and experimental setup behind these results.")
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

    st.markdown(
        "<div style='text-align:center;color:var(--mr-muted);font-size:0.9rem'>"
        "MR-Rank · ResNet-18 × CIFAR-10 · "
        "Deployed on Streamlit Community Cloud"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown('<a class="back-to-top" href="#page-top" target="_self">↑ Back to top</a>', unsafe_allow_html=True)
