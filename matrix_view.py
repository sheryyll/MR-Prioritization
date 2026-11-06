"""Readable summaries and individual outcomes from the existing kill matrix."""
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_theme import plot_chart


def family_columns(matrix):
    names = {"WF": "Weight fuzzing", "WN": "Weight negation",
             "WZ": "Weight zeroing", "LC": "Label corruption"}
    groups = {}
    for column in matrix.columns:
        family = names.get(column.split("_")[0], "Other")
        groups.setdefault(family, []).append(column)
    def natural(value):
        return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]
    return {name: sorted(groups[name], key=natural)
            for name in [*names.values(), "Other"] if name in groups}


def detection_summary(matrix, groups):
    summary = pd.DataFrame(index=matrix.index)
    for family, columns in groups.items():
        counts = matrix[columns].astype(bool).sum(axis=1)
        summary[family] = [f"{n}/{len(columns)} · {n / len(columns):.0%}" for n in counts]
    totals = matrix.astype(bool).sum(axis=1)
    summary["All mutants"] = [f"{n}/{len(matrix.columns)} · {n / len(matrix.columns):.0%}" for n in totals]
    return summary


def render_matrix(matrix, library, greedy_order):
    st.subheader("Detection by mutation family")
    st.caption("Each cell reports detected / total mutants and the detection rate within that family. "
               "The overall rate uses all mutants, including those no relation detects.")
    by_rank = st.checkbox("Order rows by Greedy rank", value=True)
    ordered = matrix.loc[greedy_order if by_rank else matrix.index].astype(bool)
    groups = family_columns(ordered)
    names = library.set_index("mr_id")["name"].to_dict()
    labels = [f"{mr} · {names.get(mr, mr)}" for mr in ordered.index]
    summary = detection_summary(ordered, groups)
    summary.index = pd.Index(labels, name="Metamorphic relation")
    st.dataframe(summary, width="stretch", height=740)

    st.subheader("Individual mutant outcomes")
    st.caption("Amber = detected · Charcoal = not detected. Select a family for larger cells; "
               "hover for the full mutant ID and outcome. M01–M80 are display identifiers.")
    selection = st.selectbox("Mutation family", ["All families", *groups])
    visible = groups if selection == "All families" else {selection: groups[selection]}
    all_columns = [c for cols in groups.values() for c in cols]
    codes = {c: f"M{i + 1:02d}" for i, c in enumerate(all_columns)}
    columns = [c for cols in visible.values() for c in cols]
    values = ordered[columns].astype(int)
    details = [[[column, next(g for g, cols in groups.items() if column in cols),
                 "Detected" if ordered.loc[mr, column] else "Not detected"]
                for column in columns] for mr in ordered.index]
    totals = ordered.sum(axis=1)
    row_labels = [f"{label} ({totals.loc[mr]}/{len(ordered.columns)})"
                  for mr, label in zip(ordered.index, labels)]
    fig = go.Figure(go.Heatmap(
        z=values.values, x=list(range(len(columns))), y=row_labels,
        customdata=details, zmin=0, zmax=1, xgap=1, ygap=2,
        colorscale=[[0, "#303235"], [0.499, "#303235"], [0.5, "#D4A65A"], [1, "#D4A65A"]],
        colorbar=dict(tickvals=[0, 1], ticktext=["Not detected", "Detected"], len=0.25),
        hovertemplate="%{y}<br>%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]}<extra></extra>",
    ))
    offset = 0
    for family, members in visible.items():
        fig.add_annotation(x=offset + (len(members)-1)/2, y=1.04, yref="paper",
                           text=family, showarrow=False, font=dict(size=11))
        if offset:
            fig.add_vline(x=offset-0.5, line_width=3, line_color="#E8E6E1")
        offset += len(members)
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(columns))),
                     ticktext=[codes[c] for c in columns], tickangle=-90,
                     tickfont=dict(size=9), title="Mutant display identifier", showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=True)
    fig.update_layout(height=820, margin=dict(l=10, r=10, t=65, b=70))
    plot_chart(fig, use_container_width=True)
    with st.expander("Mutant identifier key"):
        st.dataframe(pd.DataFrame({"Display ID": [codes[c] for c in columns],
                                  "Mutant ID": columns}), hide_index=True, width="stretch")
