"""Presentation helpers shared by the dashboard and custom MR workbench."""
from html import escape

import streamlit as st

CATEGORY_COLORS = {
    "geometric": "#91A4B5", "photometric": "#D4A65A",
    "noise": "#B5AA86", "composite": "#A7A0BA",
}


def apply_theme():
    st.markdown("""
<style>
.st-key-main_navigation { position: sticky; top: 3.75rem; z-index: 50; background: #18191B; padding: .6rem 0; border-bottom: 1px solid #383A3D; }
.st-key-main_navigation [role="radiogroup"] { gap: .5rem; flex-wrap: wrap; }
.st-key-main_navigation [role="radiogroup"] > label { border: 1px solid #55585C; border-radius: 5px; padding: .65rem 1rem; margin: 0; cursor: pointer; }
.st-key-main_navigation [role="radiogroup"] > label:has(input:checked) { background: #383125; border-color: #D4A65A; }
.st-key-main_navigation [role="radiogroup"] > label:focus-within { outline: 2px solid #D4A65A; outline-offset: 2px; }
.section-links { display: flex; flex-wrap: wrap; align-items: center; gap: .6rem; margin: .5rem 0 1.2rem; }
.section-links span { color: #B0B0AD; font-size: .85rem; margin-right: .25rem; }
.section-links a { color: #E8E6E1 !important; text-decoration: none; background: #232528; border: 1px solid #55585C; border-radius: 4px; padding: .45rem .75rem; font-size: .9rem; }
.section-links a:hover { border-color: #D4A65A; }
.section-anchor, #page-top { scroll-margin-top: 11rem; }
.back-to-top { display: inline-block; padding: .7rem 0; color: #D4A65A !important; }

[data-testid="stAppViewContainer"] { background: #18191B; }
.block-container { max-width: 1560px; padding-top: 2.4rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #1D1E20; border-right: 1px solid #383A3D; }
h1 { font-size: clamp(1.8rem, 3vw, 2.2rem) !important; letter-spacing: -.045em; font-weight: 700; }
h2 { letter-spacing: -.025em; }
h3 { font-size: 1.18rem !important; letter-spacing: -.015em; }
.eyebrow { color: #D4A65A; font-size: .69rem; font-weight: 700; letter-spacing: .16em; margin-top: .5rem; }
.section-heading { margin: 1.8rem 0 1.4rem; }
.section-heading h2 { font-size: 1.7rem; margin: .3rem 0; padding: 0; }
.section-heading p { color: #B0B0AD; font-size: .94rem; margin: .4rem 0; }
[data-testid="stMetric"] { background: #232528; border: 1px solid #383A3D;  border-radius: 4px; padding: 1rem 1.2rem; height: 100%; }
[data-testid="stMetricLabel"] { color: #B0B0AD; }
[data-testid="stMetricValue"] { font-size: 2rem; letter-spacing: -.03em; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 4px; }
[data-testid="stPlotlyChart"] { border: 1px solid #303235; border-radius: 4px; overflow: hidden; }
.stTabs [data-baseweb="tab-list"] { gap: 1.1rem; border-bottom: 1px solid #383A3D; padding-top: 1rem; }
.stTabs [data-baseweb="tab"] { padding: .7rem .4rem; color: #B0B0AD; }
.stTabs [aria-selected="true"] { color: #D4A65A !important; }
.stTabs [data-baseweb="tab-highlight"] { background: #D4A65A; }
[data-testid="stFileUploader"] { border: 1px dashed #66625A; border-radius: 4px; padding: .7rem; background: #18191B; }
[data-testid="stDataFrame"], [data-testid="stExpander"] { border-radius: 4px; overflow: hidden; }
hr { border-color: #383A3D !important; margin: 1.6rem 0 !important; }
@media (max-width: 760px) {
    .block-container { padding: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: .4rem; overflow-x: auto; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)


def section_header(label, title, description):
    st.markdown(
        f'<div class="section-heading"><div class="eyebrow">{escape(label)}</div>'
        f'<h2>{escape(title)}</h2><p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def plot_chart(fig, **kwargs):
    """Apply shared presentation without altering traces, axes, or hover data."""
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#18191B", plot_bgcolor="#18191B",
        font=dict(family="Arial, sans-serif", color="#D5D3CE", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    st.plotly_chart(fig, **kwargs)
