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
/* Streamlit updates the app's color-scheme immediately in the browser.
   Resolve these colors there instead of freezing them during a Python run. */
.stApp {
    --mr-bg: light-dark(#FAFAF8, #18191B);
    --mr-surface: light-dark(#F0F1EE, #232528);
    --mr-sidebar: light-dark(#F0F1EE, #1D1E20);
    --mr-text: light-dark(#252724, #E8E6E1);
    --mr-muted: light-dark(#595D56, #B0B0AD);
    --mr-border: light-dark(#D4D7CF, #383A3D);
    --mr-outline: light-dark(#858B7E, #55585C);
    --mr-accent: light-dark(#805B20, #D4A65A);
    --mr-selected: light-dark(#EEE4D1, #383125);
}
.st-key-main_navigation { position: sticky; top: 3.75rem; z-index: 50; background: var(--mr-bg); padding: .6rem 0; border-bottom: 1px solid var(--mr-border); }
.st-key-main_navigation [role="radiogroup"] { gap: .5rem; flex-wrap: wrap; }
.st-key-main_navigation [role="radiogroup"] > label { border: 1px solid var(--mr-outline); border-radius: 5px; padding: .65rem 1rem; margin: 0; cursor: pointer; }
.st-key-main_navigation [role="radiogroup"] > label:has(input:checked) { background: var(--mr-selected); border-color: var(--mr-accent); }
.st-key-main_navigation [role="radiogroup"] > label:focus-within { outline: 2px solid var(--mr-accent); outline-offset: 2px; }
.section-links { display: flex; flex-wrap: wrap; align-items: center; gap: .6rem; margin: .5rem 0 1.2rem; }
.section-links span { color: var(--mr-muted); font-size: .94rem; margin-right: .25rem; }
.section-links a { color: var(--mr-text) !important; text-decoration: none; background: var(--mr-surface); border: 1px solid var(--mr-outline); border-radius: 4px; padding: .45rem .75rem; font-size: 1rem; }
.section-links a:hover { border-color: var(--mr-accent); }
.section-anchor, #page-top { scroll-margin-top: 11rem; }
.back-to-top { display: inline-block; padding: .7rem 0; color: var(--mr-accent) !important; }

[data-testid="stAppViewContainer"] { background: var(--mr-bg); }
.block-container { max-width: 1400px; padding-top: 2.4rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: var(--mr-sidebar); border-right: 1px solid var(--mr-border); }
/* Scope text sizing to HTML content so Plotly labels keep their own sizes. */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { font-size: 1.05rem; line-height: 1.55; }
[data-testid="stCaptionContainer"] p { font-size: .95rem; line-height: 1.5; }
[data-testid="stWidgetLabel"] p,
[data-testid="stMetricLabel"] p,
.st-key-main_navigation label p,
.stTabs [data-baseweb="tab"] p { font-size: 1rem; }
h1 { font-size: clamp(1.8rem, 3vw, 2.2rem) !important; letter-spacing: -.045em; font-weight: 700; }
h2 { font-size: 1.65rem; letter-spacing: -.025em; }
h3 { font-size: 1.3rem !important; letter-spacing: -.015em; }
.eyebrow { color: var(--mr-accent); font-size: .8rem; font-weight: 700; letter-spacing: .16em; margin-top: .5rem; }
.section-heading { margin: 1.8rem 0 1.4rem; }
.section-heading h2 { font-size: 1.7rem; margin: .3rem 0; padding: 0; }
.section-heading p { color: var(--mr-muted); font-size: 1.05rem; line-height: 1.55; margin: .4rem 0; }
[data-testid="stMetric"] { background: var(--mr-surface); border: 1px solid var(--mr-border);  border-radius: 4px; padding: 1rem 1.2rem; height: 100%; }
[data-testid="stMetricLabel"] { color: var(--mr-muted); }
[data-testid="stMetricValue"] { font-size: 2rem; letter-spacing: -.03em; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 4px; }
[data-testid="stPlotlyChart"] { border: 1px solid var(--mr-border); border-radius: 4px; overflow: hidden; }
.stTabs [data-baseweb="tab-list"] { gap: 1.1rem; border-bottom: 1px solid var(--mr-border); padding-top: 1rem; }
.stTabs [data-baseweb="tab"] { padding: .7rem .4rem; color: var(--mr-muted); }
.stTabs [aria-selected="true"] { color: var(--mr-accent) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--mr-accent); }
[data-testid="stFileUploader"] { border: 1px dashed var(--mr-outline); border-radius: 4px; padding: .7rem; background: var(--mr-bg); }
[data-testid="stDataFrame"], [data-testid="stExpander"] { border-radius: 4px; overflow: hidden; }
hr { border-color: var(--mr-border) !important; margin: 1.6rem 0 !important; }
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
    # The native Plotly theme responds to appearance changes without a rerun.
    fig.update_layout(
        template="streamlit",
        paper_bgcolor=None, plot_bgcolor=None,
        font=dict(family="Arial, sans-serif", color=None, size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    )
    st.plotly_chart(fig, theme="streamlit", **kwargs)
