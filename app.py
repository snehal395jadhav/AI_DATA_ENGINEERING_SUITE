"""
╔══════════════════════════════════════════════════════════════╗
║       AI Data Engineering Suite — Self-Healing Pipeline      ║
║       Powered by OpenRouter AI + Three.js + Streamlit        ║
╚══════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import os
import sys
import html
from io import StringIO, BytesIO
import base64
from datetime import datetime
import zipfile
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from modules.self_healing import SelfHealingPipeline
from modules.data_quality import DataQualityChecker
from modules.ai_agent import DataAIAgent, AVAILABLE_MODELS
from modules.vector_store import VectorStore
from modules import ml_analytics, agent_graph
from modules import security, monitoring, analytics
from modules import data_hub as dh
from modules import analyst
from modules import sqlite_store
import config
import plotly.io as pio

# ─── Professional global Plotly theme ──────────────────────────────────────────
PLOT_FONT = "Inter, 'Segoe UI', sans-serif"
COLORWAY = ["#00d4ff", "#7c3aed", "#22c55e", "#f59e0b", "#ec4899", "#38bdf8", "#a855f7", "#10b981"]
pio.templates["datasuite"] = go.layout.Template(
    layout=dict(
        font=dict(family=PLOT_FONT, color="#94a3b8", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(font=dict(family=PLOT_FONT, size=12), bgcolor="#0f1629", bordercolor="#00d4ff"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.12)", zerolinecolor="rgba(148,163,184,0.2)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, x=0),
        title=dict(font=dict(family="'Space Grotesk', Inter, sans-serif", size=15, color="#e2e8f0")),
    )
)
pio.templates.default = "plotly_dark+datasuite"


def style_fig(fig, height: int = 300):
    """Apply the house style + height to any Plotly figure."""
    fig.update_layout(template="plotly_dark+datasuite", height=height,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def render_chart_gallery(df, max_charts: int = 9):
    """Auto-build many chart types from a dataframe and render them in a 2-col grid.
    Used by 'Analyze dataset' in AI Agent Chat and the Data Explorer."""
    numf = df.apply(pd.to_numeric, errors="coerce")
    num_cols = [c for c in df.columns if numf[c].notna().mean() > 0.6]
    cat_cols = [c for c in df.select_dtypes(include="object").columns if 1 < df[c].nunique() < 25]
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    figs = []  # (title, figure)

    if date_cols and num_cols:
        try:
            t = pd.DataFrame({"d": pd.to_datetime(df[date_cols[0]], errors="coerce"),
                              "v": numf[num_cols[0]]}).dropna().sort_values("d")
            if len(t) > 1:
                figs.append(("Trend", px.area(t, x="d", y="v", title=f"{num_cols[0]} over time")))
        except Exception:
            pass
    for c in num_cols[:3]:
        figs.append((f"Dist {c}", px.histogram(numf, x=c, nbins=25, title=f"Distribution: {c}")))
    if num_cols:
        figs.append(("Box", px.box(numf, y=num_cols[:4], title="Box plots")))
    if len(num_cols) > 1:
        corr = numf[num_cols].corr()
        figs.append(("Correlation", go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                                          colorscale="RdBu", zmid=0))))
        figs.append((f"Scatter", px.scatter(numf, x=num_cols[0], y=num_cols[1],
                                             title=f"{num_cols[0]} vs {num_cols[1]}")))
    for c in cat_cols[:2]:
        vc = df[c].value_counts().head(8)
        figs.append((f"Bar {c}", px.bar(x=vc.index.astype(str), y=vc.values, title=f"Top {c}")))
    if cat_cols:
        vc = df[cat_cols[0]].value_counts().head(6)
        figs.append((f"Share {cat_cols[0]}", px.pie(values=vc.values, names=vc.index.astype(str),
                                                     hole=0.5, title=f"Share: {cat_cols[0]}")))

    figs = figs[:max_charts]
    if not figs:
        st.info("Not enough numeric/categorical columns to build charts.")
        return
    cols = st.columns(2)
    for i, (_, fig) in enumerate(figs):
        with cols[i % 2]:
            style_fig(fig, 300)
            st.plotly_chart(fig, use_container_width=True)


def mini_gauge(value, title, height=200, ref=80, suffix="", vmax=100):
    """A compact Plotly gauge used across engineering/monitoring/quality pages."""
    try:
        color = score_color(value) if value <= 100 else "#00d4ff"
    except Exception:
        color = "#00d4ff"
    g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"size": 26}},
        gauge={
            "axis": {"range": [0, vmax], "tickcolor": "#94a3b8"},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, vmax * 0.6], "color": "rgba(239,68,68,0.15)"},
                {"range": [vmax * 0.6, vmax * 0.8], "color": "rgba(245,158,11,0.15)"},
                {"range": [vmax * 0.8, vmax], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {"line": {"color": "#e2e8f0", "width": 2}, "value": ref},
        },
        title={"text": title, "font": {"size": 13}},
    ))
    style_fig(g, height)
    return g


def render_ai_brief(prompt, *, key, system=None, fallback=None,
                    button_label="🤖 Generate AI insights"):
    """Render an AI insight panel (summary / risks / actions) using call_openrouter_json.
    Token-light, reusable across Engineer / Monitoring / Quality / Self-Healing pages.
    Result is cached in session_state so it survives reruns."""
    sk = f"_aibrief_{key}"
    if st.button(button_label, key=f"btn_{sk}", use_container_width=True):
        if not st.session_state.api_key:
            st.session_state[sk] = {
                "summary": fallback or "Add an OpenRouter API key in `.env` to enable AI insights. "
                                       "All metrics above are still computed locally.",
                "risks": [], "actions": []}
        else:
            with st.spinner("🧠 AI analyzing your pipeline…"):
                data = call_openrouter_json(
                    prompt,
                    system or ('You are a senior data engineer. Respond ONLY with JSON '
                               '{"summary":str,"risks":[str],"actions":[str]} — concise, specific, no markdown.'))
            st.session_state[sk] = data or {
                "summary": "The model did not return a parseable result — try again.",
                "risks": [], "actions": []}
    data = st.session_state.get(sk)
    if not data:
        return
    st.markdown(f'<div class="data-card"><b>🧠 AI Summary</b><br><br>{data.get("summary","")}</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**⚠️ Key risks**")
        for r in (data.get("risks") or ["No critical risks flagged."]):
            st.markdown(f'<div class="alert-warning">{r}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("**✅ Recommended actions**")
        for a in (data.get("actions") or ["No actions required."]):
            st.markdown(f'<div class="alert-ok">{a}</div>', unsafe_allow_html=True)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Engineering Suite",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Theme & CSS ──────────────────────────────────────────────────────────────
def get_css(dark: bool) -> str:
    if dark:
        bg, bg2, card, text, accent, border = (
            "#0a0e1a", "#0f1629", "#111827",
            "#e2e8f0", "#00d4ff", "rgba(0,212,255,0.2)"
        )
        metric_bg = "linear-gradient(135deg,#1a2035,#0d1526)"
        sidebar_bg = "#0c1220"
    else:
        bg, bg2, card, text, accent, border = (
            "#f0f4ff", "#e8edf8", "#ffffff",
            "#1a202c", "#2563eb", "rgba(37,99,235,0.2)"
        )
        metric_bg = "linear-gradient(135deg,#dbeafe,#eff6ff)"
        sidebar_bg = "#e8edf8"

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

.stApp {{ background:
    radial-gradient(1200px 600px at 80% -10%, {accent}14, transparent 60%),
    radial-gradient(900px 500px at -10% 10%, #7c3aed14, transparent 55%),
    {bg};
    color: {text}; font-family: 'Inter', sans-serif; letter-spacing: 0.1px; }}
h1, h2, h3, h4, .hero-title, .gradient-text {{ font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -0.4px; }}
code, .fix-item, pre, .stCodeBlock {{ font-family: 'JetBrains Mono', monospace !important; }}
section[data-testid="stSidebar"] {{ background: {sidebar_bg} !important; border-right: 1px solid {border}; }}
section[data-testid="stSidebar"] * {{ color: {text} !important; }}

/* Cards */
.data-card {{
    background: {card};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 20px 24px;
    margin: 8px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.data-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.25); }}

/* Metric Cards */
.metric-card {{
    background: {metric_bg};
    border: 1px solid {border};
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
}}
.metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {accent}, #7c3aed);
}}
.metric-value {{ font-size: 2rem; font-weight: 800; color: {accent}; word-break: break-word; }}
.metric-label {{ font-size: 0.78rem; color: {"#94a3b8" if dark else "#64748b"}; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }}
.metric-delta {{ font-size: 0.8rem; margin-top: 4px; }}

/* Alert Boxes */
.alert-critical {{ background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.4); border-radius: 10px; padding: 12px 16px; margin: 6px 0; }}
.alert-warning  {{ background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.4); border-radius: 10px; padding: 12px 16px; margin: 6px 0; }}
.alert-ok       {{ background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.4); border-radius: 10px; padding: 12px 16px; margin: 6px 0; }}
.alert-info     {{ background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.4); border-radius: 10px; padding: 12px 16px; margin: 6px 0; }}

/* Fix Items */
.fix-item {{ background: rgba(34,197,94,0.08); border-left: 3px solid #22c55e; border-radius: 8px; padding: 10px 14px; margin: 5px 0; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; }}
.fix-item.high {{ border-left-color: #ef4444; background: rgba(239,68,68,0.08); }}
.fix-item.medium {{ border-left-color: #f59e0b; background: rgba(245,158,11,0.08); }}

/* Score Ring */
.score-container {{ display: flex; justify-content: center; align-items: center; padding: 20px; }}
.score-badge {{
    width: 120px; height: 120px;
    border-radius: 50%;
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
    font-weight: 800;
    background: conic-gradient({accent} var(--pct), {border} 0);
    box-shadow: 0 0 30px {border};
}}

/* Chat */
.chat-message {{ padding: 12px 16px; border-radius: 12px; margin: 8px 0; max-width: 85%; }}
.chat-user {{ background: rgba(37,99,235,0.15); border: 1px solid rgba(37,99,235,0.3); margin-left: auto; text-align: right; }}
.chat-ai {{ background: {card}; border: 1px solid {border}; }}
.chat-ai-label {{ color: {accent}; font-weight: 600; font-size: 0.8rem; margin-bottom: 4px; }}

/* Buttons */
div.stButton > button {{
    background: linear-gradient(135deg, {accent}, #7c3aed);
    color: white !important;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 20px;
    transition: all 0.2s;
    font-family: 'Inter', sans-serif;
}}
div.stButton > button:hover {{ opacity: 0.9; transform: translateY(-1px); box-shadow: 0 4px 15px rgba(0,0,0,0.3); }}

/* Progress bars */
.stProgress > div > div {{ background: linear-gradient(90deg, {accent}, #7c3aed); border-radius: 10px; }}

/* Tabs */
div[data-baseweb="tab-list"] {{ background: {bg2}; border-radius: 12px; padding: 4px; border: 1px solid {border}; }}
div[data-baseweb="tab"] {{ border-radius: 8px; font-weight: 500; }}

/* Sidebar Logo */
.sidebar-logo {{
    text-align: center;
    padding: 20px 10px;
    border-bottom: 1px solid {border};
    margin-bottom: 16px;
}}
.sidebar-logo h2 {{ font-size: 1.1rem; font-weight: 800; color: {accent}; letter-spacing: -0.5px; }}
.sidebar-logo p {{ font-size: 0.7rem; color: {"#64748b" if not dark else "#475569"}; }}

/* Landing sections */
.hero-title {{ font-size: 3rem; font-weight: 900; line-height: 1.15; letter-spacing: -1px; }}
.hero-sub {{ font-size: 1.1rem; color: {"#94a3b8" if dark else "#64748b"}; line-height: 1.6; }}
.gradient-text {{ background: linear-gradient(135deg, {accent}, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.feature-chip {{
    display: inline-block;
    background: {border};
    border: 1px solid {border};
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 3px;
    color: {accent};
}}

/* DataFrames */
.dataframe {{ font-size: 0.8rem; }}
div[data-testid="stDataFrameResizable"] {{ border-radius: 12px; overflow: hidden; border: 1px solid {border}; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {bg}; }}
::-webkit-scrollbar-thumb {{ background: {border}; border-radius: 3px; }}

/* Code blocks */
.stCodeBlock {{ border-radius: 10px !important; }}

/* Layout reset — keep the header (it holds the sidebar-expand button on
   Streamlit 1.4x+), just strip its decoration so the toggle stays usable. */
[data-testid="stDecoration"] {{ display: none !important; }}
header[data-testid="stHeader"] {{
    background: transparent !important; box-shadow: none !important;
}}
[data-testid="stAppViewContainer"] > .main .block-container {{
    max-width: 1280px;
    padding-top: 0 !important;
    padding-bottom: 1.25rem;
}}

/* Animations */
@keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:.5; }} }}
@keyframes slideIn {{ from {{ transform:translateY(20px);opacity:0; }} to {{ transform:translateY(0);opacity:1; }} }}
.animate-in {{ animation: slideIn 0.4s ease forwards; }}

/* Badge */
.status-badge {{
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 0.7rem; font-weight: 600;
}}
.badge-green {{ background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }}
.badge-red {{ background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); }}
.badge-yellow {{ background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid rgba(245,158,11,0.3); }}
.badge-blue {{ background: rgba(59,130,246,0.15); color: #3b82f6; border: 1px solid rgba(59,130,246,0.3); }}

/* ─── Advanced Sidebar ─────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {sidebar_bg}, {bg}) !important;
    border-right: 1px solid {border};
    box-shadow: inset -1px 0 0 rgba(255,255,255,0.03);
}}
.brand-wrap {{
    text-align:center; padding: 18px 12px 14px;
    border-bottom: 1px solid {border}; margin-bottom: 14px;
    position: relative;
}}
.brand-orb {{
    width:54px; height:54px; margin:0 auto 8px; border-radius:16px;
    display:flex; align-items:center; justify-content:center;
    background:
      url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2036%2036'%3E%3Crect%20x='8.5'%20y='18'%20width='4.3'%20height='9'%20rx='1.6'%20fill='white'/%3E%3Crect%20x='15.85'%20y='13'%20width='4.3'%20height='14'%20rx='1.6'%20fill='white'/%3E%3Crect%20x='23.2'%20y='9'%20width='4.3'%20height='18'%20rx='1.6'%20fill='white'/%3E%3C/svg%3E") center/58% no-repeat,
      {"linear-gradient(135deg, #22d3ee, #6366f1, #a855f7)"};
    box-shadow: {"0 8px 24px rgba(99,102,241,0.40), inset 0 0 0 1px rgba(255,255,255,0.18)"};
    animation: floaty 4s ease-in-out infinite;
}}
/* Old "NS" monogram removed — bar-chart mark is drawn via the background above */
.brand-orb::before {{ content: none; }}
.brand-orb-wordmark {{ font-size: 0 !important; color: transparent !important; }}
@keyframes floaty {{ 0%,100%{{transform:translateY(0)}} 50%{{transform:translateY(-5px)}} }}
.brand-name {{ font-family:'Space Grotesk',sans-serif; font-size:1.15rem; font-weight:700;
    background: linear-gradient(135deg,{accent},#a855f7,#ec4899);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.brand-tag {{ font-size:0.66rem; letter-spacing:2px; text-transform:uppercase;
    color: {"#64748b" if not dark else "#475569"}; margin-top:2px; }}
.side-label {{ font-size:0.68rem; font-weight:700; letter-spacing:1.5px; text-transform:uppercase;
    color:{"#64748b" if not dark else "#64748b"}; margin:10px 2px 4px; }}

/* Keep the sidebar OPEN (»») control clearly visible when collapsed.
   Cover every Streamlit version's test id so it always shows. */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"],
[data-testid="collapsedControl"] {{
    display: flex !important; opacity: 1 !important; visibility: visible !important;
    position: fixed !important; top: 12px !important; left: 12px !important; z-index: 999999 !important;
    background: linear-gradient(135deg, {accent}, #7c3aed) !important;
    border-radius: 10px !important; padding: 6px 8px !important;
    box-shadow: 0 6px 18px {accent}66 !important;
}}
[data-testid="stSidebarCollapsedControl"] *,
[data-testid="stExpandSidebarButton"] *,
[data-testid="collapsedControl"] * {{
    color:#ffffff !important; fill:#ffffff !important;
}}
[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="collapsedControl"] svg {{ width:1.45rem !important; height:1.45rem !important; }}

.side-nav {{
    display: grid;
    gap: 8px;
}}
.side-nav-active {{
    background: linear-gradient(135deg, {accent}20, #7c3aed20);
    border: 1px solid {accent}55;
    border-radius: 12px;
    padding: 10px 12px;
    color: {accent};
    font-weight: 700;
    box-shadow: 0 8px 20px {accent}14;
}}
.side-nav-note {{
    font-size: 0.76rem;
    color: {"#94a3b8" if dark else "#64748b"};
    margin-bottom: 8px;
}}
section[data-testid="stSidebar"] div.stButton > button {{
    width: 100%;
    justify-content: flex-start;
    gap: 8px;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.24);
    border: 1px solid rgba(148,163,184,0.16);
    color: {text} !important;
    box-shadow: none;
}}
section[data-testid="stSidebar"] div.stButton > button:hover {{
    background: linear-gradient(135deg, {accent}1a, #7c3aed18);
    border-color: {accent}55;
    color: {accent} !important;
    transform: translateX(3px);
}}

/* Glassy data cards */
.data-card {{ backdrop-filter: blur(6px); }}

/* Hero CTA button */
.cta-row div.stButton > button {{
    font-size: 1.02rem; padding: 14px 30px; border-radius: 14px;
    box-shadow: 0 10px 30px {accent}44;
}}

/* Landing split-hero */
.land-eyebrow {{
    display:inline-block; font-family:'JetBrains Mono',monospace; font-size:0.72rem;
    letter-spacing:3px; text-transform:uppercase; color:{accent};
    border:1px solid {accent}55; border-radius:20px; padding:4px 14px; margin-bottom:10px;
    background:{accent}12;
}}
.nav-brand {{
    display:flex; align-items:center; gap:12px; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.15rem;
    letter-spacing:-0.3px; padding:6px 2px; white-space:nowrap;
}}
.nav-brand-mark {{
    width: 36px;
    height: 36px;
    border-radius: 11px;
    background: linear-gradient(135deg, #22d3ee 0%, #6366f1 55%, #a855f7 100%);
    position: relative;
    display:flex; align-items:center; justify-content:center;
    box-shadow: 0 6px 18px rgba(99,102,241,0.45), inset 0 0 0 1px rgba(255,255,255,0.18);
    overflow: hidden;
}}
.nav-brand-mark::after {{
    /* subtle shine */
    content:""; position:absolute; top:-40%; left:-10%; width:60%; height:160%;
    background: linear-gradient(120deg, rgba(255,255,255,0.45), rgba(255,255,255,0));
    transform: rotate(18deg);
}}
.nav-brand-mark::before {{
    content: "NS";
    position: relative; z-index:1;
    font-family:'Space Grotesk',sans-serif;
    font-size:13px;
    font-weight:800;
    color:#ffffff;
    letter-spacing:0.5px;
    text-shadow: 0 1px 2px rgba(0,0,0,0.25);
}}
.nav-brand-text {{
    background: linear-gradient(135deg,{accent},#a855f7);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.top-nav-shell {{
    position: sticky;
    top: 0;
    z-index: 10;
    backdrop-filter: blur(16px);
}}
.site-footer {{
    text-align:center; padding:22px 10px 8px; margin-top:26px;
    border-top:1px solid {border}; color:{"#94a3b8" if dark else "#64748b"}; font-size:0.8rem;
}}
.site-footer b {{ color:{accent}; }}
.contact-pill {{
    display:inline-flex; align-items:center; gap:8px; margin:6px;
    background:{card}; border:1px solid {border}; border-radius:30px; padding:8px 16px;
    font-size:0.85rem; color:{text};
}}
/* Top navigation bar: bordered container becomes the bar */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: linear-gradient(135deg, {card}, {bg2});
    border: 1px solid {border} !important; border-radius: 14px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.18);
}}
[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button {{
    background: {card} !important;
    color: {text} !important;
    border: 1px solid {border} !important;
    text-align: center; justify-content:center;
    font-weight: 600; font-size: 0.82rem; padding: 8px 6px; border-radius: 9px;
    box-shadow: none;
}}
[data-testid="stVerticalBlockBorderWrapper"] div.stButton > button:hover {{
    background: linear-gradient(135deg, {accent}26, #7c3aed26) !important;
    border-color: {accent}77 !important;
    transform: translateY(-2px);
    color: {accent} !important;
    box-shadow: 0 6px 16px {accent}22;
}}
</style>
"""

# ─── Session State Init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "dark_mode": True,
        "df": None,
        "df_fixed": None,
        "df_name": "",
        "issues": None,
        "fixes": None,
        "alerts": None,
        "quality_report": None,
        "pipeline": SelfHealingPipeline(),
        "checker": DataQualityChecker(),
        "api_key": config.get_api_key(),          # hidden, loaded from config (never shown in UI)
        "selected_model": config.DEFAULT_CHAT_MODEL,
        "chat_history": [],
        "agent": None,
        "page": "Landing",
        "run_complete": False,
        "semantic_results": None,
        "ai_report": None,
        "vector_store": None,
        "vs_status": "",
        "ml_predict": None,
        "ml_anomaly": None,
        "agent_run": None,
        "rag_history": [],
        "land_view": "Home",
        "healed_csv": None,
        "ai_fix_advice": None,
        "_loaded_upload": None,
        "auth": None,            # logged-in user dict {username, name, role} or None
        "audit": [],             # in-session audit log
        "_last_audit_page": None,
        "mask_reveal": False,    # admin toggle to reveal masked PII
        "sqlite_dataset_id": None,
        "sqlite_meta": None,
        "df_profiling_result": None,
        "df_quality_checks": None,
        "df_quality_score": None,
        "cleaning_log": [],
        "transformation_log": [],
        "df_schema_info": None,
        "insights_narrative": None,
        "hub_active_tab": 0,
        "signup_done": False,
        # ─── Analyst upgrade keys ───────────────────────────────────────────────
        "analyst_period": "Monthly",
        "analyst_date_from": None,
        "analyst_date_to": None,
        "df_dashboard": None,
        "exec_report_type": "Sales Performance Report",
        "ml_clusters": None,
        "ts_decomposition": None,
        "segment_ai_names": None,
        "analyst_kpis": None,
        "financial_commentary": None,
        "export_commentary": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
sqlite_store.init_db()
dark = st.session_state.dark_mode
st.markdown(get_css(dark), unsafe_allow_html=True)

# Full-screen landing → hide the sidebar entirely for a clean first impression
if st.session_state.page == "Landing":
    st.markdown("""<style>
        section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display:none !important; }
        header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
            display:none !important; height:0 !important;
        }
        .block-container,
        [data-testid="stMainBlockContainer"],
        [data-testid="stAppViewContainer"] .main .block-container {
            padding-top: 0.5rem !important; margin-top: 0 !important;
            max-width: 100% !important; padding-left: 2.2rem !important; padding-right: 2.2rem !important;
        }
        [data-testid="stAppViewContainer"], section.main { padding-top: 0 !important; }
    </style>""", unsafe_allow_html=True)

# ─── Brand logo (clean inline-SVG bar-chart mark) ──────────────────────────────
BRAND_LOGO_HTML = (
    '<div class="nav-brand">'
    '<svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<defs><linearGradient id="nsLogo" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#22d3ee"/><stop offset="0.55" stop-color="#6366f1"/>'
    '<stop offset="1" stop-color="#a855f7"/></linearGradient></defs>'
    '<rect x="1" y="1" width="34" height="34" rx="10" fill="url(#nsLogo)"/>'
    '<rect x="8.5" y="18" width="4.3" height="9" rx="1.6" fill="#ffffff" fill-opacity="0.95"/>'
    '<rect x="15.85" y="13" width="4.3" height="14" rx="1.6" fill="#ffffff" fill-opacity="0.95"/>'
    '<rect x="23.2" y="9" width="4.3" height="18" rx="1.6" fill="#ffffff" fill-opacity="0.95"/>'
    '</svg>'
    '<span class="nav-brand-text">Navneet Data Studio</span>'
    '</div>'
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_three_js(html_file: str, height: int = 350) -> None:
    path = os.path.join(os.path.dirname(__file__), "components", html_file)
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        st.components.v1.html(content, height=height, scrolling=False)
    except Exception:
        # Graceful fallback if the components API changes/is removed.
        st.caption(" 3D visualization unavailable in this environment.")

def render_three_analytics(scene_id: str, payload: dict, height: int = 360) -> None:
    payload_json = json.dumps(payload).replace("</", "<\\/")
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; }}
      body {{
        overflow: hidden;
        background: radial-gradient(circle at top, rgba(0,212,255,0.08), transparent 45%), transparent;
        font-family: Inter, sans-serif;
      }}
      #scene {{ width: 100vw; height: 100vh; }}
      #badge {{
        position: absolute; top: 14px; left: 14px; z-index: 2;
        padding: 8px 12px; border-radius: 999px;
        background: rgba(10,14,26,0.72); border: 1px solid rgba(0,212,255,0.22);
        color: #dbeafe; font-size: 11px; letter-spacing: 1.6px; text-transform: uppercase;
        backdrop-filter: blur(10px);
      }}
    </style>
    </head>
    <body>
    <div id="badge">{payload.get("title", "3D Analytics")}</div>
    <div id="scene"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    const payload = {payload_json};
    const mount = document.getElementById("scene");
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(52, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 7, 20);
    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    mount.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0x224466, 1.9));
    const keyLight = new THREE.PointLight(0x18b6ff, 2.1, 120);
    keyLight.position.set(10, 16, 18);
    scene.add(keyLight);
    const fillLight = new THREE.PointLight(0xa855f7, 1.4, 100);
    fillLight.position.set(-14, 10, -6);
    scene.add(fillLight);

    const grid = new THREE.GridHelper(24, 24, 0x145374, 0x11253d);
    grid.position.y = -2.8;
    scene.add(grid);

    const root = new THREE.Group();
    scene.add(root);

    function addBars(data) {{
      const spacing = 2.1;
      const origin = -((data.length - 1) * spacing) / 2;
      data.forEach((item, index) => {{
        const height = Math.max(item.value, 0.35);
        const geometry = new THREE.BoxGeometry(1.15, height, 1.15);
        const material = new THREE.MeshPhongMaterial({{
          color: item.color || 0x18b6ff,
          emissive: item.color || 0x18b6ff,
          emissiveIntensity: 0.25,
          transparent: true,
          opacity: 0.95
        }});
        const bar = new THREE.Mesh(geometry, material);
        bar.position.set(origin + index * spacing, height / 2 - 2.8, 0);
        root.add(bar);

        const cap = new THREE.Mesh(
          new THREE.CylinderGeometry(0.58, 0.58, 0.1, 24),
          new THREE.MeshBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.12 }})
        );
        cap.position.set(bar.position.x, bar.position.y + height / 2 + 0.05, 0);
        root.add(cap);
      }});
    }}

    function addScatter(data) {{
      const group = new THREE.Group();
      data.forEach((item) => {{
        const geometry = new THREE.SphereGeometry(item.size || 0.22, 18, 18);
        const material = new THREE.MeshPhongMaterial({{
          color: item.color || 0x22c55e,
          emissive: item.color || 0x22c55e,
          emissiveIntensity: 0.35
        }});
        const point = new THREE.Mesh(geometry, material);
        point.position.set(item.x, item.y, item.z);
        group.add(point);
      }});
      root.add(group);
    }}

    if (payload.type === "bars") {{
      addBars(payload.data || []);
    }} else {{
      addScatter(payload.data || []);
    }}

    const starsGeometry = new THREE.BufferGeometry();
    const starPositions = new Float32Array(1200);
    for (let i = 0; i < starPositions.length; i++) {{
      starPositions[i] = (Math.random() - 0.5) * 60;
    }}
    starsGeometry.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
    const stars = new THREE.Points(starsGeometry, new THREE.PointsMaterial({{ color: 0x264766, size: 0.08 }}));
    scene.add(stars);

    let isDragging = false, prevX = 0, prevY = 0, rotX = -0.18, rotY = 0.35;
    renderer.domElement.addEventListener("mousedown", (event) => {{
      isDragging = true;
      prevX = event.clientX;
      prevY = event.clientY;
    }});
    window.addEventListener("mouseup", () => isDragging = false);
    window.addEventListener("mousemove", (event) => {{
      if (!isDragging) return;
      rotY += (event.clientX - prevX) * 0.01;
      rotX += (event.clientY - prevY) * 0.01;
      prevX = event.clientX;
      prevY = event.clientY;
    }});
    renderer.domElement.addEventListener("wheel", (event) => {{
      camera.position.z = Math.min(30, Math.max(10, camera.position.z + event.deltaY * 0.01));
    }});

    let tick = 0;
    function animate() {{
      requestAnimationFrame(animate);
      tick += 0.01;
      root.rotation.y = rotY + tick * 0.12;
      root.rotation.x = rotX;
      renderer.render(scene, camera);
    }}
    animate();

    window.addEventListener("resize", () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});
    </script>
    </body>
    </html>
    """
    st.components.v1.html(html, height=height, scrolling=False)

def score_color(score: float) -> str:
    if score >= 80: return "#22c55e"
    if score >= 60: return "#f59e0b"
    return "#ef4444"

def score_emoji(score: float) -> str:
    if score >= 80: return ""
    if score >= 60: return ""
    return ""

def make_zip_download():
    """Create a ZIP of current session results."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        if st.session_state.df is not None:
            zf.writestr("original_data.csv", st.session_state.df.to_csv(index=False))
        if st.session_state.df_fixed is not None:
            zf.writestr("healed_data.csv", st.session_state.df_fixed.to_csv(index=False))
        if st.session_state.quality_report is not None:
            zf.writestr("quality_report.json", json.dumps(st.session_state.quality_report, indent=2, default=str))
        if st.session_state.fixes:
            zf.writestr("healing_log.json", json.dumps(st.session_state.fixes, indent=2))
        if st.session_state.alerts:
            zf.writestr("alerts.json", json.dumps(st.session_state.alerts, indent=2))
        if st.session_state.get("ai_report"):
            zf.writestr("executive_report.md", st.session_state.ai_report)
        if st.session_state.get("semantic_results") is not None:
            zf.writestr("semantic_matches.csv", st.session_state.semantic_results.to_csv(index=False))
    buf.seek(0)
    return buf

def load_sample_data(name: str) -> pd.DataFrame:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    files = {
        "Walmart Sales": "walmart_sales.csv",
        "Walmart Inventory": "walmart_inventory.csv",
        "Walmart Customers": "walmart_customers.csv",
        "Navneet Products": "navneet_products.csv",
        "Navneet Sales": "navneet_sales.csv",
        "Navneet Financials": "navneet_financials.csv",
    }
    path = os.path.join(data_dir, files[name])
    return pd.read_csv(path)

def apply_dataset(df: pd.DataFrame, name: str) -> None:
    """Set the active dataset and clear every derived cache."""
    st.session_state.df = df
    st.session_state.df_name = name
    for k in ["df_fixed", "issues", "fixes", "alerts", "quality_report",
              "vector_store", "ml_predict", "ml_anomaly", "agent_run",
              "semantic_results", "ai_report", "healed_csv"]:
        st.session_state[k] = None
    st.session_state.run_complete = False
    st.session_state.vs_status = ""
    st.session_state.rag_history = []
    st.session_state.pipeline = SelfHealingPipeline()
    # reset Data Hub AI caches when the dataset changes
    for k in ["df_profiling_result", "df_quality_checks", "df_quality_score",
              "df_schema_info", "insights_narrative"]:
        st.session_state[k] = None
    st.session_state.cleaning_log = []
    st.session_state.transformation_log = []


def call_openrouter_json(prompt: str,
                         system: str = "You are a senior data analyst. Always respond with valid JSON only.") -> "dict | None":
    """Call OpenRouter and parse the first JSON object from the reply. None on failure."""
    import requests
    key = st.session_state.api_key
    model = st.session_state.selected_model
    if not key:
        return None
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": config.APP_REFERER, "X-Title": config.APP_TITLE}
    payload = {"model": model, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ], "temperature": 0.2}
    try:
        r = requests.post(config.OPENROUTER_CHAT_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        start = text.find("{"); end = text.rfind("}") + 1
        return json.loads(text[start:end]) if start != -1 else None
    except Exception:
        return None


def set_active_sqlite_dataset(dataset_id: int | None, meta: dict | None = None) -> None:
    st.session_state.sqlite_dataset_id = dataset_id
    st.session_state.sqlite_meta = meta


def attempt_login(username: str, password: str) -> tuple[bool, str]:
    user = sqlite_store.authenticate_user(username, password) or security.authenticate(username, password)
    if user:
        st.session_state.auth = user
        log_action("LOGIN", f"role={user['role']}")
        st.session_state.page = security.DEFAULT_PAGE.get(user["role"], "Data Hub")
        return True, ""
    log_action("LOGIN_FAILED", f"user={username}")
    return False, "Invalid credentials. Try a demo account below or create a new account."

def render_metric(label, value, delta=None, icon=""):
    delta_html = f'<div class="metric-delta" style="color:{"#22c55e" if delta and "+" in str(delta) else "#ef4444" if delta else "gray"}">{delta or ""}</div>' if delta else ""
    return f"""<div class="metric-card animate-in">
        <div style="font-size:1.6rem">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>"""

# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_metric_safe(label, value, delta=None, icon=""):
    value_safe = html.escape(str(value))
    label_safe = html.escape(str(label))
    icon_safe = html.escape(str(icon))
    delta_safe = html.escape(str(delta)) if delta is not None else ""
    delta_color = "#22c55e" if delta and "+" in str(delta) else "#ef4444" if delta else "gray"
    delta_html = f'<div class="metric-delta" style="color:{delta_color}">{delta_safe}</div>' if delta else ""
    return (
        '<div class="metric-card animate-in">'
        f'<div style="font-size:1.6rem">{icon_safe}</div>'
        f'<div class="metric-value">{value_safe}</div>'
        f'<div class="metric-label">{label_safe}</div>'
        f'{delta_html}'
        '</div>'
    )

render_metric = render_metric_safe


# ─── Auth & audit helpers ──────────────────────────────────────────────────────
def log_action(action: str, detail: str = "") -> None:
    """Append an entry to the in-session audit log."""
    auth = st.session_state.get("auth")
    user = auth["username"] if auth else "guest"
    role = auth["role"] if auth else "-"
    st.session_state.audit.insert(0, security.audit_entry(user, role, action, detail))
    st.session_state.audit = st.session_state.audit[:200]  # keep last 200


def render_login(show_back: bool = True, key_prefix: str = "login") -> None:
    """Login form used by both the landing page and the auth gate."""
    c1, c2, c3 = st.columns([1, 1.1, 1])
    with c2:
        st.markdown("""<div style="text-align:center;padding-top:1.5vh">
            <div class="brand-orb brand-orb-wordmark" style="margin-bottom:12px">NS</div>
            <div class="hero-title" style="font-size:2rem">Sign in to <span class="gradient-text">Navneet Data Studio</span></div>
            <p class="hero-sub">Welcome back — your data workspace is ready.</p>
        </div>""", unsafe_allow_html=True)
        if st.session_state.get("signup_done"):
            st.success("✅ Account created — please sign in below.")
            st.session_state.signup_done = False
        with st.container(border=True):
            username = st.text_input("Username", placeholder="Your username", key=f"{key_prefix}_user")
            password = st.text_input("Password", type="password", placeholder="Enter password", key=f"{key_prefix}_pw")
            if st.button("Sign In", use_container_width=True, type="primary", key=f"{key_prefix}_submit"):
                ok, msg = attempt_login(username, password)
                if ok:
                    st.rerun()
                st.error(msg)
        # Don't have an account? → go to signup
        st.markdown("<div style='text-align:center;font-size:0.85rem;opacity:0.8;margin-top:12px'>Don't have an account?</div>", unsafe_allow_html=True)
        if st.button("Create a new account", use_container_width=True, key=f"{key_prefix}_gosignup"):
            st.session_state.page = "Landing"
            st.session_state.land_view = "Signup"
            st.rerun()
        if show_back and st.button("Back to home", use_container_width=True, key=f"{key_prefix}_back"):
            st.session_state.page = "Landing"
            st.session_state.land_view = "Home"
            st.rerun()


def render_signup(show_back: bool = True, key_prefix: str = "signup") -> None:
    """Public signup form stored in SQLite."""
    c1, c2, c3 = st.columns([1, 1.1, 1])
    with c2:
        st.markdown("""<div style="text-align:center;padding-top:1.5vh">
            <div class="brand-orb brand-orb-wordmark" style="margin-bottom:12px">NS</div>
            <div class="hero-title" style="font-size:2rem">Create your <span class="gradient-text">account</span></div>
            <p class="hero-sub">Pick your role — Data Engineer or Data Analyst.</p>
        </div>""", unsafe_allow_html=True)
        with st.container(border=True):
            full_name = st.text_input("Full name", placeholder="Your name", key=f"{key_prefix}_name")
            username = st.text_input("Username", placeholder="Choose a username", key=f"{key_prefix}_user")
            role = st.radio("I am signing up as", [security.ENGINEER, security.ANALYST],
                            horizontal=True, key=f"{key_prefix}_role")
            password = st.text_input("Password", type="password", key=f"{key_prefix}_pw")
            confirm_password = st.text_input("Confirm password", type="password", key=f"{key_prefix}_confirm")
            if st.button("Create Account", use_container_width=True, type="primary", key=f"{key_prefix}_submit"):
                err = security.validate_signup(full_name, username, password, confirm_password, role)
                if err:
                    st.error(err)
                else:
                    ok, msg = sqlite_store.create_user(full_name, username, password, role)
                    if ok:
                        log_action("SIGNUP", f"user={username.strip().lower()} role={role}")
                        st.session_state.signup_done = True
                        st.session_state.page = "Landing"
                        st.session_state.land_view = "Login"
                        st.rerun()
                    else:
                        st.error(msg)
        # Already have an account? → go to the login screen
        st.markdown("<div style='text-align:center;font-size:0.85rem;opacity:0.8;margin-top:12px'>Already have an account?</div>", unsafe_allow_html=True)
        if st.button("Go to Login", use_container_width=True, key=f"{key_prefix}_gologin"):
            st.session_state.page = "Landing"
            st.session_state.land_view = "Login"
            st.rerun()
        if show_back and st.button("Back to home", use_container_width=True, key=f"{key_prefix}_back"):
            st.session_state.page = "Landing"
            st.session_state.land_view = "Home"
            st.rerun()


# Authentication gate ───────────────────────────────────────────────────────
# Landing is public; every other page requires a logged-in user.
if st.session_state.page != "Landing" and not st.session_state.auth:
    st.markdown("""<style>
        section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display:none !important; }
    </style>""", unsafe_allow_html=True)
    render_login()
    st.stop()

with st.sidebar:
    # Brand
    st.markdown(f"""<div class="brand-wrap">
        <div class="brand-orb brand-orb-wordmark">NS</div>
        <div class="brand-name">Navneet Data Studio</div>
        <div class="brand-tag">Pro · Agentic Edition</div>
    </div>""", unsafe_allow_html=True)

    # Logged-in user card + logout
    _auth = st.session_state.auth
    if _auth:
        role = _auth["role"]
        rb = {"Admin": "badge-red", "Data Engineer": "badge-blue", "Data Analyst": "badge-green"}.get(role, "badge-blue")
        st.markdown(f"""<div class="data-card" style="padding:12px;text-align:center">
            <div style="font-size:0.95rem;letter-spacing:0.16em;text-transform:uppercase;color:#38bdf8">User</div>
            <div style="font-weight:700">{_auth['name']}</div>
            <span class="status-badge {rb}">{role}</span>
        </div>""", unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True, key="logout_btn"):
            log_action("LOGOUT")
            st.session_state.auth = None
            st.session_state.page = "Landing"
            st.rerun()
    else:
        role = None

    # Theme toggle (clean full-width emoji button)
    st.markdown('<div class="side-label">Appearance</div>', unsafe_allow_html=True)
    if st.button(("🌙  Dark mode" if dark else "☀️  Light mode"), use_container_width=True,
                 key="side_theme", help="Toggle light / dark theme"):
        st.session_state.dark_mode = not dark
        st.rerun()

    # Role-based navigation (grouped). Admin sees everything.
    nav_groups = [
        ("WORKSPACE", [
            ("Developer", "Developer"),
            ("Data Hub", "Data Hub"),
            ("AI Agent Chat", "AI Agent Chat"),
            ("Data Explorer", "Data Explorer"),
        ]),
        ("DATA ENGINEERING", [
            ("Data Engineer View", "Engineer View"),
            ("Pipeline Monitoring", "Pipeline Monitoring"),
            ("Agentic Pipeline", "Agentic Pipeline"),
            ("Self-Healing Pipeline", "Self-Healing"),
            ("Data Quality AI", "Data Quality AI"),
            ("Semantic Search", "Semantic Search"),
            ("Data Engineering Guide", "Engineering Guide"),
        ]),
        ("DATA ANALYTICS", [
            ("Data Analyst View", "Analyst View"),
            ("Dashboard", "Dashboard"),
            ("AI Analytics", "AI Analytics"),
            ("Executive Report", "Executive Report"),
        ]),
        ("ADMINISTRATION", [
            ("Admin / Security", "Admin / Security"),
        ]),
    ]
    for group_name, items in nav_groups:
        visible = [(pv, lb) for pv, lb in items if security.can_access(role, pv)]
        if not visible:
            continue
        st.markdown(f'<div class="side-label">{group_name}</div>', unsafe_allow_html=True)
        for page_value, label in visible:
            if st.session_state.page == page_value:
                st.markdown(f'<div class="side-nav-active">{label}</div>', unsafe_allow_html=True)
            else:
                if st.button(label, key=f"side_{page_value}", use_container_width=True):
                    st.session_state.page = page_value
                    st.rerun()
    page = st.session_state.page

    st.divider()

    # Data status — uploading happens on the dedicated Data Hub page
    st.markdown('<div class="side-label">Data</div>', unsafe_allow_html=True)
    if st.session_state.df is None:
        st.info("No dataset loaded. Open Data Hub to upload a CSV.")
        if st.button("Open Data Hub", use_container_width=True):
            st.session_state.page = "Data Hub"
            st.rerun()
    else:
        st.markdown(
            f'<span class="status-badge badge-green"> {st.session_state.df_name}</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    # AI Config — API key is embedded in config.py and intentionally hidden.
    st.markdown('<div class="side-label">AI Engine</div>', unsafe_allow_html=True)
    st.session_state.api_key = config.get_api_key()  # always available, never displayed
    key_ok = bool(st.session_state.api_key)
    st.markdown(
        f'<span class="status-badge {"badge-green" if key_ok else "badge-red"}">'
        f'{" AI Connected" if key_ok else " No key"}</span>'
        '<span class="status-badge badge-blue" style="margin-left:6px">OpenRouter</span>',
        unsafe_allow_html=True,
    )
    model_label = st.selectbox("AI Model", list(AVAILABLE_MODELS.keys()), index=0,
                               help="Reasoning-capable free models are listed first.")
    st.session_state.selected_model = AVAILABLE_MODELS[model_label]
    st.caption(" Reasoning enabled for Nemotron & GPT-OSS · key managed securely")

    st.divider()

    # Dataset info
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown(f"### Dataset Info")
        st.markdown(f"""
        <div class="data-card" style="font-size:0.8rem">
         <b>{st.session_state.df_name}</b><br>
         {df.shape[0]:,} rows × {df.shape[1]} cols<br>
         {df.select_dtypes(include=[np.number]).shape[1]} numeric<br>
         {df.select_dtypes(include=['object']).shape[1]} categorical<br>
         {df.isnull().sum().sum()} missing cells
        </div>
        """, unsafe_allow_html=True)

        # Download ZIP
        zip_buf = make_zip_download()
        st.download_button("Download Results ZIP", zip_buf, "ai_pipeline_results.zip", "application/zip", use_container_width=True)

    # Sidebar footer — copyright
    st.markdown("""<div style="margin-top:18px;padding-top:12px;border-top:1px solid rgba(148,163,184,0.2);
        font-size:0.68rem;line-height:1.5;text-align:center;opacity:0.75">
        &copy; 2026 <b>Snehal Laxman Jadhav</b><br>AI Engineer | Navneet Education Limited
    </div>""", unsafe_allow_html=True)

# ─── Audit page visits + enforce role-based access ─────────────────────────────
if st.session_state.auth:
    if st.session_state._last_audit_page != page:
        log_action("VIEW_PAGE", page)
        st.session_state._last_audit_page = page
    if page != "Landing" and not security.can_access(st.session_state.auth["role"], page):
        st.error(" Access denied — your role does not have permission to open this page.")
        st.info(f"You are signed in as **{st.session_state.auth['role']}**. Use the sidebar to open an allowed page.")
        st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: LANDING
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Landing":
    # ─── Clean single-row top navbar ────────────────────────────────────────────
    site_nav = ["Home", "Features", "Developer", "Contact"]
    topbar = st.container(border=True)
    with topbar:
        # brand | 4 nav links | theme | Launch
        cols = st.columns([2.1, 1.0, 1.05, 1.1, 1.0, 0.55, 1.35])
        with cols[0]:
            st.markdown(BRAND_LOGO_HTML, unsafe_allow_html=True)
        for name, c in zip(site_nav, cols[1:5]):
            with c:
                if st.button(name, use_container_width=True, key=f"snav_{name}"):
                    st.session_state.land_view = name
                    st.rerun()
        with cols[5]:
            # Light / dark mode toggle (emoji)
            if st.button("🌙" if dark else "☀️", use_container_width=True, key="land_theme", help="Toggle light / dark"):
                st.session_state.dark_mode = not dark
                st.rerun()
        with cols[6]:
            if st.button("Launch App", use_container_width=True, key="launch_top"):
                st.session_state.page = "Data Hub"
                st.rerun()

    view = st.session_state.land_view

    # ─── HOME ───────────────────────────────────────────────────────────────────
    if view == "Home":
        hero_left, hero_right = st.columns([1, 1], gap="large")
        with hero_left:
            st.markdown(f"""
            <div style="padding-top:26px">
                <span class="land-eyebrow"> Pro · Agentic Edition</span>
                <div class="hero-title" style="font-size:3.4rem;line-height:1.06;margin-top:10px">
                    <span class="gradient-text">AI Data<br>Engineering</span><br>
                    <span style="color:{'#e2e8f0' if dark else '#1a202c'}">Suite</span>
                </div>
                <p class="hero-sub" style="margin:16px 0;max-width:540px">
                    Turn raw data into decisions — clean it, understand it, and act on it,
                    all in one place. Smart pipelines fix issues automatically, AI explains
                    what matters, and every insight is just a click away.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="cta-row">', unsafe_allow_html=True)
            if st.button("Launch the Suite", use_container_width=True, key="launch_btn"):
                st.session_state.page = "Data Hub"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption("No setup · no API key needed · upload your CSV and go")

            sb = st.columns(4)
            stats = [("10", "Modules"), ("6+", "AI Models"), ("∞", "Your Data"), ("100%", "Out-of-box")]
            for (v, l), c in zip(stats, sb):
                with c:
                    st.markdown(f"""<div class="metric-card" style="padding:12px">
                        <div class="metric-value" style="font-size:1.4rem">{v}</div>
                        <div class="metric-label">{l}</div></div>""", unsafe_allow_html=True)
        with hero_right:
            load_three_js("three_sphere.html", height=480)

        st.markdown("<br>", unsafe_allow_html=True)
        features = ["Agentic LangGraph", "Self-Healing Pipeline", "Data Quality AI", "XGBoost Analytics",
                    "Vector DB (Chroma)", "Reasoning AI Agent", "IsolationForest Anomalies",
                    "Semantic Search", "Smart Alerts", "AI Executive Report", "Dark and Light", "OpenRouter"]
        st.markdown('<div style="text-align:center;margin:8px 0">' + "".join(f'<span class="feature-chip">{f}</span>' for f in features) + '</div>', unsafe_allow_html=True)

    # ─── FEATURES ───────────────────────────────────────────────────────────────
    elif view == "Features":
        st.markdown('<div class="hero-title" style="font-size:2.2rem">What\'s <span class="gradient-text">inside</span></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        cards = [
            ("", "Agentic Pipeline", "A LangGraph state machine autonomously runs profile → detect → heal → quality → ML → AI summary in one click.", ["LangGraph orchestration", "Live execution trace", "AI run summary"]),
            ("", "Self-Healing Pipeline", "Detects broken data and applies intelligent statistical fixes with a full audit trail and one-click corrected-file export.", ["9 issue detectors", "AI fix suggestions", "One-click clean CSV"]),
            ("", "Data Quality AI", "Quality scoring across 6 dimensions with gauge + radar visuals and SQL/Python fix generation.", ["Score 0-100", "Gauge & radar", "SQL fix code"]),
            ("", "AI Analytics", "XGBoost predictive modeling, IsolationForest anomaly detection, and RAG 'ask your data'.", ["Feature importance", "Anomaly scoring", "RAG Q&A"]),
            ("", "Vector Database", "ChromaDB indexes your rows via embeddings for true meaning-based semantic search.", ["ChromaDB", "Cosine HNSW", "Embeddings"]),
            ("", "Reasoning Agent", "DataSage chats with reasoning traces using OpenRouter free reasoning models.", ["Nemotron / GPT-OSS", "Reasoning traces", "Code generation"]),
        ]
        for row in range(0, len(cards), 3):
            ccols = st.columns(3)
            for (icon, title, desc, bullets), col in zip(cards[row:row+3], ccols):
                with col:
                    st.markdown(f"""<div class="data-card">
                        <div style="font-size:2.3rem;margin-bottom:10px">{icon}</div>
                        <h3 style="margin-bottom:8px;font-size:1.05rem">{title}</h3>
                        <p style="font-size:0.83rem;opacity:0.72;margin-bottom:12px">{desc}</p>
                        {"".join(f'<span class="feature-chip">{b}</span>' for b in bullets)}
                    </div>""", unsafe_allow_html=True)

    # ─── ABOUT ──────────────────────────────────────────────────────────────────
    elif view == "About":
        ac1, ac2 = st.columns([1.05, 1], gap="large")
        with ac1:
            st.markdown('<div class="hero-title" style="font-size:2.2rem">About the <span class="gradient-text">Studio</span></div>', unsafe_allow_html=True)
            st.markdown("""<div class="data-card" style="margin-top:14px">
                Navneet Data Studio brings together data quality review, pipeline repair, analytics, AI workflows,
                semantic search, and reporting in a single Streamlit experience built for practical day-to-day work.
                <br><br>
                The interface is tuned for clear decision-making: structured monitoring, polished charts, secure local
                configuration, and a more focused presentation layer.
            </div>""", unsafe_allow_html=True)
            st.markdown("""<div class="data-card" style="margin-top:14px">
                <h3 style="margin-top:0">Developer</h3>
                <p style="opacity:0.8;margin-bottom:0">Snehal Laxman Jadhav<br>AI Engineer at Navneet Education Limited<br>Copyright 2026</p>
            </div>""", unsafe_allow_html=True)
        with ac2:
            st.markdown("""<div class="data-card" style="text-align:center;padding:26px 24px">
                <svg width="180" height="180" viewBox="0 0 180 180" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Developer portrait">
                  <defs>
                    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
                      <stop offset="0%" stop-color="#22d3ee"/>
                      <stop offset="100%" stop-color="#2563eb"/>
                    </linearGradient>
                  </defs>
                  <rect x="12" y="12" width="156" height="156" rx="38" fill="#0f172a" stroke="rgba(34,211,238,0.35)"/>
                  <circle cx="90" cy="68" r="28" fill="url(#g)"/>
                  <path d="M48 142c8-24 28-38 42-38s34 14 42 38" fill="url(#g)" opacity="0.9"/>
                  <text x="90" y="162" text-anchor="middle" fill="#e2e8f0" font-size="16" font-family="Space Grotesk, sans-serif">SJ</text>
                </svg>
                <h3 style="margin:10px 0 4px">Snehal Laxman Jadhav</h3>
                <p style="opacity:0.75;font-size:0.92rem">AI Engineer</p>
                <p style="opacity:0.65;font-size:0.82rem">Navneet Education Limited</p>
                <div style="display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:10px">
                    <span class="contact-pill">Developer Profile</span>
                    <span class="contact-pill">Data Engineering</span>
                    <span class="contact-pill">AI Systems</span>
                </div>
            </div>""", unsafe_allow_html=True)

    # CONTACT ────────────────────────────────────────────────────────────────
    elif view == "Developer":
        dev_left, dev_right = st.columns([1.05, 1], gap="large")
        with dev_left:
            st.markdown('<div class="hero-title" style="font-size:2.2rem">Developer <span class="gradient-text">profile</span></div>', unsafe_allow_html=True)
            st.markdown("""<div class="data-card" style="margin-top:14px">
                Snehal Laxman Jadhav is the AI Engineer behind Navneet Data Studio, focused on practical data tooling,
                analytics systems, retrieval workflows, and user-facing AI experiences for business teams.
                <br><br>
                Copyright 2026 | Navneet Education Limited
            </div>""", unsafe_allow_html=True)
            st.markdown("""<div class="data-card" style="margin-top:14px">
                <h3 style="margin-top:0">Direct links</h3>
                <p style="margin-bottom:0">
                    <a class="contact-pill" href="mailto:contact@navneet.com">Gmail</a>
                    <a class="contact-pill" href="https://www.linkedin.com/" target="_blank">LinkedIn</a>
                    <a class="contact-pill" href="https://www.navneet.com/" target="_blank">Navneet</a>
                </p>
            </div>""", unsafe_allow_html=True)
        with dev_right:
            load_three_js("three_particles.html", height=360)
            st.markdown("""<div class="data-card" style="margin-top:14px">
                <h3 style="margin-top:0">Focus areas</h3>
                <div style="display:flex;gap:10px;flex-wrap:wrap">
                    <span class="feature-chip">Data Engineering</span>
                    <span class="feature-chip">AI Agents</span>
                    <span class="feature-chip">Vector Search</span>
                    <span class="feature-chip">Analytics UX</span>
                </div>
            </div>""", unsafe_allow_html=True)

    elif view == "Contact":
        st.markdown('<div class="hero-title" style="font-size:2.2rem">Developer <span class="gradient-text">contact</span></div>', unsafe_allow_html=True)
        st.markdown("""<div class="data-card" style="margin-top:14px;text-align:center">
            <p style="opacity:0.82">Built by Snehal Laxman Jadhav, AI Engineer at Navneet Education Limited.</p>
            <div style="display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:14px">
                <a class="contact-pill" href="mailto:contact@navneet.com">Gmail</a>
                <a class="contact-pill" href="https://www.linkedin.com/" target="_blank">LinkedIn</a>
                <a class="contact-pill" href="https://www.navneet.com/" target="_blank">Navneet</a>
                <span class="contact-pill">Copyright 2026</span>
            </div>
        </div>""", unsafe_allow_html=True)
        cc = st.columns([1, 1, 1.4])
        with cc[1]:
            if st.button("Launch the App", use_container_width=True, key="launch_contact"):
                st.session_state.page = "Data Hub"
                st.rerun()

    # Footer (always) ────────────────────────────────────────────────────────
    elif view == "SQLite":
        st.markdown('<div class="hero-title" style="font-size:2.2rem">SQLite <span class="gradient-text">workspace</span></div>', unsafe_allow_html=True)
        st.markdown("""<div class="data-card" style="margin-top:14px">
            SQLite now powers user accounts, signup, and dataset CRUD in this project. Save a dataset from the Data Hub,
            then manage rows with add, edit, and delete in Data Explorer or administer users from the security page.
        </div>""", unsafe_allow_html=True)
        lib_col1, lib_col2 = st.columns(2)
        with lib_col1:
            saved_df = sqlite_store.list_datasets()
            st.markdown("#### Saved datasets")
            st.dataframe(saved_df, use_container_width=True, height=260)
        with lib_col2:
            users_df = sqlite_store.list_users()[["username", "full_name", "role", "created_at"]]
            st.markdown("#### Registered users")
            st.dataframe(users_df, use_container_width=True, height=260)

    elif view == "Login":
        render_login(show_back=False, key_prefix="landing_login")

    elif view == "Signup":
        render_signup(show_back=False, key_prefix="landing_signup")

    st.markdown("""<div class="site-footer">
        &copy; 2026 <b>Snehal Laxman Jadhav</b> | AI Engineer at <b>Navneet Education Limited</b><br>
        Navneet Data Studio built with Streamlit | Three.js | LangGraph | ChromaDB | XGBoost
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DEVELOPER
elif page == "Developer":
    st.markdown("## Developer")
    st.markdown('<p class="hero-sub">Profile, branding, contact links, and project ownership details.</p>', unsafe_allow_html=True)
    dev_a, dev_b = st.columns([1.05, 1], gap="large")
    with dev_a:
        st.markdown("""<div class="data-card">
            <h3 style="margin-top:0">Snehal Laxman Jadhav</h3>
            <p style="opacity:0.82">AI Engineer at Navneet Education Limited</p>
            <p style="opacity:0.74">Copyright 2026 | Navneet Education Limited. This workspace is built for data engineering, analytics, and practical AI operations.</p>
            <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:12px">
                <a class="contact-pill" href="mailto:contact@navneet.com">Gmail</a>
                <a class="contact-pill" href="https://www.linkedin.com/" target="_blank">LinkedIn</a>
                <a class="contact-pill" href="https://www.navneet.com/" target="_blank">Navneet</a>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="data-card">
            <h3 style="margin-top:0">Brand system</h3>
            <p style="opacity:0.74">The project now uses the custom <b>NS</b> mark for Navneet Studio across the sidebar, auth screens, and landing navbar.</p>
            <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:10px">
                <div class="brand-orb brand-orb-wordmark" style="margin:0">NS</div>
                <span class="contact-pill">Three.js motion scene enabled</span>
                <span class="contact-pill">Theme-ready logo</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with dev_b:
        load_three_js("three_particles.html", height=380)
        st.markdown("""<div class="data-card" style="margin-top:14px">
            <h3 style="margin-top:0">Focus areas</h3>
            <div style="display:flex;gap:10px;flex-wrap:wrap">
                <span class="feature-chip">Data Engineering</span>
                <span class="feature-chip">AI Systems</span>
                <span class="feature-chip">Vector Search</span>
                <span class="feature-chip">Analytics UX</span>
            </div>
        </div>""", unsafe_allow_html=True)

# PAGE: DATA HUB  (upload — replaces the sidebar data source)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Data Hub":
    st.markdown("## Data Hub")
    st.markdown('<p class="hero-sub">Enterprise upload · AI profiling · quality · cleaning · transformation · export.</p>', unsafe_allow_html=True)

    _role = st.session_state.auth["role"] if st.session_state.auth else None
    _is_admin = _role == security.ADMIN

    def _cache_key():
        d = st.session_state.df
        return f"{st.session_state.df_name}_{0 if d is None else len(d)}_{0 if d is None else d.shape[1]}"

    t_up, t_prof, t_qual, t_clean, t_tx = st.tabs(
        ["Upload & Preview", "AI Profiling", "Quality Checks", "Cleaning", "Transform"])

    # ─── TAB 1: UPLOAD & PREVIEW ────────────────────────────────────────────────
    with t_up:
        if security.has_permission(_role, security.PERM_UPLOAD):
            ups = st.file_uploader("Drag and drop CSV or Excel files here (up to 200 MB each)",
                                   type=["csv", "xlsx", "xls"], accept_multiple_files=True)
            if ups:
                names = [f.name for f in ups]
                pick = names[0] if len(ups) == 1 else st.selectbox("Choose the active file", names)
                f = next(x for x in ups if x.name == pick)
                if st.session_state.get("_loaded_upload") != f.name:
                    try:
                        if f.name.lower().endswith((".xlsx", ".xls")):
                            df_new = pd.read_excel(f, engine="openpyxl")
                        else:
                            df_new = pd.read_csv(f)
                        apply_dataset(df_new, f.name.rsplit(".", 1)[0])
                        set_active_sqlite_dataset(None, None)
                        st.session_state._loaded_upload = f.name
                        st.session_state._upload_size_kb = round(getattr(f, "size", 0) / 1024, 1)
                        log_action("UPLOAD", f.name)
                        st.success(f"Loaded **{f.name}** — {df_new.shape[0]:,} rows × {df_new.shape[1]} columns")
                    except Exception as e:
                        st.error(f"Could not read this file: {e}")
        else:
            st.info("Your role does not have upload permission. Load a demo dataset below instead.")

        with st.expander("Load a ready-made sample dataset"):
            files = {
                "Navneet Export Sales": "navneet_export.csv", "Employee Access Log": "employee_access_log.csv",
                "Pipeline Audit Log": "pipeline_audit_log.csv", "Walmart Customers": "walmart_customers.csv",
                "Walmart Inventory": "walmart_inventory.csv", "Walmart Sales": "walmart_sales.csv",
                "Navneet Products": "navneet_products.csv", "Navneet Sales": "navneet_sales.csv",
                "Navneet Financials": "navneet_financials.csv",
            }
            dc1, dc2 = st.columns([2, 1])
            with dc1:
                demo = st.selectbox("Sample dataset", list(files.keys()), label_visibility="collapsed")
            with dc2:
                if st.button("Load sample", use_container_width=True):
                    try:
                        path = os.path.join(os.path.dirname(__file__), "data", files[demo])
                        apply_dataset(pd.read_csv(path), demo)
                        set_active_sqlite_dataset(None, None)
                        st.session_state._loaded_upload = None
                        st.success(f"Loaded sample: {demo}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"{e}")

        if st.session_state.df is None:
            st.info("Upload a file or load a sample to begin.")
        else:
            df = st.session_state.df
            schema = dh.detect_schema(df)
            m = st.columns(5)
            size_kb = st.session_state.get("_upload_size_kb")
            size_txt = (f"{size_kb/1024:.1f} MB" if size_kb and size_kb > 1024 else f"{size_kb} KB") if size_kb else "—"
            with m[0]: st.markdown(render_metric("Rows", f"{len(df):,}"), unsafe_allow_html=True)
            with m[1]: st.markdown(render_metric("Columns", df.shape[1]), unsafe_allow_html=True)
            with m[2]: st.markdown(render_metric("Numeric", df.select_dtypes(include=[np.number]).shape[1]), unsafe_allow_html=True)
            with m[3]: st.markdown(render_metric("Missing", f"{int(df.isnull().sum().sum()):,}"), unsafe_allow_html=True)
            with m[4]: st.markdown(render_metric("File Size", size_txt), unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown("#### Schema Detection")
                schema_rows = [{
                    "column": c, "type": s["kind"], "dtype": s["dtype"], "nulls": s["nulls"],
                    "null_%": s["null_pct"], "unique": s["unique"], "samples": ", ".join(s["samples"]),
                } for c, s in schema.items()]
                st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, height=260)
                badges = "".join(
                    f'<span class="status-badge {dh.kind_badge(s["kind"])}" style="margin:2px">{c}: {s["kind"]}</span>'
                    for c, s in schema.items())
                st.markdown(f'<div style="margin-top:6px">{badges}</div>', unsafe_allow_html=True)

            st.markdown("#### Preview (first 20 rows)")
            prev_df, _masked = security.mask_dataframe(df.head(20), reveal=bool(_is_admin))
            if _masked and not _is_admin:
                st.markdown(f'<div class="alert-info">PII masked for your role: {", ".join(f"<code>{c}</code>" for c in _masked)}.</div>', unsafe_allow_html=True)
            st.dataframe(prev_df, use_container_width=True, height=300)

            qa = st.columns(4)
            quick = [("Run AI Profiling", "AI Profiling"), ("Run Quality Check", "Quality"),
                     ("Clean This Data", "Cleaning"), ("Go to AI Analytics", "AI Analytics")]
            for (lbl, tgt), c in zip(quick, qa):
                with c:
                    if st.button(lbl, use_container_width=True, key=f"hub_q_{lbl}"):
                        if tgt == "AI Analytics":
                            st.session_state.page = "AI Analytics"; st.rerun()
                        else:
                            st.info(f"Open the **{tgt}** tab above.")

            # Save to Database
            if security.has_permission(_role, security.PERM_DB_SAVE):
                st.markdown('<div class="data-card" style="border:1px solid var(--accent,#00d4ff)">', unsafe_allow_html=True)
                st.markdown("#### Save to Database")
                save_name = st.text_input("Dataset name", value=st.session_state.df_name or "dataset", key="hub_save_name")
                if st.button("Save to SQLite Database", use_container_width=True, type="primary", key="hub_save_db"):
                    actor = st.session_state.auth["username"] if st.session_state.auth else "guest"
                    to_save = st.session_state.df_fixed if st.session_state.df_fixed is not None else df
                    ok, msg = sqlite_store.save_dataset(save_name or "dataset", to_save, actor)
                    if ok:
                        log_action("DB_SAVE", save_name)
                        score = st.session_state.df_quality_score
                        st.success(f"Saved. Table: {save_name} | Rows: {len(to_save):,} | "
                                   f"Quality: {score if score is not None else '—'}/100 | "
                                   f"{datetime.now().strftime('%H:%M:%S')}")
                    else:
                        st.error(msg)
                st.markdown('</div>', unsafe_allow_html=True)

            # SQLite library (existing load/delete kept intact)
            with st.expander("SQLite Library — load / delete saved datasets"):
                saved_df = sqlite_store.list_datasets()
                if saved_df.empty:
                    st.info("No SQLite datasets yet.")
                else:
                    option_map = {f"{row.id}: {row.name} ({row.row_count} rows)": int(row.id) for row in saved_df.itertuples(index=False)}
                    sel = st.selectbox("Saved datasets", list(option_map.keys()))
                    ac = st.columns(2)
                    with ac[0]:
                        if st.button("Load from SQLite", use_container_width=True):
                            df_sql, meta, msg = sqlite_store.load_dataset(option_map[sel])
                            if df_sql is not None:
                                apply_dataset(df_sql.drop(columns=["_rowid_"], errors="ignore"), meta["name"])
                                set_active_sqlite_dataset(int(meta["id"]), meta)
                                log_action("SQLITE_LOAD", meta["name"]); st.success(msg); st.rerun()
                            else:
                                st.error(msg)
                    with ac[1]:
                        if st.button("Delete SQLite dataset", use_container_width=True):
                            ok, msg = sqlite_store.delete_dataset(option_map[sel])
                            if ok:
                                if st.session_state.sqlite_dataset_id == option_map[sel]:
                                    set_active_sqlite_dataset(None, None)
                                log_action("SQLITE_DELETE", sel); st.success(msg); st.rerun()
                            else:
                                st.error(msg)

            # Download Center
            if security.has_permission(_role, security.PERM_DOWNLOAD):
                st.markdown("#### Download Center")
                clean = st.session_state.df_fixed if st.session_state.df_fixed is not None else df
                dl = st.columns(3)
                with dl[0]:
                    if st.download_button("Cleaned CSV", clean.to_csv(index=False).encode(),
                                          f"{st.session_state.df_name}_clean.csv", "text/csv", use_container_width=True):
                        log_action("DOWNLOAD", "cleaned_csv")
                with dl[1]:
                    if st.session_state.df_quality_checks:
                        qjson = json.dumps({"score": st.session_state.df_quality_score,
                                            "checks": st.session_state.df_quality_checks}, indent=2)
                        if st.download_button("Quality Report (JSON)", qjson.encode(),
                                              "quality_report.json", "application/json", use_container_width=True):
                            log_action("DOWNLOAD", "quality_report")
                    else:
                        st.caption("Run Quality tab first")
                with dl[2]:
                    try:
                        xls = dh.build_download_excel(clean, df, st.session_state.df_quality_checks,
                                                      st.session_state.transformation_log + st.session_state.cleaning_log,
                                                      dh.build_ai_summary_md(st.session_state.df_name,
                                                                             st.session_state.df_profiling_result,
                                                                             st.session_state.df_quality_checks,
                                                                             st.session_state.df_quality_score or 0,
                                                                             st.session_state.insights_narrative))
                        if st.download_button("Final Workbook (Excel)", xls,
                                              f"{st.session_state.df_name}_final.xlsx",
                                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                              use_container_width=True):
                            log_action("DOWNLOAD", "final_xlsx")
                    except Exception as e:
                        st.caption(f"Excel unavailable: {e}")

    # ─── TAB 2: AI PROFILING ────────────────────────────────────────────────────
    with t_prof:
        if st.session_state.df is None:
            st.info("Load a dataset first (Upload & Preview tab).")
        else:
            df = st.session_state.df
            key = _cache_key()
            if st.button("Run AI Profiling", type="primary") or (st.session_state.df_profiling_result and st.session_state.get("_prof_key") == key):
                if st.session_state.get("_prof_key") != key or not st.session_state.df_profiling_result:
                    schema = dh.detect_schema(df)
                    col_info = {c: s["kind"] for c, s in schema.items()}
                    prompt = (f"You are a Senior Data Analyst. Analyze this dataset and return valid JSON.\n"
                              f"Dataset name: {st.session_state.df_name}\nShape: {df.shape[0]} rows x {df.shape[1]} columns\n"
                              f"Columns and types: {json.dumps(col_info)}\nSample: {df.head(5).to_dict('records')}\n"
                              f"Missing: {df.isnull().sum().to_dict()}\nDuplicates: {int(df.duplicated().sum())}\n"
                              "Return JSON keys: dataset_summary, dataset_type, business_meaning, primary_key_candidates, "
                              "date_columns, numeric_columns, text_columns, currency_columns, sensitive_columns, "
                              "outlier_columns, kpi_suggestions, dashboard_ideas, anomaly_signals.")
                    with st.spinner("AI is analyzing your data..."):
                        res = call_openrouter_json(prompt)
                    if res is None:
                        st.error("AI analysis failed. Check your API key.")
                    else:
                        st.session_state.df_profiling_result = res
                        st.session_state._prof_key = key
            prof = st.session_state.df_profiling_result
            if prof:
                pm = st.columns(4)
                with pm[0]: st.markdown(render_metric("Type", str(prof.get("dataset_type", "—"))[:16]), unsafe_allow_html=True)
                with pm[1]: st.markdown(render_metric("PK", str((prof.get("primary_key_candidates") or ["—"])[0])[:14]), unsafe_allow_html=True)
                with pm[2]: st.markdown(render_metric("Sensitive", len(prof.get("sensitive_columns") or [])), unsafe_allow_html=True)
                with pm[3]: st.markdown(render_metric("KPIs", len(prof.get("kpi_suggestions") or [])), unsafe_allow_html=True)
                lc, rc = st.columns([1.3, 1])
                with lc:
                    st.markdown(f'<div class="data-card"><b>Summary</b><br>{prof.get("dataset_summary","")}<br><br>'
                                f'<b>Business meaning</b><br>{prof.get("business_meaning","")}</div>', unsafe_allow_html=True)
                with rc:
                    chips = "".join(f'<span class="feature-chip">{k}</span>' for k in (prof.get("kpi_suggestions") or []))
                    st.markdown(f'<div class="data-card"><b>Suggested KPIs</b><br>{chips}</div>', unsafe_allow_html=True)
                with st.expander("Schema Intelligence"):
                    for label, kkey in [("Primary keys", "primary_key_candidates"), ("Date", "date_columns"),
                                        ("Numeric", "numeric_columns"), ("Currency", "currency_columns"),
                                        ("Sensitive", "sensitive_columns"), ("Outliers", "outlier_columns")]:
                        vals = prof.get(kkey) or []
                        chips = "".join(f'<span class="status-badge badge-blue" style="margin:2px">{v}</span>' for v in vals) or "—"
                        st.markdown(f"**{label}:** {chips}", unsafe_allow_html=True)
                with st.expander("Dashboard Ideas & Anomaly Signals"):
                    for idea in (prof.get("dashboard_ideas") or []):
                        st.markdown(f"- {idea}")
                    for a in (prof.get("anomaly_signals") or []):
                        st.markdown(f'<div class="alert-warning">{a}</div>', unsafe_allow_html=True)

            # Pure-pandas statistical profiling + charts (no AI)
            st.markdown("#### Statistical Profile")
            nprof = dh.numeric_profile(df)
            if not nprof.empty:
                st.dataframe(nprof, use_container_width=True)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = [c for c in df.select_dtypes(include="object").columns if df[c].nunique() < 30]
            gc = st.columns(2)
            if len(num_cols) > 1:
                with gc[0]:
                    corr = df[num_cols].apply(pd.to_numeric, errors="coerce").corr()
                    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale="RdBu", zmid=0))
                    style_fig(fig, 300); fig.update_layout(title="Correlation")
                    st.plotly_chart(fig, use_container_width=True)
            if cat_cols:
                with gc[1]:
                    vc = df[cat_cols[0]].value_counts().head(8)
                    fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f"Top {cat_cols[0]}")
                    style_fig(fig, 300)
                    st.plotly_chart(fig, use_container_width=True)

            # AI Insights narrative + risk banner
            if prof and st.button("Generate Business Narrative"):
                with st.spinner("AI is writing the executive narrative..."):
                    agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
                    st.session_state.insights_narrative = agent.summarize_dashboard_insights(prof, st.session_state.df_name)
            if st.session_state.insights_narrative:
                st.markdown(f'<div class="data-card"><b>AI Business Narrative</b></div>', unsafe_allow_html=True)
                st.markdown(st.session_state.insights_narrative)
            if st.session_state.df_quality_score is not None and st.session_state.df_quality_score < 60:
                st.markdown('<div class="alert-critical">This dataset has critical quality issues. Do not use for reporting until resolved.</div>', unsafe_allow_html=True)

    # ─── TAB 3: QUALITY CHECKS ──────────────────────────────────────────────────
    with t_qual:
        if st.session_state.df is None:
            st.info("Load a dataset first (Upload & Preview tab).")
        else:
            df = st.session_state.df
            key = _cache_key()
            if st.session_state.get("_qual_key") != key or not st.session_state.df_quality_checks:
                st.session_state.df_quality_checks = dh.run_quality_checks(df)
                st.session_state.df_quality_score = dh.compute_quality_score(st.session_state.df_quality_checks)
                st.session_state._qual_key = key
            checks = st.session_state.df_quality_checks
            score = st.session_state.df_quality_score
            col = "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
            g = go.Figure(go.Indicator(mode="gauge+number", value=score,
                gauge={"axis": {"range": [0, 100]}, "bar": {"color": col},
                       "steps": [{"range": [0, 60], "color": "rgba(239,68,68,0.15)"},
                                 {"range": [60, 80], "color": "rgba(245,158,11,0.15)"},
                                 {"range": [80, 100], "color": "rgba(34,197,94,0.15)"}]},
                title={"text": "Data Quality Score"}))
            style_fig(g, 280)
            st.plotly_chart(g, use_container_width=True)
            passed = [c for c in checks if c["status"] == "Passed"]
            warn = [c for c in checks if c["status"] == "Warning"]
            failed = [c for c in checks if c["status"] == "Failed"]
            cc = st.columns(3)
            with cc[0]: st.markdown(f'<div class="alert-ok"><b>Passed ({len(passed)})</b></div>', unsafe_allow_html=True)
            with cc[1]: st.markdown(f'<div class="alert-warning"><b>Warnings ({len(warn)})</b></div>', unsafe_allow_html=True)
            with cc[2]: st.markdown(f'<div class="alert-critical"><b>Failed ({len(failed)})</b></div>', unsafe_allow_html=True)
            for grp, css in [(failed, "alert-critical"), (warn, "alert-warning"), (passed, "alert-ok")]:
                for c in grp:
                    st.markdown(f'<div class="{css}"><b>{c["name"]}</b> — {c["detail"]}</div>', unsafe_allow_html=True)
            if (failed or warn) and st.button("Get AI Recommended Fixes"):
                bad = [f"{c['name']}: {c['detail']}" for c in failed + warn]
                with st.spinner("AI is analyzing your data..."):
                    agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
                    rec = agent.chat("Give specific pandas code fixes for these data quality issues:\n" + "\n".join(bad))
                st.markdown(rec)

    # ─── TAB 4: CLEANING ────────────────────────────────────────────────────────
    with t_clean:
        if st.session_state.df is None:
            st.info("Load a dataset first (Upload & Preview tab).")
        elif not security.has_permission(_role, security.PERM_CLEAN):
            st.warning("Your role does not have cleaning permission.")
        else:
            base = st.session_state.df_fixed if st.session_state.df_fixed is not None else st.session_state.df
            st.caption("Cleaning works on a copy — your original stays intact.")
            actions = [
                ("Remove Duplicate Rows", "remove_duplicates"), ("Fill Missing Values", "fill_missing"),
                ("Standardize Column Names", "standardize_columns"), ("Convert Date Formats", "convert_dates"),
                ("Convert Numeric Columns", "convert_numeric"), ("Trim Whitespace", "trim_whitespace"),
                ("Standardize Text Case", "standardize_case"), ("Remove Special Characters", "remove_special"),
                ("Fix Currency Values", "fix_currency"), ("Fix Percentage Values", "fix_percentage"),
                ("Clean Phone Numbers", "clean_phone"),
            ]
            grid = st.columns(2)
            for i, (lbl, act) in enumerate(actions):
                with grid[i % 2]:
                    if st.button(lbl, use_container_width=True, key=f"clean_{act}"):
                        old = st.session_state.df_quality_score or dh.compute_quality_score(dh.run_quality_checks(base))
                        new_df, info = dh.apply_cleaning_action(base, act)
                        st.session_state.df_fixed = new_df
                        info["timestamp"] = datetime.now().strftime("%H:%M:%S")
                        st.session_state.cleaning_log.append(info)
                        log_action("CLEAN", act)
                        new_checks = dh.run_quality_checks(new_df)
                        new_score = dh.compute_quality_score(new_checks)
                        st.session_state.df_quality_checks = new_checks
                        st.session_state.df_quality_score = new_score
                        st.session_state._qual_key = f"{st.session_state.df_name}_{len(new_df)}_{new_df.shape[1]}"
                        st.success(f"{lbl}: changed {len(info.get('columns', []))} column(s). "
                                   f"Quality {old} → {new_score} ({'+' if new_score>=old else ''}{new_score-old}).")
            if st.button("Apply All Recommended Fixes", type="primary"):
                cur = base
                for act in dh.ALL_ACTIONS:
                    cur, info = dh.apply_cleaning_action(cur, act)
                    info["timestamp"] = datetime.now().strftime("%H:%M:%S")
                    st.session_state.cleaning_log.append(info)
                st.session_state.df_fixed = cur
                st.session_state.df_quality_checks = dh.run_quality_checks(cur)
                st.session_state.df_quality_score = dh.compute_quality_score(st.session_state.df_quality_checks)
                log_action("CLEAN", "apply_all")
                st.success(f"All fixes applied. New quality score: {st.session_state.df_quality_score}/100.")
            if st.session_state.df_fixed is not None:
                st.markdown("#### Cleaned Preview")
                st.dataframe(st.session_state.df_fixed.head(20), use_container_width=True, height=280)
            if st.session_state.cleaning_log:
                with st.expander("Cleaning Log"):
                    st.dataframe(pd.DataFrame(st.session_state.cleaning_log), use_container_width=True)

    # ─── TAB 5: TRANSFORM ───────────────────────────────────────────────────────
    with t_tx:
        if st.session_state.df is None:
            st.info("Load a dataset first (Upload & Preview tab).")
        elif not security.has_permission(_role, security.PERM_TRANSFORM):
            st.warning("Your role does not have transformation permission.")
        else:
            base = st.session_state.df_fixed if st.session_state.df_fixed is not None else st.session_state.df
            num_cols = base.select_dtypes(include=[np.number]).columns.tolist()
            st.markdown("#### Calculated Column")
            cc = st.columns([1, 1, 1, 1])
            with cc[0]: new_name = st.text_input("New column name", "new_col")
            with cc[1]: op = st.selectbox("Operation", ["multiply", "divide", "add", "subtract", "ratio", "cumsum", "rank"])
            with cc[2]: a = st.selectbox("Column A", num_cols or base.columns.tolist(), key="tx_a")
            with cc[3]: b = st.selectbox("Column B", num_cols or base.columns.tolist(), key="tx_b")
            if st.button("Add Calculated Column"):
                try:
                    out = base.copy()
                    va, vb = pd.to_numeric(out[a], errors="coerce"), pd.to_numeric(out[b], errors="coerce")
                    out[new_name] = {"multiply": va*vb, "divide": va/vb.replace(0, np.nan),
                                     "add": va+vb, "subtract": va-vb, "ratio": va/vb.replace(0, np.nan),
                                     "cumsum": va.cumsum(), "rank": va.rank()}[op]
                    st.session_state.df_fixed = out
                    st.session_state.transformation_log.append({"action": f"calc:{new_name}={a} {op} {b}",
                                                                "timestamp": datetime.now().strftime("%H:%M:%S")})
                    log_action("TRANSFORM", f"calc_{new_name}")
                    st.success(f"Added column '{new_name}'.")
                except Exception as e:
                    st.error(f"{e}")

            st.markdown("#### Aggregation Builder")
            gcols = st.multiselect("Group by", base.columns.tolist())
            acol = st.selectbox("Aggregate column", num_cols or base.columns.tolist(), key="tx_agg_col")
            afunc = st.selectbox("Function", ["sum", "mean", "count", "min", "max"], key="tx_agg_fn")
            if gcols and st.button("Preview Aggregation"):
                try:
                    agg = base.groupby(gcols)[acol].agg(afunc).reset_index()
                    st.dataframe(agg.head(20), use_container_width=True)
                    if st.button("Set Aggregation as Active Dataset", key="tx_set_agg"):
                        apply_dataset(agg, f"{st.session_state.df_name}_agg")
                        st.rerun()
                except Exception as e:
                    st.error(f"{e}")

            st.markdown("#### Pivot Table")
            pc = st.columns(4)
            with pc[0]: pidx = st.selectbox("Index", base.columns.tolist(), key="pv_i")
            with pc[1]: pcol = st.selectbox("Columns", base.columns.tolist(), key="pv_c")
            with pc[2]: pval = st.selectbox("Values", num_cols or base.columns.tolist(), key="pv_v")
            with pc[3]: pfn = st.selectbox("Aggfunc", ["sum", "mean", "count"], key="pv_f")
            if st.button("Build Pivot"):
                try:
                    pv = pd.pivot_table(base, index=pidx, columns=pcol, values=pval, aggfunc=pfn).reset_index()
                    st.dataframe(pv.head(20), use_container_width=True)
                except Exception as e:
                    st.error(f"{e}")
            if st.session_state.transformation_log:
                with st.expander("Transformation Log"):
                    st.dataframe(pd.DataFrame(st.session_state.transformation_log), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA ENGINEER VIEW  (role dashboard)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Data Engineer View":
    st.markdown("## Data Engineer View")
    st.markdown('<p class="hero-sub">Pipelines · ETL status · data quality · error logs — your engineering control room.</p>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Upload a dataset in the Data Hub to populate the engineering metrics.")
        load_three_js("three_network.html", height=300)
        st.stop()

    df = st.session_state.df
    metrics = monitoring.compute_metrics(df)
    alerts = monitoring.build_alerts(metrics, df)
    fresh_date, fresh_days = monitoring.freshness(df)

    # Status + KPI cards
    s1, s2, s3, s4 = st.columns(4)
    status_badge = " HEALTHY" if metrics["success"] else " NEEDS ATTENTION"
    with s1: st.markdown(render_metric("Pipeline Status", status_badge, icon=""), unsafe_allow_html=True)
    with s2: st.markdown(render_metric("Records Processed", f"{metrics['records']:,}", icon=""), unsafe_allow_html=True)
    with s3: st.markdown(render_metric("Errors Detected", metrics["errors"], icon=""), unsafe_allow_html=True)
    with s4: st.markdown(render_metric("Health Score", f"{metrics['health']}/100", icon=""), unsafe_allow_html=True)

    t_over, t_quality, t_errors, t_ai, t_concepts = st.tabs(
        [" Overview", " Data Quality", " Error Logs", "🤖 AI Insights", " Concepts"])

    with t_over:
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(render_metric("Processing Time", f"{metrics['processing_ms']} ms", icon="⏱"), unsafe_allow_html=True)
        with c2: st.markdown(render_metric("Data Freshness", f"{fresh_days}d old" if fresh_days is not None else "n/a", icon=""), unsafe_allow_html=True)
        with c3: st.markdown(render_metric("Last Run", metrics["last_run"].split(' ')[1], icon=""), unsafe_allow_html=True)

        g1, g2, g3 = st.columns(3)
        with g1:
            st.plotly_chart(mini_gauge(metrics["health"], "Health Score", ref=80), use_container_width=True)
        with g2:
            integrity = max(0, 100 - metrics["error_ratio"])
            st.plotly_chart(mini_gauge(round(integrity, 1), "Data Integrity %", ref=90), use_container_width=True)
        with g3:
            completeness = round(100 * (1 - metrics["missing"] / max(1, df.size)), 1)
            st.plotly_chart(mini_gauge(completeness, "Completeness %", ref=95), use_container_width=True)

        st.markdown("#### Live Alerts")
        css_map = {"critical": "alert-critical", "warning": "alert-warning", "ok": "alert-ok"}
        for level, title, msg in alerts:
            st.markdown(f'<div class="{css_map[level]}"><b>{title}</b><br>{msg}</div>', unsafe_allow_html=True)

    with t_quality:
        comp = {
            "Missing values": metrics["missing"],
            "Duplicate rows": metrics["duplicates"],
            "Type errors": metrics["type_errors"],
            "Anomalies": metrics["anomalies"],
        }
        qfig = px.bar(x=list(comp.keys()), y=list(comp.values()),
                      color=list(comp.values()), color_continuous_scale="Reds",
                      title="Detected issues by category")
        style_fig(qfig, 320)
        st.plotly_chart(qfig, use_container_width=True)
        st.caption("Run the  Self-Healing Pipeline to auto-fix these and export a clean file.")

    with t_errors:
        st.markdown("#### Pipeline Run Audit Log")
        audit_df = monitoring.load_audit_log()
        if audit_df is not None:
            def _row_style(s):
                color = {"SUCCESS": "#22c55e", "WARNING": "#f59e0b", "FAILED": "#ef4444"}.get(s, "")
                return color
            st.dataframe(audit_df, use_container_width=True, height=300)
            run_fig = px.bar(audit_df, x="run_id", y="duration_sec", color="status",
                             color_discrete_map={"SUCCESS": "#22c55e", "WARNING": "#f59e0b", "FAILED": "#ef4444"},
                             title="Run duration by status")
            style_fig(run_fig, 300); run_fig.update_xaxes(showticklabels=False)
            st.plotly_chart(run_fig, use_container_width=True)
        else:
            st.info("Audit log not found.")

    with t_ai:
        st.markdown("#### AI Engineering Co-Pilot")
        st.caption("AI reviews live pipeline metrics and recommends concrete engineering actions.")
        schema_preview = ", ".join(f"{c}:{str(t)}" for c, t in list(df.dtypes.items())[:20])
        de_prompt = (
            f"Pipeline metrics for dataset '{st.session_state.df_name}': "
            f"records={metrics['records']}, errors={metrics['errors']}, health={metrics['health']}/100, "
            f"missing_cells={metrics['missing']}, duplicate_rows={metrics['duplicates']}, "
            f"type_errors={metrics['type_errors']}, anomalies={metrics['anomalies']}, "
            f"error_ratio={metrics['error_ratio']}%, freshness_days={fresh_days}. "
            f"Schema (first cols): {schema_preview}. "
            "As a senior data engineer, summarize pipeline health, the top risks to reliability/SLAs, "
            "and prioritized engineering actions (imputation, schema enforcement, monitoring, partitioning, alerting)."
        )
        render_ai_brief(de_prompt, key="de_view", button_label="🤖 Analyze pipeline with AI")

    with t_concepts:
        st.markdown("""
**The Data Engineering lifecycle demonstrated by this app:**

1. **Data Ingestion** — load raw CSVs from source systems (Data Hub upload).
2. **ETL / ELT Pipeline** — Extract → Transform → Load. We extract on upload, transform in the
   self-healing pipeline, and load the cleaned output for analytics.
3. **Data Cleaning** — impute missing values, fix type errors, remove duplicates, trim whitespace.
4. **Data Transformation** — standardize schemas, cap outliers, derive numeric columns.
5. **Data Warehouse** — the cleaned, conformed data is the "single source of truth" the Analyst view reads.
6. **Data Quality Checks** — 6-dimension scoring (completeness, accuracy, consistency, …).
7. **Pipeline Automation** — the  Agentic Pipeline chains every step in one orchestrated run.
        """)
        load_three_js("three_network.html", height=300)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PIPELINE MONITORING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Pipeline Monitoring":
    st.markdown("## Pipeline Monitoring")
    st.markdown('<p class="hero-sub">Real-time pipeline health, freshness, throughput and alerts.</p>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Upload a dataset in the Data Hub to monitor a live pipeline run.")
        st.stop()

    df = st.session_state.df
    metrics = monitoring.compute_metrics(df)
    alerts = monitoring.build_alerts(metrics, df)
    fresh_date, fresh_days = monitoring.freshness(df)

    ok = metrics["success"]
    st.markdown(f"""<div class="{'alert-ok' if ok else 'alert-critical'}">
        <b>{' Pipeline SUCCESS' if ok else ' Pipeline FAILURE'}</b> — last run {metrics['last_run']}
    </div>""", unsafe_allow_html=True)

    m = st.columns(3)
    with m[0]: st.markdown(render_metric("Records Processed", f"{metrics['records']:,}", icon=""), unsafe_allow_html=True)
    with m[1]: st.markdown(render_metric("Error Count", metrics["errors"], icon=""), unsafe_allow_html=True)
    with m[2]: st.markdown(render_metric("Processing Time", f"{metrics['processing_ms']} ms", icon="⏱"), unsafe_allow_html=True)
    m2 = st.columns(3)
    with m2[0]: st.markdown(render_metric("Data Freshness", f"{fresh_date} ({fresh_days}d)" if fresh_date else "n/a", icon=""), unsafe_allow_html=True)
    with m2[1]: st.markdown(render_metric("Error Ratio", f"{metrics['error_ratio']}%", icon=""), unsafe_allow_html=True)
    with m2[2]: st.markdown(render_metric("Last Run", metrics["last_run"].split(' ')[1], icon=""), unsafe_allow_html=True)

    # ─── SLA / throughput gauges ────────────────────────────────────────────────
    audit_df = monitoring.load_audit_log()
    if audit_df is not None and "status" in audit_df.columns:
        uptime = round(100 * (audit_df["status"] == "SUCCESS").mean(), 1)
    else:
        uptime = 100.0 if metrics["success"] else 60.0
    throughput_score = min(100, round(metrics["records"] / 100, 1))
    speed_score = max(0, 100 - min(100, metrics["processing_ms"] / 20))
    g = st.columns(3)
    with g[0]:
        st.plotly_chart(mini_gauge(uptime, "Pipeline Uptime / SLA %", ref=95), use_container_width=True)
    with g[1]:
        st.plotly_chart(mini_gauge(throughput_score, "Throughput Index", ref=50), use_container_width=True)
    with g[2]:
        st.plotly_chart(mini_gauge(round(speed_score, 1), "Processing Speed Score", ref=70), use_container_width=True)

    st.markdown("#### Alerts")
    css_map = {"critical": "alert-critical", "warning": "alert-warning", "ok": "alert-ok"}
    for level, title, msg in alerts:
        st.markdown(f'<div class="{css_map[level]}"><b>{title}</b> — {msg}</div>', unsafe_allow_html=True)

    audit_df = monitoring.load_audit_log()
    if audit_df is not None:
        st.markdown("#### Recent Pipeline Runs")
        counts = audit_df["status"].value_counts().reindex(["SUCCESS", "WARNING", "FAILED"], fill_value=0)
        cc = st.columns([1.2, 1.2, 1.6])
        with cc[0]:
            run_bar = px.bar(
                x=counts.index,
                y=counts.values,
                color=counts.index,
                color_discrete_map={"SUCCESS": "#22c55e", "WARNING": "#f59e0b", "FAILED": "#ef4444"},
                title="Run outcomes",
            )
            run_bar.update_layout(showlegend=False, xaxis_title="Status", yaxis_title="Runs")
            style_fig(run_bar, 300)
            st.plotly_chart(run_bar, use_container_width=True)
        with cc[1]:
            trend_df = audit_df.copy().head(12)
            trend_df = trend_df.iloc[::-1].reset_index(drop=True)
            trend_df["run_number"] = trend_df.index + 1
            trend_df["error_count"] = pd.to_numeric(trend_df.get("error_count", 0), errors="coerce").fillna(0)
            trend_fig = px.line(trend_df, x="run_number", y="error_count", markers=True, title="Recent error trend")
            trend_fig.update_traces(line_color="#38bdf8")
            trend_fig.update_layout(xaxis_title="Run", yaxis_title="Errors")
            style_fig(trend_fig, 300)
            st.plotly_chart(trend_fig, use_container_width=True)
        with cc[2]:
            st.dataframe(audit_df, use_container_width=True, height=300)

    st.markdown("#### 🤖 AI Incident Triage")
    st.caption("AI reviews live metrics + recent runs and proposes an on-call response plan.")
    fail_runs = int((audit_df["status"] == "FAILED").sum()) if audit_df is not None and "status" in audit_df.columns else 0
    mon_prompt = (
        f"Monitoring snapshot for '{st.session_state.df_name}': status={'SUCCESS' if metrics['success'] else 'FAILURE'}, "
        f"records={metrics['records']}, errors={metrics['errors']}, error_ratio={metrics['error_ratio']}%, "
        f"processing_ms={metrics['processing_ms']}, uptime={uptime}%, failed_runs_recent={fail_runs}, "
        f"freshness_days={fresh_days}. As an SRE/data-platform engineer, give an incident-triage summary, "
        "the most likely failure risks, and an ordered on-call action plan (rollback, retry, alerting, root-cause)."
    )
    render_ai_brief(mon_prompt, key="monitoring", button_label="🚨 Run AI incident triage")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA ENGINEERING GUIDE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Data Engineering Guide":
    st.markdown("## Data Engineering — Concepts & Demo")
    st.markdown('<p class="hero-sub">The seven building blocks of a modern data pipeline, demonstrated on your data.</p>', unsafe_allow_html=True)
    load_three_js("three_network.html", height=300)

    steps = [
        (" Data Ingestion", "Collect raw data from sources (files, APIs, databases). Here: CSV upload in theData Hub."),
        (" ETL / ELT", "Extract → Transform → Load. ETL transforms before loading; ELT loads first then transforms in the warehouse."),
        (" Data Cleaning", "Handle missing values, fix wrong types, remove duplicates, standardize text — the Self-Healing Pipeline."),
        (" Data Transformation", "Reshape, aggregate, derive and conform data to a target schema for analytics."),
        (" Data Warehouse", "A central, query-optimized store (star schema: fact + dimension tables) — the single source of truth."),
        (" Data Quality Checks", "Validate completeness, accuracy, consistency, uniqueness, timeliness & validity."),
        (" Pipeline Automation", "Schedule & orchestrate the whole flow (Airflow / LangGraph) — see the Agentic Pipeline."),
    ]
    for i in range(0, len(steps), 2):
        cc = st.columns(2)
        for (title, desc), c in zip(steps[i:i+2], cc):
            with c:
                st.markdown(f"""<div class="data-card">
                    <h3 style="font-size:1.05rem;margin-bottom:6px">{title}</h3>
                    <p style="font-size:0.86rem;opacity:0.75">{desc}</p>
                </div>""", unsafe_allow_html=True)

    if st.session_state.df is not None:
        st.markdown("#### Mini ETL demo on your data")
        df = st.session_state.df
        before = df.isnull().sum().sum()
        p = SelfHealingPipeline()
        iss = p.detect_issues(df)
        fixed, _ = p.auto_fix(df.copy(), iss)
        after = fixed.isnull().sum().sum()
        d1, d2, d3 = st.columns(3)
        with d1: st.markdown(render_metric("Missing — Before", int(before), icon=""), unsafe_allow_html=True)
        with d2: st.markdown(render_metric("Missing — After", int(after), icon=""), unsafe_allow_html=True)
        with d3: st.markdown(render_metric("Issue Groups", p.stats["total_issues"], icon=""), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA ANALYST VIEW  (Power BI-style)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Data Analyst View":
    st.markdown("## Data Analyst View")
    st.markdown('<p class="hero-sub">KPIs · trends · regional & product performance · customers · forecasting · insights.</p>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Upload a dataset in the Data Hub (try the Navneet / Walmart sales demos) to see analytics.")
        st.stop()

    df = st.session_state.df
    k = analytics.kpis(df)                       # kept: used by the AI insights button

    # ─── Step 1: smart auto-detecting KPI band (top 6 non-None) ─────────────────
    kpis = analyst.detect_kpis(df)
    st.session_state.analyst_kpis = kpis
    _kpi_meta = [
        ("total_revenue",    "Total Revenue",    "💰", lambda v: f"₹{v:,.0f}"),
        ("total_quantity",   "Total Quantity",   "📦", lambda v: f"{v:,.0f}"),
        ("avg_transaction",  "Avg Transaction",  "🧾", lambda v: f"₹{v:,.0f}"),
        ("unique_customers", "Unique Customers", "👥", lambda v: f"{v:,}"),
        ("growth_pct",       "Growth",           "📈", lambda v: f"{v:+.1f}%"),
        ("top_region",       "Top Region",       "🌍", lambda v: str(v)),
        ("avg_discount",     "Avg Discount",     "🏷️", lambda v: f"{v:.1f}%"),
        ("profit_margin",    "Profit Margin",    "💹", lambda v: f"{v:.1f}%"),
        ("total_records",    "Total Records",    "🗂️", lambda v: f"{v:,}"),
    ]
    _shown = [(lbl, ic, fmt(kpis[key]), key) for key, lbl, ic, fmt in _kpi_meta
              if kpis.get(key) is not None][:6]
    for (lbl, ic, valstr, key), col in zip(_shown, st.columns(6)):
        with col:
            if key == "growth_pct":
                gcolor = "#22c55e" if kpis["growth_pct"] >= 0 else "#ef4444"
                st.markdown(f'<div class="metric-card animate-in"><div style="font-size:1.6rem">{ic}</div>'
                            f'<div class="metric-value" style="color:{gcolor}">{valstr}</div>'
                            f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(render_metric(lbl, valstr, icon=ic), unsafe_allow_html=True)
    _mrd = analyst.most_recent_date(df)
    st.markdown(f'<div class="alert-info">🕒 {"Data as of " + str(_mrd.date()) if _mrd is not None else "Live session data"}</div>',
                unsafe_allow_html=True)

    # ─── Step 2: global filters bar → df_view (never mutate session df) ─────────
    date_col = analyst.detect_date_col(df)
    cat_col_f = analyst.detect_category_col(df)
    reg_col_f = analyst.detect_region_col(df)
    df_view = df.copy()
    with st.container(border=True):
        st.markdown("**🔎 Global Filters**")
        fcols = st.columns([2, 2, 2, 1])
        if date_col:
            dser = pd.to_datetime(df[date_col], errors="coerce")
            dmin, dmax = dser.min(), dser.max()
            if pd.notna(dmin) and pd.notna(dmax):
                with fcols[0]:
                    dr = st.date_input("Date range", value=(dmin.date(), dmax.date()), key="av_date")
                if isinstance(dr, (list, tuple)) and len(dr) == 2:
                    df_view = analyst.filter_by_date(df_view, date_col, dr[0], dr[1])
        if cat_col_f and df[cat_col_f].nunique() <= 60:
            opts = sorted(df[cat_col_f].dropna().astype(str).unique().tolist())
            with fcols[1]:
                sel = st.multiselect(str(cat_col_f), opts, default=opts, key="av_cats")
            if sel:
                df_view = df_view[df_view[cat_col_f].astype(str).isin(sel)]
        if reg_col_f and reg_col_f != cat_col_f and df[reg_col_f].nunique() <= 60:
            optsr = sorted(df[reg_col_f].dropna().astype(str).unique().tolist())
            with fcols[2]:
                selr = st.multiselect(str(reg_col_f), optsr, default=optsr, key="av_regs")
            if selr:
                df_view = df_view[df_view[reg_col_f].astype(str).isin(selr)]
        with fcols[3]:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            if st.button("Reset", use_container_width=True, key="av_reset"):
                for kkey in ("av_date", "av_cats", "av_regs"):
                    st.session_state.pop(kkey, None)
                st.rerun()
    _pct = (len(df_view) / max(1, len(df))) * 100
    st.markdown(f'<div class="alert-info">Showing <b>{len(df_view):,}</b> of <b>{len(df):,}</b> rows ({_pct:.0f}% of dataset)</div>',
                unsafe_allow_html=True)

    _PERIOD_RULE = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Quarterly": "QE"}
    _PERIOD_FREQ = {"D": "D", "W": "W", "ME": "M", "QE": "Q"}
    rev_col = analyst.detect_revenue_col(df_view)

    ta, tb, tc, td, te, tf, tg = st.tabs(
        [" Sales Trend", " Regional", " Product", " Customers",
         " Forecast & Insights", " Financial Analysis", " Export Analysis"])

    # ─── Tab: Sales Trend ───────────────────────────────────────────────────────
    with ta:
        period = st.radio("Period", ["Daily", "Weekly", "Monthly", "Quarterly"],
                          horizontal=True, key="analyst_period")
        rule = _PERIOD_RULE[period]
        ser = analyst.resample_revenue(df_view, date_col, rev_col, rule) if (date_col and rev_col) else pd.Series(dtype=float)
        ser = ser[ser != 0] if not ser.empty else ser
        if not ser.empty:
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                ma = ser.rolling(3, min_periods=1).mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ser.index, y=ser.values, name="Revenue", mode="lines",
                                         fill="tozeroy", fillcolor="rgba(0,212,255,0.15)", line=dict(color="#00d4ff", width=2)))
                fig.add_trace(go.Scatter(x=ma.index, y=ma.values, name="3-period MA", line=dict(color="#f59e0b", width=2, dash="dot")))
                fig.update_layout(title=f"{period} revenue trend")
                style_fig(fig, 320); st.plotly_chart(fig, use_container_width=True)
            with c2:
                growth = ser.pct_change().fillna(0) * 100
                colors = ["#22c55e" if g >= 0 else "#ef4444" for g in growth.values]
                gfig = go.Figure(go.Bar(x=[str(i)[:10] for i in growth.index], y=growth.values, marker_color=colors))
                gfig.update_layout(title="Period Growth %")
                style_fig(gfig, 320); st.plotly_chart(gfig, use_container_width=True)
            with c3:
                slope = np.polyfit(np.arange(len(ser)), ser.values, 1)[0] if len(ser) > 1 else 0
                direction = "Upward" if slope > 0 else "Downward" if slope < 0 else "Flat"
                st.markdown(f'''<div class="data-card"><b>Trend Summary</b><br><br>
                    🔼 Peak: <b>{str(ser.idxmax())[:10]}</b> — ₹{ser.max():,.0f}<br>
                    🔽 Lowest: <b>{str(ser.idxmin())[:10]}</b> — ₹{ser.min():,.0f}<br>
                    📊 Avg/period: ₹{ser.mean():,.0f}<br>
                    📈 Trend: <b>{direction}</b></div>''', unsafe_allow_html=True)
            chan_col = analyst.find_col(df_view, ["sales_channel", "channel"])
            if chan_col:
                tmp = df_view.copy()
                tmp["_p"] = pd.to_datetime(tmp[date_col], errors="coerce").dt.to_period(_PERIOD_FREQ[rule]).astype(str)
                tmp["_v"] = pd.to_numeric(tmp[rev_col], errors="coerce")
                piv = tmp.dropna(subset=["_v"]).groupby(["_p", chan_col])["_v"].sum().reset_index()
                st.markdown("##### Sales Channel Breakdown")
                ch = px.bar(piv, x="_p", y="_v", color=chan_col, barmode="stack", title="Revenue by channel")
                style_fig(ch, 320); st.plotly_chart(ch, use_container_width=True)
        else:
            trend = analytics.sales_trend(df_view)
            if trend is not None:
                fig = px.area(trend, x="date", y="sales", title="Sales over time")
                fig.update_traces(line_color="#00d4ff", fillcolor="rgba(0,212,255,0.18)")
                style_fig(fig, 340); st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No date + revenue columns detected for a trend.")

    # ─── Tab: Regional ──────────────────────────────────────────────────────────
    with tb:
        reg = analytics.group_performance(df_view, "region")
        if reg is not None:
            fig = px.bar(reg, x=reg.columns[0], y="value", color="value", color_continuous_scale="Tealgrn", title="Performance by region")
            style_fig(fig, 320); st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No region column detected.")
        reg_col = analyst.detect_region_col(df_view)
        if reg_col and rev_col:
            grp = df_view.assign(_v=pd.to_numeric(df_view[rev_col], errors="coerce")).groupby(reg_col)["_v"].sum().sort_values(ascending=False)
            grp = grp[grp.notna()]
            st.markdown("##### State-Level Drill Down")
            top15 = grp.head(15)
            hb = go.Figure(go.Bar(x=top15.values, y=top15.index.astype(str), orientation="h",
                                  marker=dict(color=top15.values, colorscale="Tealgrn"),
                                  text=[f"{v:,.0f}" for v in top15.values], textposition="outside"))
            hb.update_layout(title="Top regions by revenue", yaxis=dict(autorange="reversed"))
            style_fig(hb, 380); st.plotly_chart(hb, use_container_width=True)
            top6 = grp.head(6); others = grp.iloc[6:].sum()
            pie_vals = list(top6.values) + ([others] if others > 0 else [])
            pie_names = list(top6.index.astype(str)) + (["Others"] if others > 0 else [])
            pf = px.pie(values=pie_vals, names=pie_names, hole=0.45, title="Regional revenue share")
            style_fig(pf, 320); st.plotly_chart(pf, use_container_width=True)
            avg = grp.mean()
            def _reg_card(s, title, border):
                rows = "".join(f"<li><b>{idx}</b> — ₹{val:,.0f} ({((val-avg)/avg*100 if avg else 0):+.0f}% vs avg)</li>" for idx, val in s.items())
                return f'<div class="data-card" style="border:1px solid {border}"><b>{title}</b><ul style="margin:8px 0 0 16px">{rows}</ul></div>'
            cc1, cc2 = st.columns(2)
            with cc1: st.markdown(_reg_card(grp.head(3), "Top 3 Regions", "#22c55e"), unsafe_allow_html=True)
            with cc2: st.markdown(_reg_card(grp.tail(3), "Bottom 3 Regions", "#ef4444"), unsafe_allow_html=True)
            if date_col:
                st.markdown("##### Top Regions — Revenue Trend")
                top5 = grp.head(5).index.astype(str).tolist()
                tmp = df_view.copy()
                tmp["_p"] = pd.to_datetime(tmp[date_col], errors="coerce").dt.to_period("M").astype(str)
                tmp["_v"] = pd.to_numeric(tmp[rev_col], errors="coerce")
                tmp = tmp[tmp[reg_col].astype(str).isin(top5)]
                line = tmp.groupby(["_p", reg_col])["_v"].sum().reset_index()
                lf = px.line(line, x="_p", y="_v", color=reg_col, markers=True, title="Top Regions — Revenue Trend")
                style_fig(lf, 320); st.plotly_chart(lf, use_container_width=True)

    # ─── Tab: Product ───────────────────────────────────────────────────────────
    with tc:
        prod = analytics.group_performance(df_view, "product")
        if prod is not None:
            fig = px.bar(prod, x="value", y=prod.columns[0], orientation="h", color="value", color_continuous_scale="Purp", title="Top products")
            style_fig(fig, 320); fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No product column detected.")
        prod_col = analyst.find_col(df_view, ["product_name", "product", "item"])
        cat_col = analyst.find_col(df_view, ["category", "subject", "class"])
        qty_col = analyst.detect_quantity_col(df_view)
        disc_col = analyst.find_col(df_view, ["discount"], numeric=True)
        if prod_col and rev_col and qty_col:
            st.markdown("##### Product Revenue vs Volume Matrix")
            g = df_view.assign(_rev=pd.to_numeric(df_view[rev_col], errors="coerce"),
                               _qty=pd.to_numeric(df_view[qty_col], errors="coerce"),
                               _disc=pd.to_numeric(df_view[disc_col], errors="coerce") if disc_col else 1.0)
            agg = g.groupby(prod_col).agg(_rev=("_rev", "sum"), _qty=("_qty", "sum"), _disc=("_disc", "mean"))
            agg["_cat"] = g.groupby(prod_col)[cat_col].first() if (cat_col and cat_col != prod_col) else "All"
            agg = agg.sort_values("_rev", ascending=False).head(30).reset_index()
            agg["_disc"] = agg["_disc"].fillna(0).clip(lower=0.1)
            bub = px.scatter(agg, x="_qty", y="_rev", size="_disc", color="_cat", hover_name=prod_col,
                             title="Product Revenue vs Volume Matrix (bubble = avg discount)",
                             labels={"_qty": "Quantity sold", "_rev": "Revenue", "_cat": "Category"})
            style_fig(bub, 360); st.plotly_chart(bub, use_container_width=True)
        if cat_col and rev_col:
            st.markdown("##### Category Analysis")
            g = df_view.assign(_rev=pd.to_numeric(df_view[rev_col], errors="coerce"),
                               _qty=pd.to_numeric(df_view[qty_col], errors="coerce") if qty_col else 0,
                               _disc=pd.to_numeric(df_view[disc_col], errors="coerce") if disc_col else 0)
            cg = g.groupby(cat_col).agg(revenue=("_rev", "sum"), quantity=("_qty", "sum"), avg_discount=("_disc", "mean")).reset_index()
            cg["revenue_share_%"] = (cg["revenue"] / cg["revenue"].sum() * 100).round(1)
            cg = cg.sort_values("revenue", ascending=False)
            tm = px.treemap(cg, path=[cat_col], values="revenue", color="revenue", color_continuous_scale="Blues", title="Revenue by category")
            style_fig(tm, 320); st.plotly_chart(tm, use_container_width=True)
            st.dataframe(cg, use_container_width=True)
        stock_col = analyst.find_col(df_view, ["stock", "inventory", "units_available"], numeric=True)
        if stock_col and prod_col:
            st.markdown("##### Stock Status")
            sdf = df_view.assign(_s=pd.to_numeric(df_view[stock_col], errors="coerce")).dropna(subset=["_s"])
            sc = sdf.groupby(prod_col)["_s"].sum()
            top20 = sc.sort_values(ascending=False).head(20)
            sb = go.Figure(go.Bar(x=top20.index.astype(str), y=top20.values, marker_color="#38bdf8"))
            sb.add_hline(y=top20.max() * 0.1, line_dash="dash", line_color="#ef4444", annotation_text="Reorder threshold")
            sb.update_layout(title="Stock levels (top 20)")
            style_fig(sb, 320); st.plotly_chart(sb, use_container_width=True)
            mc = st.columns(3)
            with mc[0]: st.markdown(render_metric("In Stock", int((sc >= 500).sum()), icon="✅"), unsafe_allow_html=True)
            with mc[1]: st.markdown(render_metric("Low Stock (<500)", int(((sc > 0) & (sc < 500)).sum()), icon="⚠️"), unsafe_allow_html=True)
            with mc[2]: st.markdown(render_metric("Out of Stock", int((sc <= 0).sum()), icon="⛔"), unsafe_allow_html=True)

    # ─── Tab: Customers ─────────────────────────────────────────────────────────
    with td:
        cust = analytics.customer_analysis(df_view)
        if cust is not None:
            fig = px.bar(cust, x=cust.columns[0], y="spend", color="spend", color_continuous_scale="Blues", title="Top customers")
            style_fig(fig, 320); st.plotly_chart(fig, use_container_width=True)
        cust_col = analyst.detect_customer_col(df_view)
        seg_col = analyst.find_col(df_view, ["segment", "customer_type"])
        if seg_col:
            st.markdown("##### Customer Segment Summary")
            scv = df_view[seg_col].astype(str).value_counts()
            do = px.pie(values=scv.values, names=scv.index, hole=0.55, title="Customers by segment")
            style_fig(do, 300); st.plotly_chart(do, use_container_width=True)
            if rev_col:
                g = df_view.assign(_rev=pd.to_numeric(df_view[rev_col], errors="coerce"))
                seg_tab = g.groupby(seg_col).agg(count=(seg_col, "size"), total_rev=("_rev", "sum"))
                seg_tab["avg_rev_per_customer"] = (seg_tab["total_rev"] / seg_tab["count"]).round(0)
                seg_tab["pct_of_revenue"] = (seg_tab["total_rev"] / seg_tab["total_rev"].sum() * 100).round(1)
                st.dataframe(seg_tab.reset_index(), use_container_width=True)
        if cust_col:
            st.markdown("##### RFM-Inspired Scoring")
            try:
                rfm = analyst.compute_rfm(df_view, cust_col, date_col, rev_col)
                def _hl(v):
                    c = "#22c55e" if v >= 4 else "#f59e0b" if v >= 2.5 else "#ef4444"
                    return f"color:{c};font-weight:700"
                st.dataframe(rfm.head(10).style.map(_hl, subset=["RFM_Score"]).format({"RFM_Score": "{:.2f}", "Monetary": "{:,.0f}", "Recency": "{:.0f}"}),
                             use_container_width=True)
                rfm["_size"] = rfm["Frequency"].clip(lower=1)
                rfm["_Rrank"] = rfm["Recency"].rank()
                rfm["_Mrank"] = rfm["Monetary"].rank()
                if seg_col:
                    seg_map = df_view.groupby(cust_col)[seg_col].first().astype(str)
                    rfm["_seg"] = rfm[cust_col].map(seg_map)
                sc2 = px.scatter(rfm, x="_Rrank", y="_Mrank", size="_size", color="_seg" if seg_col else None,
                                 title="Customer RFM Landscape", labels={"_Rrank": "Recency rank", "_Mrank": "Monetary rank"})
                style_fig(sc2, 320); st.plotly_chart(sc2, use_container_width=True)
            except Exception as _e:
                st.info(f"RFM unavailable for this dataset: {_e}")
        loy_col = analyst.find_col(df_view, ["loyalty_points", "loyalty", "points"], numeric=True)
        if loy_col:
            st.markdown("##### Loyalty Points Analysis")
            lp = pd.to_numeric(df_view[loy_col], errors="coerce")
            lc1, lc2 = st.columns(2)
            with lc1:
                h = px.histogram(x=lp.dropna(), nbins=20, title="Loyalty points distribution")
                style_fig(h, 300); st.plotly_chart(h, use_container_width=True)
            with lc2:
                name_col = cust_col or df_view.columns[0]
                topl = df_view.assign(_lp=lp).groupby(name_col)["_lp"].sum().sort_values(ascending=False).head(10)
                hb = go.Figure(go.Bar(x=topl.values, y=topl.index.astype(str), orientation="h", marker_color="#a855f7"))
                hb.update_layout(title="Top 10 by loyalty points", yaxis=dict(autorange="reversed"))
                style_fig(hb, 300); st.plotly_chart(hb, use_container_width=True)
            lpd_col = analyst.find_col(df_view, ["last_purchase", "last_order", "last_seen"])
            if lpd_col:
                days = (pd.Timestamp.now() - pd.to_datetime(df_view[lpd_col], errors="coerce")).dt.days
                risk = df_view[(days > 90) & (lp < lp.median())]
                st.markdown(f'<div class="alert-warning">⚠️ <b>{len(risk)}</b> customers flagged as <b>Churn Risk</b> (last purchase &gt; 90 days &amp; loyalty &lt; median).</div>', unsafe_allow_html=True)
                if len(risk):
                    st.dataframe(risk.head(20), use_container_width=True, height=200)

    # ─── Tab: Forecast & Insights ───────────────────────────────────────────────
    with te:
        trend = analytics.sales_trend(df_view)
        fc = analytics.forecast(trend)
        if trend is not None and fc is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend["date"], y=trend["sales"], name="Actual", line=dict(color="#00d4ff", width=2)))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], name="Forecast", line=dict(color="#a855f7", width=2, dash="dash")))
            style_fig(fig, 320); fig.update_layout(title="Sales forecast (linear trend)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need a date + sales series for forecasting.")

        period = st.session_state.get("analyst_period", "Monthly")
        rule = _PERIOD_RULE[period]
        if date_col and rev_col:
            ser = analyst.resample_revenue(df_view, date_col, rev_col, rule)
            ser = ser[ser != 0]
            if len(ser) >= 3:
                st.markdown("##### Revenue Forecast — Next 6 Periods")
                fdf = analyst.forecast_linear(ser, 6)
                hist = fdf[fdf["kind"] == "history"]; fut = fdf[fdf["kind"] == "forecast"]
                ff = go.Figure()
                ff.add_trace(go.Scatter(x=list(fut["period"]) + list(fut["period"][::-1]),
                                        y=list(fut["upper"]) + list(fut["lower"][::-1]),
                                        fill="toself", fillcolor="rgba(168,85,247,0.15)",
                                        line=dict(color="rgba(0,0,0,0)"), name="Confidence band"))
                ff.add_trace(go.Scatter(x=hist["period"], y=hist["forecast"], name="Actual", line=dict(color="#00d4ff", width=2)))
                ff.add_trace(go.Scatter(x=fut["period"], y=fut["forecast"], name="Forecast", line=dict(color="#a855f7", width=2, dash="dash")))
                ff.add_vline(x=hist["period"].iloc[-1], line_dash="dot", line_color="#94a3b8")
                ff.update_layout(title="Revenue Forecast — Next 6 Periods")
                style_fig(ff, 340); st.plotly_chart(ff, use_container_width=True)
            monthly = analyst.resample_revenue(df_view, date_col, rev_col, "ME")
            if len(monthly) >= 12:
                st.markdown("##### Revenue Seasonality Pattern")
                mdf = pd.DataFrame({"v": monthly.values, "m": monthly.index.month})
                seas = mdf.groupby("m")["v"].mean().reindex(range(1, 13)).fillna(0)
                months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                rad = go.Figure(go.Scatterpolar(r=list(seas.values) + [seas.values[0]], theta=months + [months[0]],
                                                fill="toself", line=dict(color="#00d4ff")))
                rad.update_layout(title="Revenue Seasonality Pattern", polar=dict(bgcolor="rgba(0,0,0,0)"))
                style_fig(rad, 360); st.plotly_chart(rad, use_container_width=True)
                idxv = seas.max() / seas.mean() if seas.mean() else 0
                st.markdown(f'<div class="alert-info">Peak season: <b>{months[int(seas.idxmax())-1]}</b> · '
                            f'Off-peak: <b>{months[int(seas.idxmin())-1]}</b> · Seasonality index: <b>{idxv:.2f}x</b></div>',
                            unsafe_allow_html=True)

        st.markdown("##### Business Health Score")
        hs, drivers, drags = analyst.business_health_score(df_view, st.session_state.quality_report)
        hc1, hc2 = st.columns([1, 1.4])
        with hc1:
            st.plotly_chart(mini_gauge(hs, "Business Health", ref=80), use_container_width=True)
        with hc2:
            st.markdown("**What's driving your score**")
            for d in (drivers or ["—"]):
                st.markdown(f'<div class="alert-ok">{d}</div>', unsafe_allow_html=True)
            st.markdown("**What's dragging it down**")
            for d in (drags or ["Nothing major — healthy across the board."]):
                st.markdown(f'<div class="alert-warning">{d}</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button(" AI: Summarize insights & recommend actions", key="da_ai_insights"):
            if st.session_state.api_key:
                agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
                with st.spinner(" Analyzing KPIs..."):
                    summary = agent.summarize_dashboard_insights(k, st.session_state.df_name)
                    actions = agent.recommend_business_actions(k, st.session_state.df_name)
                log_action("AI_INSIGHTS", st.session_state.df_name)
                st.markdown("#### Insights"); st.markdown(summary)
                st.markdown("#### Recommended Business Actions"); st.markdown(actions)
            else:
                st.warning(" Add your OpenRouter API key in `.env` to enable AI insights.")

    # ─── Tab: Financial Analysis ────────────────────────────────────────────────
    with tf:
        fin = analyst.detect_financial_cols(df_view)
        if len(fin) < 2:
            st.info("Load Navneet Financials or a dataset with revenue/profit/margin columns.")
        else:
            fdf = df_view.copy()
            if "quarter" in fin and "year" in fin:
                fdf["_period"] = fdf[fin["quarter"]].astype(str) + " " + fdf[fin["year"]].astype(str)
            elif "year" in fin:
                fdf["_period"] = fdf[fin["year"]].astype(str)
            elif "quarter" in fin:
                fdf["_period"] = fdf[fin["quarter"]].astype(str)
            else:
                fdf["_period"] = [f"P{i+1}" for i in range(len(fdf))]
            _num = lambda c: pd.to_numeric(fdf[c], errors="coerce")
            rev = _num(fin["revenue"]) if "revenue" in fin else None
            np_ = _num(fin["net_profit"]) if "net_profit" in fin else None
            gm = _num(fin["gross_margin"]) if "gross_margin" in fin else None
            yoy = _num(fin["yoy_growth"]) if "yoy_growth" in fin else None
            p = st.columns(4)
            with p[0]: st.markdown(render_metric("Total Revenue", f"{rev.sum():,.0f}" if rev is not None else "n/a", icon="💰"), unsafe_allow_html=True)
            with p[1]: st.markdown(render_metric("Total Net Profit", f"{np_.sum():,.0f}" if np_ is not None else "n/a", icon="💹"), unsafe_allow_html=True)
            with p[2]: st.markdown(render_metric("Avg Gross Margin %", f"{gm.mean():.1f}%" if gm is not None else "n/a", icon="📊"), unsafe_allow_html=True)
            with p[3]: st.markdown(render_metric("Avg YoY Growth %", f"{yoy.mean():.1f}%" if yoy is not None else "n/a", icon="📈"), unsafe_allow_html=True)
            if rev is not None and np_ is not None:
                st.markdown("##### Revenue, Profit, and Margin Over Time")
                sub = make_subplots(specs=[[{"secondary_y": True}]])
                sub.add_trace(go.Bar(x=fdf["_period"], y=rev, name="Revenue", marker_color="#00d4ff"), secondary_y=False)
                sub.add_trace(go.Bar(x=fdf["_period"], y=np_, name="Net Profit", marker_color="#22c55e"), secondary_y=True)
                if gm is not None:
                    sub.add_trace(go.Scatter(x=fdf["_period"], y=gm, name="Gross Margin %", line=dict(color="#f59e0b", width=2)), secondary_y=True)
                sub.update_layout(title="Revenue, Profit, and Margin Over Time", barmode="group")
                style_fig(sub, 360); st.plotly_chart(sub, use_container_width=True)
            if "gross_margin" in fin and "operating_margin" in fin:
                st.markdown("##### Margin Waterfall")
                gmv = _num(fin["gross_margin"]).mean(); omv = _num(fin["operating_margin"]).mean()
                nmv = (np_.sum() / rev.sum() * 100) if (np_ is not None and rev is not None and rev.sum()) else omv * 0.7
                wf = go.Figure(go.Waterfall(x=["Gross Margin", "Operating Margin", "Net Margin"],
                                            y=[gmv, omv - gmv, nmv - omv], measure=["absolute", "relative", "relative"]))
                wf.update_layout(title="Margin bridge (%)")
                style_fig(wf, 320); st.plotly_chart(wf, use_container_width=True)
            st.markdown("##### Financial Ratios")
            ratios = pd.DataFrame({"Period": fdf["_period"]})
            if rev is not None: ratios["Revenue"] = rev.values
            if np_ is not None and rev is not None: ratios["Profit Margin %"] = (np_ / rev * 100).round(1).values
            if "ebitda" in fin and rev is not None: ratios["EBITDA Margin %"] = (_num(fin["ebitda"]) / rev * 100).round(1).values
            if "eps" in fin: ratios["EPS"] = _num(fin["eps"]).values
            st.dataframe(ratios, use_container_width=True)
            if "domestic_revenue" in fin and "export_revenue" in fin:
                st.markdown("##### Domestic vs Export Revenue")
                dom = _num(fin["domestic_revenue"]); exr = _num(fin["export_revenue"])
                ds = go.Figure()
                ds.add_trace(go.Bar(x=fdf["_period"], y=dom, name="Domestic", marker_color="#38bdf8"))
                ds.add_trace(go.Bar(x=fdf["_period"], y=exr, name="Export", marker_color="#a855f7"))
                ds.update_layout(barmode="stack", title="Domestic vs Export Revenue")
                style_fig(ds, 320); st.plotly_chart(ds, use_container_width=True)
                tot = exr.sum() + dom.sum()
                st.markdown(f'<div class="alert-info">Export Revenue Share: <b>{(exr.sum()/tot*100 if tot else 0):.1f}%</b> of total.</div>', unsafe_allow_html=True)
            if st.button("Generate AI Financial Commentary", key="fin_ai"):
                if not st.session_state.api_key:
                    st.warning("Add your OpenRouter API key in `.env` to enable AI commentary.")
                else:
                    prompt = (f"As a CFO, analyze financials for {st.session_state.df_name}: "
                              f"total revenue={rev.sum() if rev is not None else 'n/a'}, net profit={np_.sum() if np_ is not None else 'n/a'}, "
                              f"avg gross margin={gm.mean() if gm is not None else 'n/a'}, avg YoY growth={yoy.mean() if yoy is not None else 'n/a'}. "
                              "Cover revenue trend, profitability health, and the top 2 financial risks. Be concise.")
                    try:
                        agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
                        with st.spinner("🧠 CFO analysis..."):
                            st.session_state.financial_commentary = agent.chat(prompt)
                    except Exception as e:
                        st.error(f"AI error: {e}")
            if st.session_state.financial_commentary:
                st.markdown(f'<div class="data-card"><b>🧠 AI Financial Commentary</b><br><br>{st.session_state.financial_commentary}</div>', unsafe_allow_html=True)

    # ─── Tab: Export Analysis ───────────────────────────────────────────────────
    with tg:
        exp = analyst.detect_export_cols(df_view)
        if "country" not in exp:
            st.info("Load Navneet Export data or a dataset with country/market/destination columns.")
        else:
            edf = df_view.copy()
            ctry = exp["country"]
            val_col = exp.get("total_value") or rev_col
            qty_col_e = exp.get("quantity")
            cat_col_e = exp.get("category")
            status_col = exp.get("status")
            _en = lambda c: pd.to_numeric(edf[c], errors="coerce")
            val = _en(val_col) if val_col else None
            p = st.columns(5)
            with p[0]: st.markdown(render_metric("Total Export Value", f"{val.sum():,.0f}" if val is not None else "n/a", icon="💵"), unsafe_allow_html=True)
            with p[1]: st.markdown(render_metric("Total Quantity", f"{_en(qty_col_e).sum():,.0f}" if qty_col_e else "n/a", icon="📦"), unsafe_allow_html=True)
            with p[2]: st.markdown(render_metric("Markets", edf[ctry].nunique(), icon="🌍"), unsafe_allow_html=True)
            with p[3]:
                _gv = edf.assign(_v=val).groupby(ctry)["_v"].sum().dropna() if val is not None else pd.Series(dtype=float)
                if len(_gv) and _gv.abs().sum() > 0:
                    top_mkt = _gv.idxmax()
                elif edf[ctry].notna().any():
                    top_mkt = edf[ctry].mode().iloc[0]
                else:
                    top_mkt = "n/a"
                st.markdown(render_metric("Top Market", str(top_mkt), icon="🏆"), unsafe_allow_html=True)
            with p[4]:
                pend = int(edf[status_col].astype(str).str.contains("pend|transit", case=False, na=False).sum()) if status_col else 0
                st.markdown(render_metric("Pending Shipments", pend, icon="⏳"), unsafe_allow_html=True)
            if val is not None:
                st.markdown("##### Top Export Markets by Value")
                tmk = edf.assign(_v=val).groupby(ctry)["_v"].sum().sort_values(ascending=False).head(10)
                hb = go.Figure(go.Bar(x=tmk.values, y=tmk.index.astype(str), orientation="h",
                                      marker=dict(color=tmk.values, colorscale="Tealgrn"),
                                      text=[f"{v:,.0f}" for v in tmk.values], textposition="outside"))
                hb.update_layout(title="Top Export Markets by Value", yaxis=dict(autorange="reversed"))
                style_fig(hb, 360); st.plotly_chart(hb, use_container_width=True)
            if cat_col_e and val is not None:
                st.markdown("##### Product Category Export Mix")
                cg = edf.assign(_v=val).groupby(cat_col_e)["_v"].sum().reset_index()
                e1, e2 = st.columns(2)
                with e1:
                    tmf = px.treemap(cg, path=[cat_col_e], values="_v", color="_v", color_continuous_scale="Blues", title="Export value by category")
                    style_fig(tmf, 300); st.plotly_chart(tmf, use_container_width=True)
                with e2:
                    pf = px.pie(cg, values="_v", names=cat_col_e, hole=0.45, title="Category share %")
                    style_fig(pf, 300); st.plotly_chart(pf, use_container_width=True)
            if status_col:
                st.markdown("##### Export Status Distribution")
                scs = edf[status_col].astype(str).value_counts()
                do = px.pie(values=scs.values, names=scs.index, hole=0.55, title="Shipment status")
                style_fig(do, 300); st.plotly_chart(do, use_container_width=True)
                total = scs.sum()
                for (nm, cnt), scol in zip(scs.items(), st.columns(min(len(scs), 5))):
                    with scol: st.markdown(render_metric(str(nm), f"{cnt/total*100:.0f}%", icon="📦"), unsafe_allow_html=True)
                rej = sum(v for kk, v in scs.items() if "reject" in str(kk).lower())
                if total and rej / total * 100 > 5:
                    st.markdown(f'<div class="alert-warning">⚠️ Rejected shipments are {rej/total*100:.1f}% of total (&gt;5%).</div>', unsafe_allow_html=True)
            if exp.get("date") and val is not None:
                st.markdown("##### Export Trend Over Time")
                eser = analyst.resample_revenue(edf, exp["date"], val_col, "ME")
                if not eser.empty:
                    roll = eser.rolling(3, min_periods=1).mean()
                    lf = go.Figure()
                    lf.add_trace(go.Scatter(x=eser.index, y=eser.values, name="Monthly", line=dict(color="#00d4ff")))
                    lf.add_trace(go.Scatter(x=roll.index, y=roll.values, name="3-mo avg", line=dict(color="#f59e0b", dash="dot")))
                    lf.update_layout(title="Export value over time")
                    style_fig(lf, 320); st.plotly_chart(lf, use_container_width=True)
                top5c = edf.assign(_v=val).groupby(ctry)["_v"].sum().sort_values(ascending=False).head(5).index.astype(str).tolist()
                tmp = edf.assign(_v=val, _p=pd.to_datetime(edf[exp["date"]], errors="coerce").dt.to_period("M").astype(str))
                tmp = tmp[tmp[ctry].astype(str).isin(top5c)]
                cl = tmp.groupby(["_p", ctry])["_v"].sum().reset_index()
                clf = px.line(cl, x="_p", y="_v", color=ctry, markers=True, title="Top 5 markets over time")
                style_fig(clf, 320); st.plotly_chart(clf, use_container_width=True)
            if val is not None and qty_col_e:
                st.markdown("##### Market Opportunity Map — Volume vs Unit Value")
                g = edf.assign(_v=val, _q=_en(qty_col_e))
                grp = g.groupby(ctry).agg(_v=("_v", "sum"), _q=("_q", "sum"))
                grp["_upv"] = grp["_v"] / grp["_q"].replace(0, np.nan)
                if cat_col_e:
                    grp["_cat"] = g.groupby(ctry)[cat_col_e].first()
                grp = grp.reset_index().dropna(subset=["_upv"])
                om = px.scatter(grp, x="_q", y="_upv", size="_v", color="_cat" if cat_col_e else None, hover_name=ctry,
                                title="Market Opportunity Map — Volume vs Unit Value",
                                labels={"_q": "Quantity", "_upv": "Revenue per unit"})
                style_fig(om, 340); st.plotly_chart(om, use_container_width=True)
            if st.button("Generate AI Export Insights", key="exp_ai"):
                if not st.session_state.api_key:
                    st.warning("Add your OpenRouter API key in `.env` to enable AI insights.")
                else:
                    top_list = ", ".join(edf.assign(_v=val).groupby(ctry)["_v"].sum().sort_values(ascending=False).head(5).index.astype(str)) if val is not None else ""
                    prompt = (f"As an export strategy analyst for {st.session_state.df_name}, top markets by value: {top_list}. "
                              "Identify the strongest markets, underperforming markets, and 3 growth recommendations. Be concise.")
                    try:
                        agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
                        with st.spinner("🧠 Export analysis..."):
                            st.session_state.export_commentary = agent.chat(prompt)
                    except Exception as e:
                        st.error(f"AI error: {e}")
            if st.session_state.export_commentary:
                st.markdown(f'<div class="data-card"><b>🧠 AI Export Insights</b><br><br>{st.session_state.export_commentary}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ADMIN / SECURITY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Admin / Security":
    st.markdown("## Admin / Security Center")
    st.markdown('<p class="hero-sub">User access · data masking · audit logs · backups · encryption.</p>', unsafe_allow_html=True)

    ts1, ts2, ts3, ts4, ts5 = st.tabs([" Users & Access", " Data Masking", " Audit Log", " Backups", " Encryption"])

    with ts1:
        st.markdown("#### Registered users")
        users_df = sqlite_store.list_users()
        st.dataframe(users_df, use_container_width=True, height=240)

        add_col, edit_col, delete_col = st.columns(3)
        with add_col:
            with st.form("admin_add_user"):
                st.markdown("##### Add user")
                add_name = st.text_input("Full name", key="admin_add_name")
                add_username = st.text_input("Username", key="admin_add_user")
                add_role = st.selectbox("Role", [security.ADMIN, security.ENGINEER, security.ANALYST], key="admin_add_role")
                add_password = st.text_input("Password", type="password", key="admin_add_pw")
                if st.form_submit_button("Add user", use_container_width=True):
                    err = None
                    if len(add_name.strip()) < 3:
                        err = "Full name must be at least 3 characters."
                    elif len(add_username.strip()) < 3:
                        err = "Username must be at least 3 characters."
                    elif len(add_password or "") < 6:
                        err = "Password must be at least 6 characters."
                    if err:
                        st.error(err)
                    else:
                        ok, msg = sqlite_store.create_user(add_name, add_username, add_password, add_role)
                        if ok:
                            log_action("USER_ADD", add_username.strip().lower())
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
        if users_df.empty:
            st.info("No users found.")
        else:
            user_options = {f"{row.id}: {row.username} ({row.role})": int(row.id) for row in users_df.itertuples(index=False)}
            selected_user_label = edit_col.selectbox("Edit user", list(user_options.keys()), key="edit_user_select")
            selected_user_id = user_options[selected_user_label]
            selected_user = users_df[users_df["id"] == selected_user_id].iloc[0]
            with edit_col.form("admin_edit_user"):
                edit_name = st.text_input("Full name", value=str(selected_user["full_name"]), key="edit_user_name")
                edit_role = st.selectbox("Role", [security.ADMIN, security.ENGINEER, security.ANALYST], index=[security.ADMIN, security.ENGINEER, security.ANALYST].index(selected_user["role"]), key="edit_user_role")
                edit_password = st.text_input("New password (optional)", type="password", key="edit_user_pw")
                if st.form_submit_button("Update user", use_container_width=True):
                    ok, msg = sqlite_store.update_user(selected_user_id, edit_name, edit_role, edit_password)
                    if ok:
                        log_action("USER_EDIT", str(selected_user_id))
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            with delete_col.form("admin_delete_user"):
                st.markdown("##### Delete user")
                del_label = st.selectbox("User", list(user_options.keys()), key="delete_user_select")
                if st.form_submit_button("Delete user", use_container_width=True):
                    ok, msg = sqlite_store.delete_user(user_options[del_label])
                    if ok:
                        log_action("USER_DELETE", del_label)
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        st.markdown("#### Access matrix (page -> roles)")
        rows = []
        for pg, roles in security.PAGE_ACCESS.items():
            rows.append({"page": pg, "Admin": "yes",
                         "Data Engineer": "yes" if security.ENGINEER in roles else "-",
                         "Data Analyst": "yes" if security.ANALYST in roles else "-"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=320)

        st.markdown("#### Permission matrix (role -> permission)")
        prows = []
        for perm in security.ALL_PERMS:
            prows.append({"permission": perm,
                          "Admin": "yes",
                          "Data Engineer": "yes" if security.has_permission(security.ENGINEER, perm) else "-",
                          "Data Analyst": "yes" if security.has_permission(security.ANALYST, perm) else "-"})
        st.dataframe(pd.DataFrame(prows), use_container_width=True)

    with ts2:
        st.markdown("#### PII data masking")
        demo_df = st.session_state.df
        masked_cols_preview = [c for c in (demo_df.columns if demo_df is not None else []) if security._detect_kind(c)]
        if demo_df is None or not masked_cols_preview:
            st.caption("Loaded data has no PII columns — showing the Walmart Customers demo instead.")
            try:
                demo_df = load_sample_data("Walmart Customers")
            except Exception:
                demo_df = None
        if demo_df is not None:
            reveal = st.toggle(" Reveal raw PII (Admin only)", value=st.session_state.mask_reveal)
            st.session_state.mask_reveal = reveal
            if reveal:
                log_action("MASK_TOGGLE", "revealed")
            shown, masked_cols = security.mask_dataframe(demo_df, reveal=reveal)
            st.markdown(f"Masked columns: " + (", ".join(f"`{c}`" for c in masked_cols) if masked_cols else "none detected"))
            st.dataframe(shown.head(30), use_container_width=True, height=320)

    with ts3:
        st.markdown("#### Live session audit log")
        if st.session_state.audit:
            st.dataframe(pd.DataFrame(st.session_state.audit), use_container_width=True, height=240)
        else:
            st.caption("No session activity yet.")
        st.markdown("#### Employee access log (system)")
        access_path = os.path.join(os.path.dirname(__file__), "data", "employee_access_log.csv")
        if os.path.exists(access_path):
            acc = pd.read_csv(access_path)
            st.dataframe(acc, use_container_width=True, height=260)
            fails = (acc["status"] == "FAILED").sum()
            if fails:
                st.markdown(f'<div class="alert-warning"><b>{fails}</b> failed login attempt(s) detected.</div>', unsafe_allow_html=True)

    with ts4:
        st.markdown("#### Backups")
        backups = pd.DataFrame([
            {"backup_id": "BKP-20260601-0300", "scope": "warehouse_snapshot", "size_mb": 48.2, "status": " OK", "created": "2026-06-01 03:10"},
            {"backup_id": "BKP-20260531-0300", "scope": "warehouse_snapshot", "size_mb": 47.9, "status": " OK", "created": "2026-05-31 03:10"},
            {"backup_id": "BKP-20260530-0300", "scope": "warehouse_snapshot", "size_mb": 47.1, "status": " OK", "created": "2026-05-30 03:10"},
        ])
        st.dataframe(backups, use_container_width=True)
        if st.session_state.df is not None:
            if st.button(" Create backup of current dataset"):
                log_action("BACKUP", st.session_state.df_name)
                st.download_button("⬇ Download backup (CSV)",
                                   st.session_state.df.to_csv(index=False).encode(),
                                   f"backup_{(st.session_state.df_name or 'data')}.csv", "text/csv")
                st.success(" Backup created.")

    with ts5:
        st.markdown(security.encryption_explainer())

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE REMOVED: IT DATA ROLES GUIDE

# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Dashboard":
    st.markdown("## Analytics Dashboard")

    if st.session_state.df is None:
        st.info("Upload a dataset in the Data Hub to see the dashboard.")
        st.stop()

    df = st.session_state.df
    df_fixed = st.session_state.df_fixed

    # ─── Section A: full-width business summary card ────────────────────────────
    _qs = st.session_state.quality_report["quality_score"] if st.session_state.quality_report else "— (run Quality AI)"
    st.markdown(f'''<div class="data-card">
        <b>📋 {st.session_state.df_name or "Unnamed dataset"}</b> &nbsp;·&nbsp;
        {df.shape[0]:,} rows × {df.shape[1]} cols &nbsp;·&nbsp;
        loaded {datetime.now().strftime("%Y-%m-%d %H:%M")} &nbsp;·&nbsp;
        Quality score: <b>{_qs}</b></div>''', unsafe_allow_html=True)

    # Top metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    total_missing = df.isnull().sum().sum() + (df.astype(str).isin(["NULL","NaN","None","nan",""]).sum().sum())
    numeric_df = df.apply(pd.to_numeric, errors='coerce')

    with m1: st.markdown(render_metric("Total Rows", f"{len(df):,}", icon=""), unsafe_allow_html=True)
    with m2: st.markdown(render_metric("Columns", df.shape[1], icon=""), unsafe_allow_html=True)
    with m3: st.markdown(render_metric("Missing Values", f"{int(total_missing):,}", icon=""), unsafe_allow_html=True)
    with m4: st.markdown(render_metric("Numeric Cols", df.select_dtypes(include=[np.number]).shape[1], icon=""), unsafe_allow_html=True)
    with m5:
        q_score = st.session_state.quality_report["quality_score"] if st.session_state.quality_report else "—"
        col = score_color(q_score) if isinstance(q_score, (int,float)) else "gray"
        st.markdown(f"""<div class="metric-card animate-in">
            <div style="font-size:1.4rem"></div>
            <div class="metric-value" style="color:{col}">{q_score}</div>
            <div class="metric-label">Quality Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3D Network View
    col1, col2 = st.columns([3,2])
    with col1:
        st.markdown("#### Data Pipeline Network (3D)")
        load_three_js("three_network.html", height=360)
    with col2:
        st.markdown("#### Missing Values Heatmap")
        null_counts = {}
        for col in df.columns:
            n = df[col].isnull().sum() + (df[col].astype(str).isin(["NULL","NaN","None","nan",""]).sum())
            null_counts[col] = int(n)
        null_series = pd.Series(null_counts)
        if null_series.sum() > 0:
            fig = go.Figure(go.Bar(
                x=null_series[null_series>0].values,
                y=null_series[null_series>0].index,
                orientation='h',
                marker=dict(
                    color=null_series[null_series>0].values,
                    colorscale='RdYlGn_r',
                    showscale=True
                )
            ))
            fig.update_layout(
                height=340, margin=dict(l=0,r=0,t=10,b=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94a3b8", size=11),
                xaxis_title="Missing Count",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success(" No missing values detected!")

    # Data distribution charts
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.markdown("#### Numeric Column Distributions")
        sel_col = st.selectbox("Select column", numeric_cols)
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = px.histogram(df, x=sel_col, nbins=30, title=f"Distribution: {sel_col}",
                               color_discrete_sequence=["#00d4ff"])
            fig.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              margin=dict(l=0,r=0,t=30,b=0), font=dict(color="#94a3b8"))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(df, y=sel_col, title=f"Box Plot: {sel_col}", color_discrete_sequence=["#a855f7"])
            fig.update_layout(height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              margin=dict(l=0,r=0,t=30,b=0), font=dict(color="#94a3b8"))
            st.plotly_chart(fig, use_container_width=True)
        with col3:
            if df_fixed is not None and sel_col in df_fixed.columns:
                comp = pd.DataFrame({
                    "Original": pd.to_numeric(df[sel_col], errors='coerce').dropna(),
                    "Healed": pd.to_numeric(df_fixed[sel_col], errors='coerce').dropna()
                })
                fig = go.Figure()
                fig.add_trace(go.Violin(y=comp["Original"].values, name="Original", fillcolor="rgba(239,68,68,0.3)", line_color="#ef4444"))
                fig.add_trace(go.Violin(y=comp["Healed"].values, name="Healed", fillcolor="rgba(34,197,94,0.3)", line_color="#22c55e"))
                fig.update_layout(title="Before vs After", height=280, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=30,b=0), font=dict(color="#94a3b8"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Run the pipeline to see Before vs After comparison")

    # Correlation Heatmap
    if len(numeric_cols) > 1:
        st.markdown("#### Correlation Matrix")
        num_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce').dropna(how='all')
        corr = num_df.corr()
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale='RdBu', zmid=0,
            text=np.round(corr.values, 2), texttemplate='%{text}', textfont={"size": 10}
        ))
        fig.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(l=0, r=0, t=10, b=0), font=dict(color="#94a3b8"))
        st.plotly_chart(fig, use_container_width=True)

        extra_a, extra_b = st.columns(2)
        with extra_a:
            scatter_fig = px.scatter(
                num_df.head(400),
                x=numeric_cols[0],
                y=numeric_cols[1],
                color=numeric_cols[0],
                color_continuous_scale='Tealgrn',
                title=f"Relationship view: {numeric_cols[0]} vs {numeric_cols[1]}",
            )
            style_fig(scatter_fig, 320)
            st.plotly_chart(scatter_fig, use_container_width=True)
        with extra_b:
            if df.select_dtypes(include=['object']).columns.any():
                cat_col = df.select_dtypes(include=['object']).columns[0]
                cat_counts = df[cat_col].astype(str).value_counts().head(12).reset_index()
                cat_counts.columns = [cat_col, 'count']
                tree_fig = px.treemap(cat_counts, path=[cat_col], values='count', color='count', color_continuous_scale='Blues', title=f"Category share: {cat_col}")
                style_fig(tree_fig, 320)
                st.plotly_chart(tree_fig, use_container_width=True)
            else:
                density_fig = px.density_heatmap(num_df.head(500), x=numeric_cols[0], y=numeric_cols[1], title="Density map")
                style_fig(density_fig, 320)
                st.plotly_chart(density_fig, use_container_width=True)

    # Data Preview
    st.markdown("#### Data Preview")
    tab1, tab2 = st.tabs(["Original Data", "Healed Data"])
    with tab1:
        st.dataframe(df, use_container_width=True, height=250)
    with tab2:
        if df_fixed is not None:
            st.dataframe(df_fixed, use_container_width=True, height=250)
        else:
            st.info("Run the Self-Healing Pipeline first")

    # ─── Section B: advanced filters → df_dashboard ─────────────────────────────
    st.markdown("---")
    with st.expander("🔧 Advanced Filters"):
        dfd = df.copy()
        date_col_d = analyst.detect_date_col(df)
        if date_col_d:
            ds = pd.to_datetime(df[date_col_d], errors="coerce")
            if ds.notna().any():
                dd = st.date_input("Date range", value=(ds.min().date(), ds.max().date()), key="dash_date")
                if isinstance(dd, (list, tuple)) and len(dd) == 2:
                    dfd = analyst.filter_by_date(dfd, date_col_d, dd[0], dd[1])
        for i, nc in enumerate(df.select_dtypes(include=[np.number]).columns.tolist()[:2]):
            s = pd.to_numeric(df[nc], errors="coerce")
            lo, hi = float(s.min()), float(s.max())
            if lo < hi:
                rng = st.slider(str(nc), lo, hi, (lo, hi), key=f"dash_num_{i}")
                sv = pd.to_numeric(dfd[nc], errors="coerce")
                dfd = dfd[(sv >= rng[0]) & (sv <= rng[1])]
        cat_d = analyst.detect_category_col(df)
        if cat_d and df[cat_d].nunique() <= 60:
            opts = sorted(df[cat_d].dropna().astype(str).unique().tolist())
            selc = st.multiselect(str(cat_d), opts, default=opts, key="dash_cat")
            if selc:
                dfd = dfd[dfd[cat_d].astype(str).isin(selc)]
        if st.button("Apply Filters", key="dash_apply"):
            st.session_state.df_dashboard = dfd
            st.success(f"Filters applied — {len(dfd):,} rows selected.")
    dash_df = st.session_state.df_dashboard if st.session_state.df_dashboard is not None else df
    rev_c = analyst.detect_revenue_col(dash_df)
    qty_c = analyst.detect_quantity_col(dash_df)

    # ─── Section C: KPI comparison (first half vs second half) ──────────────────
    st.markdown("#### KPI Comparison — First vs Second Half")
    half = len(dash_df) // 2 or 1
    first, second = dash_df.iloc[:half], dash_df.iloc[half:]
    def _cmp(label, a, b, fmt="{:,.0f}"):
        chg = ((b - a) / abs(a) * 100) if a else 0.0
        arrow = "▲" if chg >= 0 else "▼"; color = "#22c55e" if chg >= 0 else "#ef4444"
        return (f'<div class="metric-card animate-in"><div class="metric-label">{label}</div>'
                f'<div class="metric-value" style="font-size:1.4rem">{fmt.format(b)}</div>'
                f'<div style="color:{color};font-weight:700">{arrow} {chg:+.1f}% vs prev</div></div>')
    kcc = st.columns(4)
    if rev_c:
        kcc[0].markdown(_cmp("Revenue", pd.to_numeric(first[rev_c], errors="coerce").sum(),
                             pd.to_numeric(second[rev_c], errors="coerce").sum()), unsafe_allow_html=True)
    if qty_c:
        kcc[1].markdown(_cmp("Quantity", pd.to_numeric(first[qty_c], errors="coerce").sum(),
                             pd.to_numeric(second[qty_c], errors="coerce").sum()), unsafe_allow_html=True)
    if rev_c:
        kcc[2].markdown(_cmp("Avg Transaction", pd.to_numeric(first[rev_c], errors="coerce").mean() or 0,
                             pd.to_numeric(second[rev_c], errors="coerce").mean() or 0), unsafe_allow_html=True)
    kcc[3].markdown(_cmp("Missing Rate %", first.isnull().mean().mean() * 100,
                         second.isnull().mean().mean() * 100, "{:.1f}"), unsafe_allow_html=True)

    # ─── Section D: smart chart gallery (priority-based) ────────────────────────
    st.markdown("#### Smart Chart Gallery")
    date_c = analyst.detect_date_col(dash_df)
    reg_c = analyst.detect_region_col(dash_df)
    cust_c = analyst.detect_customer_col(dash_df)
    built = False
    if rev_c and date_c:
        ser = analyst.resample_revenue(dash_df, date_c, rev_c, "ME")
        if not ser.empty:
            bar = go.Figure(go.Bar(x=[str(i)[:7] for i in ser.index], y=ser.values, marker_color="#00d4ff"))
            bar.update_layout(title="Monthly revenue")
            style_fig(bar, 300); st.plotly_chart(bar, use_container_width=True); built = True
    if reg_c and rev_c:
        g = dash_df.assign(_v=pd.to_numeric(dash_df[rev_c], errors="coerce"),
                           _q=pd.to_numeric(dash_df[qty_c], errors="coerce") if qty_c else 1)
        agg = g.groupby(reg_c).agg(_v=("_v", "sum"), _q=("_q", "sum")).reset_index()
        agg["_q"] = agg["_q"].fillna(1).clip(lower=1)
        scd = px.scatter(agg, x=reg_c, y="_v", size="_q", color=reg_c, title="Revenue by region (size = quantity)")
        style_fig(scd, 300); st.plotly_chart(scd, use_container_width=True); built = True
    if cust_c and rev_c:
        cv = dash_df.assign(_v=pd.to_numeric(dash_df[rev_c], errors="coerce")).groupby(cust_c)["_v"].sum().sort_values(ascending=False)
        if cv.sum():
            cum = cv.cumsum() / cv.sum() * 100
            par = make_subplots(specs=[[{"secondary_y": True}]])
            par.add_trace(go.Bar(x=cv.index.astype(str)[:30], y=cv.values[:30], name="Revenue", marker_color="#7c3aed"), secondary_y=False)
            par.add_trace(go.Scatter(x=cv.index.astype(str)[:30], y=cum.values[:30], name="Cumulative %", line=dict(color="#f59e0b")), secondary_y=True)
            par.update_layout(title="Customer Pareto — top contributors")
            style_fig(par, 320); st.plotly_chart(par, use_container_width=True); built = True
    if not built:
        cats = dash_df.select_dtypes(include="object").columns.tolist()[:2]
        nums = dash_df.select_dtypes(include=[np.number]).columns.tolist()[:4]
        if cats:
            sb = px.sunburst(dash_df, path=cats, title="Categorical breakdown")
            style_fig(sb, 320); st.plotly_chart(sb, use_container_width=True)
        if len(nums) >= 2:
            pc = px.parallel_coordinates(dash_df[nums].apply(pd.to_numeric, errors="coerce").dropna(), title="Numeric parallel coordinates")
            style_fig(pc, 320); st.plotly_chart(pc, use_container_width=True)

    # ─── Section E: anomaly highlight panel (z-score) ───────────────────────────
    st.markdown("#### Anomaly Highlight Panel")
    mask, _z = analyst.zscore_anomalies(dash_df, 3.0)
    if not mask.empty and bool(mask.values.any()):
        row_has = mask.any(axis=1)
        st.markdown(f'<div class="alert-warning">⚠️ <b>{int(row_has.sum())}</b> rows contain anomalous values (|Z| &gt; 3).</div>', unsafe_allow_html=True)
        anom_rows = dash_df[row_has].head(10)
        try:
            st.dataframe(analyst.highlight_anomaly_cells(anom_rows, mask[row_has].head(10)), use_container_width=True)
        except Exception:
            st.dataframe(anom_rows, use_container_width=True)
        st.download_button("⬇ Download anomaly rows CSV", dash_df[row_has].to_csv(index=False).encode(),
                           "anomaly_rows.csv", "text/csv")
    else:
        st.success("✅ No |Z| > 3 anomalies detected in numeric columns.")

    # ─── Section F: data completeness scorecard ─────────────────────────────────
    st.markdown("#### Data Completeness Scorecard")
    comp = (1 - dash_df.isnull().mean()) * 100
    groups = {"🟢 Complete (≥95%)": (comp[comp >= 95], "#22c55e"),
              "🟡 Partial (50–95%)": (comp[(comp >= 50) & (comp < 95)], "#f59e0b"),
              "🔴 Sparse (<50%)": (comp[comp < 50], "#ef4444")}
    for gname, (gs, gcolor) in groups.items():
        if len(gs):
            rows = "".join(
                f'<div style="margin:6px 0">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:3px">'
                f'<span>{html.escape(str(cname))}</span><span style="opacity:0.7">{val:.0f}%</span></div>'
                f'<div style="height:8px;background:rgba(148,163,184,0.18);border-radius:4px;overflow:hidden">'
                f'<div style="width:{min(100, val):.0f}%;height:100%;background:{gcolor}"></div></div></div>'
                for cname, val in gs.items())
            st.markdown(f'<div style="margin-bottom:10px"><b>{gname}</b>{rows}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SELF-HEALING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Self-Healing Pipeline":
    st.markdown("## Self-Healing Data Pipeline")
    st.markdown('<p class="hero-sub">Automatically detects and fixes broken data with full audit trail and intelligent alerts</p>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Upload a dataset in the Data Hub to begin.")
        load_three_js("three_network.html", height=300)
        st.stop()

    df = st.session_state.df
    pipeline = st.session_state.pipeline

    # Controls
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        strategy = st.radio("Fix Strategy", ["smart", "mean", "median"], horizontal=True,
                            help="smart=uses median for skewed data, mean otherwise")
    with col2:
        fix_outliers = st.checkbox("Fix Outliers", value=True)
    with col3:
        fix_dupes = st.checkbox("Remove Duplicates", value=True)

    col_run, col_reset = st.columns([1,1])
    with col_run:
        run_btn = st.button("▶ Run Self-Healing Pipeline", use_container_width=True)
    with col_reset:
        if st.button(" Reset Pipeline", use_container_width=True):
            st.session_state.df_fixed = None
            st.session_state.issues = None
            st.session_state.fixes = None
            st.session_state.alerts = None
            st.session_state.run_complete = False
            st.session_state.healed_csv = None
            st.session_state.ai_fix_advice = None
            pipeline.reset()
            st.rerun()

    # ───  AI Fix Assistant: suggestions + one-click corrected file ─────────────
    st.markdown("---")
    st.markdown("### AI Fix Assistant")
    st.caption("Detect problems, get AI recommendations, and export a corrected file in one click.")
    ai1, ai2 = st.columns(2)
    with ai1:
        oneclick = st.button(" One-Click AI Fix & Download", use_container_width=True, type="primary")
    with ai2:
        suggest = st.button(" Ask AI: should I fix these?", use_container_width=True)

    def _heal_now(_df, _strategy, _fo, _fd):
        p = SelfHealingPipeline()
        iss = p.detect_issues(_df)
        fixed, fxs = p.auto_fix(_df.copy(), iss, strategy=_strategy)
        if not _fo:
            fxs = [f for f in fxs if f["type"] != "OUTLIER"]
        if not _fd:
            fxs = [f for f in fxs if f["type"] != "DUPLICATE"]
        return p, iss, fixed, fxs

    if oneclick:
        try:
            with st.spinner(" Detecting →  healing your file..."):
                p, iss, fixed, fxs = _heal_now(df, strategy, fix_outliers, fix_dupes)
                alerts = p.generate_alerts(iss, st.session_state.df_name)
                st.session_state.pipeline = p
                st.session_state.issues = iss
                st.session_state.df_fixed = fixed
                st.session_state.fixes = fxs
                st.session_state.alerts = alerts
                st.session_state.run_complete = True
                st.session_state.healed_csv = fixed.to_csv(index=False).encode()
            st.success(f" Healed **{p.stats['total_issues']}** detected issue group(s); applied **{len(fxs)}** fixes.")
            # AI explanation of what changed (best-effort)
            if st.session_state.api_key and p.stats["total_issues"] > 0:
                try:
                    agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
                    with st.spinner(" AI summarizing the corrections..."):
                        st.session_state.ai_fix_advice = agent.analyze_healing_results(
                            p.get_pipeline_summary(), st.session_state.df_name)
                except Exception:
                    st.session_state.ai_fix_advice = None
        except Exception as e:
            st.error(f" Healing hit an error on this file: {e}")
            st.session_state.ai_fix_advice = None

    if suggest:
        try:
            p, iss, _, fxs = _heal_now(df, strategy, fix_outliers, fix_dupes)
            n = p.stats["total_issues"]
            if n == 0:
                st.session_state.ai_fix_advice = " No issues detected — your data already looks clean."
            elif st.session_state.api_key:
                agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
                with st.spinner(" DataSage reviewing your data..."):
                    st.session_state.ai_fix_advice = agent.analyze_healing_results(
                        p.get_pipeline_summary(), st.session_state.df_name)
            else:
                st.session_state.ai_fix_advice = " AI key unavailable — but issues were found; use One-Click Fix to clean them."
        except Exception as e:
            st.session_state.ai_fix_advice = f"Could not analyze: {e}"

    if st.session_state.get("ai_fix_advice"):
        st.markdown(f'<div class="alert-info"><b> AI Recommendation</b></div>', unsafe_allow_html=True)
        st.markdown(st.session_state.ai_fix_advice)

    if st.session_state.get("healed_csv"):
        st.download_button("⬇ Download Corrected CSV",
                           st.session_state.healed_csv,
                           f"{(st.session_state.df_name or 'data').replace(' ', '_')}_healed.csv",
                           "text/csv", use_container_width=True)

    st.markdown("---")

    if run_btn:
        st.session_state.pipeline = SelfHealingPipeline()
        pipeline = st.session_state.pipeline
        
        with st.spinner(" Scanning for data issues..."):
            progress = st.progress(0)
            status = st.empty()

            status.markdown(" Detecting issues...")
            issues = pipeline.detect_issues(df)
            st.session_state.issues = issues
            progress.progress(33)
            time.sleep(0.4)

            status.markdown(" Applying intelligent fixes...")
            df_fixed, fixes = pipeline.auto_fix(df.copy(), issues, strategy=strategy)
            if not fix_outliers:
                fixes = [f for f in fixes if f["type"] != "OUTLIER"]
            if not fix_dupes:
                fixes = [f for f in fixes if f["type"] != "DUPLICATE"]
            st.session_state.df_fixed = df_fixed
            st.session_state.fixes = fixes
            progress.progress(66)
            time.sleep(0.4)

            status.markdown(" Generating intelligent alerts...")
            alerts = pipeline.generate_alerts(issues, st.session_state.df_name)
            st.session_state.alerts = alerts
            progress.progress(100)
            time.sleep(0.2)

            st.session_state.run_complete = True
            status.empty()
            progress.empty()
            st.success(f" Pipeline complete! Found {pipeline.stats['total_issues']} issues, fixed {pipeline.stats['fixed_issues']} automatically.")

    if st.session_state.run_complete and st.session_state.issues:
        issues = st.session_state.issues
        fixes = st.session_state.fixes
        alerts = st.session_state.alerts

        # Summary metrics
        st.markdown("---")
        st.markdown("### Pipeline Results")
        m1,m2,m3,m4,m5 = st.columns(5)
        with m1: st.markdown(render_metric("Issues Found", pipeline.stats["total_issues"], icon=""), unsafe_allow_html=True)
        with m2: st.markdown(render_metric("Auto Fixed", pipeline.stats["fixed_issues"], icon=""), unsafe_allow_html=True)
        with m3: st.markdown(render_metric("Critical Alerts", pipeline.stats["alerts_raised"], icon=""), unsafe_allow_html=True)
        with m4: st.markdown(render_metric("Missing Values", sum(i["count"] for i in issues.get("missing_values",[])), icon=""), unsafe_allow_html=True)
        with m5: st.markdown(render_metric("Outliers Fixed", sum(i["count"] for i in issues.get("outliers",[])), icon=""), unsafe_allow_html=True)

        # ─── Health improvement: before vs after ────────────────────────────────
        if st.session_state.df_fixed is not None:
            try:
                h_before = monitoring.compute_metrics(st.session_state.df)["health"]
                h_after = monitoring.compute_metrics(st.session_state.df_fixed)["health"]
                st.markdown("#### Health Improvement")
                hg = st.columns([1, 1, 1])
                with hg[0]:
                    st.plotly_chart(mini_gauge(h_before, "Before Healing", ref=80), use_container_width=True)
                with hg[1]:
                    st.plotly_chart(mini_gauge(h_after, "After Healing", ref=80), use_container_width=True)
                with hg[2]:
                    delta = h_after - h_before
                    st.markdown(render_metric("Quality Gain", f"+{delta} pts" if delta >= 0 else f"{delta} pts",
                                              icon="📈"), unsafe_allow_html=True)
                    st.markdown(render_metric("Rows Retained",
                                              f"{len(st.session_state.df_fixed):,}/{len(st.session_state.df):,}",
                                              icon="🧾"), unsafe_allow_html=True)
            except Exception:
                pass

        tab1, tab2, tab3, tab4 = st.tabs([" Alerts", " Detected Issues", " Applied Fixes", " Visual Diff"])

        with tab1:
            for alert in alerts:
                level = alert["level"]
                if "CRITICAL" in level: css = "alert-critical"
                elif "WARNING" in level: css = "alert-warning"
                elif "OK" in level: css = "alert-ok"
                else: css = "alert-info"
                details_html = "<ul style='margin:6px 0 0 16px;font-size:0.8rem'>" + "".join(f"<li>{d}</li>" for d in alert.get("details",[])) + "</ul>" if alert.get("details") else ""
                st.markdown(f"""<div class="{css}">
                    <b>{level} — {alert["category"]}</b><br>
                    {alert["message"]}
                    {details_html}
                    <div style="font-size:0.72rem;opacity:0.6;margin-top:6px"> {alert["timestamp"]}</div>
                </div>""", unsafe_allow_html=True)

        with tab2:
            categories = {
                " Missing Values": issues.get("missing_values", []),
                " Outliers": issues.get("outliers", []),
                " Type Errors": issues.get("type_errors", []),
                " Negative Anomalies": issues.get("negative_anomalies", []),
                " Duplicates": issues.get("duplicate_rows", []),
                " Date Errors": issues.get("date_errors", []),
                " Text in Numeric": issues.get("string_in_numeric", []),
                "␣ Whitespace": issues.get("whitespace", []),
                " Constant Columns": issues.get("constant_columns", []),
            }
            for cat_name, cat_issues in categories.items():
                if cat_issues:
                    with st.expander(f"{cat_name} ({len(cat_issues)} column{'s' if len(cat_issues)>1 else ''})"):
                        for issue in cat_issues:
                            col_name = issue.get("column", "Multiple")
                            count = issue.get("count", "?")
                            pct = issue.get("pct", "")
                            pct_str = f" ({pct}%)" if pct else ""
                            st.markdown(f"""<div class="data-card" style="padding:12px 16px">
                                <b>Column: <code>{col_name}</code></b> — {count} affected rows{pct_str}
                            </div>""", unsafe_allow_html=True)

        with tab3:
            if fixes:
                for fix in fixes:
                    severity = fix.get("severity", "low")
                    icon = "" if severity=="high" else "" if severity=="medium" else ""
                    st.markdown(f"""<div class="fix-item {severity}">
                        {icon} <b>[{fix["type"]}]</b> Column: <code>{fix["column"]}</code><br>
                        └─ {fix["action"]}
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No fixes were applied.")

        with tab4:
            if st.session_state.df_fixed is not None:
                df_orig = st.session_state.df
                df_fix = st.session_state.df_fixed
                numeric_cols = [c for c in df_orig.columns if c in df_fix.columns and pd.to_numeric(df_orig[c], errors='coerce').notna().sum() > 3]
                if numeric_cols:
                    sel = st.selectbox("Compare column", numeric_cols, key="diff_col")
                    orig_vals = pd.to_numeric(df_orig[sel], errors='coerce')
                    fix_vals  = pd.to_numeric(df_fix[sel], errors='coerce')
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=orig_vals, name="Original", line=dict(color="#ef4444", width=2), mode="lines+markers", marker=dict(size=5)))
                    fig.add_trace(go.Scatter(y=fix_vals, name="Healed", line=dict(color="#22c55e", width=2), mode="lines+markers", marker=dict(size=5)))
                    fig.update_layout(title=f"Before vs After: {sel}", height=340,
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      font=dict(color="#94a3b8"), legend=dict(bgcolor='rgba(0,0,0,0)'))
                    st.plotly_chart(fig, use_container_width=True)

                    # Stats comparison
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Original Stats**")
                        st.dataframe(orig_vals.describe().to_frame("Original").round(3), use_container_width=True)
                    with c2:
                        st.markdown("**Healed Stats**")
                        st.dataframe(fix_vals.describe().to_frame("Healed").round(3), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA QUALITY AI
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Data Quality AI":
    st.markdown("## Data Quality AI")
    st.markdown('<p class="hero-sub">Comprehensive quality scoring with AI-powered explanations and fix suggestions</p>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Upload a dataset in the Data Hub to begin.")
        st.stop()

    df = st.session_state.df

    if st.button(" Run Full Quality Scan", use_container_width=False):
        with st.spinner("Running comprehensive quality analysis..."):
            checker = DataQualityChecker()
            report = checker.full_quality_scan(df)
            st.session_state.quality_report = report
            st.session_state.checker = checker
        st.success(" Quality scan complete!")

    if st.session_state.quality_report:
        report = st.session_state.quality_report
        score = report["quality_score"]
        checker = st.session_state.checker

        # Score display
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            color = score_color(score)
            grade = "A" if score>=90 else "B" if score>=80 else "C" if score>=70 else "D" if score>=60 else "F"
            st.markdown(f"""<div style="text-align:center;padding:30px">
                <div style="font-size:4rem;font-weight:900;color:{color}">{score}</div>
                <div style="font-size:1.2rem;color:{color};font-weight:700">Grade {grade} — {score_emoji(score)} {"Excellent" if score>=80 else "Needs Work" if score>=60 else "Poor"}</div>
                <div style="width:100%;height:12px;background:rgba(255,255,255,0.1);border-radius:6px;margin:12px 0">
                    <div style="width:{score}%;height:100%;background:linear-gradient(90deg,{color},#7c3aed);border-radius:6px;transition:width 1s"></div>
                </div>
                <p style="opacity:0.6;font-size:0.85rem">Overall data quality score across 6 dimensions</p>
            </div>""", unsafe_allow_html=True)

        # Dimension scores
        st.markdown("### Quality Dimensions")
        dims = [
            ("Completeness", 100 - min(100, len(report["completeness"]["issues"]) * 10), ""),
            ("Accuracy",     100 - min(100, len(report["accuracy"]["issues"]) * 15), ""),
            ("Consistency",  100 - min(100, len(report["consistency"]["issues"]) * 12), ""),
            ("Uniqueness",   100 - min(100, report["uniqueness"]["duplicate_rows"] * 5), ""),
            ("Timeliness",   100 - min(100, len(report["timeliness"]["issues"]) * 20), "⏱"),
            ("Validity",     100 - min(100, len(report["validity"]["issues"]) * 8), ""),
        ]
        cols = st.columns(6)
        for (dim, dim_score, icon), col in zip(dims, cols):
            c = score_color(dim_score)
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div style="font-size:1.3rem">{icon}</div>
                    <div class="metric-value" style="color:{c};font-size:1.5rem">{dim_score}</div>
                    <div class="metric-label">{dim}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge + Radar (professional viz)
        gcol1, gcol2 = st.columns([1, 1])
        with gcol1:
            gfig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                delta={"reference": 80, "increasing": {"color": "#22c55e"}, "decreasing": {"color": "#ef4444"}},
                number={"suffix": " /100", "font": {"size": 34}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": score_color(score)},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 60], "color": "rgba(239,68,68,0.18)"},
                        {"range": [60, 80], "color": "rgba(245,158,11,0.18)"},
                        {"range": [80, 100], "color": "rgba(34,197,94,0.18)"},
                    ],
                    "threshold": {"line": {"color": "#e2e8f0", "width": 3}, "value": score},
                },
                title={"text": "Overall Quality"},
            ))
            style_fig(gfig, 320)
            st.plotly_chart(gfig, use_container_width=True)
        with gcol2:
            dim_names = [d[0] for d in dims]
            dim_vals = [d[1] for d in dims]
            rfig = go.Figure(go.Scatterpolar(
                r=dim_vals + [dim_vals[0]], theta=dim_names + [dim_names[0]],
                fill="toself", fillcolor="rgba(0,212,255,0.18)", line=dict(color="#00d4ff", width=2),
                name="Score",
            ))
            rfig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,0.2)"),
                                          bgcolor="rgba(0,0,0,0)", angularaxis=dict(gridcolor="rgba(148,163,184,0.2)")),
                               title="Quality Dimensions Radar")
            style_fig(rfig, 320)
            st.plotly_chart(rfig, use_container_width=True)

        # Detailed tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([" Completeness", " Accuracy", " Consistency", " Distributions", " Fix Suggestions"])

        with tab1:
            issues_c = report["completeness"]["issues"]
            if issues_c:
                df_miss = pd.DataFrame(issues_c)[["column","missing_count","missing_pct","severity","suggestion"]]
                st.dataframe(df_miss, use_container_width=True)
                # Chart
                fig = px.bar(df_miss, x="column", y="missing_pct", color="severity",
                             color_discrete_map={"HIGH":"#ef4444","MEDIUM":"#f59e0b","LOW":"#22c55e"},
                             title="Missing % by Column")
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color="#94a3b8"), height=300, margin=dict(t=30,b=0,l=0,r=0))
                st.plotly_chart(fig, use_container_width=True)
                for issue in issues_c:
                    st.markdown(f"""<div class="alert-{'critical' if issue['severity']=='HIGH' else 'warning' if issue['severity']=='MEDIUM' else 'info'}">
                        <b>{issue['column']}</b>: {issue['missing_count']} missing ({issue['missing_pct']}%)<br>
                         <i>{issue['suggestion']}</i>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success(" No completeness issues found!")

        with tab2:
            issues_a = report["accuracy"]["issues"]
            if issues_a:
                for issue in issues_a:
                    st.markdown(f"""<div class="data-card">
                        <b> {issue['type']}</b> — Column: <code>{issue.get('column','N/A')}</code><br>
                        {issue.get('description', issue.get('count',''))} values affected<br>
                         <i>{issue['suggestion']}</i>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success(" No accuracy issues found!")

        with tab3:
            issues_con = report["consistency"]["issues"]
            if issues_con:
                for issue in issues_con:
                    st.markdown(f"""<div class="alert-warning">
                        <b>{issue['type']}</b> — Column: <code>{issue.get('column','N/A')}</code><br>
                        {issue.get('description','')}<br>
                         <i>{issue['suggestion']}</i>
                    </div>""", unsafe_allow_html=True)
            else:
                st.success(" No consistency issues found!")

        with tab4:
            dist = report.get("distribution", {})
            if dist:
                cols_list = list(dist.keys())
                sel_d = st.selectbox("Column", cols_list, key="dist_sel")
                d = dist[sel_d]
                col1, col2 = st.columns(2)
                with col1:
                    stats_df = pd.DataFrame([
                        ["Mean", d["mean"]], ["Median", d["median"]], ["Std Dev", d["std"]],
                        ["Skewness", d["skewness"]], ["Kurtosis", d["kurtosis"]],
                        ["Normal?", "Yes" if d["is_normal"] else "No"],
                        ["Normality p-value", d["normality_p_value"]]
                    ], columns=["Statistic","Value"])
                    st.dataframe(stats_df, use_container_width=True)
                with col2:
                    series = pd.to_numeric(df[sel_d], errors='coerce').dropna()
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=series, nbinsx=25, name="Distribution",
                                               marker_color="#00d4ff", opacity=0.75))
                    fig.update_layout(title=f"{sel_d} — {'Normal ' if d['is_normal'] else 'Non-Normal '}",
                                      height=280, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      font=dict(color="#94a3b8"), margin=dict(t=30,b=0,l=0,r=0))
                    st.plotly_chart(fig, use_container_width=True)

        with tab5:
            st.markdown("### AI Fix Suggestions by Issue Type")
            issue_types = ["MISSING_VALUE","OUTLIER","TYPE_ERROR","NEGATIVE_ANOMALY","DUPLICATE_ROWS"]
            sel_type = st.selectbox("Select Issue Type", issue_types)
            col_name = st.text_input("Column Name (optional)", "your_column")
            suggestions = checker.suggest_fix(sel_type, col_name, df)
            explanation = checker.explain_issue(sel_type, col_name, "")
            st.markdown(f"""<div class="alert-info">
                <b> Explanation</b><br>{explanation}
            </div>""", unsafe_allow_html=True)
            st.markdown("#### 🔧 Code Solutions")
            for i, sug in enumerate(suggestions, 1):
                st.code(sug, language="python")

        # ─── AI action plan (structured) + downloadable report ──────────────────
        st.markdown("---")
        st.markdown("### 🤖 AI Quality Action Plan")
        dim_summary = ", ".join(f"{d[0]}={d[1]}" for d in dims)
        dq_prompt = (
            f"Data-quality report for '{st.session_state.df_name}': overall_score={score}/100 (grade {grade}). "
            f"Dimension scores: {dim_summary}. "
            f"Completeness issues={len(report['completeness']['issues'])}, "
            f"accuracy issues={len(report['accuracy']['issues'])}, "
            f"consistency issues={len(report['consistency']['issues'])}, "
            f"duplicate rows={report['uniqueness']['duplicate_rows']}. "
            "As a data-quality lead, summarize the state of this dataset, the biggest data-quality risks, "
            "and a prioritized remediation plan (which dimension to fix first and how)."
        )
        render_ai_brief(dq_prompt, key="quality_ai",
                        system='You are a data-quality lead. Respond ONLY with JSON {"summary":str,"risks":[str],"actions":[str]}.',
                        button_label="🧠 Generate AI quality action plan")

        # Downloadable markdown report
        report_md = (
            f"# Data Quality Report — {st.session_state.df_name}\n\n"
            f"**Overall Score:** {score}/100 (Grade {grade})\n\n"
            f"## Dimension Scores\n" + "".join(f"- {d[0]}: {d[1]}/100\n" for d in dims) +
            f"\n## Issue Summary\n"
            f"- Completeness issues: {len(report['completeness']['issues'])}\n"
            f"- Accuracy issues: {len(report['accuracy']['issues'])}\n"
            f"- Consistency issues: {len(report['consistency']['issues'])}\n"
            f"- Duplicate rows: {report['uniqueness']['duplicate_rows']}\n"
            f"\n_Generated by Navneet Data Studio · Snehal Laxman Jadhav © 2026_\n"
        )
        st.download_button("⬇ Download Quality Report (Markdown)", report_md.encode(),
                           f"{(st.session_state.df_name or 'dataset').replace(' ', '_')}_quality_report.md",
                           "text/markdown")

        # AI Analysis button (free-form, existing)
        if st.session_state.api_key:
            if st.button(" Get AI Analysis of Quality Report"):
                agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
                with st.spinner(" AI analyzing quality report..."):
                    ai_response = agent.analyze_quality_report(report, st.session_state.df_name)
                st.markdown(f"""<div class="data-card"><b> AI Analysis</b><br><br>{ai_response}</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI AGENT CHAT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "AI Agent Chat":
    st.markdown("## AI Data Agent — DataSage")
    st.markdown('<p class="hero-sub">Multi-turn conversational AI for data analysis, code generation, and business insights</p>', unsafe_allow_html=True)

    # Instant visual analysis — generates many chart types (no API key required)
    if st.session_state.df is not None:
        with st.expander("Auto-Analyze Dataset — generate many charts", expanded=False):
            st.caption(f"Visual analysis of {st.session_state.df_name}: trends, distributions, correlations, categories.")
            ac = st.columns([1, 1])
            with ac[0]:
                if st.button("Generate Visual Analysis", use_container_width=True, key="agent_gallery_btn"):
                    st.session_state._show_gallery = True
            with ac[1]:
                if st.button("AI Narrative + Actions", use_container_width=True, key="agent_narr_btn") and st.session_state.api_key:
                    with st.spinner("DataSage analyzing..."):
                        _ag = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
                        st.session_state._agent_narr = _ag.get_quick_insights(st.session_state.df, st.session_state.df_name)
            if st.session_state.get("_show_gallery"):
                render_chart_gallery(st.session_state.df)
            if st.session_state.get("_agent_narr"):
                st.markdown(st.session_state._agent_narr)

    if not st.session_state.api_key:
        st.warning(" Please enter your OpenRouter API key in the sidebar to use the AI Agent.")
        st.markdown("""<div class="data-card">
            <h4> What can DataSage do?</h4>
            <ul style="margin-top:12px;line-height:2">
                <li> Analyze your dataset and explain quality issues in plain English</li>
                <li> Generate Python & SQL code to fix data problems</li>
                <li> Design self-healing pipeline architecture</li>
                <li> Provide business impact analysis of data issues</li>
                <li> Answer any data engineering questions</li>
                <li> Perform root cause analysis on anomalies</li>
            </ul>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # Initialize agent
    if st.session_state.agent is None or st.session_state.agent.api_key != st.session_state.api_key:
        st.session_state.agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)

    agent = st.session_state.agent
    agent.model = st.session_state.selected_model

    # Quick action buttons
    st.markdown("#### ⚡ Quick Actions")
    qcols = st.columns(4)
    quick_prompts = [
        (" Analyze Dataset", f"Analyze the {st.session_state.df_name or 'loaded'} dataset and give me 5 key insights about its quality and patterns"),
        (" Generate Fix Code", f"Generate Python code to fix all data quality issues in the {st.session_state.df_name or 'loaded'} dataset"),
        (" Business Impact", "What is the business impact of missing data and outliers in this dataset? Estimate revenue risk."),
        (" Pipeline Design", "Design a complete production-ready self-healing data pipeline architecture for this dataset type"),
    ]
    for (label, prompt), col in zip(quick_prompts, qcols):
        with col:
            if st.button(label, use_container_width=True, key=f"qp_{label}"):
                st.session_state.chat_history.append({"role":"user","content":prompt})
                with st.spinner(" DataSage thinking..."):
                    response = agent.chat(prompt, st.session_state.df)
                st.session_state.chat_history.append({"role":"assistant","content":response,"reasoning":agent.last_reasoning})
                st.rerun()

    # Specialized analysis
    st.markdown("#### 🎯 Specialized Analysis")
    spec_cols = st.columns(3)
    with spec_cols[0]:
        if st.button(" Analyze Healing Results", use_container_width=True) and st.session_state.fixes:
            summary = st.session_state.pipeline.get_pipeline_summary()
            with st.spinner(" Analyzing..."):
                resp = agent.analyze_healing_results(summary, st.session_state.df_name)
            st.session_state.chat_history.append({"role":"user","content":"Analyze the pipeline healing results"})
            st.session_state.chat_history.append({"role":"assistant","content":resp,"reasoning":agent.last_reasoning})
            st.rerun()
    with spec_cols[1]:
        if st.button(" Generate SQL Fixes", use_container_width=True) and st.session_state.issues:
            with st.spinner(" Generating SQL..."):
                resp = agent.generate_sql_fixes(st.session_state.issues, st.session_state.df_name.replace(" ","_").lower())
            st.session_state.chat_history.append({"role":"user","content":"Generate SQL fix queries"})
            st.session_state.chat_history.append({"role":"assistant","content":resp,"reasoning":agent.last_reasoning})
            st.rerun()
    with spec_cols[2]:
        if st.button(" Quick Insights", use_container_width=True) and st.session_state.df is not None:
            with st.spinner(" Getting insights..."):
                resp = agent.get_quick_insights(st.session_state.df, st.session_state.df_name)
            st.session_state.chat_history.append({"role":"user","content":"Give quick data insights"})
            st.session_state.chat_history.append({"role":"assistant","content":resp,"reasoning":agent.last_reasoning})
            st.rerun()

    st.markdown("---")

    # Chat display
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("""<div style="text-align:center;padding:40px;opacity:0.5">
                <div style="font-size:3rem"></div>
                <p>DataSage is ready. Ask anything about your data!</p>
            </div>""", unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-message chat-user"> {msg["content"][:200]}{"..." if len(msg["content"])>200 else ""}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="chat-ai-label"> DataSage</div>', unsafe_allow_html=True)
                # Render markdown content properly (code blocks, tables, lists all render)
                st.markdown(msg["content"])
                if msg.get("reasoning"):
                    with st.expander(" View reasoning trace"):
                        st.markdown(msg["reasoning"])
                st.markdown("<hr style='opacity:0.1'>", unsafe_allow_html=True)

    # Input
    col_input, col_send, col_clear = st.columns([5,1,1])
    with col_input:
        user_input = st.text_input("Ask DataSage...", placeholder="e.g. Why is the data quality score low? How do I fix outliers in Weekly_Sales?", key="chat_input", label_visibility="collapsed")
    with col_send:
        send = st.button("Send ", use_container_width=True)
    with col_clear:
        if st.button("Clear ", use_container_width=True):
            st.session_state.chat_history = []
            agent.clear_history()
            st.rerun()

    if send and user_input.strip():
        st.session_state.chat_history.append({"role":"user","content":user_input})
        with st.spinner(" DataSage is thinking..."):
            response = agent.chat(user_input, st.session_state.df)
        st.session_state.chat_history.append({"role":"assistant","content":response,"reasoning":agent.last_reasoning})
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AGENTIC PIPELINE  (LangGraph orchestration)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Agentic Pipeline":
    st.markdown("## Agentic Data Pipeline")
    st.markdown('<p class="hero-sub">An autonomous <b>LangGraph</b> workflow chains every step — profile → detect → heal → quality → ML → AI summary — in one run.</p>', unsafe_allow_html=True)

    engine = "LangGraph" if agent_graph._LANGGRAPH else "Sequential (fallback)"
    st.markdown(f"""<div class="alert-info">
         Orchestration engine: <span class="status-badge badge-blue">{engine}</span>
        &nbsp; Model: <span class="status-badge badge-green">{st.session_state.selected_model.split('/')[-1]}</span>
    </div>""", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Upload a dataset in the Data Hub to run the agent.")
        load_three_js("three_network.html", height=320)
        st.stop()

    df = st.session_state.df
    crun = st.columns([1, 1, 2])
    with crun[0]:
        use_ai = st.toggle("Include AI summary", value=True, help="Adds a reasoning-LLM run-summary node")
    with crun[1]:
        run_agent = st.button("▶ Run Agentic Workflow", use_container_width=True)

    if run_agent:
        agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model) if use_ai else None
        node_labels = {"profile": " Profiling", "detect": " Detecting issues", "heal": " Healing",
                       "quality": " Quality scan", "ml_analytics": " ML anomaly scan", "ai_summary": " AI summary"}
        prog = st.progress(0, text="Starting agent...")
        steps_total = 6
        done = {"n": 0}

        def cb(node):
            done["n"] += 1
            prog.progress(min(1.0, done["n"] / steps_total), text=node_labels.get(node, node))

        with st.spinner(" Agent orchestrating the pipeline..."):
            result = agent_graph.run_workflow(df, st.session_state.df_name or "Dataset", agent=agent, step_cb=cb)
        prog.progress(1.0, text="Done")
        # persist results into the rest of the app
        st.session_state.agent_run = {k: result.get(k) for k in ["profile", "health", "trace", "engine", "ai_summary", "ai_reasoning"]}
        st.session_state.issues = result.get("issues")
        st.session_state.df_fixed = result.get("df_fixed")
        st.session_state.fixes = result.get("fixes")
        st.session_state.alerts = result.get("alerts")
        st.session_state.quality_report = result.get("quality_report")
        st.session_state.pipeline = result.get("pipeline", st.session_state.pipeline)
        st.session_state.ml_anomaly = result.get("anomalies")
        st.session_state.run_complete = True
        st.success(f" Workflow complete via {result.get('engine')} engine.")

    run = st.session_state.agent_run
    if run:
        st.markdown("---")
        m = st.columns(4)
        prof = run.get("profile", {}) or {}
        with m[0]: st.markdown(render_metric("Rows", f"{prof.get('rows','—'):,}" if isinstance(prof.get('rows'), int) else "—", icon=""), unsafe_allow_html=True)
        with m[1]: st.markdown(render_metric("Health Score", f"{run.get('health','—')}", icon=""), unsafe_allow_html=True)
        with m[2]:
            qs = (st.session_state.quality_report or {}).get("quality_score", "—")
            st.markdown(render_metric("Quality", qs, icon=""), unsafe_allow_html=True)
        with m[3]:
            na = (st.session_state.ml_anomaly or {}).get("n_anomalies", "—")
            st.markdown(render_metric("ML Anomalies", na, icon=""), unsafe_allow_html=True)

        ctrace, csum = st.columns([1, 1.3])
        with ctrace:
            st.markdown("#### Execution Trace")
            for t in run.get("trace", []):
                st.markdown(f'<div class="fix-item">{t}</div>', unsafe_allow_html=True)
        with csum:
            st.markdown("#### AI Orchestrator Summary")
            if run.get("ai_summary"):
                st.markdown(run["ai_summary"])
                if run.get("ai_reasoning"):
                    with st.expander(" View reasoning trace"):
                        st.markdown(run["ai_reasoning"])
            else:
                st.info("Enable 'Include AI summary' and re-run for an AI verdict.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI ANALYTICS  (XGBoost predictive + IsolationForest + RAG Q&A)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "AI Analytics":
    st.markdown("## AI Analytics")
    st.markdown('<p class="hero-sub">Predictive ML, unsupervised anomaly detection, and <b>RAG "ask your data"</b> grounded in the vector DB.</p>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Upload a dataset in the Data Hub to run analytics.")
        st.stop()

    df = st.session_state.df_fixed if st.session_state.df_fixed is not None else st.session_state.df
    numeric_frame = df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    tab_pred, tab_anom, tab_rag, tab_ts, tab_seg = st.tabs(
        [" Predictive Modeling", " Anomaly Detection", " Ask Your Data (RAG)",
         " Time Series Analysis", " Segmentation Analysis"])

    # ─ Predictive modeling ─
    with tab_pred:
        st.caption(f"Engine: **{ml_analytics.backend_name()}** · auto target selection · feature importance")
        num_cols = numeric_frame.columns.tolist()
        target = "(auto)"
        if not num_cols:
            st.warning("Add at least 2 numeric columns to unlock predictive modeling visuals.")
        else:
            cta_col, info_col = st.columns([1.5, 1])
            with cta_col:
                target = st.selectbox("Target column (auto if blank)", ["(auto)"] + num_cols)
            with info_col:
                st.markdown(
                    """<div class="data-card" style="padding:16px 18px">
                    <b>How to read this</b><br><br>
                    The model learns a numeric target, then ranks the strongest drivers so the result is easy to explain.
                    </div>""",
                    unsafe_allow_html=True,
                )
        if num_cols and st.button("Train Model", use_container_width=False):
            with st.spinner("Training gradient-boosted model..."):
                res = ml_analytics.run_predictive_model(df, None if target == "(auto)" else target)
            st.session_state.ml_predict = res
        res = st.session_state.ml_predict
        if res:
            if not res.get("ok"):
                st.warning(res.get("error"))
            else:
                mc = st.columns(4)
                with mc[0]: st.markdown(render_metric("Task", res["task"].title(), icon=""), unsafe_allow_html=True)
                with mc[1]: st.markdown(render_metric("Target", res["target"][:14], icon=""), unsafe_allow_html=True)
                metric_items = list(res["metrics"].items())
                for i, (k, v) in enumerate(metric_items[:2]):
                    with mc[2 + i]: st.markdown(render_metric(k.upper(), v, icon=""), unsafe_allow_html=True)
                imp = res["importance"]
                if isinstance(imp, pd.DataFrame) and not imp.empty:
                    st.markdown("#### Feature Importance")
                    imp_3d = imp.head(12).copy().sort_values("importance", ascending=False).reset_index(drop=True)
                    max_importance = max(float(imp_3d["importance"].max()), 1e-6)
                    pcol1, pcol2 = st.columns([1.6, 1], gap="large")
                    with pcol1:
                        render_three_analytics(
                            "feature_importance",
                            {
                                "type": "bars",
                                "title": "Feature Importance 3D",
                                "data": [
                                    {
                                        "label": row["feature"][:18],
                                        "value": round((float(row["importance"]) / max_importance) * 10.0, 3),
                                        "color": "#18b6ff" if idx % 2 == 0 else "#8b5cf6",
                                    }
                                    for idx, row in imp_3d.iterrows()
                                ],
                            },
                            height=380,
                        )
                    with pcol2:
                        top_features = "<br>".join(
                            f"{idx + 1}. <b>{html.escape(str(row['feature']))}</b> · {float(row['importance']):.4f}"
                            for idx, row in imp_3d.head(4).iterrows()
                        )
                        st.markdown(
                            f"""<div class="data-card" style="min-height:380px">
                            <h4 style="margin-bottom:10px">Model Summary</h4>
                            <p style="opacity:0.82;margin-bottom:12px">
                                The model is solving a <b>{html.escape(res["task"])}</b> problem for
                                <b>{html.escape(str(res["target"]))}</b>.
                            </p>
                            <div class="fix-item">{top_features}</div>
                            <p style="margin-top:14px;opacity:0.78">
                                Focus first on the top-ranked drivers above. They explain most of the model movement.
                            </p>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                st.caption(f"Trained on {res['n_train']} rows · tested on {res['n_test']} rows.")

                # ─ Actual vs Predicted + residuals ─
                yt = np.array(res.get("y_test", [])); yp = np.array(res.get("y_pred", []))
                if len(yt) and len(yt) == len(yp):
                    st.markdown("#### Actual vs Predicted")
                    avp1, avp2 = st.columns(2)
                    with avp1:
                        err = np.abs(yt - yp)
                        sc = go.Figure()
                        sc.add_trace(go.Scatter(x=yt, y=yp, mode="markers",
                                                marker=dict(color=err, colorscale="Tealgrn_r", size=7, showscale=True),
                                                name="Predictions"))
                        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
                        sc.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                                line=dict(color="#ef4444", dash="dash"), name="Perfect fit"))
                        sc.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
                        style_fig(sc, 320); st.plotly_chart(sc, use_container_width=True)
                    with avp2:
                        resid = yt - yp
                        rh = px.histogram(x=resid, nbins=30, title="Residual distribution")
                        style_fig(rh, 320); st.plotly_chart(rh, use_container_width=True)

                # ─ What-If simulator ─
                model = res.get("model"); means = res.get("feature_means") or {}
                feats = res.get("features", [])
                if model is not None and means and feats:
                    st.markdown("#### What-If Simulator")
                    imp_df = res.get("importance")
                    top_feats = (imp_df["feature"].head(3).tolist()
                                 if isinstance(imp_df, pd.DataFrame) and not imp_df.empty else feats[:3])
                    wcols = st.columns(max(1, len(top_feats)))
                    overrides = {}
                    for fcol, wc in zip(top_feats, wcols):
                        with wc:
                            overrides[fcol] = st.number_input(str(fcol), value=float(means.get(fcol, 0.0)), key=f"wi_{fcol}")
                    if st.button("Predict", key="whatif_predict"):
                        row = [overrides.get(f, means.get(f, 0.0)) for f in feats]
                        try:
                            pred_val = model.predict(pd.DataFrame([row], columns=feats))[0]
                            lbls = res.get("labels")
                            if lbls and res["task"] == "classification":
                                inv = {v: kk for kk, v in lbls.items()}
                                pred_val = inv.get(int(round(float(pred_val))), pred_val)
                            st.markdown(render_metric(f"Predicted {res['target']}", f"{pred_val:,.2f}" if isinstance(pred_val, (int, float)) else str(pred_val), icon="🎯"), unsafe_allow_html=True)
                        except Exception as _e:
                            st.error(f"Prediction failed: {_e}")

    # ─ Anomaly detection ─
    with tab_anom:
        st.caption("IsolationForest over numeric features — flags multivariate outliers.")
        contamination = st.slider("Expected anomaly rate", 0.01, 0.25, 0.08, 0.01)
        if st.button(" Detect Anomalies", use_container_width=False):
            with st.spinner("Scoring rows with IsolationForest..."):
                st.session_state.ml_anomaly = ml_analytics.detect_ml_anomalies(df, contamination)
        an = st.session_state.ml_anomaly
        if an:
            if not an.get("ok"):
                st.warning(an.get("error"))
            else:
                mc = st.columns(3)
                with mc[0]: st.markdown(render_metric("Anomalies", an["n_anomalies"], icon=""), unsafe_allow_html=True)
                with mc[1]: st.markdown(render_metric("Rows Scanned", an["total"], icon=""), unsafe_allow_html=True)
                with mc[2]: st.markdown(render_metric("Features", len(an["features_used"]), icon=""), unsafe_allow_html=True)
                tbl = an["table"]
                feats = an["features_used"]
                if len(feats) >= 2:
                    coords = tbl[feats[:3]].apply(pd.to_numeric, errors="coerce").fillna(0.0).head(120).copy()
                    score_series = pd.to_numeric(tbl["anomaly_score"], errors="coerce").fillna(0.0)
                    score_min = float(score_series.min())
                    score_range = float(score_series.max() - score_min) or 1.0
                    for axis in feats[:3]:
                        axis_min = float(coords[axis].min())
                        axis_span = float(coords[axis].max() - axis_min) or 1.0
                        coords[axis] = ((coords[axis] - axis_min) / axis_span - 0.5) * 12.0
                    render_three_analytics(
                        "anomaly_scatter",
                        {
                            "type": "scatter",
                            "title": f"{feats[0]} / {feats[1]} / {feats[2] if len(feats) >= 3 else 'anomaly score'}",
                            "data": [
                                {
                                    "x": round(float(coords.iloc[idx][feats[0]]), 3),
                                    "y": round(float(coords.iloc[idx][feats[1]]), 3),
                                    "z": round(float(coords.iloc[idx][feats[2]]) if len(feats) >= 3 else float(((score_series.iloc[idx] - score_min) / score_range - 0.5) * 10.0), 3),
                                    "size": round(0.24 + ((float(score_series.iloc[idx]) - score_min) / score_range) * 0.52, 3),
                                    "color": "#ef4444" if tbl.iloc[idx]["anomaly"] == " anomaly" else "#22c55e",
                                }
                                for idx in range(len(coords))
                            ],
                        },
                        height=380,
                    )
                    anomaly_rate = (an["n_anomalies"] / max(an["total"], 1)) * 100
                    st.markdown(
                        f"""<div class="data-card">
                        <b>Anomaly Story</b><br><br>
                        <b>{an["n_anomalies"]}</b> rows are unusual out of <b>{an["total"]}</b> scanned
                        records, which is an anomaly rate of <b>{anomaly_rate:.2f}%</b>.<br><br>
                        The 3D view is scaled to make the separation easier to read. Red markers indicate the
                        most suspicious records and larger points carry stronger anomaly scores.
                        </div>""",
                        unsafe_allow_html=True,
                    )
                st.dataframe(tbl.head(25), use_container_width=True, height=300)

                # ─ Column anomaly contribution ─
                try:
                    znum = tbl[feats].apply(pd.to_numeric, errors="coerce")
                    zz = ((znum - znum.mean()) / znum.std(ddof=0).replace(0, np.nan)).abs()
                    anom_mask_rows = tbl["anomaly"].astype(str).str.contains("anomaly")
                    contrib = zz[anom_mask_rows.values].mean().sort_values(ascending=False)
                    if contrib.notna().any():
                        st.markdown("#### Column Anomaly Contribution")
                        cb = go.Figure(go.Bar(x=contrib.values, y=contrib.index.astype(str), orientation="h",
                                              marker=dict(color=contrib.values, colorscale="Reds")))
                        cb.update_layout(title="Avg |Z| among anomalies (which columns drive anomalies)",
                                         yaxis=dict(autorange="reversed"))
                        style_fig(cb, 320); st.plotly_chart(cb, use_container_width=True)
                except Exception:
                    pass

                # ─ Anomaly timeline ─
                date_a = analyst.detect_date_col(tbl)
                if date_a:
                    tl = tbl.copy()
                    tl["_d"] = pd.to_datetime(tl[date_a], errors="coerce")
                    tl = tl.dropna(subset=["_d"])
                    if not tl.empty:
                        st.markdown("#### Anomaly Timeline")
                        tlf = px.scatter(tl, x="_d", y="anomaly_score", color="anomaly",
                                         color_discrete_map={"🔴 anomaly": "#ef4444", "🟢 normal": "#22c55e"},
                                         title="Anomaly score over time")
                        style_fig(tlf, 320); st.plotly_chart(tlf, use_container_width=True)

                # ─ Explain top anomalies with AI ─
                if st.button(" Explain Top Anomalies with AI", key="anom_ai"):
                    if not st.session_state.api_key:
                        st.warning("Add your OpenRouter API key in `.env` to enable AI explanations.")
                    else:
                        rows = [" | ".join(f"{c}={r[c]}" for c in tbl.columns[:10]) for _, r in tbl.head(5).iterrows()]
                        try:
                            agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
                            with st.spinner("🧠 Explaining anomalies..."):
                                ans = agent.answer_with_context(
                                    "Why might these rows be flagged as anomalies and what should we check?",
                                    rows, st.session_state.df_name or "Dataset")
                            st.markdown(f'<div class="data-card"><b>🧠 Anomaly Explanation</b><br><br>{ans}</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"AI error: {e}")

    # ─ RAG ask your data ─
    with tab_rag:
        st.caption("Retrieval-augmented Q&A: top-k rows are pulled from the vector DB and used as evidence.")
        for turn in st.session_state.rag_history:
            st.markdown(f'<div class="chat-message chat-user"> {turn["q"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="chat-ai-label"> DataSage (RAG)</div>', unsafe_allow_html=True)
            st.markdown(turn["a"])
            with st.expander(" Retrieved evidence rows"):
                for ev in turn["ctx"]:
                    st.markdown(f"<div class='fix-item'>{ev}</div>", unsafe_allow_html=True)
        rcol1, rcol2 = st.columns([5, 1])
        with rcol1:
            rag_q = st.text_input("Ask a question about your data", placeholder="Which records drive the most risk and why?", label_visibility="collapsed")
        with rcol2:
            ask = st.button("Ask ", use_container_width=True)
        if ask and rag_q.strip():
            agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
            if st.session_state.vector_store is None or not st.session_state.vector_store.is_ready():
                vstore = VectorStore(agent)
                with st.spinner(" Building vector index..."):
                    ok, _ = vstore.build(df)
                if ok:
                    st.session_state.vector_store = vstore
                    st.session_state.vs_status = f"{vstore.backend} · {vstore.n_rows} rows"
            vs = st.session_state.vector_store
            if vs is None or not vs.is_ready():
                st.error(" Vector index unavailable (embedding model rate limited). Try again.")
            else:
                with st.spinner(" Retrieving + reasoning..."):
                    ctx = vs.retrieve_context(rag_q, df, k=6)
                    answer = agent.answer_with_context(rag_q, ctx, st.session_state.df_name or "Dataset")
                st.session_state.rag_history.append({"q": rag_q, "a": answer, "ctx": ctx})
                st.rerun()

    # ─ Time Series Analysis ─
    with tab_ts:
        date_t = analyst.detect_date_col(df)
        rev_t = analyst.detect_revenue_col(df)
        if date_t and rev_t:
            series = analyst.resample_revenue(df, date_t, rev_t, "ME")
        else:
            _nc = numeric_frame.columns.tolist()
            series = numeric_frame[_nc[0]].dropna().reset_index(drop=True) if _nc else pd.Series(dtype=float)
        if len(series) < 6:
            st.info("Need a longer numeric/time series (≥6 points) for time-series analysis.")
        else:
            st.markdown("#### Decomposition (trend / seasonal / residual)")
            period = 12 if len(series) >= 24 else max(2, len(series) // 3)
            dec = analyst.decompose_series(series, period)
            xs = list(dec["observed"].index)
            sub = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Trend", "Seasonal", "Residual"])
            sub.add_trace(go.Scatter(x=xs, y=dec["trend"].values, line=dict(color="#00d4ff"), name="Trend"), row=1, col=1)
            sub.add_trace(go.Scatter(x=xs, y=dec["seasonal"].values, line=dict(color="#a855f7"), name="Seasonal"), row=2, col=1)
            sub.add_trace(go.Scatter(x=xs, y=dec["residual"].values, line=dict(color="#f59e0b"), name="Residual"), row=3, col=1)
            sub.update_layout(showlegend=False, title="Series decomposition (rolling-mean based)")
            style_fig(sub, 420); st.plotly_chart(sub, use_container_width=True)
            st.session_state.ts_decomposition = {"period": period, "n": len(series)}

            st.markdown("#### Autocorrelation (ACF)")
            acf = analyst.compute_acf(series, min(24, len(series) - 1))
            if not acf.empty:
                af = go.Figure()
                af.add_trace(go.Bar(x=acf["lag"], y=acf["acf"], marker_color="#00d4ff", name="ACF"))
                af.add_trace(go.Scatter(x=acf["lag"], y=acf["upper"], line=dict(color="#ef4444", dash="dash"), name="95% bound"))
                af.add_trace(go.Scatter(x=acf["lag"], y=acf["lower"], line=dict(color="#ef4444", dash="dash"), showlegend=False))
                af.update_layout(title="ACF with ±1.96/√n significance bounds")
                style_fig(af, 300); st.plotly_chart(af, use_container_width=True)

            st.markdown("#### Rolling Statistics")
            rs = pd.DataFrame({"value": series.values})
            rs["mean7"] = rs["value"].rolling(7, min_periods=1).mean()
            rs["mean30"] = rs["value"].rolling(30, min_periods=1).mean()
            rs["std"] = rs["value"].rolling(7, min_periods=1).std()
            rf = go.Figure()
            rf.add_trace(go.Scatter(y=rs["value"], name="Original", line=dict(color="#94a3b8")))
            rf.add_trace(go.Scatter(y=rs["mean7"], name="7-period mean", line=dict(color="#00d4ff")))
            rf.add_trace(go.Scatter(y=rs["mean30"], name="30-period mean", line=dict(color="#a855f7")))
            rf.add_trace(go.Scatter(y=rs["std"], name="Rolling std", line=dict(color="#f59e0b", dash="dot")))
            rf.update_layout(title="Rolling statistics")
            style_fig(rf, 320); st.plotly_chart(rf, use_container_width=True)

            if isinstance(series.index, pd.DatetimeIndex) and series.index.year.nunique() >= 2:
                st.markdown("#### Year-over-Year Monthly Comparison")
                yoy = pd.DataFrame({"v": series.values, "year": series.index.year, "month": series.index.month})
                pv = yoy.pivot_table(index="month", columns="year", values="v", aggfunc="sum")
                yf = go.Figure()
                for yr in pv.columns:
                    yf.add_trace(go.Scatter(x=pv.index, y=pv[yr].values, mode="lines+markers", name=str(yr)))
                yf.update_layout(title="YoY monthly comparison", xaxis_title="Month")
                style_fig(yf, 320); st.plotly_chart(yf, use_container_width=True)

    # ─ Segmentation Analysis ─
    with tab_seg:
        _nc = numeric_frame.columns.tolist()
        if len(_nc) < 2:
            st.info("Need ≥2 numeric columns for clustering.")
        else:
            kk = st.slider("Number of clusters (k)", 2, 8, 3, key="seg_k")
            if st.button("Run K-Means", key="seg_run"):
                with st.spinner("Clustering..."):
                    try:
                        dfc, prof = analyst.compute_kmeans(df, _nc, kk)
                        st.session_state.ml_clusters = {"df": dfc, "profile": prof, "cols": _nc}
                    except Exception as e:
                        st.error(f"Clustering failed: {e}")
            clus = st.session_state.ml_clusters
            if clus:
                dfc = clus["df"]; prof = clus["profile"]; cols_used = clus["cols"]
                sc1, sc2 = st.columns(2)
                with sc1:
                    sizes = dfc["cluster"].value_counts().sort_index()
                    bsz = go.Figure(go.Bar(x=[f"C{i}" for i in sizes.index], y=sizes.values, marker_color="#00d4ff"))
                    bsz.update_layout(title="Cluster sizes")
                    style_fig(bsz, 300); st.plotly_chart(bsz, use_container_width=True)
                with sc2:
                    sct = px.scatter(dfc, x=cols_used[0], y=cols_used[1], color=dfc["cluster"].astype(str),
                                     title=f"Clusters: {cols_used[0]} vs {cols_used[1]}", labels={"color": "cluster"})
                    style_fig(sct, 300); st.plotly_chart(sct, use_container_width=True)
                st.markdown("#### Cluster Profiles (mean values)")
                st.dataframe(prof.round(2), use_container_width=True)
                seg_col = analyst.find_col(df, ["segment", "customer_type"])
                if seg_col and len(dfc) == len(df):
                    ct = pd.crosstab(df[seg_col].astype(str), dfc["cluster"])
                    hm = go.Figure(go.Heatmap(z=ct.values, x=[f"C{c}" for c in ct.columns], y=ct.index.astype(str), colorscale="Blues"))
                    hm.update_layout(title=f"{seg_col} vs cluster")
                    style_fig(hm, 300); st.plotly_chart(hm, use_container_width=True)
                if st.button("Name and Describe Clusters with AI", key="seg_ai"):
                    if not st.session_state.api_key:
                        st.warning("Add your OpenRouter API key in `.env` to enable AI cluster naming.")
                    else:
                        prompt = (f"These are KMeans cluster profiles (mean feature values) for {st.session_state.df_name}:\n"
                                  f"{prof.round(2).to_string(index=False)}\n"
                                  "For each cluster give a short business-friendly name, a one-sentence profile, and one action. Be concise.")
                        try:
                            agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
                            with st.spinner("🧠 Naming clusters..."):
                                st.session_state.segment_ai_names = agent.chat(prompt)
                        except Exception as e:
                            st.error(f"AI error: {e}")
                if st.session_state.segment_ai_names:
                    st.markdown(f'<div class="data-card"><b>🧠 AI Cluster Profiles</b><br><br>{st.session_state.segment_ai_names}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SEMANTIC SEARCH  (AI embeddings)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Semantic Search":
    st.markdown("## Semantic Search")
    st.markdown('<p class="hero-sub">Search your data by <b>meaning</b>, not keywords — backed by a real <b>vector database</b>.</p>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Upload a dataset in the Data Hub to search it semantically.")
        st.stop()

    df = st.session_state.df
    vs = st.session_state.vector_store

    cstat = st.columns([3, 1])
    with cstat[0]:
        st.markdown(f"""<div class="alert-info">
             Rows are embedded with <code>{config.DEFAULT_EMBED_MODEL}</code> and indexed in a
            <b>vector DB</b>. Status:
            <span class="status-badge badge-{'green' if vs and vs.is_ready() else 'yellow'}">
            {st.session_state.vs_status or 'index not built'}</span>
        </div>""", unsafe_allow_html=True)
    with cstat[1]:
        if st.button(" Build / Refresh Index", use_container_width=True):
            agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
            vstore = VectorStore(agent)
            with st.spinner(" Embedding rows & building vector index..."):
                ok, msg = vstore.build(df)
            if ok:
                st.session_state.vector_store = vstore
                st.session_state.vs_status = f"{vstore.backend} · {vstore.n_rows} rows · dim {vstore.dim}"
                st.success(f" {msg}")
            else:
                st.error(f" {msg}")
            st.rerun()

    st.markdown("###### 💡 Try an example")
    examples = ["unusually large sales", "low stock items", "premium customers", "records with anomalies", "highest revenue region"]
    ex_cols = st.columns(len(examples))
    for ex, ecol in zip(examples, ex_cols):
        with ecol:
            if st.button(ex, key=f"semex_{ex}", use_container_width=True):
                st.session_state["sem_query"] = ex
                st.rerun()

    c1, c2, c3 = st.columns([4, 1, 1])
    with c1:
        query = st.text_input("Search query", placeholder="e.g. unusually large sales in the south region",
                              label_visibility="collapsed", key="sem_query")
    with c2:
        top_k = st.number_input("Top K", 1, 25, 5)
    with c3:
        run_search = st.button(" Search", use_container_width=True)

    if run_search and query.strip():
        if st.session_state.vector_store is None or not st.session_state.vector_store.is_ready():
            agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
            vstore = VectorStore(agent)
            with st.spinner(" Building index (first search)..."):
                ok, msg = vstore.build(df)
            if ok:
                st.session_state.vector_store = vstore
                st.session_state.vs_status = f"{vstore.backend} · {vstore.n_rows} rows · dim {vstore.dim}"
        vs = st.session_state.vector_store
        if vs is None or not vs.is_ready():
            st.error(" Could not build vector index (embedding model unavailable / rate limited).")
        else:
            with st.spinner(" Querying vector DB..."):
                results, status = vs.search(query, df, k=int(top_k))
            st.session_state.semantic_results = results
            st.success(f" {status}") if results is not None else st.error(f" {status}")

    if st.session_state.semantic_results is not None:
        res = st.session_state.semantic_results
        st.markdown("### Top Matches")
        st.dataframe(res, use_container_width=True, height=320)
        if " similarity" in res.columns:
            fig = px.bar(res, x=" similarity", y=res.index.astype(str), orientation="h",
                         color=" similarity", color_continuous_scale="Tealgrn",
                         title="Similarity score by row")
            style_fig(fig, 300)
            fig.update_layout(yaxis_title="row index")
            st.plotly_chart(fig, use_container_width=True)
        st.download_button("⬇ Download Matches CSV", res.to_csv(index=False).encode(),
                           "semantic_matches.csv", "text/csv")

        # ─── RAG: ask AI to reason over the retrieved rows ──────────────────────
        st.markdown("### 🤖 Ask AI about these matches")
        st.caption("Retrieval-augmented answer — the AI reasons only over the top rows above (grounded, with no hallucinated data).")
        rag_q = st.text_input("Your question about the matches",
                              placeholder="e.g. what do these rows have in common?", key="rag_question")
        if st.button("💬 Answer from matches", use_container_width=True):
            if not st.session_state.api_key:
                st.warning("Add an OpenRouter API key in `.env` to enable AI answers.")
            elif not (rag_q or "").strip():
                st.warning("Type a question first.")
            else:
                context_rows = [" | ".join(f"{c}={r[c]}" for c in res.columns) for _, r in res.head(8).iterrows()]
                try:
                    agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
                    with st.spinner("🧠 Reasoning over retrieved rows…"):
                        answer = agent.answer_with_context(rag_q, context_rows, st.session_state.df_name)
                    st.markdown(f'<div class="data-card"><b>🧠 Grounded Answer</b><br><br>{answer}</div>',
                                unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not generate an answer: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EXECUTIVE REPORT  (AI generated)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Executive Report":
    st.markdown("## AI Executive Report")
    st.markdown('<p class="hero-sub">One click → a board-ready data-quality report written by AI.</p>', unsafe_allow_html=True)

    if st.session_state.df is None:
        st.info("Upload a dataset in the Data Hub first.")
        st.stop()

    df = st.session_state.df
    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown(f"""<div class="data-card">
            <b> Report inputs</b><br><br>
             Dataset: <b>{st.session_state.df_name or 'Unnamed'}</b><br>
             {df.shape[0]:,} rows × {df.shape[1]} cols<br>
             Quality score: <b>{st.session_state.quality_report['quality_score'] if st.session_state.quality_report else '— (run Quality AI)'}</b><br>
             Pipeline run: <b>{' yes' if st.session_state.run_complete else '⏳ not yet'}</b>
        </div>""", unsafe_allow_html=True)
    with colB:
        st.markdown("""<div class="data-card">
            <b> Tip</b><br><br>
            For the richest report, first run the <b>Quality AI</b> scan and the
            <b>Self-Healing Pipeline</b>, then generate the report so the AI can reference real scores.
        </div>""", unsafe_allow_html=True)

    # ─── Enhancement A: report type selector ────────────────────────────────────
    report_types = ["Sales Performance Report", "Financial Health Report", "Data Quality Report",
                    "Customer Intelligence Report", "Export Market Report"]
    rtype = st.radio("Report type", report_types, horizontal=True, key="exec_report_type")

    # ─── Enhancement B: structure preview (feature-chip per section) ─────────────
    _structures = {
        "Sales Performance Report":     ["Executive Summary", "Revenue Trend", "Regional Performance", "Top Products", "Recommendations"],
        "Financial Health Report":      ["Executive Summary", "P&L Overview", "Margins & Ratios", "Risks", "Outlook"],
        "Data Quality Report":          ["Executive Summary", "Quality Score", "Issues by Dimension", "Remediation Plan"],
        "Customer Intelligence Report": ["Executive Summary", "Segments", "RFM Insights", "Retention Actions"],
        "Export Market Report":         ["Executive Summary", "Top Markets", "Category Mix", "Growth Opportunities"],
    }
    st.markdown("".join(f'<span class="feature-chip">{s}</span> ' for s in _structures[rtype]), unsafe_allow_html=True)

    if st.button(" Generate Executive Report", use_container_width=True):
        agent = DataAIAgent(st.session_state.api_key, st.session_state.selected_model)
        healing_summary = st.session_state.pipeline.get_pipeline_summary() if st.session_state.run_complete else None
        with st.spinner(" Writing your executive report..."):
            report_md = agent.generate_executive_report(
                df, st.session_state.quality_report, healing_summary,
                f"{st.session_state.df_name or 'Dataset'} — {rtype}")
        st.session_state.ai_report = report_md

    if st.session_state.ai_report:
        st.markdown("---")
        import re as _re
        report = st.session_state.ai_report

        def _inline_chart(text):
            t = text.lower()
            try:
                if any(w in t for w in ["revenue", "trend", "sales"]):
                    tr = analytics.sales_trend(df)
                    if tr is not None:
                        f = px.area(tr, x="date", y="sales", title="Revenue trend"); style_fig(f, 280); st.plotly_chart(f, use_container_width=True); return
                if "region" in t:
                    rg = analytics.group_performance(df, "region")
                    if rg is not None:
                        f = px.bar(rg, x=rg.columns[0], y="value", color="value", color_continuous_scale="Tealgrn", title="By region"); style_fig(f, 280); st.plotly_chart(f, use_container_width=True); return
                if "product" in t:
                    pr = analytics.group_performance(df, "product")
                    if pr is not None:
                        f = px.bar(pr, x="value", y=pr.columns[0], orientation="h", color="value", color_continuous_scale="Purp", title="Top products"); style_fig(f, 280); st.plotly_chart(f, use_container_width=True); return
                if any(w in t for w in ["customer", "segment"]):
                    seg = analyst.find_col(df, ["segment", "customer_type"])
                    if seg:
                        vc = df[seg].astype(str).value_counts()
                        f = px.pie(values=vc.values, names=vc.index, hole=0.5, title="Segments"); style_fig(f, 280); st.plotly_chart(f, use_container_width=True); return
                if "quality" in t:
                    comp = (1 - df.isnull().mean()) * 100
                    f = go.Figure(go.Bar(x=comp.index.astype(str), y=comp.values, marker_color="#22c55e"))
                    f.update_layout(title="Completeness %"); style_fig(f, 280); st.plotly_chart(f, use_container_width=True); return
            except Exception:
                pass

        # ─── Enhancement C + D: section-by-section render with inline charts ─────
        sections = _re.split(r'\n(?=##\s)', report)
        for i, sec in enumerate(sections):
            lines = sec.strip().split("\n", 1)
            title = lines[0].lstrip("# ").strip() or f"Section {i+1}"
            body = lines[1] if len(lines) > 1 else ""
            with st.expander(title, expanded=(i == 0)):
                st.markdown(body if body else sec)
                _inline_chart(sec)

        # ─── Enhancement E: three download buttons ──────────────────────────────
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("⬇ Markdown (.md)", report.encode(), "executive_report.md", "text/markdown")
        with d2:
            plain = _re.sub(r'[#*`_>]', '', report)
            st.download_button("⬇ Plain Text (.txt)", plain.encode(), "executive_report.txt", "text/plain")
        with d3:
            payload = {
                "report_type": rtype,
                "dataset": st.session_state.df_name,
                "rows": int(df.shape[0]), "cols": int(df.shape[1]),
                "quality_score": st.session_state.quality_report.get("quality_score") if st.session_state.quality_report else None,
                "kpis": st.session_state.analyst_kpis,
                "report_markdown": report,
            }
            st.download_button("⬇ Report + Data (.json)", json.dumps(payload, indent=2, default=str).encode(),
                               "executive_report.json", "application/json")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown("## Data Explorer")

    if st.session_state.df is None:
        st.info("Upload a dataset in the Data Hub to explore.")
        st.stop()

    df = st.session_state.df

    # Filter controls
    st.markdown("### Filter & Search")
    col1, col2, col3 = st.columns(3)
    with col1:
        search = st.text_input("Search in any column", "")
    with col2:
        sel_cols = st.multiselect("Select Columns", df.columns.tolist(), default=df.columns.tolist()[:8])
    with col3:
        show_n = st.slider("Rows to show", 5, min(500, len(df)), min(50, len(df)))

    filtered_df = df[sel_cols] if sel_cols else df
    if search:
        mask = filtered_df.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
        filtered_df = filtered_df[mask]

    st.dataframe(filtered_df.head(show_n), use_container_width=True, height=400)
    st.caption(f"Showing {min(show_n, len(filtered_df))} of {len(filtered_df)} rows")

    # Advanced per-column filters
    with st.expander("Advanced column filters"):
        adv = df.copy()
        fcols = st.multiselect("Columns to filter", df.columns.tolist(), key="exp_fcols")
        for fc in fcols:
            snum = pd.to_numeric(df[fc], errors="coerce")
            if snum.notna().mean() > 0.6:
                lo, hi = float(snum.min()), float(snum.max())
                if lo < hi:
                    r = st.slider(f"{fc} range", lo, hi, (lo, hi), key=f"exp_r_{fc}")
                    av = pd.to_numeric(adv[fc], errors="coerce")
                    adv = adv[(av >= r[0]) & (av <= r[1])]
            else:
                opts = df[fc].dropna().astype(str).unique().tolist()[:50]
                sel = st.multiselect(f"{fc} values", opts, key=f"exp_v_{fc}")
                if sel:
                    adv = adv[adv[fc].astype(str).isin(sel)]
        st.dataframe(adv.head(200), use_container_width=True, height=280)
        st.caption(f"{len(adv):,} rows after filters")
        st.download_button("Download filtered subset", adv.to_csv(index=False).encode(),
                           "explorer_subset.csv", "text/csv", key="exp_dl")

    # Visual analysis — many auto-generated charts
    st.markdown("### Visual Analysis")
    if st.button("Generate charts", key="exp_charts_btn"):
        st.session_state._exp_gallery = True
    if st.session_state.get("_exp_gallery"):
        render_chart_gallery(df)

    if st.session_state.sqlite_dataset_id:
        st.markdown("### SQLite Row Actions")
        st.caption("Add, edit, and delete rows for the active SQLite dataset.")
        row_cols = [c for c in df.columns if c != "_rowid_"]
        action_a, action_b, action_c = st.columns(3)
        with action_a:
            with st.form("sqlite_add_row"):
                st.markdown("#### Add row")
                add_values = {col: st.text_input(col, key=f"add_{col}") for col in row_cols}
                if st.form_submit_button("Add row", use_container_width=True):
                    ok, msg = sqlite_store.add_dataset_row(st.session_state.sqlite_dataset_id, add_values)
                    if ok:
                        log_action("ROW_ADD", st.session_state.df_name)
                        df_sql, meta, _ = sqlite_store.load_dataset(st.session_state.sqlite_dataset_id)
                        apply_dataset(df_sql.drop(columns=["_rowid_"], errors="ignore"), meta["name"])
                        set_active_sqlite_dataset(int(meta["id"]), meta)
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
        df_sql, meta, _ = sqlite_store.load_dataset(st.session_state.sqlite_dataset_id)
        if df_sql is not None and not df_sql.empty:
            row_map = {f"row {int(row._rowid_)}": int(row._rowid_) for row in df_sql.head(100).itertuples(index=False)}
            chosen = action_b.selectbox("Row to edit", list(row_map.keys()), key="sqlite_edit_row")
            current = df_sql[df_sql["_rowid_"] == row_map[chosen]].iloc[0]
            with action_b.form("sqlite_update_row"):
                st.markdown("#### Edit row")
                edit_values = {col: st.text_input(col, value="" if pd.isna(current[col]) else str(current[col]), key=f"edit_{col}") for col in row_cols}
                if st.form_submit_button("Update row", use_container_width=True):
                    ok, msg = sqlite_store.update_dataset_row(st.session_state.sqlite_dataset_id, row_map[chosen], edit_values)
                    if ok:
                        log_action("ROW_EDIT", f"{st.session_state.df_name}:{row_map[chosen]}")
                        df_sql, meta, _ = sqlite_store.load_dataset(st.session_state.sqlite_dataset_id)
                        apply_dataset(df_sql.drop(columns=["_rowid_"], errors="ignore"), meta["name"])
                        set_active_sqlite_dataset(int(meta["id"]), meta)
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            with action_c.form("sqlite_delete_row"):
                st.markdown("#### Delete row")
                del_row = st.selectbox("Row", list(row_map.keys()), key="sqlite_delete_row")
                if st.form_submit_button("Delete row", use_container_width=True):
                    ok, msg = sqlite_store.delete_dataset_row(st.session_state.sqlite_dataset_id, row_map[del_row])
                    if ok:
                        log_action("ROW_DELETE", f"{st.session_state.df_name}:{row_map[del_row]}")
                        df_sql, meta, _ = sqlite_store.load_dataset(st.session_state.sqlite_dataset_id)
                        apply_dataset(df_sql.drop(columns=["_rowid_"], errors="ignore"), meta["name"])
                        set_active_sqlite_dataset(int(meta["id"]), meta)
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
    else:
        st.info("Save or load this dataset through SQLite in Data Hub to enable row add, edit, and delete actions.")


    # Column profiling
    st.markdown("### Column Profiler")
    profile_col = st.selectbox("Select a column to profile", df.columns.tolist())
    series = df[profile_col]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Unique Values", series.nunique())
    c2.metric("Missing", int(series.isnull().sum()))
    c3.metric("Data Type", str(series.dtype))
    c4.metric("Most Common", str(series.mode().iloc[0]) if series.notna().any() else "N/A")

    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().sum() > 3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(numeric.dropna(), nbins=20, title=f"Distribution: {profile_col}")
            fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color="#94a3b8"), margin=dict(t=30,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(numeric.dropna(), title=f"Box Plot: {profile_col}")
            fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font=dict(color="#94a3b8"), margin=dict(t=30,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        vc = series.value_counts().head(15)
        fig = px.bar(x=vc.index.astype(str), y=vc.values, title=f"Value Counts: {profile_col}", color_discrete_sequence=["#00d4ff"])
        fig.update_layout(height=260, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(color="#94a3b8"), margin=dict(t=30,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    # Download (role-based: non-admins get PII masked)
    _is_admin = st.session_state.auth and st.session_state.auth["role"] == security.ADMIN
    out_df, _mcols = security.mask_dataframe(filtered_df.head(show_n), reveal=bool(_is_admin))
    if _mcols and not _is_admin:
        st.caption(f" Exported file has masked PII columns: {', '.join(_mcols)} (Admin role exports raw values).")
    if st.download_button("⬇ Download Filtered CSV", out_df.to_csv(index=False).encode(),
                          "filtered_data.csv", "text/csv", use_container_width=False):
        log_action("DOWNLOAD", "filtered_data.csv")
