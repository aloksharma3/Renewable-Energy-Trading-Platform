"""
AI Energy Trading Dashboard (v3 — Professional)
==================================================
Production-grade Streamlit dashboard for the renewable energy trading platform.

DATA SOURCES (all via API calls):
    data_service:8001     → weather, ERCOT RT prices, DAM prices, RT-DAM spread
    forecast:8002         → ML predictions, model metrics, feature importance
    trading:8003          → portfolio, trade history, position
    rag:8004              → market intelligence, risk assessment

LAYOUT:
    Sidebar:  System status, controls, service health
    Main:
        Row 1 — KPI cards (position, profit, trades, win rate, RT price, DAM spread)
        Row 2 — Price chart (RT vs DAM vs Predicted + confidence band + trade markers)
        Row 3 — Energy output chart + Grid demand chart
        Row 4 — Weather conditions + RAG market intelligence
        Row 5 — Latest trade card + ERCOT market overview
        Row 6 — Trade history table
        Row 7 — Cumulative profit chart
        Row 8 — Model performance + Feature importance
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Energy Trading Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Service URLs ────────────────────────────────────────
DATA_URL = os.getenv("DATA_SERVICE_URL", "http://data_service:8001")
FORECAST_URL = os.getenv("FORECAST_SERVICE_URL", "http://forecast:8002")
TRADING_URL = os.getenv("TRADING_SERVICE_URL", "http://trading:8003")
RAG_URL = os.getenv("RAG_SERVICE_URL", "http://rag:8004")


# ═══════════════════════════════════════════════════════════
#  CUSTOM CSS — Dark Professional Theme
# ═══════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #111827;
        --bg-card: #1a1f2e;
        --border-subtle: #2a3040;
        --border-accent: #3b82f6;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-blue: #3b82f6;
        --accent-cyan: #06b6d4;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-amber: #f59e0b;
        --accent-purple: #8b5cf6;
    }

    .stApp {
        background: var(--bg-primary);
        font-family: 'DM Sans', sans-serif;
    }

    .stApp header { background: transparent !important; }

    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle);
    }

    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 16px 20px;
        transition: border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover { border-color: var(--border-accent); }

    [data-testid="stMetricLabel"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.78rem;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.5rem;
        color: var(--text-primary) !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
    }

    .section-header {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        color: var(--text-primary);
        padding: 8px 0 12px 0;
        margin-top: 16px;
        border-bottom: 2px solid var(--border-subtle);
        letter-spacing: -0.01em;
    }
    .section-header .icon { margin-right: 8px; }

    .info-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
    }
    .info-card:hover { border-color: var(--border-accent); }
    .info-card h4 {
        font-family: 'DM Sans', sans-serif;
        color: var(--text-primary);
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin: 0 0 12px 0;
    }
    .info-card .value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.2;
    }
    .info-card .sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 6px;
    }

    .badge-buy {
        display:inline-block; background:rgba(16,185,129,0.15); color:var(--accent-green);
        border:1px solid rgba(16,185,129,0.3); padding:4px 14px; border-radius:20px;
        font-family:'JetBrains Mono',monospace; font-weight:600; font-size:0.8rem;
    }
    .badge-sell {
        display:inline-block; background:rgba(239,68,68,0.15); color:var(--accent-red);
        border:1px solid rgba(239,68,68,0.3); padding:4px 14px; border-radius:20px;
        font-family:'JetBrains Mono',monospace; font-weight:600; font-size:0.8rem;
    }
    .badge-hold {
        display:inline-block; background:rgba(245,158,11,0.15); color:var(--accent-amber);
        border:1px solid rgba(245,158,11,0.3); padding:4px 14px; border-radius:20px;
        font-family:'JetBrains Mono',monospace; font-weight:600; font-size:0.8rem;
    }

    .spread-positive { color: var(--accent-green); }
    .spread-negative { color: var(--accent-red); }
    .spread-neutral  { color: var(--text-muted); }

    .status-dot {
        display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px;
        animation: pulse 2s ease-in-out infinite;
    }
    .status-dot.online  { background: var(--accent-green); }
    .status-dot.offline { background: var(--accent-red); }
    @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.5;} }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        overflow: hidden;
    }

    hr { border:none; border-top:1px solid var(--border-subtle); margin:24px 0; }

    .stPlotlyChart {
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        overflow: hidden;
    }

    .footer {
        font-family:'DM Sans',sans-serif; font-size:0.73rem; color:var(--text-muted);
        text-align:center; padding:24px 0 16px 0;
        border-top:1px solid var(--border-subtle); margin-top:32px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PLOTLY THEME
# ═══════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(26, 31, 46, 1)",
    plot_bgcolor="rgba(26, 31, 46, 1)",
    font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=16, r=16, t=32, b=16),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        font=dict(size=11, color="#94a3b8"), bgcolor="rgba(0,0,0,0)",
    ),
    xaxis=dict(gridcolor="rgba(42,48,64,0.5)", zerolinecolor="rgba(42,48,64,0.5)"),
    yaxis=dict(gridcolor="rgba(42,48,64,0.5)", zerolinecolor="rgba(42,48,64,0.5)"),
)


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════

def safe_get(url, default=None):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return default


def safe_post(url, json_data=None, default=None):
    try:
        r = requests.post(url, json=json_data or {}, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return default


def check_service(url):
    try:
        r = requests.get(f"{url}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════
#  BACKTEST ENGINE  (imported from backtest_engine.py)
# ═══════════════════════════════════════════════════════════

from backtest_engine import run_backtest


# ═══════════════════════════════════════════════════════════
#  FETCH ALL DATA
# ═══════════════════════════════════════════════════════════


portfolio = safe_get(f"{TRADING_URL}/portfolio", {})
forecast_history = safe_get(f"{FORECAST_URL}/forecast/history?limit=50", [])  # keep low for fast render
trade_history = safe_get(f"{TRADING_URL}/trades?limit=100", [])
weather = safe_get(f"{DATA_URL}/weather/latest")
weather_hist = safe_get(f"{DATA_URL}/weather/history?limit=48", [])
ercot = safe_get(f"{DATA_URL}/ercot/latest")
dam_today = safe_get(f"{DATA_URL}/dam/today")
spread = safe_get(f"{DATA_URL}/spread/rt-dam")
latest_forecast = safe_get(f"{FORECAST_URL}/forecast/latest", {})
metrics = safe_get(f"{FORECAST_URL}/models/metrics", {})
importance = safe_get(f"{FORECAST_URL}/models/feature-importance", {})
from zoneinfo import ZoneInfo

CT_TZ = ZoneInfo("America/Chicago")
now_ct = datetime.now(CT_TZ)


# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 16px 0;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; color:#f1f5f9; letter-spacing:-0.02em;">
            ⚡ AI Energy<br/>Trading Platform
        </div>
        <div style="font-family:'DM Sans',sans-serif; font-size:0.75rem; color:#64748b; margin-top:4px;">
            ERCOT HB_NORTH · Dallas, TX
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown('<h4 style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:12px;">System Status</h4>', unsafe_allow_html=True)

    for name, url in {"Data Service": DATA_URL, "Forecast Engine": FORECAST_URL,
                       "Trading Engine": TRADING_URL, "RAG Intelligence": RAG_URL}.items():
        online = check_service(url)
        dot = "online" if online else "offline"
        clr = "#10b981" if online else "#ef4444"
        txt = "Online" if online else "Offline"
        st.markdown(
            f'<div style="display:flex; align-items:center; padding:4px 0; font-size:0.8rem; color:#94a3b8;">'
            f'<span class="status-dot {dot}"></span> {name} — '
            f'<span style="color:{clr}; margin-left:4px;">{txt}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown(f"""
    <div style="font-size:0.75rem; color:#64748b;">
        <div style="margin-bottom:8px;">
            <span style="color:#94a3b8;">LOCAL (CT)</span>&nbsp;&nbsp;
            <span style="font-family:'JetBrains Mono',monospace; color:#f1f5f9;">{now_ct.strftime("%I:%M:%S %p")}</span>
        </div>
        <div style="margin-bottom:8px;">
            <span style="color:#94a3b8;">DATE</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <span style="font-family:'JetBrains Mono',monospace; color:#f1f5f9;">{now_ct.strftime("%b %d, %Y")}</span>
        </div>
        <div>Pipeline: every 15 min</div>
        <div>DAM refresh: 2:15 PM CT</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()

    with st.expander("⚙️ Manual Controls"):
        if st.button("▶ Run Full Pipeline", use_container_width=True):
            safe_post(f"{DATA_URL}/weather/update")
            safe_post(f"{DATA_URL}/ercot/update")
            safe_post(f"{FORECAST_URL}/forecast/run")
            safe_post(f"{TRADING_URL}/trade/execute")
            st.rerun()
        if st.button("📥 Fetch DAM Prices", use_container_width=True):
            safe_post(f"{DATA_URL}/dam/update")
            st.rerun()
        if st.button("🔃 Reset Portfolio", use_container_width=True):
            safe_post(f"{TRADING_URL}/reset")
            st.rerun()


# ═══════════════════════════════════════════════════════════
#  MAIN HEADER
# ═══════════════════════════════════════════════════════════

st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:flex-end; padding-bottom:8px; border-bottom:2px solid #2a3040;">
    <div>
        <h1 style="font-family:'DM Sans',sans-serif; font-weight:700; font-size:1.8rem; color:#f1f5f9; margin:0; letter-spacing:-0.02em;">
            Market Overview
        </h1>
        <p style="font-family:'DM Sans',sans-serif; font-size:0.8rem; color:#64748b; margin:4px 0 0 0;">
            Real-time ERCOT market data · ML ensemble forecasting · Automated trading
        </p>
    </div>
    <div style="text-align:right; font-family:'JetBrains Mono',monospace;">
        <div style="font-size:0.7rem; color:#64748b;">LAST UPDATE</div>
        <div style="font-size:0.85rem; color:#94a3b8;">{}</div>
    </div>
</div>
""".format(latest_forecast.get("timestamp", now_ct.strftime("%Y-%m-%d %I:%M %p CT"))), unsafe_allow_html=True)

st.markdown("")


# ═══════════════════════════════════════════════════════════
#  ROW 1 — KPI CARDS (5 primary + secondary row)
# ═══════════════════════════════════════════════════════════

# ── Primary KPIs — most important metrics ────────────────
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    profit = portfolio.get("realized_profit", 0)
    st.metric("REALIZED P&L", f"${profit:,.2f}",
              delta=f"${profit:+,.2f}" if profit != 0 else None, delta_color="normal")
with k2:
    rt_price = ercot.get("price_usd_mwh") if ercot else None
    freshness = f"updated {latest_forecast.get('timestamp', '—')[-8:-3]} CT" if latest_forecast.get('timestamp') else "awaiting data"
    if rt_price and rt_price > 0:
        st.metric("RT PRICE", f"${rt_price:.2f}", delta=freshness)
    else:
        st.metric("RT PRICE", "—", delta="awaiting ERCOT data")
with k3:
    pred_price = latest_forecast.get("price", {}).get("predicted")
    if pred_price is not None:
        delta_text = f"{'▲' if pred_price > (rt_price or 0) else '▼'} vs RT · signal {'+' if pred_price > (rt_price or 0) else ''}{(pred_price - (rt_price or 0)):.2f}"
        st.metric("ML PREDICTED", f"${pred_price:.2f}", delta=delta_text)
    else:
        st.metric("ML PREDICTED", "—", delta="initializing")
with k4:
    win_rate = portfolio.get("win_rate", 0)
    sells_count = portfolio.get("sell_count", 0)
    total = portfolio.get("total_trades", 0)
    buys = portfolio.get("buy_count", 0)
    sells = portfolio.get("sell_count", 0)
    holds = total - buys - sells
    st.metric("WIN RATE", f"{win_rate}%" if sells_count > 0 else "—",
              delta=f"{buys}B · {sells}S · {holds}H")
with k5:
    st.metric("POSITION", f"{portfolio.get('current_position_mwh', 0)} MWh",
              delta=f"avg cost ${portfolio.get('avg_cost', 0):.2f}/MWh" if portfolio.get('avg_cost') else None)

# ── Secondary KPIs ────────────────────────────────────────
st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
s1, s2, s3 = st.columns(3)

with s1:
    pred_energy = latest_forecast.get("energy_output", {}).get("predicted")
    current_ghi = latest_forecast.get("weather", {}).get("irradiance", 0)
    night_note  = " · nighttime" if current_ghi == 0 else f" · GHI {current_ghi} W/m²"
    if pred_energy is not None:
        st.metric("SOLAR OUTPUT", f"{pred_energy:.1f} MW", delta=f"{'☀️' if current_ghi > 0 else '🌙'}{night_note}")
    else:
        st.metric("SOLAR OUTPUT", "—", delta="initializing")
with s2:
    if spread and spread.get("spread") is not None:
        sp = spread["spread"]
        st.metric("RT-DAM SPREAD", f"${sp:+.2f}",
                  delta=f"{spread.get('signal','—')} · {spread.get('signal_strength','—')}",
                  delta_color="normal" if sp >= 0 else "inverse")
    else:
        st.metric("RT-DAM SPREAD", "N/A", delta="awaiting DAM data")
with s3:
    price_history_depth = len(recent_prices) if 'recent_prices' in dir() else 0
    sell_thresh = portfolio.get("sell_threshold", "—")
    buy_thresh  = portfolio.get("buy_threshold", "—")
    st.metric("PIPELINE CYCLES", f"{len(forecast_history or [])}", delta="15-min intervals · DB persisted")


# ═══════════════════════════════════════════════════════════
#  TABS — Navigation
# ═══════════════════════════════════════════════════════════

tab_market, tab_trading, tab_model = st.tabs([
    "📈 Market & Forecasts",
    "💰 Trading & Performance",
    "🧪 Model Analytics",
])

with tab_market:

    # ── PRICE CHART ──────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="icon">📈</span>Price Analysis — Predicted vs Actual vs Day-Ahead</div>', unsafe_allow_html=True)

if forecast_history and len(forecast_history) > 1:
    timestamps, predicted_prices, actual_prices, dam_prices_series = [], [], [], []
    conf_lowers, conf_uppers = [], []

    for f in forecast_history:
        timestamps.append(f.get("timestamp", ""))
        pd_ = f.get("price", {})
        predicted_prices.append(pd_.get("predicted", 0))
        conf_lowers.append(pd_.get("confidence_lower", 0))
        conf_uppers.append(pd_.get("confidence_upper", 0))
        actual_prices.append(f.get("actual_market_price", None))
        dam_prices_series.append(f.get("dam_price", None))

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=timestamps + timestamps[::-1], y=conf_uppers + conf_lowers[::-1],
        fill="toself", fillcolor="rgba(59,130,246,0.08)",
        line=dict(color="rgba(59,130,246,0)"), name="95% CI", hoverinfo="skip",
    ))

    # DAM price
    dam_t = [t for t, d in zip(timestamps, dam_prices_series) if d is not None]
    dam_v = [d for d in dam_prices_series if d is not None]
    if dam_v:
        fig.add_trace(go.Scatter(
            x=dam_t, y=dam_v, mode="lines", name="Day-Ahead (DAM)",
            line=dict(color="#f59e0b", width=1.5, dash="dash"),
        ))

    # Predicted
    fig.add_trace(go.Scatter(
        x=timestamps, y=predicted_prices, mode="lines+markers", name="ML Predicted",
        line=dict(color="#3b82f6", width=2.5), marker=dict(size=4),
    ))

    # Actual RT
    act_t = [t for t, a in zip(timestamps, actual_prices) if a is not None]
    act_v = [a for a in actual_prices if a is not None]
    if act_v:
        fig.add_trace(go.Scatter(
            x=act_t, y=act_v, mode="lines+markers", name="Actual RT",
            line=dict(color="#06b6d4", width=2), marker=dict(size=4),
        ))

    # Trade markers
    if trade_history:
        for action_type, color, symbol in [("BUY", "#10b981", "triangle-up"), ("SELL", "#ef4444", "triangle-down")]:
            t_times = [t["timestamp"] for t in trade_history if t["action"] == action_type]
            t_prices = [t["price"] for t in trade_history if t["action"] == action_type]
            if t_times:
                fig.add_trace(go.Scatter(
                    x=t_times, y=t_prices, mode="markers", name=action_type,
                    marker=dict(symbol=symbol, size=11, color=color, line=dict(width=1, color="#0a0e17")),
                ))

    fig.update_layout(**PLOTLY_LAYOUT, height=420, yaxis_title="$/MWh", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown("""
    <div class="info-card" style="text-align:center; padding:40px 20px;">
        <div style="font-size:2rem; margin-bottom:8px;">📊</div>
        <div style="font-size:0.9rem; color:#94a3b8; font-weight:500;">Price chart populates as the pipeline runs</div>
        <div style="font-size:0.78rem; color:#64748b; margin-top:4px;">Data points added every 15 minutes · Use sidebar to trigger manually</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  ROW 3 — ENERGY + DEMAND
# ═══════════════════════════════════════════════════════════

col_energy, col_demand = st.columns(2)

with col_energy:
    st.markdown('<div class="section-header"><span class="icon">☀️</span>Solar Energy Output</div>', unsafe_allow_html=True)
    if forecast_history and len(forecast_history) > 1:
        ts = [f["timestamp"] for f in forecast_history]
        vals = [f.get("energy_output", {}).get("predicted", 0) for f in forecast_history]
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=ts, y=vals, mode="lines", fill="tozeroy",
            fillcolor="rgba(245,158,11,0.1)", line=dict(color="#f59e0b", width=2), name="Predicted",
        ))
        fig_e.update_layout(**PLOTLY_LAYOUT, height=260, yaxis_title="MW")
        st.plotly_chart(fig_e, use_container_width=True)
    else:
        st.markdown('<div class="info-card" style="text-align:center; padding:30px;"><div style="color:#64748b; font-size:0.8rem;">Accumulating data...</div></div>', unsafe_allow_html=True)

with col_demand:
    st.markdown('<div class="section-header"><span class="icon">🏭</span>Grid Demand Forecast</div>', unsafe_allow_html=True)
    if forecast_history and len(forecast_history) > 1:
        ts = [f["timestamp"] for f in forecast_history]
        vals = [f.get("demand", {}).get("predicted", 0) for f in forecast_history]
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(
            x=ts, y=vals, mode="lines", fill="tozeroy",
            fillcolor="rgba(139,92,246,0.1)", line=dict(color="#8b5cf6", width=2), name="Predicted",
        ))
        fig_d.update_layout(**PLOTLY_LAYOUT, height=260, yaxis_title="MW")
        st.plotly_chart(fig_d, use_container_width=True)
    else:
        st.markdown('<div class="info-card" style="text-align:center; padding:30px;"><div style="color:#64748b; font-size:0.8rem;">Accumulating data...</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  ROW 4 — WEATHER + RAG
# ═══════════════════════════════════════════════════════════

col_weather, col_rag = st.columns(2)

with col_weather:
    st.markdown('<div class="section-header"><span class="icon">🌤️</span>Weather Conditions — Dallas, TX</div>', unsafe_allow_html=True)
    if weather:
        w1, w2, w3, w4 = st.columns(4)
        with w1: st.metric("TEMP", f"{weather.get('temp', 'N/A')}°C")
        with w2: st.metric("WIND", f"{weather.get('wind_speed', 'N/A')} m/s")
        with w3: st.metric("GHI", f"{weather.get('irradiance', 0)} W/m²")
        with w4: st.metric("CLOUDS", f"{weather.get('cloud_coverage', 'N/A')}%")

        if weather_hist and len(weather_hist) > 2:
            wh_ts = [w.get("timestamp", "") for w in weather_hist]
            wh_ghi = [w.get("irradiance", 0) for w in weather_hist]
            wh_temp = [w.get("temp", 0) for w in weather_hist]
            fig_w = go.Figure()
            fig_w.add_trace(go.Scatter(
                x=wh_ts, y=wh_ghi, mode="lines", name="GHI (W/m²)",
                fill="tozeroy", fillcolor="rgba(245,158,11,0.08)", line=dict(color="#f59e0b", width=1.5),
            ))
            fig_w.add_trace(go.Scatter(
                x=wh_ts, y=wh_temp, mode="lines", name="Temp (°C)",
                line=dict(color="#ef4444", width=1.5), yaxis="y2",
            ))
            fig_w.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(26, 31, 46, 1)",
                plot_bgcolor="rgba(26, 31, 46, 1)",
                font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=11),
                height=180,
                yaxis=dict(title="W/m²", gridcolor="rgba(42,48,64,0.5)"),
                yaxis2=dict(title="°C", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
                margin=dict(l=8, r=8, t=8, b=8),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                            font=dict(size=10, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_w, use_container_width=True)

        st.caption(f"Open-Meteo NOAA HRRR · DNI: {weather.get('dni', 0)} W/m² · Humidity: {weather.get('humidity', 0)}%")
    else:
        st.warning("Weather data unavailable")

with col_rag:
    st.markdown('<div class="section-header"><span class="icon">🧠</span>Market Intelligence (RAG)</div>', unsafe_allow_html=True)

    # Build context-rich query using live market data so Gemini can give
    # a meaningful risk score instead of a generic 0.50 default
    _rt   = rt_price or (ercot.get("price_usd_mwh") if ercot else None)
    _pred = pred_price or latest_forecast.get("price", {}).get("predicted")

    if _rt and _pred:
        rag_query = (
            f"Current ERCOT conditions: "
            f"RT price ${_rt:.2f}/MWh, "
            f"ML forecast ${_pred:.2f}/MWh, "
            f"Temperature {weather.get('temp', 'N/A')}°C, "
            f"Wind {weather.get('wind_speed', 'N/A')} m/s, "
            f"Cloud cover {weather.get('cloud_coverage', 'N/A')}%, "
            f"Solar GHI {weather.get('irradiance', 0)} W/m², "
            f"Time: {now_ct.strftime('%I:%M %p CT on %A')}. "
            f"Based on these conditions and your energy market knowledge, "
            f"what is the risk of an ERCOT HB_NORTH price spike in the next 2 hours?"
        )
    else:
        rag_query = "What factors might affect ERCOT electricity prices?"

    # Cache RAG call for 15 min — Streamlit reruns the entire script on every
    # interaction so without caching this would burn Gemini quota on every click
    @st.cache_data(ttl=900, show_spinner=False)
    def fetch_rag(query: str, rag_url: str):
        try:
            r = requests.post(f"{rag_url}/analyze", json={"query": query}, timeout=35)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None

    rag = fetch_rag(rag_query, RAG_URL)

    # Fallback to trade history if RAG service is down
    if not rag and trade_history:
        latest_t = trade_history[0]
        rag_risk = latest_t.get("rag_risk_score")
        rag_dir  = latest_t.get("rag_direction")
        if rag_risk is not None:
            rag = {
                "risk_score": rag_risk,
                "risk_level": "high" if rag_risk > 0.7 else ("medium" if rag_risk > 0.4 else "low"),
                "price_direction": rag_dir or "stable",
                "fallback": True,
            }

    if rag and not rag.get("fallback"):
        risk_score = rag.get("risk_score", 0.5)
        risk_level = rag.get("risk_level", "medium")
        direction  = rag.get("price_direction", "stable")

        risk_colors    = {"low": "#10b981", "medium": "#f59e0b", "high": "#ef4444"}
        direction_icons = {"up": "↗ RISING", "down": "↘ FALLING", "stable": "→ STABLE"}

        # ── Risk + Direction cards ────────────────────────────
        r1, r2 = st.columns(2)
        with r1:
            clr = risk_colors.get(risk_level, "#f59e0b")
            st.markdown(f"""
            <div class="info-card">
                <h4>Risk Level</h4>
                <div class="value" style="color:{clr};">{risk_level.upper()}</div>
                <div class="sub">Score: {risk_score:.2f}</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="info-card">
                <h4>Price Direction</h4>
                <div class="value">{direction_icons.get(direction, "→ STABLE")}</div>
                <div class="sub">RAG-assessed outlook</div>
            </div>""", unsafe_allow_html=True)

        # ── Key Market Signals ────────────────────────────────
        factors = rag.get("factors", [])
        if factors:
            st.markdown("""
            <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase;
                 letter-spacing:0.05em; margin:12px 0 6px 0;">
                Key Market Signals
            </div>""", unsafe_allow_html=True)
            for factor in factors[:5]:
                st.markdown(
                    f"<div style='color:#94a3b8; font-size:0.83rem; padding:5px 0;"
                    f"border-bottom:1px solid #2a3040;'>› {factor}</div>",
                    unsafe_allow_html=True
                )

        # ── Gemini Analysis Summary ───────────────────────────
        summary = rag.get("summary", "")
        if summary:
            st.markdown(f"""
            <div style="margin-top:12px; background:rgba(59,130,246,0.05);
                 border-left:3px solid #3b82f6; padding:10px 14px;
                 border-radius:0 8px 8px 0;">
                <div style="font-size:0.72rem; color:#64748b; text-transform:uppercase;
                     letter-spacing:0.04em; margin-bottom:6px;">
                    Gemini Analysis
                </div>
                <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6;">
                    {summary[:400]}
                </div>
            </div>""", unsafe_allow_html=True)

        # ── Source Documents ──────────────────────────────────
        sources = rag.get("sources", [])
        if sources:
            with st.expander(f"📰 {len(sources)} source documents"):
                for s in sources[:3]:
                    meta = s.get("metadata", {})
                    src  = meta.get("source", "Knowledge Base")
                    st.markdown(
                        f"<div style='font-size:0.72rem; color:#3b82f6; margin-bottom:2px;'>"
                        f"{src}</div>"
                        f"<div style='font-size:0.8rem; color:#94a3b8; margin-bottom:12px;'>"
                        f"{s['content'][:200]}...</div>",
                        unsafe_allow_html=True
                    )
    else:
        st.markdown("""
        <div class="info-card" style="text-align:center; padding:30px;">
            <div style="font-size:0.85rem; color:#94a3b8;">
                RAG initializing — market intelligence loads on next pipeline cycle
            </div>
        </div>""", unsafe_allow_html=True)


with tab_trading:

    # ── CUMULATIVE P&L — shown first (most important visual) ─
    st.markdown('<div class="section-header"><span class="icon">💰</span>Cumulative P&L</div>', unsafe_allow_html=True)

    if trade_history:
        trades_chrono = list(reversed(trade_history))
        cumulative, running, times = [], 0, []
        for t in trades_chrono:
            running += t.get("profit", 0)
            cumulative.append(running)
            times.append(t["timestamp"])

        fill_c = "rgba(16,185,129,0.08)" if running >= 0 else "rgba(239,68,68,0.08)"
        line_c = "#10b981" if running >= 0 else "#ef4444"

        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=times, y=cumulative, mode="lines", fill="tozeroy",
            fillcolor=fill_c, line=dict(color=line_c, width=2.5), name="Cumulative P&L",
        ))
        fig_pnl.add_hline(y=0, line_dash="dot", line_color="#64748b", line_width=1)
        fig_pnl.update_layout(**PLOTLY_LAYOUT, height=280, yaxis_title="Profit ($)")
        st.plotly_chart(fig_pnl, use_container_width=True)
    else:
        st.markdown('<div class="info-card" style="text-align:center; padding:24px;"><div style="color:#64748b; font-size:0.8rem;">P&L chart populates as trades execute</div></div>', unsafe_allow_html=True)

    # ── LATEST TRADE + ERCOT MARKET ──────────────────────────
    col_trade, col_market = st.columns(2)

    with col_trade:
        st.markdown('<div class="section-header"><span class="icon">🔄</span>Latest Trade</div>', unsafe_allow_html=True)
        if trade_history:
            latest = trade_history[0]
            action = latest["action"]
            badge = {"BUY": "badge-buy", "SELL": "badge-sell", "HOLD": "badge-hold"}.get(action, "badge-hold")

            html = f'<div class="info-card"><div class="{badge}">{action}</div>'
            if action != "HOLD":
                pnl_clr = "#10b981" if latest["profit"] >= 0 else "#ef4444"
                html += f"""
                <div style="display:flex; gap:32px; margin-top:16px;">
                    <div><div style="font-size:0.7rem; color:#64748b; text-transform:uppercase;">Price</div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:1.2rem; color:#f1f5f9;">${latest['price']:.2f}</div></div>
                    <div><div style="font-size:0.7rem; color:#64748b; text-transform:uppercase;">Quantity</div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:1.2rem; color:#f1f5f9;">{latest['quantity']} MWh</div></div>
                    <div><div style="font-size:0.7rem; color:#64748b; text-transform:uppercase;">P&L</div>
                        <div style="font-family:'JetBrains Mono',monospace; font-size:1.2rem; color:{pnl_clr};">${latest['profit']:+,.2f}</div></div>
                </div>"""

            reason = latest.get("reason", "N/A")
            reason_parts = [p.strip() for p in reason.split("|")]
            reason_html  = "<br/>".join(f"› {p}" for p in reason_parts[:6])
            html += f"""
            <div style="margin-top:12px;">
                <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase;
                     letter-spacing:0.04em; margin-bottom:6px;">Signal Reasoning</div>
                <div style="font-size:0.78rem; color:#94a3b8; line-height:1.8;">
                    {reason_html}
                </div>
            </div>
            <div style="margin-top:8px; font-size:0.75rem; color:#64748b;">
                Position: {latest.get('position_after', 0)} MWh · {latest.get('timestamp', '')}</div></div>"""
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card" style="text-align:center; padding:30px;">
                <div style="font-size:0.85rem; color:#94a3b8;">First trade executes on next pipeline cycle</div>
            </div>""", unsafe_allow_html=True)

    with col_market:
        st.markdown('<div class="section-header"><span class="icon">💲</span>ERCOT Market Overview</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            rt = ercot.get("price_usd_mwh") if ercot else None
            rt_display = f"${rt:.2f}" if rt and rt > 0 else "—"
            rt_color = "#06b6d4" if rt and rt > 0 else "#64748b"
            st.markdown(f"""
            <div class="info-card">
                <h4>Real-Time Price</h4>
                <div class="value" style="color:{rt_color};">{rt_display}</div>
                <div class="sub">HB_NORTH · 15-min intervals</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            if spread and spread.get("dam_price") is not None:
                dp = spread["dam_price"]
                sv = spread.get("spread", 0)
                sc = "spread-positive" if sv > 0 else ("spread-negative" if sv < 0 else "spread-neutral")
                st.markdown(f"""
                <div class="info-card">
                    <h4>Day-Ahead Price</h4>
                    <div class="value" style="color:#f59e0b;">${dp:.2f}</div>
                    <div class="sub {sc}">Spread: ${sv:+.2f}/MWh</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-card">
                    <h4>Day-Ahead Price</h4>
                    <div class="value">N/A</div>
                    <div class="sub">Fetch via sidebar controls</div>
                </div>""", unsafe_allow_html=True)

        if dam_today and dam_today.get("hours"):
            hours = dam_today["hours"]
            he = [f"HE{h.get('hour_ending', '')}" for h in hours]
            dv = [h.get("dam_price_usd_mwh", 0) for h in hours]
            fig_dam2 = go.Figure()
            fig_dam2.add_trace(go.Bar(
                x=he, y=dv, marker_color=["#f59e0b" if v > 50 else "#2563eb" for v in dv],
                marker_line=dict(width=0), name="DAM Price",
            ))
            fig_dam2.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(26, 31, 46, 1)",
                plot_bgcolor="rgba(26, 31, 46, 1)",
                font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=11),
                height=170, yaxis_title="$/MWh",
                margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
                xaxis=dict(gridcolor="rgba(42,48,64,0.5)"),
                yaxis=dict(gridcolor="rgba(42,48,64,0.5)"),
            )
            st.plotly_chart(fig_dam2, use_container_width=True)

    # ── TRADE HISTORY with color-coded action badges ──────────
    st.markdown('<div class="section-header"><span class="icon">📋</span>Trade History</div>', unsafe_allow_html=True)

    if trade_history:
        df = pd.DataFrame(trade_history)
        cols = ["timestamp", "action", "price", "quantity", "profit", "position_after", "rag_risk_score"]
        available = [c for c in cols if c in df.columns]
        df_d = df[available].copy()
        rename = {"timestamp": "Time", "action": "Action", "price": "Price ($/MWh)",
                  "quantity": "Qty (MWh)", "profit": "P&L ($)", "position_after": "Position",
                  "rag_risk_score": "RAG Risk"}
        df_d.rename(columns={k: v for k, v in rename.items() if k in df_d.columns}, inplace=True)
        if "Price ($/MWh)" in df_d.columns:
            df_d["Price ($/MWh)"] = df_d["Price ($/MWh)"].apply(lambda x: f"${x:.2f}" if pd.notna(x) and x != 0 else "—")
        if "P&L ($)" in df_d.columns:
            df_d["P&L ($)"] = df_d["P&L ($)"].apply(lambda x: f"${x:+,.2f}" if pd.notna(x) else "—")
        # Color code Action column
        if "Action" in df_d.columns:
            df_d["Action"] = df_d["Action"].apply(
                lambda x: f"🟢 {x}" if x == "BUY" else (f"🔴 {x}" if x == "SELL" else f"🟡 {x}")
            )
        st.dataframe(df_d, use_container_width=True, height=320, hide_index=True)
    else:
        st.markdown('<div class="info-card" style="text-align:center; padding:24px;"><div style="color:#64748b; font-size:0.8rem;">Trade history populates as the automated pipeline executes. Click ▶ Run Full Pipeline in the sidebar to start.</div></div>', unsafe_allow_html=True)

    # ── BACKTESTING ───────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="icon">📊</span>Backtesting — ML Strategy vs Baselines</div>', unsafe_allow_html=True)

    @st.cache_data(ttl=900)
    def load_backtest_history():
        return safe_get(f"{FORECAST_URL}/forecast/history?limit=2880", [])

    bt_history = load_backtest_history()
    bt = run_backtest(bt_history or forecast_history)

if bt:
    s = bt["stats"]

    # ── KPI Row ──────────────────────────────────────────────
    bk1, bk2, bk3, bk4, bk5, bk6 = st.columns(6)

    with bk1:
        st.metric("BACKTEST PERIOD", s["period"], delta=f"{s['n_intervals']} intervals")
    with bk2:
        st.metric("TOTAL TRADES", str(s["total_trades"]), delta="ML strategy")
    with bk3:
        wr_delta = "above random" if s["win_rate"] > 50 else "below 50%"
        st.metric("WIN RATE", f"{s['win_rate']}%", delta=wr_delta,
                  delta_color="normal" if s["win_rate"] > 50 else "inverse")
    with bk4:
        if s["sharpe"] is not None:
            sh_delta = "good" if s["sharpe"] >= 1.0 else ("neutral" if s["sharpe"] >= 0 else "negative")
            st.metric("SHARPE RATIO", f"{s['sharpe']}", delta=sh_delta,
                      delta_color="normal" if s["sharpe"] >= 1.0 else "off")
        else:
            st.metric("SHARPE RATIO", "—", delta=f"needs 96+ intervals · have {s['n_intervals']}")
    with bk5:
        st.metric("MAX DRAWDOWN", f"${s['max_drawdown']:,.0f}",
                  delta="peak-to-trough", delta_color="off")
    with bk6:
        vs_bh = s["total_pnl"] - s["bh_pnl"]
        st.metric("ML vs BUY-HOLD", f"${vs_bh:+,.0f}",
                  delta="alpha generated", delta_color="normal" if vs_bh >= 0 else "inverse")

    st.markdown("")

    # ── Equity Curves ────────────────────────────────────────
    col_chart, col_summary = st.columns([3, 1])

    with col_chart:
        fig_bt = go.Figure()

        fig_bt.add_trace(go.Scatter(
            x=bt["times"], y=bt["ml_equity"],
            mode="lines", name="ML Strategy",
            line=dict(color="#3b82f6", width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.06)",
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt["times"], y=bt["bh_equity"],
            mode="lines", name="Buy & Hold",
            line=dict(color="#f59e0b", width=1.8, dash="dash"),
        ))
        fig_bt.add_trace(go.Scatter(
            x=bt["times"], y=bt["nr_equity"],
            mode="lines", name="Naive Mean-Reversion",
            line=dict(color="#8b5cf6", width=1.5, dash="dot"),
        ))
        fig_bt.add_hline(y=0, line_dash="dot", line_color="#64748b", line_width=1)

        # Mark trade entries/exits
        if bt["trades_log"]:
            sell_times = [t["time"] for t in bt["trades_log"]]
            sell_pnls  = []
            running    = 0.0
            pnl_map    = {t["time"]: t["pnl"] for t in bt["trades_log"]}
            for ts in bt["times"]:
                if ts in pnl_map:
                    running += pnl_map[ts]
                sell_pnls.append(running if ts in pnl_map else None)

            win_times  = [t["time"] for t in bt["trades_log"] if t["win"]]
            win_vals   = []
            running2   = 0.0
            for t in bt["trades_log"]:
                running2 += t["pnl"]
                if t["win"]:
                    win_vals.append(running2)

            loss_times = [t["time"] for t in bt["trades_log"] if not t["win"]]
            loss_vals  = []
            running3   = 0.0
            for t in bt["trades_log"]:
                running3 += t["pnl"]
                if not t["win"]:
                    loss_vals.append(running3)

            if win_times:
                fig_bt.add_trace(go.Scatter(
                    x=win_times, y=win_vals, mode="markers", name="Winning Trade",
                    marker=dict(symbol="triangle-up", size=10, color="#10b981",
                                line=dict(width=1, color="#0a0e17")),
                ))
            if loss_times:
                fig_bt.add_trace(go.Scatter(
                    x=loss_times, y=loss_vals, mode="markers", name="Losing Trade",
                    marker=dict(symbol="triangle-down", size=10, color="#ef4444",
                                line=dict(width=1, color="#0a0e17")),
                ))

        fig_bt.update_layout(
            **PLOTLY_LAYOUT, height=380,
            yaxis_title="Cumulative P&L ($)",
            title=dict(text="Cumulative Returns: ML Strategy vs Baselines",
                       font=dict(size=13, color="#94a3b8"), x=0),
            hovermode="x unified",
        )
        st.plotly_chart(fig_bt, use_container_width=True)

    with col_summary:
        # Strategy comparison table
        ml_pnl = s["total_pnl"]
        bh_pnl = s["bh_pnl"]
        nr_pnl = s["nr_pnl"]
        best   = max(ml_pnl, bh_pnl, nr_pnl)

        def pnl_color(v):
            return "#10b981" if v >= 0 else "#ef4444"

        def crown(v):
            return " 👑" if v == best else ""

        sharpe_color  = "#10b981" if s["sharpe"] and s["sharpe"] >= 1 else "#f59e0b"
        sharpe_display = s["sharpe"] if s["sharpe"] is not None else "— (need 96+ intervals)"

        st.markdown(f"""
        <div class="info-card" style="margin-top:0;">
            <h4>Strategy Comparison</h4>
            <div style="font-size:0.75rem; color:#64748b; margin-bottom:12px;">
                Period: {s['period']}
            </div>

            <div style="margin-bottom:14px;">
                <div style="font-size:0.72rem; color:#3b82f6; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;">
                    ML Strategy{crown(ml_pnl)}
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; color:{pnl_color(ml_pnl)};">
                    ${ml_pnl:+,.0f}
                </div>
                <div style="font-size:0.72rem; color:#64748b;">
                    {s['total_trades']} trades · {s['win_rate']}% win
                </div>
            </div>

            <div style="margin-bottom:14px;">
                <div style="font-size:0.72rem; color:#f59e0b; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;">
                    Buy &amp; Hold{crown(bh_pnl)}
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; color:{pnl_color(bh_pnl)};">
                    ${bh_pnl:+,.0f}
                </div>
                <div style="font-size:0.72rem; color:#64748b;">passive · no rebalancing</div>
            </div>

            <div style="margin-bottom:14px;">
                <div style="font-size:0.72rem; color:#8b5cf6; font-weight:600; text-transform:uppercase; letter-spacing:0.04em;">
                    Naive Mean-Rev{crown(nr_pnl)}
                </div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; color:{pnl_color(nr_pnl)};">
                    ${nr_pnl:+,.0f}
                </div>
                <div style="font-size:0.72rem; color:#64748b;">no ML · rule-based</div>
            </div>

            <div style="border-top:1px solid #2a3040; padding-top:12px; margin-top:4px;">
                <div style="font-size:0.72rem; color:#64748b;">Sharpe Ratio</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; color:{sharpe_color};">
                    {sharpe_display}
                </div>
                <div style="font-size:0.72rem; color:#64748b; margin-top:6px;">Max Drawdown</div>
                <div style="font-family:'JetBrains Mono',monospace; font-size:1rem; color:#ef4444;">
                    ${s['max_drawdown']:,.0f}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Per-Trade Log ─────────────────────────────────────────
    if bt["trades_log"]:
        with st.expander(f"📋 Trade Log — {len(bt['trades_log'])} backtest trades"):
            df_log = pd.DataFrame(bt["trades_log"])
            df_log["pnl"]    = df_log["pnl"].apply(lambda x: f"${x:+,.2f}")
            df_log["win"]    = df_log["win"].apply(lambda x: "✅ Win" if x else "❌ Loss")
            df_log.columns   = ["Time", "Buy Price", "Sell Price", "Signal Entry", "Signal Exit", "P&L", "Result"]
            st.dataframe(df_log, use_container_width=True, hide_index=True, height=260)

    st.caption(
        "Backtest runs ML signal logic on historical forecast data. "
        "Signal = ML Predicted − Actual Market Price. "
        "Dynamic threshold = 35th percentile of recent |signals| (min $1.50). "
        "Trade size: 50 MWh. No transaction costs applied."
    )

else:
    st.markdown("""
    <div class="info-card" style="text-align:center; padding:32px;">
        <div style="font-size:1.5rem; margin-bottom:8px;">📊</div>
        <div style="font-size:0.9rem; color:#94a3b8;">
            Backtest available after the pipeline collects at least 20 forecast intervals
        </div>
        <div style="font-size:0.78rem; color:#64748b; margin-top:6px;">
            Each 15-minute pipeline cycle adds one interval · Run pipeline from the sidebar to populate
        </div>
    </div>
    """, unsafe_allow_html=True)


with tab_model:

    # ── MODEL PERFORMANCE with train/test period ──────────────
    st.markdown('<div class="section-header"><span class="icon">🧪</span>Model Performance</div>', unsafe_allow_html=True)

    if metrics:
        # Show train/test split info
        # train_samples/val_samples only available after retraining via API
        # Fall back to showing training CSV row count
        price_m = metrics.get("price", {})
        train_samples = price_m.get("train_samples")
        val_samples   = price_m.get("val_samples")
        if train_samples and val_samples:
            split_info = f"Train: {train_samples:,} samples · Val: {val_samples:,} samples"
        else:
            split_info = "80% of ERCOT + weather data for training · 20% held out for validation"
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#64748b; margin-bottom:12px; padding:8px 12px;
             background:rgba(59,130,246,0.05); border-radius:8px; border-left:3px solid #3b82f6;">
            <strong style="color:#94a3b8;">Validation methodology:</strong>
            Chronological 80/20 split — trained on first 80% of data, validated on last 20%.
            No data leakage: model never saw validation prices during training.
            {split_info}
        </div>""", unsafe_allow_html=True)

        col_e, col_p, col_d = st.columns(3)
        for col, key, label, color in [
            (col_e, "energy_output", "Energy Output", "#f59e0b"),
            (col_p, "price", "Price Forecast", "#3b82f6"),
            (col_d, "demand", "Grid Demand", "#8b5cf6"),
        ]:
            with col:
                m = metrics.get(key, {})
                improvement = m.get('improvement_over_naive_pct', 0)
                imp_color = "#10b981" if improvement and float(str(improvement).replace('%','') or 0) > 20 else "#f59e0b"
                st.markdown(f"""
                <div class="info-card">
                    <h4 style="color:{color};">{label}</h4>
                    <div class="value">{m.get('ensemble_mape', 'N/A')}%</div>
                    <div class="sub">Ensemble MAPE · lower is better</div>
                    <div style="margin-top:12px; font-size:0.75rem; color:#64748b;">
                        RF: {m.get('rf_mape', 'N/A')}% · XGB: {m.get('xgb_mape', 'N/A')}%<br/>
                        Naive baseline: {m.get('naive_mape', 'N/A')}%
                    </div>
                    <div style="margin-top:8px; font-size:0.78rem; color:{imp_color}; font-weight:600;">
                        ↑ {m.get('improvement_over_naive_pct', 'N/A')}% better than naive
                    </div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("Model metrics not available — run training first.")

    # ── FEATURE IMPORTANCE with interpretive context ──────────
    st.markdown('<div class="section-header"><span class="icon">🔍</span>Feature Importance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#64748b; margin-bottom:12px; padding:8px 12px;
         background:rgba(139,92,246,0.05); border-radius:8px; border-left:3px solid #8b5cf6;">
        <strong style="color:#94a3b8;">How to read this:</strong>
        Higher % = the model relies more on that feature to make predictions.
        Price model dominated by recent price lags means it uses momentum patterns.
        Energy model dominated by wind speed means weather is the key driver.
    </div>""", unsafe_allow_html=True)

    if importance:
        # Clean feature name mapping
        label_map = {
        "price_lag_1h": "Price (1h ago)", "price_lag_2h": "Price (2h ago)",
        "price_lag_3h": "Price (3h ago)", "price_lag_24h": "Price (24h ago)",
        "price_lag_168h": "Price (1 week ago)", "price_diff_1h": "Price Δ (1h)",
        "price_diff_24h": "Price Δ (24h)", "price_rolling_6h": "Avg Price (6h)",
        "price_rolling_24h": "Avg Price (24h)", "demand_lag_1h": "Demand (1h ago)",
        "demand_lag_24h": "Demand (24h ago)", "demand_rolling_6h": "Avg Demand (6h)",
        "irradiance": "Solar GHI", "direct_radiation": "Direct Radiation",
        "dni": "DNI", "temp": "Temperature", "humidity": "Humidity",
        "wind_speed": "Wind Speed", "cloud_coverage": "Cloud Cover",
        "hour": "Hour of Day", "day_of_week": "Day of Week",
        "is_weekend": "Weekend", "is_peak_hour": "Peak Hour",
    }

    def render_importance(imp_data, base_rgb, top_n=8):
        """Render a single model's feature importance chart."""
        if not imp_data:
            st.info("No feature importance data available")
            return

        features = list(imp_data.keys())
        rf_v = [imp_data[f].get("rf", 0) for f in features]
        xgb_v = [imp_data[f].get("xgb", 0) for f in features]
        ensemble_v = [(r + x) / 2 for r, x in zip(rf_v, xgb_v)]

        srt = sorted(zip(features, ensemble_v), key=lambda x: x[1], reverse=True)[:top_n]
        total_imp = sum(s[1] for s in srt) or 1
        top_pct = [(label_map.get(s[0], s[0]), round(s[1] / total_imp * 100, 1)) for s in srt]

        labels = [s[0] for s in top_pct][::-1]
        values = [s[1] for s in top_pct][::-1]

        # base_rgb is like "245, 158, 11" — build rgba colors with fading opacity
        colors = [f"rgba({base_rgb}, {max(0.35, 1.0 - i * 0.08)})" for i in range(len(top_pct))][::-1]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=labels, x=values, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v}%" for v in values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono, monospace", size=11, color="#94a3b8"),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
            showlegend=False,
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(26, 31, 46, 1)",
            plot_bgcolor="rgba(26, 31, 46, 1)",
            font=dict(family="DM Sans, sans-serif", color="#94a3b8", size=12),
            margin=dict(l=16, r=16, t=8, b=16),
            height=280,
            xaxis=dict(title="Relative Importance (%)", range=[0, max(values) * 1.3],
                       gridcolor="rgba(42,48,64,0.3)", zerolinecolor="rgba(42,48,64,0.5)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11)),
            bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)

    fi1, fi2, fi3 = st.columns(3)

    with fi1:
        st.markdown("""
        <div style="font-size:0.85rem; font-weight:600; color:#f59e0b; margin-bottom:4px;">
            ☀️ Energy Output Model
        </div>
        <div style="font-size:0.72rem; color:#64748b; margin-bottom:8px;">
            7 weather features · No lag data
        </div>
        """, unsafe_allow_html=True)
        render_importance(importance.get("energy_output", {}), "245, 158, 11", top_n=7)

    with fi2:
        st.markdown("""
        <div style="font-size:0.85rem; font-weight:600; color:#3b82f6; margin-bottom:4px;">
            💲 Price Forecast Model
        </div>
        <div style="font-size:0.72rem; color:#64748b; margin-bottom:8px;">
            23 features · Weather + lags + time
        </div>
        """, unsafe_allow_html=True)
        render_importance(importance.get("price", {}), "59, 130, 246", top_n=8)

    with fi3:
        st.markdown("""
        <div style="font-size:0.85rem; font-weight:600; color:#8b5cf6; margin-bottom:4px;">
            🏭 Grid Demand Model
        </div>
        <div style="font-size:0.72rem; color:#64748b; margin-bottom:8px;">
            14 features · Weather + demand lags + time
        </div>
        """, unsafe_allow_html=True)
        render_importance(importance.get("demand", {}), "139, 92, 246", top_n=8)

    st.caption(
        "Each model uses different feature sets optimized for its prediction target · "
        "Energy model relies on weather, price model on recent prices, demand model on temporal patterns"
    )


# ═══════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════

st.markdown("""
<div class="footer">
    <strong>AI Renewable Energy Trading Platform</strong><br/>
    Data: Open-Meteo (NOAA HRRR) · ERCOT API (RT-SPP + DAM) · Models: RandomForest + XGBoost Ensemble · Intelligence: LangChain RAG + Gemini<br/>
    Microservices: FastAPI · Docker Compose · SQLite · Streamlit
</div>
""", unsafe_allow_html=True)