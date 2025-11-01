import os
import requests
import streamlit as st
import pandas as pd
from glob import glob
from datetime import datetime
import pytz

st.set_page_config(page_title="AI Energy Trading", layout="wide")

def get_forecast_files(folder="data/processed"):
    return sorted(glob(os.path.join(folder, "forecast_*.csv")), reverse=True)

def trigger_forecast_update():
    try:
        response = requests.post("http://forecast:8002/forecast/batch")
        if response.status_code == 200:
            return "✅ Forecast updated!", response.json()
        else:
            return f"❌ Forecast update failed: {response.status_code}", response.text
    except Exception as e:
        return "❌ Forecast service error", str(e)

def call_trading_service(output, price, interval_minutes=5):
    try:
        response = requests.post("http://trading:8003/trade", json={
            "predicted_output": output,
            "predicted_price": price,
            "interval_minutes": interval_minutes
        })
        if response.status_code == 200:
            result = response.json()
            return (
                result["decision"],
                result["reason"],
                result["profit"],
                result["quantity_mwh"]
            )
        else:
            return "Hold", "Trading service error", 0.0, 0.0
    except Exception as e:
        return "Hold", f"Error: {e}", 0.0, 0.0

# === Sidebar ===
st.sidebar.header("⚙️ Controls")
forecast_files = get_forecast_files()
selected_file = st.sidebar.selectbox("📂 Select forecast file", forecast_files)

if st.sidebar.button("🔁 Refresh Forecast"):
    with st.spinner("Calling forecast microservice..."):
        status, result = trigger_forecast_update()
    st.sidebar.success(status)
    st.sidebar.code(result if isinstance(result, str) else str(result))

# === Main Dashboard ===
if selected_file:
    df = pd.read_csv(selected_file)
    st.success(f"Loaded file: `{os.path.basename(selected_file)}`")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
    df.set_index("timestamp", inplace=True)

    # Get trading decisions
    decisions = df.apply(
        lambda row: call_trading_service(row["predicted_output"], row["predicted_price"]),
        axis=1,
        result_type="expand"
    )
    decisions.columns = ["Decision", "Reason", "Profit ($)", "Quantity (MWh)"]
    df = pd.concat([df, decisions], axis=1)
    df["Cumulative Profit ($)"] = df["Profit ($)"].cumsum()

    # Header
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h1 style="margin: 0;">🔋 AI Renewable Energy Trading Dashboard</h1>
            <h1 style="margin: 0; font-size: 1.5rem; color: gray;">
                🕒 {datetime.now(pytz.timezone("America/New_York")).strftime('%Y-%m-%d %H:%M:%S')}
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Forecast Plot ---
    st.subheader("📈 Forecasted Power Output (MW)")
    st.line_chart(df["predicted_output"])

    # --- Market Price ---
    st.subheader("📉 Market Price (USD/MWh)")
    st.line_chart(df["predicted_price"])

    # --- Table with Details ---
    st.subheader("📋 Forecast & Trading Decisions")
    df_display = df.reset_index().rename(columns={
        "timestamp": "Timestamp",
        "predicted_output": "Predicted Output (MW)",
        "predicted_price": "Market Price (USD/MWh)",
        "grid_demand": "Grid Demand (MW)",
        "Decision": "Trading Action",
        "Reason": "Decision Reason",
        "Profit ($)": "Profit ($)",
        "Cumulative Profit ($)": "Cumulative Profit ($)",
        "Quantity (MWh)": "Quantity (MWh)"
    })

    def highlight_action(val):
        if val == "Buy":
            return "background-color: #d1f7d6"
        elif val == "Sell":
            return "background-color: #fddcdc"
        return ""

    styled_table = df_display[[
        "Timestamp", "Predicted Output (MW)", "Market Price (USD/MWh)",
        "Grid Demand (MW)" if "Grid Demand (MW)" in df_display else None,
        "Quantity (MWh)", "Trading Action", "Decision Reason",
        "Profit ($)", "Cumulative Profit ($)"
    ]].dropna(axis=1, how="all").style.applymap(highlight_action, subset=["Trading Action"])

    st.dataframe(styled_table, use_container_width=True)

    # --- Cumulative Profit ---
    st.subheader("💸 Cumulative Profit Simulation")
    st.line_chart(df["Cumulative Profit ($)"])

else:
    st.warning("No forecast files found. Click 'Refresh Forecast' to create one.")
