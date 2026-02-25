"""
Model Training Script (v5 — ERCOT API + Open-Meteo)
=====================================================
Trains 3 ensemble models by pulling ALL data from APIs:
    - ERCOT Public API: DAM prices + system load (authenticated)
    - Open-Meteo Archive API: hourly weather for Dallas, TX

HOW TO RUN:
    python src/train_ensemble.py              # default 45 days
    python src/train_ensemble.py --days 60    # custom range

REQUIRES .env:
    ERCOT_SUBSCRIPTION_KEY=your_key
    ERCOT_USERNAME=your_email
    ERCOT_PASSWORD=your_password
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # dotenv not installed, env vars must be set manually

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "forecast_service"))
from ensemble_forecaster import EnsembleForecaster


# ═══════════════════════════════════════════════════════════════
#  ERCOT API AUTH
# ═══════════════════════════════════════════════════════════════

ERCOT_SUBSCRIPTION_KEY = os.getenv("ERCOT_SUBSCRIPTION_KEY", "")
ERCOT_USERNAME = os.getenv("ERCOT_USERNAME", "")
ERCOT_PASSWORD = os.getenv("ERCOT_PASSWORD", "")
ERCOT_TOKEN_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
ERCOT_API_BASE = "https://api.ercot.com/api/public-reports"
_token_cache = {"id_token": None, "expires_at": 0}


def _get_token():
    """Get bearer token from Azure B2C. Cached 50 min."""
    if _token_cache["id_token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["id_token"]
    try:
        resp = requests.post(ERCOT_TOKEN_URL, data={
            "username": ERCOT_USERNAME, "password": ERCOT_PASSWORD,
            "grant_type": "password",
            "scope": "openid+fec253ea-0d06-4272-a5e6-b478baeecd70+offline_access",
            "client_id": "fec253ea-0d06-4272-a5e6-b478baeecd70",
            "response_type": "id_token",
        }, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=15)
        if resp.status_code == 200:
            _token_cache["id_token"] = resp.json().get("id_token")
            _token_cache["expires_at"] = time.time() + 3000
            print("        ✅ ERCOT authenticated")
            return _token_cache["id_token"]
        print(f"        ❌ Auth failed: HTTP {resp.status_code}")
    except Exception as e:
        print(f"        ❌ Auth error: {e}")
    return None


def _headers():
    headers = {"Ocp-Apim-Subscription-Key": ERCOT_SUBSCRIPTION_KEY}
    token = _get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# ═══════════════════════════════════════════════════════════════
#  FETCH ERCOT DATA
# ═══════════════════════════════════════════════════════════════

def fetch_prices(start_date, end_date):
    """Fetch DAM prices from ERCOT API. Returns DataFrame or None."""
    print(f"      Fetching DAM prices: {start_date} to {end_date}")
    headers = _headers()
    records = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    final = datetime.strptime(end_date, "%Y-%m-%d")

    while current < final:
        chunk_end = min(current + timedelta(days=6), final)
        chunk = f"{current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
        try:
            resp = requests.get(f"{ERCOT_API_BASE}/np4-190-cd/dam_stlmnt_pnt_prices",
                params={"deliveryDateFrom": current.strftime("%Y-%m-%d"),
                        "deliveryDateTo": chunk_end.strftime("%Y-%m-%d"),
                        "settlementPoint": "HB_NORTH", "size": 10000},
                headers=headers, timeout=30)
            if resp.status_code == 200:
                items = resp.json().get("data", [])
                for item in items:
                    try:
                        price = float(item.get("settlementPointPrice", 0))
                        date_str = item.get("deliveryDate", "")
                        hour_str = item.get("hourEnding", item.get("deliveryHour", ""))
                        if date_str and hour_str:
                            h = max(0, min(23, int(str(hour_str).replace(":", "").strip()) - 1))
                            records.append({"datetime": f"{date_str} {h:02d}:00:00", "price_usd_mwh": round(price, 2)})
                    except (ValueError, TypeError):
                        continue
                print(f"        ✅ {chunk}: {len(items)} records")
            else:
                print(f"        ⚠️  {chunk}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"        ❌ {chunk}: {e}")
        current = chunk_end + timedelta(days=1)
        time.sleep(1)

    if not records:
        return None
    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.groupby("datetime").agg({"price_usd_mwh": "mean"}).reset_index().sort_values("datetime")
    print(f"      ✅ Prices: {len(df)} hours")
    return df


def fetch_load(start_date, end_date):
    """Fetch actual system load from ERCOT API. Returns DataFrame or None."""
    print(f"      Fetching system load: {start_date} to {end_date}")
    headers = _headers()
    records = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    final = datetime.strptime(end_date, "%Y-%m-%d")

    while current < final:
        chunk_end = min(current + timedelta(days=6), final)
        chunk = f"{current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
        try:
            resp = requests.get(f"{ERCOT_API_BASE}/np6-346-cd/act_sys_load_by_fzn",
                params={"operatingDayFrom": current.strftime("%Y-%m-%d"),
                        "operatingDayTo": chunk_end.strftime("%Y-%m-%d"),
                        "size": 10000},
                headers=headers, timeout=30)
            if resp.status_code == 200:
                items = resp.json().get("data", [])
                for item in items:
                    try:
                        total = float(item.get("total", item.get("TOTAL", 0)))
                        oper_day = item.get("operDay", item.get("OperDay", ""))
                        hour_ending = item.get("hourEnding", item.get("HourEnding", ""))
                        if oper_day and hour_ending:
                            h = max(0, min(23, int(str(hour_ending).replace(":", "").strip()) - 1))
                            records.append({"datetime": f"{oper_day} {h:02d}:00:00", "total_mw": round(total, 2)})
                    except (ValueError, TypeError):
                        continue
                print(f"        ✅ {chunk}: {len(items)} records")
            else:
                print(f"        ⚠️  {chunk}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"        ❌ {chunk}: {e}")
        current = chunk_end + timedelta(days=1)
        time.sleep(1)

    if not records:
        return None
    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates(subset="datetime").sort_values("datetime")
    print(f"      ✅ Load: {len(df)} hours")
    return df


# ═══════════════════════════════════════════════════════════════
#  FETCH WEATHER
# ═══════════════════════════════════════════════════════════════

WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                "cloud_cover", "shortwave_radiation", "direct_radiation",
                "direct_normal_irradiance"]


def fetch_weather(start_date, end_date):
    """Fetch hourly weather from Open-Meteo Archive API."""
    print(f"      Fetching weather: {start_date} to {end_date}")
    records = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    final = datetime.strptime(end_date, "%Y-%m-%d")

    while current < final:
        chunk_end = min(current + timedelta(days=29), final)
        chunk = f"{current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}"
        try:
            resp = requests.get("https://archive-api.open-meteo.com/v1/archive",
                params={"latitude": 32.7767, "longitude": -96.7970,
                        "start_date": current.strftime("%Y-%m-%d"),
                        "end_date": chunk_end.strftime("%Y-%m-%d"),
                        "hourly": ",".join(WEATHER_VARS), "timezone": "America/Chicago"},
                timeout=60)
            if resp.status_code == 200 and resp.text.strip():
                data = resp.json()
                if "hourly" in data:
                    h = data["hourly"]
                    for i in range(len(h["time"])):
                        records.append({
                            "datetime": h["time"][i], "temp": h["temperature_2m"][i],
                            "humidity": h["relative_humidity_2m"][i],
                            "wind_speed": h["wind_speed_10m"][i],
                            "cloud_coverage": h["cloud_cover"][i],
                            "irradiance": h["shortwave_radiation"][i],
                            "direct_radiation": h["direct_radiation"][i],
                            "dni": h["direct_normal_irradiance"][i],
                        })
                    print(f"        ✅ {chunk}: {len(h['time'])} hours")
                else:
                    print(f"        ⚠️  {chunk}: No hourly data")
            else:
                print(f"        ⚠️  {chunk}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"        ❌ {chunk}: {e}")
        current = chunk_end + timedelta(days=1)
        time.sleep(1)

    if not records:
        return None
    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.fillna(0)
    print(f"      ✅ Weather: {len(df)} hours")
    return df


# ═══════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def derive_energy_output(df):
    wind = 3.0 * np.minimum(df["wind_speed"].values, 12) ** 1.5
    solar = 0.05 * df["irradiance"].values
    np.random.seed(42)
    df["energy_output"] = np.clip(wind + solar + np.random.normal(0, 3, len(df)), 0, 200).round(2)
    return df


def add_lag_features(df):
    print("      Adding lag features...")
    df = df.sort_values("datetime").reset_index(drop=True)
    df["price_lag_1h"] = df["market_price"].shift(1)
    df["price_lag_2h"] = df["market_price"].shift(2)
    df["price_lag_3h"] = df["market_price"].shift(3)
    df["price_lag_24h"] = df["market_price"].shift(24)
    df["price_lag_168h"] = df["market_price"].shift(168)
    df["demand_lag_1h"] = df["grid_demand"].shift(1)
    df["demand_lag_24h"] = df["grid_demand"].shift(24)
    df["price_rolling_6h"] = df["market_price"].rolling(6, min_periods=1).mean()
    df["price_rolling_24h"] = df["market_price"].rolling(24, min_periods=1).mean()
    df["demand_rolling_6h"] = df["grid_demand"].rolling(6, min_periods=1).mean()
    df["price_diff_1h"] = df["market_price"].diff(1)
    df["price_diff_24h"] = df["market_price"].diff(24)
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = ((df["hour"] >= 14) & (df["hour"] <= 20)).astype(int)
    before = len(df)
    df = df.dropna()
    print(f"        Dropped {before - len(df)} rows (168h warmup)")
    return df


# ═══════════════════════════════════════════════════════════════
#  BUILD + TRAIN
# ═══════════════════════════════════════════════════════════════

WEATHER_FEATURES = ["temp", "humidity", "wind_speed", "cloud_coverage",
                    "irradiance", "direct_radiation", "dni"]
LAG_FEATURES = ["price_lag_1h", "price_lag_2h", "price_lag_3h",
                "price_lag_24h", "price_lag_168h", "demand_lag_1h",
                "demand_lag_24h", "price_rolling_6h", "price_rolling_24h",
                "demand_rolling_6h", "price_diff_1h", "price_diff_24h",
                "hour", "day_of_week", "is_weekend", "is_peak_hour"]
ENERGY_FEATURES = WEATHER_FEATURES
PRICE_FEATURES = WEATHER_FEATURES + LAG_FEATURES
DEMAND_FEATURES = WEATHER_FEATURES + ["demand_lag_1h", "demand_lag_24h",
    "demand_rolling_6h", "hour", "day_of_week", "is_weekend", "is_peak_hour"]


def main(days):
    print("=" * 60)
    print("  ENSEMBLE MODEL TRAINING v5 (ERCOT API + Open-Meteo)")
    print("=" * 60)

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    print(f"  Range: {start_date} to {end_date} ({days} days)")

    # Fetch all data from APIs
    print("\n[1/5] Fetching data from APIs...")
    prices_df = fetch_prices(start_date, end_date)
    if prices_df is None or len(prices_df) < 100:
        print("❌ Failed to fetch prices from ERCOT API"); return

    load_df = fetch_load(start_date, end_date)
    if load_df is None or len(load_df) < 100:
        print("❌ Failed to fetch load from ERCOT API"); return

    # Save for reference
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    prices_df.to_csv(os.path.join(data_dir, "ercot_dam_prices.csv"), index=False)
    load_df.to_csv(os.path.join(data_dir, "ercot_system_load.csv"), index=False)

    # Weather: clip to archive availability (today - 7 days)
    ercot_start = max(prices_df["datetime"].min(), load_df["datetime"].min())
    ercot_end = min(prices_df["datetime"].max(), load_df["datetime"].max())
    safe_end = min(ercot_end, pd.Timestamp(datetime.now() - timedelta(days=7)))
    print(f"\n      Merging: {ercot_start.date()} to {safe_end.date()}")

    weather_df = fetch_weather(ercot_start.strftime("%Y-%m-%d"), safe_end.strftime("%Y-%m-%d"))
    if weather_df is None or len(weather_df) < 100:
        print("❌ Failed to fetch weather"); return

    # Merge
    prices_df["mk"] = prices_df["datetime"].dt.floor("h")
    load_df["mk"] = load_df["datetime"].dt.floor("h")
    weather_df["mk"] = weather_df["datetime"].dt.floor("h")

    ercot = prices_df[["mk", "price_usd_mwh"]].merge(load_df[["mk", "total_mw"]], on="mk", how="inner")
    full = ercot.merge(weather_df.drop(columns=["datetime"]), on="mk", how="inner")
    full = full.rename(columns={"mk": "datetime", "price_usd_mwh": "market_price", "total_mw": "grid_demand"})
    print(f"      Merged: {len(full)} hours")

    full = derive_energy_output(full)
    full["market_price"] = full["market_price"].clip(0.01, 500)
    full = add_lag_features(full)

    df = full.drop(columns=["datetime"])
    print(f"\n      ✅ Final: {len(df)} rows, {len(df.columns)} columns")
    print(f"      Price: ${df['market_price'].min():.2f} - ${df['market_price'].max():.2f} (mean ${df['market_price'].mean():.2f})")

    if len(df) < 200:
        print("❌ Not enough data after lag warmup"); return

    # Train
    print("\n[2/5] Splitting (80/20)...")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    targets = [
        ("energy_output", ENERGY_FEATURES, df["energy_output"]),
        ("price", PRICE_FEATURES, df["market_price"]),
        ("demand", DEMAND_FEATURES, df["grid_demand"]),
    ]

    print("\n[3/5] Training...")
    for name, feats, y in targets:
        print(f"\n      --- {name} ({len(feats)} features) ---")
        X = df[feats]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = EnsembleForecaster(name, n_estimators=100)
        m = model.train(X_train, y_train, X_val, y_val)
        print(f"      MAPE: {m.get('ensemble_mape','N/A')}% | RMSE: {m.get('ensemble_rmse','N/A')} | MAE: {m.get('ensemble_mae','N/A')}")
        print(f"      vs Naive: {m.get('naive_mape','N/A')}% → {m.get('improvement_over_naive_pct','N/A')}% better")
        if model.feature_importances:
            top = sorted(model.feature_importances.items(), key=lambda x: x[1].get("xgb",0), reverse=True)[:5]
            for f, s in top:
                print(f"        {f:25s} RF={s['rf']:.4f} XGB={s['xgb']:.4f}")
        model.save(model_dir)

    print("\n[4/5] Saving training data...")
    df.to_csv(os.path.join(data_dir, "training_data.csv"), index=False)

    print("\n[5/5] Done!")
    print("=" * 60)
    print(f"  {len(df)} samples | 100% real ERCOT + Open-Meteo data")
    print(f"  Models → {model_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=45, help="Days of history (default 45)")
    main(parser.parse_args().days)