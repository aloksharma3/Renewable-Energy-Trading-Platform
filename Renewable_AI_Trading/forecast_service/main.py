"""
ML Forecasting Service (v4 — with Lag Features + DAM Price)
=============================================================
This service is the "brain" of the trading platform.

WHAT IT DOES:
    1. Loads trained ensemble models (RandomForest + XGBoost)
    2. Gets current weather + market data from the data service
    3. Fetches recent price/demand history to compute lag features
    4. Runs predictions: energy output, price, and grid demand
    5. Returns predictions WITH confidence intervals

WHAT CHANGED FROM V3:
    - /data/combined now includes dam_price from ERCOT DAM API (np4-190-cd)
    - DAM price is passed through to the trading service for context
    - Price model still uses same 23 features (no retraining needed)

LAG FEATURES COMPUTED AT INFERENCE:
    price_lag_1h, 2h, 3h, 24h, 168h  — from ERCOT price history
    demand_lag_1h, 24h                — from forecast demand history
    price_rolling_6h, 24h             — rolling mean of recent prices
    demand_rolling_6h                 — rolling mean of recent demand
    price_diff_1h, 24h                — price momentum
    hour, day_of_week, is_weekend, is_peak_hour — from current timestamp

ADDITIONAL DATA (passthrough, not ML features):
    dam_price                         — Day-Ahead Market price for current hour
"""

import os
import logging
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from fastapi import FastAPI, HTTPException

from ensemble_forecaster import EnsembleForecaster
from database import ForecastDatabase

# ─── Setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("forecast_service")

app = FastAPI(
    title="Energy Forecasting Service",
    description="Ensemble ML predictions with lag features + DAM price for energy, price, and demand",
    version="4.0.0",
)

# ─── Configuration ──────────────────────────────────────────
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://data_service:8001")
MODEL_DIR        = os.getenv("MODEL_DIR", "models")
DB_PATH          = os.getenv("FORECAST_DB_PATH", "data/forecast.db")

# ─── Persistent Forecast Database ───────────────────────────
# Replaces the in-memory forecast_history [] list.
# Records survive container restarts via the forecast_data Docker volume.
db = ForecastDatabase(DB_PATH)
logger.info(f"Forecast DB ready at {DB_PATH} ({db.count()} existing records)")

# ─── Feature Lists (must match train_ensemble.py v4) ────────
WEATHER_FEATURES = [
    "temp", "humidity", "wind_speed", "cloud_coverage",
    "irradiance", "direct_radiation", "dni",
]

LAG_FEATURES = [
    "price_lag_1h", "price_lag_2h", "price_lag_3h",
    "price_lag_24h", "price_lag_168h",
    "demand_lag_1h", "demand_lag_24h",
    "price_rolling_6h", "price_rolling_24h", "demand_rolling_6h",
    "price_diff_1h", "price_diff_24h",
    "hour", "day_of_week", "is_weekend", "is_peak_hour",
]

ENERGY_FEATURES = WEATHER_FEATURES  # 7 features
PRICE_FEATURES = WEATHER_FEATURES + LAG_FEATURES  # 23 features
DEMAND_FEATURES = WEATHER_FEATURES + [
    "demand_lag_1h", "demand_lag_24h", "demand_rolling_6h",
    "hour", "day_of_week", "is_weekend", "is_peak_hour",
]  # 14 features

# ─── Load Models ────────────────────────────────────────────
energy_model = EnsembleForecaster("energy_output")
price_model = EnsembleForecaster("price")
demand_model = EnsembleForecaster("demand")


def load_models():
    """Load all 3 models at startup."""
    for model in [energy_model, price_model, demand_model]:
        try:
            model.load(MODEL_DIR)
            logger.info(f"Loaded {model.target_name} model")
        except FileNotFoundError as e:
            logger.error(f"Could not load {model.target_name}: {e}")

load_models()

# ─── In-Memory Buffers ──────────────────────────────────────
# latest_forecast: cached in RAM for the fast /forecast/latest endpoint
#                  (avoids a DB read on every dashboard refresh)
# forecast_history: REMOVED — now persisted in forecast.db via `db`
#
# price_history / demand_history: still in-memory — these are used only
# for computing lag features at inference time and do not need to persist
# across restarts (the data service provides real ERCOT history on startup).
latest_forecast = {}

# Store recent prices and demand predictions for lag features
# Each entry: {"timestamp": ..., "price": ..., "demand": ...}
price_history  = []   # last 200 hourly prices from ERCOT
demand_history = []   # last 200 hourly demand values


# ═══════════════════════════════════════════════════════════
#  DATA FETCHING
# ═══════════════════════════════════════════════════════════

def get_data_from_service():
    """Fetch current weather + market data (RT + DAM) from the data service."""
    try:
        response = requests.get(f"{DATA_SERVICE_URL}/data/combined", timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.info(
            f"Got data: temp={data.get('temp')}°C, "
            f"RT=${data.get('market_price')}, "
            f"DAM=${data.get('dam_price')}"
        )
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Data service unavailable: {e}")
        raise HTTPException(status_code=503, detail=f"Data service unavailable: {e}")


def fetch_ercot_history():
    """
    Fetch recent ERCOT price history from data service.
    Used to compute price lag features.

    Returns list of {"timestamp": ..., "price": ...} sorted by time.
    """
    try:
        response = requests.get(
            f"{DATA_SERVICE_URL}/ercot/history?limit=200",
            timeout=10
        )
        if response.status_code == 200:
            records = response.json()
            # Extract timestamp and price, sort by time
            history = []
            for r in records:
                try:
                    history.append({
                        "timestamp": r.get("timestamp", ""),
                        "price": float(r.get("price_usd_mwh", 0)),
                    })
                except (ValueError, TypeError):
                    continue
            history.sort(key=lambda x: x["timestamp"])
            return history
    except Exception as e:
        logger.warning(f"Could not fetch ERCOT history: {e}")
    return []


def fetch_forecast_demand_history():
    """
    Get recent demand values from our own forecast history.
    Since we don't have a separate demand history endpoint,
    we use our stored predictions as proxy.
    """
    demands = []
    for f in db.get_history(limit=200, order="asc"):
        demand_data = f.get("demand", {})
        if "predicted" in demand_data:
            demands.append({
                "timestamp": f.get("timestamp", ""),
                "demand": demand_data["predicted"],
            })
    return demands


# ═══════════════════════════════════════════════════════════
#  LAG FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════

def compute_lag_features(current_price, current_timestamp):
    """
    Compute all lag features for price and demand models.

    Uses in-memory price_history and demand_history buffers.
    Falls back to 0 if not enough history (first few runs).

    Args:
        current_price: Current ERCOT price (from data service)
        current_timestamp: Current timestamp string

    Returns:
        dict with all lag feature values
    """
    # Parse current time for time features
    try:
        if isinstance(current_timestamp, str):
            # Try multiple formats
            for fmt in ["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    now = datetime.strptime(current_timestamp, fmt)
                    break
                except ValueError:
                    continue
            else:
                now = datetime.utcnow()
        else:
            now = datetime.utcnow()
    except Exception:
        now = datetime.utcnow()

    # ─── Time Features ───────────────────────────────────────
    hour = now.hour
    day_of_week = now.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    is_peak_hour = 1 if 14 <= hour <= 20 else 0

    # ─── Price Lag Features ──────────────────────────────────
    # price_history is a list of floats, most recent last
    prices = [p for p in price_history]  # copy
    n = len(prices)

    price_lag_1h = prices[-1] if n >= 1 else (current_price or 0)
    price_lag_2h = prices[-2] if n >= 2 else price_lag_1h
    price_lag_3h = prices[-3] if n >= 3 else price_lag_2h
    price_lag_24h = prices[-24] if n >= 24 else price_lag_1h
    price_lag_168h = prices[-168] if n >= 168 else price_lag_24h

    # Rolling averages
    if n >= 6:
        price_rolling_6h = np.mean(prices[-6:])
    elif n >= 1:
        price_rolling_6h = np.mean(prices)
    else:
        price_rolling_6h = current_price or 0

    if n >= 24:
        price_rolling_24h = np.mean(prices[-24:])
    elif n >= 1:
        price_rolling_24h = np.mean(prices)
    else:
        price_rolling_24h = current_price or 0

    # Price momentum (diff)
    price_diff_1h = (current_price or 0) - price_lag_1h
    price_diff_24h = (current_price or 0) - price_lag_24h

    # ─── Demand Lag Features ─────────────────────────────────
    demands = [d for d in demand_history]
    nd = len(demands)

    demand_lag_1h = demands[-1] if nd >= 1 else 50000  # reasonable default
    demand_lag_24h = demands[-24] if nd >= 24 else demand_lag_1h

    if nd >= 6:
        demand_rolling_6h = np.mean(demands[-6:])
    elif nd >= 1:
        demand_rolling_6h = np.mean(demands)
    else:
        demand_rolling_6h = 50000

    lags = {
        # Price lags
        "price_lag_1h": round(price_lag_1h, 2),
        "price_lag_2h": round(price_lag_2h, 2),
        "price_lag_3h": round(price_lag_3h, 2),
        "price_lag_24h": round(price_lag_24h, 2),
        "price_lag_168h": round(price_lag_168h, 2),
        # Demand lags
        "demand_lag_1h": round(demand_lag_1h, 2),
        "demand_lag_24h": round(demand_lag_24h, 2),
        # Rolling averages
        "price_rolling_6h": round(price_rolling_6h, 2),
        "price_rolling_24h": round(price_rolling_24h, 2),
        "demand_rolling_6h": round(demand_rolling_6h, 2),
        # Momentum
        "price_diff_1h": round(price_diff_1h, 2),
        "price_diff_24h": round(price_diff_24h, 2),
        # Time
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_peak_hour": is_peak_hour,
    }

    logger.info(
        f"Lag features: price_lag_1h=${lags['price_lag_1h']}, "
        f"rolling_6h=${lags['price_rolling_6h']}, "
        f"hour={hour}, history_depth={n} prices, {nd} demands"
    )

    return lags


def update_history(price, demand_predicted):
    """
    Update in-memory history buffers after each forecast run.
    Called after every prediction to maintain lag data for next run.
    """
    if price is not None:
        price_history.append(float(price))
        # Keep last 200 entries
        while len(price_history) > 200:
            price_history.pop(0)

    if demand_predicted is not None:
        demand_history.append(float(demand_predicted))
        while len(demand_history) > 200:
            demand_history.pop(0)


def seed_history_from_data_service():
    """
    On startup or when history is empty, seed from data service.
    This gives us lag features from the first forecast run.
    """
    if len(price_history) > 0:
        return  # already seeded

    logger.info("Seeding price history from data service...")
    ercot_records = fetch_ercot_history()
    if ercot_records:
        for r in ercot_records:
            price_history.append(r["price"])
        logger.info(f"Seeded {len(price_history)} price records")
    else:
        logger.warning("No ERCOT history available for seeding")


# ═══════════════════════════════════════════════════════════
#  FEATURE PREPARATION
# ═══════════════════════════════════════════════════════════

def prepare_energy_features(data):
    """Prepare 7 weather features for energy output model."""
    feature_dict = {f: data.get(f, 0) for f in ENERGY_FEATURES}
    return pd.DataFrame([feature_dict])


def prepare_price_features(data, lag_features):
    """Prepare 23 features (weather + lags) for price model."""
    feature_dict = {f: data.get(f, 0) for f in WEATHER_FEATURES}
    feature_dict.update(lag_features)
    return pd.DataFrame([feature_dict])[PRICE_FEATURES]


def prepare_demand_features(data, lag_features):
    """Prepare 14 features (weather + demand lags + time) for demand model."""
    feature_dict = {f: data.get(f, 0) for f in WEATHER_FEATURES}
    # Only include demand-relevant lag features
    for f in ["demand_lag_1h", "demand_lag_24h", "demand_rolling_6h",
              "hour", "day_of_week", "is_weekend", "is_peak_hour"]:
        feature_dict[f] = lag_features.get(f, 0)
    return pd.DataFrame([feature_dict])[DEMAND_FEATURES]


# ═══════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """Health check — reports models loaded and history depth."""
    return {
        "status": "healthy",
        "service": "forecasting",
        "version": "3.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": {
            "energy_output": energy_model.is_trained,
            "price": price_model.is_trained,
            "demand": demand_model.is_trained,
        },
        "forecast_count": db.count(),
        "price_history_depth": len(price_history),
        "demand_history_depth": len(demand_history),
    }


@app.post("/forecast/run")
def run_forecast():
    """
    Run the full forecasting pipeline.

    Called by scheduler every 15 minutes.

    Pipeline:
        1. GET /data/combined → current weather + ERCOT RT price + DAM price
        2. Seed price history from data service (if first run)
        3. Compute lag features from history buffers
        4. Run energy model (weather features only)
        5. Run price model (weather + lag features)
        6. Run demand model (weather + demand lags + time)
        7. Update history buffers for next run
        8. Return predictions + DAM price for trading service
    """
    global latest_forecast

    # Step 1: Get current data (now includes dam_price)
    data = get_data_from_service()

    # Step 2: Seed history on first run
    seed_history_from_data_service()

    # Step 3: Compute lag features
    current_price = data.get("market_price", 0) or 0
    current_timestamp = data.get("timestamp", datetime.utcnow().isoformat())
    lag_features = compute_lag_features(current_price, current_timestamp)

    # Step 4: Build results
    results = {
        "timestamp": current_timestamp,
        "weather": {
            "temp": data.get("temp"),
            "wind_speed": data.get("wind_speed"),
            "irradiance": data.get("irradiance"),
            "humidity": data.get("humidity"),
        },
        "actual_market_price": data.get("market_price"),
        # DAM price passthrough for trading service
        "dam_price": data.get("dam_price"),
    }

    # Step 5: Predict energy output (weather only — 7 features)
    if energy_model.is_trained:
        try:
            X_energy = prepare_energy_features(data)
            energy_pred = energy_model.predict_with_confidence(X_energy)

            # Clamp: solar energy output cannot be negative.
            # At night (irradiance=0), force to 0.
            raw_energy = float(energy_pred["ensemble"][0])
            ghi = data.get("irradiance", 0) or 0
            clamped_energy = max(0.0, raw_energy) if ghi > 0 else 0.0

            results["energy_output"] = {
                "predicted": round(clamped_energy, 2),
                "rf_predicted": round(max(0.0, float(energy_pred["rf"][0])), 2),
                "xgb_predicted": round(max(0.0, float(energy_pred["xgb"][0])), 2),
                "confidence_lower": round(max(0.0, float(energy_pred["confidence_lower"][0])), 2),
                "confidence_upper": round(max(0.0, float(energy_pred["confidence_upper"][0])), 2),
                "unit": "MW",
            }
        except Exception as e:
            logger.error(f"Energy prediction failed: {e}")
            results["energy_output"] = {"error": str(e)}
    else:
        results["energy_output"] = {"error": "Model not loaded"}

    # Step 6: Predict price (weather + lags — 23 features)
    if price_model.is_trained:
        try:
            X_price = prepare_price_features(data, lag_features)
            price_pred = price_model.predict_with_confidence(X_price)

            # Clamp: ERCOT prices can go slightly negative (wind oversupply)
            # but not below -$10/MWh in practice. Cap at $5000 (ERCOT system cap).
            raw_price = float(price_pred["ensemble"][0])
            clamped_price = max(0.0, min(5000.0, raw_price))

            # If lag features are mostly defaults (first few runs), trust RF more
            # since XGBoost extrapolates badly with zero-filled lags.
            rf_price = float(price_pred["rf"][0])
            xgb_price = float(price_pred["xgb"][0])
            if abs(xgb_price) > 3 * abs(rf_price) and abs(rf_price) > 1:
                # XGBoost is extrapolating wildly — weight RF heavier
                clamped_price = max(0.0, min(5000.0, rf_price * 0.7 + xgb_price * 0.3))
                logger.warning(
                    f"XGBoost extrapolating: RF=${rf_price:.2f}, XGB=${xgb_price:.2f} "
                    f"→ using weighted avg ${clamped_price:.2f}"
                )

            results["price"] = {
                "predicted": round(clamped_price, 2),
                "rf_predicted": round(max(0.0, rf_price), 2),
                "xgb_predicted": round(max(0.0, xgb_price), 2),
                "confidence_lower": round(max(0.0, float(price_pred["confidence_lower"][0])), 2),
                "confidence_upper": round(min(5000.0, float(price_pred["confidence_upper"][0])), 2),
                "unit": "$/MWh",
            }
        except Exception as e:
            logger.error(f"Price prediction failed: {e}")
            results["price"] = {"error": str(e)}
    else:
        results["price"] = {"error": "Model not loaded"}

    # Step 7: Predict demand (weather + demand lags + time — 14 features)
    if demand_model.is_trained:
        try:
            X_demand = prepare_demand_features(data, lag_features)
            demand_pred = demand_model.predict_with_confidence(X_demand)

            # Clamp: grid demand is always positive.
            # ERCOT typical range: 25,000–80,000 MW.
            raw_demand = float(demand_pred["ensemble"][0])
            clamped_demand = max(20000.0, min(85000.0, raw_demand))

            results["demand"] = {
                "predicted": round(clamped_demand, 2),
                "rf_predicted": round(max(0.0, float(demand_pred["rf"][0])), 2),
                "xgb_predicted": round(max(0.0, float(demand_pred["xgb"][0])), 2),
                "confidence_lower": round(max(20000.0, float(demand_pred["confidence_lower"][0])), 2),
                "confidence_upper": round(min(85000.0, float(demand_pred["confidence_upper"][0])), 2),
                "unit": "MW",
            }
        except Exception as e:
            logger.error(f"Demand prediction failed: {e}")
            results["demand"] = {"error": str(e)}
    else:
        results["demand"] = {"error": "Model not loaded"}

    # Step 8: Update history buffers for next run's lag computation
    update_history(
        price=data.get("market_price"),
        demand_predicted=results.get("demand", {}).get("predicted"),
    )

    # Persist forecast to SQLite and update in-memory latest cache
    latest_forecast = results
    db.save_forecast(results)

    logger.info(
        f"Forecast complete: energy={results.get('energy_output', {}).get('predicted')}MW, "
        f"price=${results.get('price', {}).get('predicted')}/MWh, "
        f"demand={results.get('demand', {}).get('predicted')}MW"
    )
    return results


@app.get("/forecast/latest")
def get_latest_forecast():
    """Return the most recent forecast — served from in-memory cache for speed."""
    if latest_forecast:
        return latest_forecast
    # Fallback: read from DB on first request after a cold restart
    record = db.get_latest()
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="No forecast available. Call POST /forecast/run first."
        )
    return record


@app.get("/forecast/history")
def get_forecast_history(limit: int = 96):
    """
    Return recent forecast history from the persistent SQLite database.
    Newest first (desc) so the dashboard chart and backtest engine both
    receive the same ordering they previously expected from the in-memory list.
    """
    return db.get_history(limit=limit, order="desc")


@app.get("/forecast/count")
def get_forecast_count():
    """Return total number of stored forecast records."""
    return {"count": db.count()}


@app.get("/models/metrics")
def get_model_metrics():
    """Return evaluation metrics for all models."""
    return {
        "energy_output": energy_model.metrics,
        "price": price_model.metrics,
        "demand": demand_model.metrics,
    }


@app.get("/models/feature-importance")
def get_feature_importance():
    """Return feature importance for all models."""
    return {
        "energy_output": energy_model.feature_importances,
        "price": price_model.feature_importances,
        "demand": demand_model.feature_importances,
    }


@app.post("/models/retrain")
def retrain_models(min_rows: int = 200):
    """
    Retrain all 3 ensemble models using accumulated real data.

    RETRAINING PIPELINE:
        1. Fetch collected training data from data service CSV
        2. Merge with existing training_data.csv (if any)
        3. Compute lag features + derived targets
        4. Split 80/20 and retrain all 3 models
        5. Save new models to disk
        6. Reload models into memory

    WHY RETRAIN?
        The initial models were trained on limited data.
        As the platform runs, it collects real weather + ERCOT prices.
        Retraining on this accumulated data improves predictions because:
        - More diverse conditions (day/night, weekday/weekend, weather events)
        - Real price patterns instead of synthetic approximations
        - Lag features computed from actual market data

    Call manually or via scheduler (weekly).
    """
    from sklearn.model_selection import train_test_split

    logger.info("Starting model retraining...")

    # Step 1: Get collected data from data service
    try:
        # Try to read collected CSV directly (shared volume)
        collected_path = "/app/data/collected_data.csv"
        original_path = "/app/data/training_data.csv"

        dfs = []

        # Load collected real data
        if os.path.exists(collected_path):
            df_collected = pd.read_csv(collected_path)
            df_collected["datetime"] = pd.to_datetime(df_collected["datetime"])
            logger.info(f"Loaded {len(df_collected)} collected records")
            dfs.append(df_collected)

        # Also try fetching from data service
        try:
            resp = requests.get(f"{DATA_SERVICE_URL}/data/training-stats", timeout=5)
            if resp.status_code == 200:
                stats = resp.json()
                logger.info(f"Data service training stats: {stats}")
        except Exception:
            pass

        # Load original training data as base
        df_original_ready = None
        if os.path.exists(original_path):
            df_original = pd.read_csv(original_path)
            if "datetime" not in df_original.columns and "market_price" in df_original.columns:
                # Original training data already has lag features pre-computed
                # Use it directly — no need to recompute lags
                df_original_ready = df_original
                logger.info(f"Original training data has {len(df_original)} rows (pre-computed lags, ready to use)")
            else:
                df_original["datetime"] = pd.to_datetime(df_original["datetime"])
                dfs.append(df_original)
                logger.info(f"Loaded {len(df_original)} original training records")

        if not dfs and df_original_ready is None:
            return {"error": "No training data found", "hint": "Run the pipeline for a while to collect data, then call POST /data/save-training"}

        # Merge collected data sources (those with datetime)
        if dfs:
            if len(dfs) > 1:
                df_new = pd.concat(dfs, ignore_index=True)
                df_new = df_new.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
            else:
                df_new = dfs[0].sort_values("datetime").reset_index(drop=True) if "datetime" in dfs[0].columns else dfs[0]
            logger.info(f"New collected data: {len(df_new)} rows")
        else:
            df_new = None

        logger.info(f"Data sources — original: {len(df_original_ready) if df_original_ready is not None else 0}, new: {len(df_new) if df_new is not None else 0}")

    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return {"error": f"Failed to load data: {e}"}

    # TWO PATHS:
    # Path A: Original training_data.csv already has lag features — use directly
    # Path B: New collected data needs lag features computed, then combine with Path A

    frames_to_train = []

    # Path A: Original pre-computed data
    if df_original_ready is not None and len(df_original_ready) > 0:
        # Verify it has all required columns
        required_cols = WEATHER_FEATURES + LAG_FEATURES + ["market_price", "energy_output"]
        missing = [c for c in required_cols if c not in df_original_ready.columns]
        if not missing:
            frames_to_train.append(df_original_ready)
            logger.info(f"Path A: {len(df_original_ready)} original rows ready")
        else:
            logger.warning(f"Original data missing columns: {missing}")

    # Path B: New collected data — compute lag features
    if df_new is not None and len(df_new) > 50:
        df = df_new.copy()

        # Ensure required weather columns exist
        for col in WEATHER_FEATURES:
            if col not in df.columns:
                df[col] = 0

        if "market_price" not in df.columns:
            logger.warning("New data missing market_price, skipping")
        else:
            # Derive energy output if missing
            if "energy_output" not in df.columns:
                wind = 3.0 * np.minimum(df["wind_speed"].values, 12) ** 1.5
                solar = 0.05 * df["irradiance"].values
                np.random.seed(42)
                df["energy_output"] = np.clip(wind + solar + np.random.normal(0, 3, len(df)), 0, 200).round(2)

            # Derive grid demand if missing
            if "grid_demand" not in df.columns:
                if "hour" not in df.columns and "datetime" in df.columns:
                    df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
                h = df.get("hour", pd.Series([12]*len(df)))
                base_demand = 40000 + 15000 * np.sin((h - 6) * np.pi / 12)
                df["grid_demand"] = np.clip(base_demand + df["market_price"] * 200 + np.random.normal(0, 2000, len(df)), 25000, 75000).round(2)

            # Clamp prices
            df["market_price"] = df["market_price"].clip(0.01, 500)

            # Add lag features
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

            if "datetime" in df.columns:
                df["hour"] = pd.to_datetime(df["datetime"]).dt.hour
                df["day_of_week"] = pd.to_datetime(df["datetime"]).dt.dayofweek
            elif "hour" not in df.columns:
                df["hour"] = 12
                df["day_of_week"] = 3

            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
            df["is_peak_hour"] = ((df["hour"] >= 14) & (df["hour"] <= 20)).astype(int)

            before = len(df)
            df = df.dropna()
            logger.info(f"Path B: {len(df)} new rows after lag warmup (dropped {before - len(df)})")

            if len(df) > 0:
                frames_to_train.append(df)

    if not frames_to_train:
        return {
            "error": "No usable training data",
            "hint": "Need original training_data.csv or 200+ collected records",
        }

    # Combine all training frames
    # Select only the columns needed for training
    all_needed = list(set(WEATHER_FEATURES + LAG_FEATURES + ["market_price", "energy_output", "grid_demand"]))
    combined_frames = []
    for frame in frames_to_train:
        available = [c for c in all_needed if c in frame.columns]
        combined_frames.append(frame[available])

    df = pd.concat(combined_frames, ignore_index=True)
    df = df.dropna()
    logger.info(f"Combined training data: {len(df)} rows")

    if len(df) < min_rows:
        return {
            "error": f"Not enough data ({len(df)} < {min_rows})",
            "rows_available": len(df),
            "rows_needed": min_rows,
        }

    # Step 6: Train all 3 models
    results = {}
    targets = [
        ("energy_output", ENERGY_FEATURES, df["energy_output"]),
        ("price", PRICE_FEATURES, df["market_price"]),
        ("demand", DEMAND_FEATURES, df["grid_demand"]),
    ]

    for name, feats, y in targets:
        try:
            X = df[feats]
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            model = EnsembleForecaster(name, n_estimators=100)
            metrics = model.train(X_train, y_train, X_val, y_val)
            model.save(MODEL_DIR)

            results[name] = {
                "status": "success",
                "metrics": metrics,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
            }
            logger.info(f"Retrained {name}: MAPE={metrics.get('ensemble_mape')}%")
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}
            logger.error(f"Failed to retrain {name}: {e}")

    # Step 7: Reload models into memory
    load_models()
    logger.info("Models reloaded after retraining")

    return {
        "message": "Retraining complete",
        "total_rows": len(df),
        "models": results,
    }