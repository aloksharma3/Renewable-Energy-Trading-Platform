"""
Data Ingestion Service (v2)
============================
This service is the "eyes" of the trading platform. It fetches TWO types of real data:

1. WEATHER + SOLAR DATA from Open-Meteo API (Dallas, Texas)
   - Temperature, wind speed, humidity, cloud coverage
   - REAL solar irradiance (GHI, DNI, direct radiation) — not calculated
   - Free, no API key needed, 15-minute resolution for North America
   - Uses NOAA HRRR model (High Resolution Rapid Refresh) for the US

2. MARKET DATA from ERCOT (Electric Reliability Council of Texas)
   - Real settlement point prices ($/MWh) every 15 minutes
   - HB_NORTH hub (Dallas region) to match our weather location
   - Falls back to realistic synthetic prices if API is unavailable

WHY OPEN-METEO INSTEAD OF OPENWEATHERMAP?
   - Open-Meteo gives REAL solar irradiance (GHI, DNI, direct radiation)
   - OpenWeatherMap does NOT provide irradiance (we had to fake it with formulas)
   - Open-Meteo is completely free with no API key
   - It has 15-minute data for North America (matches our ERCOT 15-min intervals)
   - Uses NOAA's HRRR model which is the gold standard for US weather

WHAT IS GHI vs DNI vs DIRECT RADIATION?
   - GHI (Global Horizontal Irradiance): total sunlight hitting a flat surface
     → This is what a solar panel laying flat on a roof receives
   - DNI (Direct Normal Irradiance): sunlight hitting a surface pointed at the sun
     → This is what a solar tracker panel receives (higher than GHI)
   - Direct Radiation: the portion of sunlight coming straight from the sun
     → Excludes scattered light from clouds/atmosphere

   We use GHI because most residential solar panels are fixed (not tracking).
"""

import os
import logging
import requests
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException

# ─── Setup Logging ──────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("data_service")

# ─── Create FastAPI App ─────────────────────────────────────
app = FastAPI(
    title="Energy Data Ingestion Service",
    description="Fetches weather (Open-Meteo) and market data (ERCOT RT-SPP + DAM) for ML pipeline",
    version="3.0.0",
)

# ─── Configuration ──────────────────────────────────────────
# Dallas, TX coordinates (center of ERCOT North Hub)
LAT = os.getenv("LAT", "32.7767")
LON = os.getenv("LON", "-96.7970")

# ─── In-Memory Storage ─────────────────────────────────────
latest_weather = {}
latest_ercot = {}
latest_dam = {}          # Day-Ahead Market latest price
weather_history = []
ercot_history = []
dam_history = []         # Day-Ahead Market price history


# ═══════════════════════════════════════════════════════════
#  WEATHER + SOLAR DATA (Open-Meteo)
# ═══════════════════════════════════════════════════════════

def fetch_weather():
    """
    Fetch current weather + real solar irradiance from Open-Meteo API.

    API call explained:
        https://api.open-meteo.com/v1/forecast
            ?latitude=32.7767              → Dallas, TX
            &longitude=-96.7970
            &current=temperature_2m,...     → get CURRENT values (not forecast)
            &timezone=America/Chicago       → Central Time (Texas)

    The "current" parameter returns the most recent 15-minute observation.

    Variables we fetch:
        - temperature_2m:               Air temperature at 2 meters height (°C)
        - relative_humidity_2m:         Relative humidity at 2 meters (%)
        - wind_speed_10m:               Wind speed at 10 meters height (m/s)
        - cloud_cover:                  Total cloud coverage (%)
        - shortwave_radiation:          GHI - total sunlight on flat surface (W/m²)
        - direct_radiation:             Direct beam radiation (W/m²)
        - direct_normal_irradiance:     DNI - sunlight on sun-tracking surface (W/m²)

    Returns a dict with all weather features for the ML model.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "current": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",       # GHI (W/m²) — real, not calculated
            "direct_radiation",           # Direct beam (W/m²)
            "direct_normal_irradiance",   # DNI (W/m²)
        ]),
        "timezone": "America/Chicago",    # Central Time for Texas
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data["current"]

        record = {
            "timestamp": current.get("time", datetime.utcnow().strftime("%Y-%m-%dT%H:%M")),
            "temp": current.get("temperature_2m", 0),
            "humidity": current.get("relative_humidity_2m", 0),
            "wind_speed": current.get("wind_speed_10m", 0),
            "cloud_coverage": current.get("cloud_cover", 0),
            "irradiance": current.get("shortwave_radiation", 0),       # GHI — REAL
            "direct_radiation": current.get("direct_radiation", 0),     # Direct beam
            "dni": current.get("direct_normal_irradiance", 0),          # DNI
            "source": "open_meteo",
        }

        logger.info(
            f"Weather fetched: {record['temp']}°C, "
            f"wind={record['wind_speed']}m/s, "
            f"GHI={record['irradiance']}W/m², "
            f"DNI={record['dni']}W/m²"
        )
        return record

    except requests.exceptions.RequestException as e:
        logger.error(f"Open-Meteo API failed: {e} — using synthetic data")
        return _generate_synthetic_weather()


def fetch_weather_history_from_api(hours=24):
    """
    Fetch recent hourly weather history from Open-Meteo.

    WHY THIS FUNCTION?
        When the service first starts, weather_history is empty.
        This function backfills the last 24 hours of weather data
        so the dashboard and ML models have something to work with immediately.

    Uses the "past_hours" parameter to get historical data.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
            "direct_radiation",
            "direct_normal_irradiance",
        ]),
        "past_hours": hours,
        "forecast_hours": 1,
        "timezone": "America/Chicago",
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        hourly = data["hourly"]

        records = []
        for i in range(len(hourly["time"])):
            records.append({
                "timestamp": hourly["time"][i],
                "temp": hourly["temperature_2m"][i] or 0,
                "humidity": hourly["relative_humidity_2m"][i] or 0,
                "wind_speed": hourly["wind_speed_10m"][i] or 0,
                "cloud_coverage": hourly["cloud_cover"][i] or 0,
                "irradiance": hourly["shortwave_radiation"][i] or 0,
                "direct_radiation": hourly["direct_radiation"][i] or 0,
                "dni": hourly["direct_normal_irradiance"][i] or 0,
                "source": "open_meteo_historical",
            })

        logger.info(f"Fetched {len(records)} hours of weather history")
        return records

    except Exception as e:
        logger.error(f"Failed to fetch weather history: {e}")
        return []


def _generate_synthetic_weather():
    """
    Generate realistic synthetic weather data for Dallas, TX.
    Used as fallback when Open-Meteo API is unavailable.

    Even the synthetic data follows realistic Dallas patterns:
    - Summer highs: 35-40°C (very hot)
    - Wind: averages 4-5 m/s (Texas is windy)
    - Irradiance: peaks ~900 W/m² at noon in summer
    - Cloud cover: generally low (Texas is sunny)
    """
    hour = (datetime.utcnow().hour - 6) % 24  # UTC → CST

    base_temp = 28 + 8 * np.sin((hour - 6) * np.pi / 12)
    temp = round(base_temp + np.random.normal(0, 2), 1)
    wind_speed = round(max(0.5, 4.5 + np.random.normal(0, 2)), 1)
    cloud_coverage = round(max(0, min(100, 30 + np.random.normal(0, 20))), 1)
    humidity = round(max(20, min(95, 50 + np.random.normal(0, 15))), 1)

    if 6 <= hour <= 18:
        solar_factor = max(0, np.sin((hour - 6) * np.pi / 12))
        irradiance = round(950 * solar_factor * (1 - cloud_coverage / 100.0), 1)
        direct_radiation = round(irradiance * 0.7, 1)
        dni = round(irradiance * 1.2, 1)
    else:
        irradiance = 0.0
        direct_radiation = 0.0
        dni = 0.0

    return {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M"),
        "temp": temp,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "cloud_coverage": cloud_coverage,
        "irradiance": irradiance,
        "direct_radiation": direct_radiation,
        "dni": dni,
        "source": "synthetic",
    }


# ═══════════════════════════════════════════════════════════
#  ERCOT MARKET DATA (Authenticated API)
# ═══════════════════════════════════════════════════════════

# ERCOT API Authentication
# Requires 3 credentials (set in .env):
#   ERCOT_SUBSCRIPTION_KEY  → from apiexplorer.ercot.com Profile → Subscriptions
#   ERCOT_USERNAME          → email used to register
#   ERCOT_PASSWORD          → password for apiexplorer.ercot.com
#
# Auth flow:
#   1. POST to Azure B2C token endpoint with username/password → get id_token
#   2. Include id_token as Bearer token + subscription key in API requests
#   3. Token valid for 1 hour, then must re-authenticate

ERCOT_SUBSCRIPTION_KEY = os.getenv("ERCOT_SUBSCRIPTION_KEY", "")
ERCOT_USERNAME = os.getenv("ERCOT_USERNAME", "")
ERCOT_PASSWORD = os.getenv("ERCOT_PASSWORD", "")

# Second ERCOT account for RT-LMP (np6-970-cd)
# Separate subscription needed because ERCOT limits one API per subscription
ERCOT_SUBSCRIPTION_KEY_RT = os.getenv("ERCOT_SUBSCRIPTION_KEY_RT", "")
ERCOT_USERNAME_RT = os.getenv("ERCOT_USERNAME_RT", "")
ERCOT_PASSWORD_RT = os.getenv("ERCOT_PASSWORD_RT", "")

ERCOT_TOKEN_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
ERCOT_API_BASE = "https://api.ercot.com/api/public-reports"

# Cache tokens (valid for 1 hour) — separate cache per account
_ercot_token = {"id_token": None, "expires_at": 0}
_ercot_token_rt = {"id_token": None, "expires_at": 0}


def _get_ercot_token(username=None, password=None, token_cache=None):
    """
    Get ERCOT API bearer token via Azure B2C ROPC flow.

    Supports multiple accounts by accepting credentials and cache as params.
    The token is cached for 50 minutes (token valid for 60 min).
    Returns the id_token string, or None if auth fails.
    """
    import time as _time

    if token_cache is None:
        token_cache = _ercot_token
    if username is None:
        username = ERCOT_USERNAME
    if password is None:
        password = ERCOT_PASSWORD

    # Return cached token if still valid
    if token_cache["id_token"] and _time.time() < token_cache["expires_at"]:
        return token_cache["id_token"]

    if not username or not password:
        logger.warning("ERCOT username or password not set")
        return None

    try:
        data = {
            "username": username,
            "password": password,
            "grant_type": "password",
            "scope": "openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access",
            "client_id": "fec253ea-0d06-4272-a5e6-b478baeecd70",
            "response_type": "id_token",
        }

        response = requests.post(
            ERCOT_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )

        if response.status_code == 200:
            token_data = response.json()
            token_cache["id_token"] = token_data.get("id_token")
            # Cache for 50 minutes (token valid for 60)
            token_cache["expires_at"] = _time.time() + 3000
            logger.info(f"ERCOT authentication successful (user: {username[:10]}...)")
            return token_cache["id_token"]
        else:
            logger.error(f"ERCOT auth failed: HTTP {response.status_code} - {response.text[:200]}")
            return None

    except Exception as e:
        logger.error(f"ERCOT auth error: {e}")
        return None


def _build_ercot_headers(account="dam"):
    """Build ERCOT API request headers. account='dam' for DAM, 'rt' for RT-LMP."""
    if account == "rt" and ERCOT_SUBSCRIPTION_KEY_RT:
        headers = {"Ocp-Apim-Subscription-Key": ERCOT_SUBSCRIPTION_KEY_RT}
        token = _get_ercot_token(ERCOT_USERNAME_RT, ERCOT_PASSWORD_RT, _ercot_token_rt)
    else:
        headers = {"Ocp-Apim-Subscription-Key": ERCOT_SUBSCRIPTION_KEY}
        token = _get_ercot_token(ERCOT_USERNAME, ERCOT_PASSWORD, _ercot_token)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_ercot_prices():
    """
    Fetch real-time LMP prices from ERCOT API.

    ENDPOINT: np6-970-cd/rtd_lmp_node_zone_hub
        Real-Time Dispatch LMP (Locational Marginal Prices) for hubs/zones/nodes.
        Updated every 5 minutes — more frequent than 15-min settlement prices.

    USES SECOND ERCOT ACCOUNT:
        ERCOT limits one API subscription per account. The first account is used
        for DAM prices (np4-190-cd). This function uses the second account
        (ERCOT_SUBSCRIPTION_KEY_RT) for real-time LMP data.

    WHAT IS LMP?
        Locational Marginal Price = the cost of delivering 1 MWh of electricity
        to a specific location, accounting for:
        - Energy cost (generation)
        - Congestion cost (transmission bottlenecks)
        - Loss cost (energy lost in transmission)

    API PARAMS (np6-970-cd):
        RTDTimestampFrom/To      — timestamp range filter
        settlementPoint          — hub name (e.g., HB_NORTH)
        settlementPointType      — filter by type (Hub, Zone, Node)
        LMPFrom/To               — price range filter
        size                     — records per page
        sort / dir               — sorting
    """
    # Check if RT credentials are configured
    if not ERCOT_SUBSCRIPTION_KEY_RT:
        logger.warning("ERCOT_SUBSCRIPTION_KEY_RT not set — using synthetic prices")
        return _generate_synthetic_ercot()

    try:
        url = f"{ERCOT_API_BASE}/np6-970-cd/rtd_lmp_node_zone_hub"

        params = {
            "settlementPoint": "HB_NORTH",
            "settlementPointType": "Hub",
            "size": 12,              # Just last hour of 5-min data (12 intervals)
            "sort": "RTDTimestamp",
            "dir": "DESC",
        }

        response = requests.get(
            url, params=params, headers=_build_ercot_headers(account="rt"), timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                field_names = [f["name"] for f in data.get("fields", [])]

                records = []
                for row in data["data"]:
                    try:
                        if field_names:
                            item = dict(zip(field_names, row))
                        elif isinstance(row, dict):
                            item = row
                        else:
                            continue

                        # RTD LMP uses different field names than SPP
                        timestamp = str(item.get("RTDTimestamp", item.get("intervalEnding", "")))
                        settlement_point = str(item.get("settlementPoint", "HB_NORTH"))
                        lmp = float(item.get("LMP", item.get("settlementPointPrice", 0)))

                        records.append({
                            "timestamp": timestamp,
                            "settlement_point": settlement_point,
                            "price_usd_mwh": round(lmp, 2),
                            "source": "ercot_rtd_lmp",
                        })
                    except (ValueError, TypeError, IndexError):
                        continue

                if records:
                    logger.info(f"Fetched {len(records)} REAL ERCOT RTD-LMP records")
                    return records

        logger.warning(
            f"ERCOT RTD-LMP API: HTTP {response.status_code} — "
            f"{response.text[:200] if response.text else 'empty response'}"
        )
        return _generate_synthetic_ercot()

    except Exception as e:
        logger.warning(f"ERCOT RTD-LMP API failed: {e} — using synthetic prices")
        return _generate_synthetic_ercot()


# ═══════════════════════════════════════════════════════════
#  ERCOT DAY-AHEAD MARKET (DAM) PRICES
# ═══════════════════════════════════════════════════════════

def fetch_dam_prices(date_str=None):
    """
    Fetch Day-Ahead Market (DAM) Settlement Point Prices from ERCOT API.

    ENDPOINT: np4-190-cd/dam_stlmnt_pnt_prices
        Hourly DAM prices published the day before for each settlement point.

    WHY DAM PRICES MATTER FOR TRADING:
        DAM prices are set via an auction the day before delivery.
        Comparing DAM vs Real-Time (RT) prices reveals:
        - RT > DAM → real-time demand exceeded day-ahead forecast → SELL signal
        - RT < DAM → real-time demand lower than expected → BUY signal (cheap power)
        - The spread (RT - DAM) is one of the strongest short-term trading signals

    API PARAMS (np4-190-cd):
        deliveryDateFrom/To      — date range (YYYY-MM-DD)
        hourEnding               — specific hour (1-24, Hour Ending format)
        settlementPoint          — hub name (e.g., HB_NORTH)
        settlementPointPriceFrom/To — price range filter
        DSTFlag                  — daylight saving time flag
        size                     — records per page
        sort / dir               — sorting

    Args:
        date_str: Date to fetch DAM prices for (YYYY-MM-DD).
                  Defaults to today. Pass tomorrow's date to get
                  the freshly published DAM prices for the next day.

    Returns:
        List of dicts with hourly DAM prices, or synthetic fallback.
    """
    if not ERCOT_SUBSCRIPTION_KEY:
        logger.warning("ERCOT_SUBSCRIPTION_KEY not set — using synthetic DAM prices")
        return _generate_synthetic_dam()

    if date_str is None:
        from zoneinfo import ZoneInfo
        date_str = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")

    try:
        url = f"{ERCOT_API_BASE}/np4-190-cd/dam_stlmnt_pnt_prices"
        params = {
            "deliveryDateFrom": date_str,
            "deliveryDateTo": date_str,
            "settlementPoint": "HB_NORTH",
            "size": 24,                    # 24 hours in a day
            "sort": "hourEnding",
            "dir": "ASC",
        }

        response = requests.get(
            url, params=params, headers=_build_ercot_headers(account="dam"), timeout=15
        )

        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                # ERCOT API returns data as list of lists, not list of dicts.
                # The "fields" key defines column order.
                # Typical DAM fields: [deliveryDate, hourEnding, settlementPoint,
                #                      settlementPointPrice, DSTFlag]
                field_names = [f["name"] for f in data.get("fields", [])]

                records = []
                for row in data["data"]:
                    try:
                        # Convert list to dict using field names
                        if field_names:
                            item = dict(zip(field_names, row))
                        elif isinstance(row, dict):
                            item = row
                        else:
                            # Fallback: assume standard DAM column order
                            item = {
                                "deliveryDate": row[0] if len(row) > 0 else date_str,
                                "hourEnding": row[1] if len(row) > 1 else "",
                                "settlementPoint": row[2] if len(row) > 2 else "HB_NORTH",
                                "settlementPointPrice": row[3] if len(row) > 3 else 0,
                                "DSTFlag": row[4] if len(row) > 4 else False,
                            }

                        delivery_date = str(item.get("deliveryDate", date_str))
                        hour_ending = str(item.get("hourEnding", "")).replace(":00", "")
                        price = round(
                            float(item.get("settlementPointPrice", 0)), 2
                        )

                        records.append({
                            "timestamp": f"{delivery_date} HE{hour_ending}",
                            "delivery_date": delivery_date,
                            "hour_ending": hour_ending,
                            "settlement_point": str(item.get("settlementPoint", "HB_NORTH")),
                            "dam_price_usd_mwh": price,
                            "dst_flag": item.get("DSTFlag", False),
                            "source": "ercot_dam_api",
                        })
                    except (ValueError, TypeError, IndexError):
                        continue

                logger.info(
                    f"Fetched {len(records)} REAL ERCOT DAM prices for {date_str}"
                )
                return records

        logger.warning(
            f"ERCOT DAM API: HTTP {response.status_code} — "
            f"{response.text[:200] if response.text else 'empty response'}"
        )
        return _generate_synthetic_dam()

    except Exception as e:
        logger.warning(f"ERCOT DAM API failed: {e} — using synthetic DAM prices")
        return _generate_synthetic_dam()


def get_dam_price_for_hour(hour=None):
    """
    Get the DAM price for a specific hour from cached history.

    Used by the combined endpoint and trading service to compute
    the RT-DAM spread for the current hour.

    Args:
        hour: Hour ending (1-24). Defaults to current CST hour.

    Returns:
        DAM price (float) or None if not available.
    """
    if hour is None:
        # Convert UTC to CST (UTC-6) and use hour ending (1-24)
        cst_hour = (datetime.utcnow().hour - 6) % 24
        hour = cst_hour if cst_hour > 0 else 24

    hour_str = str(hour)
    for record in reversed(dam_history):
        if record.get("hour_ending") == hour_str:
            return record.get("dam_price_usd_mwh")

    return None


def _generate_synthetic_dam():
    """
    Generate 24 hours of synthetic DAM prices for HB_NORTH.

    DAM prices follow similar patterns to RT but are smoother
    (no 15-min spikes) since they're set by auction the day before.

    Typical DAM patterns:
        Off-peak (HE1-HE6):     $18-28/MWh
        Morning ramp (HE7-10):  $30-45/MWh
        Afternoon peak (HE14-18): $45-70/MWh
        Evening (HE19-22):      $30-45/MWh
        Night (HE23-24):        $20-30/MWh
    """
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")
    records = []

    for he in range(1, 25):
        if he <= 6:
            base = 22 + np.random.normal(0, 3)
        elif he <= 10:
            base = 38 + np.random.normal(0, 5)
        elif he <= 13:
            base = 42 + np.random.normal(0, 6)
        elif he <= 18:
            base = 55 + np.random.normal(0, 8)
        elif he <= 22:
            base = 38 + np.random.normal(0, 5)
        else:
            base = 24 + np.random.normal(0, 3)

        price = round(max(5, base), 2)

        records.append({
            "timestamp": f"{today} HE{he}",
            "delivery_date": today,
            "hour_ending": str(he),
            "settlement_point": "HB_NORTH",
            "dam_price_usd_mwh": price,
            "dst_flag": False,
            "source": "synthetic_dam",
        })

    return records


def _generate_synthetic_ercot():
    """
    Generate 24 hours of realistic ERCOT pricing data (96 intervals of 15 min).

    Based on real ERCOT price patterns:
        Night (10pm-5am):    $15-25/MWh  (low demand, wind keeps blowing)
        Morning (6am-10am):  $30-50/MWh  (people waking up, AC turning on)
        Afternoon (2pm-6pm): $50-80/MWh  (peak AC demand, hottest part of day)
        Evening (7pm-9pm):   $35-50/MWh  (demand cooling off)

    5% chance of price spikes — this is realistic for ERCOT.
    In August 2023, ERCOT prices hit $5,000/MWh during a heat wave.
    """
    now = datetime.utcnow()
    records = []

    for i in range(96):
        interval_time = now - timedelta(minutes=15 * (95 - i))
        hour = (interval_time.hour - 6) % 24  # UTC → CST

        if 22 <= hour or hour <= 5:
            base_price = 20 + np.random.normal(0, 5)
        elif 6 <= hour <= 10:
            base_price = 38 + np.random.normal(0, 8)
        elif 14 <= hour <= 18:
            base_price = 60 + np.random.normal(0, 15)
        else:
            base_price = 35 + np.random.normal(0, 10)

        if np.random.random() < 0.05:
            base_price *= np.random.uniform(2, 5)

        price = round(max(5, base_price), 2)

        records.append({
            "timestamp": interval_time.strftime("%Y-%m-%d %H:%M:%S"),
            "settlement_point": "HB_NORTH",
            "price_usd_mwh": price,
            "source": "synthetic_ercot",
        })

    return records


# ═══════════════════════════════════════════════════════════
#  API ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """
    Health check endpoint.

    WHY EVERY SERVICE NEEDS THIS:
    - Docker HEALTHCHECK pings this to know if the container is alive
    - Cloud platforms (Render, AWS ECS, Kubernetes) use it for auto-restart
    - Load balancers use it to route traffic only to healthy instances
    - Your monitoring dashboard can track service uptime
    """
    return {
        "status": "healthy",
        "service": "data_ingestion",
        "version": "3.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "weather_source": "open_meteo",
        "market_source": "ercot",
        "weather_records_in_memory": len(weather_history),
        "ercot_records_in_memory": len(ercot_history),
        "dam_records_in_memory": len(dam_history),
    }


# ─── Weather Endpoints ───────────────────────────────────

@app.post("/weather/update")
def update_weather():
    """
    Fetch fresh weather data from Open-Meteo and store it.

    Called by cron every 15 minutes:
        curl -X POST http://data_service:8001/weather/update

    Flow:
        1. Call Open-Meteo API → get current Dallas weather + real irradiance
        2. Store in memory (latest_weather, weather_history)
        3. Save to CSV file for persistence
        4. Return the data
    """
    global latest_weather

    record = fetch_weather()
    latest_weather = record
    weather_history.append(record)

    logger.info(f"Weather updated: {record.get('temp')}°C, GHI={record.get('irradiance')}W/m²")
    return {"message": "Weather data updated", "data": record}


@app.get("/weather/latest")
def get_latest_weather():
    """
    Return the most recent weather reading.

    Used by:
    - Forecast service → input features for ML predictions
    - Dashboard → display current conditions
    """
    if not latest_weather:
        raise HTTPException(
            status_code=404,
            detail="No weather data available. Call POST /weather/update first."
        )
    return latest_weather


@app.get("/weather/history")
def get_weather_history(limit: int = 100):
    """Return recent weather history. Used by dashboard for trend charts."""
    return weather_history[-limit:]


@app.post("/weather/backfill")
def backfill_weather(hours: int = 24):
    """
    Backfill weather history from Open-Meteo.

    WHY THIS ENDPOINT?
        When the service first starts, weather_history is empty.
        Call this once at startup to load the last 24 hours of data.
        This way the dashboard and ML models have data immediately
        instead of waiting hours for enough data to accumulate.

    Example:
        curl -X POST "http://data_service:8001/weather/backfill?hours=48"
    """
    records = fetch_weather_history_from_api(hours=min(hours, 168))
    weather_history.extend(records)

    return {
        "message": f"Backfilled {len(records)} hours of weather data",
        "count": len(records),
    }


# ─── ERCOT Endpoints ─────────────────────────────────────

@app.post("/ercot/update")
def update_ercot():
    """
    Fetch latest ERCOT prices and store them.

    Called by cron every 15 minutes:
        curl -X POST http://data_service:8001/ercot/update
    """
    global latest_ercot

    records = fetch_ercot_prices()

    if records:
        latest_ercot = records[-1]
        ercot_history.extend(records)

        # Deduplicate by timestamp
        seen = set()
        unique = []
        for r in ercot_history:
            if r["timestamp"] not in seen:
                seen.add(r["timestamp"])
                unique.append(r)
        ercot_history.clear()
        ercot_history.extend(unique[-2000:])

    logger.info(f"ERCOT updated: {len(records)} records")
    return {
        "message": "ERCOT prices updated",
        "count": len(records),
        "latest_price": latest_ercot.get("price_usd_mwh", "N/A"),
    }


@app.get("/ercot/latest")
def get_latest_ercot():
    """Return the most recent ERCOT price."""
    if not latest_ercot:
        raise HTTPException(
            status_code=404,
            detail="No ERCOT data. Call POST /ercot/update first."
        )
    return latest_ercot


@app.get("/ercot/history")
def get_ercot_history(limit: int = 100):
    """Return recent ERCOT real-time price history."""
    return ercot_history[-limit:]


# ─── DAM (Day-Ahead Market) Endpoints ───────────────────

@app.post("/dam/update")
def update_dam(date: str = None):
    """
    Fetch latest Day-Ahead Market prices and store them.

    Called by scheduler once daily (after DAM results publish ~1:30pm CT):
        curl -X POST http://data_service:8001/dam/update

    Or fetch for a specific date:
        curl -X POST "http://data_service:8001/dam/update?date=2026-02-16"

    WHY ONCE A DAY?
        Unlike real-time prices (every 15 min), DAM prices are published
        once per day for the next day's 24 hours. So we only need to
        fetch once, typically in the afternoon when results are posted.
    """
    global latest_dam

    records = fetch_dam_prices(date_str=date)

    if records:
        latest_dam = records[-1]
        dam_history.extend(records)

        # Deduplicate by timestamp
        seen = set()
        unique = []
        for r in dam_history:
            if r["timestamp"] not in seen:
                seen.add(r["timestamp"])
                unique.append(r)
        dam_history.clear()
        dam_history.extend(unique[-500:])  # ~20 days of hourly data

    logger.info(f"DAM updated: {len(records)} hourly prices")
    return {
        "message": "DAM prices updated",
        "count": len(records),
        "latest_dam_price": latest_dam.get("dam_price_usd_mwh", "N/A"),
    }


@app.get("/dam/latest")
def get_latest_dam():
    """Return the most recent DAM price record."""
    if not latest_dam:
        raise HTTPException(
            status_code=404,
            detail="No DAM data. Call POST /dam/update first."
        )
    return latest_dam


@app.get("/dam/history")
def get_dam_history(limit: int = 100):
    """Return recent DAM price history."""
    return dam_history[-limit:]


@app.get("/dam/today")
def get_dam_today():
    """
    Return all 24 hourly DAM prices for today.

    Useful for the dashboard to show the full day-ahead price curve
    and for the trading service to compute RT-DAM spread at any hour.
    """
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")
    today_prices = [r for r in dam_history if r.get("delivery_date") == today]

    if not today_prices:
        raise HTTPException(
            status_code=404,
            detail="No DAM data for today. Call POST /dam/update first."
        )

    return {
        "date": today,
        "settlement_point": "HB_NORTH",
        "hours": sorted(today_prices, key=lambda x: int(x.get("hour_ending", 0))),
        "count": len(today_prices),
    }


@app.get("/spread/rt-dam")
def get_rt_dam_spread():
    """
    Calculate the Real-Time vs Day-Ahead price spread.

    THE MOST IMPORTANT TRADING SIGNAL:
        spread = RT price - DAM price

        spread > 0 → RT is MORE expensive than expected
                    → SELL signal (sell into expensive real-time market)

        spread < 0 → RT is CHEAPER than expected
                    → BUY signal (buy cheap real-time power)

        spread magnitude indicates strength of the signal:
        ±$5   → normal variation, weak signal
        ±$15  → moderate divergence, actionable signal
        ±$50+ → significant event, strong signal
    """
    rt_price = latest_ercot.get("price_usd_mwh") if latest_ercot else None
    dam_price = get_dam_price_for_hour()

    if rt_price is None:
        raise HTTPException(status_code=404, detail="No RT price available")

    if dam_price is None:
        return {
            "rt_price": rt_price,
            "dam_price": None,
            "spread": None,
            "spread_pct": None,
            "signal": "NEUTRAL",
            "signal_strength": "none",
            "note": "DAM data unavailable — call POST /dam/update",
        }

    spread = round(rt_price - dam_price, 2)
    spread_pct = round((spread / dam_price) * 100, 2) if dam_price != 0 else 0

    abs_spread = abs(spread)
    if abs_spread >= 50:
        strength = "strong"
    elif abs_spread >= 15:
        strength = "moderate"
    elif abs_spread >= 5:
        strength = "weak"
    else:
        strength = "negligible"

    if spread > 5:
        signal = "SELL"
    elif spread < -5:
        signal = "BUY"
    else:
        signal = "NEUTRAL"

    return {
        "rt_price": rt_price,
        "dam_price": dam_price,
        "spread": spread,
        "spread_pct": spread_pct,
        "signal": signal,
        "signal_strength": strength,
    }


# ─── Combined Endpoint ───────────────────────────────────

@app.get("/data/combined")
def get_combined_data():
    """
    Return weather + market data combined in one response.

    This is the PRIMARY endpoint the forecast service uses.
    One call gives it everything needed for ML prediction.

    Returns:
        {
            "timestamp": "2025-06-25T14:30",
            "temp": 35.2,
            "humidity": 45,
            "wind_speed": 5.8,
            "cloud_coverage": 20,
            "irradiance": 743.0,       ← REAL from Open-Meteo (GHI)
            "direct_radiation": 520.0,  ← REAL from Open-Meteo
            "dni": 890.0,              ← REAL from Open-Meteo
            "market_price": 52.30,     ← REAL from ERCOT (Real-Time SPP)
            "settlement_point": "HB_NORTH",
            "dam_price": 48.50,        ← REAL from ERCOT (Day-Ahead)
            "rt_dam_spread": 3.80      ← Calculated (RT - DAM)
        }
    """
    if not latest_weather:
        raise HTTPException(
            status_code=404,
            detail="No weather data. Call /weather/update first."
        )

    combined = {**latest_weather}

    if latest_ercot:
        combined["market_price"] = latest_ercot.get("price_usd_mwh", 0)
        combined["settlement_point"] = latest_ercot.get("settlement_point", "unknown")
    else:
        combined["market_price"] = None
        combined["settlement_point"] = "no_data"

    # Add DAM price and RT-DAM spread
    dam_price = get_dam_price_for_hour()
    combined["dam_price"] = dam_price

    rt_price = combined.get("market_price")
    if rt_price is not None and dam_price is not None:
        combined["rt_dam_spread"] = round(rt_price - dam_price, 2)
    else:
        combined["rt_dam_spread"] = None

    return combined


# ─── Data Persistence for Retraining ─────────────────────

@app.post("/data/save-training")
def save_training_data():
    """
    Save accumulated weather + ERCOT data to CSV for model retraining.

    Merges weather_history and ercot_history into a single training-ready CSV.
    Called periodically by the scheduler or manually before retraining.

    The CSV matches the format expected by the training script:
        datetime, market_price, temp, humidity, wind_speed, cloud_coverage,
        irradiance, direct_radiation, dni
    """
    import csv

    if len(weather_history) < 10 or len(ercot_history) < 10:
        return {
            "message": "Not enough data to save",
            "weather_records": len(weather_history),
            "ercot_records": len(ercot_history),
        }

    # Build lookup of ERCOT prices by rounded timestamp
    price_lookup = {}
    for r in ercot_history:
        ts = r.get("timestamp", "")
        price = r.get("price_usd_mwh", 0)
        if ts and price:
            # Normalize timestamp to hour for matching
            try:
                for fmt in ["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        dt = datetime.strptime(ts[:19], fmt)
                        hour_key = dt.strftime("%Y-%m-%d %H:00")
                        price_lookup[hour_key] = price
                        break
                    except ValueError:
                        continue
            except Exception:
                continue

    # Merge weather records with closest ERCOT price
    rows = []
    for w in weather_history:
        ts = w.get("timestamp", "")
        if not ts:
            continue
        try:
            for fmt in ["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                try:
                    dt = datetime.strptime(ts[:19], fmt)
                    break
                except ValueError:
                    continue
            else:
                continue

            hour_key = dt.strftime("%Y-%m-%d %H:00")
            price = price_lookup.get(hour_key)

            if price is not None and price > 0:
                rows.append({
                    "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "market_price": price,
                    "temp": w.get("temp", 0),
                    "humidity": w.get("humidity", 0),
                    "wind_speed": w.get("wind_speed", 0),
                    "cloud_coverage": w.get("cloud_coverage", 0),
                    "irradiance": w.get("irradiance", 0),
                    "direct_radiation": w.get("direct_radiation", 0),
                    "dni": w.get("dni", 0),
                })
        except Exception:
            continue

    if not rows:
        return {"message": "No matched weather-price records found", "count": 0}

    # Append to existing training data CSV (or create new)
    csv_path = "/app/data/collected_data.csv"
    os.makedirs("/app/data", exist_ok=True)

    file_exists = os.path.exists(csv_path)
    existing_timestamps = set()

    # Read existing timestamps to avoid duplicates
    if file_exists:
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_timestamps.add(row.get("datetime", ""))
        except Exception:
            pass

    # Filter out duplicates
    new_rows = [r for r in rows if r["datetime"] not in existing_timestamps]

    if not new_rows:
        return {"message": "No new records to save (all duplicates)", "count": 0}

    # Append new rows
    fieldnames = ["datetime", "market_price", "temp", "humidity", "wind_speed",
                  "cloud_coverage", "irradiance", "direct_radiation", "dni"]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(new_rows)

    total_rows = len(existing_timestamps) + len(new_rows)
    logger.info(f"Saved {len(new_rows)} new training records (total: {total_rows})")
    return {
        "message": f"Saved {len(new_rows)} new records",
        "new_records": len(new_rows),
        "total_records": total_rows,
        "csv_path": csv_path,
    }


@app.get("/data/training-stats")
def get_training_stats():
    """Return stats about collected training data."""
    csv_path = "/app/data/collected_data.csv"
    if not os.path.exists(csv_path):
        return {"exists": False, "total_records": 0}

    try:
        import csv
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            count = sum(1 for _ in reader)
        return {"exists": True, "total_records": count, "csv_path": csv_path}
    except Exception as e:
        return {"exists": True, "error": str(e)}