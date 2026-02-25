"""
Pipeline Scheduler
===================
Triggers the full trading pipeline every 15 minutes.

WHY A SEPARATE SCHEDULER?
    Each service does ONE thing. The scheduler's job is orchestration —
    calling services in the right order at the right time.

PIPELINE ORDER (every 15 minutes):
    1. POST data_service/weather/update     → fetch fresh Dallas weather
    2. POST data_service/ercot/update       → fetch fresh ERCOT prices
    3. POST forecast/forecast/run           → run ML predictions
    4. POST trading/trade/execute           → make trading decision
    5. POST rag/refresh-news               → refresh market news (every 6 hours)

WHY THIS ORDER?
    - Weather + ERCOT must be fetched BEFORE forecast runs (forecast needs fresh data)
    - Forecast must run BEFORE trading (trading needs predictions)
    - RAG news refreshes less frequently (EIA publishes a few times per week)

ON STARTUP:
    - Triggers /weather/backfill to load 24 hours of weather history
    - Runs the full pipeline once immediately
    - Then schedules it every 15 minutes

REPLACES: old forecast_cron service (which used raw cron + curl)
IMPROVEMENT: Python-based, has logging, handles errors, runs in correct order
"""

import os
import time
import logging
import requests
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

# ─── Setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("scheduler")

DATA_URL = os.getenv("DATA_SERVICE_URL", "http://data_service:8001")
FORECAST_URL = os.getenv("FORECAST_SERVICE_URL", "http://forecast:8002")
TRADING_URL = os.getenv("TRADING_SERVICE_URL", "http://trading:8003")
RAG_URL = os.getenv("RAG_SERVICE_URL", "http://rag:8004")


def call_service(method, url, name, timeout=30):
    """
    Call a service endpoint with error handling.

    If a service is down, log the error and continue.
    The pipeline should not stop because one service failed.
    """
    try:
        if method == "POST":
            r = requests.post(url, timeout=timeout)
        else:
            r = requests.get(url, timeout=timeout)

        if r.status_code == 200:
            logger.info(f"✅ {name}: OK")
            return r.json()
        else:
            logger.warning(f"⚠️ {name}: HTTP {r.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        logger.error(f"❌ {name}: Service not reachable")
        return None
    except requests.exceptions.Timeout:
        logger.error(f"❌ {name}: Timeout after {timeout}s")
        return None
    except Exception as e:
        logger.error(f"❌ {name}: {e}")
        return None


def run_pipeline():
    """
    Execute the full trading pipeline.

    Called every 15 minutes by APScheduler.
    Each step depends on the previous one, but failures don't stop the chain.
    """
    logger.info("=" * 50)
    logger.info(f"Pipeline started at {datetime.utcnow().isoformat()}")
    logger.info("=" * 50)

    # Step 1: Fetch fresh weather
    call_service("POST", f"{DATA_URL}/weather/update", "Weather Update")

    # Step 2: Fetch fresh ERCOT prices
    call_service("POST", f"{DATA_URL}/ercot/update", "ERCOT Update")

    # Step 3: Run ML forecast
    call_service("POST", f"{FORECAST_URL}/forecast/run", "ML Forecast")

    # Step 4: Execute trading decision
    result = call_service("POST", f"{TRADING_URL}/trade/execute", "Trading Decision")
    if result:
        logger.info(
            f"Trade: {result.get('action')} {result.get('quantity', 0)}MWh "
            f"at ${result.get('price', 0)} | Profit: ${result.get('profit', 0)}"
        )

    logger.info("Pipeline complete")
    logger.info("=" * 50)


def refresh_news():
    """
    Refresh RAG knowledge base with latest news.

    Called every 6 hours — EIA publishes a few times per week,
    so checking every 15 minutes is wasteful.
    """
    logger.info("Refreshing RAG news...")
    call_service("POST", f"{RAG_URL}/refresh-news", "RAG News Refresh", timeout=60)


def startup():
    """
    Run once on startup.

    1. Wait for services to be ready (they take a few seconds to start)
    2. Backfill 24 hours of weather history
    3. Run the pipeline immediately (don't wait 15 minutes for first run)
    """
    logger.info("Scheduler starting — waiting for services...")

    # Wait for services to be ready
    for i in range(30):
        try:
            r = requests.get(f"{DATA_URL}/health", timeout=5)
            if r.status_code == 200:
                logger.info("Services are ready")
                break
        except Exception:
            pass
        logger.info(f"Waiting for services... ({i+1}/30)")
        time.sleep(5)

    # Backfill weather history
    logger.info("Backfilling 24h of weather history...")
    call_service("POST", f"{DATA_URL}/weather/backfill?hours=24", "Weather Backfill")

    # Run pipeline immediately
    logger.info("Running initial pipeline...")
    run_pipeline()

    # Refresh news on startup
    refresh_news()


def save_training_data():
    """
    Save accumulated weather + ERCOT data to CSV for model retraining.
    Called every 6 hours to persist collected data.
    """
    logger.info("Saving training data...")
    call_service("POST", f"{DATA_URL}/data/save-training", "Save Training Data")


def retrain_models():
    """
    Retrain all ML models on accumulated real data.

    Called weekly (Sunday 3 AM CT / 9 AM UTC).
    First saves latest data, then triggers retraining.

    WHY WEEKLY?
        - Enough new data accumulates in a week (672 data points at 15-min intervals)
        - Retraining is compute-intensive, don't want to do it too often
        - Weekly captures different day patterns (weekday vs weekend)
    """
    logger.info("=" * 50)
    logger.info("Starting weekly model retraining...")
    logger.info("=" * 50)

    # First save any unsaved data
    call_service("POST", f"{DATA_URL}/data/save-training", "Save Training Data (pre-retrain)")

    # Then retrain
    result = call_service("POST", f"{FORECAST_URL}/models/retrain", "Model Retraining", timeout=120)
    if result:
        logger.info(f"Retraining result: {result.get('message', 'unknown')}")
        for name, info in result.get("models", {}).items():
            if info.get("status") == "success":
                mape = info.get("metrics", {}).get("ensemble_mape", "N/A")
                logger.info(f"  {name}: MAPE={mape}%")
            else:
                logger.warning(f"  {name}: {info.get('error', 'failed')}")

    logger.info("=" * 50)


if __name__ == "__main__":
    # Run startup tasks
    startup()

    # Schedule recurring jobs
    scheduler = BlockingScheduler()

    # Main pipeline: every 15 minutes
    scheduler.add_job(run_pipeline, "interval", minutes=15, id="pipeline")

    # Save training data: every 6 hours
    scheduler.add_job(save_training_data, "interval", hours=6, id="save_data")

    # Retrain models: weekly on Sunday at 3 AM CT (9 AM UTC)
    scheduler.add_job(retrain_models, "cron", day_of_week="sun", hour=9, minute=0, id="retrain")

    # News refresh: every 6 hours
    scheduler.add_job(refresh_news, "interval", hours=6, id="news_refresh")

    logger.info(
        "Scheduler running — pipeline every 15 min, "
        "data save every 6h, retrain weekly Sun 3AM CT, "
        "news every 6h"
    )
    scheduler.start()