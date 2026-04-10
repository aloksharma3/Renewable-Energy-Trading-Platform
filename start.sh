#!/bin/bash
# ============================================================
# Startup script for Hugging Face Spaces
# ============================================================
# HF injects secrets as environment variables at container runtime.
# Supervisor child processes inherit env from this script.
# This wrapper ensures all secrets are available to all services.
# ============================================================

# Default env vars for inter-service communication
export DATA_SERVICE_URL="${DATA_SERVICE_URL:-http://localhost:8001}"
export FORECAST_SERVICE_URL="${FORECAST_SERVICE_URL:-http://localhost:8002}"
export TRADING_SERVICE_URL="${TRADING_SERVICE_URL:-http://localhost:8003}"
export RAG_SERVICE_URL="${RAG_SERVICE_URL:-http://localhost:8004}"
export MODEL_DIR="${MODEL_DIR:-/app/models}"
export DB_PATH="${DB_PATH:-/app/data/trading/trading.db}"
export FORECAST_DB_PATH="${FORECAST_DB_PATH:-/app/data/forecast/forecast.db}"

# Ensure data directories exist and are writable
mkdir -p /app/data/trading /app/data/forecast /app/data/training

echo "=============================================="
echo "  ML Energy Trading Platform — HF Spaces"
echo "=============================================="
echo "  Data service:     $DATA_SERVICE_URL"
echo "  Forecast service: $FORECAST_SERVICE_URL"
echo "  Trading service:  $TRADING_SERVICE_URL"
echo "  RAG service:      $RAG_SERVICE_URL"
echo "  ERCOT API:        $([ -n "$ERCOT_SUBSCRIPTION_KEY" ] && echo 'configured' || echo 'not set')"
echo "  Gemini API:       $([ -n "$GEMINI_API_KEY" ] && echo 'configured' || echo 'not set')"
echo "=============================================="

exec supervisord -c /etc/supervisor/conf.d/supervisord.conf
