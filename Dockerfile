# ============================================================
# Unified Dockerfile for Hugging Face Spaces
# ============================================================
# HF Spaces runs ONE container with ONE exposed port (7860).
# This Dockerfile packs all 6 services + Nginx reverse proxy
# into a single image using Supervisor as the process manager.
#
# Internal layout:
#   Nginx    :7860  → reverse proxy (the only port HF exposes)
#   data     :8001  → weather + ERCOT data
#   forecast :8002  → ML ensemble predictions
#   trading  :8003  → automated trading decisions
#   rag      :8004  → LangChain + FAISS + Gemini
#   dashboard:8501  → Streamlit UI (proxied as default route)
#   scheduler       → background pipeline (no port)
# ============================================================

FROM python:3.10-slim

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        nginx \
        supervisor \
        curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- HF Spaces requires user with UID 1000 ---
RUN useradd -m -u 1000 user

# --- Create application directories ---
RUN mkdir -p /app/data_service \
             /app/forecast_service \
             /app/trading_service \
             /app/rag_service \
             /app/scheduler \
             /app/dashboard \
             /app/models \
             /app/data/training \
             /app/data/trading \
             /app/data/forecast \
    && chown -R user:user /app

WORKDIR /app

# --- Install all Python dependencies in one layer ---
COPY requirements-unified.txt /app/requirements-unified.txt
RUN pip install --no-cache-dir -r /app/requirements-unified.txt

# --- Copy service code ---
COPY Renewable_AI_Trading/data_service/main.py         /app/data_service/
COPY Renewable_AI_Trading/forecast_service/main.py     /app/forecast_service/
COPY Renewable_AI_Trading/forecast_service/ensemble_forecaster.py /app/forecast_service/
COPY Renewable_AI_Trading/forecast_service/database.py /app/forecast_service/
COPY Renewable_AI_Trading/trading_service/main.py      /app/trading_service/
COPY Renewable_AI_Trading/trading_service/database.py  /app/trading_service/
COPY Renewable_AI_Trading/rag_service/main.py          /app/rag_service/
COPY Renewable_AI_Trading/rag_service/rag_engine.py    /app/rag_service/
COPY Renewable_AI_Trading/rag_service/energy_documents.py /app/rag_service/
COPY Renewable_AI_Trading/rag_service/news_fetcher.py  /app/rag_service/
COPY Renewable_AI_Trading/scheduler/main.py            /app/scheduler/
COPY Renewable_AI_Trading/dashboard/app.py             /app/dashboard/
COPY Renewable_AI_Trading/dashboard/backtest_engine.py /app/dashboard/

# --- Copy pre-trained models ---
COPY Renewable_AI_Trading/models/ /app/models/

# --- Copy training data if present ---
COPY Renewable_AI_Trading/data/ /app/data/training/

# --- Nginx config: single port 7860 → internal services ---
COPY nginx.conf /etc/nginx/nginx.conf

# --- Supervisor config: manages all processes ---
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# --- Startup script ---
COPY start.sh /app/start.sh

# --- Fix permissions ---
RUN chown -R user:user /app \
    && chmod -R 755 /app \
    && chmod +x /app/start.sh \
    && mkdir -p /var/log/supervisor /var/log/nginx /var/lib/nginx/body \
    && chown -R user:user /var/log/supervisor /var/log/nginx /var/lib/nginx \
    && chown -R user:user /run

# --- Environment variables (defaults — secrets injected by HF) ---
ENV DATA_SERVICE_URL=http://localhost:8001 \
    FORECAST_SERVICE_URL=http://localhost:8002 \
    TRADING_SERVICE_URL=http://localhost:8003 \
    RAG_SERVICE_URL=http://localhost:8004 \
    MODEL_DIR=/app/models \
    DB_PATH=/app/data/trading/trading.db \
    FORECAST_DB_PATH=/app/data/forecast/forecast.db \
    PYTHONUNBUFFERED=1

EXPOSE 7860

USER user

CMD ["bash", "/app/start.sh"]
