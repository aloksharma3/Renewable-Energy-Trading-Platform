# ML-Powered Renewable Energy Trading Platform

An end-to-end automated energy trading system that predicts ERCOT electricity prices using ensemble ML models and makes buy/sell decisions informed by RAG-based market intelligence.

> **Live Demo:** [Coming Soon]

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ML-Powered Energy Trading Platform                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Open-Meteo API ──→ ┌──────────────┐                               │
│   (Dallas weather)    │    Data      │                               │
│                       │   Service    │──→ Weather + ERCOT data       │
│   ERCOT API ────────→ │   :8001      │                               │
│   (market prices)     └──────┬───────┘                               │
│                              │                                       │
│                              ▼                                       │
│                       ┌──────────────┐    ┌──────────────┐          │
│                       │  Forecast    │    │     RAG      │          │
│                       │  Service     │    │   Service    │          │
│                       │  :8002       │    │   :8004      │          │
│                       │              │    │              │          │
│                       │ RF + XGBoost │    │ LangChain +  │          │
│                       │  Ensemble    │    │ FAISS +      │          │
│                       │              │    │ Gemini       │          │
│                       └──────┬───────┘    └──────┬───────┘          │
│                              │                   │                   │
│                              ▼                   ▼                   │
│                       ┌──────────────────────────────┐              │
│                       │      Trading Service         │              │
│                       │          :8003               │              │
│                       │                              │              │
│                       │  ML prediction + RAG risk    │              │
│                       │  → Adjusted thresholds       │              │
│                       │  → Confidence-based sizing   │              │
│                       │  → Position management       │              │
│                       │  → SQLite persistence        │              │
│                       └──────────────┬───────────────┘              │
│                                      │                               │
│                                      ▼                               │
│                       ┌──────────────────────────────┐              │
│                       │    Streamlit Dashboard        │              │
│                       │         :8501                │              │
│                       │                              │              │
│                       │  Price chart with CI bands   │              │
│                       │  Trade history + P&L         │              │
│                       │  RAG insights + Weather      │              │
│                       │  Model metrics + Importance  │              │
│                       └──────────────────────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

**Real-Time Data Ingestion**
- Live weather data from Open-Meteo (NOAA HRRR model) for Dallas, TX — real GHI, DNI, direct radiation
- Real ERCOT settlement point prices from HB_NORTH hub every 15 minutes
- Graceful fallback to synthetic data when APIs are unavailable

**Ensemble ML Forecasting**
- RandomForest + XGBoost averaged via VotingRegressor
- Bootstrap confidence intervals (30 iterations) for uncertainty quantification
- Permutation-based feature importance analysis
- Naive baseline comparison for model validation

**RAG Market Intelligence**
- LangChain orchestration with FAISS vector store
- Google Gemini embeddings + chat for document retrieval and synthesis
- Ingests real EIA energy articles via RSS and ERCOT grid conditions
- Returns structured risk assessments that adjust trading behavior

**Intelligent Trading Logic**
- RAG-adjusted thresholds: high upward risk lowers sell threshold
- Confidence-based position sizing: trade more when model is certain
- Full position tracking with weighted average cost basis
- Profit calculation, win rate tracking, trade audit trail
- SQLite persistence for trade history and portfolio state

**Production Architecture**
- 5 Docker microservices communicating via REST APIs
- Health check endpoints on every service
- Environment-based configuration for local and cloud deployment
- Graceful degradation: each service handles failures independently

---

## Tech Stack

| Layer | Technologies |
|---|---|
| ML Models | scikit-learn, XGBoost, RandomForest, VotingRegressor |
| RAG Pipeline | LangChain, FAISS, Google Gemini (embeddings + chat) |
| Backend APIs | FastAPI, Uvicorn |
| Database | SQLite |
| Data Sources | Open-Meteo API, ERCOT Public API, EIA RSS Feed |
| Frontend | Streamlit, Plotly |
| Infrastructure | Docker, Docker Compose |
| Language | Python 3.10 |

---

## Performance Metrics

| Metric | Value |
|---|---|
| Demand Forecast MAPE | ~12% |
| Price Forecast MAPE | ~14% |
| Energy Output MAPE | ~10% |
| Improvement Over Naive Baseline | 5-8 percentage points |
| Orchestration Interval | 15 minutes (aligned with ERCOT RTM) |
| Time Intervals Tested | 1,500+ |

---

## Project Structure

```
Renewable_AI_Trading/
├── data_service/                  # Weather + ERCOT data ingestion
│   ├── main.py                    # Open-Meteo + ERCOT API endpoints
│   ├── Dockerfile
│   └── requirements.txt
│
├── forecast_service/              # Ensemble ML predictions
│   ├── main.py                    # FastAPI service with prediction endpoints
│   ├── ensemble_forecaster.py     # RF + XGBoost + confidence intervals
│   ├── Dockerfile
│   └── requirements.txt
│
├── rag_service/                   # RAG market intelligence
│   ├── main.py                    # FastAPI endpoints for analysis
│   ├── rag_engine.py              # LangChain + FAISS + Gemini pipeline
│   ├── energy_documents.py        # ERCOT domain knowledge base
│   ├── news_fetcher.py            # Real EIA + ERCOT news ingestion
│   ├── Dockerfile
│   └── requirements.txt
│
├── trading_service/               # Automated trading decisions
│   ├── main.py                    # Trading logic with RAG integration
│   ├── database.py                # SQLite for trades + positions
│   ├── Dockerfile
│   └── requirements.txt
│
├── dashboard/                     # Streamlit visualization
│   ├── app.py                     # Full dashboard with charts + metrics
│   ├── Dockerfile
│   └── requirements.txt
│
├── src/
│   └── train_ensemble.py          # Train RF + XGBoost ensemble models
│
├── models/                        # Trained model files (.joblib)
├── data/                          # Training data + SQLite database
├── docker-compose.yml             # Orchestrates all services
└── .env.example                   # Environment variables template
```

---

## Quick Start

**Prerequisites:** Docker, Docker Compose, Python 3.10+

```bash
# 1. Clone the repo
git clone https://github.com/aloksharma3/Renewable-Energy-Trading-Platform.git
cd Renewable-Energy-Trading-Platform/Renewable_AI_Trading

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (free from https://aistudio.google.com/apikey)

# 3. Train the ML models
pip install scikit-learn xgboost joblib pandas numpy
python src/train_ensemble.py

# 4. Start all services
docker-compose up --build

# 5. Open the dashboard
# http://localhost:8501
```

---

## How Trading Decisions Are Made

```
Every 15 minutes:

1. DATA SERVICE fetches:
   → Dallas weather from Open-Meteo (temp, wind, GHI, DNI)
   → ERCOT HB_NORTH settlement point price

2. FORECAST SERVICE predicts:
   → Energy output, price, and demand using RF + XGBoost ensemble
   → Returns confidence intervals via bootstrap sampling

3. RAG SERVICE analyzes:
   → Searches FAISS for relevant market documents
   → Generates risk assessment via Gemini
   → Returns risk_score (0-1) and price_direction (up/down/stable)

4. TRADING SERVICE decides:
   → Adjusts thresholds: high upward risk → lower sell threshold
   → Sizes trade: tight confidence → trade more
   → Checks position limits: can't sell more than held
   → Executes trade and logs to SQLite with full reasoning
```

---

## Environment Variables

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key    # Free from Google AI Studio

# Optional (defaults shown)
LAT=32.7767                           # Dallas, TX
LON=-96.7970
SELL_THRESHOLD=55                     # Base sell threshold ($/MWh)
BUY_THRESHOLD=30                      # Base buy threshold ($/MWh)
MAX_POSITION_MWH=500                  # Maximum position size
LOG_LEVEL=INFO
```

---

## Why These Design Decisions

| Decision | Reasoning |
|---|---|
| ERCOT HB_NORTH | Highest demand zone in Texas, matches Dallas weather location |
| 15-minute intervals | Aligned with ERCOT Real-Time Market settlement periods |
| Open-Meteo over OpenWeatherMap | Real GHI/DNI solar irradiance, free, no API key |
| Ensemble over single model | Errors cancel out, more robust predictions |
| RAG over static rules | Markets driven by events ML can't see in numbers |
| SQLite over PostgreSQL | Zero-setup for MVP, same interface for production swap |
| Microservices over monolith | Independent scaling, fault isolation, production-realistic |

---

## Author

**Alok Sharma**
- M.S. Applied Machine Intelligence, Northeastern University
- 5+ years in SCADA/EMS systems and industrial automation at Schneider Electric, L&T, NCDEX
- [LinkedIn](https://linkedin.com/in/aloksharma28) | [GitHub](https://github.com/aloksharma3)