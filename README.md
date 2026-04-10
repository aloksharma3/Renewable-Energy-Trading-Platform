---
title: ML Energy Trading Platform
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
---

# ML-Powered Renewable Energy Trading Platform

An end-to-end automated energy trading system that predicts ERCOT electricity prices using ensemble ML models and makes buy/sell decisions informed by RAG-based market intelligence.

## Architecture

6 microservices running in a single Docker container:

| Service | Port | Role |
|---------|------|------|
| Data Service | 8001 | Real ERCOT + weather data ingestion |
| Forecast Service | 8002 | RF + XGBoost ensemble predictions |
| Trading Service | 8003 | Automated BUY/SELL/HOLD decisions |
| RAG Service | 8004 | LangChain + FAISS + Gemini intelligence |
| Dashboard | 8501 | Streamlit UI (exposed via Nginx on 7860) |
| Scheduler | — | 15-min pipeline orchestration |

## Tech Stack

**ML:** scikit-learn, XGBoost, VotingRegressor, bootstrap confidence intervals  
**RAG:** LangChain, FAISS, Google Gemini embeddings + chat  
**Backend:** FastAPI, SQLite persistence, Nginx reverse proxy  
**Infra:** Docker, Supervisor, GitHub Actions CI/CD  
**Data:** ERCOT RTD-LMP (HB_NORTH), Open-Meteo weather API  

## API Endpoints

Once running, API endpoints are available at:
- `/api/data/health` — Data service status
- `/api/forecast/health` — Forecast service status
- `/api/trading/health` — Trading service status + position
- `/api/rag/health` — RAG engine status

## Source Code

[GitHub Repository](https://github.com/aloksharma3/Renewable-Energy-Trading-Platform)
