"""
RAG Market Intelligence Service
=================================
Provides market intelligence by analyzing energy documents using RAG.

WHAT THIS SERVICE DOES:
    1. On startup: loads energy knowledge base into FAISS vector store
    2. On /ingest: accepts new documents (ERCOT notices, news, reports)
    3. On /analyze: retrieves relevant docs + generates risk assessment via Gemini
    4. On /search: finds relevant document chunks without LLM generation

HOW THE TRADING SERVICE USES THIS:
    Every 15 minutes, trading service calls POST /analyze with:
        "Assess current market risks for ERCOT HB_NORTH"
    
    RAG returns:
        {"risk_score": 0.85, "direction": "up", "factors": [...]}
    
    Trading service adjusts buy/sell thresholds based on risk_score.

ENDPOINTS:
    GET  /health              → service status
    POST /analyze             → full RAG analysis (retrieve + generate)
    POST /search              → search documents without LLM
    POST /ingest              → add new documents to knowledge base
    GET  /status              → document count, engine status
"""

import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from rag_engine import RAGEngine
from energy_documents import ENERGY_DOCUMENTS
from news_fetcher import fetch_all_news

# ─── Setup ──────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_service")

app = FastAPI(
    title="RAG Market Intelligence Service",
    description="Energy market analysis using Retrieval-Augmented Generation",
    version="2.0.0",
)

# ─── Initialize RAG Engine ──────────────────────────────────
engine = RAGEngine()

# Load knowledge base on startup
def load_knowledge_base():
    """
    Load pre-written energy documents into the vector store.

    These documents contain knowledge about:
        - ERCOT conservation alerts and their price impact
        - Generator outages and supply-demand dynamics
        - Texas wind and solar energy patterns
        - Natural gas price relationships
        - Demand patterns and weather impacts
        - Market structure (DAM vs RTM)

    This gives the RAG engine domain knowledge to draw from
    even before any real-time news is ingested.
    """
    if not engine.enabled:
        logger.warning("RAG engine not enabled — skipping knowledge base load")
        return

    texts = [doc["text"] for doc in ENERGY_DOCUMENTS]
    metadata = [doc["metadata"] for doc in ENERGY_DOCUMENTS]
    chunks = engine.add_documents(texts, metadata)
    logger.info(f"Loaded {len(ENERGY_DOCUMENTS)} knowledge base documents ({chunks} chunks)")

load_knowledge_base()


def load_real_news():
    """
    Fetch real energy news from EIA and ERCOT on startup.

    This gives the RAG engine current real-world context:
        - EIA: latest energy market articles and analysis
        - ERCOT: current grid conditions and stress levels

    Called once on startup. Can also be triggered via POST /refresh-news.
    """
    if not engine.enabled:
        return

    texts, metadata = fetch_all_news()
    if texts:
        chunks = engine.add_documents(texts, metadata)
        logger.info(f"Loaded {len(texts)} real news articles ({chunks} chunks)")
    else:
        logger.warning("No real news fetched — RAG will use knowledge base only")

load_real_news()


# ─── Request/Response Models ────────────────────────────────
class AnalyzeRequest(BaseModel):
    """Request body for /analyze endpoint."""
    query: str = "What factors might affect ERCOT electricity prices in the next few hours?"

class SearchRequest(BaseModel):
    """Request body for /search endpoint."""
    query: str
    k: int = 5

class IngestRequest(BaseModel):
    """Request body for /ingest endpoint."""
    texts: list[str]
    sources: Optional[list[str]] = None


# ─── API Endpoints ──────────────────────────────────────────

@app.get("/health")
def health():
    """Health check with RAG engine status."""
    status = engine.get_status()
    return {
        "status": "healthy",
        "service": "rag_intelligence",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "rag_enabled": status["enabled"],
        "documents_loaded": status["documents_loaded"],
        "chunks_in_store": status["chunks_in_store"],
    }


@app.post("/analyze")
def analyze_market(request: AnalyzeRequest):
    """
    Full RAG analysis: retrieve relevant documents + generate risk assessment.

    This is the PRIMARY endpoint the trading service calls.

    Flow:
        1. Take the query (e.g., "What risks affect ERCOT prices?")
        2. Embed the query into a vector
        3. FAISS finds the 5 most relevant document chunks
        4. Send chunks + query to Gemini
        5. Gemini returns structured risk assessment

    Request:
        {"query": "What factors might affect ERCOT prices?"}

    Response:
        {
            "risk_score": 0.85,
            "risk_level": "high",
            "price_direction": "up",
            "factors": ["conservation alert", "generator outage", "heat wave"],
            "summary": "Multiple supply-demand stress indicators...",
            "sources": [{"content": "...", "metadata": {...}}],
            "timestamp": "2025-06-25T14:30:00"
        }
    """
    result = engine.analyze_market(request.query)
    return result


@app.post("/search")
def search_documents(request: SearchRequest):
    """
    Search for relevant document chunks WITHOUT generating an LLM response.

    Useful for:
        - Debugging: see what documents the RAG retrieves
        - Dashboard: display relevant excerpts to the user
        - Cost saving: doesn't use Gemini API quota

    Request:
        {"query": "wind energy output", "k": 3}

    Response:
        {
            "query": "wind energy output",
            "results": [
                {"content": "Texas leads the US in wind...", "metadata": {...}},
                ...
            ],
            "count": 3
        }
    """
    results = engine.search(request.query, k=request.k)
    return {
        "query": request.query,
        "results": results,
        "count": len(results),
    }


@app.post("/ingest")
def ingest_documents(request: IngestRequest):
    """
    Add new documents to the RAG knowledge base.

    Use this to feed in:
        - ERCOT market notices
        - EIA weekly reports
        - Energy news articles
        - Your own analysis documents

    The documents are split into chunks, embedded, and stored in FAISS.
    They become searchable immediately.

    Request:
        {
            "texts": [
                "ERCOT issued a conservation alert today...",
                "Natural gas prices surged 15% this week..."
            ],
            "sources": ["ercot_notice", "eia_report"]
        }

    Response:
        {"message": "Ingested 2 documents (12 chunks)", "chunks": 12}
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    metadata_list = None
    if request.sources:
        metadata_list = [{"source": s} for s in request.sources]

    chunks = engine.add_documents(request.texts, metadata_list)
    return {
        "message": f"Ingested {len(request.texts)} documents ({chunks} chunks)",
        "documents": len(request.texts),
        "chunks": chunks,
    }


@app.post("/refresh-news")
def refresh_news():
    """
    Fetch latest energy news and add to knowledge base.

    Called by the scheduler every few hours to keep RAG updated
    with real-world information.

    Flow:
        1. Fetch EIA RSS feed (latest energy articles)
        2. Fetch ERCOT grid conditions (current grid stress)
        3. Embed and add to FAISS vector store
        4. Next /analyze call will use this fresh context
    """
    texts, metadata = fetch_all_news()
    if not texts:
        return {"message": "No new articles found", "count": 0}

    chunks = engine.add_documents(texts, metadata)
    return {
        "message": f"Refreshed with {len(texts)} articles ({chunks} chunks)",
        "articles": len(texts),
        "chunks": chunks,
    }


@app.get("/status")
def get_status():
    """Return detailed RAG engine status."""
    return engine.get_status()