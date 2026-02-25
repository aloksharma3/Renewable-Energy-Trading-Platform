"""
RAG Engine
===========
The core retrieval-augmented generation pipeline.

PIPELINE:
    1. LOAD documents (text, URLs)
    2. SPLIT into chunks (~500 words each, 100 word overlap)
    3. EMBED using Google's gemini-embedding-001 model (free)
    4. STORE in FAISS vector database (in-memory, fast search)
    5. QUERY: question → find similar chunks → send to Gemini → get answer

LIBRARIES:
    LangChain:              Orchestrates the pipeline (load → split → embed → query)
    FAISS:                  Facebook's vector similarity search (finds relevant chunks)
    langchain-google-genai: Connects LangChain to Gemini for embeddings and chat
    
WHY GEMINI?
    - Free tier: 500 requests/day, 15 requests/min
    - Our project uses ~96 requests/day (every 15 min)
    - Gemini embedding model is also free
    - Quality is good enough for summarization + risk classification
"""

import os
import json
import logging
import time
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger("rag_engine")

# Cache duration in seconds (15 minutes matches the pipeline interval)
CACHE_TTL = int(os.getenv("RAG_CACHE_TTL_SECONDS", "900"))


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for energy market intelligence.
    
    Usage:
        engine = RAGEngine()
        engine.add_documents(["ERCOT issued conservation alert...", ...])
        result = engine.analyze_market("What risks affect prices?")
    """

    def __init__(self):
        """
        Initialize the RAG engine with Gemini models.

        Two models are used:
            1. Embedding model (gemini-embedding-001):
               Converts text → vector (list of numbers)
               Used for: storing and searching documents
               
            2. Chat model (gemini-2.0-flash):
               Takes text input → generates text output
               Used for: synthesizing answers from retrieved chunks
        """
        api_key = os.getenv("GEMINI_API_KEY", "")

        if not api_key:
            logger.warning("No GEMINI_API_KEY set — RAG will run in fallback mode")
            self.enabled = False
            return

        # Embedding model: converts text to 768-dimensional vectors
        # "models/gemini-embedding-001" is free with no usage limits
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )

        # Chat model: generates answers from retrieved context
        # "gemini-1.5-flash" is fast and free (500 req/day)
        # temperature=0.3 means less creative, more factual answers
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0.3,
        )

        # Text splitter: breaks documents into searchable chunks
        # chunk_size=500: each chunk is ~500 characters
        # chunk_overlap=100: chunks share 100 chars at boundaries
        #   so context isn't lost between chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # FAISS vector store: initialized when first documents are added
        self.vectorstore = None
        self.enabled = True
        self.document_count = 0

        # Cache: stores last analysis result to avoid redundant Gemini calls.
        # The trading service and dashboard can both call /analyze, but
        # Gemini only gets hit once per CACHE_TTL (default 15 minutes).
        self._cache = {}          # {query_key: result_dict}
        self._cache_time = {}     # {query_key: timestamp}

        logger.info("RAG engine initialized with Gemini models")

    def add_documents(self, texts, metadata_list=None):
        """
        Add documents to the vector store.

        Args:
            texts: List of strings (document content)
                   ["ERCOT alert: conservation...", "EIA report: gas prices..."]
            metadata_list: Optional list of dicts with source info
                   [{"source": "ercot", "date": "2025-06-25"}, ...]

        What happens:
            1. Each text is split into chunks
            2. Each chunk is embedded (text → vector)
            3. Vectors are stored in FAISS for fast search
        """
        if not self.enabled:
            logger.warning("RAG not enabled (no API key)")
            return 0

        # Create Document objects (LangChain's standard format)
        documents = []
        for i, text in enumerate(texts):
            metadata = metadata_list[i] if metadata_list else {}
            metadata["added_at"] = datetime.utcnow().isoformat()
            documents.append(Document(page_content=text, metadata=metadata))

        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(texts)} documents into {len(chunks)} chunks")

        # Add to FAISS vector store
        if self.vectorstore is None:
            # First time: create new vector store from chunks
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        else:
            # Subsequent times: add to existing store
            self.vectorstore.add_documents(chunks)

        self.document_count += len(texts)
        logger.info(f"Added {len(texts)} documents ({len(chunks)} chunks) to vector store")
        return len(chunks)

    def search(self, query, k=5):
        """
        Search for documents most relevant to a query.

        Args:
            query: The question to search for
            k: Number of results to return (default 5)

        How it works:
            1. Convert query to vector using same embedding model
            2. FAISS finds the k vectors most similar to query vector
            3. Return the original text chunks for those vectors

        This is the "retrieval" part of RAG.
        """
        if not self.enabled or self.vectorstore is None:
            return []

        results = self.vectorstore.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in results
        ]

    def analyze_market(self, query="What factors might affect ERCOT electricity prices?"):
        """
        Full RAG pipeline: retrieve relevant chunks + generate analysis.

        This is the main method the trading service calls.

        Args:
            query: Market question to answer

        Returns:
            {
                "risk_score": 0.85,
                "risk_level": "high",
                "price_direction": "up",
                "factors": ["conservation alert", "generator outage"],
                "summary": "Multiple indicators suggest...",
                "sources": [{"content": "...", "metadata": {...}}],
                "timestamp": "2025-06-25T14:30:00"
            }
        """
        if not self.enabled:
            return self._fallback_assessment()

        if self.vectorstore is None:
            logger.warning("No documents in vector store")
            return self._fallback_assessment()

        # Check cache: return previous result if still fresh.
        # This prevents burning Gemini quota when multiple callers
        # (trading service + dashboard) hit /analyze within the same cycle.
        cache_key = query[:100]  # Normalize similar queries
        now = time.time()
        if cache_key in self._cache:
            age = now - self._cache_time[cache_key]
            if age < CACHE_TTL:
                logger.info(f"RAG cache hit (age={age:.0f}s, ttl={CACHE_TTL}s)")
                return self._cache[cache_key]

        # Step 1: Retrieve relevant chunks
        relevant_docs = self.vectorstore.similarity_search(query, k=5)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Step 2: Send context + question to Gemini
        prompt = f"""You are an energy market analyst specializing in ERCOT (Texas electricity market).

Based ONLY on the following document excerpts, provide a market risk assessment.

DOCUMENTS:
{context}

QUESTION: {query}

Respond in this exact JSON format (no markdown, no code blocks, just raw JSON):
{{
    "risk_score": <float 0.0 to 1.0, where 1.0 is highest risk of price spike>,
    "risk_level": "<low/medium/high>",
    "price_direction": "<up/down/stable>",
    "factors": ["<factor 1>", "<factor 2>", "<factor 3>"],
    "summary": "<2-3 sentence analysis>"
}}"""

        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()

            # Clean response: remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("\n", 1)[1]
                response_text = response_text.rsplit("```", 1)[0]

            result = json.loads(response_text)

            # Add metadata
            result["sources"] = [
                {"content": doc.page_content[:200], "metadata": doc.metadata}
                for doc in relevant_docs
            ]
            result["timestamp"] = datetime.utcnow().isoformat()
            result["query"] = query

            logger.info(
                f"RAG analysis: risk={result['risk_score']}, "
                f"direction={result['price_direction']}"
            )

            # Cache the result so subsequent calls within CACHE_TTL
            # don't hit Gemini again
            self._cache[cache_key] = result
            self._cache_time[cache_key] = time.time()

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {response_text}")
            return self._fallback_assessment()

        except Exception as e:
            logger.error(f"RAG analysis failed: {e}")
            return self._fallback_assessment()

    def _fallback_assessment(self):
        """
        Return a neutral assessment when RAG is unavailable.

        This ensures the trading service always gets a response,
        even if Gemini API is down or no documents are loaded.
        A neutral assessment means trading uses ML predictions only
        without any RAG adjustment.
        """
        return {
            "risk_score": 0.5,
            "risk_level": "medium",
            "price_direction": "stable",
            "factors": ["RAG analysis unavailable"],
            "summary": "Unable to perform market analysis. Using default risk assessment.",
            "sources": [],
            "timestamp": datetime.utcnow().isoformat(),
            "fallback": True,
        }

    def get_status(self):
        """Return current status of the RAG engine."""
        return {
            "enabled": self.enabled,
            "documents_loaded": self.document_count,
            "chunks_in_store": (
                self.vectorstore.index.ntotal
                if self.vectorstore is not None
                else 0
            ),
        }