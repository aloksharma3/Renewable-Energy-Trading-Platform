FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_engine.py .
COPY energy_documents.py .
COPY news_fetcher.py .
COPY main.py .

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]