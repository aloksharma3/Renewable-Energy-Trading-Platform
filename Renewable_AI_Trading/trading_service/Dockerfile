FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY database.py .
COPY main.py .

# Create data directory for SQLite database
RUN mkdir -p data

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]