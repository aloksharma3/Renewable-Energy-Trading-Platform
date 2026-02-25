# Base image: Python 3.10 (slim = smaller image, no extra system packages)
FROM python:3.10-slim

# Set the working directory inside the container
# All subsequent commands run from /app
WORKDIR /app

# Install curl for health checks (Docker uses this to check if service is alive)
RUN apt-get update && apt-get install -y curl && apt-get clean

# Copy requirements first (Docker caches this layer — if requirements don't change,
# it skips reinstalling packages on rebuild, saving time)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual application code
COPY . .

# Tell Docker this container listens on port 8001
EXPOSE 8001

# Health check: Docker pings /health every 30 seconds
# If it fails 3 times in a row, Docker marks the container as unhealthy
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Start the FastAPI server using uvicorn
# --host 0.0.0.0 means "listen on all network interfaces" (required in Docker)
# --port 8001 matches the EXPOSE above
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]