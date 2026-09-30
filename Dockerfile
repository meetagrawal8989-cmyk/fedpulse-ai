# ==============================================================================
# Dockerfile for FedPulse AI - National Health & Supply Chain Resilience Platform
# Target Deployments: Google Cloud Run, Render, AWS App Runner, Docker Hub
# ==============================================================================

# Use official lightweight Python slim image
FROM python:3.11-slim

# Prevent Python from writing .pyc files & buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt-get/lists/*

# Copy & install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose default port (8080 for Cloud Run / Render)
EXPOSE 8080

# Health check endpoint for container lifecycle monitoring
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Execute Streamlit server bound to 0.0.0.0 and port 8080
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true"]
