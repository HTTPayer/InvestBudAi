# InvestBud AI Graph Confirming Node
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create cache directory
RUN mkdir -p /app/cache

# Set environment defaults
ENV CACHE_DIR=/app/cache
ENV POLL_INTERVAL=3600

# Run the confirming node
CMD ["gcn"]
