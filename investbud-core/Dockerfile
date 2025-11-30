# MacroCrypto API Dockerfile
# Multi-stage build for smaller final image

# =============================================================================
# Stage 1: Builder - Install dependencies with uv
# =============================================================================
FROM python:3.13-slim AS builder

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first (better layer caching)
COPY backend/pyproject.toml backend/uv.lock backend/README.md ./backend/

# Copy the x402 patch script
COPY backend/scripts/patch_x402.py ./backend/scripts/

# Install dependencies into a virtual environment
RUN cd backend && uv sync --frozen --no-dev

# Apply x402 patch (fix for missing 'error' field validation)
RUN cd backend && .venv/bin/python scripts/patch_x402.py

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM python:3.13-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/backend/.venv /app/backend/.venv

# Copy backend source code
COPY backend/ ./backend/

# Copy contracts (for ABI files in contracts/out/)
COPY contracts/out/ ./contracts/out/
COPY contracts/deployments/ ./contracts/deployments/

# Set environment variables
ENV PATH="/app/backend/.venv/bin:$PATH"
ENV PYTHONPATH="/app/backend/src:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Default port (configurable via PORT env var)
ENV PORT=8015

EXPOSE 8015

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Set working directory to backend
WORKDIR /app/backend

# Run the API server
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
