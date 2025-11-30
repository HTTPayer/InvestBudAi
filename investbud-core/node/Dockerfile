# DKG Node Dockerfile
# Multi-stage build for optimized production image

# ============================================
# Stage 1: Build stage
# ============================================
FROM node:22-bookworm-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./
COPY turbo.json ./
COPY apps/agent/package*.json ./apps/agent/
COPY apps/agent/fix-uuid.js ./apps/agent/
COPY packages/plugins/package*.json ./packages/plugins/
COPY packages/plugin-auth/package*.json ./packages/plugin-auth/
COPY packages/plugin-dkg-essentials/package*.json ./packages/plugin-dkg-essentials/
COPY packages/plugin-dkg-publisher/package*.json ./packages/plugin-dkg-publisher/
COPY packages/plugin-example/package*.json ./packages/plugin-example/
COPY packages/plugin-oauth/package*.json ./packages/plugin-oauth/
COPY packages/plugin-swagger/package*.json ./packages/plugin-swagger/
COPY packages/webpage2dkg/package*.json ./packages/webpage2dkg/
COPY packages/expo-forcegraph/package*.json ./packages/expo-forcegraph/
COPY packages/internal/package*.json ./packages/internal/
COPY packages/eslint-config/package*.json ./packages/eslint-config/
COPY packages/typescript-config/package*.json ./packages/typescript-config/

# Install dependencies
RUN npm ci --legacy-peer-deps

# Copy source code
COPY . .

# Build the application
RUN npm run build

# ============================================
# Stage 2: Production stage
# ============================================
FROM node:22-bookworm-slim AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r dkgnode && useradd -r -g dkgnode dkgnode

WORKDIR /app

# Copy built application from builder stage
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/turbo.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/apps ./apps
COPY --from=builder /app/packages ./packages

# Create data directory for SQLite database
RUN mkdir -p /app/data && chown -R dkgnode:dkgnode /app

# Environment variables
ENV NODE_ENV=production
ENV PORT=9200
ENV DATABASE_URL=/app/data/dkg.db

# Expose ports
# 9200 - DKG Agent API/MCP Server (production)
# 8081 - DKG Agent Web UI (development)
EXPOSE 9200 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Switch to non-root user
USER dkgnode

# Default command - run the production server
CMD ["node", "apps/agent/dist/index.js"]
