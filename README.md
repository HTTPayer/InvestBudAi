<p align="center">
  <img src="img/investbud_logo.png" alt="InvestBud AI Logo" width="400">
</p>

# InvestBud AI

**AI-powered macro regime oracle with x402 micropayments and decentralized verification**

InvestBud AI predicts crypto market regimes (Risk-On/Risk-Off) using macroeconomic indicators, publishes cryptographically signed predictions to OriginTrail's DKG, and monetizes via x402 micropayments - including **real-time WebSocket streaming with payment at handshake**.

## Why InvestBud AI?

- **42.7% CAGR** vs 31.7% buy-and-hold (+11% alpha)
- **91.7% accuracy** on test set
- **Verifiable predictions** - all signals published to DKG with signatures
- **No subscriptions** - pay per call with USDC micropayments

## Novel: WebSocket Streaming with x402

InvestBud AI is one of the first APIs to implement **real-time streaming with x402 payment at WebSocket handshake**:

```
Client                          Server
  │                               │
  ├── GET /wallet/live ──────────►│
  │◄── 402 Payment Required ──────┤
  │                               │
  ├── Pay USDC on Base ──────────►│ (on-chain)
  │                               │
  ├── WS connect + X-PAYMENT ────►│
  │◄── {"type": "connected"} ─────┤
  │                               │
  │◄── portfolio_update ──────────┤ (every 30s)
  │◄── portfolio_update ──────────┤
  │◄── session_ending ────────────┤ (30s warning)
  │◄── connection closed ─────────┤ (5 min max)
```

One payment unlocks a full streaming session. No polling, no rate limits, no API keys.

## Architecture

<p align="center">
  <img src="img/investbud_diagram.png" alt="InvestBud AI Architecture" width="800">
</p>

```
┌─────────────────────────────────────────────────────────────┐
│                    INVESTBUD AI ORACLE                       │
├─────────────────┬─────────────────┬─────────────────────────┤
│ Regime Oracle   │ Portfolio API   │ WebSocket Live Stream   │
│ (ML classifier) │ (wallet metrics)│ (x402 at handshake)     │
└────────┬────────┴─────────────────┴─────────────────────────┘
         │
         │ Signed snapshots
         ▼
┌─────────────────────────────────────────────────────────────┐
│                  ORIGINTRAIL DKG                             │
│  Immutable, timestamped, cryptographically signed records   │
└─────────────────────────────────────────────────────────────┘
         │
         │ Query & Verify
         ▼
┌─────────────────────────────────────────────────────────────┐
│              GRAPH CONFIRMING NODES                          │
│  Independent verifiers that validate oracle accuracy         │
└─────────────────────────────────────────────────────────────┘
```

## Features

| Category            | Features                                                              |
| ------------------- | --------------------------------------------------------------------- |
| **Oracle**          | ML regime classifier, 35 macro/crypto features, daily predictions     |
| **Payments**        | x402 micropayments (USDC on Base), pay-per-call, no subscriptions     |
| **Streaming**       | WebSocket live portfolio updates with x402 payment at handshake       |
| **Verification**    | DKG integration, cryptographic signatures, confirming nodes           |
| **Portfolio**       | On-chain wallet analysis, Sharpe/Sortino/VaR metrics, historical data |
| **Smart Contracts** | On-chain oracle for DeFi integrations (Base Sepolia)                  |

## Deployments

| Resource            | URL                                                                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **API**             | https://r3467d7khd8b94sfguhrr273lo.ingress.akashprovid.com                                                                                     |
| **Docs**            | https://r3467d7khd8b94sfguhrr273lo.ingress.akashprovid.com/docs                                                                                |
| **Backend Signer**  | `0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF` (Base Sepolia + NeuroWeb)                                                                         |
| **Oracle Contract** | [`0x21AF011807411bA1f81eA58963A55F641d0e9BF7`](https://sepolia.basescan.org/address/0x21AF011807411bA1f81eA58963A55F641d0e9BF7) (Base Sepolia) |
| **DKG Network**     | NeuroWeb Testnet (`otp:20430`)                                                                                                                 |

## Quick Start

```bash
# Get latest regime (FREE)
curl https://r3467d7khd8b94sfguhrr273lo.ingress.akashprovid.com/latest_report

# With TypeScript client (handles x402 payments automatically)
cd client && npm install
npm run client regime              # $0.01
npm run client portfolio 0x...     # $0.05
npm run client live 0x...          # $0.05 - WebSocket stream!
```

## API Endpoints

### Free Endpoints

| Endpoint                | Description                       |
| ----------------------- | --------------------------------- |
| `GET /`                 | Service info                      |
| `GET /health`           | Health check                      |
| `GET /latest_report`    | Cached regime + model performance |
| `GET /model/historical` | Backtest results (JSON/CSV)       |
| `GET /model/metrics`    | Model accuracy metrics            |
| `GET /dkg/info`         | DKG node status                   |
| `GET /dkg/latest`       | Latest published UAL              |

### Paid Endpoints (x402 USDC)

| Endpoint                   | Cost  | Description                        |
| -------------------------- | ----- | ---------------------------------- |
| `GET /regime`              | $0.01 | Current macro regime prediction    |
| `POST /portfolio`          | $0.05 | Wallet analysis                    |
| `POST /advise`             | $0.10 | Full LLM advisory                  |
| `POST /chat`               | $0.02 | Stateful conversation              |
| `POST /signal`             | $0.10 | Push to on-chain oracle            |
| `GET /news`                | $0.10 | Smart money flow analysis          |
| `POST /wallet/performance` | $0.05 | Sharpe, VaR, max drawdown          |
| `POST /wallet/historical`  | $0.05 | Historical returns                 |
| `POST /wallet/composition` | $0.05 | Portfolio weights over time        |
| `POST /wallet/rolling`     | $0.05 | Rolling window metrics             |
| `WS /wallet/live`          | $0.05 | **Live streaming** (5 min session) |

### DKG Endpoints (Free)

| Endpoint                  | Description                    |
| ------------------------- | ------------------------------ |
| `POST /dkg/publish`       | Publish regime to DKG          |
| `POST /dkg/query`         | Query historical snapshots     |
| `GET /dkg/snapshot/{ual}` | Get snapshot by UAL            |
| `POST /dkg/verify`        | Verify snapshot independently  |
| `GET /dkg/uals`           | List UALs for confirming nodes |

## Project Structure

```
InvestBud AI/
├── backend/          # FastAPI + ML models + DKG integration
├── client/           # TypeScript CLI with x402 + WebSocket
├── contracts/        # Solidity oracle (Foundry)
├── analytics/        # Jupyter notebooks
├── gcn/              # Graph Confirming Node CLI
└── node/             # Docker deployment configs
```

## Environment Variables

```bash
# Required
FRED_API_KEY=...              # Macro data from FRED
ALCHEMY_API_KEY=...           # On-chain wallet data
PRIVATE_KEY=0x...             # Oracle signing key
PAYTO_ADDRESS=0x...           # Receive x402 payments

# DKG (OriginTrail)
DKG_NODE_ENDPOINT=https://v6-pegasus-node-02.origin-trail.network:8900
DKG_BLOCKCHAIN_ID=otp:20430

# Optional
DATABASE_URL=postgresql://... # For /chat persistence
OPENAI_API_KEY=...            # For LLM advisory
```

## Running Locally

```bash
# Backend
cd backend
uv sync
uv run python scripts/train_classifier.py  # First time only
uv run python start_api.py                  # http://localhost:8015

# Client
cd client
npm install
npm run client help
```

## Docker Deployment

```bash
# Full stack with DKG node
docker-compose up -d

# Just the API
docker run -p 8015:8015 brandynham/dkg-node:latest
```

## Model Performance

| Metric          | Value |
| --------------- | ----- |
| Accuracy        | 91.7% |
| Precision       | 100%  |
| ROC AUC         | 0.998 |
| CAGR (backtest) | 42.7% |
| Alpha vs B&H    | +11%  |

## Technology Integration

### MCP (Model Context Protocol)

InvestBud AI exposes an [MCP server](https://modelcontextprotocol.io/) that allows AI assistants like Claude Desktop to access the API with automatic x402 payment handling:

**Tools Available:**

| Tool                 | Description                         |
| -------------------- | ----------------------------------- |
| `get_macro_regime`   | Current Risk-On/Risk-Off prediction |
| `analyze_wallet`     | Wallet composition and risk metrics |
| `advise_portfolio`   | Full LLM-powered advisory           |
| `wallet_performance` | Sharpe, VaR, drawdown metrics       |

**Payment Bridge:** When a tool requires payment, the MCP server automatically opens a browser window for wallet approval (MetaMask or Polkadot.js), then continues the request.

```
Claude Desktop → MCP Server → 402 Payment Required
                     ↓
              Payment Bridge opens browser
                     ↓
              User approves in wallet
                     ↓
              Request retries with X-PAYMENT header
                     ↓
              Claude Desktop receives data
```

### x402 Micropayments

[x402](https://www.x402.org/) is an HTTP payment protocol that enables pay-per-request APIs using stablecoin micropayments:

```
1. Client requests paid endpoint
2. Server returns 402 Payment Required + payment details
3. Client pays USDC on Base Sepolia (on-chain)
4. Client retries with X-PAYMENT header containing receipt
5. Server verifies payment and returns data
```

**Libraries**: `x402-fetch` (TypeScript), `x402` (Python)

InvestBud AI extends x402 to **WebSocket streaming** - one payment at handshake unlocks a full streaming session.

### OriginTrail DKG (Decentralized Knowledge Graph)

Every regime prediction is published as a **Knowledge Asset** to OriginTrail's DKG:

- **Immutable**: Snapshots cannot be modified after publishing
- **Timestamped**: Blockchain-anchored proof of when prediction was made
- **Cryptographically signed**: Verifiable authorship via `signerAddress`
- **Queryable**: SPARQL queries across the decentralized graph

```sparql
PREFIX mc: <https://macrocrypto.io/ontology#>
SELECT ?regime ?confidence ?timestamp
WHERE {
    ?snapshot a mc:RegimeSnapshot .
    ?snapshot mc:signerAddress "0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF" .
    ?snapshot mc:regime ?regime .
    ?snapshot mc:confidence ?confidence .
}
```

### Graph Confirming Nodes

Independent verifiers that validate oracle predictions:

```
Oracle publishes: "Risk-Off, 85% confidence"
        │
        ├── Node A runs same model → "Risk-Off, 84%" → CONFIRMED
        ├── Node B runs same model → "Risk-Off, 86%" → CONFIRMED
        └── Node C runs same model → "Risk-On, 60%"  → CHALLENGED
```

Confirmations are also published to DKG, creating a **trust network** where predictions with multiple independent confirmations are more trustworthy.

### NeuroWeb (Polkadot Parachain)

DKG interactions are anchored on **NeuroWeb Testnet** (`otp:20430`), a Polkadot parachain:

- Knowledge Assets get a UAL (Universal Asset Locator): `did:dkg:otp:20430/0xcdb.../405801`
- Publishing requires NEURO tokens for gas
- Immutable once anchored on-chain

## Documentation

- [Backend README](backend/README.md) - Full API docs, DKG integration, confirming nodes
- [DKG Integration](backend/docs/DKG_INTEGRATION.md) - OriginTrail publishing
- [Confirming Nodes](backend/docs/DKG_CONFIRMING_NODE.md) - Verification architecture
- [GCN CLI](gcn/README.md) - Graph Confirming Node standalone CLI

## License

TBA
