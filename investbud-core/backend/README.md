# InvestBud AI

**AI-powered macro regime classifier for crypto portfolio advisory**

InvestBud AI analyzes macroeconomic indicators and crypto signals to classify the current market regime as **Risk-On** or **Risk-Off**, providing actionable portfolio recommendations and backtested trading strategies.

## TL;DR

- **What:** ML oracle that predicts crypto market regimes (Risk-On/Risk-Off) using macro indicators
- **How:** Analyzes Fed rates, inflation, VIX, BTC technicals → outputs regime + confidence score
- **Results:** 42.7% CAGR vs 31.7% buy-and-hold, 91.7% accuracy on test set
- **Monetization:** x402 micropayments (USDC on Base Sepolia) - pay per API call, no subscriptions
- **Verification:** Predictions published to OriginTrail DKG with cryptographic signatures
- **Decentralization:** Independent Graph Confirming Nodes verify oracle accuracy

## Try It Now

```bash
# Free endpoint - get latest cached regime report
curl https://your-api.com/latest_report

# Paid endpoint - get current regime ($0.01 USDC)
# Requires x402 client for payment
curl -H "X-PAYMENT: <receipt>" https://your-api.com/regime
```

See [Client Libraries](#client-libraries) for x402 payment integration.

## Deployed Addresses

### Smart Contracts

| Network | Contract | Address |
|---------|----------|---------|
| Base Sepolia | InvestBud AIOracle | [`0x21AF011807411bA1f81eA58963A55F641d0e9BF7`](https://sepolia.basescan.org/address/0x21AF011807411bA1f81eA58963A55F641d0e9BF7) |

### Oracle & Verifier Addresses

| Role | Address |
|------|---------|
| Oracle Signer | `0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF` |
| Verifier Node | `0x417dE910F1AfbA54A7eB623be7AEC02d2c46754c` |

### DKG (OriginTrail Decentralized Knowledge Graph)

| Network | Blockchain ID | Node Endpoint |
|---------|---------------|---------------|
| Neuroweb Testnet | `otp:20430` | `https://v6-pegasus-node-02.origin-trail.network:8900` |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENTS                                   │
│  (Wallets, Apps, Bots that want macro signals + portfolio data) │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Pay with x402 (USDC micropayments)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INVESTBUD AI SERVER                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Regime Oracle   │  │ Portfolio API   │  │ WebSocket Live  │  │
│  │ (ML classifier) │  │ (wallet metrics)│  │ (streaming)     │  │
│  └────────┬────────┘  └─────────────────┘  └─────────────────┘  │
│           │                                                      │
│           │ Publishes signed snapshots                          │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    DKG Service                               ││
│  │  - Signs regime predictions with oracle private key          ││
│  │  - Publishes to OriginTrail as "Knowledge Assets"            ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Knowledge Assets (JSON-LD)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ORIGINTRAIL DKG                               │
│                (Decentralized Knowledge Graph)                   │
│                                                                  │
│  Stores verifiable, immutable history of:                       │
│  - Regime predictions (risk-on/risk-off)                        │
│  - Confidence scores                                            │
│  - Macro indicators (Fed funds, CPI, VIX, etc.)                 │
│  - Signatures from the oracle                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Query & Verify
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  GRAPH CONFIRMING NODES                          │
│                                                                  │
│  Independent nodes that:                                        │
│  1. Monitor new snapshots published to DKG                      │
│  2. Verify signature matches official oracle address            │
│  3. Optionally challenge incorrect predictions                  │
│  4. Build reputation by confirming accurate signals             │
└─────────────────────────────────────────────────────────────────┘
```

### How the Components Connect

**InvestBud AI Server** is an AI oracle that predicts whether crypto markets are in "risk-on" (bullish) or "risk-off" (bearish) mode. It analyzes macro indicators like Fed rates, inflation, VIX, and BTC technicals, then runs them through an ML model to output a regime classification. It also provides portfolio analysis for any wallet address.

**x402** is how you pay for API access. Instead of subscriptions or API keys, every request includes a micropayment in USDC. The client pays 1-10 cents per call, the server verifies payment on-chain, and the request goes through. It's built into the HTTP layer - you get a 402 response with payment requirements, pay, retry with receipt, done.

**OriginTrail DKG** is where the oracle publishes its predictions permanently. Every time the server makes a regime call, it signs the prediction with the oracle's private key and publishes it to OriginTrail's Decentralized Knowledge Graph as a "Knowledge Asset." This creates an immutable, timestamped, cryptographically signed record. The oracle can't lie about its past predictions because they're all on-chain with signatures.

**Graph Confirming Nodes** are independent operators who validate the oracle. They run software that monitors new snapshots on the DKG, verifies the signature matches the official oracle address, and tracks prediction accuracy over time. They can also challenge bad predictions. This creates decentralized accountability - you don't have to trust the InvestBud AI team, you can verify the oracle's track record yourself or trust the confirming nodes to do it.

### Why This Matters

- **Accountability** - Oracle can't claim "I predicted that" after the fact. Everything is timestamped and signed on DKG.
- **Decentralization** - Confirming nodes create a network of verifiers, not just trusting one server.
- **Monetization** - x402 lets anyone pay for signals without KYC, subscriptions, or payment processors.
- **Composability** - Other apps can query the DKG directly to get InvestBud AI signals without hitting our server.

## Features

- **Macro Data Integration**: Fetches M2, GDP, CPI, Fed Funds, recession indicators from FRED API
- **Bitcoin Momentum Analysis**: Calculates returns, RSI, drawdowns, volatility
- **ML Regime Classifier**: Logistic regression model to predict Risk-On/Risk-Off (91.7% accuracy)
- **Backtesting Framework**: Simulates trading strategies with performance metrics
- **Portfolio Metrics**: Sharpe, Sortino, VaR, CVaR, CAGR, Max Drawdown, Win Rate
- **Wallet Analysis**: Fetch on-chain holdings and get regime-based recommendations
- **DKG Integration**: Publish regime snapshots to OriginTrail's Decentralized Knowledge Graph
- **Graph Confirming Nodes**: Independent verifiers that validate oracle predictions
- **x402 Payment Integration**: USDC micropayments for API access
- **WebSocket Live Streaming**: Real-time portfolio updates with x402 payment at handshake

## Key Results

The model-based trading strategy achieves:
- **42.7% CAGR** vs 31.7% for buy-and-hold (+11% alpha)
- **Sharpe Ratio of 1.22** vs 0.80 (+53% improvement)
- **28% lower volatility** (31.65% vs 44.40%)
- **96.9% prediction accuracy** on test set

## Quick Start

### 1. Get API Keys

**Required:**
- **FRED API Key**: Get from https://fredapi.stlouisfed.org/api_key.html

**Optional (for wallet analysis):**
- **Alchemy API Key**: Get from https://www.alchemy.com/
- **CoinGecko API Key**: Get from https://www.coingecko.com/en/api

### 2. Setup Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your API keys
# FRED_API_KEY=your_actual_api_key_here
# ALCHEMY_API_KEY=your_alchemy_key_here (for wallet analysis)
```

### 3. Train the Classifier

```bash
uv run python train_classifier.py
```

This will:
- Fetch 2 years of macro + BTC data
- Train logistic regression classifier
- Save model to `models/regime_classifier.pkl`
- Show test accuracy and feature importance

### 4. Run Backtest

```bash
uv run python run_backtest.py
```

This will:
- Load trained model
- Backtest 4 strategies (Buy & Hold, Model-Based, Always Risk-On, Always Risk-Off)
- Calculate performance metrics
- Show trade log and recommendations

### 5. Analyze a Wallet

```bash
uv run python analyze_wallet.py 0xYourWalletAddress
```

This will:
- Fetch wallet holdings from on-chain
- Calculate portfolio composition
- Get current regime prediction
- Provide investment recommendations

## Usage Examples

### Check Current Market Regime

```python
from src.macrocrypto.models import MacroRegimeClassifier

classifier = MacroRegimeClassifier()
classifier.load('models/regime_classifier.pkl')

result = classifier.predict_current_regime()
print(f"{result['regime']} - {result['confidence']*100:.0f}% confidence")
```

### Analyze Your Portfolio

```python
from src.macrocrypto.data import WalletAnalyzer
from src.macrocrypto.models import MacroRegimeClassifier

# Analyze wallet
analyzer = WalletAnalyzer()
analysis = analyzer.analyze_wallet('0xYourAddress')

# Get recommendations
classifier = MacroRegimeClassifier()
classifier.load('models/regime_classifier.pkl')
recommendation = analyzer.get_regime_recommendation(analysis, classifier)

print(f"Action: {recommendation['action']}")
print(f"Suggestion: {recommendation['suggestion']}")
```

### Custom Backtest

```python
from src.macrocrypto.models import RegimeBacktester, MacroRegimeClassifier
from src.macrocrypto.data import CombinedDataPipeline

# Fetch data
pipeline = CombinedDataPipeline()
df = pipeline.fetch_combined_data(start_date='2023-01-01')
df = pipeline.create_risk_labels(df)

# Load classifier
classifier = MacroRegimeClassifier()
classifier.load('models/regime_classifier.pkl')

# Run backtest
backtester = RegimeBacktester(initial_capital=100000)
backtest_df = backtester.prepare_backtest_data(df, classifier)
results = backtester.run_backtest(backtest_df)

# Print results
backtester.print_results()
```

## Project Structure

```
InvestBud AI/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI application
│   │
│   ├── src/macrocrypto/
│   │   ├── data/                # Data fetching modules
│   │   │   ├── fred_data.py     # FRED macro indicators
│   │   │   ├── btc_data.py      # Bitcoin momentum
│   │   │   ├── combined_data.py # Combined pipeline
│   │   │   └── wallet_analyzer.py
│   │   │
│   │   ├── models/              # ML models
│   │   │   ├── regime_classifier.py
│   │   │   └── backtest.py
│   │   │
│   │   ├── services/            # External integrations
│   │   │   ├── dkg_service.py   # OriginTrail DKG client
│   │   │   └── dkg_confirming_node.py  # Verifier node
│   │   │
│   │   └── utils/
│   │       ├── metrics.py
│   │       └── scheduler.py     # Background jobs
│   │
│   ├── examples/                # Client examples
│   │   ├── ws_live_client.py    # Python WebSocket client
│   │   └── advanced_metrics_demo.py
│   │
│   ├── models/
│   │   └── regime_classifier.pkl
│   │
│   ├── docs/
│   │   ├── DKG_CONFIRMING_NODE.md
│   │   └── DKG_INTEGRATION.md
│   │
│   ├── Dockerfile.confirming-node
│   ├── pyproject.toml
│   └── README.md
│
├── client/                      # TypeScript clients
│   └── src/
│       ├── client.ts            # Full API client with x402
│       └── x402_ws_client.ts    # WebSocket streaming client
│
├── contracts/                   # Solidity smart contracts
│   ├── src/
│   │   └── MacroSignalOracle.sol
│   └── deployments/
│       └── InvestBud AIOracle.json
│
├── analytics/                   # Jupyter notebooks & research
│   └── notebooks/
│       └── eda.ipynb            # Exploratory data analysis
│
├── Dockerfile                   # API Dockerfile
├── docker-compose.yml           # Full stack deployment
└── .env.example
```

## Data Sources

### Macro Indicators (FRED)
- **M2 Money Supply** (`M2SL`)
- **GDP** (`GDP`)
- **CPI Inflation** (`CPIAUCSL`)
- **Fed Funds Rate** (`FEDFUNDS`)
- **Recession Indicator** (`USREC`)
- **Dollar Index** (`DTWEXBGS`)
- **Unemployment Rate** (`UNRATE`)
- **VIX** (`VIXCLS`)
- **Treasury Yields** (`DGS10`, `DGS2`)

### Crypto Data (Bitcoin)
- **Price** (via yfinance)
- **Returns** (1d, 7d, 30d, 60d, 90d, 180d)
- **RSI** (Relative Strength Index)
- **Drawdown** from all-time high
- **Volatility** (7d, 30d, 90d rolling)
- **Moving Averages** (7d, 30d, 90d, 200d)

### On-Chain Data (via Alchemy)
- ERC-20 token balances
- ETH balance
- Token metadata (symbol, decimals)
- Portfolio composition

## API Server

InvestBud AI includes a FastAPI server with X402 micropayment integration.

### Setup Database (for /chat endpoint)

```bash
# Start PostgreSQL with Docker
docker run -d \
  --name macrocrypto-postgres \
  -e POSTGRES_USER=macrocrypto \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=macrocrypto \
  -p 5433:5432 \
  -v pgdata:/var/lib/postgresql/data \
  postgres:15

# Initialize database tables
uv run python init_db.py
```

### Start API Server

```bash
uv run python start_api.py
# Server runs on http://localhost:8001
# Docs available at http://localhost:8001/docs
```

### Endpoints

**FREE Endpoints:**
- `GET /health` - Health check
- `GET /latest_report` - Cached regime report + model performance
- `GET /historical` - Backtest results (CSV/JSON)
- `GET /model/metrics` - Model accuracy and performance metrics

**PAID Endpoints (x402 micropayments):**
- `GET /regime` - **$0.01** - Current macro regime
- `POST /portfolio` - **$0.05** - Wallet analysis (no LLM)
- `POST /advise` - **$0.10** - Full advisory (wallet + regime + LLM)
- `POST /chat` - **$0.02** - Stateful conversation
- `WS /wallet/live` - **$0.02** - Live portfolio streaming (5 min session)

### Chat Endpoint Features

The `/chat` endpoint provides conversational AI with full portfolio and regime context:

**Features:**
- Auto-generated session IDs (or bring your own)
- Conversation history persistence (last 20 messages)
- Current regime context (Risk-On/Risk-Off with indicators)
- Full portfolio analysis (holdings, allocation, value)
- 5-minute portfolio caching (reduces API calls)
- Multi-network support (Ethereum, Arbitrum, Polygon, etc.)

**Example Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "x-payment-receipt: <x402_receipt>" \
  -d '{
    "message": "What should I do with my portfolio?",
    "wallet_address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
    "network": "eth-mainnet"
  }'
```

**Example Response:**
```json
{
  "session_id": "8f3a4b2c-1234-5678-90ab-cdef12345678",
  "response": "Based on the current Risk-On regime with 85% confidence and your portfolio allocation...",
  "message_count": 2,
  "context_used": {
    "wallet": true,
    "portfolio": true,
    "regime": true
  }
}
```

**Security:** Session IDs are cryptographically secure 128-bit UUIDs. See [CHAT_SECURITY.md](CHAT_SECURITY.md) for details.

⚠️ **Privacy Warning:** Do not share your session_id - anyone with it can access your conversation history.

### WebSocket Live Streaming

The `/wallet/live` endpoint provides real-time portfolio updates via WebSocket with x402 payment at handshake.

**Connection Flow:**
1. Client makes HTTP GET to `/wallet/live` → receives 402 with payment requirements
2. Pay via x402 → obtain receipt
3. Connect to WebSocket with `X-PAYMENT` header or `?receipt=` query param
4. Receive portfolio updates every 30 seconds until session expires (max 5 minutes)

**Configuration:**
| Variable | Default | Description |
|----------|---------|-------------|
| `LIVE_STREAM_COST` | `0.02` | Cost per session (USDC) |
| `LIVE_STREAM_INTERVAL` | `30` | Update interval (seconds) |
| `WEBSOCKET_MAX_TIME` | `300` | Max session duration (seconds) |

**Message Types:**
```typescript
// Connection established
{ "type": "connected", "session_id": "...", "expires_at": "..." }

// Portfolio update (every 30s)
{ "type": "portfolio_update", "total_value_usd": 12345.67, "positions": [...] }

// Session ending warning (30s before expiry)
{ "type": "session_ending", "seconds_remaining": 30 }

// Errors
{ "type": "error", "code": "session_expired", "message": "..." }
```

**Python Client Example:**
```bash
# Install dependencies
pip install websockets eth-account x402

# Run client
python examples/ws_live_client.py \
  --address 0x742d35Cc6634C0532925a3b844Bc9e7595f2bD45 \
  --private-key $CLIENT_PRIVATE_KEY \
  --url http://localhost:8001
```

## x402 Payment Integration

InvestBud AI uses [x402](https://www.x402.org/) for micropayments - every paid API call includes a USDC payment on Base Sepolia.

### How x402 Works

1. **Client makes request** → Server responds with `402 Payment Required` + payment details
2. **Client pays** → Signs and broadcasts USDC transfer to server's address
3. **Client retries with receipt** → `X-PAYMENT` header contains payment proof
4. **Server verifies** → Confirms payment on-chain, processes request

### Payment Configuration

| Variable | Description |
|----------|-------------|
| `PAYTO_ADDRESS` | Server's wallet address to receive payments |
| `NETWORK` | Blockchain network (`base-sepolia`) |
| `FACILITATOR_URL` | x402 facilitator for payment verification |

### Pricing

| Endpoint | Cost (USDC) |
|----------|-------------|
| `/regime` | $0.01 |
| `/portfolio` | $0.05 |
| `/advise` | $0.10 |
| `/chat` | $0.02 |
| `/wallet/live` (WebSocket) | $0.02 per session |

## Client Libraries

### TypeScript Client

Located in `client/src/`:

**`client.ts`** - Full API client with x402 integration:
```typescript
import { wrapFetchWithPayment, createSigner } from "x402-fetch";

const signer = await createSigner("base-sepolia", PRIVATE_KEY);
const fetchWithPay = wrapFetchWithPayment(fetch, signer);

// Paid endpoints work automatically
const response = await fetchWithPay("http://localhost:8001/regime");
```

**`x402_ws_client.ts`** - WebSocket client for live streaming:
```typescript
import { X402WebSocketClient } from './x402_ws_client';

const client = new X402WebSocketClient({
  baseUrl: 'http://localhost:8001',
  walletAddress: '0x...',
  paymentProvider: async (requirements) => {
    // Implement x402 payment
    return receipt;
  },
  onUpdate: (update) => console.log(`Portfolio: $${update.total_value_usd}`),
});

await client.connect();
```

### Python Client

Located in `examples/`:

**`ws_live_client.py`** - WebSocket client with x402:
```python
from x402.clients.requests import x402_requests
from eth_account import Account

account = Account.from_key(PRIVATE_KEY)
session = x402_requests(account)

# Automatic 402 handling
response = session.get("http://localhost:8001/regime")
```

**`advanced_metrics_demo.py`** - Portfolio metrics examples

## Smart Contracts

### MacroSignalOracle.sol

On-chain oracle contract for storing regime signals. Allows the oracle to publish signed predictions that can be verified by anyone.

**Deployed:** Base Sepolia at [`0x21AF011807411bA1f81eA58963A55F641d0e9BF7`](https://sepolia.basescan.org/address/0x21AF011807411bA1f81eA58963A55F641d0e9BF7)

**Key Functions:**
```solidity
// Submit a new signal (oracle only)
function submitSignal(
    bool riskOn,
    uint8 confidence,
    bytes32 snapshotHash,
    bytes calldata signature
) external;

// Get latest signal
function getLatestSignal() external view returns (Signal memory);

// Verify a signature
function verifySignature(
    bytes32 snapshotHash,
    bytes calldata signature
) external view returns (bool);
```

**Building & Testing:**
```bash
cd contracts
forge build
forge test
forge script scripts/Deploy.s.sol --broadcast --rpc-url base-sepolia
```

## DKG Integration (OriginTrail)

InvestBud AI publishes regime snapshots to OriginTrail's Decentralized Knowledge Graph, creating a verifiable, immutable record of predictions.

### Publishing Snapshots

```bash
# Publish current regime to DKG
curl -X POST http://localhost:8001/dkg/publish
```

Response:
```json
{
  "success": true,
  "ual": "did:dkg:otp:20430/0xcdb28e93ed340ec10a71bba00a31dbfcf1bd5d37/405836",
  "dataset_root": "0x..."
}
```

### DKG Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /dkg/publish` | Publish regime snapshot to DKG |
| `GET /dkg/latest` | Get latest published UAL |
| `GET /dkg/uals` | List published UALs (for confirming nodes) |
| `GET /dkg/snapshot/{ual}` | Retrieve snapshot from DKG |
| `GET /dkg/status` | Check DKG node connection |

### JSON-LD Schema

Published snapshots use the `mc:RegimeSnapshot` type:

```json
{
  "@type": ["mc:RegimeSnapshot", "Dataset"],
  "regime": "Risk-Off",
  "confidence": 0.85,
  "signerAddress": "0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF",
  "btcPrice": 95000,
  "vix": 18.5,
  "fedFunds": 4.5
}
```

## Graph Confirming Node

Independent nodes that verify oracle predictions by:
1. Polling for new snapshots from the oracle API
2. Independently computing the same prediction
3. Publishing confirmation or challenge reports to DKG

### Running a Confirming Node

```bash
# Set environment variables
export ORACLE_API_URL=http://localhost:8001
export DKG_NODE_ENDPOINT=https://v6-pegasus-node-02.origin-trail.network:8900
export VERIFIER_ADDRESS=0xYourAddress
export ORACLE_SIGNER_ADDRESS=0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF
export MODEL_PATH=models/regime_classifier.pkl
export POLL_INTERVAL=3600  # 1 hour

# Run the node
python -m macrocrypto.services.dkg_confirming_node
```

### Verification Results

| Result | Description |
|--------|-------------|
| `CONFIRMED` | Regime matches, confidence within 10% |
| `CHALLENGED` | Regime differs from independent prediction |
| `PARTIAL` | Regime matches but confidence differs >10% |
| `ERROR` | Verification failed |

### Trust Levels

| Confirmations | Trust Level |
|---------------|-------------|
| 0 | Unverified |
| 1-2 | Low |
| 3-5 | Medium |
| 5+ | High |
| Any CHALLENGED | Disputed |

See [DKG_CONFIRMING_NODE.md](docs/DKG_CONFIRMING_NODE.md) for full architecture details.

## Deployment

### Akash Network (Production)

InvestBud AI is deployed on [Akash Network](https://akash.network/), a decentralized cloud computing marketplace.

```bash
# Deploy using akash.yaml
akash tx deployment create akash.yaml --from wallet --chain-id akashnet-2
```

The deployment includes:
- **API service** - FastAPI server with x402 payments
- **PostgreSQL** - Persistent storage for chat sessions

See `akash.yaml` for the full deployment configuration.

### Docker Compose (Local/Self-hosted)

```bash
# Build and run all services
docker-compose up --build

# Or run specific services
docker-compose up api postgres           # Just API
docker-compose up confirming-node        # Just confirming node
```

### Services

| Service | Description | Port |
|---------|-------------|------|
| `api` | InvestBud AI API server | 8001 |
| `postgres` | PostgreSQL for chat sessions | 5433 |
| `confirming-node` | Graph Confirming Node | - |

### Environment Variables

Create a `.env` file:

```bash
# API Keys
FRED_API_KEY=your_fred_api_key
ALCHEMY_API_KEY=your_alchemy_key

# Wallet & Signing
PRIVATE_KEY=0x...
PAYTO_ADDRESS=0x...

# DKG Configuration
DKG_NODE_ENDPOINT=https://v6-pegasus-node-02.origin-trail.network:8900
DKG_BLOCKCHAIN_ID=otp:20430
ORACLE_SIGNER_ADDRESS=0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF

# Confirming Node
VERIFIER_ADDRESS=0xYourVerifierAddress
POLL_INTERVAL=3600

# Database
POSTGRES_PASSWORD=secret
```

### Standalone Confirming Node

```bash
cd backend
docker build -f Dockerfile.confirming-node -t macrocrypto-confirming-node .
docker run \
  -e ORACLE_API_URL=http://host.docker.internal:8001 \
  -e DKG_NODE_ENDPOINT=https://v6-pegasus-node-02.origin-trail.network:8900 \
  -e VERIFIER_ADDRESS=0xYourAddress \
  -e ORACLE_SIGNER_ADDRESS=0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF \
  -v $(pwd)/models:/app/models:ro \
  macrocrypto-confirming-node
```

## Roadmap

**Completed:**
- [x] FRED API integration
- [x] BTC data fetching
- [x] Combined data pipeline
- [x] Risk regime labeling
- [x] Logistic regression classifier
- [x] Model training & backtesting
- [x] Portfolio metrics calculation
- [x] Wallet analysis (fetch on-chain holdings)
- [x] LLM advisory summaries
- [x] FastAPI endpoints with X402 payments
- [x] Chat endpoint with PostgreSQL persistence
- [x] Portfolio caching and context
- [x] DKG integration (OriginTrail)
- [x] Graph Confirming Node
- [x] Docker deployment

**In Progress:**
- [ ] DKG mainnet deployment
- [ ] Multiple confirming node operators
- [ ] Staking for verifiers

**Planned:**
- [ ] Frontend dashboard
- [ ] Multi-asset support (ETH, SOL, etc.)
- [ ] Wallet signature authentication
- [ ] Session expiration (30-day TTL)
- [ ] Historical portfolio tracking

## Performance Metrics

### Model Accuracy (Test Set)
| Metric | Value |
|--------|-------|
| Accuracy | 91.7% |
| Precision | 100% |
| Recall | 66.7% |
| F1 Score | 0.80 |
| ROC AUC | 0.998 |

### Backtest Results (Sep 2024 - Nov 2025)
| Strategy | CAGR | Sharpe | Volatility | Max DD | Final Value |
|----------|------|--------|------------|--------|-------------|
| **Model-Based** | **42.70%** | **1.217** | **31.65%** | **-31.04%** | **$14,993** |
| Buy & Hold | 31.67% | 0.795 | 44.40% | -30.56% | $13,680 |
| Always Risk-Off | 0.00% | 0.000 | 0.00% | 0.00% | $10,000 |

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Detailed system architecture and component guide
- **[idea.md](docs/idea.md)**: Original project concept
- **[api.md](docs/api.md)**: API endpoint specifications (planned)

## Development

Built with:
- **uv** for fast Python package management
- **pandas** for data manipulation
- **scikit-learn** for ML models
- **yfinance** for crypto data
- **fredapi** for economic data
- **web3** for blockchain interaction
- **requests** for API calls

## License

MIT
