# InvestBud AI TypeScript Client

TypeScript CLI and library for interacting with the InvestBud AI API, featuring automatic x402 micropayments and WebSocket streaming support.

## Installation

```bash
npm install
```

## Configuration

Create a `.env` file:

```bash
# Required: Your private key for x402 payments (Base Sepolia)
CLIENT_PRIVATE_KEY=0x...

# Optional: API endpoint (defaults to localhost)
MACRO_CRYPTO_URL=https://r3467d7khd8b94sfguhrr273lo.ingress.akashprovid.com
```

## CLI Usage

```bash
npm run client help
```

### Paid Endpoints (x402 USDC)

```bash
# Regime & Advisory
npm run client regime                    # $0.01 - Current macro regime
npm run client portfolio 0x...           # $0.05 - Wallet analysis
npm run client advise 0x...              # $0.10 - Full LLM advisory
npm run client chat "message"            # $0.02 - Stateful conversation
npm run client signal base-sepolia       # $0.10 - Push to on-chain oracle
npm run client news                      # $0.10 - Smart money analysis

# Wallet Analytics
npm run client wallet:performance 0x...  # $0.05 - Sharpe, VaR, drawdown
npm run client wallet:historical 0x...   # $0.05 - Historical returns
npm run client wallet:composition 0x...  # $0.05 - Portfolio weights
npm run client wallet:rolling 0x... 30   # $0.05 - Rolling metrics (30d window)

# Live Streaming
npm run client live 0x...                # $0.05 - WebSocket stream (5 min)
```

### Free Endpoints

```bash
npm run client model:historical          # Backtest results
npm run client model:metrics             # Model accuracy
npm run client latest-report             # Cached regime report

# DKG
npm run client dkg:info                  # Node status
npm run client dkg:latest                # Latest UAL
npm run client dkg:uals                  # UAL history
npm run client dkg:publish               # Publish to DKG
npm run client dkg:query Risk-On 0.8     # Query snapshots
npm run client dkg:snapshot <ual>        # Get snapshot
npm run client dkg:verify <ual>          # Verify snapshot
```

## Library Usage

### Basic API Calls

```typescript
import { wrapFetchWithPayment, createSigner } from "x402-fetch";

const signer = await createSigner("base-sepolia", PRIVATE_KEY);
const fetchWithPay = wrapFetchWithPayment(fetch, signer);

// Paid endpoint - x402 payment handled automatically
const response = await fetchWithPay("http://localhost:8015/regime");
const data = await response.json();
console.log(data.regime, data.confidence);
```

### WebSocket Streaming

```typescript
import { LivePortfolioClient } from "./client";

const client = new LivePortfolioClient("http://localhost:8015");

await client.connect("0xYourWallet", "eth-mainnet");
await client.stream((update) => {
  console.log(`Portfolio: $${update.total_value_usd}`);
  for (const pos of update.positions) {
    console.log(`  ${pos.symbol}: $${pos.value_usd}`);
  }
});
```

### Standalone WebSocket Client

```typescript
import { X402WebSocketClient } from "./x402_ws_client";

const client = new X402WebSocketClient({
  baseUrl: "http://localhost:8015",
  walletAddress: "0x...",
  network: "eth-mainnet",
  paymentProvider: async (requirements) => {
    // Implement x402 payment, return receipt
    return receipt;
  },
  onUpdate: (update) => {
    console.log(`Value: $${update.total_value_usd}`);
  },
  onSessionEnding: (seconds) => {
    console.log(`Session ending in ${seconds}s`);
  },
});

await client.connect();
```

## Source Files

| File | Description |
|------|-------------|
| `src/client.ts` | Full CLI client with all endpoints and `LivePortfolioClient` class |
| `src/x402_ws_client.ts` | Standalone WebSocket client with x402 payment handling |

## How x402 Payments Work

1. Client makes request to paid endpoint
2. Server responds with `402 Payment Required` + payment details
3. `x402-fetch` automatically pays USDC on Base Sepolia
4. Request retries with `X-PAYMENT` header containing receipt
5. Server verifies payment and returns data

For WebSocket:
1. HTTP GET to `/wallet/live` returns 402 with requirements
2. Pay via x402, get receipt
3. Connect WebSocket with `X-PAYMENT` header
4. Stream updates for session duration (5 min max)

## Dependencies

- `x402-fetch` - Automatic x402 payment handling
- `viem` - Ethereum interactions
- `ws` - WebSocket client for Node.js
- `dotenv` - Environment configuration

## Building

```bash
npm run build
```

## License

TBA
