# InvestBud AI Analytics

Jupyter notebooks for exploring the InvestBud AI API, visualizing portfolio data, and analyzing model performance.

## Setup

```bash
cd analytics

# Install dependencies
uv sync

# Start Jupyter
uv run jupyter notebook
```

## Notebooks

### `notebooks/eda.ipynb` - Exploratory Data Analysis

Interactive notebook demonstrating all InvestBud AI API endpoints with x402 payments:

**API Endpoints Covered:**

| Endpoint | Description |
|----------|-------------|
| `/regime` | Current macro regime (Risk-On/Risk-Off) |
| `/portfolio` | Wallet holdings and metrics |
| `/advise` | Full LLM advisory |
| `/news` | Smart money flow analysis |
| `/chat` | Stateful conversation |
| `/signal` | On-chain oracle submission |
| `/wallet/historical` | Historical returns and composition |
| `/wallet/performance` | Sharpe, Sortino, VaR, drawdown metrics |
| `/model/historical` | Backtest cumulative returns |
| `/model/metrics` | Model accuracy and feature importance |
| `/wallet/live` | WebSocket streaming (x402 at handshake) |

**Visualizations Generated:**

- Cumulative return percentage over time
- Total portfolio value USD
- Portfolio composition (stacked area)
- Normalized returns comparison (model vs buy-and-hold)

## Configuration

Create a `.env` file:

```bash
# Required: Private key for x402 payments
PRIVATE_KEY=0x...

# Wallet to analyze
WALLET_ADDRESS=0x...
NETWORK=eth-mainnet  # or opt-mainnet, arb-mainnet, etc.

# API endpoint
BASE_URL=https://r3467d7khd8b94sfguhrr273lo.ingress.akashprovid.com
```

## Generated Charts

Charts are saved to `img/`:

| File | Description |
|------|-------------|
| `cumulative_return_pct.png` | Portfolio cumulative returns over time |
| `total_value_usd.png` | Portfolio value in USD |
| `composition.png` | Token allocation over time (stacked area) |
| `normalized_returns.png` | Model vs buy-and-hold backtest comparison |

## Dependencies

- `pandas` - Data manipulation
- `chartengineer` - Chart generation (Plotly-based)
- `x402` - Micropayment client
- `eth-account` - Ethereum signing
- `websockets` - WebSocket streaming
- `python-dotenv` - Environment configuration

## Example: Fetching Regime

```python
from eth_account import Account
from x402.clients.requests import x402_requests

account = Account.from_key(PRIVATE_KEY)
session = x402_requests(account)

response = session.get("https://api.example.com/regime")
data = response.json()

print(f"{data['regime']} - {data['confidence']*100:.0f}% confidence")
# Risk-Off - 100% confidence
```

## Example: WebSocket Streaming

```python
client = X402WebSocketClient(
    base_url=BASE_URL,
    wallet_address=WALLET_ADDRESS,
    private_key=PRIVATE_KEY,
    network=NETWORK,
)

await client.connect()  # Handles x402 payment automatically
await client.stream()   # Receives updates every 30s
```

## License

TBA
