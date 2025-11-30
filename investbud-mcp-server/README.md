# InvestBud MCP Server 🚀

A Model Context Protocol (MCP) server that provides AI-powered crypto investment tools with automatic x402 payments. Get macro regime analysis, smart money flows, wallet analytics, and on-chain oracle integration—all with seamless crypto payments via MetaMask.

## 🌟 Features

### 🔧 Tools Available

1. **get_macro_regime** - Get current market regime (Risk-On/Risk-Off) with probabilities
2. **analyze_wallet** - Deep analysis of wallet composition and risk metrics
3. **advise_portfolio** - AI-powered portfolio recommendations with GPT-4
4. **get_news** - Smart money flow analysis from Nansen + market news ($0.10 USDC)
5. **submit_signal** - Publish macro regime to on-chain oracle ($0.10 USDC)

### 💡 Smart Prompts

Pre-built workflows for common use cases:

- **market_analysis** - Complete market overview (regime + smart money)
- **wallet_deep_dive** - Comprehensive wallet analysis with recommendations
- **smart_money_tracker** - Track institutional money movements
- **publish_oracle_signal** - Submit signals to blockchain oracles
- **portfolio_rebalance** - Get rebalancing strategies
- **regime_change_alert** - Market regime change notifications

### 💳 x402 Payment Integration

- **Automatic payment handling** - Server intercepts 402 responses
- **MetaMask integration** - Payment bridge opens browser for authorization
- **Seamless UX** - Users approve once, tools work automatically
- **Transparent pricing** - Clear costs displayed upfront

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- MetaMask wallet
- USDC on Base Sepolia (for testing) or Base Mainnet

### Installation

```bash
# Clone the repository
git clone https://github.com/HTTPayer/investbud-MCP-Server.git
cd investbud-MCP-Server

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Add your wallet private key (for x402 payments)
echo "WALLET_PRIVATE_KEY=your_private_key_here" >> .env
```

### Usage with Claude Desktop

1. Build the server:
```bash
npm run build
```

2. Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "investbud": {
      "command": "node",
      "args": [
        "/absolute/path/to/investbud-MCP-Server/dist/server.js",
        "--stdio"
      ]
    }
  }
}
```

3. Restart Claude Desktop

4. Start using the tools! Try:
   - "What's the current macro regime?"
   - "Analyze wallet 0x... on base-mainnet"
   - "Show me the latest smart money flows"

### HTTP Mode (Remote Access)

Run as a web server for remote MCP clients:

```bash
# Start the server
PORT=3030 npm start

# Server runs at http://localhost:3030
# MCP endpoint: POST http://localhost:3030/mcp
```

## 📚 API Endpoints

### Tools

#### `get_macro_regime`
Get current macro regime classification.

```typescript
// No parameters
// Returns: { regime: "risk_on" | "risk_off", probability: number, features: {...} }
```

#### `analyze_wallet`
Analyze a crypto wallet's composition.

```typescript
{
  network: "base-mainnet" | "eth-mainnet" | "polygon-mainnet" | "arbitrum",
  address: "0x..." // Wallet address
}
// Returns: Portfolio composition, risk metrics, token holdings
```

#### `advise_portfolio`
Get AI-powered investment advice.

```typescript
{
  network: string,
  address: string,
  riskPreference?: "low" | "medium" | "high",
  notes?: string // Additional context
}
// Returns: Personalized recommendations, rebalancing suggestions
```

#### `get_news`
Get smart money flow analysis. **Cost: $0.10 USDC**

```typescript
// No parameters
// Returns: Top 100 tokens by netflow, accumulation patterns, GPT-4 analysis
```

#### `submit_signal`
Publish macro signal to on-chain oracle. **Cost: $0.10 USDC**

```typescript
{
  network: "base-sepolia" | "base-mainnet"
}
// Returns: Signed oracle update, transaction details
```

## 🏗️ Architecture

```
┌─────────────────┐
│  Claude/Client  │
└────────┬────────┘
         │ MCP Protocol
┌────────▼────────┐
│   MCP Server    │
│  (this repo)    │
└────────┬────────┘
         │
    ┌────▼─────┬──────────────┐
    │          │              │
┌───▼────┐ ┌──▼───────┐ ┌───▼──────┐
│InvestBud│ │ Payment  │ │ MetaMask │
│   API   │ │  Bridge  │ │ (x402)   │
└─────────┘ └──────────┘ └──────────┘
```

### Payment Flow

1. Tool called → Request to InvestBud API
2. API returns `402 Payment Required` with payment details
3. Payment Bridge opens MetaMask in browser
4. User approves payment signature
5. Request retried with payment header
6. API returns data

## 🔐 Security

- Private keys stored in `.env` (never committed)
- Payment signatures validated on-chain
- All API calls over HTTPS
- MetaMask manages key security

## 💰 Pricing

| Tool | Cost |
|------|------|
| get_macro_regime | Free |
| analyze_wallet | Free |
| advise_portfolio | Free |
| get_news | $0.10 USDC |
| submit_signal | $0.10 USDC |

Payments processed via x402 on Base network.

## 🌐 Deployment

### Deploy to Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Create app
flyctl launch

# Set environment variables
flyctl secrets set WALLET_PRIVATE_KEY=your_key

# Deploy
flyctl deploy
```

### Deploy to Railway

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Deploy
railway up
```

### Deploy to Render

1. Connect your GitHub repo
2. Create new Web Service
3. Set build command: `npm install && npm run build`
4. Set start command: `npm start`
5. Add environment variable: `WALLET_PRIVATE_KEY`

## 🔧 Development

```bash
# Development mode (auto-reload)
npm run dev

# Build
npm run build

# Run tests
npm test

# Type check
npx tsc --noEmit
```

## 📝 Example Usage

### Get Market Analysis

```typescript
// In Claude or any MCP client:
"Use the market_analysis prompt to give me a complete market overview"

// This will:
// 1. Get macro regime
// 2. Get smart money flows
// 3. Synthesize into actionable insights
```

### Analyze a Wallet

```typescript
"Use wallet_deep_dive for address 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb with network base-mainnet"

// Returns:
// - Portfolio composition
// - Risk metrics
// - Personalized advice
// - Comparison with smart money
```

### Track Smart Money

```typescript
"Use smart_money_tracker to show me what institutions are buying"

// Returns:
// - Top accumulation tokens
// - Distribution patterns
// - Opportunities aligned with macro
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file

## 🆘 Support

- Issues: [GitHub Issues](https://github.com/HTTPayer/investbud-MCP-Server/issues)
- Docs: [InvestBud API Docs](https://investbud.xyz/docs)
- x402: [x402 Protocol](https://x402.org)

## 🙏 Acknowledgments

- Built with [MCP SDK](https://github.com/modelcontextprotocol/sdk)
- Powered by [InvestBud API](https://investbud.xyz)
- Payments via [x402 Protocol](https://x402.org)
- Uses [Nansen](https://nansen.ai) smart money data

---

Made with ❤️ for the crypto community

**Start earning with x402 payments today!** 🚀💰
