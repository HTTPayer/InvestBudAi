# 🚀 Deployment Guide

## Overview

This guide covers deploying the InvestBud MCP Server to various platforms so anyone in the world can connect to it and use the tools with x402 payments.

## Prerequisites

Before deploying:

1. ✅ Build locally and test: `npm run build && npm start`
2. ✅ Have a wallet with USDC on Base network
3. ✅ Set `WALLET_PRIVATE_KEY` environment variable
4. ✅ Test x402 payments work locally

## Deployment Options

### 🐳 Docker (Recommended for Production)

**Build and run locally:**

```bash
# Build image
npm run docker:build

# Run container
npm run docker:run

# Or use docker-compose
npm run docker:compose
```

**Deploy to any Docker host:**

```bash
# Build
docker build -t investbud-mcp .

# Run
docker run -d \
  -p 3030:3030 \
  -e WALLET_PRIVATE_KEY=your_key \
  --name investbud-mcp \
  investbud-mcp
```

### ☁️ Fly.io (Best for Global Edge Deployment)

**Why Fly.io:**
- ✅ Global edge network
- ✅ Automatic HTTPS
- ✅ $0 free tier
- ✅ WebSocket support

**Deploy:**

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login
flyctl auth login

# Launch (first time)
flyctl launch

# Set secrets
flyctl secrets set WALLET_PRIVATE_KEY=your_private_key_here

# Deploy
flyctl deploy

# Check status
flyctl status

# View logs
flyctl logs

# Get URL
flyctl info
```

**Your MCP endpoint:** `https://your-app.fly.dev/mcp`

**Connect from anywhere:**

```json
{
  "mcpServers": {
    "investbud-remote": {
      "url": "https://your-app.fly.dev/mcp",
      "transport": "http"
    }
  }
}
```

### 🚂 Railway (Easiest Deploy)

**Why Railway:**
- ✅ GitHub auto-deploy
- ✅ Super simple UI
- ✅ Built-in env vars
- ✅ Free $5/month credit

**Deploy:**

1. Go to [railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your forked repo
5. Add environment variable:
   - Key: `WALLET_PRIVATE_KEY`
   - Value: Your private key
6. Click "Deploy"

**Your MCP endpoint:** `https://your-app.up.railway.app/mcp`

### 🎨 Render (Free with Auto-Deploy)

**Why Render:**
- ✅ Free tier available
- ✅ GitHub auto-deploy
- ✅ Easy SSL
- ✅ Good performance

**Deploy:**

1. Go to [render.com](https://render.com)
2. New → Web Service
3. Connect GitHub repo
4. Use these settings:
   - Build Command: `npm install && npm run build`
   - Start Command: `npm start`
5. Add environment variable:
   - Key: `WALLET_PRIVATE_KEY`
   - Value: Your private key
6. Create Web Service

**Your MCP endpoint:** `https://your-app.onrender.com/mcp`

### 🔷 Vercel (Serverless)

**Why Vercel:**
- ✅ Instant deploys
- ✅ Great DX
- ✅ Edge network

**Note:** Vercel has 10s timeout on hobby plan, which might be too short for some tools. Consider Fly.io or Railway for production.

**Deploy:**

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Set secret
vercel env add WALLET_PRIVATE_KEY

# Redeploy
vercel --prod
```

### 🔵 DigitalOcean App Platform

**Why DigitalOcean:**
- ✅ Simple pricing
- ✅ Good performance
- ✅ Docker support

**Deploy:**

1. Go to [digitalocean.com/apps](https://cloud.digitalocean.com/apps)
2. Create App
3. Connect GitHub
4. Configure:
   - Run Command: `npm start`
   - HTTP Port: `3030`
5. Add environment variables
6. Launch

### 🏠 Self-Hosted (VPS)

**Why Self-Host:**
- ✅ Full control
- ✅ No platform limits
- ✅ Best performance

**Deploy on any VPS (AWS, GCP, Azure, etc):**

```bash
# SSH into your server
ssh user@your-server.com

# Clone repo
git clone https://github.com/HTTPayer/investbud-MCP-Server.git
cd investbud-MCP-Server

# Install dependencies
npm install

# Build
npm run build

# Create .env file
echo "WALLET_PRIVATE_KEY=your_key" > .env

# Install PM2 for process management
npm install -g pm2

# Start server
pm2 start dist/server.js --name investbud-mcp

# Setup auto-restart on reboot
pm2 startup
pm2 save

# Setup Nginx reverse proxy (optional)
sudo apt install nginx
sudo nano /etc/nginx/sites-available/investbud
```

**Nginx config:**

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3030;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## Security Considerations

### 🔐 Environment Variables

**NEVER commit these:**
- ✅ Keep `WALLET_PRIVATE_KEY` in environment only
- ✅ Use platform secrets management
- ✅ Rotate keys regularly

### 🛡️ Rate Limiting

Consider adding rate limiting for public deployments:

```typescript
// Add to server.ts
import rateLimit from 'express-rate-limit';

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});

app.use('/mcp', limiter);
```

### 🔒 Authentication (Optional)

Add API key auth for production:

```typescript
// Add to server.ts
const apiKeyAuth = (req, res, next) => {
  const apiKey = req.headers['x-api-key'];
  if (!process.env.API_KEY || apiKey === process.env.API_KEY) {
    next();
  } else {
    res.status(401).json({ error: 'Unauthorized' });
  }
};

app.use('/mcp', apiKeyAuth);
```

## Monitoring

### Health Checks

All platforms will use: `GET /`

Response:
```json
{
  "status": "ok",
  "server": "investbud-mcp",
  "version": "0.1.0"
}
```

### Logs

**Fly.io:**
```bash
flyctl logs
```

**Railway:**
```bash
railway logs
```

**Render:**
Check dashboard logs

**PM2 (Self-hosted):**
```bash
pm2 logs investbud-mcp
```

## Cost Estimates

| Platform | Free Tier | Paid (Low Traffic) | Notes |
|----------|-----------|-------------------|-------|
| Fly.io | 3 VMs free | ~$2-5/mo | Best value |
| Railway | $5 credit/mo | ~$5-10/mo | Easiest |
| Render | 750 hrs/mo | $7/mo | Good free tier |
| DigitalOcean | - | $5/mo | Droplet |
| Vercel | Hobby free | $20/mo | 10s timeout |

## Making Money with x402

### Revenue Model

1. **Deploy the server publicly**
2. **Users pay per tool usage** (via x402)
3. **You receive payments** in your wallet
4. **No payment processing fees** (direct crypto)

### Pricing Strategy

Current InvestBud pricing:
- `get_news`: $0.10 USDC
- `submit_signal`: $0.10 USDC
- Other tools: Free (for now)

**You can:**
- Keep default pricing
- Add markup for your service
- Create subscription plans
- Bundle tools together

### Marketing Your Deployment

1. **Share your MCP endpoint URL**
2. **Create tutorials** on using the tools
3. **Join MCP communities** (Discord, Reddit)
4. **Add to MCP directory** (when available)
5. **Tweet about it** with #MCP #x402

### Example Promotion

```
🚀 New: InvestBud MCP Server - AI Crypto Tools!

Get macro regime analysis, smart money flows, and portfolio advice via Claude/MCP.

💰 Pay-per-use with $USDC (x402)
🌐 Global deployment: https://your-app.fly.dev/mcp
📖 Docs: github.com/HTTPayer/investbud-MCP-Server

#MCP #AI #Crypto #DeFi
```

## Testing Your Deployment

### 1. Health Check

```bash
curl https://your-app.fly.dev/
```

Should return:
```json
{"status":"ok","server":"investbud-mcp","version":"0.1.0"}
```

### 2. MCP Connection

Add to Claude Desktop config:

```json
{
  "mcpServers": {
    "investbud-prod": {
      "url": "https://your-app.fly.dev/mcp",
      "transport": "http"
    }
  }
}
```

### 3. Test a Tool

In Claude:
```
"What's the current macro regime?"
```

Should trigger payment flow and return data.

## Troubleshooting

### 502 Bad Gateway
- Check logs for startup errors
- Verify `PORT` environment variable
- Ensure build completed successfully

### Payment Failures
- Verify `WALLET_PRIVATE_KEY` is set
- Check wallet has USDC on Base
- Try on Base Sepolia first (testnet)

### Slow Responses
- Check server location (use Fly.io edge)
- Increase timeout limits
- Monitor InvestBud API latency

### Tools Not Appearing
- Check MCP endpoint is accessible
- Verify transport type is correct
- Check Claude Desktop logs

## Next Steps

After deployment:

1. ✅ Test all tools end-to-end
2. ✅ Set up monitoring/alerting
3. ✅ Share your endpoint URL
4. ✅ Start earning from x402 payments!
5. ✅ Join MCP community
6. ✅ Contribute improvements

## Support

- GitHub Issues: [investbud-MCP-Server/issues](https://github.com/HTTPayer/investbud-MCP-Server/issues)
- MCP Discord: [Join](https://discord.gg/mcp)
- x402 Docs: [x402.org](https://x402.org)

---

**Ready to deploy? Pick a platform above and let's go! 🚀**
