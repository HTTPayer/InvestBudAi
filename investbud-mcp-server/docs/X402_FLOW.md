# x402 Payment Flow - InvestBud MCP Server

## Architecture Overview

The system uses **manual x402 implementation** with the **Payment Bridge** for browser-based wallet interaction.

```
User Request → InvestBud API
     ↓
402 Payment Required
     ↓
Payment Bridge (Local Server)
     ↓
Browser Opens → MetaMask Sign
     ↓
X-PAYMENT Header Generated
     ↓
Retry Request → Success
```

## How It Works

### 1. Initial Request
When you make a request to InvestBud API:
```typescript
await analyzeWallet({ network: "base-sepolia", address: "0x..." })
```

### 2. 402 Response
The API returns `402 Payment Required` with payment requirements:
```json
{
  "x402Version": 1,
  "accepts": [{
    "scheme": "exact",
    "network": "base-sepolia",
    "payTo": "0xRecipientAddress...",
    "asset": "0xUSDCAddress...",
    "maxAmountRequired": "100000"
  }]
}
```

### 3. Payment Bridge Activation
The axios interceptor detects the 402 and:
- Starts local server (port 3402-3502)
- Opens browser to `http://localhost:3402/pay/{sessionId}`
- User connects MetaMask and signs payment authorization

### 4. Payment Header Construction
Browser JavaScript constructs the x402 payment header:

```javascript
const paymentData = {
  x402Version: 1,
  scheme: "exact",
  network: "base-sepolia",  // From 402 response
  payload: {
    signature: "0x1234...",  // From MetaMask signature
    authorization: {
      from: "0xUserWallet...",      // User's wallet address
      to: "0xRecipientAddress...",  // From 402 response
      value: "100000",              // From 402 response
      validAfter: "1732800000",     // Current timestamp
      validBefore: "1732800900",    // +15 minutes
      nonce: "0xabc123..."          // Random 32 bytes
    }
  }
};

// Encode to base64
const paymentHeader = btoa(JSON.stringify(paymentData));
```

### 5. Request Retry
The interceptor retries the original request with:

**Headers:**
```
X-PAYMENT: eyJ4NDAyVmVyc2lvbiI6MSwic2NoZW1lIjoiZXhhY3QiLCJuZXR3b3JrIjoi...
Content-Type: application/json
```

**Body:**
```json
{
  "network": "base-sepolia",
  "address": "0xUserWallet..."
}
```

### 6. Success
The API validates the payment and returns the requested data.

## Code Structure

### `investbudClient.ts`
- Creates axios instance
- Adds 402 interceptor
- Handles payment bridge integration
- Retries with X-PAYMENT header

### `paymentBridge.ts`
- Local Express server (3402-3502)
- Serves payment UI HTML
- Manages payment sessions
- Returns signed payment header

### `x402Client.ts`
- Helper functions for x402
- Payment info extraction
- Revocation support

## Payment Header Structure

The `X-PAYMENT` header contains a **base64-encoded JSON object**:

```json
{
  "x402Version": 1,
  "scheme": "exact",
  "network": "base-sepolia",
  "payload": {
    "signature": "0x...",
    "authorization": {
      "from": "0x...",    // Payer address
      "to": "0x...",      // Recipient address  
      "value": "100000",  // Amount (string)
      "validAfter": "...", // Unix timestamp (string)
      "validBefore": "...", // Unix timestamp (string)
      "nonce": "0x..."    // 32-byte hex string
    }
  }
}
```

## Request Payload

The request body contains:
```json
{
  "network": "base-sepolia",  // Chain identifier
  "address": "0x..."          // User's wallet address
}
```

## Why Not Use x402-axios?

The `x402-axios` library expects a **viem wallet client** with private keys for automatic signing. This doesn't work for our use case because:

1. **No browser access** - MCP server runs in Node.js
2. **User interaction required** - We need MetaMask for signatures
3. **No private keys** - Users shouldn't share private keys

Our **Payment Bridge** solution:
- ✅ Opens browser for MetaMask interaction
- ✅ Secure - no private keys in Node.js
- ✅ User-friendly - familiar MetaMask UI
- ✅ Works with Claude Desktop

## Testing

```bash
# Build the server
npm run build

# Restart Claude Desktop

# Test with a query:
"Analyze wallet 0x364307720164378324965c27fae21242fd5807ee on base-sepolia"
```

Browser will open automatically for payment approval.
