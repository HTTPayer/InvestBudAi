# Backend x402 Response Format

Tu backend debe devolver este formato exacto cuando recibe el primer request:

## Primera llamada (sin X-PAYMENT header):

**Status Code:** `402`

**Body:**
```json
{
  "status": 402,
  "payment_instructions": {
    "recipient": "0xYourRelayWalletAddress",
    "amount": "1000000",
    "token": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    "chain_id": 84532,
    "payment_id": "unique-payment-id-123"
  },
  "message": "Payment required"
}
```

**Importante:**
- `amount` debe ser string en wei (para USDC con 6 decimales: "1000000" = 1 USDC)
- `token` es la dirección del contrato USDC
- `chain_id` debe ser 84532 para Base Sepolia
- `recipient` es la wallet que recibirá los fondos (tu relay)

## Segunda llamada (con X-PAYMENT header):

**Headers:**
```
X-PAYMENT: eyJ4NDAyVmVyc2lvbiI6MSwic2NoZW1lIjoiZXhhY3Qi...
```

**Status Code:** `200`

**Body:**
```json
{
  "reply": "Your AI response here",
  "regime_signal": {
    "current": "risk-on",
    "last_updated": "2025-11-29T10:00:00Z"
  },
  "portfolio_summary": {
    "total_value_usd": 10000,
    "top_holdings": [
      { "symbol": "BTC", "percentage": 45.5 },
      { "symbol": "ETH", "percentage": 30.2 }
    ]
  }
}
```

El backend debe:
1. Decodificar el header X-PAYMENT (base64)
2. Extraer la firma y autorización
3. Llamar al contrato USDC con `receiveWithAuthorization()`
4. Procesar y responder
