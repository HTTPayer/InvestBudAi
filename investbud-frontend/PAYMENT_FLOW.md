# Payment Flow - Frontend to Backend

## Flujo completo de pago x402

### 1️⃣ Primera llamada (sin pago)

**Request:**
```http
POST /api/chat
Content-Type: application/json

{
  "session_id": "uuid-here",
  "message": "What's the current macro regime?",
  "metadata": {
    "user_agent": "...",
    "timezone": "...",
    "wallet_address": "0x364307..."
  }
}
```

**Response esperada del backend:**
```http
HTTP/1.1 402 Payment Required
Content-Type: application/json

{
  "status": 402,
  "payment_instructions": {
    "recipient": "0x1d4ba461fdba577dfe0400252cdf462e5a1ff13f",
    "amount": "20000",
    "token": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    "chain_id": 84532,
    "payment_id": "payment-uuid-from-backend"
  },
  "message": "Payment required"
}
```

**O formato x402 estándar:**
```http
HTTP/1.1 402 Payment Required
Content-Type: application/json

{
  "x402Version": 1,
  "accepts": [{
    "payTo": "0x1d4ba461fdba577dfe0400252cdf462e5a1ff13f",
    "maxAmountRequired": "20000",
    "asset": "0x036CbD53842c5426634e7929541eC2318f3dCF7e",
    "network": "base-sepolia",
    "description": "AI Query Payment"
  }],
  "payment_instructions": {
    "payment_id": "payment-uuid-from-backend"
  }
}
```

---

### 2️⃣ Usuario firma en MetaMask

El frontend hace lo siguiente:

1. **Genera la firma EIP-712** (TransferWithAuthorization):
   - Domain: USD Coin v2 en Base Sepolia (chainId: 84532)
   - Message: `{ from, to, value, validAfter, validBefore, nonce }`
   - El usuario ve el popup de MetaMask y firma

2. **Construye el header X-PAYMENT**:
```json
{
  "x402Version": 1,
  "scheme": "exact",
  "network": "base-sepolia",
  "payload": {
    "signature": "0xdcf30008e11f7f4d92aa9577aa86aec3fd97a4a66bbf7cb2ee3bddf784f6241b2177fa119c5256ac334394b61242de5e611a8300c5e3490a13a47e6d3f84528e1c",
    "authorization": {
      "from": "0x364307720164378324965C27FAE21242Fd5807eE",
      "to": "0x1d4ba461fdba577dfe0400252cdf462e5a1ff13f",
      "value": "20000",
      "validAfter": "1764441632",
      "validBefore": "1764443132",
      "nonce": "0x7c237fab3b18db94f9c5974bac484b17b91e6d3293cb80af5bb45d547a5f268d"
    }
  }
}
```

3. **Codifica en base64**: `btoa(JSON.stringify(paymentData))`

---

### 3️⃣ Segunda llamada (con pago firmado)

**Request:**
```http
POST /api/chat
Content-Type: application/json
X-PAYMENT: eyJ4NDAyVmVyc2lvbiI6MSwic2NoZW1lIjoiZXhhY3QiLCJuZXR3b3JrIjoiYmFzZS1zZXBvbGlhIiwicGF5bG9hZCI6eyJzaWduYXR1cmUiOiIweGRjZjMwMDA4ZTExZjdmNGQ5MmFhOTU3N2FhODZhZWMzZmQ5N2E0YTY2YmJmN2NiMmVlM2JkZGY3ODRmNjI0MWIyMTc3ZmExMTljNTI1NmFjMzM0Mzk0YjYxMjQyZGU1ZTYxMWE4MzAwYzVlMzQ5MGExM2E0N2U2ZDNmODQ1MjhlMWMiLCJhdXRob3JpemF0aW9uIjp7ImZyb20iOiIweDM2NDMwNzcyMDE2NDM3ODMyNDk2NUM...

{
  "session_id": "uuid-here",
  "message": "What's the current macro regime?",
  "payment_id": "payment-uuid-from-backend",
  "metadata": {
    "user_agent": "...",
    "timezone": "...",
    "wallet_address": "0x364307..."
  }
}
```

**Response esperada del backend:**
```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "reply": "Based on current market conditions...",
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

---

## ¿Qué debe hacer el backend?

### Al recibir la segunda llamada (con X-PAYMENT):

1. **Decodificar el header X-PAYMENT**:
```python
import base64
import json

payment_header = request.headers.get('X-PAYMENT')
payment_data = json.loads(base64.b64decode(payment_header))

signature = payment_data['payload']['signature']
auth = payment_data['payload']['authorization']
```

2. **Verificar y ejecutar el pago en blockchain**:
```python
# Call USDC contract: receiveWithAuthorization()
usdc_contract.receiveWithAuthorization(
    from=auth['from'],
    to=auth['to'],
    value=auth['value'],
    validAfter=auth['validAfter'],
    validBefore=auth['validBefore'],
    nonce=auth['nonce'],
    v=signature_v,
    r=signature_r,
    s=signature_s
)
```

3. **Si el pago es exitoso**: Procesar el mensaje y devolver respuesta 200
4. **Si el pago falla**: Devolver 402 de nuevo con error específico

---

## Debugging

Si ves el error "HTTP 402: Payment Required" después de firmar:

### Posibles causas:

1. **El backend no está decodificando el header X-PAYMENT correctamente**
2. **El payment_id no coincide** entre las llamadas
3. **La firma es inválida** (problemas con EIP-712)
4. **El contrato rechaza la transacción** (nonce duplicado, balance insuficiente, etc.)

### Logs importantes del frontend:

```
📡 Backend response status: 402
💳 x402 response: {...}
✅ Using payment option from accepts: {...}
🔔 About to trigger MetaMask signature popup...
✅ Signature received: 0xdcf30008...
✅ Payment header created: eyJ4NDAyVmVy...
📨 Sending message with payment_id: payment-uuid
📤 Sending request with payment header...
📥 Response status after payment: 402 ❌ (should be 200)
```

### Verifica en el backend:

1. ¿Recibe el header `X-PAYMENT`?
2. ¿Puede decodificarlo correctamente?
3. ¿El `payment_id` en el body coincide con el generado en el primer 402?
4. ¿La llamada al contrato USDC tiene éxito?
5. ¿El backend devuelve 200 después de un pago exitoso?

---

## Contrato USDC en Base Sepolia

**Address**: `0x036CbD53842c5426634e7929541eC2318f3dCF7e`

**Función a llamar**: `receiveWithAuthorization()`

**ABI snippet**:
```json
{
  "name": "receiveWithAuthorization",
  "type": "function",
  "inputs": [
    { "name": "from", "type": "address" },
    { "name": "to", "type": "address" },
    { "name": "value", "type": "uint256" },
    { "name": "validAfter", "type": "uint256" },
    { "name": "validBefore", "type": "uint256" },
    { "name": "nonce", "type": "bytes32" },
    { "name": "v", "type": "uint8" },
    { "name": "r", "type": "bytes32" },
    { "name": "s", "type": "bytes32" }
  ]
}
```

**¿Cómo separar la firma?**:
```python
signature = "0xdcf30008...1c"  # 132 chars (0x + 130 hex)
r = signature[0:66]   # First 32 bytes
s = "0x" + signature[66:130]  # Next 32 bytes
v = int(signature[130:132], 16)  # Last byte
```
