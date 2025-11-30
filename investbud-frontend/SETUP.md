# Investbud Landing - Setup Guide

## 🚀 Funcionalidades Implementadas

### 1. **Conexión con MetaMask**
- Detección automática de MetaMask
- Conexión/desconexión de wallet
- Auto-switch a Base Sepolia
- Muestra dirección conectada en el header

### 2. **Flujo x402 Payment**
- Llama al endpoint para obtener instrucciones de pago
- Ejecuta pago en USDC en Base Sepolia
- Envía prueba de pago al backend
- Manejo de errores completo

### 3. **RAG + LLM con OpenAI**
- Procesa respuestas del backend con GPT-4
- Alimenta contexto desde cookies y localStorage del usuario
- Guarda historial de conversación en sessionStorage
- Personaliza respuestas basadas en contexto del usuario

## 📋 Configuración

### 1. Instalar Dependencias
```bash
npm install
```

### 2. Configurar Variables de Entorno

Crea un archivo `.env.local` en la raíz del proyecto:

```bash
# Backend URL (ya configurado)
NEXT_PUBLIC_BACKEND_URL=https://r3467d7khd8b94sfguhrr273lo.ingress.akashprovid.com

# OpenAI API Key (REQUERIDO)
NEXT_PUBLIC_OPENAI_API_KEY=sk-...tu_api_key_aqui
```

⚠️ **IMPORTANTE**: Necesitas agregar tu OpenAI API Key

### 3. Ejecutar en Desarrollo
```bash
npm run dev
```

## 🔧 Estructura de Archivos Creados/Modificados

```
/components
  └── InvestbudChat.tsx          # Componente principal actualizado

/hooks
  └── useMetaMask.ts             # Hook para MetaMask

/lib
  ├── constants.ts               # Configuración de Base Sepolia y USDC
  ├── x402.ts                    # Funciones para x402 payment
  └── rag.ts                     # Funciones RAG/LLM con OpenAI

/.env.local.example              # Ejemplo de variables de entorno
```

## 🌊 Flujo de Uso

1. **Usuario conecta MetaMask**: Click en "Connect Wallet"
2. **Sistema verifica red**: Auto-switch a Base Sepolia
3. **Usuario escribe mensaje**: Input habilitado
4. **Sistema obtiene instrucciones**: Llamada x402 al backend
5. **Usuario aprueba pago**: MetaMask popup para transferir USDC
6. **Sistema confirma pago**: Envía tx hash al backend
7. **Backend responde**: Retorna datos del chat
8. **RAG procesa respuesta**: OpenAI enriquece con contexto del usuario
9. **Muestra respuesta final**: Con régimen de mercado y portfolio

## 🔐 Seguridad

- ⚠️ OpenAI API Key en frontend es **temporal** para prototipado
- Para producción: mover RAG/LLM a API route o servidor
- USDC en Base Sepolia es testnet (sin valor real)

## 💰 Testnet Tokens

Para testear, necesitas:
- ETH en Base Sepolia (para gas): https://www.alchemy.com/faucets/base-sepolia
- USDC en Base Sepolia: https://faucet.circle.com/

## 📝 Notas Importantes

1. **Dirección del contrato USDC**: `0x036CbD53842c5426634e7929541eC2318f3dCF7e`
2. **Chain ID Base Sepolia**: `84532`
3. **Cada mensaje requiere pago en USDC**
4. **El RAG usa cookies/localStorage del navegador del usuario**

## 🐛 Troubleshooting

### Error: "MetaMask is not installed"
- Instala MetaMask: https://metamask.io

### Error: "Insufficient USDC balance"
- Consigue USDC testnet del faucet de Circle

### Error: OpenAI API
- Verifica que tu API key esté en `.env.local`
- Revisa que tengas créditos en tu cuenta OpenAI

### Error: Network
- Asegúrate de estar en Base Sepolia
- El sistema debería cambiarte automáticamente

## 🚀 Próximos Pasos (Opcional)

1. **Mover OpenAI a API Route** (recomendado para producción)
2. **Implementar cache de respuestas**
3. **Agregar vector store para RAG más robusto**
4. **Implementar retry logic para pagos fallidos**
5. **Agregar analytics de pagos**

---

¿Necesitas ayuda? Revisa los logs del navegador (F12 → Console) para ver errores detallados.
