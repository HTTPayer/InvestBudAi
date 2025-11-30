// src/services/investbudClient.ts
import axios from 'axios';
import { getPaymentBridge } from './paymentBridge.js';

// URL del API original de InvestBud
const INVESTBUD_API_URL = "https://r3467d7khd8b94sfguhrr273lo.ingress.akashprovid.com";

const BASE_URL = process.env.INVESTBUD_BASE_URL || INVESTBUD_API_URL;

// Crear instancia de axios
const client = axios.create({
  baseURL: BASE_URL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Interceptor de respuesta para manejar 402 Payment Required
client.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    
    // Si es 402 y no hemos reintentado ya
    if (error.response?.status === 402 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      try {
        // Obtener payment requirements del 402 response
        const paymentRequirements = error.response.data;
        
        // Solicitar pago via payment bridge (abre navegador con MetaMask)
        const bridge = await getPaymentBridge();
        const paymentHeaderBase64 = await bridge.requestPayment(paymentRequirements);
        
        // El payment bridge devuelve el header completo en base64:
        // { x402Version, scheme, network, payload: { signature, authorization } }
        
        // Reintentar request con el header X-PAYMENT (asegurar que sea string)
        if (!originalRequest.headers) {
          originalRequest.headers = {};
        }
        originalRequest.headers['X-PAYMENT'] = String(paymentHeaderBase64);
        
        return client(originalRequest);
      } catch (paymentError) {
        return Promise.reject(paymentError);
      }
    }
    
    return Promise.reject(error);
  }
);

// Helper: fetch con timeout y soporte x402 automático
async function httpRequest<T>(
  method: "GET" | "POST",
  path: string,
  body?: unknown
): Promise<T> {
  try {
    const response = await client.request<T>({
      method,
      url: path,
      data: body
    });
    
    return response.data;
  } catch (err: any) {
    if (err.code === 'ECONNABORTED') {
      throw new Error(`Request timeout for ${method} ${path}`);
    }
    
    const status = err.response?.status || 'unknown';
    const statusText = err.response?.statusText || '';
    const text = typeof err.response?.data === 'string' 
      ? err.response.data 
      : JSON.stringify(err.response?.data || '');
    
    throw new Error(
      `InvestBud API error on ${method} ${path}: ${status} ${statusText} ${text}`
    );
  }
}

// Tip: los tipos de respuesta los puedes refinar luego según tu OpenAPI.
// Por ahora uso `any` para que compile y funcione.

export async function getMacroRegime(): Promise<any> {
  return httpRequest<any>("GET", "/regime");
}

export interface AnalyzeWalletInput {
  network: string; // e.g. "eth-mainnet", "polygon-mainnet", etc.
  address: string;
}

export async function analyzeWallet(
  input: AnalyzeWalletInput
): Promise<any> {
  // Formato que espera el API de InvestBud según su schema
  return httpRequest<any>("POST", "/portfolio", {
    wallet_address: input.address,
    network: input.network,
    chain_id: getChainId(input.network),
    include_metrics: true
  });
}

// Helper: obtener chain ID según la red
function getChainId(network: string): number {
  const chainIds: Record<string, number> = {
    'ethereum': 1,
    'eth-mainnet': 1,
    'base': 8453,
    'base-mainnet': 8453,
    'polygon': 137,
    'polygon-mainnet': 137,
    'arbitrum': 42161,
    'optimism': 10,
    'base-sepolia': 84532
  };
  return chainIds[network.toLowerCase()] || 8453; // Default Base Mainnet
}

export interface AdvisePortfolioInput {
  network: string;
  address: string;
  riskPreference?: "low" | "medium" | "high";
  notes?: string;
}

export async function advisePortfolio(
  input: AdvisePortfolioInput
): Promise<any> {
  return httpRequest<any>("POST", "/advise", {
    network: input.network,
    address: input.address,
    risk_preference: input.riskPreference,
    notes: input.notes
  });
}

// GET /news - Smart money flow analysis
export async function getNews(): Promise<any> {
  return httpRequest<any>("GET", "/news");
}

// POST /signal - Submit macro regime signal to on-chain oracle
export interface SubmitSignalInput {
  network: string; // e.g. "base-sepolia", "base-mainnet"
}

export async function submitSignal(
  input: SubmitSignalInput
): Promise<any> {
  return httpRequest<any>("POST", "/signal", {
    network: input.network
  });
}
