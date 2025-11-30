// src/services/x402Client.ts
// Abstracción para manejar la capa de pago x402 como CLIENTE.
// Integración con HTTPayer relay endpoint (relay.httpayer.com)

import { getPaymentBridge } from './paymentBridge.js';

/**
 * Esquema x402 PaymentRequirements devuelto por el relay en 402
 */
export interface X402PaymentRequirements {
  x402Version: number;
  accepts: Array<{
    scheme: string;           // "exact"
    network: string;          // "base", "ethereum", "solana", etc.
    payTo: string;            // Dirección del destinatario
    asset: string;            // Token/asset a pagar
    maxAmountRequired: string; // Monto total (incluye fee del relay 3% + $0.002 min)
    resource?: string;        // Recurso que se está pagando
    description?: string;     // Descripción del pago
    mimeType?: string;        // "application/json"
    maxTimeoutSeconds?: number; // 180 por defecto
  }>;
}

/**
 * Interfaz genérica para mantener compatibilidad
 */
export interface PaymentRequirements {
  [key: string]: unknown;
}

/**
 * Información del pago realizado para poder revocarlo después
 */
export interface PaymentInfo {
  network: string;
  spender: string;
  asset: string;
  userAddress: string;
}

/**
 * Dado un objeto PaymentRequirements devuelto por el servidor (402),
 * crea y firma un payload de pago compatible con x402 y devuélvelo
 * como string para el header `X-PAYMENT`.
 *
 * FLUJO:
 * 1. El relay devuelve 402 con instrucciones (accepts array)
 * 2. Esta función debe comunicarse con tu wallet para firmar el pago
 * 3. Tu wallet devuelve el header X-PAYMENT firmado
 * 4. El MCP server reintenta con ese header
 *
 * IMPORTANTE: Aquí debes integrar tu wallet (HTTPayer SDK, wallet extension, etc.)
 * 
 * Opciones de integración:
 * - HTTPayer SDK (si existe): import { signPayment } from '@httpayer/sdk'
 * - Wallet browser extension: window.ethereum o window.solana
 * - Wallet API/RPC directa
 * - Facilitator service
 */
export async function fulfillX402Payment(
  requirements: PaymentRequirements
): Promise<{ paymentHeader: string; paymentInfo: PaymentInfo }> {
  const x402Req = requirements as unknown as X402PaymentRequirements;

  // Validar que tenemos el formato correcto
  if (!x402Req.accepts || !Array.isArray(x402Req.accepts) || x402Req.accepts.length === 0) {
    throw new Error("Invalid x402 payment requirements: missing 'accepts' array");
  }

  // Tomar la primera opción de pago disponible
  const paymentOption = x402Req.accepts[0];
  
  // ============================================================================
  // PAYMENT BRIDGE - Abre navegador para aprobar pago
  // ============================================================================
  
  try {
    // Obtener el bridge (inicia servidor local si no está corriendo)
    const bridge = await getPaymentBridge();
    
    // Solicitar pago - esto abrirá el navegador y esperará la aprobación
    const paymentHeader = await bridge.requestPayment(x402Req);
    
    // Extraer información del pago para poder revocarlo después
    // El paymentHeader completo está codificado en base64
    // Estructura: { x402Version, scheme, network, payload: { signature, authorization } }
    const decodedHeader = JSON.parse(Buffer.from(paymentHeader, 'base64').toString());
    const paymentInfo: PaymentInfo = {
      network: decodedHeader.network,                          // Del header decodificado
      spender: decodedHeader.payload.authorization.to,         // Del authorization en payload
      asset: paymentOption.asset,                              // Del 402 response
      userAddress: decodedHeader.payload.authorization.from    // Del authorization en payload
    };
    
    return { paymentHeader, paymentInfo };
  } catch (error: any) {
    // Si falla el bridge, intentar fallback a wallet de navegador
    if (typeof window !== 'undefined') {
      const paymentHeader = await attemptBrowserWalletPayment(paymentOption);
      const decodedHeader = JSON.parse(Buffer.from(paymentHeader, 'base64').toString());
      const paymentInfo: PaymentInfo = {
        network: decodedHeader.network,
        spender: decodedHeader.payload.authorization.to,
        asset: paymentOption.asset,
        userAddress: decodedHeader.payload.authorization.from
      };
      return { paymentHeader, paymentInfo };
    }
    
    throw error;
  }
}

/**
 * Fallback: Intenta usar wallets de navegador directamente (solo si estamos en browser)
 */
async function attemptBrowserWalletPayment(paymentOption: any): Promise<string> {
  // Determinar qué wallet usar según la red
  const isEVMNetwork = ['ethereum', 'base', 'polygon', 'arbitrum', 'optimism'].includes(
    paymentOption.network.toLowerCase()
  );
  const isPolkadotNetwork = ['polkadot', 'kusama', 'westend'].includes(
    paymentOption.network.toLowerCase()
  );

  // METAMASK - Para redes EVM
  if (isEVMNetwork && (window as any).ethereum) {
    return await signWithMetaMask(paymentOption);
  }

  // POLKADOT.JS - Para redes Polkadot/Substrate
  if (isPolkadotNetwork && (window as any).injectedWeb3?.['polkadot-js']) {
    return await signWithPolkadotJS(paymentOption);
  }

  // Si no hay wallet disponible
  throw new Error(
    `No compatible wallet found for network: ${paymentOption.network}\n` +
    `Please install:\n` +
    `- MetaMask for EVM networks (Ethereum, Base, Polygon, etc.)\n` +
    `- Polkadot.js Extension for Polkadot/Substrate networks`
  );
}

/**
 * Opcional: Helper para calcular el fee del relay HTTPayer
 * Formula: max(3% × target_amount, $0.002)
 */
export function calculateRelayFee(targetAmountUSD: number): number {
  const threePercent = targetAmountUSD * 0.03;
  const minFee = 0.002;
  return Math.max(threePercent, minFee);
}

/**
 * Opcional: Helper para calcular el total que necesitas pagar
 * (target amount + relay fee)
 */
export function calculateTotalAmount(targetAmountUSD: number): number {
  return targetAmountUSD + calculateRelayFee(targetAmountUSD);
}

/**
 * Helper para obtener chainId desde el nombre de la red
 */
function getChainIdFromNetwork(network: string): number {
  const chainIds: Record<string, number> = {
    'base': 8453,
    'base-mainnet': 8453,
    'base-sepolia': 84532,
    'ethereum': 1,
    'eth-mainnet': 1,
    'polygon': 137,
    'polygon-mainnet': 137,
    'arbitrum': 42161,
    'optimism': 10
  };
  return chainIds[network.toLowerCase()] || 8453; // Default Base
}

// ============================================================================
// WALLET INTEGRATIONS
// ============================================================================

/**
 * Firma un pago usando MetaMask (para redes EVM) - EIP-3009 TransferWithAuthorization
 */
async function signWithMetaMask(paymentOption: any): Promise<string> {
  const ethereum = (window as any).ethereum;
  
  try {
    // 1. Solicitar acceso a la cuenta
    const accounts = await ethereum.request({ 
      method: 'eth_requestAccounts' 
    });
    const userAddress = accounts[0];

    // 2. Generar nonce aleatorio (32 bytes)
    const nonceBytes = new Uint8Array(32);
    crypto.getRandomValues(nonceBytes);
    const nonce = '0x' + Array.from(nonceBytes).map(b => b.toString(16).padStart(2, '0')).join('');

    // 3. Timestamps
    const validAfter = Math.floor(Date.now() / 1000);
    const validBefore = validAfter + 900; // Válido por 15 minutos

    // 4. Construir el mensaje EIP-712 para TransferWithAuthorization
    const domain = {
      name: paymentOption.extra?.name || 'USD Coin',
      version: paymentOption.extra?.version || '2',
      chainId: getChainIdFromNetwork(paymentOption.network),
      verifyingContract: paymentOption.asset
    };

    const types = {
      TransferWithAuthorization: [
        { name: 'from', type: 'address' },
        { name: 'to', type: 'address' },
        { name: 'value', type: 'uint256' },
        { name: 'validAfter', type: 'uint256' },
        { name: 'validBefore', type: 'uint256' },
        { name: 'nonce', type: 'bytes32' }
      ]
    };

    const message = {
      from: userAddress,
      to: paymentOption.payTo,
      value: paymentOption.maxAmountRequired,
      validAfter: validAfter,
      validBefore: validBefore,
      nonce: nonce
    };

    // 5. Firmar con eth_signTypedData_v4
    const signature = await ethereum.request({
      method: 'eth_signTypedData_v4',
      params: [userAddress, JSON.stringify({ domain, types, primaryType: 'TransferWithAuthorization', message })]
    });

    // 6. Construir el payment header con formato x402 completo
    const paymentData = {
      x402Version: 1,
      scheme: 'exact',
      network: paymentOption.network,
      payload: {
        signature: signature,
        authorization: {
          from: userAddress,
          to: paymentOption.payTo,
          value: String(paymentOption.maxAmountRequired),
          validAfter: String(validAfter),
          validBefore: String(validBefore),
          nonce: nonce
        }
      }
    };
    
    const paymentHeader = Buffer.from(JSON.stringify(paymentData)).toString('base64');
    return paymentHeader;

  } catch (error: any) {
    throw new Error(`MetaMask payment failed: ${error.message}`);
  }
}

/**
 * Firma un pago usando Polkadot.js Extension (para redes Substrate)
 */
async function signWithPolkadotJS(paymentOption: any): Promise<string> {
  const injectedWeb3 = (window as any).injectedWeb3;
  
  try {
    // 1. Obtener la extensión Polkadot.js
    const extension = injectedWeb3['polkadot-js'];
    await extension.enable('InvestBud MCP');

    // 2. Obtener cuentas disponibles
    const allAccounts = await extension.accounts.get();
    if (allAccounts.length === 0) {
      throw new Error('No Polkadot accounts found');
    }
    const userAccount = allAccounts[0];

    // 3. Crear el mensaje de pago según el formato x402
    const paymentMessage = {
      version: 1,
      from: userAccount.address,
      to: paymentOption.payTo,
      asset: paymentOption.asset,
      amount: paymentOption.maxAmountRequired,
      network: paymentOption.network,
      timestamp: Date.now(),
      description: paymentOption.description || 'HTTPayer relay payment'
    };

    // 4. Firmar el mensaje con Polkadot.js
    const message = JSON.stringify(paymentMessage);
    const signRaw = extension.signer?.signRaw;
    
    if (!signRaw) {
      throw new Error('Polkadot.js signing not available');
    }

    const { signature } = await signRaw({
      address: userAccount.address,
      data: Buffer.from(message).toString('hex'),
      type: 'bytes'
    });

    // 5. Construir el header X-PAYMENT con formato x402 completo
    const paymentData = {
      x402Version: 1,
      scheme: 'exact',
      network: paymentOption.network,
      payload: {
        signature: signature,
        authorization: paymentMessage
      }
    };
    
    const paymentHeader = Buffer.from(JSON.stringify(paymentData)).toString('base64');
    return paymentHeader;

  } catch (error: any) {
    throw new Error(`Polkadot.js payment failed: ${error.message}`);
  }
}

/**
 * Revoca la aprobación de un spender para un token ERC20
 * Establece el allowance en 0 para cerrar el spending cap
 */
export async function revokeApproval(paymentInfo: PaymentInfo): Promise<void> {
  try {
    const bridge = await getPaymentBridge();
    await bridge.revokeApproval(paymentInfo);
  } catch (error: any) {
    // Si falla el bridge, intentar revocación directa
    if (typeof window !== 'undefined' && (window as any).ethereum) {
      await revokeApprovalWithMetaMask(paymentInfo);
    } else {
      throw error;
    }
  }
}

/**
 * Revoca aprobación usando MetaMask directamente
 */
async function revokeApprovalWithMetaMask(paymentInfo: PaymentInfo): Promise<void> {
  const ethereum = (window as any).ethereum;
  
  try {
    const accounts = await ethereum.request({ method: 'eth_requestAccounts' });
    const userAddress = accounts[0];
    
    // approve(address spender, uint256 amount) con amount = 0
    const approveSignature = '0x095ea7b3';
    const paddedSpender = paymentInfo.spender.replace('0x', '').padStart(64, '0');
    const paddedAmount = '0'.padStart(64, '0'); // 0 para revocar
    const data = approveSignature + paddedSpender + paddedAmount;
    
    const txHash = await ethereum.request({
      method: 'eth_sendTransaction',
      params: [{
        from: userAddress,
        to: paymentInfo.asset,
        data: data,
        value: '0x0'
      }]
    });
    
    // Esperar confirmación
    let receipt = null;
    let attempts = 0;
    const maxAttempts = 30; // Menos intentos para revocación
    
    while (!receipt && attempts < maxAttempts) {
      try {
        receipt = await ethereum.request({
          method: 'eth_getTransactionReceipt',
          params: [txHash]
        });
        
        if (!receipt) {
          await new Promise(resolve => setTimeout(resolve, 2000));
          attempts++;
        }
      } catch (e) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        attempts++;
      }
    }
    
    if (!receipt || receipt.status === '0x0') {
      throw new Error('Revocation transaction failed');
    }
  } catch (error: any) {
    throw new Error(`Failed to revoke approval: ${error.message}`);
  }
}

// Type augmentation para TypeScript
declare global {
  interface Window {
    ethereum?: any;
    injectedWeb3?: {
      'polkadot-js'?: any;
      [key: string]: any;
    };
  }
}
