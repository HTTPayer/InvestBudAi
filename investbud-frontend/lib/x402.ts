import { ethers } from 'ethers';
import { wrapFetchWithPayment } from 'x402-fetch';
import { createWalletClient, custom, publicActions, http } from 'viem';
import { baseSepolia } from 'viem/chains';
import { USDC_ADDRESS, BACKEND_URL } from './constants';

/**
 * Send message with automatic x402 payment handling
 */
export async function sendMessageWithX402(
  provider: ethers.BrowserProvider,
  sessionId: string,
  message: string,
  metadata?: Record<string, unknown>
): Promise<{
  response?: string;
  reply?: string;
  regime_signal?: string;
  portfolio_summary?: unknown;
  error?: string;
  [key: string]: unknown;
}> {
  try {
    // Get the Ethereum provider from window.ethereum
    if (!window.ethereum) {
      throw new Error('No Ethereum provider found');
    }

    // Get the connected account
    const signer = await provider.getSigner();
    const account = await signer.getAddress();

    // Create a viem wallet client from the browser provider
    // Extend with publicActions to create a complete SignerWallet type
    const walletClient = createWalletClient({
      account: account as `0x${string}`,
      chain: baseSepolia,
      transport: custom(window.ethereum),
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    }).extend(publicActions) as any;

    // Wrap fetch with payment handling
    // maxValue: 0.1 USDC = 100000 (6 decimals)
    const fetchWithPay = wrapFetchWithPayment(fetch, walletClient, BigInt(100000));

    // Make the request - x402-fetch will handle the payment flow automatically
    const response = await fetchWithPay(BACKEND_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: sessionId,
        message,
        metadata,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    
    return data;
  } catch (error) {
    throw error;
  }
}

export interface PaymentInstructions {
  recipient?: string;
  amount?: string;
  token?: string;
  chain_id?: number;
  payment_id?: string;
  // x402 format
  payTo?: string;
  asset?: string;
  maxAmountRequired?: string;
  network?: string;
  description?: string;
}

export interface X402Response {
  status?: 402;
  x402Version: number;
  accepts: PaymentInstructions[];
  error?: string;
  payment_instructions?: PaymentInstructions;
  message?: string;
}

/**
 * Get chain ID from network name
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
  return chainIds[network.toLowerCase()] || 84532;
}

/**
 * Get token metadata for EIP-712 signing
 */
function getTokenMetadata(asset: string): { name: string; version: string } {
  const assetLower = asset.toLowerCase();
  
  // USDC addresses
  const usdcAddresses = [
    '0x036cbd53842c5426634e7929541ec2318f3dcf7e', // Base Sepolia
    '0x833589fcd6edb6e08f4c7c32d4f71b54bda02913', // Base Mainnet
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48', // Ethereum Mainnet
  ];
  
  if (usdcAddresses.includes(assetLower)) {
    return { name: 'USD Coin', version: '2' };
  }
  
  // Default for unknown tokens
  return { name: 'Token', version: '1' };
}

/**
 * Call the endpoint to get x402 payment instructions
 */
export async function getPaymentInstructions(
  sessionId: string,
  message: string,
  metadata?: Record<string, unknown>
): Promise<PaymentInstructions> {
  const response = await fetch(BACKEND_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      metadata,
    }),
  });

  if (response.status === 402) {
    const data: X402Response = await response.json();
    
    // Try x402 standard format (accepts array)
    if (data.accepts && data.accepts.length > 0) {
      const paymentOption = data.accepts[0];
      
      // Get payment_id from response if available
      const paymentId = data.payment_instructions?.payment_id || crypto.randomUUID();
      
      // Transform to our format
      return {
        recipient: paymentOption.payTo || paymentOption.recipient || '',
        amount: paymentOption.maxAmountRequired || paymentOption.amount || '',
        token: paymentOption.asset || paymentOption.token || '',
        chain_id: getChainIdFromNetwork(paymentOption.network || 'base-sepolia'),
        payment_id: paymentId,
        ...paymentOption,
      };
    }
    
    // Fallback to our custom payment_instructions format
    if (data.payment_instructions) {
      return {
        ...data.payment_instructions,
        // Ensure payment_id exists
        payment_id: data.payment_instructions.payment_id || crypto.randomUUID(),
      };
    }
    
    throw new Error('Invalid x402 response: missing accepts or payment_instructions');
  }

  throw new Error(`Expected 402 status, got ${response.status}`);
}

/**
 * Execute USDC payment using TransferWithAuthorization (EIP-3009)
 */
export async function executePayment(
  provider: ethers.BrowserProvider,
  instructions: PaymentInstructions
): Promise<string> {
  try {
    const signer = await provider.getSigner();
    const userAddress = await signer.getAddress();
    
    // Get values with fallbacks for x402 format
    const recipient = instructions.payTo || instructions.recipient || '';
    const amount = instructions.maxAmountRequired || instructions.amount || '';
    const tokenAddress = instructions.asset || instructions.token || USDC_ADDRESS;
    const network = instructions.network || 'base-sepolia';
    const chainId = instructions.chain_id || getChainIdFromNetwork(network);
    
    if (!recipient || !amount) {
      throw new Error('Missing recipient or amount in payment instructions');
    }
    
    // Generate random nonce (32 bytes)
    const nonceBytes = ethers.randomBytes(32);
    const nonce = ethers.hexlify(nonceBytes);
    
    // Timestamps
    const validAfter = Math.floor(Date.now() / 1000) - 600;
    const validBefore = Math.floor(Date.now() / 1000) + 900;
    
    // Get token metadata
    const tokenMetadata = getTokenMetadata(tokenAddress);
    
    // EIP-712 Domain
    const domain = {
      name: tokenMetadata.name,
      version: tokenMetadata.version,
      chainId: chainId,
      verifyingContract: tokenAddress,
    };
    
    // EIP-712 Types
    const types = {
      TransferWithAuthorization: [
        { name: 'from', type: 'address' },
        { name: 'to', type: 'address' },
        { name: 'value', type: 'uint256' },
        { name: 'validAfter', type: 'uint256' },
        { name: 'validBefore', type: 'uint256' },
        { name: 'nonce', type: 'bytes32' },
      ],
    };
    
    // Message to sign
    const message = {
      from: userAddress,
      to: recipient,
      value: amount,
      validAfter: validAfter.toString(),
      validBefore: validBefore.toString(),
      nonce: nonce,
    };
    
    // Sign the message - MetaMask popup should appear here
    const signature = await signer.signTypedData(domain, types, message);
    
    // Build X-PAYMENT header with x402 exact EVM format
    // CRITICAL: All numeric values MUST be strings in the final payload
    // CRITICAL: Addresses MUST be in checksum format
    const paymentData = {
      x402Version: 1,
      scheme: 'exact',
      network: network,  // Use the network from instructions
      payload: {
        signature: signature,
        authorization: {
          from: ethers.getAddress(userAddress),  // Checksum format
          to: ethers.getAddress(recipient),  // Checksum format
          value: amount,  // Already a string
          validAfter: validAfter.toString(),  // String
          validBefore: validBefore.toString(),  // String
          nonce: nonce
        }
      }
    };
    
    const paymentHeader = btoa(JSON.stringify(paymentData));
    
    return paymentHeader;
  } catch (error) {
    throw new Error(`Failed to sign payment: ${error instanceof Error ? error.message : 'Unknown error'}`);
  }
}

/**
 * Send message with payment proof (X-PAYMENT header)
 */
export async function sendMessageWithPayment(
  sessionId: string,
  message: string,
  paymentId: string | undefined,
  paymentHeader: string,
  metadata?: Record<string, unknown>
): Promise<{
  response?: string;
  reply?: string;
  regime_signal?: string;
  portfolio_summary?: unknown;
  error?: string;
  [key: string]: unknown;
}> {
  const response = await fetch(BACKEND_URL, {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      'X-PAYMENT': paymentHeader,
    },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      payment_id: paymentId,
      metadata,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP ${response.status}: ${response.statusText} - ${errorText}`);
  }

  return response.json();
}
