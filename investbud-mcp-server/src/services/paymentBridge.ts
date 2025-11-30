// src/services/paymentBridge.ts
// Bridge para manejar pagos x402 abriendo una ventana de navegador

import express from 'express';
import { Server } from 'http';
import { X402PaymentRequirements } from './x402Client.js';


interface PaymentSession {
  id: string;
  requirements: X402PaymentRequirements;
  paymentOption: any;
  resolve: (value: string) => void;
  reject: (error: Error) => void;
  expiresAt: number;
}

// Helper para obtener chainId desde el nombre de la red
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
  return chainIds[network.toLowerCase()] || 8453; // Default Base Mainnet
}

// Helper para obtener metadata del token EIP-712
function getTokenMetadata(asset: string, network: string): { name: string; version: string } {
  const assetLower = asset.toLowerCase();
  
  // USDC addresses
  const usdcAddresses = [
    '0x036cbd53842c5426634e7929541ec2318f3dcf7e', // Base Sepolia
    '0x833589fcd6edb6e08f4c7c32d4f71b54bda02913', // Base Mainnet
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48', // Ethereum
    '0x2791bca1f2de4661ed88a30c99a7a9449aa84174', // Polygon
    '0xff970a61a04b1ca14834a43f5de4533ebddb5cc8', // Arbitrum
  ];
  
  if (usdcAddresses.includes(assetLower)) {
    return { name: 'USD Coin', version: '2' };
  }
  
  // USDT
  if (assetLower === '0xdac17f958d2ee523a2206206994597c13d831ec7') {
    return { name: 'Tether USD', version: '1' };
  }
  
  // DAI
  if (assetLower === '0x6b175474e89094c44da98b954eedeac495271d0f') {
    return { name: 'Dai Stablecoin', version: '1' };
  }
  
  // Default
  return { name: 'ERC20 Token', version: '1' };
}

class PaymentBridgeServer {
  private app: express.Application;
  private server: Server | null = null;
  private port: number = 0;
  private sessions: Map<string, PaymentSession> = new Map();

  constructor() {
    this.app = express();
    this.app.use(express.json());
    this.setupRoutes();
  }

  private setupRoutes() {
    // Página de pago HTML
    this.app.get('/pay/:sessionId', (req, res) => {
      const { sessionId } = req.params;
      const session = this.sessions.get(sessionId);

      if (!session) {
        res.status(404).send('Payment session not found or expired');
        return;
      }

      res.send(this.getPaymentHTML(sessionId, session));
    });

    // API para obtener detalles de la sesión
    this.app.get('/api/session/:sessionId', (req, res) => {
      const { sessionId } = req.params;
      const session = this.sessions.get(sessionId);

      if (!session) {
        res.status(404).json({ error: 'Session not found' });
        return;
      }

      res.json({
        paymentOption: session.paymentOption,
        expiresAt: session.expiresAt
      });
    });

    // API para enviar el pago firmado
    this.app.post('/api/submit/:sessionId', (req, res) => {
      const { sessionId } = req.params;
      const { paymentHeader } = req.body;

      const session = this.sessions.get(sessionId);

      if (!session) {
        res.status(404).json({ error: 'Session not found' });
        return;
      }

      if (!paymentHeader) {
        res.status(400).json({ error: 'Missing paymentHeader' });
        return;
      }

      // Resolver la promesa con el payment header
      session.resolve(paymentHeader);
      this.sessions.delete(sessionId);

      res.json({ success: true });
    });

    // API para cancelar el pago
    this.app.post('/api/cancel/:sessionId', (req, res) => {
      const { sessionId } = req.params;
      const session = this.sessions.get(sessionId);

      if (!session) {
        res.status(404).json({ error: 'Session not found' });
        return;
      }

      session.reject(new Error('Payment cancelled by user'));
      this.sessions.delete(sessionId);

      res.json({ success: true });
    });

    // Página para revocar aprobación
    this.app.get('/revoke', (req, res) => {
      const { network, spender, asset, from } = req.query;
      
      if (!network || !spender || !asset || !from) {
        res.status(400).send('Missing required parameters');
        return;
      }

      res.send(this.getRevokeHTML(
        network as string,
        spender as string,
        asset as string,
        from as string
      ));
    });
  }

  private formatTokenAmount(asset: string, amount: string, network: string): string {
    // Mapa de tokens conocidos con sus decimales
    const tokenDecimals: { [key: string]: { decimals: number; symbol: string } } = {
      // USDC en diferentes redes
      '0x036CbD53842c5426634e7929541eC2318f3dCF7e': { decimals: 6, symbol: 'USDC' }, // Base Sepolia
      '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913': { decimals: 6, symbol: 'USDC' }, // Base Mainnet
      '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48': { decimals: 6, symbol: 'USDC' }, // Ethereum
      '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174': { decimals: 6, symbol: 'USDC' }, // Polygon
      '0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8': { decimals: 6, symbol: 'USDC' }, // Arbitrum
      
      // USDT
      '0xdAC17F958D2ee523a2206206994597C13D831ec7': { decimals: 6, symbol: 'USDT' }, // Ethereum
      '0xc2132D05D31c914a87C6611C10748AEb04B58e8F': { decimals: 6, symbol: 'USDT' }, // Polygon
      
      // DAI
      '0x6B175474E89094C44Da98b954EedeAC495271d0F': { decimals: 18, symbol: 'DAI' }, // Ethereum
      '0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063': { decimals: 18, symbol: 'DAI' }, // Polygon
      
      // Native tokens (address 0x0 o similar)
      '0x0000000000000000000000000000000000000000': { decimals: 18, symbol: 'ETH' },
      '0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE': { decimals: 18, symbol: 'ETH' },
    };

    // Normalizar dirección
    const normalizedAsset = asset.toLowerCase();
    
    // Buscar info del token
    let decimals = 18; // Default para tokens ERC20
    let symbol = 'tokens';
    
    for (const [addr, info] of Object.entries(tokenDecimals)) {
      if (addr.toLowerCase() === normalizedAsset) {
        decimals = info.decimals;
        symbol = info.symbol;
        break;
      }
    }
    
    // Si es base o ethereum y no se reconoce, asumir ETH
    if (symbol === 'tokens' && (network.includes('base') || network.includes('eth'))) {
      symbol = 'ETH';
      decimals = 18;
    }
    
    // Convertir a número con decimales
    const numAmount = parseFloat(amount);
    const divisor = Math.pow(10, decimals);
    const readableAmount = numAmount / divisor;
    
    // Formatear con precisión apropiada
    let formattedValue: string;
    if (readableAmount < 0.01) {
      formattedValue = readableAmount.toFixed(6);
    } else if (readableAmount < 1) {
      formattedValue = readableAmount.toFixed(4);
    } else {
      formattedValue = readableAmount.toFixed(2);
    }
    
    // Eliminar ceros trailing
    formattedValue = formattedValue.replace(/\.?0+$/, '');
    
    return `${formattedValue} ${symbol}`;
  }

  private getPaymentHTML(sessionId: string, session: PaymentSession): string {
    const { paymentOption } = session;
    
    // Formatear el monto con decimales correctos
    const formattedAmount = this.formatTokenAmount(
      paymentOption.asset,
      paymentOption.maxAmountRequired,
      paymentOption.network
    );
    
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>InvestBud Payment - x402</title>
  <script type="module">
    // Import viem from CDN
    import { createWalletClient, custom, parseUnits, getAddress } from 'https://esm.sh/viem@2.21.54';
    import { baseSepolia, base, mainnet, polygon, arbitrum, optimism } from 'https://esm.sh/viem@2.21.54/chains';
    
    // Make viem available globally
    window.viem = { createWalletClient, custom, parseUnits, getAddress, chains: { baseSepolia, base, mainnet, polygon, arbitrum, optimism } };
  </script>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
      background: linear-gradient(135deg, #080808ff 0%, #3e3e3eff 100%);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }
    .container {
      background: white;
      border-radius: 16px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      max-width: 500px;
      width: 100%;
      padding: 40px;
      animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn {
      from {
        opacity: 0;
        transform: translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    h1 {
      color: #333;
      font-size: 24px;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .subtitle {
      color: #666;
      font-size: 14px;
      margin-bottom: 30px;
    }
    .payment-details {
      background: #f7f9fc;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
    }
    .detail-row {
      display: flex;
      justify-content: space-between;
      margin-bottom: 12px;
      font-size: 14px;
    }
    .detail-row:last-child {
      margin-bottom: 0;
    }
    .detail-label {
      color: #666;
      font-weight: 500;
    }
    .detail-value {
      color: #333;
      font-weight: 600;
      word-break: break-all;
      text-align: right;
      max-width: 60%;
    }
    .amount {
      font-size: 18px;
      color: #667eea;
    }
    .status {
      padding: 12px;
      border-radius: 8px;
      margin-bottom: 20px;
      font-size: 14px;
      display: none;
    }
    .status.info {
      background: #e3f2fd;
      color: #1976d2;
      display: block;
    }
    .status.success {
      background: #e8f5e9;
      color: #388e3c;
    }
    .status.error {
      background: #ffebee;
      color: #d32f2f;
    }
    .buttons {
      display: flex;
      gap: 12px;
    }
    button {
      flex: 1;
      padding: 14px 24px;
      border-radius: 8px;
      border: none;
      font-size: 16px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }
    .btn-primary {
      background: linear-gradient(135deg, #000000ff 0%, #303030ff 100%);
      color: white;
    }
    .btn-primary:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .btn-secondary {
      background: #f5f5f5;
      color: #666;
    }
    .btn-secondary:hover:not(:disabled) {
      background: #e0e0e0;
    }
    .spinner {
      display: inline-block;
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin-right: 8px;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>💳 Payment Required</h1>
    <p class="subtitle">x402 Payment Protocol - InvestBud</p>

    <div class="payment-details">
      <div class="detail-row">
        <span class="detail-label">Network:</span>
        <span class="detail-value">${paymentOption.network}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Pay To:</span>
        <span class="detail-value">${paymentOption.payTo}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Asset:</span>
        <span class="detail-value">${paymentOption.asset}</span>
      </div>
      <div class="detail-row">
        <span class="detail-label">Amount:</span>
        <span class="detail-value amount">${formattedAmount}</span>
      </div>
      ${paymentOption.description ? `
      <div class="detail-row">
        <span class="detail-label">Description:</span>
        <span class="detail-value">${paymentOption.description}</span>
      </div>
      ` : ''}
    </div>

    <div class="status info" id="status">
      Please connect your wallet and approve the payment
    </div>

    <div class="buttons">
      <button class="btn-secondary" id="cancelBtn" onclick="cancelPayment()">
        Cancel
      </button>
      <button class="btn-primary" id="payBtn" onclick="initiatePayment()">
        Connect Wallet & Pay
      </button>
    </div>
  </div>

  <script>
    const sessionId = '${sessionId}';
    const isEVM = ['ethereum', 'base', 'polygon', 'arbitrum', 'optimism', 'eth-mainnet', 'base-mainnet', 'base-sepolia'].includes('${paymentOption.network}'.toLowerCase());
    
    function setStatus(message, type = 'info') {
      const statusEl = document.getElementById('status');
      statusEl.textContent = message;
      statusEl.className = 'status ' + type;
      statusEl.style.display = 'block';
    }

    function disableButtons() {
      document.getElementById('payBtn').disabled = true;
      document.getElementById('cancelBtn').disabled = true;
    }

    function enableButtons() {
      document.getElementById('payBtn').disabled = false;
      document.getElementById('cancelBtn').disabled = false;
    }

    async function cancelPayment() {
      try {
        await fetch('/api/cancel/' + sessionId, { method: 'POST' });
        setStatus('Payment cancelled', 'info');
        setTimeout(() => window.close(), 1500);
      } catch (error) {
        console.error('Cancel error:', error);
        window.close();
      }
    }

    async function initiatePayment() {
      const payBtn = document.getElementById('payBtn');
      const originalText = payBtn.innerHTML;
      payBtn.innerHTML = '<span class="spinner"></span>Processing...';
      disableButtons();

      try {
        if (isEVM) {
          await payWithMetaMask();
        } else {
          await payWithPolkadot();
        }
      } catch (error) {
        console.error('Payment error:', error);
        setStatus(error.message, 'error');
        payBtn.innerHTML = originalText;
        enableButtons();
      }
    }

    async function payWithMetaMask() {
      // Check if MetaMask is installed
      if (typeof window.ethereum === 'undefined') {
        throw new Error('MetaMask not detected. Please install MetaMask extension.');
      }

      // Wait for viem to load
      while (!window.viem) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      const { createWalletClient, custom, chains, getAddress } = window.viem;

      // Helper function to get numeric chain ID
      function getChainIdFromNetwork(network) {
        const chainIds = {
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

      // Helper to get viem chain object
      function getViemChain(network) {
        const networkLower = network.toLowerCase();
        if (networkLower.includes('base-sepolia')) return chains.baseSepolia;
        if (networkLower.includes('base')) return chains.base;
        if (networkLower.includes('polygon')) return chains.polygon;
        if (networkLower.includes('arbitrum')) return chains.arbitrum;
        if (networkLower.includes('optimism')) return chains.optimism;
        return chains.mainnet;
      }

      setStatus('Connecting to MetaMask...', 'info');

      // Fetch payment details
      const response = await fetch('/api/session/' + sessionId);
      const { paymentOption } = await response.json();

      // Get the appropriate chain
      const chain = getViemChain(paymentOption.network);

      // Create wallet client with viem
      const walletClient = createWalletClient({
        chain,
        transport: custom(window.ethereum)
      });

      // Request account access
      const [userAddress] = await walletClient.requestAddresses();

      // Recreate wallet client with account for signing
      const walletClientWithAccount = createWalletClient({
        account: userAddress,
        chain,
        transport: custom(window.ethereum)
      });

      // Map network names to chain IDs
      const networkToChainId = {
        'base': '0x2105',           // Base Mainnet (8453)
        'base-mainnet': '0x2105',
        'base-sepolia': '0x14a34',  // Base Sepolia (84532)
        'ethereum': '0x1',           // Ethereum Mainnet (1)
        'eth-mainnet': '0x1',
        'polygon': '0x89',           // Polygon Mainnet (137)
        'polygon-mainnet': '0x89',
        'arbitrum': '0xa4b1',        // Arbitrum One (42161)
        'arb-mainnet': '0xa4b1',
        'optimism': '0xa',           // Optimism (10)
        'opt-mainnet': '0xa',
      };

      const requiredChainId = networkToChainId[paymentOption.network.toLowerCase()] || '0x14a34';

      // Check current network
      const currentChainId = await window.ethereum.request({ method: 'eth_chainId' });

      // Switch network if needed
      if (currentChainId !== requiredChainId) {
        setStatus(\`Switching to \${paymentOption.network}...\`, 'info');
        
        try {
          await window.ethereum.request({
            method: 'wallet_switchEthereumChain',
            params: [{ chainId: requiredChainId }],
          });
        } catch (switchError) {
          // This error code indicates that the chain has not been added to MetaMask
          if (switchError.code === 4902) {
            // Add Base Sepolia network
            if (requiredChainId === '0x14a34') {
              await window.ethereum.request({
                method: 'wallet_addEthereumChain',
                params: [{
                  chainId: '0x14a34',
                  chainName: 'Base Sepolia',
                  nativeCurrency: {
                    name: 'Ethereum',
                    symbol: 'ETH',
                    decimals: 18
                  },
                  rpcUrls: ['https://sepolia.base.org'],
                  blockExplorerUrls: ['https://sepolia.basescan.org']
                }],
              });
            } else if (requiredChainId === '0x2105') {
              await window.ethereum.request({
                method: 'wallet_addEthereumChain',
                params: [{
                  chainId: '0x2105',
                  chainName: 'Base',
                  nativeCurrency: {
                    name: 'Ethereum',
                    symbol: 'ETH',
                    decimals: 18
                  },
                  rpcUrls: ['https://mainnet.base.org'],
                  blockExplorerUrls: ['https://basescan.org']
                }],
              });
            } else {
              throw new Error(\`Please add \${paymentOption.network} network to MetaMask\`);
            }
          } else {
            throw switchError;
          }
        }
      }

      setStatus('Approving token spending...', 'info');

      // Generar nonce aleatorio (32 bytes)
      const nonceBytes = new Uint8Array(32);
      crypto.getRandomValues(nonceBytes);
      const nonce = '0x' + Array.from(nonceBytes).map(b => b.toString(16).padStart(2, '0')).join('');

      // Timestamps (convert to BigInt then string to exactly match x402 behavior)
      const validAfter = BigInt(Math.floor(Date.now() / 1000) - 600).toString();
      const validBefore = BigInt(Math.floor(Date.now() / 1000) + 900).toString();

      // Construir el mensaje EIP-712 para TransferWithAuthorization
      // Usar name y version del 402 response (campo extra) si están disponibles
      const tokenName = paymentOption.extra?.name || 'USD Coin';
      const tokenVersion = paymentOption.extra?.version || '2';
      
      const chainId = getChainIdFromNetwork(paymentOption.network);
      
      const domain = {
        name: tokenName,
        version: tokenVersion,
        chainId: chainId,
        verifyingContract: getAddress(paymentOption.asset)  // Normalize to checksum format
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

      // CRITICAL: All values MUST be strings for EIP-712 uint256 types to match x402 behavior
      // CRITICAL: Addresses MUST be in checksum format using getAddress() like x402 does
      const message = {
        from: getAddress(userAddress),  // Checksum format
        to: getAddress(paymentOption.payTo),  // Checksum format
        value: paymentOption.maxAmountRequired,  // Already a string from 402 response
        validAfter: validAfter,  // String from BigInt
        validBefore: validBefore,  // String from BigInt
        nonce: nonce
      };

      // DEBUG: Log all signing parameters
      console.log('🔍 DEBUG - EIP-712 Signing Parameters:');
      console.log('Domain:', JSON.stringify(domain, null, 2));
      console.log('Message:', JSON.stringify(message, null, 2));
      console.log('Types:', JSON.stringify(types, null, 2));
      console.log('User Address:', userAddress);
      console.log('PaymentOption:', JSON.stringify(paymentOption, null, 2));
      
      // Log value types
      console.log('\\n🔍 DEBUG - Value Types:');
      console.log('typeof value:', typeof message.value, '| value:', message.value);
      console.log('typeof validAfter:', typeof message.validAfter, '| validAfter:', message.validAfter);
      console.log('typeof validBefore:', typeof message.validBefore, '| validBefore:', message.validBefore);
      console.log('typeof nonce:', typeof message.nonce, '| nonce:', message.nonce);
      console.log('typeof chainId:', typeof chainId, '| chainId:', chainId);

      setStatus('📝 Please sign the payment authorization...', 'info');

      // Firmar usando viem walletClient (mismo método que x402)
      // Esto asegura que la firma se genere exactamente como x402 lo hace
      const signature = await walletClientWithAccount.signTypedData({
        domain,
        types,
        primaryType: 'TransferWithAuthorization',
        message
      });

      console.log('\\n✅ Signature received:', signature);

      // Construir el X-PAYMENT header según el formato x402 completo
      // CRITICAL: Todos los valores numéricos DEBEN ser strings en el payload final
      // CRITICAL: Addresses MUST be in checksum format
      const paymentData = {
        x402Version: 1,
        scheme: 'exact',
        network: paymentOption.network,
        payload: {
          signature: signature,
          authorization: {
            from: getAddress(userAddress),  // Checksum format
            to: getAddress(paymentOption.payTo),  // Checksum format
            value: paymentOption.maxAmountRequired,  // Already a string
            validAfter: validAfter,  // Already a string
            validBefore: validBefore,  // Already a string
            nonce: nonce
          }
        }
      };
      
      console.log('\\n🔍 DEBUG - Final Payment Data:');
      console.log(JSON.stringify(paymentData, null, 2));
      
      const paymentHeader = btoa(JSON.stringify(paymentData));
      console.log('\\n🔍 DEBUG - X-PAYMENT Header (base64):', paymentHeader.substring(0, 100) + '...');

      // Submit payment proof
      setStatus('💸 Submitting payment proof...', 'info');
      const submitResponse = await fetch('/api/submit/' + sessionId, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paymentHeader })
      });

      if (!submitResponse.ok) {
        throw new Error('Failed to submit payment proof');
      }

      setStatus('✅ Payment successful! Returning to Claude...', 'success');
      setTimeout(() => window.close(), 3000);
    }

    async function payWithPolkadot() {
      // Check if Polkadot.js is installed
      if (typeof window.injectedWeb3 === 'undefined' || !window.injectedWeb3['polkadot-js']) {
        throw new Error('Polkadot.js extension not detected. Please install it.');
      }

      setStatus('Connecting to Polkadot.js...', 'info');

      const extension = window.injectedWeb3['polkadot-js'];
      await extension.enable('InvestBud MCP Payment');

      const allAccounts = await extension.accounts.get();
      if (allAccounts.length === 0) {
        throw new Error('No Polkadot accounts found');
      }
      const userAccount = allAccounts[0];

      setStatus('Please sign the payment in Polkadot.js...', 'info');

      // Fetch payment details
      const response = await fetch('/api/session/' + sessionId);
      const { paymentOption } = await response.json();

      // Create payment message
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

      const message = JSON.stringify(paymentMessage);
      const signRaw = extension.signer?.signRaw;
      
      if (!signRaw) {
        throw new Error('Polkadot.js signing not available');
      }

      const { signature } = await signRaw({
        address: userAccount.address,
        data: Array.from(new TextEncoder().encode(message)).map(b => b.toString(16).padStart(2, '0')).join(''),
        type: 'bytes'
      });

      // Create payment header
      const paymentHeader = btoa(signature + ':' + message);

      // Submit payment
      setStatus('Submitting payment...', 'info');
      const submitResponse = await fetch('/api/submit/' + sessionId, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ paymentHeader })
      });

      if (!submitResponse.ok) {
        throw new Error('Failed to submit payment');
      }

      setStatus('✅ Payment successful! Returning to Claude...', 'success');
      setTimeout(() => window.close(), 3000);
    }
  </script>
</body>
</html>`;
  }

  private getRevokeHTML(network: string, spender: string, asset: string, from: string): string {
    return `<!DOCTYPE html>
<html>
<head>
  <title>Revoke Token Approval</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      max-width: 600px;
      margin: 50px auto;
      padding: 20px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
    }
    .card {
      background: white;
      border-radius: 12px;
      padding: 30px;
      box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    h1 {
      color: #333;
      margin-top: 0;
      font-size: 24px;
    }
    .info {
      background: #f5f5f5;
      padding: 15px;
      border-radius: 8px;
      margin: 20px 0;
      font-size: 14px;
    }
    .info-label {
      font-weight: bold;
      color: #666;
      margin-bottom: 5px;
    }
    .info-value {
      font-family: monospace;
      word-break: break-all;
      color: #333;
    }
    .btn {
      width: 100%;
      padding: 15px;
      font-size: 16px;
      font-weight: bold;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      margin-top: 10px;
      transition: all 0.3s;
    }
    .btn-primary {
      background: #667eea;
      color: white;
    }
    .btn-primary:hover {
      background: #5568d3;
    }
    .btn-secondary {
      background: #e0e0e0;
      color: #333;
    }
    .btn-secondary:hover {
      background: #d0d0d0;
    }
    .status {
      padding: 12px;
      border-radius: 8px;
      margin: 15px 0;
      display: none;
      font-size: 14px;
    }
    .status.info { background: #e3f2fd; color: #1976d2; display: block; }
    .status.error { background: #ffebee; color: #c62828; display: block; }
    .status.success { background: #e8f5e9; color: #2e7d32; display: block; }
    .warning {
      background: #fff3cd;
      border: 1px solid #ffc107;
      padding: 15px;
      border-radius: 8px;
      margin: 20px 0;
      font-size: 14px;
      color: #856404;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>🔒 Revoke Token Approval</h1>
    <div class="warning">
      ⚠️ <strong>Security Action</strong><br>
      You are about to revoke the approval given to the payment relay. This prevents it from accessing your tokens in the future.
    </div>
    
    <div class="info">
      <div class="info-label">Network:</div>
      <div class="info-value">${network}</div>
    </div>
    
    <div class="info">
      <div class="info-label">Token Contract:</div>
      <div class="info-value">${asset}</div>
    </div>
    
    <div class="info">
      <div class="info-label">Spender (Relay):</div>
      <div class="info-value">${spender}</div>
    </div>
    
    <div id="status" class="status"></div>
    
    <button class="btn btn-primary" onclick="revokeApproval()">
      Revoke Approval
    </button>
    <button class="btn btn-secondary" onclick="window.close()">
      Skip & Close
    </button>
  </div>

  <script>
    const network = '${network}';
    const spender = '${spender}';
    const asset = '${asset}';
    const from = '${from}';

    function setStatus(message, type) {
      const statusEl = document.getElementById('status');
      statusEl.textContent = message;
      statusEl.className = 'status ' + type;
    }

    async function revokeApproval() {
      try {
        if (typeof window.ethereum === 'undefined') {
          throw new Error('MetaMask not detected. Please install MetaMask.');
        }

        setStatus('Connecting to MetaMask...', 'info');

        // Request account access
        const accounts = await window.ethereum.request({ 
          method: 'eth_requestAccounts' 
        });
        const userAddress = accounts[0];

        setStatus('Please confirm the revocation in MetaMask...', 'info');

        // approve(address spender, uint256 amount) with amount = 0
        const approveSignature = '0x095ea7b3';
        const paddedSpender = spender.replace('0x', '').padStart(64, '0');
        const paddedAmount = '0'.padStart(64, '0'); // 0 to revoke
        const data = approveSignature + paddedSpender + paddedAmount;

        const txHash = await window.ethereum.request({
          method: 'eth_sendTransaction',
          params: [{
            from: userAddress,
            to: asset,
            data: data,
            value: '0x0'
          }]
        });

        setStatus('⏳ Waiting for transaction confirmation...', 'info');

        // Wait for transaction receipt
        let receipt = null;
        let attempts = 0;
        const maxAttempts = 30;

        while (!receipt && attempts < maxAttempts) {
          try {
            receipt = await window.ethereum.request({
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

        setStatus('✅ Approval successfully revoked! Closing...', 'success');
        setTimeout(() => window.close(), 3000);

      } catch (error) {
        setStatus('❌ Error: ' + error.message, 'error');
      }
    }
  </script>
</body>
</html>`;
  }

  async start(): Promise<number> {
    return new Promise((resolve, reject) => {
      // Intentar puertos en el rango 3402-3502
      const tryPort = (port: number) => {
        this.server = this.app.listen(port)
          .on('listening', () => {
            this.port = port;
            resolve(port);
          })
          .on('error', (err: any) => {
            if (err.code === 'EADDRINUSE' && port < 3502) {
              tryPort(port + 1);
            } else {
              reject(err);
            }
          });
      };
      tryPort(3402);
    });
  }

  stop() {
    if (this.server) {
      this.server.close();
      this.server = null;
    }
    this.sessions.clear();
  }

  async requestPayment(
    requirements: X402PaymentRequirements
  ): Promise<string> {
    // Validar requirements
    if (!requirements.accepts || requirements.accepts.length === 0) {
      throw new Error('Invalid payment requirements');
    }

    const paymentOption = requirements.accepts[0];
    const sessionId = this.generateSessionId();
    const expiresAt = Date.now() + 5 * 60 * 1000; // 5 minutos

    // Crear promesa que será resuelta cuando se complete el pago
    const paymentPromise = new Promise<string>((resolve, reject) => {
      this.sessions.set(sessionId, {
        id: sessionId,
        requirements,
        paymentOption,
        resolve,
        reject,
        expiresAt
      });

      // Timeout automático
      setTimeout(() => {
        if (this.sessions.has(sessionId)) {
          this.sessions.delete(sessionId);
          reject(new Error('Payment timeout - session expired'));
        }
      }, 5 * 60 * 1000);
    });

    // Abrir navegador con la página de pago
    const paymentUrl = `http://localhost:${this.port}/pay/${sessionId}`;
    await this.openBrowser(paymentUrl);

    // Esperar a que se complete el pago
    return paymentPromise;
  }

  async revokeApproval(paymentInfo: any): Promise<void> {
    // Abrir navegador para revocar la aprobación
    const revokeUrl = `http://localhost:${this.port}/revoke?network=${paymentInfo.network}&spender=${paymentInfo.spender}&asset=${paymentInfo.asset}&from=${paymentInfo.userAddress}`;
    await this.openBrowser(revokeUrl);
    
    // Dar tiempo para que el usuario confirme la transacción
    // En una implementación más robusta, esto también usaría una sesión con promesa
    await new Promise(resolve => setTimeout(resolve, 15000));
  }

  private generateSessionId(): string {
    return `pay_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }

  private async openBrowser(url: string): Promise<void> {
    const { exec } = await import('child_process');
    const { promisify } = await import('util');
    const execAsync = promisify(exec);

    // Detectar sistema operativo y abrir navegador
    const platform = process.platform;
    
    try {
      if (platform === 'darwin') {
        await execAsync(`open "${url}"`);
      } else if (platform === 'win32') {
        await execAsync(`start "${url}"`);
      } else {
        await execAsync(`xdg-open "${url}"`);
      }
    } catch (error) {
      throw new Error(
        `Failed to open browser. Please manually open: ${url}`
      );
    }
  }
}

// Singleton instance
let bridgeInstance: PaymentBridgeServer | null = null;

export async function getPaymentBridge(): Promise<PaymentBridgeServer> {
  if (!bridgeInstance) {
    bridgeInstance = new PaymentBridgeServer();
    await bridgeInstance.start();
  }
  return bridgeInstance;
}

export function stopPaymentBridge() {
  if (bridgeInstance) {
    bridgeInstance.stop();
    bridgeInstance = null;
  }
}
