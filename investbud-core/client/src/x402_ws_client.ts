/**
 * x402 WebSocket Client for MacroCrypto Live Portfolio Streaming
 *
 * Flow:
 * 1. GET /wallet/live to get payment requirements
 * 2. Pay via x402 and obtain receipt
 * 3. Connect to WebSocket with X-PAYMENT header (or ?receipt= query param)
 * 4. Receive portfolio updates until session expires (max 5 minutes)
 */

// Message types from server
export interface WSConnected {
  type: 'connected';
  session_id: string;
  wallet: string;
  network: string;
  update_interval: number;
  max_duration: number;
  expires_at: string;
}

export interface WSPortfolioUpdate {
  type: 'portfolio_update';
  timestamp: string;
  total_value_usd: number;
  change_24h: number | null;
  positions: Array<{
    symbol: string;
    name: string;
    balance: number;
    price_usd: number;
    value_usd: number;
    weight: number;
  }>;
  metrics: Record<string, any> | null;
}

export interface WSSessionEnding {
  type: 'session_ending';
  seconds_remaining: number;
}

export interface WSError {
  type: 'error';
  code: string;
  message: string;
}

export type WSServerMessage = WSConnected | WSPortfolioUpdate | WSSessionEnding | WSError;

// Payment requirements from GET /wallet/live
export interface PaymentRequirements {
  endpoint: string;
  protocol: string;
  description: string;
  cost_per_session: string;
  max_session_seconds: number;
  update_interval_seconds: number;
  payment_network: string;
  pay_to: string;
  connection_url: string;
}

// Client configuration
export interface X402WSClientConfig {
  baseUrl: string;
  walletAddress: string;
  network?: string;
  chainId?: number;
  onUpdate?: (update: WSPortfolioUpdate) => void;
  onError?: (error: WSError) => void;
  onConnected?: (info: WSConnected) => void;
  onSessionEnding?: (info: WSSessionEnding) => void;
  onDisconnect?: (reason: string) => void;
  // Payment function - implement with your x402 client
  paymentProvider?: (requirements: PaymentRequirements) => Promise<string>;
}

export class X402WebSocketClient {
  private ws: WebSocket | null = null;
  private config: X402WSClientConfig;
  private sessionInfo: WSConnected | null = null;

  constructor(config: X402WSClientConfig) {
    this.config = config;
  }

  /**
   * Get payment requirements from the API
   */
  async getPaymentRequirements(): Promise<PaymentRequirements> {
    const response = await fetch(`${this.config.baseUrl}/wallet/live`);
    if (!response.ok) {
      throw new Error(`Failed to get payment requirements: ${response.status}`);
    }
    return response.json() as Promise<PaymentRequirements>;
  }

  /**
   * Connect to the live portfolio WebSocket endpoint
   */
  async connect(): Promise<void> {
    // 1. Get payment requirements
    console.log('[X402WS] Fetching payment requirements...');
    const requirements = await this.getPaymentRequirements();
    console.log(`[X402WS] Cost: $${requirements.cost_per_session} for ${requirements.max_session_seconds}s session`);

    // 2. Pay via x402
    let receipt = '';
    if (this.config.paymentProvider) {
      console.log('[X402WS] Processing payment...');
      receipt = await this.config.paymentProvider(requirements);
      console.log('[X402WS] Payment complete');
    } else {
      console.warn('[X402WS] No payment provider configured - connection may fail');
    }

    // 3. Build WebSocket URL
    const params = new URLSearchParams({
      address: this.config.walletAddress,
    });
    if (this.config.network) {
      params.set('network', this.config.network);
    }
    if (this.config.chainId) {
      params.set('chain_id', this.config.chainId.toString());
    }
    if (receipt) {
      // Pass receipt as query param for browser compatibility
      params.set('receipt', receipt);
    }

    const wsUrl = `${this.config.baseUrl.replace('http', 'ws')}/wallet/live?${params}`;
    console.log(`[X402WS] Connecting to ${wsUrl.substring(0, 80)}...`);

    // 4. Connect
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        console.log('[X402WS] WebSocket connected, waiting for session...');
      };

      this.ws.onmessage = (event) => {
        try {
          const msg: WSServerMessage = JSON.parse(event.data);
          this.handleMessage(msg, resolve, reject);
        } catch (e) {
          console.error('[X402WS] Failed to parse message:', e);
        }
      };

      this.ws.onerror = (error) => {
        console.error('[X402WS] WebSocket error:', error);
        reject(new Error('WebSocket connection failed'));
      };

      this.ws.onclose = (event) => {
        console.log(`[X402WS] WebSocket closed: ${event.code} ${event.reason}`);
        this.sessionInfo = null;
        this.config.onDisconnect?.(event.reason || 'Connection closed');
      };
    });
  }

  /**
   * Handle incoming WebSocket messages
   */
  private handleMessage(
    msg: WSServerMessage,
    resolveConnect?: (value: void) => void,
    rejectConnect?: (reason: any) => void
  ): void {
    switch (msg.type) {
      case 'connected':
        console.log(`[X402WS] Session started: ${msg.session_id}`);
        console.log(`[X402WS] Expires at: ${msg.expires_at}`);
        this.sessionInfo = msg;
        this.config.onConnected?.(msg);
        resolveConnect?.();
        break;

      case 'portfolio_update':
        console.log(`[X402WS] Portfolio: $${msg.total_value_usd.toLocaleString()}`);
        this.config.onUpdate?.(msg);
        break;

      case 'session_ending':
        console.warn(`[X402WS] Session ending in ${msg.seconds_remaining}s`);
        this.config.onSessionEnding?.(msg);
        break;

      case 'error':
        console.error(`[X402WS] Error: ${msg.code} - ${msg.message}`);
        this.config.onError?.(msg);
        if (msg.code === 'session_expired') {
          this.disconnect();
        }
        // Reject connection if this is during setup
        if (rejectConnect && !this.sessionInfo) {
          rejectConnect(new Error(msg.message));
        }
        break;

      default:
        console.log('[X402WS] Unknown message:', msg);
    }
  }

  /**
   * Send a ping to keep connection alive
   */
  ping(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'ping' }));
    }
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    this.ws?.close(1000, 'Client disconnect');
    this.ws = null;
    this.sessionInfo = null;
  }

  /**
   * Check if connected
   */
  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN && this.sessionInfo !== null;
  }

  /**
   * Get current session info
   */
  get session(): WSConnected | null {
    return this.sessionInfo;
  }
}

// ============================================================================
// Example Usage
// ============================================================================

/*
import { X402WebSocketClient } from './x402_ws_client';
import { x402 } from 'x402';  // Your x402 client library

const client = new X402WebSocketClient({
  baseUrl: 'http://localhost:8001',
  walletAddress: '0x742d35Cc6634C0532925a3b844Bc9e7595f2bD45',
  network: 'eth-mainnet',

  // Implement payment with your x402 library
  paymentProvider: async (requirements) => {
    const receipt = await x402.pay({
      payTo: requirements.pay_to,
      maxAmountRequired: requirements.cost_per_session,
      network: requirements.payment_network,
    });
    return receipt;
  },

  onConnected: (info) => {
    console.log(`Session: ${info.session_id}`);
    console.log(`Expires: ${info.expires_at}`);
  },

  onUpdate: (update) => {
    console.log(`Portfolio: $${update.total_value_usd.toLocaleString()}`);
    update.positions.forEach(pos => {
      console.log(`  ${pos.symbol}: $${pos.value_usd.toFixed(2)} (${(pos.weight * 100).toFixed(1)}%)`);
    });
  },

  onSessionEnding: (info) => {
    console.log(`Session ending in ${info.seconds_remaining}s - reconnect with new payment`);
  },

  onError: (error) => {
    console.error(`Error: ${error.code} - ${error.message}`);
  },

  onDisconnect: (reason) => {
    console.log(`Disconnected: ${reason}`);
  },
});

// Connect and start streaming
await client.connect();

// Session will auto-expire after max_duration (default 5 min)
// Reconnect with new payment to continue
*/

export default X402WebSocketClient;
