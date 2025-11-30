import { wrapFetchWithPayment, createSigner } from "x402-fetch";
import dotenv from "dotenv";
import WebSocket from "ws";

dotenv.config();

const MACRO_CRYPTO_URL = process.env.MACRO_CRYPTO_URL || "http://localhost:8015";

// =============================================================================
// Type definitions for API responses
// =============================================================================

interface SignalResponse {
  backend_submitted: boolean;
  transaction_hash?: string;
  error?: string;
  risk_on: boolean;
  confidence: number;
  signature: string;
  snapshot_hash: string;
  signer_address: string;
}

interface NewsResponse {
  generated_at: string;
  update_frequency: string;
  analysis: string;
  data_sources: string[];
  note: string;
}

// Wallet Performance
interface WalletPerformanceResponse {
  wallet_address: string;
  total_return: number;
  cagr: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  volatility: number;
  var_95: number;
  cvar_95: number;
  max_drawdown: number;
  beta?: number;
  alpha?: number;
  treynor_ratio?: number;
  information_ratio?: number;
  start_value: number;
  end_value: number;
  start_date: string;
  end_date: string;
  days: number;
  gas_spent_eth?: number;
}

// Wallet Historical
interface WalletHistoricalResponse {
  wallet_address: string;
  start_date: string;
  end_date: string;
  returns: Array<Record<string, unknown>>;
  composition: Record<string, unknown>;
  daily_balances: Record<string, unknown>;
}

// Wallet Composition
interface WalletCompositionResponse {
  wallet_address: string;
  start_date: string;
  end_date: string;
  dates: string[];
  composition: Record<string, number[]>;
  total_value_usd: number[];
}

// Wallet Rolling Metrics
interface WalletRollingResponse {
  wallet_address: string;
  window: number;
  metrics_calculated: string[];
  start_date: string;
  end_date: string;
  dates: string[];
  rolling_sharpe?: (number | null)[];
  rolling_sortino?: (number | null)[];
  rolling_volatility?: (number | null)[];
  rolling_beta?: (number | null)[];
}

// Model Metrics
interface ModelMetricsResponse {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc?: number;
  confusion_matrix?: number[][];
}

// DKG Types
interface DKGInfoResponse {
  status: string;
  node_endpoint: string;
  node_info?: Record<string, unknown>;
  error?: string;
}

interface DKGPublishResponse {
  success: boolean;
  ual?: string;
  dataset_root?: string;
  error?: string;
}

interface DKGSnapshotResponse {
  ual: string;
  regime: string;
  confidence: number;
  timestamp: string;
  btc_price?: number;
  snapshot_hash?: string;
}

interface DKGVerifyResponse {
  ual: string;
  verification_result: string;
  regime_matches: boolean;
  confidence_difference: number;
  computed_regime?: string;
  computed_confidence?: number;
  error?: string;
}

interface DKGLatestResponse {
  ual?: string;
  status: string;
  message?: string;
}

interface DKGUalsResponse {
  uals: Array<{
    ual: string;
    timestamp: number;
    regime: string;
    signer: string;
  }>;
  total: number;
}

// WebSocket Messages
interface WSConnectedMessage {
  type: "connected";
  session_id: string;
  wallet: string;
  network: string;
  update_interval: number;
  max_duration: number;
  expires_at: string;
}

interface WSPortfolioUpdate {
  type: "portfolio_update";
  timestamp: string;
  total_value_usd: number;
  change_24h?: number;
  positions: Array<{
    symbol: string;
    value_usd: number;
    weight: number;
  }>;
}

interface WSSessionEnding {
  type: "session_ending";
  seconds_remaining: number;
}

interface WSError {
  type: "error";
  message: string;
  code?: string;
}

type WSMessage = WSConnectedMessage | WSPortfolioUpdate | WSSessionEnding | WSError;

// Helper type for generic JSON responses - using 'any' to allow flexible property access
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type JsonResponse = Record<string, any>;

let PRIVATE_KEY = process.env.CLIENT_PRIVATE_KEY || "";
if (!PRIVATE_KEY.startsWith("0x")) {
  PRIVATE_KEY = `0x${PRIVATE_KEY}`;
}

console.log("[client] Using private key:", PRIVATE_KEY ? `${PRIVATE_KEY.slice(0, 6)}...${PRIVATE_KEY.slice(-4)}` : "NOT SET");

// Create signer for payment
const signer = await createSigner("base-sepolia", PRIVATE_KEY as `0x${string}`);

const walletAddress = process.env.CLIENT_WALLET_ADDRESS || "0xc4179e7dc055a1cfcc99138f99520f4b6abae654";

console.log("[client] Using account address:", walletAddress);

// Wrap the fetch function with payment handling
const fetchWithPay = wrapFetchWithPayment(fetch, signer);

async function callAdvise(wallet?: string, network: string = "eth-mainnet") {
  const targetWallet = wallet || walletAddress;
  console.log(`\n[advise] Analyzing wallet: ${targetWallet} on ${network}`);

  const response1 = await fetchWithPay(`${MACRO_CRYPTO_URL}/advise`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      network,
      wallet_address: targetWallet
    })
  });

  const data1 = await response1.json();
  console.log("\n=== ADVISE RESPONSE ===");
  console.log(JSON.stringify(data1, null, 2));
}

async function callRegime() {
  console.log(`\n[regime] Fetching current macro regime...`);

  const response1 = await fetchWithPay(`${MACRO_CRYPTO_URL}/regime`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json"
    },
  });

  const data1 = await response1.json();
  console.log("\n=== REGIME RESPONSE ===");
  console.log(JSON.stringify(data1, null, 2));
}

async function callPortfolio(wallet?: string, network: string = "eth-mainnet") {
  const targetWallet = wallet || walletAddress;
  console.log(`\n[portfolio] Analyzing wallet: ${targetWallet} on ${network}`);

  const response1 = await fetchWithPay(`${MACRO_CRYPTO_URL}/portfolio`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      network,
      wallet_address: targetWallet
    })
  });

  const data1 = await response1.json();
  console.log("\n=== PORTFOLIO RESPONSE ===");
  console.log(JSON.stringify(data1, null, 2));
}

async function callChat(message?: string, sessionId?: string, wallet?: string, network: string = "eth-mainnet") {
  const targetWallet = wallet || walletAddress;
  const chatMessage = message || "What can you tell me about the current macroeconomic regime and how it affects my crypto portfolio?";
  const session = sessionId || "aeb2ebe3-56f1-4bee-8cbc-384e8d5e0bf6";

  console.log(`\n[chat] Sending message: "${chatMessage.slice(0, 50)}..."`);

  const response1 = await fetchWithPay(`${MACRO_CRYPTO_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      session_id: session,
      message: chatMessage,
      wallet_address: targetWallet,
      network
    })
  });

  const data1 = await response1.json();
  console.log("\n=== CHAT RESPONSE ===");
  console.log(JSON.stringify(data1, null, 2));
}

async function callSignal(network: string = "base-sepolia") {
  console.log(`\n[signal] Submitting on-chain regime signal update on ${network}...`);
  const response1 = await fetchWithPay(`${MACRO_CRYPTO_URL}/signal`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      network
    })
  });

  const data1 = await response1.json() as SignalResponse;
  console.log("\n=== SIGNAL RESPONSE ===");
  console.log(JSON.stringify(data1, null, 2));

  // Check if backend submitted successfully
  if (data1.backend_submitted) {
    console.log("\n✓ Backend submitted transaction successfully!");
    console.log(`TX Hash: ${data1.transaction_hash}`);
    console.log(`View: https://sepolia.basescan.org/tx/${data1.transaction_hash}`);
  } else {
    console.log("\n⚠ Backend submission failed:", data1.error || "Unknown error");
    console.log("\n📝 Signed data available for manual submission:");
    console.log(`  Risk-On: ${data1.risk_on}`);
    console.log(`  Confidence: ${data1.confidence}%`);
    console.log(`  Signature: ${data1.signature.substring(0, 20)}...`);
    console.log(`  Snapshot Hash: ${data1.snapshot_hash.substring(0, 20)}...`);
    console.log(`  Signer: ${data1.signer_address}`);
    console.log("\nYou can submit this transaction yourself using the provided signature.");
    console.log("See test_signal_with_fallback.py for an example.");
  }
}

async function callNews() {
  console.log(`\n[news] Getting smart money flow analysis...`);
  const response1 = await fetchWithPay(`${MACRO_CRYPTO_URL}/news`, {
    method: "GET",
    headers: {
      "Content-Type": "application/json"
    }
  });

  const data1 = await response1.json() as NewsResponse;
  console.log("\n=== SMART MONEY NEWS ===");
  console.log(`Generated: ${data1.generated_at}`);
  console.log(`Update Frequency: ${data1.update_frequency}`);
  console.log(`\n${data1.analysis}`);
  console.log(`\nData Sources: ${data1.data_sources.join(', ')}`);
  console.log(`\nNote: ${data1.note}`);
}

// =============================================================================
// Wallet Analytics Endpoints
// =============================================================================

async function callWalletPerformance(wallet?: string, network: string = "eth-mainnet") {
  const targetWallet = wallet || walletAddress;
  console.log(`\n[performance] Getting metrics for: ${targetWallet} on ${network}`);

  const response = await fetchWithPay(`${MACRO_CRYPTO_URL}/wallet/performance`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      wallet_address: targetWallet,
      network
    })
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText })) as JsonResponse;
    console.error(`\n=== ERROR ${response.status} ===`);
    console.error(error.detail || JSON.stringify(error));
    return;
  }

  const data = await response.json() as JsonResponse;
  console.log("\n=== WALLET PERFORMANCE ===");
  console.log(`Period: ${data.start_date || 'N/A'} to ${data.end_date || 'N/A'} (${data.days || 0} days)`);
  console.log(`Value: $${(data.start_value || 0).toLocaleString()} → $${(data.end_value || 0).toLocaleString()}`);
  console.log(`\nReturns:`);
  console.log(`  Total Return: ${((data.total_return || 0) * 100).toFixed(2)}%`);
  console.log(`  CAGR: ${((data.cagr || 0) * 100).toFixed(2)}%`);
  console.log(`\nRisk Metrics:`);
  console.log(`  Volatility: ${((data.volatility || 0) * 100).toFixed(2)}%`);
  console.log(`  Max Drawdown: ${((data.max_drawdown || 0) * 100).toFixed(2)}%`);
  console.log(`  VaR (95%): ${((data.var_95 || 0) * 100).toFixed(2)}%`);
  console.log(`  CVaR (95%): ${((data.cvar_95 || 0) * 100).toFixed(2)}%`);
  console.log(`\nRisk-Adjusted:`);
  console.log(`  Sharpe Ratio: ${(data.sharpe_ratio || 0).toFixed(3)}`);
  console.log(`  Sortino Ratio: ${(data.sortino_ratio || 0).toFixed(3)}`);
  console.log(`  Calmar Ratio: ${(data.calmar_ratio || 0).toFixed(3)}`);
  if (data.beta != null) {
    console.log(`\nBenchmark (vs BTC):`);
    console.log(`  Beta: ${data.beta.toFixed(3)}`);
    console.log(`  Alpha: ${data.alpha?.toFixed(3) || 'N/A'}`);
  }
}

async function callWalletHistorical(wallet?: string, network: string = "eth-mainnet") {
  const targetWallet = wallet || walletAddress;
  console.log(`\n[historical] Getting historical data for: ${targetWallet} on ${network}`);

  const response = await fetchWithPay(`${MACRO_CRYPTO_URL}/wallet/historical`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      wallet_address: targetWallet,
      network
    })
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText })) as JsonResponse;
    console.error(`\n=== ERROR ${response.status} ===`);
    console.error(error.detail || JSON.stringify(error));
    return;
  }

  const data = await response.json() as JsonResponse;
  console.log("\n=== WALLET HISTORICAL DATA ===");
  console.log(`Period: ${data.start_date || 'N/A'} to ${data.end_date || 'N/A'}`);
  console.log(`Returns entries: ${(data.returns || []).length}`);
  console.log(`Composition tokens: ${Object.keys(data.composition || {}).length}`);
  console.log(JSON.stringify(data, null, 2));
}

async function callWalletComposition(wallet?: string, network: string = "eth-mainnet") {
  const targetWallet = wallet || walletAddress;
  console.log(`\n[composition] Getting composition for: ${targetWallet} on ${network}`);

  const response = await fetchWithPay(`${MACRO_CRYPTO_URL}/wallet/composition`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      wallet_address: targetWallet,
      network
    })
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText })) as JsonResponse;
    console.error(`\n=== ERROR ${response.status} ===`);
    console.error(error.detail || JSON.stringify(error));
    return;
  }

  const data = await response.json() as JsonResponse;
  console.log("\n=== WALLET COMPOSITION ===");
  console.log(`Period: ${data.start_date || 'N/A'} to ${data.end_date || 'N/A'}`);
  console.log(`Data points: ${(data.dates || []).length}`);
  console.log(`\nTokens tracked:`);
  for (const [symbol, weights] of Object.entries(data.composition || {})) {
    const w = weights as number[];
    const latest = w[w.length - 1] || 0;
    console.log(`  ${symbol}: ${(latest * 100).toFixed(1)}% (latest)`);
  }
  const values = data.total_value_usd || [];
  console.log(`\nLatest total value: $${(values[values.length - 1] || 0).toLocaleString()}`);
}

async function callWalletRolling(wallet?: string, network: string = "eth-mainnet", window: number = 30) {
  const targetWallet = wallet || walletAddress;
  console.log(`\n[rolling] Getting ${window}-day rolling metrics for: ${targetWallet} on ${network}`);

  const response = await fetchWithPay(`${MACRO_CRYPTO_URL}/wallet/rolling`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      wallet_address: targetWallet,
      network,
      window,
      metrics: ["sharpe", "sortino", "volatility", "beta"]
    })
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText })) as JsonResponse;
    console.error(`\n=== ERROR ${response.status} ===`);
    console.error(error.detail || JSON.stringify(error));
    return;
  }

  const data = await response.json() as JsonResponse;
  console.log("\n=== ROLLING METRICS ===");
  console.log(`Period: ${data.start_date || 'N/A'} to ${data.end_date || 'N/A'}`);
  console.log(`Window: ${data.window || window} days`);
  console.log(`Metrics: ${(data.metrics_calculated || []).join(", ")}`);

  // Show latest values
  const dates = data.dates || [];
  const lastIdx = dates.length - 1;
  if (lastIdx >= 0) {
    console.log(`\nLatest (${dates[lastIdx]}):`);
    if (data.rolling_sharpe) console.log(`  Sharpe: ${data.rolling_sharpe[lastIdx]?.toFixed(3) || 'N/A'}`);
    if (data.rolling_sortino) console.log(`  Sortino: ${data.rolling_sortino[lastIdx]?.toFixed(3) || 'N/A'}`);
    if (data.rolling_volatility) console.log(`  Volatility: ${((data.rolling_volatility[lastIdx] || 0) * 100).toFixed(2)}%`);
    if (data.rolling_beta) console.log(`  Beta: ${data.rolling_beta[lastIdx]?.toFixed(3) || 'N/A'}`);
  } else {
    console.log("\nNo data points available");
  }
}

// =============================================================================
// Model Endpoints (FREE)
// =============================================================================

async function callModelHistorical(format: string = "json") {
  console.log(`\n[model/historical] Getting backtest results (${format})...`);

  const response = await fetch(`${MACRO_CRYPTO_URL}/model/historical?format=${format}`);

  if (format === "csv") {
    const data = await response.text();
    console.log("\n=== MODEL HISTORICAL (CSV) ===");
    // Show first 2000 chars for CSV
    console.log(data.slice(0, 2000) + (data.length > 2000 ? "\n...[truncated]" : ""));
  } else {
    const data = await response.json() as JsonResponse;
    console.log("\n=== MODEL HISTORICAL (JSON) ===");
    // For large responses, show summary first
    const cumReturns = data.cumulative_returns as unknown[] | undefined;
    if (cumReturns) {
      console.log(`Data points: ${cumReturns.length}`);
      const metricsObj = data.metrics as Record<string, Record<string, number>> | undefined;
      console.log(`Strategies: ${Object.keys(metricsObj || {}).join(', ') || 'N/A'}`);
      if (metricsObj) {
        console.log("\nMetrics Summary:");
        for (const [strategy, metrics] of Object.entries(metricsObj)) {
          console.log(`  ${strategy}: CAGR=${((metrics.cagr || 0) * 100).toFixed(2)}%, Sharpe=${(metrics.sharpe_ratio || 0).toFixed(3)}`);
        }
      }
      console.log("\nFull response:");
    }
    console.log(JSON.stringify(data, null, 2));
  }
}

async function callLatestReport() {
  console.log(`\n[latest_report] Getting cached regime report...`);

  const response = await fetch(`${MACRO_CRYPTO_URL}/latest_report`);
  const data = await response.json() as JsonResponse;
  console.log("\n=== LATEST REGIME REPORT ===");
  console.log(JSON.stringify(data, null, 2));
}

async function callModelMetrics() {
  console.log(`\n[model/metrics] Getting model performance metrics...`);

  const response = await fetch(`${MACRO_CRYPTO_URL}/model/metrics`);
  const data = await response.json() as JsonResponse;
  console.log("\n=== MODEL METRICS ===");
  console.log(JSON.stringify(data, null, 2));
}

// =============================================================================
// DKG Endpoints (FREE)
// =============================================================================

async function callDKGInfo() {
  console.log(`\n[dkg/info] Getting DKG node info...`);

  const response = await fetch(`${MACRO_CRYPTO_URL}/dkg/info`);
  const data = await response.json() as DKGInfoResponse;
  console.log("\n=== DKG NODE INFO ===");
  console.log(`Status: ${data.status}`);
  console.log(`Endpoint: ${data.node_endpoint}`);
  if (data.node_info) console.log(`Node Info: ${JSON.stringify(data.node_info, null, 2)}`);
  if (data.error) console.log(`Error: ${data.error}`);
}

async function callDKGPublish(epochsNum: number = 2) {
  console.log(`\n[dkg/publish] Publishing regime snapshot to DKG...`);

  const response = await fetch(`${MACRO_CRYPTO_URL}/dkg/publish`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ epochs_num: epochsNum })
  });

  const data = await response.json() as DKGPublishResponse;
  console.log("\n=== DKG PUBLISH RESULT ===");
  console.log(`Success: ${data.success}`);
  if (data.ual) console.log(`UAL: ${data.ual}`);
  if (data.dataset_root) console.log(`Dataset Root: ${data.dataset_root}`);
  if (data.error) console.log(`Error: ${data.error}`);
}

async function callDKGQuery(regime?: string, minConfidence?: number, limit: number = 10) {
  console.log(`\n[dkg/query] Querying DKG snapshots...`);

  const response = await fetch(`${MACRO_CRYPTO_URL}/dkg/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      regime,
      min_confidence: minConfidence,
      limit
    })
  });

  const data = await response.json() as JsonResponse | JsonResponse[];
  console.log("\n=== DKG QUERY RESULTS ===");

  // Handle both array response and object with results
  const snapshots = (Array.isArray(data) ? data : ((data as JsonResponse).results || [])) as JsonResponse[];
  console.log(`Found ${snapshots.length} snapshots:`);

  for (const snapshot of snapshots) {
    const conf = typeof snapshot.confidence === 'number' ? (snapshot.confidence * 100).toFixed(1) : 'N/A';
    console.log(`\n  [${snapshot.timestamp || 'N/A'}] ${snapshot.regime || 'N/A'} (${conf}%)`);
    console.log(`  UAL: ${snapshot.ual || 'N/A'}`);
    if (snapshot.btc_price) console.log(`  BTC: $${(snapshot.btc_price as number).toLocaleString()}`);
  }
}

async function callDKGSnapshot(ual: string) {
  console.log(`\n[dkg/snapshot] Getting snapshot: ${ual}`);

  const response = await fetch(`${MACRO_CRYPTO_URL}/dkg/snapshot/${encodeURIComponent(ual)}`);
  const data = await response.json() as JsonResponse;
  console.log("\n=== DKG SNAPSHOT ===");
  console.log(JSON.stringify(data, null, 2));
}

async function callDKGVerify(ual: string) {
  console.log(`\n[dkg/verify] Verifying snapshot: ${ual}`);

  const response = await fetch(`${MACRO_CRYPTO_URL}/dkg/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ual })
  });

  const data = await response.json() as JsonResponse;
  console.log("\n=== DKG VERIFICATION ===");
  console.log(`Result: ${data.verification_result || 'N/A'}`);
  console.log(`Regime Matches: ${data.regime_matches ?? 'N/A'}`);
  const confDiff = typeof data.confidence_difference === 'number' ? (data.confidence_difference * 100).toFixed(2) : 'N/A';
  console.log(`Confidence Difference: ${confDiff}%`);
  if (data.computed_regime) console.log(`Computed Regime: ${data.computed_regime}`);
  if (data.computed_confidence != null) console.log(`Computed Confidence: ${((data.computed_confidence as number) * 100).toFixed(1)}%`);
  if (data.error) console.log(`Error: ${data.error}`);
}

async function callDKGLatest() {
  console.log(`\n[dkg/latest] Getting latest DKG snapshot...`);

  const response = await fetch(`${MACRO_CRYPTO_URL}/dkg/latest`);
  const data = await response.json() as DKGLatestResponse;
  console.log("\n=== DKG LATEST ===");
  console.log(`Status: ${data.status}`);
  if (data.ual) console.log(`UAL: ${data.ual}`);
  if (data.message) console.log(`Message: ${data.message}`);
}

async function callDKGUals(limit: number = 10, sinceTimestamp?: number) {
  console.log(`\n[dkg/uals] Getting DKG UAL history...`);

  let url = `${MACRO_CRYPTO_URL}/dkg/uals?limit=${limit}`;
  if (sinceTimestamp) url += `&since_timestamp=${sinceTimestamp}`;

  const response = await fetch(url);
  const data = await response.json() as JsonResponse;
  console.log("\n=== DKG UAL HISTORY ===");
  console.log(`Count: ${data.count || 0}`);
  if (data.oracle_address) console.log(`Oracle: ${data.oracle_address}`);
  const uals = (data.uals || []) as JsonResponse[];
  for (const entry of uals) {
    const date = new Date((entry.timestamp as number) * 1000).toISOString();
    console.log(`  [${date}] ${entry.regime} - ${(entry.ual as string)?.slice(0, 50) || 'N/A'}...`);
  }
}

// =============================================================================
// WebSocket Live Portfolio Streaming
// =============================================================================

class LivePortfolioClient {
  private baseUrl: string;
  private ws: WebSocket | null = null;
  private sessionInfo: WSConnectedMessage | null = null;

  constructor(baseUrl: string = MACRO_CRYPTO_URL) {
    this.baseUrl = baseUrl;
  }

  private getWsUrl(wallet: string, network: string): string {
    const base = this.baseUrl.replace("http://", "ws://").replace("https://", "wss://");
    return `${base}/wallet/live?address=${wallet}&network=${network}`;
  }

  private getHttpUrl(wallet: string, network: string): string {
    return `${this.baseUrl}/wallet/live?address=${wallet}&network=${network}`;
  }

  async connect(wallet: string, network: string = "eth-mainnet"): Promise<void> {
    console.log("1. Making x402 payment via HTTP...");

    // Make payment via HTTP first (x402-fetch handles 402 → pay → retry)
    const httpUrl = this.getHttpUrl(wallet, network);
    const response = await fetchWithPay(httpUrl);

    if (!response.ok) {
      throw new Error(`Payment failed: ${response.status}`);
    }

    console.log("   ✓ Payment successful");

    // Get receipt from successful request
    const receipt = response.headers.get("X-PAYMENT") || "";

    console.log("2. Connecting to WebSocket...");
    const wsUrl = this.getWsUrl(wallet, network);

    return new Promise((resolve, reject) => {
      const headers: Record<string, string> = {};
      if (receipt) headers["X-PAYMENT"] = receipt;

      this.ws = new WebSocket(wsUrl, { headers });

      this.ws.on("open", () => {
        console.log("   ✓ WebSocket connected");
      });

      this.ws.on("message", (data: WebSocket.Data) => {
        const msg = JSON.parse(data.toString()) as WSMessage;

        if (msg.type === "connected") {
          this.sessionInfo = msg;
          console.log("3. Session established!");
          console.log(`   Session: ${msg.session_id}`);
          console.log(`   Expires: ${msg.expires_at}`);
          console.log();
          resolve();
        } else if (msg.type === "error") {
          reject(new Error(msg.message));
        }
      });

      this.ws.on("error", (err) => {
        reject(err);
      });
    });
  }

  async stream(onUpdate?: (update: WSPortfolioUpdate) => void): Promise<void> {
    if (!this.ws) throw new Error("Not connected");

    console.log("Streaming portfolio updates...");
    console.log("-".repeat(60));

    let updateCount = 0;

    return new Promise((resolve) => {
      this.ws!.on("message", (data: WebSocket.Data) => {
        const msg = JSON.parse(data.toString()) as WSMessage;

        if (msg.type === "portfolio_update") {
          updateCount++;
          this.printUpdate(msg, updateCount);
          if (onUpdate) onUpdate(msg);
        } else if (msg.type === "session_ending") {
          console.log(`\n⚠️  Session ending in ${msg.seconds_remaining}s`);
        } else if (msg.type === "error") {
          if (msg.code === "session_expired") {
            console.log("\n⏰ Session expired");
          } else {
            console.log(`\n❌ Error: ${msg.message}`);
          }
        }
      });

      this.ws!.on("close", (code, reason) => {
        console.log(`\n🔌 Connection closed: ${code} ${reason}`);
        console.log(`\nSession complete: ${updateCount} updates`);
        resolve();
      });
    });
  }

  private printUpdate(data: WSPortfolioUpdate, count: number): void {
    const ts = data.timestamp.slice(0, 19);
    const value = data.total_value_usd;
    const change = data.change_24h;

    let changeStr = "";
    if (change !== undefined) {
      const arrow = change >= 0 ? "📈" : "📉";
      changeStr = ` ${arrow} ${(change * 100).toFixed(2)}%`;
    }

    console.log(`[${ts}] #${count} 💰 $${value.toLocaleString()}${changeStr}`);

    for (const p of data.positions) {
      console.log(`   ${p.symbol}: $${p.value_usd.toLocaleString()} (${(p.weight * 100).toFixed(1)}%)`);
    }
  }

  async disconnect(): Promise<void> {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

async function callLiveStream(wallet?: string, network: string = "eth-mainnet") {
  const targetWallet = wallet || walletAddress;

  console.log();
  console.log("=".repeat(60));
  console.log("  MacroCrypto Live Portfolio Stream");
  console.log("=".repeat(60));
  console.log(`  Wallet:  ${targetWallet}`);
  console.log(`  Network: ${network}`);
  console.log(`  API:     ${MACRO_CRYPTO_URL}`);
  console.log("=".repeat(60));
  console.log();

  const client = new LivePortfolioClient();

  try {
    await client.connect(targetWallet, network);
    await client.stream();
  } catch (error) {
    console.error(`\n❌ ${error}`);
  } finally {
    await client.disconnect();
  }
}

function showHelp() {
  console.log(`
MacroCrypto CLI - AI-powered crypto portfolio advisor

USAGE:
  npm run client <command> [options]

PAID COMMANDS:
  regime                           Get current macro regime ($0.01)
  portfolio [wallet] [network]     Analyze portfolio without LLM ($0.05)
  advise [wallet] [network]        Full advisory with LLM ($0.10)
  chat [message] [sessionId]       Chat about portfolio/regime ($0.02)
  signal [network]                 Submit on-chain regime signal update ($0.10)
  news                             Smart money flow analysis ($0.10)

WALLET ANALYTICS (PAID $0.05):
  wallet:performance [wallet]      Portfolio performance metrics (Sharpe, VaR, etc.)
  wallet:historical [wallet]       Historical returns & composition
  wallet:composition [wallet]      Token allocation over time
  wallet:rolling [wallet] [window] Rolling window metrics (default: 30 days)
  live [wallet] [network]          WebSocket live portfolio stream ($0.05/session)

MODEL ENDPOINTS (FREE):
  model:historical [format]        Backtest results (json or csv)
  model:metrics                    Model accuracy & performance
  latest-report                    Cached regime report

DKG ENDPOINTS (FREE):
  dkg:info                         DKG node connection status
  dkg:latest                       Latest published snapshot UAL
  dkg:uals [limit]                 List of published snapshot UALs
  dkg:publish [epochs]             Publish current regime to DKG
  dkg:query [regime] [confidence]  Query DKG snapshots
  dkg:snapshot <ual>               Get specific snapshot by UAL
  dkg:verify <ual>                 Verify a DKG snapshot

OTHER:
  help                             Show this help message

EXAMPLES:
  npm run client regime
  npm run client portfolio 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
  npm run client wallet:performance 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
  npm run client live 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
  npm run client dkg:query Risk-On 0.8
  npm run client dkg:verify did:dkg:otp/0x1234.../1234567

ENVIRONMENT:
  API running at: ${MACRO_CRYPTO_URL}
  Client wallet address: ${walletAddress}
  Payment network: base-sepolia
`);
}

// CLI entry point
async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  if (!command || command === "help") {
    showHelp();
    return;
  }

  try {
    switch (command.toLowerCase()) {
      // Original paid endpoints
      case "regime":
        await callRegime();
        break;

      case "portfolio":
        await callPortfolio(args[1], args[2]);
        break;

      case "advise":
        await callAdvise(args[1], args[2]);
        break;

      case "chat":
        await callChat(args[1], args[2], args[3], args[4]);
        break;

      case "signal":
        await callSignal(args[1]);
        break;

      case "news":
        await callNews();
        break;

      // Wallet analytics endpoints
      case "wallet:performance":
        await callWalletPerformance(args[1], args[2]);
        break;

      case "wallet:historical":
        await callWalletHistorical(args[1], args[2]);
        break;

      case "wallet:composition":
        await callWalletComposition(args[1], args[2]);
        break;

      case "wallet:rolling":
        await callWalletRolling(args[1], args[2], args[3] ? parseInt(args[3]) : 30);
        break;

      case "live":
        await callLiveStream(args[1], args[2]);
        break;

      // Model endpoints (free)
      case "model:historical":
        await callModelHistorical(args[1] || "json");
        break;

      case "model:metrics":
        await callModelMetrics();
        break;

      case "latest-report":
        await callLatestReport();
        break;

      // DKG endpoints (free)
      case "dkg:info":
        await callDKGInfo();
        break;

      case "dkg:latest":
        await callDKGLatest();
        break;

      case "dkg:uals":
        await callDKGUals(args[1] ? parseInt(args[1]) : 10);
        break;

      case "dkg:publish":
        await callDKGPublish(args[1] ? parseInt(args[1]) : 2);
        break;

      case "dkg:query":
        await callDKGQuery(args[1], args[2] ? parseFloat(args[2]) : undefined);
        break;

      case "dkg:snapshot":
        if (!args[1]) {
          console.error("Error: UAL required. Usage: dkg:snapshot <ual>");
          process.exit(1);
        }
        await callDKGSnapshot(args[1]);
        break;

      case "dkg:verify":
        if (!args[1]) {
          console.error("Error: UAL required. Usage: dkg:verify <ual>");
          process.exit(1);
        }
        await callDKGVerify(args[1]);
        break;

      default:
        console.error(`\nUnknown command: ${command}`);
        console.error(`Run "npm run client help" for usage information\n`);
        process.exit(1);
    }
  } catch (error) {
    console.error("\n❌ Error:", error);
    process.exit(1);
  }
}

main();

// Export for programmatic use
export { LivePortfolioClient };

