import { wrapFetchWithPayment, createSigner } from "x402-fetch";
import dotenv from "dotenv";

dotenv.config();

const MACRO_CRYPTO_URL = process.env.MACRO_CRYPTO_URL || "https://r3467d7khd8b94sfguhrr273lo.ingress.akashprovid.com";

// Type definitions for API responses
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

function showHelp() {
  console.log(`
MacroCrypto CLI - AI-powered crypto portfolio advisor

USAGE:
  npm run client <command> [options]

COMMANDS:
  regime                        Get current macro regime ($0.01)
  portfolio [wallet] [network]  Analyze portfolio without LLM ($0.05)
  advise [wallet] [network]     Full advisory with LLM ($0.10)
  chat [message] [sessionId]    Chat about portfolio/regime ($0.02)
  signal [network]              Submit on-chain regime signal update ($0.10)
  news                          Smart money flow analysis ($0.10)
  help                          Show this help message

EXAMPLES:
  npm run client regime
  npm run client portfolio 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
  npm run client advise 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 eth-mainnet
  npm run client chat "What should I do with my portfolio?"
  npm run client chat "How risky is my portfolio?" my-session-123
  npm run client signal base-sepolia
  npm run client news

OPTIONS:
  wallet    - Ethereum wallet address (defaults to your wallet)
  network   - Network to query (default: eth-mainnet)
  message   - Chat message to send
  sessionId - Chat session ID (auto-generated if not provided)
  network   - Network to query (default: eth-mainnet)

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
