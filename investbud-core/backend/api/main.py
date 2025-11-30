"""
MacroCrypto API with X402 payment integration.

FREE Endpoints:
- GET /health - Health check
- GET /latest_report - Cached regime report + model performance
- GET /historical - Backtest results (CSV/JSON)
- GET /model/metrics - Model accuracy and performance metrics

PAID Endpoints:
- GET /regime - Current macro regime ($0.01)
- POST /portfolio - Wallet analysis without LLM ($0.05)
- POST /advise - Full advisory with LLM recommendations ($0.10)
- POST /chat - Stateful conversation about portfolio/regime ($0.02)
"""
from fastapi import FastAPI, HTTPException, Body, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from x402.fastapi.middleware import require_payment
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import os
import json
from web3 import Web3

from src.macrocrypto.data import WalletAnalyzer, CombinedDataPipeline
from src.macrocrypto.models import MacroRegimeClassifier
from src.macrocrypto.utils import calculate_all_metrics, get_llm_client
from src.macrocrypto.utils.cache import get_cache
from src.macrocrypto.db import init_db, get_session_maker, ChatSession
from src.macrocrypto.services import ChatService, WalletHistoryService
from src.macrocrypto.utils.web3_utils import create_and_sign_update, update_signal, get_signal
from src.macrocrypto.config import (
    SUPPORTED_NETWORKS,
    DEFAULT_NETWORK,
    is_network_supported,
    get_network_config,
    get_supported_network_ids,
    resolve_network,
)

load_dotenv()

# Environment variables
PAYTO_ADDRESS = os.getenv("PAYTO_ADDRESS")
ADVISE_COST = os.getenv("ADVISE_COST", "0.10")  # $0.10 for full analysis + LLM
REGIME_COST = os.getenv("REGIME_COST", "0.01")  # $0.01 for regime check
PORTFOLIO_COST = os.getenv("PORTFOLIO_COST", "0.05")  # $0.05 for wallet analysis (no LLM)
CHAT_COST = os.getenv("CHAT_COST", "0.02")  # $0.02 per chat message
SIGNAL_COST = os.getenv("SIGNAL_COST", "0.1")  # $0.10 for oracle signal update
NEWS_COST = os.getenv("NEWS_COST", "0.01")  # $0.01 for smart money analysis
WALLET_ANALYSIS_COST = os.getenv("WALLET_ANALYSIS_COST", "0.10")  # $0.10 for wallet analysis
LIVE_STREAM_COST = os.getenv("LIVE_STREAM_COST", "0.02")  # $0.02 per session
LIVE_STREAM_INTERVAL = int(os.getenv("LIVE_STREAM_INTERVAL", "30"))  # 30 seconds between updates
WEBSOCKET_MAX_TIME = int(os.getenv("WEBSOCKET_MAX_TIME", "300"))  # 5 minutes max per paid session
NETWORK = os.getenv("PAYMENT_NETWORK", "base-sepolia")
MAINNET = os.getenv("MAINNET", "false").lower() == "true"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"[MacroCrypto] Base Directory: {BASE_DIR}")

# Load environment variables
BASE_SEPOLIA_RPC_URL = os.getenv("BASE_SEPOLIA_RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

# Initialize database
try:
    init_db()
    SessionMaker = get_session_maker()
    print("[OK] Database initialized")
except Exception as e:
    print(f"[WARNING] Database initialization failed: {e}")
    print("[WARNING] Chat endpoint will not be available")
    SessionMaker = None

# Initialize background scheduler for daily data refresh
try:
    from src.macrocrypto.utils.scheduler import start_scheduler
    scheduler = start_scheduler()
except Exception as e:
    print(f"[WARNING] Scheduler initialization failed: {e}")
    scheduler = None

# Initialize FastAPI app
app = FastAPI(
    title="MacroCrypto API",
    description="""AI-powered macro regime classification for crypto portfolio advisory.

## Supported Networks

This API supports wallet analysis across multiple EVM-compatible networks.
See the `/config` endpoint for a complete list of supported networks.

**Networks with internal transfer tracking:** eth-mainnet, polygon-mainnet

**Other supported networks:** arb-mainnet, opt-mainnet, base-mainnet, and 25+ more.

## Quick Start

1. Check `/config` for supported networks
2. Use `/portfolio` for wallet analysis
3. Use `/advise` for AI-powered recommendations
""",
    version="1.0.0"
)

# CORS middleware - allow all origins for API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f'MacroCrypto API starting...')
print(f'Payment address: {PAYTO_ADDRESS}')
print(f'Network: {NETWORK}')
print(f'Costs: Advise=${ADVISE_COST}, Regime=${REGIME_COST}, Portfolio=${PORTFOLIO_COST}, Chat=${CHAT_COST}, News=${NEWS_COST}, LiveStream=${LIVE_STREAM_COST}/interval')

# Configure custom facilitator URL if needed
facilitator_url = os.getenv("FACILITATOR_URL")
facilitator_config = {"url": facilitator_url} if facilitator_url else None

if facilitator_config:
    print(f'X402 facilitator: {facilitator_url}')

# Global instances (lazy loaded)
_classifier = None
_wallet_analyzer = None
_wallet_history_services: Dict[str, WalletHistoryService] = {}  # Cache per network

# Alchemy API key for wallet history
ALCHEMY_API_KEY = os.getenv("ALCHEMY_API_KEY")

w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC_URL))
account = w3.eth.account.from_key(PRIVATE_KEY)
w3.eth.default_account = account.address

print(f"✓ Using account: {account.address}")

# Load contract ABI and address
oracle_abi_path = os.path.join(BASE_DIR, '..', 'contracts', 'out', 'MacroSignalOracle.sol', 'MacroSignalOracle.json')
oracle_address_path = os.path.join(BASE_DIR, '..', 'contracts', 'deployments', 'MacroCryptoOracle.json')

with open(oracle_abi_path) as f:
    abi = json.load(f)['abi']

with open(oracle_address_path) as f:
    contract_address = Web3.to_checksum_address(json.load(f)["deployedTo"])

contract = w3.eth.contract(address=contract_address, abi=abi)

def get_oracle_network(network_name: str):
    """Get Web3 instance and contract for specified network."""
    if network_name == "base-sepolia":
        w3_instance = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC_URL))
        contract_instance = w3_instance.eth.contract(address=contract_address, abi=abi)
        return w3_instance, contract_instance
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported network: {network_name}")

def update_signal_job():
    """Background job to update oracle signal daily."""
    try:
        from src.macrocrypto.utils.web3_utils import create_and_sign_update, update_signal, get_signal
        from src.macrocrypto.data import get_latest_market_data

        print("[Scheduler] Starting daily oracle signal update...")

        # Fetch latest market data
        market_data = get_latest_market_data()

        # Predict current regime
        classifier = get_classifier()
        regime_result = classifier.predict_current_regime(verbose=False)

        risk_on = bool(regime_result['regime_binary'])

        print(f"[Scheduler] Current Regime: {'Risk-On' if risk_on else 'Risk-Off'} (Confidence: {regime_result['confidence']:.2f})")

        confidence = int(regime_result['confidence'] * 100)
        timestamp = int(market_data['timestamp'])

        # Create and sign update
        signed_update = create_and_sign_update(
            market_data=market_data,
            risk_on=risk_on,
            confidence=confidence,
            timestamp=timestamp,
            private_key=PRIVATE_KEY
        )

        # Submit update to contract
        tx_hash = update_signal(
            contract=contract,
            w3=w3,
            risk_on=risk_on,
            confidence=confidence,
            snapshot_hash=signed_update['snapshot_hash'],
            timestamp=timestamp,
            signature=signed_update['signature'],
            private_key=PRIVATE_KEY
        )

        print(f"[Scheduler] Oracle signal updated successfully. TX Hash: {'0x' + tx_hash.hex()}")

    except Exception as e:
        print(f"[Scheduler] Failed to update oracle signal: {e}")

def get_classifier() -> MacroRegimeClassifier:
    """Get or load the trained classifier."""
    global _classifier
    if _classifier is None:
        _classifier = MacroRegimeClassifier()
        try:
            _classifier.load('models/regime_classifier.pkl')
            print("[OK] Classifier loaded")
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="Trained model not found. Please train the classifier first."
            )
    return _classifier

def get_wallet_analyzer() -> WalletAnalyzer:
    """Get or create the wallet analyzer."""
    global _wallet_analyzer
    if _wallet_analyzer is None:
        _wallet_analyzer = WalletAnalyzer()
        print("[OK] Wallet analyzer initialized")
    return _wallet_analyzer

def get_wallet_history_service(network: str = DEFAULT_NETWORK) -> WalletHistoryService:
    """Get or create a wallet history service for the specified network."""
    global _wallet_history_services

    # Validate network
    if not is_network_supported(network):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported network: {network}. Supported networks: {get_supported_network_ids()}"
        )

    if network not in _wallet_history_services:
        if not ALCHEMY_API_KEY:
            raise HTTPException(status_code=500, detail="ALCHEMY_API_KEY not configured")
        _wallet_history_services[network] = WalletHistoryService(ALCHEMY_API_KEY, network=network)
        print(f"[OK] Wallet history service initialized for {network}")

    return _wallet_history_services[network]

# Request/Response Models
class WalletAdviseRequest(BaseModel):
    wallet_address: str = Field(..., description="Ethereum wallet address (0x...)")
    network: Optional[str] = Field(None, description="Network to query (eth-mainnet, polygon-mainnet, etc.). See /config for full list.")
    chain_id: Optional[int] = Field(None, description="Chain ID (e.g., 1 for Ethereum, 137 for Polygon). Alternative to network.")

class RegimeResponse(BaseModel):
    regime: str = Field(..., description="Current market regime (Risk-On or Risk-Off)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    risk_on_probability: float = Field(..., description="Probability of Risk-On regime")
    date: str = Field(..., description="Prediction date")
    features: Optional[Dict] = Field(None, description="Key macro/crypto features")
    dkg_ual: Optional[str] = Field(None, description="OriginTrail DKG Universal Asset Locator")
    dkg_network: Optional[str] = Field(None, description="DKG blockchain network (e.g., otp:20430)")

class MacroSignal(BaseModel):
    regime: str = Field(..., description="Current market regime (Risk-On or Risk-Off)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    explanation: str = Field(..., description="LLM explanation of regime reasoning")
    features: Optional[Dict] = Field(None, description="Key macro/crypto features")

class PortfolioHolding(BaseModel):
    symbol: str
    name: str
    balance: float
    price_usd: float
    value_usd: float
    weight: float

class PortfolioAnalysis(BaseModel):
    wallet: str = Field(..., description="Wallet address")
    network: str = Field(..., description="Network name")
    total_value_usd: float = Field(..., description="Total portfolio value in USD")
    composition: Dict[str, float] = Field(..., description="Token allocation percentages")
    holdings: List[PortfolioHolding] = Field(..., description="Detailed holdings")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Portfolio metrics")

class Recommendation(BaseModel):
    summary: str = Field(..., description="Brief recommendation summary")
    actionable_steps: List[str] = Field(..., description="Specific action items")

class WalletAdviseResponse(BaseModel):
    advice_id: str = Field(..., description="Unique advice identifier")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    macro_signal: MacroSignal
    portfolio_analysis: PortfolioAnalysis
    recommendation: Recommendation

class PortfolioRequest(BaseModel):
    wallet_address: str = Field(..., description="Ethereum wallet address")
    network: Optional[str] = Field(None, description="Network to query (eth-mainnet, polygon-mainnet, etc.). See /config for full list.")
    chain_id: Optional[int] = Field(None, description="Chain ID (e.g., 1 for Ethereum, 137 for Polygon). Alternative to network.")
    include_metrics: bool = Field(False, description="Include historical performance metrics (Sharpe, Sortino, VaR, etc.) - slower")

class PortfolioResponse(BaseModel):
    wallet: str
    network: str
    total_value_usd: float
    composition: Dict[str, float]
    holdings: List[PortfolioHolding]
    metrics: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID (auto-generated if not provided)")
    message: str = Field(..., description="User message")
    wallet_address: Optional[str] = Field(None, description="Optional wallet address for context")
    network: Optional[str] = Field(None, description="Network to query (eth-mainnet, polygon-mainnet, etc.). See /config for full list.")
    chain_id: Optional[int] = Field(None, description="Chain ID (e.g., 1 for Ethereum, 137 for Polygon). Alternative to network.")

class ChatResponse(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    response: str = Field(..., description="AI response")
    message_count: int = Field(..., description="Total messages in conversation")
    context_used: Dict[str, bool] = Field(..., description="Context availability flags")

class OnchainSignalRequest(BaseModel):
    network: str = Field("base-sepolia", description="Network where the oracle is deployed")

class OnchainSignalResponse(BaseModel):
    risk_on: bool = Field(..., description="Current on-chain regime signal (True=Risk-On, False=Risk-Off)")
    confidence: int = Field(..., description="Confidence level (0-100)")
    last_updated: int = Field(..., description="Last updated timestamp (Unix epoch)")
    transaction_hash: Optional[str] = Field(None, description="Transaction hash of the update (if backend submitted successfully)")
    backend_submitted: bool = Field(..., description="Whether backend successfully submitted the transaction")

    # Signed data for caller to submit if backend fails
    signature: str = Field(..., description="Cryptographic signature from trusted signer")
    snapshot_hash: str = Field(..., description="Hash of market data snapshot")
    signer_address: str = Field(..., description="Address that signed the update")
    error: Optional[str] = Field(None, description="Error message if backend submission failed")

class WalletPerformanceRequest(BaseModel):
    wallet_address: str = Field(..., description="Ethereum wallet address (0x...)")
    network: Optional[str] = Field(None, description="Network to query (eth-mainnet, polygon-mainnet, etc.). See /config for full list.")
    chain_id: Optional[int] = Field(None, description="Chain ID (e.g., 1 for Ethereum, 137 for Polygon). Alternative to network.")

class WalletPerformanceResponse(BaseModel):
    wallet_address: str
    total_return: float = Field(..., description="Total return as decimal (e.g., -0.42 = -42%)")
    cagr: float = Field(..., description="Compound annual growth rate")
    sharpe_ratio: float = Field(..., description="Sharpe ratio (annualized)")
    sortino_ratio: float = Field(..., description="Sortino ratio (annualized)")
    calmar_ratio: float = Field(..., description="Calmar ratio (CAGR / Max Drawdown)")
    volatility: float = Field(..., description="Annualized volatility")
    var_95: float = Field(..., description="Value at Risk (95%)")
    cvar_95: float = Field(..., description="Conditional VaR / Expected Shortfall (95%)")
    max_drawdown: float = Field(..., description="Maximum drawdown")
    beta: Optional[float] = Field(None, description="Beta relative to BTC")
    alpha: Optional[float] = Field(None, description="Jensen's alpha (annualized)")
    treynor_ratio: Optional[float] = Field(None, description="Treynor ratio")
    information_ratio: Optional[float] = Field(None, description="Information ratio")
    start_value: float = Field(..., description="Portfolio starting value USD")
    end_value: float = Field(..., description="Portfolio ending value USD")
    start_date: str = Field(..., description="Analysis start date")
    end_date: str = Field(..., description="Analysis end date")
    days: int = Field(..., description="Number of days analyzed")
    gas_spent_eth: Optional[float] = Field(None, description="Total gas spent in ETH")

class WalletHistoricalRequest(BaseModel):
    wallet_address: str = Field(..., description="Ethereum wallet address (0x...)")
    network: Optional[str] = Field(None, description="Network to query (eth-mainnet, polygon-mainnet, etc.). See /config for full list.")
    chain_id: Optional[int] = Field(None, description="Chain ID (e.g., 1 for Ethereum, 137 for Polygon). Alternative to network.")

class WalletHistoricalResponse(BaseModel):
    wallet_address: str
    start_date: str
    end_date: str
    returns: List[Dict[str, Any]] = Field(..., description="Daily returns timeseries")
    composition: Dict[str, Any] = Field(..., description="Portfolio weights over time")
    daily_balances: Dict[str, Any] = Field(..., description="Daily balances per token")

class WalletCompositionRequest(BaseModel):
    wallet_address: str = Field(..., description="Ethereum wallet address (0x...)")
    network: Optional[str] = Field(None, description="Network to query (eth-mainnet, polygon-mainnet, etc.). See /config for full list.")
    chain_id: Optional[int] = Field(None, description="Chain ID (e.g., 1 for Ethereum, 137 for Polygon). Alternative to network.")

class WalletCompositionResponse(BaseModel):
    wallet_address: str
    start_date: str
    end_date: str
    dates: List[str] = Field(..., description="Array of dates")
    composition: Dict[str, List[float]] = Field(..., description="Token weights over time {symbol: [weights]}")
    total_value_usd: List[float] = Field(..., description="Total portfolio value per day")

class WalletRollingRequest(BaseModel):
    wallet_address: str = Field(..., description="Ethereum wallet address (0x...)")
    network: Optional[str] = Field(None, description="Network to query (eth-mainnet, polygon-mainnet, etc.). See /config for full list.")
    chain_id: Optional[int] = Field(None, description="Chain ID (e.g., 1 for Ethereum, 137 for Polygon). Alternative to network.")
    window: int = Field(30, description="Rolling window in days", ge=7, le=365)
    metrics: List[str] = Field(
        ["sharpe", "sortino", "volatility"],
        description="Metrics to calculate: sharpe, sortino, volatility, beta"
    )

class WalletRollingResponse(BaseModel):
    wallet_address: str
    window: int
    metrics_calculated: List[str]
    start_date: str
    end_date: str
    dates: List[str] = Field(..., description="Array of dates")
    rolling_sharpe: Optional[List[Optional[float]]] = Field(None, description="Rolling Sharpe ratio")
    rolling_sortino: Optional[List[Optional[float]]] = Field(None, description="Rolling Sortino ratio")
    rolling_volatility: Optional[List[Optional[float]]] = Field(None, description="Rolling annualized volatility")
    rolling_beta: Optional[List[Optional[float]]] = Field(None, description="Rolling beta vs BTC")


# WebSocket Message Models (for documentation/validation)
# Note: Actual messages are sent as plain dicts, these are for reference

class WSConnected(BaseModel):
    """Connection established message."""
    type: str = "connected"
    session_id: str = Field(..., description="WebSocket session ID")
    wallet: str = Field(..., description="Wallet address being tracked")
    network: str = Field(..., description="Network")
    update_interval: int = Field(..., description="Seconds between updates")
    max_duration: int = Field(..., description="Max session duration in seconds")
    expires_at: str = Field(..., description="ISO timestamp when session expires")


# WebSocket helper
from datetime import datetime, timezone, timedelta


# Apply X402 payment middleware
app.middleware("http")(
    require_payment(
        price=ADVISE_COST,
        pay_to_address=PAYTO_ADDRESS,
        network=NETWORK,
        path=["/advise"],
        facilitator_config=facilitator_config,
        max_deadline_seconds=300  # 5 minutes for complex wallet analysis
    )
)

app.middleware("http")(
    require_payment(
        price=REGIME_COST,
        pay_to_address=PAYTO_ADDRESS,
        network=NETWORK,
        path=["/regime"],
        facilitator_config=facilitator_config,
        max_deadline_seconds=120  # 2 minutes
    )
)

app.middleware("http")(
    require_payment(
        price=PORTFOLIO_COST,
        pay_to_address=PAYTO_ADDRESS,
        network=NETWORK,
        path=["/portfolio"],
        facilitator_config=facilitator_config,
        max_deadline_seconds=300  # 5 minutes for portfolio analysis
    )
)

app.middleware("http")(
    require_payment(
        price=CHAT_COST,
        pay_to_address=PAYTO_ADDRESS,
        network=NETWORK,
        path=["/chat"],
        facilitator_config=facilitator_config,
        max_deadline_seconds=180  # 3 minutes for chat
    )
)

app.middleware("http")(
    require_payment(
        price=SIGNAL_COST,
        pay_to_address=PAYTO_ADDRESS,
        network=NETWORK,
        path=["/signal"],
        facilitator_config=facilitator_config,
        max_deadline_seconds=120  # 2 minutes for signal update
    )
)

app.middleware("http")(
    require_payment(
        price=NEWS_COST,
        pay_to_address=PAYTO_ADDRESS,
        network=NETWORK,
        path=["/news"],
        facilitator_config=facilitator_config,
        max_deadline_seconds=30  # 30 seconds for cached data
    )
)

app.middleware("http")(
    require_payment(
        price=WALLET_ANALYSIS_COST,
        pay_to_address=PAYTO_ADDRESS,
        network=NETWORK,
        path=["/wallet/historical", "/wallet/composition", "/wallet/rolling", "/wallet/performance"],
        facilitator_config=facilitator_config,
        max_deadline_seconds=300  # 5 minutes for cached data
    )
)

# WebSocket live stream - payment validated at HTTP upgrade handshake
app.middleware("http")(
    require_payment(
        price=LIVE_STREAM_COST,
        pay_to_address=PAYTO_ADDRESS,
        network=NETWORK,
        path=["/wallet/live"],
        facilitator_config=facilitator_config,
        max_deadline_seconds=WEBSOCKET_MAX_TIME  # Session duration
    )
)

# Routes

@app.get("/")
async def root():
    """Health check endpoint (free)."""
    return {
        "service": "MacroCrypto API",
        "status": "healthy",
        "version": "1.0.0",
        "description": "AI-powered macro regime classification for crypto portfolio advisory",
        "endpoints": {
            "free": {
                "/health": "Health check",
                "/config": "API configuration (supported networks, etc.)",
                "/latest_report": "Cached regime report + model performance",
                "/model/historical": "Backtest results (CSV/JSON)",
                "/model/metrics": "Model accuracy and performance metrics",
                "GET /wallet/live": "WebSocket endpoint documentation (connect via ws:// protocol)"
            },
            "paid": {
                "/regime": f"${REGIME_COST} - Current macro regime",
                "/portfolio": f"${PORTFOLIO_COST} - Wallet analysis (no LLM)",
                "/advise": f"${ADVISE_COST} - Full advisory (wallet + regime + LLM)",
                "/chat": f"${CHAT_COST} - Stateful conversation about portfolio/regime",
                "/signal": f"${SIGNAL_COST} - Submit oracle regime signal update",
                "/news": f"${NEWS_COST} - Smart money flow analysis (Nansen + Heurist + GPT-4)",
                "/wallet/historical": f"${WALLET_ANALYSIS_COST} - Wallet historical performance and composition",
                "/wallet/composition": f"${WALLET_ANALYSIS_COST} - Wallet composition over time",
                "/wallet/rolling": f"${WALLET_ANALYSIS_COST} - Wallet rolling performance metrics",
                "/wallet/performance": f"${WALLET_ANALYSIS_COST} - Wallet performance metrics"
            },
            "websocket": {
                "WS /wallet/live": f"${LIVE_STREAM_COST}/interval - Real-time portfolio streaming with x402 micropayments"
            },
            "dkg": {
                "/dkg/info": "DKG node connection status",
                "/dkg/publish": "Publish regime snapshot to OriginTrail DKG",
                "/dkg/query": "Query historical snapshots from DKG",
                "/dkg/snapshot/{ual}": "Get specific snapshot by UAL",
                "/dkg/verify": "Verify a snapshot (act as confirming node)",
                "/dkg/latest": "Get latest published snapshot UAL"
            }
        },
        "payment_network": NETWORK,
        "docs": "/docs",
        "config": "/config"
    }


@app.get("/config")
async def get_config():
    """
    Get API configuration including supported networks.

    **Cost: FREE**

    Returns:
    - Supported blockchain networks for wallet analysis
    - Network capabilities (internal transfers support)
    - Default network
    - Chain ID to network mapping
    - API version info
    """
    networks_info = {}
    chain_id_lookup = {}
    for network_id, config in SUPPORTED_NETWORKS.items():
        networks_info[network_id] = {
            "name": config["name"],
            "chain_id": config["chain_id"],
            "internal_transfers": config["internal_transfers"],
        }
        chain_id_lookup[config["chain_id"]] = network_id

    return {
        "version": "1.0.0",
        "default_network": DEFAULT_NETWORK,
        "supported_networks": networks_info,
        "chain_id_lookup": chain_id_lookup,
        "network_count": len(SUPPORTED_NETWORKS),
        "usage_note": "You can specify network using either 'network' (string like 'eth-mainnet') or 'chain_id' (integer like 1). chain_id takes priority if both are provided.",
        "internal_transfers_note": "Internal transfer tracking (contract calls, DEX swaps) is only available on Ethereum (chain_id: 1) and Polygon (chain_id: 137) mainnets.",
        "data_source": "Alchemy Transfers API",
        "documentation": "https://www.alchemy.com/docs/data/transfers-api/transfers-endpoints/alchemy-get-asset-transfers"
    }

@app.get("/health")
async def health():
    """Detailed health check (free)."""
    try:
        classifier = get_classifier()
        return {
            "status": "healthy",
            "classifier_loaded": classifier is not None,
            "wallet_analyzer_available": True,
            "payment_network": NETWORK,
            "mainnet": MAINNET
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/news")
async def get_smart_money_news():
    """
    Get smart money flow analysis.

    **Cost: $0.10 in USDC**

    Returns AI-powered analysis of smart money token flows from Nansen,
    combined with news and market analysis from Heurist, summarized by GPT-4.

    **Key Features**:
    - Top 100 tokens by smart money netflow (24h)
    - Identifies accumulation and distribution patterns
    - Contextualizes flows with recent crypto news
    - Actionable insights from GPT-4 analysis
    - Includes token addresses and chains

    **Update Frequency**: Every 6 hours (00:00, 06:00, 12:00, 18:00 UTC)

    **Data Sources**:
    - Nansen Smart Money API (token netflows from Funds & Smart Traders)
    - Heurist AI (crypto news & market analysis from past week)

    **Note**: This endpoint serves cached data that is refreshed automatically
    every 6 hours. You get the same high-quality analysis at a fraction of the
    cost compared to calling the APIs yourself (~$0.27/update).
    """
    try:
        cache = get_cache()
        cached_analysis = cache.get('smart_money_analysis', ttl_seconds=21600)  # 6 hours

        if not cached_analysis:
            from src.macrocrypto.utils.scheduler import refresh_smart_money_analysis
            refresh_smart_money_analysis()
            cached_analysis = cache.get('smart_money_analysis', ttl_seconds=21600)  # 6 hours

            if not cached_analysis:
                raise HTTPException(status_code=500, detail="Failed to fetch smart money analysis")

        return {
            "generated_at": cached_analysis.get('generated_at'),
            "analysis": cached_analysis.get('analysis'),
            "data_sources": ["Nansen Smart Money", "Heurist AI"],
            "update_frequency": "Every 6 hours",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get smart money news: {str(e)}")

@app.post("/advise", response_model=WalletAdviseResponse)
async def advise_wallet(request: WalletAdviseRequest):
    """
    Analyze a wallet and get regime-based investment recommendations.

    **Cost: $0.10 in USDC**

    This endpoint:
    1. Fetches on-chain token holdings
    2. Calculates portfolio composition
    3. Predicts current macro regime (Risk-On/Risk-Off)
    4. Generates LLM-powered investment recommendations
    """
    try:
        from datetime import datetime
        import uuid

        # Validate wallet address
        if not request.wallet_address.startswith('0x') or len(request.wallet_address) != 42:
            raise HTTPException(status_code=400, detail="Invalid Ethereum address")

        # Resolve network from network string or chain_id
        try:
            network = resolve_network(request.network, request.chain_id)
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Get analyzer and classifier
        analyzer = get_wallet_analyzer()
        classifier = get_classifier()

        # Analyze wallet
        analysis = analyzer.analyze_wallet(request.wallet_address, network)

        if 'error' in analysis:
            raise HTTPException(status_code=404, detail=analysis['error'])

        # Get regime prediction
        regime_result = classifier.predict_current_regime(verbose=False)

        # Get quantitative recommendation
        quant_recommendation = analyzer.get_regime_recommendation(analysis, classifier)
        print(f'[OK] Quantitative recommendation: {quant_recommendation}')

        # Fetch historical performance metrics (cached per wallet + network)
        performance_data = None
        try:
            cache = get_cache()
            cache_key = f'wallet_performance:{request.wallet_address.lower()}:{network}'
            cached_perf = cache.get(cache_key, ttl_seconds=3600)  # 1 hour cache

            if cached_perf:
                performance_data = cached_perf
                print(f'[OK] Using cached performance for {request.wallet_address[:10]}... on {network}')
            else:
                # Calculate fresh metrics
                service = get_wallet_history_service(network)
                metrics = service.get_performance_metrics(request.wallet_address)
                if "error" not in metrics:
                    performance_data = metrics
                    cache.set(cache_key, metrics)
                    print(f'[OK] Calculated and cached performance for {request.wallet_address[:10]}... on {network}')
        except Exception as e:
            print(f'[!] Failed to fetch wallet performance: {e}')
            # Continue without performance data - it's optional

        # Format portfolio holdings
        portfolio_holdings = [
            PortfolioHolding(**holding) for holding in analysis['portfolio']
        ]

        # Calculate composition percentages
        composition = {}
        for holding in analysis['portfolio']:
            symbol = holding['symbol']
            weight = holding['weight']
            composition[symbol] = round(weight * 100, 1)

        # Generate LLM-powered recommendation and explanation
        try:
            llm_client = get_llm_client()
            llm_result = await llm_client.generate_recommendation(
                wallet_data={
                    'total_value_usd': analysis['total_value_usd'],
                    'risk_allocation': analysis['risk_allocation'],
                    'stable_allocation': analysis['stable_allocation'],
                    'portfolio': analysis['portfolio']
                },
                regime_data={
                    'regime': regime_result['regime'],
                    'confidence': regime_result['confidence'],
                    'features': regime_result['features']
                },
                quantitative_signal={
                    'action': quant_recommendation['action'],
                    'current_risk_allocation': quant_recommendation['current_risk_allocation'],
                    'optimal_risk_allocation': quant_recommendation['optimal_risk_allocation'],
                    'regime': regime_result['regime']
                },
                performance_data=performance_data
            )
            explanation = llm_result.get('explanation', '')
            print(f'[OK] LLM explanation: {explanation}')
            summary = llm_result.get('summary', '')
            print(f'[OK] LLM summary: {summary}')
            actionable_steps = llm_result.get('actionable_steps', [])
            print(f'[OK] LLM actionable steps: {actionable_steps}')
        except Exception as e:
            print(f"[!] Failed to generate LLM recommendation: {str(e)}")
            # Use fallback
            llm_client = get_llm_client()
            llm_result = llm_client._fallback_recommendation({
                'action': quant_recommendation['action'],
                'current_risk_allocation': quant_recommendation['current_risk_allocation'],
                'optimal_risk_allocation': quant_recommendation['optimal_risk_allocation'],
                'regime': regime_result['regime']
            })
            explanation = llm_result['explanation']
            summary = llm_result['summary']
            actionable_steps = llm_result['actionable_steps']

        # Build response
        return WalletAdviseResponse(
            advice_id=f"adv-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            macro_signal=MacroSignal(
                regime=regime_result['regime'],
                confidence=regime_result['confidence'],
                explanation=explanation,
                features={
                    'btc_price': regime_result['features'].get('btc_price'),
                    'btc_returns_30d': regime_result['features'].get('btc_returns_30d'),
                    'btc_rsi': regime_result['features'].get('btc_rsi'),
                    'btc_drawdown': regime_result['features'].get('btc_drawdown'),
                    'fed_funds': regime_result['features'].get('fed_funds'),
                    'cpi_yoy': regime_result['features'].get('cpi_yoy'),
                    'vix': regime_result['features'].get('vix'),
                    'yield_curve_spread': regime_result['features'].get('yield_curve_spread')
                }
            ),
            portfolio_analysis=PortfolioAnalysis(
                wallet=analysis['wallet_address'],
                network=network,
                total_value_usd=analysis['total_value_usd'],
                composition=composition,
                holdings=portfolio_holdings,
                metrics=performance_data
            ),
            recommendation=Recommendation(
                summary=summary,
                actionable_steps=actionable_steps
            )
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze wallet: {str(e)}")

@app.get("/regime", response_model=RegimeResponse)
async def get_regime():
    """
    Get current macro regime prediction.

    **Cost: $0.01 in USDC**

    Returns:
    - Current regime (Risk-On or Risk-Off)
    - Confidence score
    - Risk-On probability
    - Key macro/crypto features
    """
    try:
        classifier = get_classifier()
        result = classifier.predict_current_regime(verbose=False)

        # Get cached DKG info (if published)
        cache = get_cache()
        dkg_ual = cache.get("latest_dkg_ual", ttl_seconds=86400)
        dkg_network = cache.get("latest_dkg_network", ttl_seconds=86400)

        return RegimeResponse(
            regime=result['regime'],
            confidence=result['confidence'],
            risk_on_probability=result['risk_on_probability'],
            date=str(result['date']),
            features={
                'btc_price': result['features'].get('btc_price'),
                'btc_returns_30d': result['features'].get('btc_returns_30d'),
                'btc_rsi': result['features'].get('btc_rsi'),
                'btc_drawdown': result['features'].get('btc_drawdown'),
                'fed_funds': result['features'].get('fed_funds'),
                'cpi_yoy': result['features'].get('cpi_yoy'),
                'vix': result['features'].get('vix'),
                'yield_curve_spread': result['features'].get('yield_curve_spread')
            },
            dkg_ual=dkg_ual,
            dkg_network=dkg_network,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to predict regime: {str(e)}")

@app.post("/portfolio", response_model=PortfolioResponse)
async def analyze_portfolio(request: PortfolioRequest):
    """
    Analyze wallet portfolio composition and metrics.

    **Cost: $0.05 in USDC**

    This is a lightweight version of `/advise` without LLM recommendations.

    **Parameters:**
    - `wallet_address`: Ethereum wallet address (0x...)
    - `network`: Network to query (default: eth-mainnet)
    - `include_metrics`: If true, includes historical performance metrics (slower)

    **Returns:**
    - Portfolio holdings and composition
    - Basic metrics (risk allocation, diversification)
    - Historical metrics if `include_metrics=true` (Sharpe, Sortino, VaR, max drawdown, etc.)
    """
    try:
        # Validate wallet address
        if not request.wallet_address.startswith('0x') or len(request.wallet_address) != 42:
            raise HTTPException(status_code=400, detail="Invalid Ethereum address")

        # Resolve network from network string or chain_id
        try:
            network = resolve_network(request.network, request.chain_id)
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Get analyzer
        analyzer = get_wallet_analyzer()

        # Analyze wallet
        analysis = analyzer.analyze_wallet(request.wallet_address, network)
        print(f'analysis: {analysis}')

        if 'error' in analysis:
            raise HTTPException(status_code=404, detail=analysis['error'])

        # Format portfolio holdings
        portfolio_holdings = [
            PortfolioHolding(**holding) for holding in analysis['portfolio']
        ]

        # Calculate composition percentages
        composition = {}
        for holding in analysis['portfolio']:
            symbol = holding['symbol']
            weight = holding['weight']
            composition[symbol] = round(weight * 100, 1)

        # Calculate basic metrics
        metrics = {
            "risk_allocation": round(analysis['risk_allocation'], 4),
            "stable_allocation": round(analysis['stable_allocation'], 4),
            "diversification_score": round(1.0 - max([h['weight'] for h in analysis['portfolio']]), 4),
            "top_holding_concentration": round(max([h['weight'] for h in analysis['portfolio']]), 4)
        }

        # Fetch historical performance metrics if requested
        if request.include_metrics:
            try:
                cache = get_cache()
                cache_key = f'wallet_performance:{request.wallet_address.lower()}:{network}'
                cached_perf = cache.get(cache_key, ttl_seconds=3600)  # 1 hour cache

                if cached_perf:
                    print(f'[OK] Using cached performance metrics for {request.wallet_address[:10]}... on {network}')
                    # Merge historical metrics into response
                    metrics.update(cached_perf)
                else:
                    # Calculate fresh metrics
                    service = get_wallet_history_service(network)
                    perf_metrics = service.get_performance_metrics(request.wallet_address)
                    if "error" not in perf_metrics:
                        cache.set(cache_key, perf_metrics)
                        print(f'[OK] Calculated and cached performance for {request.wallet_address[:10]}... on {network}')
                        metrics.update(perf_metrics)
                    else:
                        print(f'[!] Failed to get performance metrics: {perf_metrics.get("error")}')
            except Exception as e:
                print(f'[!] Failed to fetch historical metrics: {e}')

        return PortfolioResponse(
            wallet=analysis['wallet_address'],
            network=network,
            total_value_usd=analysis['total_value_usd'],
            composition=composition,
            holdings=portfolio_holdings,
            metrics=metrics
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze portfolio: {str(e)}")

@app.post("/wallet/performance", response_model=WalletPerformanceResponse)
async def get_wallet_performance(request: WalletPerformanceRequest):
    """
    Get comprehensive performance metrics for a wallet's historical portfolio.

    Calculates institutional-grade metrics including:
    - Sharpe Ratio, Sortino Ratio, Calmar Ratio
    - VaR/CVaR (95%)
    - Maximum Drawdown
    - Beta, Alpha, Treynor Ratio (vs BTC benchmark)
    - CAGR and total return

    Note: Requires wallet to have on-chain transfer history.
    """
    try:
        # Validate wallet address
        if not request.wallet_address.startswith('0x') or len(request.wallet_address) != 42:
            raise HTTPException(status_code=400, detail="Invalid Ethereum address")

        # Resolve network from network string or chain_id
        try:
            network = resolve_network(request.network, request.chain_id)
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Get wallet history service for network
        service = get_wallet_history_service(network)

        # Calculate performance metrics
        metrics = service.get_performance_metrics(request.wallet_address)

        if "error" in metrics:
            raise HTTPException(status_code=404, detail=metrics["error"])

        return WalletPerformanceResponse(
            wallet_address=request.wallet_address.lower(),
            total_return=metrics.get("total_return", 0),
            cagr=metrics.get("cagr", 0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0),
            sortino_ratio=metrics.get("sortino_ratio", 0),
            calmar_ratio=metrics.get("calmar_ratio", 0),
            volatility=metrics.get("volatility", 0),
            var_95=metrics.get("var_95", 0),
            cvar_95=metrics.get("cvar_95", 0),
            max_drawdown=metrics.get("max_drawdown", 0),
            beta=metrics.get("beta"),
            alpha=metrics.get("alpha"),
            treynor_ratio=metrics.get("treynor_ratio"),
            information_ratio=metrics.get("information_ratio"),
            start_value=metrics.get("start_value", 0),
            end_value=metrics.get("end_value", 0),
            start_date=metrics.get("start_date", ""),
            end_date=metrics.get("end_date", ""),
            days=metrics.get("days", 0),
            gas_spent_eth=metrics.get("gas_spent_eth"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate performance: {str(e)}")

@app.post("/wallet/historical", response_model=WalletHistoricalResponse)
async def get_wallet_historical(request: WalletHistoricalRequest):
    """
    Get historical portfolio data for a wallet.

    Returns:
    - Daily returns timeseries
    - Portfolio composition (weights) over time
    - Daily balances per token with USD values

    Note: Requires wallet to have on-chain transfer history.
    """
    try:
        # Validate wallet address
        if not request.wallet_address.startswith('0x') or len(request.wallet_address) != 42:
            raise HTTPException(status_code=400, detail="Invalid Ethereum address")

        # Resolve network from network string or chain_id
        try:
            network = resolve_network(request.network, request.chain_id)
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Get wallet history service for network
        service = get_wallet_history_service(network)

        # Get historical data
        data = service.get_historical_data(request.wallet_address)

        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])

        return WalletHistoricalResponse(
            wallet_address=request.wallet_address.lower(),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            returns=data.get("returns", []),
            composition=data.get("composition", {}),
            daily_balances=data.get("daily_balances", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get historical data: {str(e)}")

@app.post("/wallet/composition", response_model=WalletCompositionResponse)
async def get_wallet_composition(request: WalletCompositionRequest):
    """
    Get portfolio composition (weights) over time.

    Returns:
    - Array of dates
    - Token weights per day {ETH: [0.5, 0.48, ...], LINK: [0.3, 0.32, ...]}
    - Total portfolio value USD per day

    Useful for visualizing how portfolio allocation has changed over time.
    """
    try:
        # Validate wallet address
        if not request.wallet_address.startswith('0x') or len(request.wallet_address) != 42:
            raise HTTPException(status_code=400, detail="Invalid Ethereum address")

        # Resolve network from network string or chain_id
        try:
            network = resolve_network(request.network, request.chain_id)
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Get wallet history service for network
        service = get_wallet_history_service(network)

        # Get composition data
        data = service.get_composition_history(request.wallet_address)

        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])

        return WalletCompositionResponse(
            wallet_address=data.get("wallet_address", request.wallet_address.lower()),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            dates=data.get("dates", []),
            composition=data.get("composition", {}),
            total_value_usd=data.get("total_value_usd", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get composition data: {str(e)}")

@app.post("/wallet/rolling", response_model=WalletRollingResponse)
async def get_wallet_rolling_metrics(request: WalletRollingRequest):
    """
    Calculate rolling window metrics for time-series visualization.

    Args:
    - window: Rolling window in days (default 30, range 7-365)
    - metrics: List of metrics to calculate (sharpe, sortino, volatility, beta)

    Returns:
    - Rolling Sharpe ratio (annualized)
    - Rolling Sortino ratio (annualized)
    - Rolling volatility (annualized)
    - Rolling beta vs BTC (if requested)

    Note: First `window` days will be null as they require full window of data.
    Beta requires BTC benchmark data and may have gaps.
    """
    try:
        # Validate wallet address
        if not request.wallet_address.startswith('0x') or len(request.wallet_address) != 42:
            raise HTTPException(status_code=400, detail="Invalid Ethereum address")

        # Resolve network from network string or chain_id
        try:
            network = resolve_network(request.network, request.chain_id)
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Validate metrics
        valid_metrics = {'sharpe', 'sortino', 'volatility', 'beta'}
        for m in request.metrics:
            if m not in valid_metrics:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid metric '{m}'. Valid options: {valid_metrics}"
                )

        # Get wallet history service for network
        service = get_wallet_history_service(network)

        # Get rolling metrics
        data = service.get_rolling_metrics(
            wallet_address=request.wallet_address,
            window=request.window,
            metrics=request.metrics
        )

        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])

        return WalletRollingResponse(
            wallet_address=data.get("wallet_address", request.wallet_address.lower()),
            window=data.get("window", request.window),
            metrics_calculated=data.get("metrics_calculated", request.metrics),
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            dates=data.get("dates", []),
            rolling_sharpe=data.get("rolling_sharpe"),
            rolling_sortino=data.get("rolling_sortino"),
            rolling_volatility=data.get("rolling_volatility"),
            rolling_beta=data.get("rolling_beta"),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate rolling metrics: {str(e)}")


@app.get("/wallet/live")
async def wallet_live_docs():
    """
    WebSocket endpoint documentation for real-time portfolio streaming.

    **This is a WebSocket endpoint** - connect via `ws://` or `wss://` protocol.

    **Cost: $0.02 per session (max {WEBSOCKET_MAX_TIME}s)**

    ## Connection Flow (x402 at handshake)

    1. Client fetches payment requirements from this endpoint
    2. Client pays via x402 and obtains receipt
    3. Client connects to WebSocket with `X-PAYMENT` header
    4. Server validates payment and starts streaming
    5. Connection auto-closes after max session time

    ## Headers Required

    | Header | Description |
    |--------|-------------|
    | X-PAYMENT | Base64-encoded x402 payment receipt |
    | X-PAYMENT-PAYLOAD | Optional payment payload |

    ## Query Parameters

    | Parameter | Type | Required | Description |
    |-----------|------|----------|-------------|
    | address | string | Yes | Wallet address (0x...) |
    | network | string | No | Network ID (default: eth-mainnet) |
    | chain_id | int | No | Chain ID (alternative to network) |

    ## Message Types (Server → Client)

    ### `connected` - Session started
    ```json
    {
      "type": "connected",
      "session_id": "uuid-...",
      "wallet": "0x...",
      "network": "eth-mainnet",
      "update_interval": 30,
      "max_duration": 300,
      "expires_at": "2024-01-15T10:35:00Z"
    }
    ```

    ### `portfolio_update` - Real-time data
    ```json
    {
      "type": "portfolio_update",
      "timestamp": "2024-01-15T10:30:00Z",
      "total_value_usd": 15234.56,
      "change_24h": 0.023,
      "positions": [...],
      "metrics": {...}
    }
    ```

    ### `error` - Error
    ```json
    {"type": "error", "code": "...", "message": "..."}
    ```

    ### `session_ending` - Warning before timeout
    ```json
    {"type": "session_ending", "seconds_remaining": 30}
    ```

    ## Example (JavaScript with x402)

    ```javascript
    import { paymentMiddleware } from 'x402';

    // 1. Get payment requirements
    const info = await fetch('/wallet/live').then(r => r.json());

    // 2. Pay and get receipt
    const receipt = await x402.pay({
      payTo: info.pay_to,
      maxAmountRequired: info.cost_per_session,
      network: info.payment_network,
    });

    // 3. Connect with payment header
    const ws = new WebSocket(
      'ws://host/wallet/live?address=0x...',
      { headers: { 'X-PAYMENT': receipt } }
    );

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'portfolio_update') {
        console.log(`Value: $${msg.total_value_usd}`);
      }
    };
    ```
    """
    return {
        "endpoint": "WS /wallet/live",
        "protocol": "WebSocket",
        "description": "Real-time portfolio streaming with x402 payment at handshake",
        "cost_per_session": LIVE_STREAM_COST,
        "max_session_seconds": WEBSOCKET_MAX_TIME,
        "update_interval_seconds": LIVE_STREAM_INTERVAL,
        "payment_network": NETWORK,
        "pay_to": PAYTO_ADDRESS,
        "connection_url": "ws://{host}/wallet/live?address={wallet}&network={network}",
        "required_headers": {
            "X-PAYMENT": "Base64-encoded x402 payment receipt",
            "X-PAYMENT-PAYLOAD": "Optional payment payload",
        },
        "query_params": {
            "address": "Wallet address (0x...) - required",
            "network": "Network ID (default: eth-mainnet)",
            "chain_id": "Chain ID (alternative to network)",
        },
        "message_types": {
            "server_to_client": ["connected", "portfolio_update", "error", "session_ending", "pong"],
            "client_to_server": ["ping"],
        },
        "flow": [
            "1. GET /wallet/live to get payment requirements",
            "2. Pay via x402 to obtain receipt",
            "3. Connect to WebSocket with X-PAYMENT header",
            "4. Receive 'connected' message with session info",
            "5. Receive portfolio_update messages every 30s",
            "6. Session auto-closes after max duration",
        ],
    }


@app.websocket("/wallet/live")
async def wallet_live_stream(
    websocket: WebSocket,
    address: str = Query(..., description="Wallet address to track"),
    network: Optional[str] = Query(None, description="Network (default: eth-mainnet)"),
    chain_id: Optional[int] = Query(None, description="Chain ID (alternative to network)"),
):
    """
    WebSocket endpoint for real-time portfolio tracking.

    Payment is verified via x402 middleware at HTTP upgrade handshake.
    Session auto-closes after WEBSOCKET_MAX_TIME seconds.
    """
    import uuid

    session_id = str(uuid.uuid4())
    print(f"[WS] New connection: {session_id} for wallet {address[:10]}...")

    # Validate wallet address
    if not address.startswith('0x') or len(address) != 42:
        await websocket.close(code=1008, reason="Invalid wallet address")
        return

    # Resolve network
    try:
        resolved_network = resolve_network(network, chain_id)
    except (KeyError, ValueError) as e:
        await websocket.close(code=1008, reason=str(e))
        return

    # Payment already verified by x402 middleware at HTTP upgrade handshake
    # Accept connection
    await websocket.accept()
    session_start = datetime.now(timezone.utc)
    session_expires = session_start + timedelta(seconds=WEBSOCKET_MAX_TIME)

    print(f"[WS] Session {session_id} started, expires at {session_expires.isoformat()}")

    try:
        # Send connected message
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "wallet": address.lower(),
            "network": resolved_network,
            "update_interval": LIVE_STREAM_INTERVAL,
            "max_duration": WEBSOCKET_MAX_TIME,
            "expires_at": session_expires.isoformat(),
        })

        analyzer = get_wallet_analyzer()
        update_count = 0
        prev_value = None
        warning_sent = False

        # Streaming loop with timeout
        while datetime.now(timezone.utc) < session_expires:
            try:
                # Check remaining time
                remaining = (session_expires - datetime.now(timezone.utc)).total_seconds()

                # Send warning 30s before expiry
                if remaining <= 30 and not warning_sent:
                    await websocket.send_json({
                        "type": "session_ending",
                        "seconds_remaining": int(remaining),
                    })
                    warning_sent = True

                # Fetch portfolio data
                analysis = analyzer.analyze_wallet(address, resolved_network)

                if 'error' in analysis:
                    await websocket.send_json({
                        "type": "error",
                        "code": "analysis_error",
                        "message": analysis['error'],
                    })
                else:
                    current_value = analysis['total_value_usd']
                    change_24h = None
                    if prev_value is not None and prev_value > 0:
                        change_24h = (current_value - prev_value) / prev_value
                    prev_value = current_value

                    positions = [
                        {
                            "symbol": h['symbol'],
                            "name": h['name'],
                            "balance": h['balance'],
                            "price_usd": h['price_usd'],
                            "value_usd": h['value_usd'],
                            "weight": h['weight'],
                        }
                        for h in analysis['portfolio']
                    ]

                    metrics = {
                        "risk_allocation": round(analysis['risk_allocation'], 4),
                        "stable_allocation": round(analysis['stable_allocation'], 4),
                        "diversification_score": round(1.0 - max([h['weight'] for h in analysis['portfolio']], default=0), 4),
                    }

                    # Try cached performance metrics
                    try:
                        cache = get_cache()
                        cache_key = f'wallet_performance:{address.lower()}:{resolved_network}'
                        cached_perf = cache.get(cache_key, ttl_seconds=3600)
                        if cached_perf:
                            metrics.update(cached_perf)
                    except Exception:
                        pass

                    await websocket.send_json({
                        "type": "portfolio_update",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "total_value_usd": current_value,
                        "change_24h": change_24h,
                        "positions": positions,
                        "metrics": metrics,
                    })
                    update_count += 1
                    print(f"[WS] Update #{update_count} to {session_id}: ${current_value:,.2f}")

                # Wait for next interval, handling pings
                wait_until = min(LIVE_STREAM_INTERVAL, remaining)
                waited = 0
                while waited < wait_until:
                    try:
                        msg = await asyncio.wait_for(
                            websocket.receive_json(),
                            timeout=min(5.0, wait_until - waited)
                        )
                        if msg.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                    except asyncio.TimeoutError:
                        pass
                    waited += 5.0

            except WebSocketDisconnect:
                print(f"[WS] Client disconnected: {session_id}")
                return
            except Exception as e:
                print(f"[WS] Error in stream: {e}")
                await websocket.send_json({
                    "type": "error",
                    "code": "internal_error",
                    "message": str(e),
                })

        # Session expired
        print(f"[WS] Session {session_id} expired after {update_count} updates")
        await websocket.send_json({
            "type": "error",
            "code": "session_expired",
            "message": f"Session expired after {WEBSOCKET_MAX_TIME}s. Reconnect with new payment.",
        })

    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: {session_id}")
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/model/historical")
async def get_historical_backtest(format: str = 'json'):
    """
    Get historical backtest results with normalized cumulative returns.

    **Cost: FREE**

    Returns model-based vs buy-and-hold performance over time.
    Available formats: json, csv
    """
    try:
        import os
        import pandas as pd
        from fastapi.responses import Response

        # Check if backtest results exist
        backtest_file = 'models/backtest_results.pkl'

        if not os.path.exists(backtest_file):
            raise HTTPException(
                status_code=404,
                detail="Backtest results not found. Please run backtest first."
            )

        # Load backtest results
        import pickle
        with open(backtest_file, 'rb') as f:
            backtest_data = pickle.load(f)

        results = backtest_data['results']
        metrics = backtest_data['metrics']

        # Normalize to starting value of 1.0
        normalized_results = {}
        for strategy, values in results.items():
            initial_value = values.iloc[0]
            normalized_results[strategy] = values / initial_value

        # Create DataFrame
        df = pd.DataFrame(normalized_results)
        df.index.name = 'date'

        if format.lower() == 'csv':
            # Return as CSV
            csv_data = df.to_csv()
            return Response(
                content=csv_data,
                media_type='text/csv',
                headers={'Content-Disposition': 'attachment; filename=backtest_results.csv'}
            )
        else:
            # Helper to safely round values, replacing NaN/inf with 0
            import math
            import numpy as np

            def clean_value(val):
                """Convert any value to JSON-safe format"""
                if val is None:
                    return None
                # Handle numpy types
                if hasattr(val, 'item'):
                    val = val.item()
                # Handle NaN and infinity
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    return 0.0
                # Handle numpy arrays
                if isinstance(val, np.ndarray):
                    return [clean_value(v) for v in val.tolist()]
                # Handle pandas Timestamp
                if hasattr(val, 'isoformat'):
                    return val.isoformat()
                return val

            def clean_dict(obj):
                """Recursively clean all values in a dict/list to be JSON-safe"""
                if isinstance(obj, dict):
                    return {k: clean_dict(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [clean_dict(item) for item in obj]
                else:
                    return clean_value(obj)

            def safe_round(val, decimals=4):
                val = clean_value(val)
                if val is None:
                    return 0.0
                try:
                    return round(float(val), decimals)
                except (ValueError, TypeError):
                    return 0.0

            # Return as JSON
            cumulative_returns = []
            for date, row in df.iterrows():
                entry = {'date': str(date)}
                for strategy in row.index:
                    entry[strategy] = safe_round(row[strategy], 4)
                cumulative_returns.append(entry)

            # Load live agent decisions if available
            live_agent_file = 'models/live_agent_decisions.pkl'
            live_agent_data = None
            if os.path.exists(live_agent_file):
                with open(live_agent_file, 'rb') as f:
                    live_agent_data = pickle.load(f)

            return {
                "start_date": str(df.index[0]),
                "end_date": str(df.index[-1]),
                "initial_capital": backtest_data.get('initial_capital', 10000),
                "strategies": {
                    strategy: {
                        "cagr": safe_round(metrics[strategy].get('cagr', 0), 4),
                        "sharpe_ratio": safe_round(metrics[strategy].get('sharpe_ratio', 0), 4),
                        "sortino_ratio": safe_round(metrics[strategy].get('sortino_ratio', 0), 4),
                        "max_drawdown": safe_round(metrics[strategy].get('max_drawdown', 0), 4),
                        "calmar_ratio": safe_round(metrics[strategy].get('calmar_ratio', 0), 4),
                        "var_95": safe_round(metrics[strategy].get('var_95', 0), 4),
                        "cvar_95": safe_round(metrics[strategy].get('cvar_95', 0), 4),
                        "volatility": safe_round(metrics[strategy].get('volatility', 0), 4),
                        "cumulative_return": safe_round(metrics[strategy].get('cumulative_return', 0), 4),
                        "final_value": safe_round(results[strategy].iloc[-1], 2)
                    }
                    for strategy in metrics.keys()
                },
                "cumulative_returns": cumulative_returns,
                "live_agent": clean_dict(live_agent_data) if live_agent_data else None
            }

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Backtest results not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get historical data: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Stateful chat about portfolio and macro regime.

    **Cost: $0.02 in USDC per message**

    Maintains conversation context using session_id.
    Can reference wallet holdings and current regime.

    **Session Management:**
    - If session_id is not provided, a new UUID will be auto-generated
    - Save your session_id to continue conversations later
    - Session IDs are cryptographically secure (128-bit UUIDs)

    **Privacy Warning:**
    ⚠️ Anyone with your session_id can access your conversation history.
    Do not share your session_id with others.
    """
    if SessionMaker is None:
        raise HTTPException(
            status_code=503,
            detail="Chat service unavailable - database not initialized"
        )

    # Auto-generate session_id if not provided
    import uuid
    session_id = request.session_id or str(uuid.uuid4())

    db_session = SessionMaker()
    try:
        # Check for stored wallet_address in session (for subsequent requests)
        existing_session = db_session.query(ChatSession).filter_by(session_id=session_id).first()
        effective_wallet = request.wallet_address or (existing_session.wallet_address if existing_session else None)

        # Resolve network: request.chain_id > request.network > session.network > default
        if request.chain_id is not None or request.network is not None:
            try:
                effective_network = resolve_network(request.network, request.chain_id)
            except (KeyError, ValueError) as e:
                raise HTTPException(status_code=400, detail=str(e))
        else:
            effective_network = (existing_session.network if existing_session else None) or DEFAULT_NETWORK

        # Initialize chat service with wallet analyzer
        analyzer = get_wallet_analyzer() if effective_wallet else None
        chat_service = ChatService(db_session, wallet_analyzer=analyzer)

        # Fetch current regime context
        try:
            classifier = get_classifier()
            regime_result = classifier.predict_current_regime(verbose=False)
            regime_context = {
                'regime': regime_result['regime'],
                'confidence': regime_result['confidence'],
                'risk_on_probability': regime_result['risk_on_probability'],
                'features': regime_result['features']
            }
        except Exception as e:
            print(f"[!] Failed to fetch regime context: {e}")
            regime_context = None

        # Fetch cached news/smart money analysis
        news_context = None
        try:
            from src.macrocrypto.utils.cache import get_cache
            cache = get_cache()
            cached_news = cache.get('smart_money_analysis', ttl_seconds=21600)  # 6 hours
            if cached_news:
                news_context = {
                    'generated_at': cached_news.get('generated_at'),
                    'analysis': cached_news.get('analysis')
                }
        except Exception as e:
            print(f"[!] Failed to fetch news context: {e}")

        # Fetch wallet performance metrics (cached per wallet + network)
        wallet_performance_context = None
        if effective_wallet:
            try:
                cache = get_cache()
                cache_key = f'wallet_performance:{effective_wallet.lower()}:{effective_network}'
                cached_perf = cache.get(cache_key, ttl_seconds=3600)  # 1 hour cache

                if cached_perf:
                    wallet_performance_context = cached_perf
                else:
                    # Calculate fresh metrics
                    service = get_wallet_history_service(effective_network)
                    metrics = service.get_performance_metrics(effective_wallet)
                    if "error" not in metrics:
                        wallet_performance_context = metrics
                        cache.set(cache_key, metrics)
                        print(f"[OK] Cached wallet performance for {effective_wallet[:10]}... on {effective_network}")
            except Exception as e:
                print(f"[!] Failed to fetch wallet performance: {e}")

        # Process chat message
        result = await chat_service.chat(
            session_id=session_id,
            message=request.message,
            wallet_address=request.wallet_address,
            network=effective_network,
            regime_context=regime_context,
            news_context=news_context,
            wallet_performance_context=wallet_performance_context
        )

        return ChatResponse(**result)

    except Exception as e:
        db_session.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )
    finally:
        db_session.close()

@app.get("/latest_report")
async def get_latest_report():
    """
    Get latest regime report with backtest summary (free).

    Returns cached information about:
    - Current regime
    - Model performance
    - Recent predictions
    """
    try:
        classifier = get_classifier()
        regime_result = classifier.predict_current_regime(verbose=False)

        # Get model metrics
        metrics = classifier.metrics

        # Load dynamic backtest results
        backtest_summary = None
        try:
            import pickle
            backtest_file = 'models/backtest_results.pkl'
            if os.path.exists(backtest_file):
                with open(backtest_file, 'rb') as f:
                    backtest_data = pickle.load(f)

                backtest_metrics = backtest_data['metrics']

                # Extract metrics for each strategy
                model_metrics = backtest_metrics.get('model', {})
                bnh_metrics = backtest_metrics.get('buy_and_hold', {})

                # Calculate alpha (model CAGR - buy-and-hold CAGR)
                model_cagr = model_metrics.get('cagr', 0)
                bnh_cagr = bnh_metrics.get('cagr', 0)
                alpha = model_cagr - bnh_cagr

                # Build dynamic summary
                backtest_summary = {
                    "note": f"Model-based strategy achieves {model_cagr*100:.1f}% CAGR vs {bnh_cagr*100:.1f}% buy-and-hold",
                    "model_based_strategy": {
                        "cagr": round(model_cagr, 4),
                        "sharpe_ratio": round(model_metrics.get('sharpe_ratio', 0), 4),
                        "sortino_ratio": round(model_metrics.get('sortino_ratio', 0), 4),
                        "volatility": round(model_metrics.get('volatility', 0), 4),
                        "max_drawdown": round(model_metrics.get('max_drawdown', 0), 4)
                    },
                    "buy_and_hold": {
                        "cagr": round(bnh_cagr, 4),
                        "sharpe_ratio": round(bnh_metrics.get('sharpe_ratio', 0), 4),
                        "sortino_ratio": round(bnh_metrics.get('sortino_ratio', 0), 4),
                        "volatility": round(bnh_metrics.get('volatility', 0), 4),
                        "max_drawdown": round(bnh_metrics.get('max_drawdown', 0), 4)
                    },
                    "alpha": round(alpha, 4)
                }
        except Exception as e:
            print(f"[!] Failed to load backtest results: {e}")
            backtest_summary = {
                "note": "Backtest results not available",
                "error": "Run backtest to see performance metrics"
            }

        return {
            "current_regime": {
                "regime": regime_result['regime'],
                "confidence": regime_result['confidence'],
                "date": str(regime_result['date'])
            },
            "model_performance": {
                "test_accuracy": metrics['test']['accuracy'] if metrics else None,
                "test_precision": metrics['test']['precision'] if metrics else None,
                "test_recall": metrics['test']['recall'] if metrics else None,
                "test_f1_score": metrics['test']['f1'] if metrics else None,
                "test_roc_auc": metrics['test']['roc_auc'] if metrics else None,
                "training_date": classifier.training_date
            },
            "top_features": [
                "btc_returns_30d",
                "btc_drawdown",
                "btc_price_vs_ma_30",
                "vix_ma_30",
                "yield_curve_spread"
            ],
            "backtest_summary": backtest_summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get report: {str(e)}")

@app.get("/model/metrics")
async def get_model_metrics():
    """
    Get comprehensive model performance metrics.

    **Cost: FREE**

    Returns detailed accuracy metrics including:
    - Train/test accuracy, precision, recall, F1, ROC-AUC
    - Time-series cross-validation results (if available)
    - Confusion matrix
    - Feature importance
    - Training metadata
    """
    try:
        classifier = get_classifier()
        metrics = classifier.metrics

        if not metrics:
            raise HTTPException(status_code=500, detail="Model metrics not available")

        # Get feature importance from model coefficients
        feature_importance = None
        if classifier.model and classifier.feature_names:
            coefficients = classifier.model.coef_[0]
            # Sort by absolute value
            importance_pairs = sorted(
                zip(classifier.feature_names, coefficients),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            feature_importance = [
                {
                    "feature": name,
                    "coefficient": round(float(coef), 6),
                    "abs_coefficient": round(abs(float(coef)), 6)
                }
                for name, coef in importance_pairs[:15]  # Top 15 features
            ]

        return {
            "model_type": "Logistic Regression",
            "training_date": classifier.training_date,
            "total_features": len(classifier.feature_names) if classifier.feature_names else None,

            "train_metrics": {
                "accuracy": round(metrics['train']['accuracy'], 4),
                "precision": round(metrics['train']['precision'], 4),
                "recall": round(metrics['train']['recall'], 4),
                "f1_score": round(metrics['train']['f1'], 4),
                "roc_auc": round(metrics['train']['roc_auc'], 4),
                "confusion_matrix": metrics['train']['confusion_matrix'].tolist() if 'confusion_matrix' in metrics['train'] else None
            },

            "test_metrics": {
                "accuracy": round(metrics['test']['accuracy'], 4),
                "precision": round(metrics['test']['precision'], 4),
                "recall": round(metrics['test']['recall'], 4),
                "f1_score": round(metrics['test']['f1'], 4),
                "roc_auc": round(metrics['test']['roc_auc'], 4),
                "confusion_matrix": metrics['test']['confusion_matrix'].tolist() if 'confusion_matrix' in metrics['test'] else None
            },

            "cross_validation": classifier.cv_metrics if classifier.cv_metrics else None,

            "feature_importance": feature_importance,

            "interpretation": {
                "confusion_matrix_labels": ["Risk-Off (0)", "Risk-On (1)"],
                "accuracy_interpretation": "Percentage of correct predictions",
                "precision_interpretation": "Of predicted Risk-On, how many were correct",
                "recall_interpretation": "Of actual Risk-On, how many were predicted",
                "f1_score_interpretation": "Harmonic mean of precision and recall",
                "roc_auc_interpretation": "Area under ROC curve (0.5 = random, 1.0 = perfect)",
                "cross_validation_interpretation": "Time-series cross-validation provides robust performance estimates across different time periods. Low std indicates model stability."
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model metrics: {str(e)}")

@app.post("/signal", response_model=OnchainSignalResponse)
async def submit_onchain_signal(request: OnchainSignalRequest):
    """
    Submit current macro regime signal to on-chain oracle.

    **Cost: $0.1 in USDC**

    This endpoint:
    1. Predicts current macro regime using the trained classifier
    2. Creates and signs an update for the on-chain oracle contract
    3. Attempts to submit the update transaction to the specified network
    4. **Returns signed data regardless of submission success**

    **Resilient Design:**
    - If backend submission succeeds → `backend_submitted=true` with `transaction_hash`
    - If backend submission fails → `backend_submitted=false` with `signature` and `snapshot_hash`
    - Caller can always submit the transaction themselves using the returned signed data
    - This prevents loss of signed data if backend runs out of gas or has network issues

    """
    try:
        classifier = get_classifier()
        result = classifier.predict_current_regime(verbose=False)

        risk_on = True if result['regime'] == 'Risk-On' else False
        confidence = int(result['confidence'] * 100)
        timestamp = int(result['date'].timestamp())

        # Create and sign update (this always happens)
        signed_update = create_and_sign_update(
            market_data=result['features'],
            risk_on=risk_on,
            confidence=confidence,
            timestamp=timestamp,
            private_key=PRIVATE_KEY
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create signed update: {str(e)}")

    # Try to submit on-chain, but return signed data regardless
    tx_hash = None
    backend_submitted = False
    error_message = None

    try:
        # Get contract for specified network
        w3, contract = get_oracle_network(request.network)

        # Submit update to contract
        tx_hash = update_signal(
            contract=contract,
            w3=w3,
            risk_on=risk_on,
            confidence=confidence,
            snapshot_hash=signed_update['snapshotHash'],
            timestamp=timestamp,
            signature=signed_update['signature'],
            private_key=PRIVATE_KEY
        )

        if tx_hash:
            backend_submitted = True
            tx_hash_hex = "0x" + tx_hash.hex()
        else:
            error_message = "Transaction submission returned None"

    except Exception as e:
        # Log the error but don't fail the request
        error_message = str(e)
        print(f"[WARNING] Backend failed to submit transaction: {error_message}")

    # Always return signed data so caller can submit if backend failed
    return OnchainSignalResponse(
        risk_on=risk_on,
        confidence=confidence,
        last_updated=timestamp,
        transaction_hash=tx_hash_hex if backend_submitted else None,
        backend_submitted=backend_submitted,
        signature=signed_update['signature'],
        snapshot_hash=signed_update['snapshotHash'],
        signer_address=signed_update['signer'],
        error=error_message
    )


# ============================================================================
# DKG (OriginTrail Decentralized Knowledge Graph) Endpoints
# ============================================================================

# DKG Request/Response Models
class DKGPublishRequest(BaseModel):
    """Request to publish current regime snapshot to DKG."""
    epochs_num: int = Field(2, description="Number of epochs for asset validity", ge=1, le=10)
    min_confirmations: int = Field(3, description="Minimum finalization confirmations", ge=1)
    min_replications: int = Field(1, description="Minimum node replications", ge=1)

class DKGPublishResponse(BaseModel):
    """Response from DKG publish operation."""
    success: bool
    ual: Optional[str] = Field(None, description="Universal Asset Locator")
    dataset_root: Optional[str] = Field(None, description="Dataset root hash")
    error: Optional[str] = None

class DKGQueryRequest(BaseModel):
    """Request to query DKG snapshots."""
    regime: Optional[str] = Field(None, description="Filter by regime (Risk-On or Risk-Off)")
    min_confidence: Optional[float] = Field(None, description="Minimum confidence threshold", ge=0, le=1)
    limit: int = Field(10, description="Maximum results", ge=1, le=100)

class DKGSnapshotResponse(BaseModel):
    """Single snapshot from DKG."""
    ual: str
    regime: str
    confidence: float
    timestamp: str
    btc_price: Optional[float] = None
    snapshot_hash: Optional[str] = None

class DKGVerifyRequest(BaseModel):
    """Request to verify a DKG snapshot."""
    ual: str = Field(..., description="Universal Asset Locator of snapshot to verify")

class DKGVerifyResponse(BaseModel):
    """Response from snapshot verification."""
    ual: str
    verification_result: str
    regime_matches: bool
    confidence_difference: float
    computed_regime: Optional[str] = None
    computed_confidence: Optional[float] = None
    error: Optional[str] = None


# Global DKG service instance
_dkg_service = None

def get_dkg_service():
    """Get or create async DKG service."""
    global _dkg_service
    if _dkg_service is None:
        try:
            from src.macrocrypto.services import AsyncDKGService
            node_endpoint = os.getenv("DKG_NODE_ENDPOINT", "http://localhost:8900")
            blockchain_env = os.getenv("DKG_BLOCKCHAIN_ENV", "testnet")
            blockchain_id = os.getenv("DKG_BLOCKCHAIN_ID", "otp:20430")

            _dkg_service = AsyncDKGService(
                node_endpoint=node_endpoint,
                blockchain_environment=blockchain_env,
                blockchain_id=blockchain_id,
            )
            print(f"[OK] DKG service initialized: {node_endpoint}")
        except Exception as e:
            print(f"[!] DKG service not available: {e}")
            return None
    return _dkg_service


@app.get("/dkg/info")
async def dkg_info():
    """
    Get DKG node information.

    **Cost: FREE**

    Returns connection status and node details.
    """
    try:
        from src.macrocrypto.services import DKGService
        node_endpoint = os.getenv("DKG_NODE_ENDPOINT", "http://localhost:8900")

        # Use sync client for simple info call
        dkg = DKGService(node_endpoint=node_endpoint)
        info = dkg.get_node_info()

        return {
            "status": "connected",
            "node_endpoint": node_endpoint,
            "node_info": info,
        }
    except Exception as e:
        return {
            "status": "disconnected",
            "node_endpoint": os.getenv("DKG_NODE_ENDPOINT", "http://localhost:8900"),
            "error": str(e),
        }


@app.post("/dkg/publish", response_model=DKGPublishResponse)
async def dkg_publish(request: DKGPublishRequest = Body(default=DKGPublishRequest())):
    """
    Publish current regime snapshot to OriginTrail DKG.

    **Cost: FREE** (but requires gas for blockchain anchoring)

    Creates a Knowledge Asset containing:
    - Current regime classification
    - Confidence score
    - Market features (BTC, macro indicators)
    - Cryptographic proof (signature, hash)

    The asset is replicated across DKG nodes and linked to previous snapshots.
    """
    try:
        from src.macrocrypto.services import DKGService, RegimeSnapshot

        # Get current regime prediction
        classifier = get_classifier()
        regime_result = classifier.predict_current_regime(verbose=False)

        # Create snapshot
        from datetime import datetime, timezone
        from src.macrocrypto.utils.web3_utils import create_and_sign_update

        timestamp = int(datetime.now(timezone.utc).timestamp())
        market_data = {
            'date': str(regime_result['date']),
            'regime': regime_result['regime'],
            'confidence': regime_result['confidence'],
            'risk_on_probability': regime_result['risk_on_probability'],
            'features': regime_result['features'],
            'timestamp': timestamp
        }

        # Sign the data
        signed_update = create_and_sign_update(
            market_data=market_data,
            risk_on=bool(regime_result['regime_binary']),
            confidence=int(regime_result['confidence'] * 100),
            timestamp=timestamp,
            private_key=PRIVATE_KEY
        )

        # Create RegimeSnapshot
        snapshot = RegimeSnapshot(
            regime=regime_result['regime'],
            regime_binary=regime_result['regime_binary'],
            confidence=regime_result['confidence'],
            risk_on_probability=regime_result['risk_on_probability'],
            timestamp=timestamp,
            date=str(regime_result['date']),
            snapshot_hash=signed_update['snapshotHash'],
            signature=signed_update['signature'],
            signer_address=signed_update['signer'],
            transaction_hash=None,
            features=regime_result['features'],
        )

        # Publish to DKG
        node_endpoint = os.getenv("DKG_NODE_ENDPOINT", "http://localhost:8900")
        blockchain_id = os.getenv("DKG_BLOCKCHAIN_ID", "otp:20430")
        dkg = DKGService(node_endpoint=node_endpoint, blockchain_id=blockchain_id)

        result = dkg.publish_snapshot(
            snapshot=snapshot,
            epochs_num=request.epochs_num,
            min_confirmations=request.min_confirmations,
            min_replications=request.min_replications,
        )

        # Cache the UAL and network for /regime endpoint
        ual = result.get("UAL")
        if ual:
            cache = get_cache()
            cache.set("latest_dkg_ual", ual)
            cache.set("latest_dkg_network", blockchain_id)

            # Store in persistent UAL store for confirming nodes
            from src.macrocrypto.services.dkg_service import get_ual_store
            ual_store = get_ual_store()
            ual_store.save_ual(
                ual=ual,
                timestamp=timestamp,
                regime=regime_result['regime'],
                regime_binary=regime_result.get('regime_binary', 1 if regime_result['regime'] == 'Risk-On' else 0),
                confidence=regime_result.get('confidence'),
                risk_on_probability=regime_result.get('risk_on_probability'),
                signer_address=signed_update['signer'],
                dataset_root=result.get("datasetRoot"),
            )

        return DKGPublishResponse(
            success=True,
            ual=ual,
            dataset_root=result.get("datasetRoot"),
        )

    except Exception as e:
        return DKGPublishResponse(
            success=False,
            error=str(e),
        )


@app.post("/dkg/query", response_model=List[DKGSnapshotResponse])
async def dkg_query(request: DKGQueryRequest = Body(default=DKGQueryRequest())):
    """
    Query regime snapshots from the DKG.

    **Cost: FREE**

    Search the decentralized knowledge graph for historical regime classifications.
    """
    try:
        from src.macrocrypto.services import DKGService

        node_endpoint = os.getenv("DKG_NODE_ENDPOINT", "http://localhost:8900")
        dkg = DKGService(node_endpoint=node_endpoint)

        results = dkg.query_snapshots(
            regime=request.regime,
            min_confidence=request.min_confidence,
            limit=request.limit,
        )

        return [
            DKGSnapshotResponse(
                ual=r.get("snapshot", ""),
                regime=r.get("regime", ""),
                confidence=float(r.get("confidence", 0)),
                timestamp=r.get("timestamp", ""),
                btc_price=float(r.get("btcPrice")) if r.get("btcPrice") else None,
                snapshot_hash=r.get("snapshotHash"),
            )
            for r in results
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DKG query failed: {str(e)}")


@app.get("/dkg/snapshot/{ual:path}")
async def dkg_get_snapshot(ual: str):
    """
    Get a specific snapshot from DKG by UAL.

    **Cost: FREE**

    Returns the full Knowledge Asset data.
    """
    try:
        from urllib.parse import unquote
        from src.macrocrypto.services import DKGService

        # URL-decode the UAL (colons get encoded as %3A)
        ual = unquote(ual)

        node_endpoint = os.getenv("DKG_NODE_ENDPOINT", "http://localhost:8900")
        dkg = DKGService(node_endpoint=node_endpoint)

        result = dkg.get_snapshot(ual)
        return result

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Snapshot not found: {str(e)}")


@app.post("/dkg/verify", response_model=DKGVerifyResponse)
async def dkg_verify(request: DKGVerifyRequest):
    """
    Verify a DKG snapshot by independently computing the prediction.

    **Cost: FREE**

    This endpoint acts as a confirming node, independently computing
    the regime prediction and comparing it to the stored snapshot.

    Returns:
    - Whether the regime matches
    - Confidence difference
    - Verification result (confirmed/challenged/partial)
    """
    try:
        from src.macrocrypto.services import DKGService, GraphConfirmingNode

        node_endpoint = os.getenv("DKG_NODE_ENDPOINT", "http://localhost:8900")
        dkg = DKGService(node_endpoint=node_endpoint)

        # Create confirming node with current classifier
        verifier_address = os.getenv("VERIFIER_ADDRESS", account.address if account else "0x0")
        node = GraphConfirmingNode(
            dkg_service=dkg,
            verifier_address=verifier_address,
            classifier=get_classifier(),
        )

        # Verify the snapshot
        report = node.verify_snapshot(request.ual)

        return DKGVerifyResponse(
            ual=request.ual,
            verification_result=report.verification_result.value,
            regime_matches=report.regime_matches,
            confidence_difference=report.confidence_difference,
            computed_regime=report.computed_regime,
            computed_confidence=report.computed_confidence,
            error=report.error_message,
        )

    except Exception as e:
        return DKGVerifyResponse(
            ual=request.ual,
            verification_result="error",
            regime_matches=False,
            confidence_difference=0,
            error=str(e),
        )


@app.get("/dkg/latest")
async def dkg_latest():
    """
    Get the latest published DKG snapshot UAL.

    **Cost: FREE**

    Returns the most recently published snapshot from the persistent store.
    """
    from src.macrocrypto.services.dkg_service import get_ual_store

    ual_store = get_ual_store()
    latest = ual_store.get_latest()

    if latest:
        return {
            "ual": latest["ual"],
            "status": "found",
            "regime": latest.get("regime"),
            "confidence": latest.get("confidence"),
            "timestamp": latest.get("timestamp"),
            "signer_address": latest.get("signer_address"),
        }
    else:
        return {
            "ual": None,
            "status": "no_snapshots_published",
            "message": "No snapshots have been published to DKG yet. Use POST /dkg/publish to create one.",
        }


@app.get("/dkg/stats")
async def dkg_stats():
    """
    Get statistics about published DKG snapshots.

    **Cost: FREE**

    Returns counts and date ranges of published snapshots.
    """
    from src.macrocrypto.services.dkg_service import get_ual_store

    ual_store = get_ual_store()
    stats = ual_store.get_stats()

    return {
        "total_published": stats["total_published"],
        "risk_on_count": stats["risk_on_count"],
        "risk_off_count": stats["risk_off_count"],
        "first_timestamp": stats["first_timestamp"],
        "last_timestamp": stats["last_timestamp"],
        "oracle_address": os.getenv("ORACLE_SIGNER_ADDRESS", ""),
    }


@app.get("/dkg/uals")
async def dkg_uals(limit: int = Query(default=10, le=100), since_timestamp: Optional[int] = None, regime: Optional[str] = None):
    """
    Get list of published DKG snapshot UALs for confirming nodes.

    **Cost: FREE**

    This endpoint is used by Graph Confirming Nodes to discover new snapshots
    that need verification.

    Args:
        limit: Maximum number of UALs to return (default 10, max 100)
        since_timestamp: Only return UALs published after this Unix timestamp
        regime: Filter by regime ('Risk-On' or 'Risk-Off')

    Returns:
        List of UAL entries with metadata
    """
    from src.macrocrypto.services.dkg_service import get_ual_store

    ual_store = get_ual_store()
    uals = ual_store.get_uals(limit=limit, since_timestamp=since_timestamp, regime=regime)

    return {
        "uals": uals,
        "count": len(uals),
        "oracle_address": os.getenv("ORACLE_SIGNER_ADDRESS", ""),
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shutdown background tasks."""
    if scheduler:
        try:
            scheduler.shutdown()
            print("[OK] Scheduler shut down gracefully")
        except Exception as e:
            print(f"[WARNING] Error shutting down scheduler: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
