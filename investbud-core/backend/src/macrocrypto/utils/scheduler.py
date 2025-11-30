"""
Background scheduler for daily data refresh.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from .cache import get_cache
import os

# Global DKG service instance (lazy loaded)
_dkg_service = None


def get_dkg_service():
    """Get or create DKG service instance."""
    global _dkg_service
    if _dkg_service is None:
        try:
            from ..services.dkg_service import DKGService
            node_endpoint = os.getenv("DKG_NODE_ENDPOINT", "http://localhost:9200")
            blockchain_env = os.getenv("DKG_BLOCKCHAIN_ENV", "testnet")
            blockchain_id = os.getenv("DKG_BLOCKCHAIN_ID", "otp:20430")

            _dkg_service = DKGService(
                node_endpoint=node_endpoint,
                blockchain_environment=blockchain_env,
                blockchain_id=blockchain_id,
            )
            print(f"[SCHEDULER] DKG service initialized: {node_endpoint}")
        except Exception as e:
            print(f"[SCHEDULER] DKG service not available: {e}")
            return None
    return _dkg_service


def publish_to_dkg(
    market_data: Dict[str, Any],
    signed_update: Dict[str, Any],
    tx_hash: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Publish regime snapshot to OriginTrail DKG.

    Args:
        market_data: Market data snapshot
        signed_update: Signed oracle update
        tx_hash: On-chain transaction hash

    Returns:
        DKG publish result or None if failed
    """
    dkg = get_dkg_service()
    if dkg is None:
        print("[SCHEDULER] DKG service not available, skipping publish")
        return None

    try:
        from ..services.dkg_service import RegimeSnapshot

        # Create RegimeSnapshot from signal data
        snapshot = RegimeSnapshot(
            regime=market_data.get("regime", "Unknown"),
            regime_binary=1 if signed_update.get("riskOn", False) else 0,
            confidence=signed_update.get("confidence", 0) / 100.0,
            risk_on_probability=market_data.get("risk_on_probability", 0),
            timestamp=signed_update.get("timestamp", 0),
            date=market_data.get("date", ""),
            snapshot_hash=signed_update.get("snapshotHash", ""),
            signature=signed_update.get("signature", ""),
            signer_address=signed_update.get("signer", ""),
            transaction_hash=tx_hash,
            features=market_data.get("features", {}),
        )

        # Publish to DKG
        result = dkg.publish_snapshot(snapshot)

        if result and "UAL" in result:
            print(f"[SCHEDULER] ✓ Published to DKG: {result['UAL']}")

            # Cache the UAL for reference
            cache = get_cache()
            cache.set("latest_dkg_ual", result["UAL"])

        return result

    except Exception as e:
        print(f"[SCHEDULER] ✗ DKG publish error: {e}")
        return None


def refresh_stablecoin_cache():
    """Refresh stablecoin symbols cache."""
    try:
        print(f"\n[SCHEDULER] Refreshing stablecoin cache at {datetime.now()}")

        # Import here to avoid circular dependencies
        from ..data.wallet_analyzer import coingecko_api

        cache = get_cache()

        # Fetch fresh stablecoin data
        stables = coingecko_api(coins='stablecoins', url="FDV", use_cached_data=False)
        # rwa = coingecko_api(coins='real-world-assets-rwa', url="FDV", use_cached_data=False)

        # Cache the results
        cache.set('stablecoins', stables)
        # cache.set('rwa_tokens', rwa)

        print(f"[SCHEDULER] ✓ Cached {len(stables)} stablecoins")

    except Exception as e:
        print(f"[SCHEDULER] ✗ Failed to refresh stablecoin cache: {e}")


def refresh_macro_data_cache():
    """Refresh macro and crypto data cache."""
    try:
        print(f"\n[SCHEDULER] Refreshing macro/crypto data at {datetime.now()}")

        # Import here to avoid circular dependencies
        from ..data import CombinedDataPipeline

        cache = get_cache()

        # Fetch fresh data (last 2 years)
        pipeline = CombinedDataPipeline()
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')

        df = pipeline.fetch_combined_data(start_date=start_date, end_date=end_date)

        # Cache the result
        cache.set('macro_crypto_data', df)

        print(f"[SCHEDULER] ✓ Cached macro/crypto data ({len(df)} rows)")

    except Exception as e:
        print(f"[SCHEDULER] ✗ Failed to refresh macro data cache: {e}")


def update_oracle_signal():
    """Update oracle signal with latest regime prediction."""
    try:
        print(f"\n[SCHEDULER] Updating oracle signal at {datetime.now()}")

        # Import here to avoid circular dependencies
        from ..models import MacroRegimeClassifier
        from .web3_utils import create_and_sign_update, update_signal, get_signal
        import os
        from web3 import Web3
        import json

        # Get environment variables
        BASE_SEPOLIA_RPC_URL = os.getenv("BASE_SEPOLIA_RPC_URL")
        PRIVATE_KEY = os.getenv("PRIVATE_KEY")

        if not BASE_SEPOLIA_RPC_URL or not PRIVATE_KEY:
            print("[SCHEDULER] ✗ Missing RPC URL or private key")
            return

        # Load classifier
        classifier = MacroRegimeClassifier()
        classifier.load('models/regime_classifier.pkl')

        # Predict current regime
        regime_result = classifier.predict_current_regime(verbose=False)

        # Connect to blockchain
        w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC_URL))

        # Load contract
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        oracle_abi_path = os.path.join(BASE_DIR, 'contracts', 'out', 'MacroSignalOracle.sol', 'MacroSignalOracle.json')
        oracle_address_path = os.path.join(BASE_DIR, 'contracts', 'deployments', 'MacroCryptoOracle.json')

        with open(oracle_abi_path) as f:
            abi = json.load(f)['abi']

        with open(oracle_address_path) as f:
            contract_address = Web3.to_checksum_address(json.load(f)["deployedTo"])

        contract = w3.eth.contract(address=contract_address, abi=abi)

        # Get current on-chain signal
        onchain_signal = get_signal(contract)
        risk_on_onchain = onchain_signal.get('risk_on')

        # Prepare update data
        risk_on = bool(regime_result['regime_binary'])
        confidence = int(regime_result['confidence'] * 100)
        timestamp = int(regime_result['date'].timestamp())

        print(f"[SCHEDULER] Current Regime: {'Risk-On' if risk_on else 'Risk-Off'} (Confidence: {confidence}%)")
        print(f"[SCHEDULER] On-chain Regime: {'Risk-On' if risk_on_onchain else 'Risk-Off'}")

        # Only update if signal changed
        if risk_on == risk_on_onchain:
            print("[SCHEDULER] ✓ On-chain signal is up-to-date. No update needed.")
            return

        # Create market data snapshot
        market_data = {
            'date': str(regime_result['date']),
            'regime': regime_result['regime'],
            'confidence': regime_result['confidence'],
            'risk_on_probability': regime_result['risk_on_probability'],
            'features': regime_result['features'],
            'timestamp': timestamp
        }

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
            snapshot_hash=signed_update['snapshotHash'],
            timestamp=timestamp,
            signature=signed_update['signature'],
            private_key=PRIVATE_KEY
        )

        if tx_hash:
            print(f"[SCHEDULER] ✓ Oracle signal updated successfully!")
            print(f"[SCHEDULER]   TX Hash: 0x{tx_hash.hex()}")
            print(f"[SCHEDULER]   View: https://sepolia.basescan.org/tx/0x{tx_hash.hex()}")

            # Publish to DKG (OriginTrail Knowledge Graph)
            try:
                publish_to_dkg(
                    market_data=market_data,
                    signed_update=signed_update,
                    tx_hash=f"0x{tx_hash.hex()}"
                )
            except Exception as dkg_error:
                print(f"[SCHEDULER] ⚠ DKG publish failed (non-blocking): {dkg_error}")
        else:
            print("[SCHEDULER] ✗ Failed to update oracle signal")

    except Exception as e:
        print(f"[SCHEDULER] ✗ Failed to update oracle signal: {e}")


def refresh_smart_money_analysis():
    """Refresh smart money analysis from Nansen + Heurist + LLM."""
    try:
        print(f"\n[SCHEDULER] Refreshing smart money analysis at {datetime.now()}")

        # Import here to avoid circular dependencies
        from eth_account import Account
        from x402.clients.requests import x402_requests
        import json
        import time

        # Get environment variables
        PRIVATE_KEY = os.getenv("CLIENT_PRIVATE_KEY")
        LLM_SERVER = os.getenv("LLM_SERVER", "https://api.httpayer.com/llm")
        SERVER_API_KEY = os.getenv("SERVER_API_KEY", "")
        RELAY_URL = os.getenv("RELAY_URL", "https://relay.httpayer.com")

        if not PRIVATE_KEY:
            print("[SCHEDULER] ✗ Missing CLIENT_PRIVATE_KEY")
            return

        # Initialize account and session
        account = Account.from_key(PRIVATE_KEY)
        session = x402_requests(account)

        # Helper function to retry X402 payment requests
        def retry_payment_request(session, url, max_retries=5, retry_delay=1, **kwargs):
            """
            Retry a session.post request if it gets 402 Payment Required.

            The first 402 triggers the payment, subsequent requests should succeed.
            """
            for attempt in range(max_retries):
                response = session.post(url, **kwargs)

                if response.status_code == 402:
                    if attempt < max_retries - 1:
                        print(f"[SCHEDULER] Got 402, retrying... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        print(f"[SCHEDULER] ✗ Failed after {max_retries} attempts (402 Payment Required)")
                        raise Exception(f"Payment failed after {max_retries} retries")

                # Success or other error
                return response

            return response

        # Step 1: Get Nansen data
        print("[SCHEDULER] Fetching Nansen Smart Money data...")
        nansen_response = retry_payment_request(
            session,
            RELAY_URL,
            json={
                "api_url": "https://nansen.api.corbits.dev/api/v1/smart-money/netflow",
                "method": "POST",
                "network": "base",
                "data": {
                    "chains": ["all"],
                    "filters": {
                        "include_native_tokens": True,
                        "include_smart_money_labels": ["Fund", "Smart Trader"],
                        "include_stablecoins": False
                    },
                    "pagination": {"page": 1, "per_page": 100},
                    "order_by": [{"field": "net_flow_24h_usd", "direction": "DESC"}]
                }
            },
            headers={"Content-Type": "application/json"}
        )
        nansen_data = nansen_response.json()
        print(f"[SCHEDULER] ✓ Nansen data received ({len(nansen_data.get('data', []))} tokens)")

        # Step 2: Get Heurist search
        print("[SCHEDULER] Searching crypto news with Heurist...")
        tokens = nansen_data.get('data', [])
        token_symbols = [t.get('token_symbol') for t in tokens if t.get('token_symbol')][:20]
        sectors = list(set([sector for t in tokens for sector in t.get('token_sectors', [])]))[:5]

        search_term = f"Recent cryptocurrency news and market analysis for tokens: {', '.join(token_symbols)}. Focus on {', '.join(sectors)} sectors."

        heurist_response = retry_payment_request(
            session,
            RELAY_URL,
            json={
                "api_url": "https://mesh.heurist.xyz/x402/agents/ExaSearchDigestAgent/exa_web_search",
                "method": "POST",
                "network": "base",
                "data": {"search_term": search_term, "limit": 5, "time_filter": "past_week"}
            },
            headers={"Content-Type": "application/json"}
        )
        heurist_data = heurist_response.json() if heurist_response.headers.get("content-type", "").startswith("application/json") else heurist_response.text
        print(f"[SCHEDULER] ✓ Heurist data received")

        # Step 3: Generate LLM summary
        print("[SCHEDULER] Generating analysis with LLM...")
        llm_response = retry_payment_request(
            session,
            f"{LLM_SERVER}/chat",
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a research analyst that analyzes cryptocurrency smart money flows and related news to generate a concise and insightful report for defi traders."
                    },
                    {
                        "role": "user",
                        "content": f"""Analyze the following data:

SMART MONEY TOKEN FLOWS (from Nansen):
{json.dumps(nansen_data, indent=2)}

RELATED NEWS & MARKET ANALYSIS (from Heurist):
{json.dumps(heurist_data, indent=2)}

Provide a concise summary that:
1. Identifies which tokens smart money is accumulating or selling
2. Explains potential reasons based on the news articles
3. Highlights key trends or opportunities

Note: Whenever you mention a token for the first time, include its token_address and chain in parentheses."""
                    }
                ]
            },
            headers={"Content-Type": "application/json", "x-api-key": SERVER_API_KEY} if SERVER_API_KEY else {"Content-Type": "application/json"}
        )
        summary_data = llm_response.json() if llm_response.headers.get("content-type", "").startswith("application/json") else llm_response.text
        print(f"[SCHEDULER] ✓ LLM analysis generated")

        # Extract text from response
        def extract_text(obj):
            if isinstance(obj, str):
                try:
                    parsed = json.loads(obj)
                    return extract_text(parsed)
                except:
                    return obj
            if isinstance(obj, dict):
                for key in ['response', 'text', 'content', 'message']:
                    if key in obj:
                        return extract_text(obj[key])
            return json.dumps(obj, indent=2)

        analysis_text = extract_text(summary_data.get('original') or summary_data.get('response') or summary_data)

        # Cache the results
        cache = get_cache()
        cache.set('smart_money_analysis', {
            'generated_at': datetime.now().isoformat(),
            'analysis': analysis_text,
            'nansen_data': nansen_data,
            'heurist_data': heurist_data,
            'raw_summary': summary_data
        })

        print(f"[SCHEDULER] ✓ Smart money analysis cached ({len(analysis_text)} chars)")

    except Exception as e:
        print(f"[SCHEDULER] ✗ Failed to refresh smart money analysis: {e}")
        import traceback
        traceback.print_exc()


def record_live_agent_decision():
    """
    Record daily agent portfolio decision and track performance over time.

    This creates a growing timeseries of:
    - Regime predictions
    - Portfolio allocations
    - Simulated portfolio value
    """
    try:
        print(f"\n[SCHEDULER] Recording live agent decision at {datetime.now()}")

        import pickle
        import pandas as pd
        import numpy as np

        # Import classifier
        from ..models import MacroRegimeClassifier

        # Load or create live decisions file
        decisions_file = 'models/live_agent_decisions.pkl'

        if os.path.exists(decisions_file):
            with open(decisions_file, 'rb') as f:
                live_data = pickle.load(f)
        else:
            live_data = {
                'decisions': [],
                'portfolio_value': 10000,  # Start with $10k
                'btc_holdings': 0,
                'cash_holdings': 10000,
                'started_at': datetime.now().isoformat()
            }

        # Load classifier and predict
        classifier = MacroRegimeClassifier()
        classifier.load('models/regime_classifier.pkl')
        regime_result = classifier.predict_current_regime(verbose=False)

        risk_on = bool(regime_result['regime_binary'])
        confidence = regime_result['confidence']
        btc_price = regime_result['features'].get('btc_price', 0)

        # Get previous state
        prev_btc_holdings = live_data.get('btc_holdings', 0)
        prev_cash = live_data.get('cash_holdings', 10000)
        prev_risk_on = live_data['decisions'][-1]['risk_on'] if live_data['decisions'] else None

        # Calculate current portfolio value before rebalancing
        current_portfolio_value = prev_cash + (prev_btc_holdings * btc_price)

        # Determine allocation based on regime
        # Risk-On: 100% BTC, Risk-Off: 100% Cash
        if risk_on:
            # Allocate to BTC
            target_btc_allocation = 1.0
            target_cash_allocation = 0.0
        else:
            # Stay in cash
            target_btc_allocation = 0.0
            target_cash_allocation = 1.0

        # Apply simple fee model if rebalancing
        rebalanced = (prev_risk_on is not None and risk_on != prev_risk_on)
        fee = 0
        if rebalanced:
            fee = current_portfolio_value * 0.01  # 1% total fee for simplicity
            current_portfolio_value -= fee

        # Calculate new holdings
        new_btc_holdings = (current_portfolio_value * target_btc_allocation) / btc_price if btc_price > 0 else 0
        new_cash_holdings = current_portfolio_value * target_cash_allocation

        # Record decision
        decision = {
            'timestamp': datetime.now().isoformat(),
            'date': str(regime_result['date']),
            'risk_on': risk_on,
            'confidence': confidence,
            'btc_price': btc_price,
            'btc_allocation': target_btc_allocation,
            'cash_allocation': target_cash_allocation,
            'btc_holdings': new_btc_holdings,
            'cash_holdings': new_cash_holdings,
            'portfolio_value': current_portfolio_value,
            'rebalanced': rebalanced,
            'fee_paid': fee,
            'features': regime_result['features']
        }

        live_data['decisions'].append(decision)
        live_data['btc_holdings'] = new_btc_holdings
        live_data['cash_holdings'] = new_cash_holdings
        live_data['portfolio_value'] = current_portfolio_value
        live_data['last_updated'] = datetime.now().isoformat()

        # Calculate cumulative metrics if we have enough data
        if len(live_data['decisions']) >= 2:
            values = [d['portfolio_value'] for d in live_data['decisions']]
            values_series = pd.Series(values)
            returns = values_series.pct_change().dropna()

            if len(returns) > 0:
                # Calculate metrics
                total_return = (values[-1] / values[0]) - 1
                volatility = returns.std() * np.sqrt(365)
                sharpe = (returns.mean() * 365) / (volatility) if volatility > 0 else 0

                # Max drawdown
                cummax = values_series.cummax()
                drawdown = (values_series - cummax) / cummax
                max_dd = drawdown.min()

                live_data['metrics'] = {
                    'total_return': float(total_return),
                    'volatility': float(volatility),
                    'sharpe_ratio': float(sharpe),
                    'max_drawdown': float(max_dd),
                    'num_trades': sum(1 for d in live_data['decisions'] if d.get('rebalanced')),
                    'total_fees': sum(d.get('fee_paid', 0) for d in live_data['decisions']),
                    'days_tracked': len(live_data['decisions'])
                }

        # Save
        os.makedirs('models', exist_ok=True)
        with open(decisions_file, 'wb') as f:
            pickle.dump(live_data, f)

        regime_str = 'Risk-On (BTC)' if risk_on else 'Risk-Off (Cash)'
        print(f"[SCHEDULER] ✓ Agent decision recorded: {regime_str}")
        print(f"[SCHEDULER]   Portfolio Value: ${current_portfolio_value:,.2f}")
        print(f"[SCHEDULER]   BTC: {new_btc_holdings:.6f} | Cash: ${new_cash_holdings:,.2f}")
        if rebalanced:
            print(f"[SCHEDULER]   Rebalanced! Fee: ${fee:,.2f}")

    except Exception as e:
        print(f"[SCHEDULER] ✗ Failed to record agent decision: {e}")
        import traceback
        traceback.print_exc()


def start_scheduler():
    """Start background scheduler for daily cache refresh."""
    scheduler = BackgroundScheduler()

    # Refresh stablecoins daily at 2 AM UTC
    scheduler.add_job(
        refresh_stablecoin_cache,
        CronTrigger(hour=2, minute=0),
        id='refresh_stablecoins',
        name='Refresh stablecoin cache',
        replace_existing=True
    )

    # Refresh macro data daily at 3 AM UTC (after market close)
    scheduler.add_job(
        refresh_macro_data_cache,
        CronTrigger(hour=3, minute=0),
        id='refresh_macro_data',
        name='Refresh macro/crypto data',
        replace_existing=True
    )

    # Update oracle signal daily at 4 AM UTC (after fresh data is cached)
    scheduler.add_job(
        update_oracle_signal,
        CronTrigger(hour=4, minute=0),
        id='update_oracle_signal',
        name='Update oracle signal on-chain',
        replace_existing=True
    )

    # Refresh smart money analysis every 6 hours
    scheduler.add_job(
        refresh_smart_money_analysis,
        CronTrigger(hour='*/6', minute=0),  # Every 6 hours: 0, 6, 12, 18
        id='refresh_smart_money',
        name='Refresh smart money analysis',
        replace_existing=True
    )

    # Record live agent decision daily at 5 AM UTC (after oracle update)
    scheduler.add_job(
        record_live_agent_decision,
        CronTrigger(hour=5, minute=0),
        id='record_agent_decision',
        name='Record live agent portfolio decision',
        replace_existing=True
    )

    # Run immediately on startup if cache is empty
    cache = get_cache()
    if cache.get('stablecoins', ttl_seconds=86400) is None:
        print("[SCHEDULER] Initial stablecoin cache refresh...")
        refresh_stablecoin_cache()

    if cache.get('macro_crypto_data', ttl_seconds=86400) is None:
        print("[SCHEDULER] Initial macro data cache refresh...")
        refresh_macro_data_cache()

    if cache.get('smart_money_analysis', ttl_seconds=21600) is None:  # 6 hours
        print("[SCHEDULER] Initial smart money analysis refresh...")
        refresh_smart_money_analysis()

    # Record initial agent decision if file doesn't exist
    if not os.path.exists('models/live_agent_decisions.pkl'):
        print("[SCHEDULER] Recording initial agent decision...")
        record_live_agent_decision()

    scheduler.start()
    print("[SCHEDULER] ✓ Background scheduler started")
    print("[SCHEDULER]   - Stablecoins refresh: Daily at 02:00 UTC")
    print("[SCHEDULER]   - Macro data refresh: Daily at 03:00 UTC")
    print("[SCHEDULER]   - Oracle signal update: Daily at 04:00 UTC")
    print("[SCHEDULER]   - Live agent decision: Daily at 05:00 UTC")
    print("[SCHEDULER]   - Smart money analysis: Every 6 hours (00:00, 06:00, 12:00, 18:00 UTC)")

    return scheduler
