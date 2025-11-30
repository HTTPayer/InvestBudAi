"""
Example: How to create and sign oracle updates for MacroSignalOracle contract
"""

import os
from dotenv import load_dotenv
from macrocrypto.utils.web3_utils import create_and_sign_update, sign_oracle_update, create_snapshot_hash
import time

load_dotenv()

# =============================================================================
# EXAMPLE 1: Complete Flow (Recommended)
# =============================================================================

# Your trusted signer's private key (keep this secure!)
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

# Market data that generated your signal
market_data = {
    "btc_price": 45000,
    "eth_price": 2500,
    "fear_greed_index": 65,
    "moving_averages": {
        "btc_50ma": 43000,
        "eth_50ma": 2400
    },
    "timestamp": int(time.time()),
    "sources": ["coinmarketcap", "coingecko", "fear_greed_api"]
}

# Your signal analysis results
risk_on = True      # Market is risk-on
confidence = 85     # 85% confidence
timestamp = int(time.time())

# Create snapshot hash and sign in one step
result = create_and_sign_update(
    market_data=market_data,
    risk_on=risk_on,
    confidence=confidence,
    timestamp=timestamp,
    private_key=PRIVATE_KEY
)

print("=== EXAMPLE 1: Complete Flow ===")
print(f"Risk On: {result['riskOn']}")
print(f"Confidence: {result['confidence']}")
print(f"Snapshot Hash: {result['snapshotHash']}")
print(f"Timestamp: {result['timestamp']}")
print(f"Signature: {result['signature']}")
print(f"Signer Address: {result['signer']}")
print()