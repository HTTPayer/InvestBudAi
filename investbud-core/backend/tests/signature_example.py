"""
Example: How to create and sign oracle updates for MacroSignalOracle contract
"""

import os
from dotenv import load_dotenv
from macrocrypto.web3_utils import create_and_sign_update, sign_oracle_update, create_snapshot_hash
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


# =============================================================================
# EXAMPLE 2: Step-by-Step (More Control)
# =============================================================================

print("=== EXAMPLE 2: Step-by-Step ===")

# Step 1: Create snapshot hash
snapshot_hash = create_snapshot_hash(market_data)
print(f"1. Snapshot Hash: {snapshot_hash}")

# Step 2: Sign the update
signed_data = sign_oracle_update(
    risk_on=risk_on,
    confidence=confidence,
    snapshot_hash=snapshot_hash,
    timestamp=timestamp,
    private_key=PRIVATE_KEY
)

print(f"2. Signature: {signed_data['signature']}")
print(f"3. Ready to submit to contract!")
print()


# =============================================================================
# EXAMPLE 3: Calling the Smart Contract
# =============================================================================

print("=== EXAMPLE 3: Contract Call Example ===")
print("""
from web3 import Web3

# Connect to your blockchain
w3 = Web3(Web3.HTTPProvider('https://rpc.ankr.com/base'))

# Load contract
with open('MacroSignalOracle.abi.json') as f:
    abi = json.load(f)

contract_address = '0xYourContractAddress'
contract = w3.eth.contract(address=contract_address, abi=abi)

# Get signed data
signed_data = create_and_sign_update(market_data, risk_on, confidence, timestamp, private_key)

# Submit transaction (from any account - doesn't need to be the signer!)
tx = contract.functions.updateSignal(
    signed_data['riskOn'],
    signed_data['confidence'],
    signed_data['snapshotHash'],
    signed_data['timestamp'],
    signed_data['signature']
).build_transaction({
    'from': w3.eth.accounts[0],  # Any account with gas
    'gas': 200000,
    'gasPrice': w3.eth.gas_price,
    'nonce': w3.eth.get_transaction_count(w3.eth.accounts[0])
})

# Sign and send
signed_tx = w3.eth.account.sign_transaction(tx, private_key=your_hot_wallet_key)
tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
print(f'Transaction sent: {tx_hash.hex()}')
""")


# =============================================================================
# EXAMPLE 4: Verify Signature Locally (Before Sending)
# =============================================================================

print("\n=== EXAMPLE 4: Verify Before Sending ===")
from eth_account import Account
from web3_utils import create_message_hash

message_hash = create_message_hash(
    result['riskOn'],
    result['confidence'],
    result['snapshotHash'],
    result['timestamp']
)

# Recover the signer from signature
recovered_address = Account.recover_message(
    message_hash,
    signature=result['signature']
)

print(f"Expected Signer: {result['signer']}")
print(f"Recovered Signer: {recovered_address}")
print(f"Match: {recovered_address.lower() == result['signer'].lower()}")
