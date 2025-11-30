"""
Example: Using the /signal endpoint with fallback submission

This demonstrates the resilient design where:
1. Backend tries to submit on-chain
2. If it fails, client can submit using the returned signature
"""

import os
import json
from web3 import Web3
from dotenv import load_dotenv
from eth_account import Account
from x402.clients.requests import x402_requests

load_dotenv()

# Configuration
API_URL = "http://localhost:8001"
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
BASE_SEPOLIA_RPC_URL = os.getenv("BASE_SEPOLIA_RPC_URL")

if not PRIVATE_KEY or not BASE_SEPOLIA_RPC_URL:
    raise ValueError("Missing PRIVATE_KEY or BASE_SEPOLIA_RPC_URL in .env")

# Initialize account and session
account = Account.from_key(PRIVATE_KEY)
session = x402_requests(account)

print(f"\n{'='*70}")
print(f"Signal Endpoint with Fallback Submission Test")
print(f"{'='*70}")
print(f"Account: {account.address}")
print()


def submit_signal_with_fallback():
    """
    Call /signal endpoint and handle both success and failure cases
    """

    # Step 1: Call the API
    print("[1] Calling /signal endpoint...")
    print("-" * 70)

    response = session.post(
        f"{API_URL}/signal",
        json={"network": "base-sepolia"}
    )

    if response.status_code != 200:
        print(f"✗ API Error: {response.status_code}")
        print(response.text)
        return

    data = response.json()
    print(f"✓ API Response received")
    print(f"  Risk-On: {data['risk_on']}")
    print(f"  Confidence: {data['confidence']}%")
    print(f"  Backend Submitted: {data['backend_submitted']}")

    # Step 2: Check if backend submitted successfully
    if data['backend_submitted']:
        print(f"\n{'='*70}")
        print(f"✓ Backend submitted successfully!")
        print(f"{'='*70}")
        print(f"TX Hash: {data['transaction_hash']}")
        print(f"View: https://sepolia.basescan.org/tx/{data['transaction_hash']}")
        return

    # Step 3: Backend failed - submit ourselves
    print(f"\n{'='*70}")
    print(f"⚠ Backend submission failed: {data.get('error', 'Unknown error')}")
    print(f"{'='*70}")
    print(f"\n[2] Submitting transaction ourselves...")
    print("-" * 70)

    # Connect to blockchain
    w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC_URL))
    if not w3.is_connected():
        print("✗ Failed to connect to Base Sepolia")
        return

    # Load contract
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    oracle_abi_path = os.path.join(BASE_DIR, '..', 'contracts', 'out', 'MacroSignalOracle.sol', 'MacroSignalOracle.json')
    oracle_address_path = os.path.join(BASE_DIR, '..', 'contracts', 'deployments', 'MacroCryptoOracle.json')

    with open(oracle_abi_path) as f:
        abi = json.load(f)['abi']

    with open(oracle_address_path) as f:
        contract_address = Web3.to_checksum_address(json.load(f)["deployedTo"])

    contract = w3.eth.contract(address=contract_address, abi=abi)

    print(f"✓ Connected to Base Sepolia")
    print(f"✓ Contract: {contract_address}")

    # Build transaction using signed data from API
    print(f"\nBuilding transaction with signed data from API...")
    print(f"  Signature: {data['signature'][:20]}...")
    print(f"  Snapshot Hash: {data['snapshot_hash'][:20]}...")
    print(f"  Signer: {data['signer_address']}")

    tx = contract.functions.updateSignal(
        data['risk_on'],
        data['confidence'],
        data['snapshot_hash'],
        data['last_updated'],
        data['signature']
    ).build_transaction({
        'from': account.address,
        'gas': 200000,
        'gasPrice': w3.eth.gas_price,
        'nonce': w3.eth.get_transaction_count(account.address)
    })

    print(f"✓ Gas: {tx['gas']}")
    print(f"✓ Gas Price: {tx['gasPrice'] / 1e9:.2f} Gwei")

    # Sign and send
    print(f"\nSigning and sending transaction...")
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

    print(f"\n{'='*70}")
    print(f"✓ Transaction sent by client!")
    print(f"{'='*70}")
    print(f"TX Hash: {tx_hash.hex()}")
    print(f"View: https://sepolia.basescan.org/tx/{tx_hash.hex()}")

    # Wait for confirmation
    print(f"\nWaiting for confirmation...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    if receipt['status'] == 1:
        print(f"✓ Transaction SUCCESSFUL!")
        print(f"Block: {receipt['blockNumber']}")
        print(f"Gas Used: {receipt['gasUsed']}")
    else:
        print(f"✗ Transaction FAILED!")


if __name__ == "__main__":
    try:
        submit_signal_with_fallback()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
