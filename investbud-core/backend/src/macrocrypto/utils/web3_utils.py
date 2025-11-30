from web3 import Web3
from eth_account import Account
import json

w3 = Web3()

def create_snapshot_hash(data):
    """
    Create a keccak256 hash of your market data
    
    Args:
        data: dict or string of your market data
    
    Returns:
        bytes32 hash
    """
    # Convert to JSON string if dict
    if isinstance(data, dict):
        data_string = json.dumps(data, sort_keys=True)
    else:
        data_string = str(data)

    # Create keccak256 hash
    snapshot_hash = w3.keccak(text=data_string)

    return "0x" + snapshot_hash.hex()


def create_message_hash(risk_on, confidence, snapshot_hash, timestamp):
    """
    Create the message hash that matches Solidity's:
    keccak256(abi.encodePacked(riskOn, confidence, snapshotHash, timestamp))

    Args:
        risk_on: bool - True for risk-on, False for risk-off
        confidence: int - 0-100 confidence level
        snapshot_hash: bytes or hex string - The snapshot hash (32 bytes)
        timestamp: int - Unix timestamp

    Returns:
        bytes32 hash
    """
    # Convert snapshot_hash to bytes if it's a hex string
    if isinstance(snapshot_hash, str):
        snapshot_hash = bytes.fromhex(snapshot_hash.replace('0x', ''))

    # Pack data tightly (no padding) to match abi.encodePacked
    packed_data = b''.join([
        (1 if risk_on else 0).to_bytes(1, 'big'),  # bool as 1 byte
        confidence.to_bytes(1, 'big'),              # uint8 as 1 byte
        snapshot_hash,                               # bytes32 (32 bytes)
        timestamp.to_bytes(32, 'big')               # uint256 as 32 bytes
    ])

    # Hash the packed data
    message_hash = w3.keccak(packed_data)

    return message_hash


def sign_oracle_update(risk_on, confidence, snapshot_hash, timestamp, private_key):
    """
    Sign an oracle update with the trusted signer's private key.

    Args:
        risk_on: bool - True for risk-on, False for risk-off
        confidence: int - 0-100 confidence level
        snapshot_hash: bytes or hex string - The snapshot hash
        timestamp: int - Unix timestamp
        private_key: str - Private key of the trusted signer (with or without 0x prefix)

    Returns:
        dict with signature and all parameters ready for contract call
    """
    # Create the message hash
    message_hash = create_message_hash(risk_on, confidence, snapshot_hash, timestamp)

    # Load account from private key
    if not private_key.startswith('0x'):
        private_key = '0x' + private_key
    account = Account.from_key(private_key)

    # Sign the hash (using sign_message for raw hash)
    from eth_account.messages import encode_defunct
    # For raw hash, we need to use unsafe_hash
    signed_message = account.unsafe_sign_hash(message_hash)
    signature = signed_message.signature

    # Ensure snapshot_hash is hex string for return
    if isinstance(snapshot_hash, bytes):
        snapshot_hash = '0x' + snapshot_hash.hex()

    return {
        'riskOn': risk_on,
        'confidence': confidence,
        'snapshotHash': snapshot_hash,
        'timestamp': timestamp,
        'signature': '0x' + signature.hex() if isinstance(signature, bytes) else signature,
        'signer': account.address
    }


def create_and_sign_update(market_data, risk_on, confidence, timestamp, private_key):
    """
    Complete flow: Create snapshot hash from market data and sign the update.

    Args:
        market_data: dict - Market data that generated the signal
        risk_on: bool - True for risk-on, False for risk-off
        confidence: int - 0-100 confidence level
        timestamp: int - Unix timestamp
        private_key: str - Private key of the trusted signer

    Returns:
        dict with all data ready to submit to contract
    """
    # Step 1: Create snapshot hash from market data
    snapshot_hash = create_snapshot_hash(market_data)

    # Step 2: Sign the update
    signed_data = sign_oracle_update(risk_on, confidence, snapshot_hash, timestamp, private_key)

    # Step 3: Include original market data for storage/audit
    signed_data['market_data'] = market_data

    return signed_data

def get_signal(contract):
    """
    Gets the latest signal from the oracle contract.

    Args:
        contract: web3.contract.Contract - The oracle contract instance
    
    Returns:
        dict with signal data
    """
    try:

        signal = contract.functions.latest().call()

        (risk_on, confidence, snapshotHash, timestamp) = signal

        return {
            'risk_on': risk_on,
            'confidence': confidence,
            'snapshotHash': snapshotHash,
            'timestamp': timestamp
        }
    except Exception as e:
        print(f"Error getting signal: {e}")
        return {}

def update_signal(contract, w3, risk_on, confidence, snapshot_hash, timestamp, signature, private_key):
    """
    Submits an updateSignal transaction to the oracle contract.

    Args:
        contract: web3.contract.Contract - The oracle contract instance
        w3: Web3 - The web3 instance
        risk_on: bool - True for risk-on, False for risk-off
        confidence: int - 0-100 confidence level
        snapshot_hash: bytes or hex string - The snapshot hash
        timestamp: int - Unix timestamp
        signature: str - Signature from the trusted signer
        private_key: str - Private key of the sender account
    """
    # Prepare the transaction
    try:
        # Get account from private key
        if not private_key.startswith('0x'):
            private_key = '0x' + private_key
        account = Account.from_key(private_key)

        tx = contract.functions.updateSignal(
            risk_on,
            confidence,
            snapshot_hash,
            timestamp,
            signature
        ).build_transaction({
            'from': account.address,
            'nonce': w3.eth.get_transaction_count(account.address),
            'gasPrice': w3.eth.gas_price
        })

        try:
            gas_estimate = w3.eth.estimate_gas(tx)
            tx['gas'] = gas_estimate + 10000  # Add buffer
        except Exception as e:
            print(f"Gas estimation failed: {e}. Using default gas limit.")
            tx['gas'] = 200000  # Fallback gas limit

        # Sign the transaction
        signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)

        # Send the transaction
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)

        # Wait for receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

        return tx_hash

    except Exception as e:
        print(f"Error updating signal: {e}")
        return None

    