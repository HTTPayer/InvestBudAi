from web3 import Web3
import json
from eth_account.messages import encode_defunct

# Initialize Web3
w3 = Web3()
# ====================================
# OPTION 1: Hash arbitrary data (e.g., JSON)
# ====================================
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
# Example usage:
market_data = {
    "btc_price": 45000,
    "eth_price": 2500,
    "fear_greed_index": 65,
    "timestamp": 1703001234,
    "sources": ["coinmarketcap", "coingecko"]
}

snapshot_hash = create_snapshot_hash(market_data)
print(f"Snapshot Hash: {snapshot_hash}")