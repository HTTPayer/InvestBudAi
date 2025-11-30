"""
Test script for DKG publish and get operations.

Usage:
    python tests/test_dkg.py                    # Use local node
    python tests/test_dkg.py --public           # Use public node
    python tests/test_dkg.py --node <endpoint>  # Use custom endpoint
"""
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from dkg import DKG
from dkg.providers import BlockchainProvider, NodeHTTPProvider

# Configuration
PUBLIC_NODE = "https://v6-pegasus-node-02.origin-trail.network:8900"
LOCAL_NODE = "http://localhost:8900"

# Parse args
if "--public" in sys.argv:
    NODE_ENDPOINT = PUBLIC_NODE
elif "--node" in sys.argv:
    idx = sys.argv.index("--node")
    NODE_ENDPOINT = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else LOCAL_NODE
else:
    NODE_ENDPOINT = os.getenv("DKG_NODE_ENDPOINT", LOCAL_NODE)

BLOCKCHAIN_ID = os.getenv("DKG_BLOCKCHAIN_ID", "otp:20430")

# Adjust timeouts based on node type
if "origin-trail.network" in NODE_ENDPOINT:
    MAX_RETRIES = 120  # 6 minutes for public node
    FREQUENCY = 3
    INITIAL_WAIT = 60
    RETRY_WAIT = 120
else:
    MAX_RETRIES = 60  # 3 minutes for local node
    FREQUENCY = 3
    INITIAL_WAIT = 10
    RETRY_WAIT = 30

print("=" * 60)
print("DKG SDK Test Script")
print("=" * 60)
print(f"Node endpoint: {NODE_ENDPOINT}")
print(f"Blockchain ID: {BLOCKCHAIN_ID}")
print(f"Max retries: {MAX_RETRIES}, Frequency: {FREQUENCY}s")
print(f"Initial wait: {INITIAL_WAIT}s, Retry wait: {RETRY_WAIT}s")
print()

# Initialize providers
print("[1] Initializing DKG client...")
node_provider = NodeHTTPProvider(
    endpoint_uri=NODE_ENDPOINT,
    api_version="v1"
)
blockchain_provider = BlockchainProvider(BLOCKCHAIN_ID)

dkg = DKG(
    node_provider,
    blockchain_provider,
    config={"max_number_of_retries": MAX_RETRIES, "frequency": FREQUENCY},
)
print("[OK] DKG client initialized")

# Check node info
print("\n[2] Checking node info...")
try:
    info = dkg.node.info
    print(f"[OK] Node version: {info}")
except Exception as e:
    print(f"[ERROR] Failed to get node info: {e}")

# Create a simple test asset
print("\n[3] Publishing test asset...")
test_content = {
    "public": {
        "@context": "https://schema.org",
        "@type": "Dataset",
        "name": "DKG Test Asset",
        "description": "Test asset for MacroCrypto DKG integration",
        "dateCreated": "2025-11-28T00:00:00Z",
        "test": True,
    }
}

try:
    result = dkg.asset.create(
        content=test_content,
        options={
            "epochs_num": 2,
            "minimum_number_of_finalization_confirmations": 1,
            "minimum_number_of_node_replications": 1,
        }
    )
    print(f"[OK] Published!")
    print(f"  UAL: {result.get('UAL')}")
    print(f"  Dataset Root: {result.get('datasetRoot')}")
    ual = result.get("UAL")
except Exception as e:
    print(f"[ERROR] Failed to publish: {e}")
    ual = None

# Wait a bit for propagation
if ual:
    print(f"\n[4] Waiting {INITIAL_WAIT} seconds for propagation...")
    time.sleep(INITIAL_WAIT)

    # Try to get the asset
    print("\n[5] Getting asset...")
    try:
        asset = dkg.asset.get(ual)
        print(f"[OK] Retrieved asset!")
        print(f"  Content: {asset}")
    except Exception as e:
        print(f"[ERROR] Failed to get asset: {e}")

        # Try with more wait time
        print(f"\n[6] Waiting {RETRY_WAIT} more seconds and retrying...")
        time.sleep(RETRY_WAIT)
        try:
            asset = dkg.asset.get(ual)
            print(f"[OK] Retrieved asset on retry!")
            print(f"  Content: {asset}")
        except Exception as e:
            print(f"[ERROR] Still failed to get asset: {e}")

            # Third attempt with even more wait
            print(f"\n[7] Final attempt - waiting {RETRY_WAIT} more seconds...")
            time.sleep(RETRY_WAIT)
            try:
                asset = dkg.asset.get(ual)
                print(f"[OK] Retrieved asset on final retry!")
                print(f"  Content: {asset}")
            except Exception as e:
                print(f"[ERROR] All retries failed: {e}")

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)
