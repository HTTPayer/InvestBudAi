from web3 import Web3
import json
import os
from dotenv import load_dotenv
from macrocrypto.utils.web3_utils import create_and_sign_update
from macrocrypto.models.regime_classifier import MacroRegimeClassifier
import time

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"[MacroCrypto] Base Directory: {BASE_DIR}")

# Load environment variables
BASE_SEPOLIA_RPC_URL = os.getenv("BASE_SEPOLIA_RPC_URL")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")

if not BASE_SEPOLIA_RPC_URL or not PRIVATE_KEY:
    raise ValueError("Missing BASE_SEPOLIA_RPC_URL or PRIVATE_KEY in .env file")

print("\n" + "="*70)
print("STEP 1: Load Regime Classifier and Predict Current Regime")
print("="*70)

# Load trained model
model_path = os.path.join(BASE_DIR, 'models', 'regime_classifier.pkl')
if not os.path.exists(model_path):
    raise ValueError(f"Model not found at {model_path}. Please train the model first.")

classifier = MacroRegimeClassifier()
classifier.load(model_path)

# Get current regime prediction
regime_result = classifier.predict_current_regime(verbose=True)

print("\n" + "="*70)
print("STEP 2: Prepare Oracle Update Data")
print("="*70)

# Convert regime data to oracle format
risk_on = bool(regime_result['regime_binary'])  # True for Risk-On, False for Risk-Off
confidence = int(regime_result['confidence'] * 100)  # Convert 0-1 to 0-100
timestamp = int(time.time())

# Market data for snapshot hash (includes features used for prediction)
market_data = {
    'date': str(regime_result['date']),
    'regime': regime_result['regime'],
    'confidence': regime_result['confidence'],
    'risk_on_probability': regime_result['risk_on_probability'],
    'features': regime_result['features'],
    'timestamp': timestamp
}

print(f"\nOracle Update Parameters:")
print(f"  Risk-On: {risk_on}")
print(f"  Confidence: {confidence}%")
print(f"  Timestamp: {timestamp}")

# Create and sign the update
print("\nSigning data...")
signed_data = create_and_sign_update(
    market_data=market_data,
    risk_on=risk_on,
    confidence=confidence,
    timestamp=timestamp,
    private_key=PRIVATE_KEY
)

print(f"\n✓ Snapshot Hash: {signed_data['snapshotHash']}")
print(f"✓ Signature: {signed_data['signature']}")
print(f"✓ Signer: {signed_data['signer']}")

print("\n" + "="*70)
print("STEP 3: Connect to Blockchain and Submit Transaction")
print("="*70)

# Connect to blockchain
w3 = Web3(Web3.HTTPProvider(BASE_SEPOLIA_RPC_URL))
if not w3.is_connected():
    raise ConnectionError("Failed to connect to Base Sepolia")

account = w3.eth.account.from_key(PRIVATE_KEY)
w3.eth.default_account = account.address

print(f"\n✓ Connected to Base Sepolia")
print(f"✓ Account: {account.address}")
print(f"✓ Balance: {w3.eth.get_balance(account.address) / 1e18:.6f} ETH")

# Load contract ABI and address
oracle_abi_path = os.path.join(BASE_DIR, '..', 'contracts', 'out', 'MacroSignalOracle.sol', 'MacroSignalOracle.json')
oracle_address_path = os.path.join(BASE_DIR, '..', 'contracts', 'deployments', 'MacroCryptoOracle.json')

with open(oracle_abi_path) as f:
    abi = json.load(f)['abi']

with open(oracle_address_path) as f:
    contract_address = Web3.to_checksum_address(json.load(f)["deployedTo"])

contract = w3.eth.contract(address=contract_address, abi=abi)

def get_signal(contract):

    signal = contract.functions.latest().call()

    (risk_on, confidence, snapshotHash, timestamp) = signal

    return {
        'risk_on': risk_on,
        'confidence': confidence,
        'snapshotHash': snapshotHash,
        'timestamp': timestamp
    }

signal = get_signal(contract)

print(f'signal before update: risk_on={signal["risk_on"]}, confidence={signal["confidence"]}, snapshotHash={signal["snapshotHash"].hex()}, timestamp={signal["timestamp"]}')
