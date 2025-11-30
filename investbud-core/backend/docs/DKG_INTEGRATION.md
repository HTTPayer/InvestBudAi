# InvestBud AI DKG Integration

OriginTrail Decentralized Knowledge Graph integration for verifiable, decentralized macro regime classifications.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         InvestBud AI DKG Integration                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │  Primary Oracle  │───▶│   OriginTrail    │◀───│ Confirming Nodes │      │
│  │  (InvestBud AI)   │    │       DKG        │    │   (Verifiers)    │      │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘      │
│           │                       │                       │                 │
│           ▼                       ▼                       ▼                 │
│     Publishes             Stores Knowledge          Verify & Extend        │
│    Regime Snapshot          Assets                   the Graph             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Primary Oracle** (InvestBud AI) generates regime predictions daily
2. Predictions are published as **Knowledge Assets** to the DKG
3. **Confirming Nodes** independently verify predictions
4. Verifications are also published, creating a trust network
5. All data is queryable via SPARQL on the decentralized graph

## Files Created

### 1. `backend/src/macrocrypto/services/dkg_service.py`

| Component | Description |
|-----------|-------------|
| `RegimeSnapshot` | Dataclass for regime data |
| `DKGService` | Sync client for publishing and querying DKG |
| `AsyncDKGService` | Async client for FastAPI integration |
| `MACROCRYPTO_CONTEXT` | JSON-LD schema/ontology |
| `create_snapshot_from_signal()` | Convert `/signal` response to snapshot |

### 2. `backend/src/macrocrypto/services/dkg_confirming_node.py`

| Component | Description |
|-----------|-------------|
| `GraphConfirmingNode` | Independent verification node |
| `AsyncGraphConfirmingNode` | Async version for FastAPI |
| `VerificationReport` | Detailed comparison results |
| `VerificationResult` | Enum: `confirmed`, `challenged`, `partial`, `error` |
| `run_confirming_node()` | Standalone runner for verification loop |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dkg/info` | GET | Check DKG node connection status |
| `/dkg/publish` | POST | Publish current regime snapshot to DKG |
| `/dkg/query` | POST | Query historical snapshots from DKG |
| `/dkg/snapshot/{ual}` | GET | Get specific snapshot by UAL |
| `/dkg/verify` | POST | Verify a snapshot (act as confirming node) |
| `/dkg/latest` | GET | Get latest published snapshot UAL |

### Example: Publish a Snapshot

```bash
curl -X POST http://localhost:8001/dkg/publish \
  -H "Content-Type: application/json" \
  -d '{"epochs_num": 2, "min_confirmations": 3, "min_replications": 1}'
```

### Example: Query Snapshots

```bash
curl -X POST http://localhost:8001/dkg/query \
  -H "Content-Type: application/json" \
  -d '{"regime": "Risk-Off", "min_confidence": 0.8, "limit": 10}'
```

### Example: Verify a Snapshot

```bash
curl -X POST http://localhost:8001/dkg/verify \
  -H "Content-Type: application/json" \
  -d '{"ual": "did:dkg:base:84532/0x.../571"}'
```

## Scheduler Integration

After each oracle signal update (04:00 UTC daily), the snapshot is automatically published to DKG:

```python
# In scheduler.py update_oracle_signal():
if tx_hash:
    publish_to_dkg(market_data, signed_update, tx_hash)
```

The DKG publish is non-blocking - if it fails, the on-chain update still succeeds.

## JSON-LD Knowledge Asset Schema

### InvestBud AI Ontology

```json
{
  "@context": {
    "mc": "https://macrocrypto.io/ontology#",
    "regime": "mc:regime",
    "confidence": "mc:confidence",
    "riskOnProbability": "mc:riskOnProbability",
    "btcPrice": "mc:btcPrice",
    "snapshotHash": "mc:snapshotHash",
    "signature": "mc:signature",
    "previousSnapshot": "mc:previousSnapshot"
  }
}
```

### Example Knowledge Asset

```json
{
  "@context": { "mc": "https://macrocrypto.io/ontology#" },
  "@id": "https://macrocrypto.io/snapshots/1732752345",
  "@type": ["mc:RegimeSnapshot", "Dataset"],
  "name": "InvestBud AI Regime Snapshot 2025-11-27",
  "regime": "Risk-Off",
  "confidence": 1.0,
  "riskOnProbability": 0.0,
  "btcPrice": 96543.12,
  "btcRsi": 45.2,
  "vix": 18.5,
  "fedFunds": 4.33,
  "snapshotHash": "0x1234...",
  "signature": "0xabcd...",
  "signerAddress": "0x5678...",
  "previousSnapshot": "did:dkg:base:84532/0x.../571"
}
```

## Environment Variables

```bash
# DKG Node Configuration
DKG_NODE_ENDPOINT=http://localhost:9200
DKG_BLOCKCHAIN_ENV=testnet
DKG_BLOCKCHAIN_ID=base:84532

# Verifier Configuration (for confirming nodes)
VERIFIER_ADDRESS=0x...

# Existing variables still needed
PRIVATE_KEY=0x...
ALCHEMY_API_KEY=...
```

## Running a Confirming Node

### As Python Module

```python
from macrocrypto.services.dkg_confirming_node import run_confirming_node

run_confirming_node(
    node_endpoint="http://localhost:9200",
    verifier_address="0x...",
    poll_interval=3600  # Check every hour
)
```

### As CLI

```bash
python -m macrocrypto.services.dkg_confirming_node
```

### As Separate Service

Create a `confirming_node.py`:

```python
import os
from dotenv import load_dotenv
from macrocrypto.services.dkg_confirming_node import run_confirming_node

load_dotenv()

if __name__ == "__main__":
    run_confirming_node(
        node_endpoint=os.getenv("DKG_NODE_ENDPOINT", "http://localhost:9200"),
        verifier_address=os.getenv("VERIFIER_ADDRESS"),
        poll_interval=3600,
    )
```

## Verification Flow

```
┌─────────────────┐
│ Fetch Snapshot  │  Query DKG for unverified snapshots
└────────┬────────┘
         ▼
┌─────────────────┐
│ Load Features   │  Get BTC price, VIX, Fed Funds, etc.
└────────┬────────┘
         ▼
┌─────────────────┐
│ Run Classifier  │  Independent regime prediction
└────────┬────────┘
         ▼
┌─────────────────┐
│    Compare      │  Check regime match, confidence diff
└────────┬────────┘
         ▼
┌─────────────────┐
│ Publish Report  │  CONFIRMED / CHALLENGED / PARTIAL
└─────────────────┘
```

## SPARQL Queries

Query the DKG directly with SPARQL:

```sparql
PREFIX mc: <https://macrocrypto.io/ontology#>

SELECT ?snapshot ?regime ?confidence ?timestamp ?btcPrice
WHERE {
    ?snapshot a mc:RegimeSnapshot .
    ?snapshot mc:regime ?regime .
    ?snapshot mc:confidence ?confidence .
    ?snapshot mc:snapshotTimestamp ?timestamp .
    ?snapshot mc:btcPrice ?btcPrice .
    FILTER(?confidence >= 0.9)
}
ORDER BY DESC(?timestamp)
LIMIT 10
```

## Next Steps for Hackathon

1. **Install DKG SDK**:
   ```bash
   pip install dkg
   ```

2. **Test connection**:
   ```bash
   curl http://localhost:8001/dkg/info
   ```

3. **Publish first snapshot**:
   ```bash
   curl -X POST http://localhost:8001/dkg/publish
   ```

4. **Verify it**:
   ```bash
   curl -X POST http://localhost:8001/dkg/verify \
     -d '{"ual": "<UAL from step 3>"}'
   ```

5. **Run a confirming node** on a separate machine to demonstrate distributed verification

## References

- [OriginTrail DKG Python SDK](https://docs.origintrail.io/build-a-dkg-node-ai-agent/advanced-features-and-toolkits/dkg-sdk/dkg-v8-py-client)
- [Knowledge Assets](https://docs.origintrail.io/general/dkg-intro/knowledge-assets)
- [JSON-LD Specification](https://json-ld.org/)
