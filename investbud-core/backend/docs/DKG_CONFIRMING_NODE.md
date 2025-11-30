# DKG Graph Confirming Node Architecture

## Overview

The Graph Confirming Node is an independent verifier that validates InvestBud AI oracle predictions by:

1. Polling the DKG for new regime snapshots
2. Independently computing the same prediction using the same ML model
3. Publishing a confirmation or challenge to the DKG

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DKG NETWORK (NeuroWeb)                              │
│                                                                               │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│   │   Node 1    │◄──►│   Node 2    │◄──►│   Node 3    │◄──►│   Node N    │   │
│   │  (Stores)   │    │  (Stores)   │    │  (Stores)   │    │  (Stores)   │   │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │
│          ▲                  ▲                  ▲                  ▲          │
│          │                  │                  │                  │          │
│          └──────────────────┴─────────┬────────┴──────────────────┘          │
│                                       │                                       │
│                            ┌──────────▼──────────┐                           │
│                            │   Blazegraph (RDF)  │                           │
│                            │   SPARQL Endpoint   │                           │
│                            └──────────┬──────────┘                           │
└───────────────────────────────────────┼───────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
            ┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
            │  InvestBud AI │   │  Confirming   │   │  Confirming   │
            │    Oracle     │   │    Node A     │   │    Node B     │
            │  (Publisher)  │   │  (Verifier)   │   │  (Verifier)   │
            └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                    │                   │                   │
                    ▼                   ▼                   ▼
              PUBLISH             QUERY & VERIFY      QUERY & VERIFY
```

## Verification Results

The confirming node produces one of five results:

| Result       | Description                                    |
| ------------ | ---------------------------------------------- |
| `CONFIRMED`  | Regime matches and confidence difference ≤ 10% |
| `CHALLENGED` | Regime prediction differs                      |
| `PARTIAL`    | Regime matches but confidence differs > 10%    |
| `STALE_DATA` | Unable to fetch current market data            |
| `ERROR`      | Verification failed due to an error            |

## The Trust Problem

### Q: Couldn't anyone publish with `mc:RegimeSnapshot` type?

**Yes, they can.** The DKG is permissionless - anyone can publish any JSON-LD with any type.

This is NOT a bug, it's a feature. The trust model works differently:

### How Trust is Established

#### 1. Cryptographic Signatures

Every InvestBud AI snapshot includes a cryptographic signature:

```json
{
  "@type": "mc:RegimeSnapshot",
  "regime": "Risk-Off",
  "confidence": 0.85,
  "snapshotHash": "0x1234...",
  "signature": "0xabcd...",
  "signerAddress": "0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF"
}
```

The `signerAddress` is the **known InvestBud AI oracle address**. Anyone can verify:

- The signature is valid for the snapshot hash
- The signer matches the official InvestBud AI address

**Fake snapshots would have a different signer address.**

#### 2. Multi-Party Verification

Confirming nodes independently verify predictions:

```
Snapshot Published: "Risk-Off, 85% confidence"
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   Node A runs          Node B runs          Node C runs
   same model           same model           same model
        │                   │                   │
        ▼                   ▼                   ▼
   "Risk-Off, 84%"     "Risk-Off, 86%"     "Risk-Off, 83%"
        │                   │                   │
        ▼                   ▼                   ▼
   CONFIRMED            CONFIRMED            CONFIRMED
```

A fake snapshot with wrong predictions would be **CHALLENGED** by all verifiers.

#### 3. Query by Signer Address

To get ONLY official InvestBud AI snapshots:

```sparql
PREFIX mc: <https://macrocrypto.io/ontology#>

SELECT ?snapshot ?regime ?confidence
WHERE {
    ?snapshot a mc:RegimeSnapshot .
    ?snapshot mc:signerAddress "0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF" .
    ?snapshot mc:regime ?regime .
    ?snapshot mc:confidence ?confidence .
}
ORDER BY DESC(?timestamp)
```

#### 4. On-Chain Anchoring

The snapshot is anchored on NeuroWeb blockchain:

- UAL contains the contract address
- Transaction hash proves when it was published
- Immutable once published

## Running a Confirming Node

### Environment Variables

```bash
# Required
export VERIFIER_ADDRESS=0xYourAddress              # Your verifier Ethereum address
export ORACLE_SIGNER_ADDRESS=0x85bADB...           # Official oracle signer to verify

# Optional
export DKG_NODE_ENDPOINT=http://localhost:8900     # DKG node endpoint
export ORACLE_API_URL=http://localhost:8015        # InvestBud AI API URL
export POLL_INTERVAL=3600                          # Poll interval in seconds (default: 1 hour)
export MODEL_PATH=models/regime_classifier.pkl     # Path to trained model
```

### Continuous Verification Loop

```bash
# Run the confirming node (polls every hour by default)
python -m macrocrypto.services.dkg_confirming_node
```

Output:

```
============================================================
InvestBud AI Graph Confirming Node
============================================================
Oracle API: http://localhost:8015
DKG Node: http://localhost:8900
Verifier: 0xYourAddress
Oracle Signer: 0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF
Poll Interval: 3600s
============================================================
[OK] Starting verification loop
    Oracle API: http://localhost:8015
    Poll interval: 3600s
[...] Found 2 snapshot(s) to verify
[...] Verifying: did:dkg:otp:20430/0xcdb.../405801
[OK] Result: CONFIRMED
    Regime match: True
    Confidence diff: 0.0100
```

### Single Snapshot Verification

Verify a specific snapshot without running the loop:

```bash
# Verify a specific UAL
python -m macrocrypto.services.dkg_confirming_node --verify-ual did:dkg:otp:20430/0xcdb.../405801

# Verify and publish the report to DKG
python -m macrocrypto.services.dkg_confirming_node --verify-ual did:dkg:otp:20430/0xcdb.../405801 --publish
```

### Programmatic Usage

```python
from macrocrypto.services.dkg_confirming_node import (
    GraphConfirmingNode,
    VerificationResult,
    VerificationReport,
)
from macrocrypto.services.dkg_service import DKGService

# Initialize
dkg = DKGService(node_endpoint="http://localhost:8900")
node = GraphConfirmingNode(
    dkg_service=dkg,
    verifier_address="0xYourAddress",
    confidence_threshold=0.1,   # 10% tolerance (default)
    feature_threshold=0.05,     # 5% feature tolerance (default)
)

# Verify a specific snapshot
report = node.verify_snapshot("did:dkg:otp:20430/0xcdb.../405801")
print(f"Result: {report.verification_result.value}")
print(f"Regime match: {report.regime_matches}")
print(f"Confidence diff: {report.confidence_difference:.4f}")
print(f"Computed regime: {report.computed_regime}")
print(f"Computed confidence: {report.computed_confidence}")

# Publish verification report to DKG
if report.verification_result != VerificationResult.ERROR:
    result = node.publish_verification(report)
    print(f"Published: {result.get('UAL')}")

# Or verify and publish in one call
report, publish_result = node.verify_and_publish("did:dkg:otp:20430/0xcdb.../405801")

# Run continuous verification loop
node.run_verification_loop(
    oracle_api_url="http://localhost:8015",
    poll_interval_seconds=3600,  # 1 hour
    max_iterations=None,         # Run forever
)
```

### Async Usage (FastAPI)

```python
from macrocrypto.services.dkg_confirming_node import AsyncGraphConfirmingNode
from macrocrypto.services.dkg_service import AsyncDKGService

# Initialize async versions
dkg = AsyncDKGService(node_endpoint="http://localhost:8900")
node = AsyncGraphConfirmingNode(
    dkg_service=dkg,
    verifier_address="0xYourAddress",
)

# Verify asynchronously
report = await node.verify_snapshot("did:dkg:otp:20430/0xcdb.../405801")
```

## Verification Flow

### Step 1: Oracle Publishes Snapshot

```python
# InvestBud AI API calls /dkg/publish
snapshot = RegimeSnapshot(
    regime="Risk-Off",
    confidence=0.85,
    signature="0x...",  # Signed by oracle's private key
    signer_address="0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF",
    ...
)
result = dkg.publish_snapshot(snapshot)
# Returns: UAL = did:dkg:otp:20430/0xcdb.../405801
```

### Step 2: Confirming Node Polls for New Snapshots

The node polls the oracle API's `/dkg/uals` endpoint:

```python
response = requests.get(
    f"{oracle_api_url}/dkg/uals",
    params={"limit": 10, "since_timestamp": last_timestamp},
)
uals = response.json().get("uals", [])
```

### Step 3: Independent Verification

For each new snapshot:

```python
# 1. Fetch original snapshot from DKG
original = dkg.get_snapshot(snapshot_ual)
public = original.get("public", {})

# 2. Run SAME model with FRESH data
computed_regime, computed_confidence, computed_risk_on_prob, computed_features = \
    node.compute_prediction()

# 3. Compare results
regime_matches = (public.get("regime") == computed_regime)
confidence_diff = abs(public.get("confidence", 0) - computed_confidence)

# 4. Determine result
if regime_matches and confidence_diff <= 0.10:
    result = VerificationResult.CONFIRMED
elif not regime_matches:
    result = VerificationResult.CHALLENGED
else:
    result = VerificationResult.PARTIAL
```

### Step 4: Publish Verification Report

```python
report = VerificationReport(
    snapshot_ual=snapshot_ual,
    original_snapshot=public,
    verification_result=result,
    timestamp=datetime.now(timezone.utc).isoformat(),
    verifier_address="0xABC...",
    computed_regime="Risk-Off",
    computed_confidence=0.84,
    computed_risk_on_probability=0.16,
    regime_matches=True,
    confidence_difference=0.01,
    feature_differences={"btcPrice": 0.002, "vix": 0.01},
)
node.publish_verification(report)
```

### Step 5: Query Confirmations

```sparql
# How many independent confirmations does this snapshot have?
PREFIX mc: <https://macrocrypto.io/ontology#>

SELECT ?verifier ?result
WHERE {
    ?report mc:verifies <did:dkg:otp:20430/0xcdb.../405801> .
    ?report mc:verificationResult ?result .
    ?report mc:verifierAddress ?verifier .
}
```

## Trust Levels

| Confirmations  | Trust Level | Interpretation                              |
| -------------- | ----------- | ------------------------------------------- |
| 0              | Unverified  | Just published, no independent verification |
| 1-2            | Low         | Some verification, could be sybil           |
| 3-5            | Medium      | Multiple independent verifiers agree        |
| 5+             | High        | Strong consensus from diverse verifiers     |
| Any CHALLENGED | Disputed    | At least one verifier disagrees             |

## Security Considerations

### Attack Vectors

1. **Fake Snapshots**: Anyone can publish `mc:RegimeSnapshot`

   - **Mitigation**: Check `signerAddress` matches official oracle

2. **Sybil Verifiers**: Attacker runs many fake confirming nodes

   - **Mitigation**: Weight verifiers by reputation/stake

3. **Stale Data Attack**: Verifier uses old data to get different result

   - **Mitigation**: Compare timestamps, require recent data

4. **Model Divergence**: Verifiers have different model versions
   - **Mitigation**: Include model version in verification report

### Best Practices

1. **Always verify signer address** before trusting a snapshot
2. **Run your own confirming node** for maximum trust
3. **Check multiple confirmations** from diverse verifiers
4. **Verify the signature** cryptographically

## JSON-LD Schemas

### RegimeSnapshot (Published by Oracle)

```json
{
  "@context": {
    "@vocab": "https://schema.org/",
    "mc": "https://macrocrypto.io/ontology#"
  },
  "@type": ["mc:RegimeSnapshot", "Dataset"],
  "@id": "https://macrocrypto.io/snapshots/1732752000",

  "regime": "Risk-Off",
  "regimeBinary": 0,
  "confidence": 0.85,
  "riskOnProbability": 0.15,

  "snapshotTimestamp": "2025-11-28T00:00:00Z",
  "snapshotHash": "0x1234567890abcdef...",
  "signature": "0xabcdef1234567890...",
  "signerAddress": "0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF",

  "btcPrice": 95000,
  "btcReturns30d": -0.05,
  "vix": 18.5,
  "fedFunds": 4.5
}
```

### VerificationReport (Published by Confirming Node)

```json
{
  "@context": {
    "@vocab": "https://schema.org/",
    "mc": "https://macrocrypto.io/ontology#"
  },
  "@type": ["mc:VerificationReport", "Dataset"],
  "@id": "https://macrocrypto.io/verifications/2025-11-28T01-00-00Z",
  "name": "Verification Report for did:dkg:otp:20430/0xcdb.../405801",
  "dateCreated": "2025-11-28T01:00:00Z",

  "mc:verifies": "did:dkg:otp:20430/0xcdb.../405801",
  "mc:verificationResult": "confirmed",
  "mc:verifierAddress": "0xABC123...",

  "mc:computedRegime": "Risk-Off",
  "mc:computedConfidence": 0.84,
  "mc:computedRiskOnProbability": 0.16,
  "mc:regimeMatches": true,
  "mc:confidenceDifference": 0.01,

  "confirmedBy": "0xABC123..."
}
```

## API Endpoints

The confirming node interacts with these oracle API endpoints:

| Endpoint              | Method | Description                          |
| --------------------- | ------ | ------------------------------------ |
| `/dkg/uals`           | GET    | List published UALs for verification |
| `/dkg/snapshot/{ual}` | GET    | Get snapshot details by UAL          |
| `/dkg/verify`         | POST   | Trigger verification via API         |

## Docker Deployment

```bash
# Build the confirming node image
docker build -f Dockerfile.confirming-node -t investbud-confirming-node .

# Run with environment variables
docker run -d \
  -e VERIFIER_ADDRESS=0xYourAddress \
  -e ORACLE_SIGNER_ADDRESS=0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF \
  -e DKG_NODE_ENDPOINT=http://dkg-node:8900 \
  -e ORACLE_API_URL=http://api:8015 \
  -e POLL_INTERVAL=3600 \
  investbud-confirming-node
```
