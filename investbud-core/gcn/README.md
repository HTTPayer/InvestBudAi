# InvestBud AI Graph Confirming Node

Independent verifier for InvestBud AI oracle predictions on OriginTrail's Decentralized Knowledge Graph (DKG).

## What It Does

1. **Polls** the oracle API for new published snapshots
2. **Fetches** each snapshot from the DKG
3. **Computes** an independent prediction using fresh market data
4. **Compares** the original and computed predictions
5. **Publishes** a verification report (CONFIRMED/CHALLENGED) to the DKG

This creates a trust network where multiple independent verifiers confirm oracle predictions.

## Quick Start

```bash
# Install
uv sync

# Configure
cp .env.sample .env
# Edit .env with your settings

# Run
uv run gcn
```

## Installation

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Configuration

Create a `.env` file:

```bash
# Required
VERIFIER_ADDRESS=0xYourVerifierAddress

# DKG Node
DKG_NODE_ENDPOINT=https://v6-pegasus-node-02.origin-trail.network:8900
DKG_BLOCKCHAIN_ID=otp:20430

# Oracle API
ORACLE_API_URL=https://r3467d7khd8b94sfguhrr273lo.ingress.akashprovid.com
ORACLE_SIGNER_ADDRESS=0x85bADB8118f98b6AD8eaB5880c8829AF8adA44cF

# Verification loop
POLL_INTERVAL=3600
```

## CLI Usage

```bash
# Run continuous verification loop
uv run gcn

# Verify a specific snapshot
uv run gcn --verify-ual did:dkg:otp:20430/0xcdb.../405801

# Verify and publish to DKG
uv run gcn --verify-ual did:dkg:otp:20430/0xcdb.../405801 --publish

# Show verification statistics
uv run gcn --stats

# Show verification history
uv run gcn --history
```

## Verification Results

| Result | Description |
|--------|-------------|
| `CONFIRMED` | Regime matches, confidence within 10% |
| `CHALLENGED` | Regime prediction differs |
| `PARTIAL` | Regime matches but confidence differs > 10% |
| `ERROR` | Verification failed |

## Architecture

```
Oracle publishes snapshot to DKG
         |
         v
+---------------------------------------+
|     Graph Confirming Node             |
+---------------------------------------+
| 1. Fetch snapshot from DKG            |
| 2. Get current prediction from API    |
| 3. Compare regime & confidence        |
| 4. Publish verification report        |
+---------------------------------------+
         |
         v
Verification report on DKG
(linked to original snapshot)
```

## How Trust Works

1. **Cryptographic Signatures**: Every snapshot has a `signerAddress`
2. **Multi-Party Verification**: Multiple independent nodes verify the same prediction
3. **On-Chain Anchoring**: All reports are immutably stored on NeuroWeb

```
Snapshot: "Risk-Off, 85%"
    |
    +-- Node A: CONFIRMED (computed: Risk-Off, 84%)
    +-- Node B: CONFIRMED (computed: Risk-Off, 86%)
    +-- Node C: CHALLENGED (computed: Risk-On, 60%)
```

Multiple confirmations = higher trust. Any challenge = disputed.

## Querying Verifications

```sparql
PREFIX mc: <https://macrocrypto.io/ontology#>

SELECT ?verifier ?result
WHERE {
    ?report mc:verifies <did:dkg:otp:20430/0xcdb.../405801> .
    ?report mc:verificationResult ?result .
    ?report mc:verifierAddress ?verifier .
}
```

## Docker

```bash
# Build
docker build -t investbud-gcn .

# Run
docker run -d \
  -e VERIFIER_ADDRESS=0xYourAddress \
  -e ORACLE_API_URL=https://r3467d7khd8b94sfguhrr273lo.ingress.akashprovid.com \
  investbud-gcn
```

## Persistent Storage

Verification history is stored in SQLite at `cache/confirming_node.db`:

- `verified_snapshots`: All verified UALs with results
- `verification_reports`: Full JSON-LD reports

This persists across restarts so the node won't re-verify the same snapshots.

## License

TBA
