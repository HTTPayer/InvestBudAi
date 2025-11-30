"""
DKG (Decentralized Knowledge Graph) Service

Simplified DKG client for the Graph Confirming Node.
Handles querying and publishing to OriginTrail DKG.
"""

import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# DKG SDK imports
try:
    from dkg import DKG
    from dkg.providers import BlockchainProvider, NodeHTTPProvider
    DKG_AVAILABLE = True
except ImportError:
    DKG_AVAILABLE = False
    print("[!] DKG SDK not installed. Run: pip install dkg")


# Schema context for Knowledge Assets
MACROCRYPTO_CONTEXT = {
    "@context": {
        "@vocab": "https://schema.org/",
        "mc": "https://macrocrypto.io/ontology#",
        "dkg": "https://origintrail.io/ontology#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",

        "regime": "mc:regime",
        "regimeBinary": {"@id": "mc:regimeBinary", "@type": "xsd:integer"},
        "confidence": {"@id": "mc:confidence", "@type": "xsd:decimal"},
        "riskOnProbability": {"@id": "mc:riskOnProbability", "@type": "xsd:decimal"},

        "snapshotTimestamp": {"@id": "mc:snapshotTimestamp", "@type": "xsd:dateTime"},
        "blockTimestamp": {"@id": "mc:blockTimestamp", "@type": "xsd:integer"},

        "snapshotHash": "mc:snapshotHash",
        "signature": "mc:signature",
        "signerAddress": "mc:signerAddress",

        "btcPrice": {"@id": "mc:btcPrice", "@type": "xsd:decimal"},
        "vix": {"@id": "mc:vix", "@type": "xsd:decimal"},
        "fedFunds": {"@id": "mc:fedFunds", "@type": "xsd:decimal"},

        "previousSnapshot": {"@id": "mc:previousSnapshot", "@type": "@id"},
        "confirmedBy": {"@id": "mc:confirmedBy", "@type": "@id"},
        "challenges": {"@id": "mc:challenges", "@type": "@id"},
    }
}


@dataclass
class RegimeSnapshot:
    """Structured regime snapshot."""
    regime: str
    regime_binary: int
    confidence: float
    risk_on_probability: float
    timestamp: int
    date: str
    snapshot_hash: str
    signature: str
    signer_address: str
    transaction_hash: Optional[str]
    features: Dict[str, Any]


class DKGService:
    """Service for querying and publishing to OriginTrail DKG."""

    def __init__(
        self,
        node_endpoint: str = None,
        blockchain_id: str = None,
        api_version: str = "v1",
    ):
        """
        Initialize DKG service.

        Args:
            node_endpoint: OriginTrail node endpoint (default from env)
            blockchain_id: Blockchain ID (default: otp:20430 for NeuroWeb Testnet)
            api_version: DKG API version
        """
        self.node_endpoint = node_endpoint or os.getenv(
            "DKG_NODE_ENDPOINT", "http://localhost:8900"
        )
        self.blockchain_id = blockchain_id or os.getenv(
            "DKG_BLOCKCHAIN_ID", "otp:20430"
        )
        self.api_version = api_version
        self._dkg = None

    def _is_public_node(self) -> bool:
        """Check if using a public DKG node."""
        return "origin-trail.network" in self.node_endpoint

    def _get_client(self) -> 'DKG':
        """Get or create DKG client."""
        if not DKG_AVAILABLE:
            raise RuntimeError("DKG SDK not installed. Run: pip install dkg")

        if self._dkg is None:
            node_provider = NodeHTTPProvider(
                endpoint_uri=self.node_endpoint,
                api_version=self.api_version
            )
            blockchain_provider = BlockchainProvider(self.blockchain_id)

            # Longer timeouts for public nodes
            if self._is_public_node():
                max_retries = 120
                frequency = 3
            else:
                max_retries = 60
                frequency = 3

            self._dkg = DKG(
                node_provider,
                blockchain_provider,
                config={"max_number_of_retries": max_retries, "frequency": frequency},
            )
            print(f"[OK] DKG client: {self.node_endpoint} (retries: {max_retries})")

        return self._dkg

    def get_node_info(self) -> Dict[str, Any]:
        """Get DKG node information."""
        dkg = self._get_client()
        return dkg.node.info

    def get_snapshot(self, ual: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Retrieve a snapshot from DKG with retry logic.

        Args:
            ual: Universal Asset Locator
            max_retries: Number of retry attempts

        Returns:
            Asset data
        """
        dkg = self._get_client()

        if self._is_public_node():
            initial_wait = 30
            retry_wait = 60
        else:
            initial_wait = 10
            retry_wait = 20

        last_error = None

        for attempt in range(max_retries):
            try:
                result = dkg.asset.get(ual)
                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = initial_wait if attempt == 0 else retry_wait
                    print(f"[...] Retry {attempt + 2}/{max_retries} in {wait_time}s...")
                    time.sleep(wait_time)

        raise last_error

    def query_snapshots(
        self,
        regime: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 100,
        signer_address: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query regime snapshots using SPARQL.

        Args:
            regime: Filter by regime ('Risk-On' or 'Risk-Off')
            min_confidence: Minimum confidence threshold
            limit: Maximum results
            signer_address: Filter by signer address

        Returns:
            List of matching snapshots
        """
        dkg = self._get_client()

        filters = []
        if signer_address:
            filters.append(f'FILTER(?signerAddress = "{signer_address}")')
        if regime:
            filters.append(f'FILTER(?regime = "{regime}")')
        if min_confidence:
            filters.append(f'FILTER(?confidence >= {min_confidence})')

        filter_clause = "\n        ".join(filters)

        query = f"""
        PREFIX mc: <https://macrocrypto.io/ontology#>

        SELECT ?snapshot ?regime ?confidence ?timestamp ?signerAddress
        WHERE {{
            ?snapshot a mc:RegimeSnapshot .
            ?snapshot mc:regime ?regime .
            ?snapshot mc:confidence ?confidence .
            ?snapshot mc:snapshotTimestamp ?timestamp .
            ?snapshot mc:signerAddress ?signerAddress .
            {filter_clause}
        }}
        ORDER BY DESC(?timestamp)
        LIMIT {limit}
        """

        return dkg.graph.query(query)
