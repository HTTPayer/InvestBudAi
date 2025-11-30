"""
DKG (Decentralized Knowledge Graph) Service

Publishes MacroCrypto regime snapshots to OriginTrail DKG as Knowledge Assets.
Enables verifiable, decentralized history of macro classifications.

Reference: https://docs.origintrail.io/build-a-dkg-node-ai-agent/advanced-features-and-toolkits/dkg-sdk/dkg-v8-py-client
"""

import os
import json
import time
import asyncio
import hashlib
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# DKG SDK imports
try:
    from dkg import DKG, AsyncDKG
    from dkg.providers import BlockchainProvider, NodeHTTPProvider
    from dkg.providers import AsyncBlockchainProvider, AsyncNodeHTTPProvider
    DKG_AVAILABLE = True
except ImportError:
    DKG_AVAILABLE = False
    print("[!] DKG SDK not installed. Run: pip install dkg")


# Official MacroCrypto Oracle signer address (for confirming node filtering)
# Confirming nodes should ONLY verify snapshots from this address
OFFICIAL_ORACLE_ADDRESS = os.getenv("ORACLE_SIGNER_ADDRESS", "")


# ============================================================================
# Persistent Storage for Published UALs
# ============================================================================

class UALStore:
    """SQLite-based persistent storage for tracking published DKG UALs."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            cache_dir = Path(os.getenv("CACHE_DIR", "cache"))
            cache_dir.mkdir(exist_ok=True)
            db_path = str(cache_dir / "dkg_uals.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS published_uals (
                    ual TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    regime TEXT,
                    regime_binary INTEGER,
                    confidence REAL,
                    risk_on_probability REAL,
                    signer_address TEXT,
                    dataset_root TEXT,
                    published_at TEXT
                )
            """)
            # Index for timestamp queries (confirming nodes filter by since_timestamp)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON published_uals(timestamp DESC)
            """)
            # Index for regime queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_regime ON published_uals(regime)
            """)
            conn.commit()

    def save_ual(
        self,
        ual: str,
        timestamp: int,
        regime: str = None,
        regime_binary: int = None,
        confidence: float = None,
        risk_on_probability: float = None,
        signer_address: str = None,
        dataset_root: str = None,
    ):
        """Save a published UAL."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO published_uals
                (ual, timestamp, regime, regime_binary, confidence,
                 risk_on_probability, signer_address, dataset_root, published_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ual,
                timestamp,
                regime,
                regime_binary,
                confidence,
                risk_on_probability,
                signer_address,
                dataset_root,
                datetime.now(timezone.utc).isoformat()
            ))
            conn.commit()

    def get_uals(
        self,
        limit: int = 10,
        since_timestamp: int = None,
        regime: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Get published UALs with optional filtering.

        Args:
            limit: Maximum number of UALs to return
            since_timestamp: Only return UALs published after this Unix timestamp
            regime: Filter by regime ('Risk-On' or 'Risk-Off')

        Returns:
            List of UAL entries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM published_uals WHERE 1=1"
            params = []

            if since_timestamp:
                query += " AND timestamp > ?"
                params.append(since_timestamp)

            if regime:
                query += " AND regime = ?"
                params.append(regime)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get the most recently published UAL."""
        uals = self.get_uals(limit=1)
        return uals[0] if uals else None

    def get_by_ual(self, ual: str) -> Optional[Dict[str, Any]]:
        """Get a specific UAL entry."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM published_uals WHERE ual = ?",
                (ual,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about published UALs."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM published_uals").fetchone()[0]
            risk_on = conn.execute(
                "SELECT COUNT(*) FROM published_uals WHERE regime = 'Risk-On'"
            ).fetchone()[0]
            risk_off = conn.execute(
                "SELECT COUNT(*) FROM published_uals WHERE regime = 'Risk-Off'"
            ).fetchone()[0]

            # Get date range
            first = conn.execute(
                "SELECT MIN(timestamp) FROM published_uals"
            ).fetchone()[0]
            last = conn.execute(
                "SELECT MAX(timestamp) FROM published_uals"
            ).fetchone()[0]

            return {
                "total_published": total,
                "risk_on_count": risk_on,
                "risk_off_count": risk_off,
                "first_timestamp": first,
                "last_timestamp": last,
            }

    def exists(self, ual: str) -> bool:
        """Check if a UAL exists in the store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM published_uals WHERE ual = ?",
                (ual,)
            )
            return cursor.fetchone() is not None


# Global UAL store instance
_ual_store: Optional["UALStore"] = None


def get_ual_store() -> UALStore:
    """Get global UAL store instance."""
    global _ual_store
    if _ual_store is None:
        _ual_store = UALStore()
        print(f"[OK] UAL store initialized: {_ual_store.db_path}")
    return _ual_store


# Schema context for MacroCrypto Knowledge Assets
MACROCRYPTO_CONTEXT = {
    "@context": {
        "@vocab": "https://schema.org/",
        "mc": "https://macrocrypto.io/ontology#",
        "dkg": "https://origintrail.io/ontology#",
        "xsd": "http://www.w3.org/2001/XMLSchema#",

        # Core regime properties
        "regime": "mc:regime",
        "regimeBinary": {"@id": "mc:regimeBinary", "@type": "xsd:integer"},
        "confidence": {"@id": "mc:confidence", "@type": "xsd:decimal"},
        "riskOnProbability": {"@id": "mc:riskOnProbability", "@type": "xsd:decimal"},

        # Temporal properties
        "snapshotTimestamp": {"@id": "mc:snapshotTimestamp", "@type": "xsd:dateTime"},
        "blockTimestamp": {"@id": "mc:blockTimestamp", "@type": "xsd:integer"},

        # Cryptographic properties
        "snapshotHash": "mc:snapshotHash",
        "signature": "mc:signature",
        "signerAddress": "mc:signerAddress",
        "transactionHash": "mc:transactionHash",

        # BTC indicators
        "btcPrice": {"@id": "mc:btcPrice", "@type": "xsd:decimal"},
        "btcReturns1d": {"@id": "mc:btcReturns1d", "@type": "xsd:decimal"},
        "btcReturns7d": {"@id": "mc:btcReturns7d", "@type": "xsd:decimal"},
        "btcReturns30d": {"@id": "mc:btcReturns30d", "@type": "xsd:decimal"},
        "btcReturns60d": {"@id": "mc:btcReturns60d", "@type": "xsd:decimal"},
        "btcRsi": {"@id": "mc:btcRsi", "@type": "xsd:decimal"},
        "btcDrawdown": {"@id": "mc:btcDrawdown", "@type": "xsd:decimal"},
        "btcVolatility30d": {"@id": "mc:btcVolatility30d", "@type": "xsd:decimal"},
        "btcPriceVsMa30": {"@id": "mc:btcPriceVsMa30", "@type": "xsd:decimal"},
        "btcPriceVsMa200": {"@id": "mc:btcPriceVsMa200", "@type": "xsd:decimal"},
        "btcMa30Above200": {"@id": "mc:btcMa30Above200", "@type": "xsd:boolean"},

        # Macro indicators
        "fedFunds": {"@id": "mc:fedFunds", "@type": "xsd:decimal"},
        "cpiYoy": {"@id": "mc:cpiYoy", "@type": "xsd:decimal"},
        "vix": {"@id": "mc:vix", "@type": "xsd:decimal"},
        "yieldCurveSpread": {"@id": "mc:yieldCurveSpread", "@type": "xsd:decimal"},
        "m2Growth": {"@id": "mc:m2Growth", "@type": "xsd:decimal"},

        # Relationships
        "previousSnapshot": {"@id": "mc:previousSnapshot", "@type": "@id"},
        "confirmedBy": {"@id": "mc:confirmedBy", "@type": "@id"},
        "challenges": {"@id": "mc:challenges", "@type": "@id"},
        "extends": {"@id": "mc:extends", "@type": "@id"},
    }
}


@dataclass
class RegimeSnapshot:
    """Structured regime snapshot for DKG publishing."""
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

    @property
    def iso_date(self) -> str:
        """Return date in ISO 8601 format (with 'T' separator)."""
        # Convert '2025-11-25 00:00:00' to '2025-11-25T00:00:00'
        return self.date.replace(' ', 'T') if ' ' in self.date else self.date

    def to_json_ld(self, asset_id: Optional[str] = None) -> Dict[str, Any]:
        """Convert to JSON-LD format for DKG publishing."""
        # Generate asset ID if not provided
        if asset_id is None:
            asset_id = f"https://macrocrypto.io/snapshots/{self.timestamp}"

        # Build the public assertion (visible to all DKG nodes)
        public_assertion = {
            **MACROCRYPTO_CONTEXT,
            "@id": asset_id,
            "@type": ["mc:RegimeSnapshot", "Dataset"],
            "name": f"MacroCrypto Regime Snapshot {self.date}",
            "description": f"AI-powered macro regime classification: {self.regime} with {self.confidence:.1%} confidence",
            "dateCreated": self.iso_date,
            "creator": {
                "@type": "Organization",
                "@id": "https://macrocrypto.io",
                "name": "MacroCrypto Oracle"
            },

            # Core regime data
            "regime": self.regime,
            "regimeBinary": self.regime_binary,
            "confidence": round(self.confidence, 4),
            "riskOnProbability": round(self.risk_on_probability, 4),

            # Temporal data
            "snapshotTimestamp": self.iso_date,
            "blockTimestamp": self.timestamp,

            # Cryptographic proof
            "snapshotHash": self.snapshot_hash,
            "signature": self.signature,
            "signerAddress": self.signer_address,
        }

        # Add transaction hash if available
        if self.transaction_hash:
            public_assertion["transactionHash"] = self.transaction_hash

        # Add BTC indicators (public)
        btc_features = {
            "btcPrice": self.features.get("btc_price"),
            "btcReturns1d": self.features.get("btc_returns_1d"),
            "btcReturns7d": self.features.get("btc_returns_7d"),
            "btcReturns30d": self.features.get("btc_returns_30d"),
            "btcReturns60d": self.features.get("btc_returns_60d"),
            "btcRsi": self.features.get("btc_rsi"),
            "btcDrawdown": self.features.get("btc_drawdown"),
            "btcVolatility30d": self.features.get("btc_volatility_30d"),
            "btcPriceVsMa30": self.features.get("btc_price_vs_ma_30"),
            "btcPriceVsMa200": self.features.get("btc_price_vs_ma_200"),
            "btcMa30Above200": self.features.get("btc_ma_30_above_200"),
        }
        # Filter None values, round floats, and convert booleans
        for key, value in btc_features.items():
            if value is not None:
                if key == "btcMa30Above200":
                    # Convert to proper boolean (may come as 0.0/1.0 or 0/1)
                    public_assertion[key] = bool(value)
                elif isinstance(value, float):
                    public_assertion[key] = round(value, 6)
                else:
                    public_assertion[key] = value

        # Add macro indicators (public)
        macro_features = {
            "fedFunds": self.features.get("fed_funds"),
            "cpiYoy": self.features.get("cpi_yoy"),
            "vix": self.features.get("vix"),
            "yieldCurveSpread": self.features.get("yield_curve_spread"),
            "m2Growth": self.features.get("m2_growth"),
        }
        for key, value in macro_features.items():
            if value is not None:
                if isinstance(value, float):
                    public_assertion[key] = round(value, 6)
                else:
                    public_assertion[key] = value

        return public_assertion

    def to_dkg_content(self, previous_ual: Optional[str] = None) -> Dict[str, Any]:
        """
        Create full DKG content with public and private assertions.

        Args:
            previous_ual: UAL of the previous snapshot for linking
        """
        public = self.to_json_ld()

        # Link to previous snapshot if available
        if previous_ual:
            public["previousSnapshot"] = previous_ual

        # Private assertion (only stored locally, not replicated)
        private = {
            "@context": MACROCRYPTO_CONTEXT["@context"],
            "@id": public["@id"],
            "@type": "mc:RegimeSnapshotPrivate",
            # Full feature set (some may be proprietary)
            "fullFeatures": self.features,
            # Model metadata
            "modelVersion": "1.0.0",
            "modelType": "LogisticRegression",
        }

        return {
            "public": public,
            "private": private,
        }


class DKGService:
    """
    Service for publishing and querying MacroCrypto knowledge assets on OriginTrail DKG.
    """

    def __init__(
        self,
        node_endpoint: str = "http://localhost:8900",
        blockchain_environment: str = "testnet",
        blockchain_id: str = "otp:20430",  # Neuroweb Testnet
        api_version: str = "v1",
    ):
        """
        Initialize DKG service.

        Args:
            node_endpoint: OriginTrail node HTTP endpoint
            blockchain_environment: 'mainnet', 'testnet', or 'development'
            blockchain_id: Blockchain identifier (e.g., 'otp:20430' for Neuroweb Testnet)
            api_version: DKG API version
        """
        self.node_endpoint = node_endpoint
        self.blockchain_environment = blockchain_environment
        self.blockchain_id = blockchain_id
        self.api_version = api_version
        self._dkg = None
        self._last_ual: Optional[str] = None  # Track last published UAL for chaining

    def _is_public_node(self) -> bool:
        """Check if using a public DKG node (slower, needs longer timeouts)."""
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
            blockchain_provider = BlockchainProvider(
                self.blockchain_id,
            )

            # Use longer timeouts for public nodes
            if self._is_public_node():
                max_retries = 120  # 6 minutes for public node
                frequency = 3
            else:
                max_retries = 60   # 3 minutes for local node
                frequency = 3

            self._dkg = DKG(
                node_provider,
                blockchain_provider,
                config={"max_number_of_retries": max_retries, "frequency": frequency},
            )
            print(f"[OK] DKG client initialized: {self.node_endpoint} (retries: {max_retries})")

        return self._dkg

    def get_node_info(self) -> Dict[str, Any]:
        """Get DKG node information."""
        dkg = self._get_client()
        return dkg.node.info

    def publish_snapshot(
        self,
        snapshot: RegimeSnapshot,
        epochs_num: int = 2,
        min_confirmations: int = 3,
        min_replications: int = 1,
    ) -> Dict[str, Any]:
        """
        Publish a regime snapshot to the DKG.

        Args:
            snapshot: RegimeSnapshot to publish
            epochs_num: Number of epochs for asset validity
            min_confirmations: Minimum finalization confirmations
            min_replications: Minimum node replications

        Returns:
            Dict with UAL, dataset root, and transaction details
        """
        dkg = self._get_client()

        # Create content with link to previous snapshot
        content = snapshot.to_dkg_content(previous_ual=self._last_ual)

        print(f"[...] Publishing snapshot to DKG: {snapshot.regime} @ {snapshot.date}")

        result = dkg.asset.create(
            content=content,
            options={
                "epochs_num": epochs_num,
                "minimum_number_of_finalization_confirmations": min_confirmations,
                "minimum_number_of_node_replications": min_replications,
            }
        )

        # Store UAL for chaining
        if "UAL" in result:
            self._last_ual = result["UAL"]
            print(f"[OK] Published to DKG: {self._last_ual}")

        return result

    def get_snapshot(self, ual: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Retrieve a snapshot from the DKG by UAL with retry logic.

        Args:
            ual: Universal Asset Locator
            max_retries: Number of retry attempts (with waits between)

        Returns:
            Asset data
        """
        dkg = self._get_client()

        # Determine wait times based on node type
        if self._is_public_node():
            initial_wait = 30   # Wait before first retry
            retry_wait = 60     # Wait between subsequent retries
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
                    print(f"[...] Get failed, waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    time.sleep(wait_time)

        # All retries failed
        raise last_error

    def query_snapshots(
        self,
        regime: Optional[str] = None,
        min_confidence: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        signer_address: Optional[str] = None,
        official_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Query regime snapshots from the DKG using SPARQL.

        Args:
            regime: Filter by regime ('Risk-On' or 'Risk-Off')
            min_confidence: Minimum confidence threshold
            start_date: Start date filter (ISO format)
            end_date: End date filter (ISO format)
            limit: Maximum results
            signer_address: Filter by specific signer address
            official_only: If True, only return snapshots from ORACLE_SIGNER_ADDRESS

        Returns:
            List of matching snapshots
        """
        dkg = self._get_client()

        # Build SPARQL query
        filters = []

        # Signer address filter (critical for confirming nodes)
        if signer_address:
            filters.append(f'FILTER(?signerAddress = "{signer_address}")')
        elif official_only and OFFICIAL_ORACLE_ADDRESS:
            filters.append(f'FILTER(?signerAddress = "{OFFICIAL_ORACLE_ADDRESS}")')

        if regime:
            filters.append(f'FILTER(?regime = "{regime}")')
        if min_confidence:
            filters.append(f'FILTER(?confidence >= {min_confidence})')
        if start_date:
            filters.append(f'FILTER(?timestamp >= "{start_date}"^^xsd:dateTime)')
        if end_date:
            filters.append(f'FILTER(?timestamp <= "{end_date}"^^xsd:dateTime)')

        filter_clause = "\n        ".join(filters)

        query = f"""
        PREFIX mc: <https://macrocrypto.io/ontology#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

        SELECT ?snapshot ?regime ?confidence ?timestamp ?btcPrice ?snapshotHash ?signerAddress
        WHERE {{
            ?snapshot a mc:RegimeSnapshot .
            ?snapshot mc:regime ?regime .
            ?snapshot mc:confidence ?confidence .
            ?snapshot mc:snapshotTimestamp ?timestamp .
            ?snapshot mc:btcPrice ?btcPrice .
            ?snapshot mc:snapshotHash ?snapshotHash .
            ?snapshot mc:signerAddress ?signerAddress .
            {filter_clause}
        }}
        ORDER BY DESC(?timestamp)
        LIMIT {limit}
        """

        result = dkg.graph.query(query)
        return result

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get the most recent regime snapshot."""
        results = self.query_snapshots(limit=1)
        return results[0] if results else None

    def verify_snapshot_chain(self, ual: str, depth: int = 10) -> List[Dict[str, Any]]:
        """
        Verify a chain of snapshots by following previousSnapshot links.

        Args:
            ual: Starting UAL
            depth: Maximum chain depth to follow

        Returns:
            List of snapshots in the chain
        """
        chain = []
        current_ual = ual

        for _ in range(depth):
            if not current_ual:
                break

            try:
                asset = self.get_snapshot(current_ual)
                chain.append({
                    "ual": current_ual,
                    "data": asset
                })

                # Get previous snapshot link
                public = asset.get("public", {})
                current_ual = public.get("previousSnapshot")
            except Exception as e:
                print(f"[!] Chain verification stopped: {e}")
                break

        return chain


class AsyncDKGService:
    """
    Async version of DKG service for use with FastAPI.
    """

    def __init__(
        self,
        node_endpoint: str = "http://localhost:8900",
        blockchain_environment: str = "testnet",
        blockchain_id: str = "otp:20430",
        api_version: str = "v1",
    ):
        self.node_endpoint = node_endpoint
        self.blockchain_environment = blockchain_environment
        self.blockchain_id = blockchain_id
        self.api_version = api_version
        self._dkg = None
        self._last_ual: Optional[str] = None

    def _is_public_node(self) -> bool:
        """Check if using a public DKG node (slower, needs longer timeouts)."""
        return "origin-trail.network" in self.node_endpoint

    async def _get_client(self) -> 'AsyncDKG':
        """Get or create async DKG client."""
        if not DKG_AVAILABLE:
            raise RuntimeError("DKG SDK not installed. Run: pip install dkg")

        if self._dkg is None:
            node_provider = AsyncNodeHTTPProvider(
                endpoint_uri=self.node_endpoint,
                api_version=self.api_version
            )
            blockchain_provider = AsyncBlockchainProvider(
                self.blockchain_id,
            )

            # Use longer timeouts for public nodes
            if self._is_public_node():
                max_retries = 120  # 6 minutes for public node
                frequency = 3
            else:
                max_retries = 60   # 3 minutes for local node
                frequency = 3

            self._dkg = AsyncDKG(
                node_provider,
                blockchain_provider,
                config={"max_number_of_retries": max_retries, "frequency": frequency},
            )
            print(f"[OK] Async DKG client initialized: {self.node_endpoint} (retries: {max_retries})")

        return self._dkg

    async def publish_snapshot(
        self,
        snapshot: RegimeSnapshot,
        epochs_num: int = 2,
        min_confirmations: int = 3,
        min_replications: int = 1,
    ) -> Dict[str, Any]:
        """Async publish a regime snapshot to the DKG."""
        dkg = await self._get_client()

        content = snapshot.to_dkg_content(previous_ual=self._last_ual)

        print(f"[...] Publishing snapshot to DKG: {snapshot.regime} @ {snapshot.date}")

        result = await dkg.asset.create(
            content=content,
            options={
                "epochs_num": epochs_num,
                "minimum_number_of_finalization_confirmations": min_confirmations,
                "minimum_number_of_node_replications": min_replications,
            }
        )

        if "UAL" in result:
            self._last_ual = result["UAL"]
            print(f"[OK] Published to DKG: {self._last_ual}")

        return result

    async def get_snapshot(self, ual: str, max_retries: int = 3) -> Dict[str, Any]:
        """Async retrieve a snapshot from the DKG with retry logic."""
        dkg = await self._get_client()

        # Determine wait times based on node type
        if self._is_public_node():
            initial_wait = 30
            retry_wait = 60
        else:
            initial_wait = 10
            retry_wait = 20

        last_error = None

        for attempt in range(max_retries):
            try:
                result = await dkg.asset.get(ual)
                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = initial_wait if attempt == 0 else retry_wait
                    print(f"[...] Get failed, waiting {wait_time}s before retry {attempt + 2}/{max_retries}...")
                    await asyncio.sleep(wait_time)

        raise last_error

    async def query_snapshots(
        self,
        regime: Optional[str] = None,
        min_confidence: Optional[float] = None,
        limit: int = 100,
        signer_address: Optional[str] = None,
        official_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Async query regime snapshots.

        Args:
            regime: Filter by regime ('Risk-On' or 'Risk-Off')
            min_confidence: Minimum confidence threshold
            limit: Maximum results
            signer_address: Filter by specific signer address
            official_only: If True, only return snapshots from ORACLE_SIGNER_ADDRESS
        """
        dkg = await self._get_client()

        filters = []

        # Signer address filter (critical for confirming nodes)
        if signer_address:
            filters.append(f'FILTER(?signerAddress = "{signer_address}")')
        elif official_only and OFFICIAL_ORACLE_ADDRESS:
            filters.append(f'FILTER(?signerAddress = "{OFFICIAL_ORACLE_ADDRESS}")')

        if regime:
            filters.append(f'FILTER(?regime = "{regime}")')
        if min_confidence:
            filters.append(f'FILTER(?confidence >= {min_confidence})')

        filter_clause = "\n        ".join(filters)

        query = f"""
        PREFIX mc: <https://macrocrypto.io/ontology#>

        SELECT ?snapshot ?regime ?confidence ?timestamp ?btcPrice ?signerAddress
        WHERE {{
            ?snapshot a mc:RegimeSnapshot .
            ?snapshot mc:regime ?regime .
            ?snapshot mc:confidence ?confidence .
            ?snapshot mc:snapshotTimestamp ?timestamp .
            ?snapshot mc:btcPrice ?btcPrice .
            ?snapshot mc:signerAddress ?signerAddress .
            {filter_clause}
        }}
        ORDER BY DESC(?timestamp)
        LIMIT {limit}
        """

        return await dkg.graph.query(query)


def create_snapshot_from_signal(signal_data: Dict[str, Any]) -> RegimeSnapshot:
    """
    Convert a signal response from /signal endpoint to a RegimeSnapshot.

    Args:
        signal_data: Response from the /signal endpoint including market_data

    Returns:
        RegimeSnapshot ready for DKG publishing
    """
    market_data = signal_data.get("market_data", {})
    features = market_data.get("features", {})

    return RegimeSnapshot(
        regime=market_data.get("regime", "Unknown"),
        regime_binary=1 if signal_data.get("risk_on", False) else 0,
        confidence=signal_data.get("confidence", 0) / 100.0,  # Convert from 0-100 to 0-1
        risk_on_probability=market_data.get("risk_on_probability", 0),
        timestamp=signal_data.get("last_updated", 0),
        date=market_data.get("date", datetime.now(timezone.utc).isoformat()),
        snapshot_hash=signal_data.get("snapshot_hash", ""),
        signature=signal_data.get("signature", ""),
        signer_address=signal_data.get("signer_address", ""),
        transaction_hash=signal_data.get("transaction_hash"),
        features=features,
    )
