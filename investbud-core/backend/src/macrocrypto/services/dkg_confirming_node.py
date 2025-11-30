"""
DKG Graph Confirming Node

Independent node that verifies MacroCrypto oracle snapshots by:
1. Fetching the latest snapshot from DKG
2. Running the same computation independently
3. Comparing results and publishing confirmation/challenge

This creates a trust network where multiple nodes verify the same predictions.

Architecture:
    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  Primary Oracle │────▶│      DKG        │◀────│ Confirming Node │
    │  (MacroCrypto)  │     │ Knowledge Graph │     │   (Verifier)    │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
           │                       │                       │
           ▼                       ▼                       ▼
    Publishes Snapshot    Stores & Replicates    Confirms/Challenges
"""

import os
import json
import asyncio
import requests
import sqlite3
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

from .dkg_service import (
    DKGService,
    AsyncDKGService,
    RegimeSnapshot,
    MACROCRYPTO_CONTEXT,
    create_snapshot_from_signal,
)


# ============================================================================
# Persistent Storage for Verified Snapshots
# ============================================================================

class VerificationStore:
    """SQLite-based persistent storage for tracking verified snapshots and reports."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to cache directory
            cache_dir = Path(os.getenv("CACHE_DIR", "cache"))
            cache_dir.mkdir(exist_ok=True)
            db_path = str(cache_dir / "confirming_node.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verified_snapshots (
                    snapshot_ual TEXT PRIMARY KEY,
                    verification_ual TEXT,
                    result TEXT,
                    computed_regime TEXT,
                    computed_confidence REAL,
                    regime_matches INTEGER,
                    confidence_difference REAL,
                    verified_at TEXT,
                    verifier_address TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verification_reports (
                    verification_ual TEXT PRIMARY KEY,
                    snapshot_ual TEXT,
                    result TEXT,
                    full_report TEXT,
                    published_at TEXT,
                    FOREIGN KEY (snapshot_ual) REFERENCES verified_snapshots(snapshot_ual)
                )
            """)
            conn.commit()

    def is_verified(self, snapshot_ual: str) -> bool:
        """Check if a snapshot has already been verified."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM verified_snapshots WHERE snapshot_ual = ?",
                (snapshot_ual,)
            )
            return cursor.fetchone() is not None

    def get_verified_uals(self) -> List[str]:
        """Get list of all verified snapshot UALs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT snapshot_ual FROM verified_snapshots")
            return [row[0] for row in cursor.fetchall()]

    def save_verification(
        self,
        snapshot_ual: str,
        verification_ual: str,
        report: "VerificationReport"
    ):
        """Save a verification result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO verified_snapshots
                (snapshot_ual, verification_ual, result, computed_regime,
                 computed_confidence, regime_matches, confidence_difference,
                 verified_at, verifier_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_ual,
                verification_ual,
                report.verification_result.value,
                report.computed_regime,
                report.computed_confidence,
                1 if report.regime_matches else 0,
                report.confidence_difference,
                report.timestamp,
                report.verifier_address
            ))

            conn.execute("""
                INSERT OR REPLACE INTO verification_reports
                (verification_ual, snapshot_ual, result, full_report, published_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                verification_ual,
                snapshot_ual,
                report.verification_result.value,
                json.dumps(report.to_json_ld()),
                report.timestamp
            ))
            conn.commit()

    def get_verification_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent verification history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM verified_snapshots
                ORDER BY verified_at DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_reports_by_result(self, result: str) -> List[Dict[str, Any]]:
        """Get all verifications with a specific result (confirmed/challenged/etc)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM verified_snapshots WHERE result = ?",
                (result,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM verified_snapshots").fetchone()[0]
            confirmed = conn.execute(
                "SELECT COUNT(*) FROM verified_snapshots WHERE result = 'confirmed'"
            ).fetchone()[0]
            challenged = conn.execute(
                "SELECT COUNT(*) FROM verified_snapshots WHERE result = 'challenged'"
            ).fetchone()[0]
            partial = conn.execute(
                "SELECT COUNT(*) FROM verified_snapshots WHERE result = 'partial'"
            ).fetchone()[0]
            errors = conn.execute(
                "SELECT COUNT(*) FROM verified_snapshots WHERE result = 'error'"
            ).fetchone()[0]

            return {
                "total_verified": total,
                "confirmed": confirmed,
                "challenged": challenged,
                "partial": partial,
                "errors": errors,
                "confirmation_rate": confirmed / total if total > 0 else 0,
            }


# ============================================================================
# Verification Types
# ============================================================================

class VerificationResult(Enum):
    """Result of snapshot verification."""
    CONFIRMED = "confirmed"      # Prediction matches independently
    CHALLENGED = "challenged"    # Prediction differs significantly
    PARTIAL = "partial"          # Some features match, some differ
    STALE_DATA = "stale_data"    # Unable to fetch current data for verification
    ERROR = "error"              # Verification failed


@dataclass
class VerificationReport:
    """Detailed verification report for a snapshot."""
    snapshot_ual: str
    original_snapshot: Dict[str, Any]
    verification_result: VerificationResult
    timestamp: str
    verifier_address: str

    # Computed values from verifier
    computed_regime: Optional[str] = None
    computed_confidence: Optional[float] = None
    computed_risk_on_probability: Optional[float] = None

    # Comparison metrics
    regime_matches: bool = False
    confidence_difference: float = 0.0
    feature_differences: Dict[str, float] = field(default_factory=dict)

    # Thresholds used
    confidence_threshold: float = 0.1  # 10% tolerance
    feature_threshold: float = 0.05    # 5% tolerance for features

    error_message: Optional[str] = None

    def to_json_ld(self) -> Dict[str, Any]:
        """Convert verification report to JSON-LD for DKG publishing."""
        report_id = f"https://macrocrypto.io/verifications/{self.timestamp.replace(':', '-')}"

        assertion = {
            **MACROCRYPTO_CONTEXT,
            "@id": report_id,
            "@type": ["mc:VerificationReport", "Dataset"],
            "name": f"Verification Report for {self.snapshot_ual}",
            "dateCreated": self.timestamp,

            # Reference to original snapshot
            "mc:verifies": self.snapshot_ual,

            # Verification outcome
            "mc:verificationResult": self.verification_result.value,
            "mc:verifierAddress": self.verifier_address,

            # Computed values
            "mc:computedRegime": self.computed_regime,
            "mc:computedConfidence": self.computed_confidence,
            "mc:computedRiskOnProbability": self.computed_risk_on_probability,

            # Comparison
            "mc:regimeMatches": self.regime_matches,
            "mc:confidenceDifference": round(self.confidence_difference, 4),
        }

        # Add relationship based on result
        if self.verification_result == VerificationResult.CONFIRMED:
            assertion["confirmedBy"] = self.verifier_address
        elif self.verification_result == VerificationResult.CHALLENGED:
            assertion["challenges"] = self.snapshot_ual

        if self.error_message:
            assertion["mc:errorMessage"] = self.error_message

        return assertion


class GraphConfirmingNode:
    """
    Independent node that verifies and confirms MacroCrypto oracle snapshots.

    This node:
    1. Polls the DKG for new snapshots
    2. Independently computes the same prediction
    3. Compares results and publishes confirmation/challenge
    4. Builds trust by creating verifiable audit trail
    """

    def __init__(
        self,
        dkg_service: DKGService,
        verifier_address: str,
        classifier=None,
        data_pipeline=None,
        confidence_threshold: float = 0.1,
        feature_threshold: float = 0.05,
        store: VerificationStore = None,
    ):
        """
        Initialize confirming node.

        Args:
            dkg_service: DKG service for querying and publishing
            verifier_address: Ethereum address of this verifier
            classifier: MacroRegimeClassifier instance (or None to create)
            data_pipeline: CombinedDataPipeline instance (or None to create)
            confidence_threshold: Max allowed confidence difference (0.1 = 10%)
            feature_threshold: Max allowed feature difference (0.05 = 5%)
            store: VerificationStore for persistent tracking (or None to create)
        """
        self.dkg = dkg_service
        self.verifier_address = verifier_address
        self.confidence_threshold = confidence_threshold
        self.feature_threshold = feature_threshold

        # Lazy-load classifier and data pipeline
        self._classifier = classifier
        self._data_pipeline = data_pipeline

        # Persistent store for verified snapshots
        self.store = store or VerificationStore()
        print(f"[OK] Confirming node: Using store at {self.store.db_path}")

    def _get_classifier(self):
        """Get or create the regime classifier."""
        if self._classifier is None:
            from ..models import MacroRegimeClassifier

            # Load pre-trained model (regime_classifier.pkl, NOT live_agent_decisions.pkl)
            model_path = os.getenv("MODEL_PATH", "models/regime_classifier.pkl")
            self._classifier = MacroRegimeClassifier()
            self._classifier.load(model_path)
            print(f"[OK] Confirming node: Classifier loaded from {model_path}")
        return self._classifier

    def _get_data_pipeline(self):
        """Get or create the data pipeline."""
        if self._data_pipeline is None:
            from ..data import CombinedDataPipeline
            self._data_pipeline = CombinedDataPipeline()
            print("[OK] Confirming node: Data pipeline initialized")
        return self._data_pipeline

    def fetch_latest_data(self) -> Dict[str, Any]:
        """
        Fetch latest macro and crypto data for independent verification.

        Returns:
            Dict with current market data
        """
        pipeline = self._get_data_pipeline()

        # Fetch fresh data
        df = pipeline.get_latest_data()

        if df.empty:
            raise RuntimeError("Failed to fetch market data")

        # Get latest row
        latest = df.iloc[-1].to_dict()

        return {
            "date": str(df.index[-1]),
            "features": latest,
        }

    def compute_prediction(self) -> Tuple[str, float, float, Dict[str, Any]]:
        """
        Independently compute regime prediction.

        Returns:
            Tuple of (regime, confidence, risk_on_probability, features)
        """
        classifier = self._get_classifier()

        # Get prediction
        result = classifier.predict_current_regime(verbose=False)
        print(f'[compute_prediction] result: {result}')

        return (
            result["regime"],
            result["confidence"],
            result["risk_on_probability"],
            result["features"],
        )

    def verify_snapshot(self, snapshot_ual: str) -> VerificationReport:
        """
        Verify a snapshot by independently computing the prediction.

        Args:
            snapshot_ual: UAL of the snapshot to verify

        Returns:
            VerificationReport with comparison results
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # 1. Fetch the original snapshot from DKG
            print(f"\n[DEBUG] Fetching snapshot: {snapshot_ual}")
            original = self.dkg.get_snapshot(snapshot_ual)

            print(f"[DEBUG] Raw DKG response keys: {original.keys() if isinstance(original, dict) else type(original)}")
            print(f"[DEBUG] Raw DKG response: {json.dumps(original, indent=2, default=str)[:2000]}")

            public = original.get("public", {})
            print(f"[DEBUG] Public assertion keys: {public.keys() if isinstance(public, dict) else type(public)}")

            original_regime = public.get("regime")
            original_confidence = public.get("confidence", 0)
            original_risk_on_prob = public.get("riskOnProbability", 0)

            print(f"[DEBUG] Original values from DKG:")
            print(f"  - regime: {original_regime} (type: {type(original_regime)})")
            print(f"  - confidence: {original_confidence} (type: {type(original_confidence)})")
            print(f"  - riskOnProbability: {original_risk_on_prob} (type: {type(original_risk_on_prob)})")

            # 2. Independently compute prediction
            print(f"\n[DEBUG] Computing independent prediction...")
            computed_regime, computed_confidence, computed_risk_on_prob, computed_features = \
                self.compute_prediction()

            print(f"[DEBUG] Computed values:")
            print(f"  - regime: {computed_regime} (type: {type(computed_regime)})")
            print(f"  - confidence: {computed_confidence} (type: {type(computed_confidence)})")
            print(f"  - risk_on_probability: {computed_risk_on_prob} (type: {type(computed_risk_on_prob)})")

            print(f"\n[DEBUG] Comparison:")
            print(f"  - Regime match: {original_regime} == {computed_regime} ? {original_regime == computed_regime}")
            print(f"  - Confidence diff: |{original_confidence} - {computed_confidence}| = {abs(original_confidence - computed_confidence)}")

            # 3. Compare results
            regime_matches = (original_regime == computed_regime)
            confidence_diff = abs(original_confidence - computed_confidence)

            # Compare features
            feature_diffs = {}
            original_features = {
                "btcPrice": public.get("btcPrice"),
                "btcRsi": public.get("btcRsi"),
                "vix": public.get("vix"),
                "fedFunds": public.get("fedFunds"),
            }
            computed_feature_map = {
                "btcPrice": computed_features.get("btc_price"),
                "btcRsi": computed_features.get("btc_rsi"),
                "vix": computed_features.get("vix"),
                "fedFunds": computed_features.get("fed_funds"),
            }
            for key in original_features:
                orig_val = original_features.get(key)
                comp_val = computed_feature_map.get(key)
                if orig_val is not None and comp_val is not None and orig_val != 0:
                    feature_diffs[key] = abs(orig_val - comp_val) / abs(orig_val)

            # 4. Determine verification result
            if regime_matches and confidence_diff <= self.confidence_threshold:
                result = VerificationResult.CONFIRMED
            elif not regime_matches:
                result = VerificationResult.CHALLENGED
            elif confidence_diff > self.confidence_threshold:
                result = VerificationResult.PARTIAL
            else:
                result = VerificationResult.CONFIRMED

            return VerificationReport(
                snapshot_ual=snapshot_ual,
                original_snapshot=public,
                verification_result=result,
                timestamp=timestamp,
                verifier_address=self.verifier_address,
                computed_regime=computed_regime,
                computed_confidence=computed_confidence,
                computed_risk_on_probability=computed_risk_on_prob,
                regime_matches=regime_matches,
                confidence_difference=confidence_diff,
                feature_differences=feature_diffs,
                confidence_threshold=self.confidence_threshold,
                feature_threshold=self.feature_threshold,
            )

        except Exception as e:
            return VerificationReport(
                snapshot_ual=snapshot_ual,
                original_snapshot={},
                verification_result=VerificationResult.ERROR,
                timestamp=timestamp,
                verifier_address=self.verifier_address,
                error_message=str(e),
            )

    def publish_verification(self, report: VerificationReport) -> Dict[str, Any]:
        """
        Publish verification report to DKG and save to persistent store.

        Args:
            report: VerificationReport to publish

        Returns:
            DKG publish result with UAL
        """
        content = {
            "public": report.to_json_ld(),
            "private": {
                "@context": MACROCRYPTO_CONTEXT["@context"],
                "@id": report.to_json_ld()["@id"],
                "@type": "mc:VerificationReportPrivate",
                "fullFeatureDifferences": report.feature_differences,
            }
        }

        result = self.dkg._get_client().asset.create(
            content=content,
            options={
                "epochs_num": 2,
                "minimum_number_of_finalization_confirmations": 1,
                "minimum_number_of_node_replications": 1,
            }
        )

        # Save to persistent store
        verification_ual = result.get("UAL", "")
        self.store.save_verification(report.snapshot_ual, verification_ual, report)

        return result

    def is_already_verified(self, snapshot_ual: str) -> bool:
        """Check if a snapshot has already been verified (persisted across restarts)."""
        return self.store.is_verified(snapshot_ual)

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics from the store."""
        return self.store.get_stats()

    def get_verification_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent verification history."""
        return self.store.get_verification_history(limit)

    def verify_and_publish(self, snapshot_ual: str) -> Tuple[VerificationReport, Dict[str, Any]]:
        """
        Verify a snapshot and publish the verification report.

        Args:
            snapshot_ual: UAL of snapshot to verify

        Returns:
            Tuple of (VerificationReport, DKG publish result)
        """
        report = self.verify_snapshot(snapshot_ual)
        publish_result = self.publish_verification(report)
        return report, publish_result

    def run_verification_loop(
        self,
        oracle_api_url: str = "http://localhost:8000",
        poll_interval_seconds: int = 3600,  # 1 hour
        max_iterations: Optional[int] = None,
    ):
        """
        Run continuous verification loop.

        Polls the oracle API for new snapshots and verifies them.

        Args:
            oracle_api_url: Base URL of the MacroCrypto oracle API
            poll_interval_seconds: How often to check for new snapshots
            max_iterations: Maximum iterations (None for infinite)
        """
        import time

        iteration = 0
        last_timestamp = 0
        print(f"[OK] Starting verification loop")
        print(f"    Oracle API: {oracle_api_url}")
        print(f"    Poll interval: {poll_interval_seconds}s")

        while max_iterations is None or iteration < max_iterations:
            try:
                # Poll oracle API for new UALs
                response = requests.get(
                    f"{oracle_api_url}/dkg/uals",
                    params={"limit": 10, "since_timestamp": last_timestamp},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                uals = data.get("uals", [])
                if uals:
                    print(f"[...] Found {len(uals)} snapshot(s) to verify")

                for entry in uals:
                    ual = entry.get("ual")
                    timestamp = entry.get("timestamp", 0)

                    # Check persistent store instead of in-memory list
                    if ual and not self.store.is_verified(ual):
                        print(f"[...] Verifying: {ual}")
                        report, result = self.verify_and_publish(ual)

                        if report.error_message:
                            print(f"[!] Error: {report.error_message}")
                        else:
                            verification_ual = result.get("UAL", "unknown")
                            print(f"[OK] Result: {report.verification_result.value.upper()}")
                            print(f"    Regime match: {report.regime_matches}")
                            print(f"    Confidence diff: {report.confidence_difference:.4f}")
                            print(f"    Published as: {verification_ual}")

                        # Update last timestamp
                        if timestamp > last_timestamp:
                            last_timestamp = timestamp
                    elif ual:
                        print(f"[SKIP] Already verified: {ual}")

            except requests.RequestException as e:
                print(f"[!] Failed to poll oracle API: {e}")
            except Exception as e:
                print(f"[!] Verification loop error: {e}")

            iteration += 1
            if max_iterations is None or iteration < max_iterations:
                time.sleep(poll_interval_seconds)


class AsyncGraphConfirmingNode:
    """Async version of GraphConfirmingNode for use with FastAPI."""

    def __init__(
        self,
        dkg_service: AsyncDKGService,
        verifier_address: str,
        classifier=None,
        data_pipeline=None,
        confidence_threshold: float = 0.1,
    ):
        self.dkg = dkg_service
        self.verifier_address = verifier_address
        self.confidence_threshold = confidence_threshold
        self._classifier = classifier
        self._data_pipeline = data_pipeline
        self.verified_snapshots: List[str] = []

    async def verify_snapshot(self, snapshot_ual: str) -> VerificationReport:
        """Async verify a snapshot."""
        # Similar implementation but using async methods
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            original = await self.dkg.get_snapshot(snapshot_ual)
            public = original.get("public", {})

            # Get classifier (sync operation, could be improved)
            if self._classifier is None:
                from ..models import MacroRegimeClassifier
                self._classifier = MacroRegimeClassifier()

            result = self._classifier.predict_current_regime(verbose=False)

            regime_matches = (public.get("regime") == result["regime"])
            confidence_diff = abs(public.get("confidence", 0) - result["confidence"])

            if regime_matches and confidence_diff <= self.confidence_threshold:
                verification_result = VerificationResult.CONFIRMED
            elif not regime_matches:
                verification_result = VerificationResult.CHALLENGED
            else:
                verification_result = VerificationResult.PARTIAL

            return VerificationReport(
                snapshot_ual=snapshot_ual,
                original_snapshot=public,
                verification_result=verification_result,
                timestamp=timestamp,
                verifier_address=self.verifier_address,
                computed_regime=result["regime"],
                computed_confidence=result["confidence"],
                computed_risk_on_probability=result["risk_on_probability"],
                regime_matches=regime_matches,
                confidence_difference=confidence_diff,
            )

        except Exception as e:
            return VerificationReport(
                snapshot_ual=snapshot_ual,
                original_snapshot={},
                verification_result=VerificationResult.ERROR,
                timestamp=timestamp,
                verifier_address=self.verifier_address,
                error_message=str(e),
            )


# ============================================================================
# Standalone Confirming Node Runner
# ============================================================================

def run_confirming_node(
    node_endpoint: str = None,
    oracle_api_url: str = None,
    verifier_address: str = None,
    poll_interval: int = None,
):
    """
    Run a standalone confirming node.

    Usage:
        python -m macrocrypto.services.dkg_confirming_node

    Environment variables:
        DKG_NODE_ENDPOINT: DKG node endpoint (default: http://localhost:8900)
        ORACLE_API_URL: MacroCrypto oracle API URL (default: http://localhost:8000)
        VERIFIER_ADDRESS: Your verifier Ethereum address
        ORACLE_SIGNER_ADDRESS: Official oracle signer address to verify

    Or from code:
        from macrocrypto.services.dkg_confirming_node import run_confirming_node
        run_confirming_node()
    """
    if node_endpoint is None:
        node_endpoint = os.getenv("DKG_NODE_ENDPOINT", "http://localhost:8900")

    if oracle_api_url is None:
        oracle_api_url = os.getenv("ORACLE_API_URL", "http://localhost:8015")

    if verifier_address is None:
        verifier_address = os.getenv("VERIFIER_ADDRESS", "0x0000000000000000000000000000000000000000")

    if poll_interval is None:
        poll_interval = int(os.getenv("POLL_INTERVAL", "3600"))

    oracle_address = os.getenv("ORACLE_SIGNER_ADDRESS", "")

    # Initialize store first to show stats
    store = VerificationStore()
    stats = store.get_stats()

    print("=" * 60)
    print("InvestBud AI Graph Confirming Node")
    print("=" * 60)
    print(f"Oracle API: {oracle_api_url}")
    print(f"DKG Node: {node_endpoint}")
    print(f"Verifier: {verifier_address}")
    print(f"Oracle Signer: {oracle_address or '(not set)'}")
    print(f"Poll Interval: {poll_interval}s")
    print(f"Store: {store.db_path}")
    print(f"Previously verified: {stats['total_verified']} ({stats['confirmed']} confirmed, {stats['challenged']} challenged)")
    print("=" * 60)

    dkg_service = DKGService(node_endpoint=node_endpoint)
    node = GraphConfirmingNode(
        dkg_service=dkg_service,
        verifier_address=verifier_address,
        store=store,
    )

    node.run_verification_loop(
        oracle_api_url=oracle_api_url,
        poll_interval_seconds=poll_interval,
    )


if __name__ == "__main__":
    import sys

    # Check for --stats flag to show verification statistics
    if "--stats" in sys.argv:
        print("=" * 60)
        print("Verification Statistics")
        print("=" * 60)

        store = VerificationStore()
        stats = store.get_stats()

        print(f"Total verified:     {stats['total_verified']}")
        print(f"Confirmed:          {stats['confirmed']}")
        print(f"Challenged:         {stats['challenged']}")
        print(f"Partial:            {stats['partial']}")
        print(f"Errors:             {stats['errors']}")
        print(f"Confirmation rate:  {stats['confirmation_rate']:.1%}")
        print("=" * 60)

        # Show recent history
        history = store.get_verification_history(limit=10)
        if history:
            print("\nRecent Verifications:")
            for h in history:
                print(f"  {h['verified_at'][:19]} | {h['result'].upper():10} | {h['snapshot_ual'][:50]}...")
        sys.exit(0)

    # Check for --history flag to show verification history
    if "--history" in sys.argv:
        store = VerificationStore()
        history = store.get_verification_history(limit=50)

        print("=" * 80)
        print("Verification History")
        print("=" * 80)

        for h in history:
            print(f"{h['verified_at'][:19]} | {h['result'].upper():10} | Match: {bool(h['regime_matches'])} | Diff: {h['confidence_difference']:.4f}")
            print(f"  Snapshot: {h['snapshot_ual']}")
            print(f"  Verification: {h['verification_ual']}")
            print()
        sys.exit(0)

    # Check for --verify-ual flag to verify a specific UAL
    if "--verify-ual" in sys.argv:
        idx = sys.argv.index("--verify-ual")
        if idx + 1 < len(sys.argv):
            ual = sys.argv[idx + 1]
            print("=" * 60)
            print("MacroCrypto Graph Confirming Node - Single Verification")
            print("=" * 60)

            node_endpoint = os.getenv("DKG_NODE_ENDPOINT", "http://localhost:8900")
            verifier_address = os.getenv("VERIFIER_ADDRESS", "0x0")

            print(f"DKG Node: {node_endpoint}")
            print(f"Verifier: {verifier_address}")
            print(f"UAL: {ual}")
            print("=" * 60)

            dkg_service = DKGService(node_endpoint=node_endpoint)
            node = GraphConfirmingNode(
                dkg_service=dkg_service,
                verifier_address=verifier_address,
            )

            print("[...] Fetching snapshot from DKG...")
            report = node.verify_snapshot(ual)

            print(f'report: {report}')

            print(f"\n{'=' * 60}")
            print("VERIFICATION RESULT")
            print("=" * 60)
            print(f"Result: {report.verification_result.value.upper()}")
            print(f"Original Regime: {report.original_snapshot.get('regime')}")
            print(f"Computed Regime: {report.computed_regime}")
            print(f"Regime Match: {report.regime_matches}")
            print(f"Original Confidence: {report.original_snapshot.get('confidence')}")
            print(f"Computed Confidence: {report.computed_confidence}")
            print(f"Confidence Diff: {report.confidence_difference:.4f}")
            if report.error_message:
                print(f"Error: {report.error_message}")
            print("=" * 60)

            # Ask if user wants to publish
            if report.verification_result != VerificationResult.ERROR:
                if "--publish" in sys.argv:
                    print("\n[...] Publishing verification report to DKG...")
                    result = node.publish_verification(report)
                    print(f"[OK] Published: {result.get('UAL')}")
                else:
                    print("\nRun with --publish to publish the verification report")
        else:
            print("Error: --verify-ual requires a UAL argument")
            print("Usage: python -m macrocrypto.services.dkg_confirming_node --verify-ual <UAL> [--publish]")
    else:
        run_confirming_node()
