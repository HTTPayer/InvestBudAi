"""
Graph Confirming Node

Independent verifier for InvestBud AI oracle snapshots. Validates predictions by:
1. Fetching the snapshot from DKG
2. Computing independent prediction (via API or local model)
3. Comparing results and publishing confirmation/challenge

Two verification modes:
- API mode: Fetches fresh prediction from oracle API (default, simpler)
- Local mode: Runs ML model locally (requires model file)
"""

import os
import json
import time
import sqlite3
import requests
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

from .dkg_service import DKGService, MACROCRYPTO_CONTEXT


# ============================================================================
# Persistent Storage
# ============================================================================

class VerificationStore:
    """SQLite-based persistent storage for tracking verified snapshots."""

    def __init__(self, db_path: str = None):
        if db_path is None:
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
                    published_at TEXT
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
    CONFIRMED = "confirmed"
    CHALLENGED = "challenged"
    PARTIAL = "partial"
    STALE_DATA = "stale_data"
    ERROR = "error"


@dataclass
class VerificationReport:
    """Detailed verification report for a snapshot."""
    snapshot_ual: str
    original_snapshot: Dict[str, Any]
    verification_result: VerificationResult
    timestamp: str
    verifier_address: str

    computed_regime: Optional[str] = None
    computed_confidence: Optional[float] = None
    computed_risk_on_probability: Optional[float] = None

    regime_matches: bool = False
    confidence_difference: float = 0.0
    feature_differences: Dict[str, float] = field(default_factory=dict)

    confidence_threshold: float = 0.1
    feature_threshold: float = 0.05

    error_message: Optional[str] = None

    def to_json_ld(self) -> Dict[str, Any]:
        """Convert verification report to JSON-LD for DKG publishing."""
        report_id = f"https://investbud.ai/verifications/{self.timestamp.replace(':', '-')}"

        assertion = {
            **MACROCRYPTO_CONTEXT,
            "@id": report_id,
            "@type": ["mc:VerificationReport", "Dataset"],
            "name": f"Verification Report for {self.snapshot_ual}",
            "dateCreated": self.timestamp,

            "mc:verifies": self.snapshot_ual,
            "mc:verificationResult": self.verification_result.value,
            "mc:verifierAddress": self.verifier_address,

            "mc:computedRegime": self.computed_regime,
            "mc:computedConfidence": self.computed_confidence,
            "mc:computedRiskOnProbability": self.computed_risk_on_probability,

            "mc:regimeMatches": self.regime_matches,
            "mc:confidenceDifference": round(self.confidence_difference, 4),
        }

        if self.verification_result == VerificationResult.CONFIRMED:
            assertion["confirmedBy"] = self.verifier_address
        elif self.verification_result == VerificationResult.CHALLENGED:
            assertion["challenges"] = self.snapshot_ual

        if self.error_message:
            assertion["mc:errorMessage"] = self.error_message

        return assertion


# ============================================================================
# Graph Confirming Node
# ============================================================================

class GraphConfirmingNode:
    """
    Independent node that verifies InvestBud AI oracle snapshots.

    Two modes:
    - API mode (default): Fetches fresh prediction from oracle API
    - Local mode: Runs ML model locally (requires model file)
    """

    def __init__(
        self,
        dkg_service: DKGService,
        verifier_address: str,
        oracle_api_url: str = None,
        confidence_threshold: float = 0.1,
        feature_threshold: float = 0.05,
        store: VerificationStore = None,
        use_local_model: bool = False,
    ):
        """
        Initialize confirming node.

        Args:
            dkg_service: DKG service for querying and publishing
            verifier_address: Ethereum address of this verifier
            oracle_api_url: Oracle API URL for API mode verification
            confidence_threshold: Max allowed confidence difference (0.1 = 10%)
            feature_threshold: Max allowed feature difference (0.05 = 5%)
            store: VerificationStore for persistent tracking
            use_local_model: If True, run ML model locally instead of API
        """
        self.dkg = dkg_service
        self.verifier_address = verifier_address
        self.oracle_api_url = oracle_api_url or os.getenv(
            "ORACLE_API_URL", "http://localhost:8015"
        )
        self.confidence_threshold = confidence_threshold
        self.feature_threshold = feature_threshold
        self.use_local_model = use_local_model

        self.store = store or VerificationStore()
        print(f"[OK] Store: {self.store.db_path}")

        # Lazy-load for local model mode
        self._classifier = None
        self._data_pipeline = None

    def compute_prediction_api(self) -> Tuple[str, float, float, Dict[str, Any]]:
        """
        Fetch current prediction from oracle API.

        Returns:
            Tuple of (regime, confidence, risk_on_probability, features)
        """
        response = requests.get(
            f"{self.oracle_api_url}/latest_report",
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        return (
            data.get("regime", "Unknown"),
            data.get("confidence", 0),
            data.get("risk_on_probability", 0),
            data.get("features", {}),
        )

    def compute_prediction_local(self) -> Tuple[str, float, float, Dict[str, Any]]:
        """
        Compute prediction using local ML model.

        Requires: pip install investbud-gcn[full]

        Returns:
            Tuple of (regime, confidence, risk_on_probability, features)
        """
        if self._classifier is None:
            try:
                import joblib
                model_path = os.getenv("MODEL_PATH", "models/regime_classifier.pkl")
                self._classifier = joblib.load(model_path)
                print(f"[OK] Model loaded: {model_path}")
            except ImportError:
                raise RuntimeError(
                    "Local model mode requires: pip install investbud-gcn[full]"
                )
            except FileNotFoundError:
                raise RuntimeError(
                    f"Model file not found: {model_path}. "
                    "Set MODEL_PATH env var or use API mode."
                )

        # This would need the full data pipeline - simplified for now
        raise NotImplementedError(
            "Local model mode not fully implemented. Use API mode instead."
        )

    def compute_prediction(self) -> Tuple[str, float, float, Dict[str, Any]]:
        """Compute prediction using configured mode."""
        if self.use_local_model:
            return self.compute_prediction_local()
        else:
            return self.compute_prediction_api()

    def verify_snapshot(self, snapshot_ual: str) -> VerificationReport:
        """
        Verify a snapshot by computing independent prediction.

        Args:
            snapshot_ual: UAL of the snapshot to verify

        Returns:
            VerificationReport with comparison results
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            # 1. Fetch the original snapshot from DKG
            print(f"[...] Fetching snapshot: {snapshot_ual}")
            original = self.dkg.get_snapshot(snapshot_ual)

            public = original.get("public", {})
            original_regime = public.get("regime")
            original_confidence = float(public.get("confidence", 0))

            print(f"[OK] Original: {original_regime} @ {original_confidence:.1%}")

            # 2. Compute independent prediction
            print(f"[...] Computing independent prediction...")
            computed_regime, computed_confidence, computed_risk_on_prob, computed_features = \
                self.compute_prediction()

            print(f"[OK] Computed: {computed_regime} @ {computed_confidence:.1%}")

            # 3. Compare results
            regime_matches = (original_regime == computed_regime)
            confidence_diff = abs(original_confidence - computed_confidence)

            print(f"[...] Regime match: {regime_matches}, Confidence diff: {confidence_diff:.4f}")

            # 4. Determine verification result
            if regime_matches and confidence_diff <= self.confidence_threshold:
                result = VerificationResult.CONFIRMED
            elif not regime_matches:
                result = VerificationResult.CHALLENGED
            else:
                result = VerificationResult.PARTIAL

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
                confidence_threshold=self.confidence_threshold,
                feature_threshold=self.feature_threshold,
            )

        except Exception as e:
            print(f"[!] Error: {e}")
            return VerificationReport(
                snapshot_ual=snapshot_ual,
                original_snapshot={},
                verification_result=VerificationResult.ERROR,
                timestamp=timestamp,
                verifier_address=self.verifier_address,
                error_message=str(e),
            )

    def publish_verification(self, report: VerificationReport) -> Dict[str, Any]:
        """Publish verification report to DKG."""
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

        verification_ual = result.get("UAL", "")
        self.store.save_verification(report.snapshot_ual, verification_ual, report)

        return result

    def verify_and_publish(self, snapshot_ual: str) -> Tuple[VerificationReport, Dict[str, Any]]:
        """Verify a snapshot and publish the report."""
        report = self.verify_snapshot(snapshot_ual)
        publish_result = self.publish_verification(report)
        return report, publish_result

    def run_verification_loop(
        self,
        poll_interval_seconds: int = 3600,
        max_iterations: Optional[int] = None,
    ):
        """
        Run continuous verification loop.

        Args:
            poll_interval_seconds: How often to check for new snapshots
            max_iterations: Maximum iterations (None for infinite)
        """
        iteration = 0
        last_timestamp = 0

        print(f"[OK] Starting verification loop")
        print(f"    Oracle API: {self.oracle_api_url}")
        print(f"    Poll interval: {poll_interval_seconds}s")

        while max_iterations is None or iteration < max_iterations:
            try:
                # Poll oracle API for new UALs
                response = requests.get(
                    f"{self.oracle_api_url}/dkg/uals",
                    params={"limit": 10, "since_timestamp": last_timestamp},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                uals = data.get("uals", [])
                if uals:
                    print(f"\n[...] Found {len(uals)} snapshot(s)")

                for entry in uals:
                    ual = entry.get("ual")
                    ts = entry.get("timestamp", 0)

                    if ual and not self.store.is_verified(ual):
                        print(f"\n[...] Verifying: {ual}")
                        report, result = self.verify_and_publish(ual)

                        if report.error_message:
                            print(f"[!] Error: {report.error_message}")
                        else:
                            print(f"[OK] Result: {report.verification_result.value.upper()}")
                            print(f"    Published: {result.get('UAL', 'unknown')}")

                        if ts > last_timestamp:
                            last_timestamp = ts
                    elif ual:
                        print(f"[SKIP] Already verified: {ual[:60]}...")

            except requests.RequestException as e:
                print(f"[!] API error: {e}")
            except Exception as e:
                print(f"[!] Loop error: {e}")

            iteration += 1
            if max_iterations is None or iteration < max_iterations:
                print(f"\n[...] Sleeping {poll_interval_seconds}s...")
                time.sleep(poll_interval_seconds)
