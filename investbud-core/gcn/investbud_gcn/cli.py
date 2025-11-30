"""
CLI for InvestBud AI Graph Confirming Node

Usage:
    gcn                           # Run continuous verification loop
    gcn --verify-ual <UAL>        # Verify specific snapshot
    gcn --verify-ual <UAL> --publish  # Verify and publish to DKG
    gcn --stats                   # Show verification statistics
    gcn --history                 # Show verification history
"""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="InvestBud AI Graph Confirming Node - Independent verifier for DKG oracle snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gcn                                    Run continuous verification loop
  gcn --verify-ual did:dkg:otp:20430/... Verify a specific snapshot
  gcn --verify-ual ... --publish         Verify and publish result to DKG
  gcn --stats                            Show verification statistics
  gcn --history                          Show verification history
        """,
    )
    parser.add_argument("--verify-ual", metavar="UAL", help="Verify a specific snapshot UAL")
    parser.add_argument("--publish", action="store_true", help="Publish verification result to DKG")
    parser.add_argument("--stats", action="store_true", help="Show verification statistics")
    parser.add_argument("--history", action="store_true", help="Show verification history")

    args = parser.parse_args()

    from .dkg_service import DKGService
    from .confirming_node import GraphConfirmingNode, VerificationStore, VerificationResult

    # Configuration from environment
    node_endpoint = os.getenv("DKG_NODE_ENDPOINT", "http://localhost:8900")
    oracle_api_url = os.getenv("ORACLE_API_URL", "http://localhost:8015")
    verifier_address = os.getenv("VERIFIER_ADDRESS", "0x0000000000000000000000000000000000000000")
    oracle_signer = os.getenv("ORACLE_SIGNER_ADDRESS", "")
    poll_interval = int(os.getenv("POLL_INTERVAL", "3600"))

    # --stats: Show verification statistics
    if args.stats:
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

        history = store.get_verification_history(limit=10)
        if history:
            print("\nRecent Verifications:")
            for h in history:
                print(f"  {h['verified_at'][:19]} | {h['result'].upper():10} | {h['snapshot_ual'][:50]}...")
        return

    # --history: Show verification history
    if args.history:
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
        return

    # --verify-ual: Verify a specific UAL
    if args.verify_ual:
        ual = args.verify_ual

        print("=" * 60)
        print("InvestBud AI Graph Confirming Node")
        print("=" * 60)
        print(f"Mode: Single Verification")
        print(f"DKG Node: {node_endpoint}")
        print(f"Oracle API: {oracle_api_url}")
        print(f"Verifier: {verifier_address}")
        print(f"UAL: {ual}")
        print("=" * 60)

        dkg_service = DKGService(node_endpoint=node_endpoint)
        node = GraphConfirmingNode(
            dkg_service=dkg_service,
            verifier_address=verifier_address,
            oracle_api_url=oracle_api_url,
        )

        print("\n[...] Verifying snapshot...")
        report = node.verify_snapshot(ual)

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

        if report.verification_result != VerificationResult.ERROR:
            if args.publish:
                print("\n[...] Publishing verification report to DKG...")
                result = node.publish_verification(report)
                print(f"[OK] Published: {result.get('UAL')}")
            else:
                print("\nRun with --publish to publish the verification report")
        return

    # Default: Run continuous verification loop
    store = VerificationStore()
    stats = store.get_stats()

    print("=" * 60)
    print("InvestBud AI Graph Confirming Node")
    print("=" * 60)
    print(f"Mode: Continuous Verification")
    print(f"DKG Node: {node_endpoint}")
    print(f"Oracle API: {oracle_api_url}")
    print(f"Verifier: {verifier_address}")
    print(f"Oracle Signer: {oracle_signer or '(not set)'}")
    print(f"Poll Interval: {poll_interval}s")
    print(f"Store: {store.db_path}")
    print(f"Previously verified: {stats['total_verified']} ({stats['confirmed']} confirmed, {stats['challenged']} challenged)")
    print("=" * 60)

    dkg_service = DKGService(node_endpoint=node_endpoint)
    node = GraphConfirmingNode(
        dkg_service=dkg_service,
        verifier_address=verifier_address,
        oracle_api_url=oracle_api_url,
        store=store,
    )

    node.run_verification_loop(poll_interval_seconds=poll_interval)


if __name__ == "__main__":
    main()
