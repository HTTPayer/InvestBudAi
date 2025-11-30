"""
InvestBud AI Graph Confirming Node

Independent verifier for DKG oracle snapshots. Validates predictions by:
1. Fetching snapshots from OriginTrail DKG
2. Computing independent predictions (via API or local model)
3. Publishing confirmation/challenge reports to DKG

Usage:
    gcn --verify-ual <UAL>           # Verify specific snapshot
    gcn --stats                       # Show verification statistics
    gcn                               # Run continuous verification loop
"""

from .dkg_service import DKGService, RegimeSnapshot, MACROCRYPTO_CONTEXT
from .confirming_node import (
    GraphConfirmingNode,
    VerificationStore,
    VerificationResult,
    VerificationReport,
)

__version__ = "0.1.0"
__all__ = [
    "DKGService",
    "RegimeSnapshot",
    "MACROCRYPTO_CONTEXT",
    "GraphConfirmingNode",
    "VerificationStore",
    "VerificationResult",
    "VerificationReport",
]
