// SPDX-License-Identifier: MIT
pragma solidity ^0.8.21;

contract MacroSignalOracle {
    address public trustedSigner; // Your service key
    uint256 public lastUpdated;

    struct Signal {
        bool riskOn; // true = risk-on, false = risk-off
        uint8 confidence; // 0–100
        bytes32 snapshotHash;
        uint256 timestamp;
    }

    Signal public latest;

    event SignalUpdated(
        bool riskOn,
        uint8 confidence,
        bytes32 snapshotHash,
        uint256 timestamp
    );

    constructor(address _trustedSigner) {
        trustedSigner = _trustedSigner;
    }

    function updateSignal(
        bool riskOn,
        uint8 confidence,
        bytes32 snapshotHash,
        uint256 timestamp,
        bytes calldata signature
    ) external {
        require(
            _verify(riskOn, confidence, snapshotHash, timestamp, signature),
            "Invalid signature"
        );

        latest = Signal(riskOn, confidence, snapshotHash, timestamp);
        lastUpdated = block.timestamp;

        emit SignalUpdated(riskOn, confidence, snapshotHash, timestamp);
    }

    function _verify(
        bool riskOn,
        uint8 confidence,
        bytes32 snapshotHash,
        uint256 timestamp,
        bytes calldata signature
    ) internal view returns (bool) {
        bytes32 digest = keccak256(
            abi.encodePacked(riskOn, confidence, snapshotHash, timestamp)
        );

        address recovered = _recover(digest, signature);
        return recovered == trustedSigner;
    }

    function _recover(
        bytes32 digest,
        bytes calldata signature
    ) internal pure returns (address) {
        return recoverSigner(digest, signature);
    }

    // ECDSA helper
    function recoverSigner(
        bytes32 message,
        bytes memory sig
    ) internal pure returns (address) {
        if (sig.length != 65) return address(0);

        bytes32 r;
        bytes32 s;
        uint8 v;
        (r, s, v) = split(sig);

        return ecrecover(message, v, r, s);
    }

    function split(
        bytes memory sig
    ) internal pure returns (bytes32 r, bytes32 s, uint8 v) {
        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }
    }
}
