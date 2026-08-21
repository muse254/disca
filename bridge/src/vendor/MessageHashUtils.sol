// SPDX-License-Identifier: MIT
// OpenZeppelin Contracts (last updated v5.0.0) (utils/cryptography/MessageHashUtils.sol)
//
// Vendored from openzeppelin-contracts v5.1.0, file
// `contracts/utils/cryptography/MessageHashUtils.sol`, MIT licensed
// (https://github.com/OpenZeppelin/openzeppelin-contracts/blob/v5.1.0/LICENSE).
//
// Reduced to the single overload this repository uses. Dropped:
// `toEthSignedMessageHash(bytes memory)` (variable-length EIP-191 0x45, which
// pulls in Strings/Math/SignedMath for the decimal length), and
// `toDataWithIntendedValidatorHash` / `toTypedDataHash` (EIP-191 0x00 and
// EIP-712 0x01). docs/bridge.md §2a signs a fixed 32-byte payload under 0x45
// and says why EIP-712 is deferred, so the omitted code has no caller here and
// carrying it would mean carrying four more files for nothing.
//
// The body below is byte-for-byte upstream. It is vendored rather than pulled
// in as a dependency because it is the one piece of third-party code on the
// trust-critical path — `docs/bridge.md` §2a names this exact function as the
// definition of the digest — and a reader checking the contract against the
// Rust in `primitives/src/attest.rs` should be able to see it without leaving
// the repository.
pragma solidity ^0.8.20;

/**
 * @dev Signature message hash utilities for producing digests to be consumed by {ECDSA} recovery
 * or signing.
 */
library MessageHashUtils {
    /**
     * @dev Returns the keccak256 digest of an EIP-191 signed data with version
     * `0x45` (`personal_sign` messages).
     *
     * The digest is calculated by prefixing a bytes32 `messageHash` with
     * `"\x19Ethereum Signed Message:\n32"` and hashing the result. It corresponds with the
     * hash signed when using the https://eth.wiki/json-rpc/API#eth_sign[`eth_sign`] JSON-RPC method.
     *
     * NOTE: The `messageHash` parameter is intended to be the result of hashing a raw message with
     * keccak256, although any bytes32 value can be safely used because the final digest will
     * be re-hashed.
     *
     * See {ECDSA-recover}.
     */
    function toEthSignedMessageHash(bytes32 messageHash) internal pure returns (bytes32 digest) {
        assembly ("memory-safe") {
            mstore(0x00, "\x19Ethereum Signed Message:\n32") // 32 is the bytes-length of messageHash
            mstore(0x1c, messageHash) // 0x1c (28) is the length of the prefix
            digest := keccak256(0x00, 0x3c) // 0x3c is the length of the prefix (0x1c) + messageHash (0x20)
        }
    }
}
