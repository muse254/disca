// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Test} from "forge-std/Test.sol";
import {DiscaBridge} from "../src/DiscaBridge.sol";
import {IDiscaBridge} from "../src/IDiscaBridge.sol";

/// @notice Shared fixtures for the bridge suites.
/// @dev The worker keys here are the on-chain counterpart of
/// `primitives::attest::WorkerKey::derive`: deterministic, public, and derived
/// from a label, so a test and a Rust integration run can be talking about the
/// same address without any key material being plumbed between them.
abstract contract BridgeHarness is Test {
    /// @dev Measured in `docs/architecture.md` §2: a `CompressedFheInt32` that
    /// has been through the evaluator is 11.8 KB, five times a freshly
    /// encrypted input, because a fresh ciphertext compresses to a replayable
    /// PRNG seed and a computed one has to carry real coefficients. Every gas
    /// number in this suite is against a blob of this size; a smaller one would
    /// make `fulfillJob` look cheaper than it is.
    /// 11.8 KB read as KiB, i.e. 11.8 * 1024 rounded down.
    uint256 internal constant RESULT_BLOB_BYTES = 12083;

    /// @dev A freshly encrypted `CompressedFheInt32`, same source: 2.3 KiB.
    uint256 internal constant INPUT_BLOB_BYTES = 2355;

    uint64 internal constant JOB_TIMEOUT = 1 hours;

    DiscaBridge internal bridge;
    address internal coordinator = makeAddr("coordinator");
    address internal poster = makeAddr("poster");

    /// @dev Private keys, not addresses, because a test that needs an
    /// attestation needs to sign.
    uint256[] internal workerKeys;

    function setUp() public virtual {
        bridge = new DiscaBridge(JOB_TIMEOUT);
        bridge.setCoordinator(coordinator);

        for (uint256 i = 0; i < 8; ++i) {
            uint256 key = uint256(keccak256(abi.encodePacked("DISCA/test-worker/", i)));
            workerKeys.push(key);
            bridge.registerWorker(vm.addr(key));
        }

        vm.deal(poster, 100 ether);
    }

    /// @notice Registers a program requiring `m` attesters.
    function _program(uint8 m) internal returns (uint256 programId) {
        return bridge.registerProgram(
            keccak256(abi.encodePacked("bytecode", m)), keccak256("server-key"), m
        );
    }

    /// @notice Posts a job with `inputs` realistic input blobs.
    function _job(uint256 programId, uint256 inputs, address callback, uint256 escrow)
        internal
        returns (uint256 jobId)
    {
        bytes[] memory blobs = new bytes[](inputs);
        bytes32[] memory commits = new bytes32[](inputs);
        for (uint256 i = 0; i < inputs; ++i) {
            blobs[i] = _blob(INPUT_BLOB_BYTES, bytes32(i));
            commits[i] = keccak256(blobs[i]);
        }

        vm.prank(poster);
        return bridge.submitJob{value: escrow}(programId, commits, blobs, callback);
    }

    /// @notice Pseudo-random bytes of a given length.
    /// @dev Random rather than zeroed on purpose: zero calldata bytes cost 4
    /// gas and non-zero ones 16, so a zero-filled blob would understate
    /// `fulfillJob` by roughly 140k gas at 11.8 KB — most of what the blob
    /// costs.
    function _blob(uint256 length, bytes32 seed) internal pure returns (bytes memory out) {
        out = new bytes(length);
        bytes32 word = keccak256(abi.encodePacked("DISCA/test-blob/", seed));
        for (uint256 i = 0; i < length; i += 32) {
            word = keccak256(abi.encodePacked(word));
            uint256 remaining = length - i;
            uint256 span = remaining < 32 ? remaining : 32;
            for (uint256 j = 0; j < span; ++j) {
                // A byte of zero would be free calldata; nudge it off zero
                // rather than re-rolling, since the point is the cost profile,
                // not the distribution.
                uint8 b = uint8(word[j]);
                out[i + j] = bytes1(b == 0 ? 1 : b);
            }
        }
    }

    /// @notice One worker's attestation over the §2a claim for this job.
    function _attest(uint256 key, uint256 jobId, bytes32 bytecodeHash, bytes32 resultHash)
        internal
        pure
        returns (IDiscaBridge.Attestation memory)
    {
        bytes32 digest = _digest(jobId, bytecodeHash, resultHash);
        (uint8 v, bytes32 r, bytes32 s) = vm.sign(key, digest);
        return IDiscaBridge.Attestation({r: r, s: s, v: v});
    }

    /// @dev Recomputed here rather than read off the contract, so the tests are
    /// checking an independent construction of the digest rather than the
    /// contract against itself.
    function _digest(uint256 jobId, bytes32 bytecodeHash, bytes32 resultHash)
        internal
        pure
        returns (bytes32)
    {
        bytes32 inner = keccak256(
            abi.encodePacked("DISCA/attest/result/v1", uint64(jobId), bytecodeHash, resultHash)
        );
        return keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", inner));
    }

    /// @notice Attestations from the first `count` worker keys, in the strictly
    /// increasing address order `fulfillJob` requires.
    function _quorum(uint256 count, uint256 jobId, bytes32 bytecodeHash, bytes32 resultHash)
        internal
        view
        returns (IDiscaBridge.Attestation[] memory)
    {
        uint256[] memory keys = new uint256[](count);
        for (uint256 i = 0; i < count; ++i) {
            keys[i] = workerKeys[i];
        }
        return _sortedQuorum(keys, jobId, bytecodeHash, resultHash);
    }

    /// @notice The lowest address among `keys`, i.e. the one `fulfillJob`
    /// reaches first once the attestations are sorted.
    function _lowestSigner(uint256[] memory keys) internal pure returns (address lowest) {
        lowest = vm.addr(keys[0]);
        for (uint256 i = 1; i < keys.length; ++i) {
            address candidate = vm.addr(keys[i]);
            if (candidate < lowest) lowest = candidate;
        }
    }

    /// @notice Asserts that an attestation does not attribute to any registered
    /// worker, and returns the revert `fulfillJob` owes for it.
    /// @dev The assertion is the property under test; the returned revert data
    /// is only how the contract is required to react to it. Recovery always
    /// succeeds for well-formed scalars and yields *some* address, so there are
    /// two shapes of rejection — that address being unregistered, or the curve
    /// arithmetic failing and `ecrecover` returning zero — and a test that
    /// accepted "any revert" would not distinguish either from a bug.
    function _rejectionFor(bytes32 digest, IDiscaBridge.Attestation memory attestation)
        internal
        view
        returns (bytes memory)
    {
        address recovered = ecrecover(digest, attestation.v, attestation.r, attestation.s);
        assertFalse(
            recovered != address(0) && bridge.isRegisteredWorker(recovered),
            "the forged attestation recovered to a registered worker"
        );

        if (recovered == address(0)) {
            return abi.encodeWithSelector(DiscaBridge.ZeroSigner.selector, 0);
        }
        return abi.encodeWithSelector(DiscaBridge.NotRegisteredWorker.selector, recovered);
    }

    /// @notice Attestations from `keys`, sorted by signer address.
    /// @dev The off-chain coordinator sorts for the same reason the contract
    /// demands it: `docs/bridge.md` §2a step 4 makes distinctness an O(n) check
    /// by requiring ascending addresses, and the caller should pay for the sort
    /// rather than the chain.
    function _sortedQuorum(
        uint256[] memory keys,
        uint256 jobId,
        bytes32 bytecodeHash,
        bytes32 resultHash
    ) internal pure returns (IDiscaBridge.Attestation[] memory attestations) {
        // Insertion sort: the arrays here are single digits long.
        for (uint256 i = 1; i < keys.length; ++i) {
            uint256 key = keys[i];
            uint256 j = i;
            while (j > 0 && vm.addr(keys[j - 1]) > vm.addr(key)) {
                keys[j] = keys[j - 1];
                --j;
            }
            keys[j] = key;
        }

        attestations = new IDiscaBridge.Attestation[](keys.length);
        for (uint256 i = 0; i < keys.length; ++i) {
            attestations[i] = _attest(keys[i], jobId, bytecodeHash, resultHash);
        }
    }
}
