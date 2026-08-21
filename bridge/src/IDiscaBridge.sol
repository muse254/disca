// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

/// @title The settlement surface of the DISCA FHE coprocessor.
/// @notice Contracts request confidential computation, DISCA workers evaluate
/// it on ciphertexts, and a quorum of worker signatures over the result
/// commitment settles the job here. Specified in `docs/bridge.md` §2; the
/// attestation rules the implementation must enforce are §2a.
/// @dev The types live in the interface so a consumer (`CommitteeTally`) and an
/// off-chain watcher can both be written against one definition.
interface IDiscaBridge {
    /// @notice Lifecycle position of a job.
    /// @dev `None` is first so that the zero value of an unwritten mapping entry
    /// means "no such job" — an enum has no null, and without a reserved zero an
    /// unknown job id would read as `Open` and be indistinguishable from a real
    /// one.
    ///
    /// `Disputed` is listed in `docs/bridge.md` §2 and §6 but nothing in this
    /// contract sets it. Divergence between workers is not visible on-chain: it
    /// shows up as attestations over different result hashes, so no quorum
    /// forms, no `fulfillJob` succeeds, and the job takes the timeout path. §6's
    /// "worker hash mismatch" row therefore collapses onto its "coordinator goes
    /// silent" row today. The value is kept so that an off-chain dispute
    /// mechanism can be added without renumbering the states that are stored.
    enum JobState {
        None,
        Open,
        Fulfilled,
        Refunded,
        Disputed
    }

    /// @notice A request for computation and the escrow backing it.
    /// @param programId Which registered program (and therefore which bytecode
    /// hash and quorum size) the job runs under.
    /// @param poster Who paid the escrow, and who a timeout refunds.
    /// @param callback Consumer contract to notify on fulfillment, or zero.
    /// @param inputCommits `keccak256` of each compressed input ciphertext. The
    /// blobs themselves are event data only (`docs/bridge.md` §1).
    /// @param deadline Unix seconds after which the job can no longer be
    /// fulfilled and can be refunded.
    /// @param escrow Wei held for the coordinator.
    /// @param state Lifecycle position.
    struct Job {
        uint256 programId;
        address poster;
        address callback;
        bytes32[] inputCommits;
        uint64 deadline;
        uint256 escrow;
        JobState state;
    }

    /// @notice One worker's recoverable secp256k1 signature over the claim in
    /// `docs/bridge.md` §2a.
    /// @dev Split into `(r, s, v)` rather than 65 packed bytes because that is
    /// the shape `ecrecover` takes; `primitives::attest::Attestation` is the
    /// same three fields for the same reason. `v` is Ethereum's 27 or 28, never
    /// the bare 0/1 that k256 uses internally.
    struct Attestation {
        bytes32 r;
        bytes32 s;
        uint8 v;
    }

    /// @notice A job was posted. Workers read the input blobs from here.
    /// @dev The blobs are in event data rather than storage because on-chain
    /// availability is what proves the inputs were not substituted, and event
    /// data costs ~8 gas/byte against storage's ~20,000 per 32 bytes
    /// (`docs/bridge.md` §1).
    event JobRequested(
        uint256 indexed jobId,
        uint256 indexed programId,
        bytes32[] inputCommits,
        bytes[] inputBlobs,
        address callback
    );

    /// @notice A quorum attested to `resultHash` and the job settled.
    /// @dev `resultBlob` is emitted, not stored, and `fulfillJob` has already
    /// checked it hashes to `resultHash`. Emitting it is what makes escrow
    /// release atomic with result availability — option B in `docs/bridge.md`
    /// §5a, and the reason that section calls emission required rather than
    /// optional.
    event JobFulfilled(uint256 indexed jobId, bytes32 resultHash, bytes resultBlob);

    /// @notice A job passed its deadline without being fulfilled and the escrow
    /// went back to the poster (`docs/bridge.md` §6).
    event JobRefunded(uint256 indexed jobId, address indexed poster, uint256 escrow);

    /// @notice A program was pinned on-chain (`docs/bridge.md` §3 step 3).
    event ProgramRegistered(
        uint256 indexed programId,
        bytes32 bytecodeHash,
        bytes32 serverKeyHash,
        uint8 attestersRequired
    );

    /// @notice An address may now have its attestations counted.
    event WorkerRegistered(address indexed worker);

    /// @notice The address allowed to call `fulfillJob` and paid on fulfillment.
    event CoordinatorUpdated(address indexed coordinator);

    /// @notice Pins a program's bytecode hash, server key hash and quorum size.
    /// @param bytecodeHash `keccak256` of the DISCA bytecode.
    /// @param serverKeyHash `keccak256` of the compressed server key the
    /// coordinator serves at `GET /keys/<serverKeyHash>`.
    /// @param attestersRequired M, the number of distinct registered workers
    /// that must sign one result hash.
    /// @return programId Identifier to pass to `submitJob`.
    function registerProgram(bytes32 bytecodeHash, bytes32 serverKeyHash, uint8 attestersRequired)
        external
        returns (uint256 programId);

    /// @notice Adds an address whose attestations count towards a quorum.
    /// @param worker The worker's Ethereum address — the last 20 bytes of
    /// `keccak256` over its uncompressed public key, exactly as an EOA.
    function registerWorker(address worker) external;

    /// @notice Posts a job and escrows `msg.value` for the coordinator.
    /// @param programId A registered program.
    /// @param inputCommits `keccak256` of each compressed input ciphertext.
    /// @param inputBlobs The ciphertexts themselves, emitted for workers to
    /// fetch; positionally matched against `inputCommits`.
    /// @param callback Consumer to notify on fulfillment, or zero for none.
    /// @return jobId Identifier of the new job.
    function submitJob(
        uint256 programId,
        bytes32[] calldata inputCommits,
        bytes[] calldata inputBlobs,
        address callback
    ) external payable returns (uint256 jobId);

    /// @notice Settles a job against a quorum of worker signatures.
    /// @param jobId The open job.
    /// @param resultHash `keccak256` of the compressed result — what the workers
    /// signed.
    /// @param resultBlob The compressed result ciphertext itself.
    /// @param attestations One signature per attesting worker, ordered by
    /// strictly increasing recovered address.
    function fulfillJob(
        uint256 jobId,
        bytes32 resultHash,
        bytes calldata resultBlob,
        Attestation[] calldata attestations
    ) external;

    /// @notice Returns the escrow to the poster once the deadline has passed.
    /// @param jobId The open, expired job.
    function refundOnTimeout(uint256 jobId) external;
}

/// @title What a consumer contract must implement to be called back.
/// @notice `CommitteeTally` in `docs/bridge.md` §5 is the demo implementation.
interface IDiscaConsumer {
    /// @notice Called by the bridge when a job this contract posted settles.
    /// @param jobId The settled job.
    /// @param resultHash `keccak256` of the compressed result ciphertext. The
    /// plaintext is not on-chain and cannot be: only the key holder can decrypt.
    function onJobFulfilled(uint256 jobId, bytes32 resultHash) external;
}
