// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {IDiscaBridge, IDiscaConsumer} from "./IDiscaBridge.sol";
import {MessageHashUtils} from "./vendor/MessageHashUtils.sol";

/// @title DiscaBridge — settlement for off-chain FHE evaluation.
/// @notice Implements `docs/bridge.md` §2. A poster escrows funds against a
/// registered program; DISCA workers evaluate the circuit on ciphertexts
/// off-chain and each signs a claim about the result; the coordinator relays
/// those signatures here, and the contract recovers every signer itself.
///
/// @dev **What this contract verifies, precisely.** Not that the computation
/// was performed correctly — it cannot; it holds no keys and evaluates no
/// ciphertexts. It verifies that M distinct registered addresses each signed
/// "job `jobId`, running program `bytecodeHash`, produced a result sealing to
/// `resultHash`", and that the emitted blob is the one that hash covers.
/// Because FHE evaluation and compression are byte-reproducible under the
/// conditions in `docs/architecture.md` §3, agreement between M workers implies
/// correct evaluation with up to N-M Byzantine workers tolerated
/// (`docs/bridge.md` §2).
///
/// **Why signatures rather than an address list.** The superseded design took
/// `address[] attesters` from the coordinator. The contract could check those
/// addresses were registered and distinct and nothing else, so any M registered
/// addresses could be named beside any result hash, and the key holder — who
/// cannot tell a wrong plaintext from a right one — would never find out.
/// `docs/bridge.md` §2b records that in full. Recovering the signer is what
/// makes the attester set a fact rather than a formatting requirement.
contract DiscaBridge is IDiscaBridge {
    /// @dev Domain tag from `docs/bridge.md` §2a. 22 ASCII bytes, no length
    /// prefix. `primitives::attest::DOMAIN_RESULT` is the same literal, and the
    /// cross-language vector in `test/AttestationVector.t.sol` is what keeps
    /// them the same literal.
    string private constant DOMAIN_RESULT = "DISCA/attest/result/v1";

    /// @dev Upper bound on a low-`s` signature: half the secp256k1 group order.
    /// EIP-2 requires it, and `primitives::attest::recover` rejects the high
    /// form for the same reason — `(r, n - s, v ^ 1)` is a second valid
    /// signature over the same claim, so without this an attestation could be
    /// rewritten in flight without being invalidated.
    uint256 private constant SECP256K1_HALF_ORDER =
        0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0;

    /// @notice A program pinned on-chain by `registerProgram`.
    /// @dev `bytecodeHash` is in the signed preimage, so it is read from here
    /// during `fulfillJob` and never from calldata — the coordinator must not be
    /// able to choose which program a signature is checked against.
    struct Program {
        bytes32 bytecodeHash;
        bytes32 serverKeyHash;
        uint8 attestersRequired;
    }

    /// @notice Registry owner. Gates the worker registry and the coordinator
    /// address; `docs/bridge.md` §2 keeps both owner-gated for the demo.
    address public owner;

    /// @notice The address allowed to call `fulfillJob`, and paid the escrow on
    /// fulfillment.
    /// @dev `docs/bridge.md` §2 says `fulfillJob` is "callable by the
    /// coordinator" but does not say how that is enforced, and the difference
    /// matters: escrow pays whoever settles, so a permissionless `fulfillJob`
    /// lets anyone copy a pending transaction's calldata out of the mempool and
    /// collect the escrow for work they did not do. The attestations stay valid
    /// either way — §2a is explicit that an attestation is valid because of who
    /// signed it, not who relayed it — so this restriction is about payment,
    /// not about trust in the result.
    address public coordinator;

    /// @notice How long a job stays fulfillable, in seconds.
    /// @dev Per deployment rather than per job: a poster-chosen deadline could
    /// be set to zero, making every job refundable before the coordinator can
    /// possibly settle it, which is free griefing against workers who have
    /// already spent seconds of FHE evaluation on it.
    uint64 public immutable jobTimeout;

    /// @notice Whether an address's attestations count towards a quorum.
    mapping(address => bool) public isRegisteredWorker;

    /// @notice Programs by id. Ids start at 1 so that zero means "unregistered".
    mapping(uint256 => Program) public programs;

    /// @notice Number of registered programs; also the last id issued.
    uint256 public programCount;

    /// @notice Jobs by id. Ids start at 1 so that zero means "no such job".
    mapping(uint256 => Job) private _jobs;

    /// @notice Number of posted jobs; also the last id issued.
    uint256 public jobCount;

    error NotOwner();
    error NotCoordinator();
    error ZeroAddress();
    /// @dev A quorum of zero would be satisfied by an empty attestation array.
    error QuorumTooSmall();
    error UnknownProgram(uint256 programId);
    error UnknownJob(uint256 jobId);
    error JobNotOpen(uint256 jobId, JobState state);
    error JobExpired(uint256 jobId, uint64 deadline);
    error JobNotExpired(uint256 jobId, uint64 deadline);
    /// @dev `inputBlobs[i]` did not hash to `inputCommits[i]`.
    error InputCommitmentMismatch(uint256 index);
    error InputLengthMismatch(uint256 commits, uint256 blobs);
    /// @dev `docs/bridge.md` §5a: the emitted blob must be the one the workers
    /// attested to, or a coordinator could pair a real attestation with a
    /// substituted ciphertext.
    error ResultBlobMismatch();
    error BadRecoveryId(uint8 v);
    error HighS();
    /// @dev `ecrecover` returns `address(0)` on failure instead of reverting.
    error ZeroSigner(uint256 index);
    error NotRegisteredWorker(address recovered);
    /// @dev Attesters must arrive in strictly increasing address order, which
    /// makes distinctness an O(n) check with no storage (`docs/bridge.md` §2a
    /// step 4).
    error AttestersOutOfOrder(address previous, address recovered);
    error QuorumNotMet(uint256 supplied, uint8 required);
    error EscrowTransferFailed(address recipient, uint256 amount);

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    /// @param timeout Seconds a job stays fulfillable before `refundOnTimeout`
    /// opens. The demo runs against Anvil and an L2 (`docs/bridge.md` §7), where
    /// a job is settled in seconds; the value only has to exceed the worst-case
    /// evaluation plus dispatch time from `docs/architecture.md` §2.
    constructor(uint64 timeout) {
        owner = msg.sender;
        coordinator = msg.sender;
        jobTimeout = timeout;
        emit CoordinatorUpdated(msg.sender);
    }

    /// @notice Sets the address allowed to call `fulfillJob` and paid on
    /// fulfillment.
    /// @param newCoordinator The coordinator's address.
    function setCoordinator(address newCoordinator) external onlyOwner {
        if (newCoordinator == address(0)) revert ZeroAddress();
        coordinator = newCoordinator;
        emit CoordinatorUpdated(newCoordinator);
    }

    /// @inheritdoc IDiscaBridge
    /// @dev Deliberately not owner-gated. Registering a program pins three
    /// values and grants nothing: the only thing that decides whether a result
    /// settles is the worker registry, which is owner-gated. A program
    /// registered with a low `attestersRequired` weakens only the jobs posted
    /// against it, and those are posted by whoever chose it.
    function registerProgram(bytes32 bytecodeHash, bytes32 serverKeyHash, uint8 attestersRequired)
        external
        returns (uint256 programId)
    {
        if (attestersRequired == 0) revert QuorumTooSmall();

        programId = ++programCount;
        programs[programId] = Program({
            bytecodeHash: bytecodeHash,
            serverKeyHash: serverKeyHash,
            attestersRequired: attestersRequired
        });

        emit ProgramRegistered(programId, bytecodeHash, serverKeyHash, attestersRequired);
    }

    /// @inheritdoc IDiscaBridge
    /// @dev Owner-gated for the demo. `docs/bridge.md` §2 records what this
    /// registry will eventually have to record beyond the address: the CPU
    /// architecture and FFT plan a worker evaluates under, since a worker that
    /// differs on either disagrees with honest workers while behaving honestly
    /// (task 2.10b). Until then disagreement is evidence of divergence, not
    /// dishonesty, which is why nothing here slashes.
    function registerWorker(address worker) external onlyOwner {
        if (worker == address(0)) revert ZeroAddress();
        isRegisteredWorker[worker] = true;
        emit WorkerRegistered(worker);
    }

    /// @inheritdoc IDiscaBridge
    /// @dev The blobs are checked against the commitments here rather than
    /// taken on trust. Without it the commitments would be decorative: they are
    /// what `docs/bridge.md` §6 relies on to stop a substituted input, and a
    /// commitment nobody checks stops nothing. Hashing 2.3 KB per input
    /// (`docs/architecture.md` §2) costs ~6 gas per 32-byte word.
    function submitJob(
        uint256 programId,
        bytes32[] calldata inputCommits,
        bytes[] calldata inputBlobs,
        address callback
    ) external payable returns (uint256 jobId) {
        if (programs[programId].attestersRequired == 0) revert UnknownProgram(programId);
        if (inputCommits.length != inputBlobs.length) {
            revert InputLengthMismatch(inputCommits.length, inputBlobs.length);
        }

        for (uint256 i = 0; i < inputBlobs.length; ++i) {
            if (keccak256(inputBlobs[i]) != inputCommits[i]) revert InputCommitmentMismatch(i);
        }

        jobId = ++jobCount;
        Job storage job = _jobs[jobId];
        job.programId = programId;
        job.poster = msg.sender;
        job.callback = callback;
        job.inputCommits = inputCommits;
        job.deadline = uint64(block.timestamp) + jobTimeout;
        job.escrow = msg.value;
        job.state = JobState.Open;

        emit JobRequested(jobId, programId, inputCommits, inputBlobs, callback);
    }

    /// @inheritdoc IDiscaBridge
    /// @dev The steps are `docs/bridge.md` §2a's list, in its order. The one
    /// thing worth restating: the digest is built from the job's stored
    /// `programId -> bytecodeHash` and the `jobId`, never from anything the
    /// coordinator supplies, because a digest the caller can steer is a digest
    /// the caller can make any signature verify against.
    ///
    /// What this does *not* check is that the bridge dispatched to the
    /// recovered workers. It never knew who was dispatched to. An attestation
    /// counts because of who signed it.
    function fulfillJob(
        uint256 jobId,
        bytes32 resultHash,
        bytes calldata resultBlob,
        Attestation[] calldata attestations
    ) external {
        if (msg.sender != coordinator) revert NotCoordinator();

        Job storage job = _jobs[jobId];
        if (job.state == JobState.None) revert UnknownJob(jobId);
        if (job.state != JobState.Open) revert JobNotOpen(jobId, job.state);
        // Past the deadline the refund path is open, and a job that could be
        // both fulfilled and refunded is a race between the coordinator and the
        // poster over the same escrow.
        if (block.timestamp > job.deadline) revert JobExpired(jobId, job.deadline);

        // §2a step 1 / §5a. Doing this first means a substituted blob is
        // rejected before any signature work is paid for.
        if (keccak256(resultBlob) != resultHash) revert ResultBlobMismatch();

        Program storage program = programs[job.programId];
        // The preimage field is 8 bytes wide (`docs/bridge.md` §2a), and job
        // ids are a counter incremented once per `submitJob`, so the cast
        // cannot truncate a job that exists.
        bytes32 digest = attestationDigest(uint64(jobId), program.bytecodeHash, resultHash);

        // §2a step 4. Strictly increasing, so distinctness costs one comparison
        // per attester instead of a set: `address(0)` is below every valid
        // signer, so a duplicate or an out-of-order entry is caught by the same
        // check that starts the loop.
        address previous = address(0);
        for (uint256 i = 0; i < attestations.length; ++i) {
            Attestation calldata attestation = attestations[i];

            if (attestation.v != 27 && attestation.v != 28) revert BadRecoveryId(attestation.v);
            if (uint256(attestation.s) > SECP256K1_HALF_ORDER) revert HighS();

            address recovered = ecrecover(digest, attestation.v, attestation.r, attestation.s);

            // §2a step 3. `ecrecover` signals failure by returning zero rather
            // than reverting, and counting that as a signer is the standard way
            // to accept a forged signature. The ordering check below would also
            // catch it at i == 0, but only by accident of `previous` starting at
            // zero; this is the rule §2a states, stated.
            if (recovered == address(0)) revert ZeroSigner(i);
            if (recovered <= previous) revert AttestersOutOfOrder(previous, recovered);
            if (!isRegisteredWorker[recovered]) revert NotRegisteredWorker(recovered);

            previous = recovered;
        }

        // §2a step 5. Every supplied attestation had to pass, so the array
        // length is the count of distinct registered signers.
        if (attestations.length < program.attestersRequired) {
            revert QuorumNotMet(attestations.length, program.attestersRequired);
        }

        // Effects before interactions: the state change lands before the
        // callback and the payout, so neither can re-enter into a second
        // fulfillment.
        job.state = JobState.Fulfilled;
        emit JobFulfilled(jobId, resultHash, resultBlob);

        address callback = job.callback;
        if (callback != address(0)) {
            // A reverting consumer takes the whole settlement down with it, and
            // the job then follows the timeout path to a refund. That is
            // deliberate: swallowing the revert would release the escrow while
            // the contract that paid for the computation believes nothing
            // happened, and the consumer is chosen by the poster, so the cost of
            // a broken one falls where the choice was made.
            IDiscaConsumer(callback).onJobFulfilled(jobId, resultHash);
        }

        uint256 escrow = job.escrow;
        if (escrow > 0) {
            job.escrow = 0;
            (bool paid,) = coordinator.call{value: escrow}("");
            if (!paid) revert EscrowTransferFailed(coordinator, escrow);
        }
    }

    /// @inheritdoc IDiscaBridge
    /// @dev Permissionless: the money can only go to the poster, so there is
    /// nothing to gain by calling it on someone else's behalf, and a poster who
    /// is a contract without a keeper still gets refunded.
    ///
    /// This is the handler for every liveness failure in `docs/bridge.md` §6 —
    /// coordinator silence, a withheld result, and workers who never agree all
    /// end here, because none of them produce a quorum and the contract cannot
    /// tell them apart.
    function refundOnTimeout(uint256 jobId) external {
        Job storage job = _jobs[jobId];
        if (job.state == JobState.None) revert UnknownJob(jobId);
        if (job.state != JobState.Open) revert JobNotOpen(jobId, job.state);
        if (block.timestamp <= job.deadline) revert JobNotExpired(jobId, job.deadline);

        job.state = JobState.Refunded;
        uint256 escrow = job.escrow;
        job.escrow = 0;
        address poster = job.poster;

        emit JobRefunded(jobId, poster, escrow);

        if (escrow > 0) {
            (bool paid,) = poster.call{value: escrow}("");
            if (!paid) revert EscrowTransferFailed(poster, escrow);
        }
    }

    /// @notice The digest a worker signs for a successful evaluation.
    /// @param jobId The job the claim is about.
    /// @param bytecodeHash `keccak256` of the program's DISCA bytecode.
    /// @param resultHash `keccak256` of the compressed result ciphertext.
    /// @return The EIP-191 digest passed to `ecrecover`.
    /// @dev The 94-byte preimage of `docs/bridge.md` §2a, and the on-chain half
    /// of `primitives::attest::Claim::preimage`. Every field is fixed width, so
    /// concatenation is injective and no length prefixes are needed:
    /// `abi.encodePacked` writes the string literal without one and the
    /// `uint64` as 8 big-endian bytes.
    ///
    /// Public because it is the one thing in this contract that has to agree
    /// byte-for-byte with another language. `test/AttestationVector.t.sol`
    /// checks it against a vector generated by the Rust implementation, which is
    /// only possible if the digest can be computed from outside.
    function attestationDigest(uint64 jobId, bytes32 bytecodeHash, bytes32 resultHash)
        public
        pure
        returns (bytes32)
    {
        return MessageHashUtils.toEthSignedMessageHash(
            keccak256(attestationPreimage(jobId, bytecodeHash, resultHash))
        );
    }

    /// @notice The exact bytes hashed into the attestation digest.
    /// @param jobId The job the claim is about.
    /// @param bytecodeHash `keccak256` of the program's DISCA bytecode.
    /// @param resultHash `keccak256` of the compressed result ciphertext.
    /// @return The 94-byte preimage.
    /// @dev Split out from `attestationDigest` so the cross-language test can
    /// compare the preimage itself, not just the hash of it. A mismatch in the
    /// preimage and a mismatch in the prefix are different bugs and a single
    /// digest comparison cannot tell them apart.
    function attestationPreimage(uint64 jobId, bytes32 bytecodeHash, bytes32 resultHash)
        public
        pure
        returns (bytes memory)
    {
        return abi.encodePacked(DOMAIN_RESULT, jobId, bytecodeHash, resultHash);
    }

    /// @notice Reads a job.
    /// @param jobId The job id.
    /// @return The stored job, or a zero-valued struct in state `None` if there
    /// is no such job.
    /// @dev Hand-written rather than a public mapping because the automatic
    /// getter for a struct containing a dynamic array omits the array.
    function jobs(uint256 jobId) external view returns (Job memory) {
        return _jobs[jobId];
    }

    /// @notice The input commitments recorded for a job.
    /// @param jobId The job id.
    /// @return `keccak256` of each compressed input ciphertext, in submission
    /// order.
    function inputCommitsOf(uint256 jobId) external view returns (bytes32[] memory) {
        return _jobs[jobId].inputCommits;
    }
}
