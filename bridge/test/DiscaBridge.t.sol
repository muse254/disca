// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {BridgeHarness} from "./BridgeHarness.sol";
import {DiscaBridge} from "../src/DiscaBridge.sol";
import {IDiscaBridge} from "../src/IDiscaBridge.sol";

/// @notice The attestation checks of `docs/bridge.md` §2a and the job state
/// machine of §2.
///
/// @dev The forgery tests come first because they are the ones the contract
/// exists for. §2b records the superseded design in which `fulfillJob` took an
/// address list: it passed every test that could be written about state
/// transitions and escrow, and was still worthless, because the coordinator
/// chose the attester array. A suite that only exercised the happy path would
/// not have noticed the difference.
contract DiscaBridgeTest is BridgeHarness {
    bytes internal resultBlob;
    bytes32 internal resultHash;

    function setUp() public override {
        super.setUp();
        resultBlob = _blob(RESULT_BLOB_BYTES, "result");
        resultHash = keccak256(resultBlob);
    }

    // ---------------------------------------------------------------------
    // Forgery. §2a, and the reason §2b exists.
    // ---------------------------------------------------------------------

    /// @dev The headline property. A coordinator that assembles a perfectly
    /// well-formed settlement — right job, right blob, right hash, the right
    /// *number* of signatures — but whose signers are keys the registry has
    /// never heard of must get a revert. Under the address-list design of §2b
    /// this transaction would have been indistinguishable from an honest one,
    /// because the coordinator supplied the names.
    function test_fulfillJob_revertsWhenTheAttesterSetIsForged() public {
        uint256 programId = _program(3);
        uint256 jobId = _job(programId, 3, address(0), 1 ether);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        // Keys that exist and can sign, but that nobody registered.
        uint256[] memory strangers = new uint256[](3);
        for (uint256 i = 0; i < 3; ++i) {
            strangers[i] = uint256(keccak256(abi.encodePacked("impostor", i)));
            assertFalse(bridge.isRegisteredWorker(vm.addr(strangers[i])));
        }

        IDiscaBridge.Attestation[] memory forged =
            _sortedQuorum(strangers, jobId, bytecodeHash, resultHash);

        vm.prank(coordinator);
        vm.expectRevert(
            abi.encodeWithSelector(
                DiscaBridge.NotRegisteredWorker.selector, _lowestSigner(strangers)
            )
        );
        bridge.fulfillJob(jobId, resultHash, resultBlob, forged);
    }

    /// @dev The cheaper forgery: do not sign at all, just put plausible bytes
    /// in `(r, s, v)`. `ecrecover` does not reject these — it returns whatever
    /// address the curve arithmetic implies, or zero — so the rejection has to
    /// come from the registry lookup, not from recovery failing.
    function test_fulfillJob_revertsOnFabricatedSignatureBytes() public {
        uint256 programId = _program(1);
        uint256 jobId = _job(programId, 1, address(0), 0);

        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        IDiscaBridge.Attestation[] memory fabricated = new IDiscaBridge.Attestation[](1);
        fabricated[0] = IDiscaBridge.Attestation({
            r: keccak256("not a signature"),
            // Forced low so the EIP-2 check is not what rejects this; the point
            // is that unsigned bytes do not recover to a registered worker.
            s: bytes32(uint256(keccak256("also not a signature")) >> 1),
            v: 27
        });

        bytes memory rejection =
            _rejectionFor(_digest(jobId, bytecodeHash, resultHash), fabricated[0]);

        vm.prank(coordinator);
        vm.expectRevert(rejection);
        bridge.fulfillJob(jobId, resultHash, resultBlob, fabricated);
    }

    /// @dev The replay the job id in the preimage exists to stop. Real workers,
    /// real signatures — for a different job. With a deterministic evaluator
    /// over a small result space two jobs producing the same bytes is the
    /// common case, not a coincidence, so without the job id these signatures
    /// would be reusable settlement for any job with the same answer.
    function test_fulfillJob_revertsWhenAttestationsAreForAnotherJob() public {
        uint256 programId = _program(1);
        uint256 first = _job(programId, 1, address(0), 0);
        uint256 second = _job(programId, 1, address(0), 0);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        // One registered worker's real signature, for the wrong job.
        IDiscaBridge.Attestation[] memory forOther = _quorum(1, first, bytecodeHash, resultHash);
        bytes memory rejection =
            _rejectionFor(_digest(second, bytecodeHash, resultHash), forOther[0]);

        vm.prank(coordinator);
        vm.expectRevert(rejection);
        bridge.fulfillJob(second, resultHash, resultBlob, forOther);
    }

    /// @dev Same job id, same answer, different circuit. Two programs can
    /// easily produce the same result bytes; the bytecode hash is what stops an
    /// attestation for a trivial circuit settling the circuit the job paid for.
    function test_fulfillJob_revertsWhenAttestationsAreForAnotherProgram() public {
        uint256 programId = _program(1);
        uint256 jobId = _job(programId, 1, address(0), 0);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        IDiscaBridge.Attestation[] memory forOther =
            _quorum(1, jobId, keccak256("some other circuit"), resultHash);
        bytes memory rejection =
            _rejectionFor(_digest(jobId, bytecodeHash, resultHash), forOther[0]);

        vm.prank(coordinator);
        vm.expectRevert(rejection);
        bridge.fulfillJob(jobId, resultHash, resultBlob, forOther);
    }

    /// @dev A quorum that signed one result, presented beside another. Note
    /// this is not the same as the blob/hash mismatch below: here the blob and
    /// the hash agree with each other, and it is the *signatures* that are
    /// about something else.
    function test_fulfillJob_revertsWhenAttestationsAreForAnotherResult() public {
        uint256 programId = _program(1);
        uint256 jobId = _job(programId, 1, address(0), 0);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        IDiscaBridge.Attestation[] memory forOther =
            _quorum(1, jobId, bytecodeHash, keccak256("a result nobody computed"));
        bytes memory rejection =
            _rejectionFor(_digest(jobId, bytecodeHash, resultHash), forOther[0]);

        vm.prank(coordinator);
        vm.expectRevert(rejection);
        bridge.fulfillJob(jobId, resultHash, resultBlob, forOther);
    }

    /// @dev §5a. A genuinely attested hash beside a substituted blob is the
    /// failure this check exists for, and it is silent without it: only the key
    /// holder would find out, after decrypting, with nothing to dispute with.
    function test_fulfillJob_revertsWhenTheBlobDoesNotMatchTheAttestedHash() public {
        uint256 programId = _program(2);
        uint256 jobId = _job(programId, 1, address(0), 0);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        IDiscaBridge.Attestation[] memory honest = _quorum(2, jobId, bytecodeHash, resultHash);
        bytes memory substituted = _blob(RESULT_BLOB_BYTES, "substituted");

        vm.prank(coordinator);
        vm.expectRevert(DiscaBridge.ResultBlobMismatch.selector);
        bridge.fulfillJob(jobId, resultHash, substituted, honest);
    }

    // ---------------------------------------------------------------------
    // Per-attestation checks. §2a steps 3 and 4.
    // ---------------------------------------------------------------------

    /// @dev `ecrecover` returns `address(0)` on failure rather than reverting.
    /// Treating that as a signer is the classic way to accept a forged
    /// signature: an all-zero `(r, s)` would otherwise be one free attester,
    /// repeatable up to the quorum size.
    function test_fulfillJob_revertsOnAZeroRecoveredSigner() public {
        uint256 programId = _program(1);
        uint256 jobId = _job(programId, 1, address(0), 0);

        IDiscaBridge.Attestation[] memory zeroed = new IDiscaBridge.Attestation[](1);
        zeroed[0] = IDiscaBridge.Attestation({r: bytes32(0), s: bytes32(0), v: 27});
        assertEq(ecrecover(bytes32(uint256(1)), 27, bytes32(0), bytes32(0)), address(0));

        vm.prank(coordinator);
        vm.expectRevert(abi.encodeWithSelector(DiscaBridge.ZeroSigner.selector, 0));
        bridge.fulfillJob(jobId, resultHash, resultBlob, zeroed);
    }

    /// @dev One worker signing twice is one worker. Ascending order is what
    /// makes that an O(n) check, so a duplicate lands as an ordering failure —
    /// an address is not strictly greater than itself.
    function test_fulfillJob_revertsOnADuplicateAttester() public {
        uint256 programId = _program(2);
        uint256 jobId = _job(programId, 1, address(0), 0);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        IDiscaBridge.Attestation[] memory twice = new IDiscaBridge.Attestation[](2);
        twice[0] = _attest(workerKeys[0], jobId, bytecodeHash, resultHash);
        twice[1] = twice[0];

        address signer = vm.addr(workerKeys[0]);
        vm.prank(coordinator);
        vm.expectRevert(
            abi.encodeWithSelector(DiscaBridge.AttestersOutOfOrder.selector, signer, signer)
        );
        bridge.fulfillJob(jobId, resultHash, resultBlob, twice);
    }

    /// @dev Two distinct, registered, honest signers — in the wrong order.
    /// Rejected rather than sorted on-chain: sorting is O(n log n) of the
    /// caller's gas or O(n) of everyone's, and the coordinator already sorts.
    function test_fulfillJob_revertsOnUnsortedAttesters() public {
        uint256 programId = _program(2);
        uint256 jobId = _job(programId, 1, address(0), 0);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        IDiscaBridge.Attestation[] memory sorted = _quorum(2, jobId, bytecodeHash, resultHash);
        IDiscaBridge.Attestation[] memory reversed = new IDiscaBridge.Attestation[](2);
        reversed[0] = sorted[1];
        reversed[1] = sorted[0];

        bytes32 digest = _digest(jobId, bytecodeHash, resultHash);
        address higher = ecrecover(digest, reversed[0].v, reversed[0].r, reversed[0].s);
        address lower = ecrecover(digest, reversed[1].v, reversed[1].r, reversed[1].s);
        assertTrue(higher > lower);

        vm.prank(coordinator);
        vm.expectRevert(
            abi.encodeWithSelector(DiscaBridge.AttestersOutOfOrder.selector, higher, lower)
        );
        bridge.fulfillJob(jobId, resultHash, resultBlob, reversed);
    }

    /// @dev `ecrecover` itself refuses anything outside {27, 28} by returning
    /// zero, but rejecting it explicitly names the bug: a coordinator that
    /// forwarded k256's bare 0/1 recovery id would otherwise see every
    /// attestation recover to nobody.
    function test_fulfillJob_revertsOnABadRecoveryId() public {
        uint256 programId = _program(1);
        uint256 jobId = _job(programId, 1, address(0), 0);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        IDiscaBridge.Attestation[] memory bad = new IDiscaBridge.Attestation[](1);
        bad[0] = _attest(workerKeys[0], jobId, bytecodeHash, resultHash);
        bad[0].v = 29;

        vm.prank(coordinator);
        vm.expectRevert(abi.encodeWithSelector(DiscaBridge.BadRecoveryId.selector, uint8(29)));
        bridge.fulfillJob(jobId, resultHash, resultBlob, bad);
    }

    /// @dev EIP-2. `(r, n - s, v ^ 1)` recovers the same signer, so it cannot
    /// inflate a quorum — the ordering check would catch the pair. What it
    /// would allow is a relay silently rewriting an attestation in flight
    /// without invalidating it, and `primitives::attest::recover` refuses the
    /// high form for the same reason.
    function test_fulfillJob_revertsOnAHighSSignature() public {
        uint256 programId = _program(1);
        uint256 jobId = _job(programId, 1, address(0), 0);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        IDiscaBridge.Attestation[] memory malleated = new IDiscaBridge.Attestation[](1);
        malleated[0] = _attest(workerKeys[0], jobId, bytecodeHash, resultHash);

        uint256 order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141;
        malleated[0].s = bytes32(order - uint256(malleated[0].s));
        malleated[0].v = malleated[0].v == 27 ? 28 : 27;

        // The malleated twin really does recover to the same worker, which is
        // what makes rejecting it a deliberate choice rather than a side effect.
        assertEq(
            ecrecover(
                _digest(jobId, bytecodeHash, resultHash),
                malleated[0].v,
                malleated[0].r,
                malleated[0].s
            ),
            vm.addr(workerKeys[0])
        );

        vm.prank(coordinator);
        vm.expectRevert(DiscaBridge.HighS.selector);
        bridge.fulfillJob(jobId, resultHash, resultBlob, malleated);
    }

    // ---------------------------------------------------------------------
    // M-of-N boundaries. §2a step 5.
    // ---------------------------------------------------------------------

    /// @dev M-1 honest signatures. Everything about them is valid; there are
    /// just not enough, which is the only thing standing between one Byzantine
    /// worker and a settled wrong answer.
    function test_fulfillJob_revertsOneAttestationShortOfQuorum() public {
        uint256 programId = _program(3);
        uint256 jobId = _job(programId, 1, address(0), 0);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        IDiscaBridge.Attestation[] memory two = _quorum(2, jobId, bytecodeHash, resultHash);

        vm.prank(coordinator);
        vm.expectRevert(abi.encodeWithSelector(DiscaBridge.QuorumNotMet.selector, 2, uint8(3)));
        bridge.fulfillJob(jobId, resultHash, resultBlob, two);
    }

    function test_fulfillJob_acceptsExactlyTheQuorum() public {
        uint256 programId = _program(3);
        uint256 jobId = _job(programId, 1, address(0), 0);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        vm.prank(coordinator);
        bridge.fulfillJob(
            jobId, resultHash, resultBlob, _quorum(3, jobId, bytecodeHash, resultHash)
        );

        assertEq(uint256(bridge.jobs(jobId).state), uint256(IDiscaBridge.JobState.Fulfilled));
    }

    /// @dev More than M is agreement, not a protocol error. §2a says "at least"
    /// and the extra signers are extra evidence.
    function test_fulfillJob_acceptsMoreThanTheQuorum() public {
        uint256 programId = _program(2);
        uint256 jobId = _job(programId, 1, address(0), 0);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        vm.prank(coordinator);
        bridge.fulfillJob(
            jobId, resultHash, resultBlob, _quorum(5, jobId, bytecodeHash, resultHash)
        );

        assertEq(uint256(bridge.jobs(jobId).state), uint256(IDiscaBridge.JobState.Fulfilled));
    }

    /// @dev A quorum of zero would be met by an empty array, so `fulfillJob`
    /// would verify nothing at all while looking exactly as it does now.
    function test_registerProgram_revertsOnAZeroQuorum() public {
        vm.expectRevert(DiscaBridge.QuorumTooSmall.selector);
        bridge.registerProgram(keccak256("bytecode"), keccak256("server-key"), 0);
    }

    // ---------------------------------------------------------------------
    // Job state machine. §2.
    // ---------------------------------------------------------------------

    function test_submitJob_opensAJobAndRecordsItsCommitments() public {
        uint256 programId = _program(2);
        uint256 jobId = _job(programId, 3, address(0), 1 ether);

        IDiscaBridge.Job memory job = bridge.jobs(jobId);
        assertEq(uint256(job.state), uint256(IDiscaBridge.JobState.Open));
        assertEq(job.programId, programId);
        assertEq(job.poster, poster);
        assertEq(job.escrow, 1 ether);
        assertEq(job.deadline, uint64(block.timestamp) + JOB_TIMEOUT);
        assertEq(bridge.inputCommitsOf(jobId).length, 3);
        assertEq(address(bridge).balance, 1 ether);
    }

    function test_submitJob_revertsForAnUnknownProgram() public {
        bytes32[] memory commits = new bytes32[](0);
        bytes[] memory blobs = new bytes[](0);

        vm.prank(poster);
        vm.expectRevert(abi.encodeWithSelector(DiscaBridge.UnknownProgram.selector, 7));
        bridge.submitJob(7, commits, blobs, address(0));
    }

    /// @dev §6 leans on the commitment to stop a substituted input. A
    /// commitment nobody checks stops nothing, so the check is here rather than
    /// only in the coordinator.
    function test_submitJob_revertsWhenAnInputBlobDoesNotMatchItsCommitment() public {
        uint256 programId = _program(2);

        bytes[] memory blobs = new bytes[](2);
        bytes32[] memory commits = new bytes32[](2);
        blobs[0] = _blob(INPUT_BLOB_BYTES, "a");
        blobs[1] = _blob(INPUT_BLOB_BYTES, "b");
        commits[0] = keccak256(blobs[0]);
        commits[1] = keccak256("a hash of something else");

        vm.prank(poster);
        vm.expectRevert(abi.encodeWithSelector(DiscaBridge.InputCommitmentMismatch.selector, 1));
        bridge.submitJob(programId, commits, blobs, address(0));
    }

    function test_submitJob_revertsOnMismatchedInputArrayLengths() public {
        uint256 programId = _program(2);
        bytes32[] memory commits = new bytes32[](2);
        bytes[] memory blobs = new bytes[](1);

        vm.prank(poster);
        vm.expectRevert(abi.encodeWithSelector(DiscaBridge.InputLengthMismatch.selector, 2, 1));
        bridge.submitJob(programId, commits, blobs, address(0));
    }

    /// @dev Open -> Fulfilled, with the escrow following. Also pins that the
    /// blob reaches the event: §5a makes emission required rather than
    /// optional, because the guarantee is the contract checking an emitted blob
    /// against an attested hash, and that does nothing if the blob is never
    /// emitted.
    function test_fulfillJob_settlesAndPaysTheCoordinator() public {
        uint256 programId = _program(2);
        uint256 jobId = _job(programId, 2, address(0), 1 ether);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        vm.expectEmit(true, false, false, true, address(bridge));
        emit IDiscaBridge.JobFulfilled(jobId, resultHash, resultBlob);

        vm.prank(coordinator);
        bridge.fulfillJob(
            jobId, resultHash, resultBlob, _quorum(2, jobId, bytecodeHash, resultHash)
        );

        assertEq(uint256(bridge.jobs(jobId).state), uint256(IDiscaBridge.JobState.Fulfilled));
        assertEq(coordinator.balance, 1 ether);
        assertEq(address(bridge).balance, 0);
        assertEq(bridge.jobs(jobId).escrow, 0);
    }

    function test_fulfillJob_revertsOnASecondFulfillment() public {
        uint256 programId = _program(2);
        uint256 jobId = _job(programId, 1, address(0), 1 ether);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);
        IDiscaBridge.Attestation[] memory quorum = _quorum(2, jobId, bytecodeHash, resultHash);

        vm.prank(coordinator);
        bridge.fulfillJob(jobId, resultHash, resultBlob, quorum);

        vm.prank(coordinator);
        vm.expectRevert(
            abi.encodeWithSelector(
                DiscaBridge.JobNotOpen.selector, jobId, IDiscaBridge.JobState.Fulfilled
            )
        );
        bridge.fulfillJob(jobId, resultHash, resultBlob, quorum);
    }

    /// @dev An unknown job is `JobState.None`, which is why the enum reserves
    /// zero: without it an unwritten mapping entry would read as `Open` and a
    /// settlement against a job that was never posted would look ordinary.
    function test_fulfillJob_revertsForAnUnknownJob() public {
        IDiscaBridge.Attestation[] memory none = new IDiscaBridge.Attestation[](0);

        vm.prank(coordinator);
        vm.expectRevert(abi.encodeWithSelector(DiscaBridge.UnknownJob.selector, 42));
        bridge.fulfillJob(42, resultHash, resultBlob, none);
    }

    /// @dev Past the deadline only the refund path is open. Allowing both would
    /// make settlement a race between the coordinator and the poster over the
    /// same escrow, decided by transaction ordering.
    function test_fulfillJob_revertsAfterTheDeadline() public {
        uint256 programId = _program(2);
        uint256 jobId = _job(programId, 1, address(0), 1 ether);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);
        uint64 deadline = bridge.jobs(jobId).deadline;

        vm.warp(deadline + 1);
        vm.prank(coordinator);
        vm.expectRevert(abi.encodeWithSelector(DiscaBridge.JobExpired.selector, jobId, deadline));
        bridge.fulfillJob(
            jobId, resultHash, resultBlob, _quorum(2, jobId, bytecodeHash, resultHash)
        );
    }

    function test_fulfillJob_isCoordinatorGated() public {
        uint256 programId = _program(1);
        uint256 jobId = _job(programId, 1, address(0), 1 ether);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);
        IDiscaBridge.Attestation[] memory quorum = _quorum(1, jobId, bytecodeHash, resultHash);

        // A valid quorum, relayed by somebody else, for somebody else's escrow.
        vm.prank(makeAddr("mempool watcher"));
        vm.expectRevert(DiscaBridge.NotCoordinator.selector);
        bridge.fulfillJob(jobId, resultHash, resultBlob, quorum);
    }

    // ---------------------------------------------------------------------
    // Timeout and refund. §6.
    // ---------------------------------------------------------------------

    function test_refundOnTimeout_returnsTheEscrowToThePoster() public {
        uint256 programId = _program(2);
        uint256 jobId = _job(programId, 1, address(0), 1 ether);
        uint256 before = poster.balance;

        vm.warp(bridge.jobs(jobId).deadline + 1);
        bridge.refundOnTimeout(jobId);

        assertEq(uint256(bridge.jobs(jobId).state), uint256(IDiscaBridge.JobState.Refunded));
        assertEq(poster.balance, before + 1 ether);
        assertEq(address(bridge).balance, 0);
    }

    function test_refundOnTimeout_revertsBeforeTheDeadline() public {
        uint256 programId = _program(2);
        uint256 jobId = _job(programId, 1, address(0), 1 ether);
        uint64 deadline = bridge.jobs(jobId).deadline;

        vm.warp(deadline);
        vm.expectRevert(abi.encodeWithSelector(DiscaBridge.JobNotExpired.selector, jobId, deadline));
        bridge.refundOnTimeout(jobId);
    }

    function test_refundOnTimeout_revertsOnASecondRefund() public {
        uint256 programId = _program(2);
        uint256 jobId = _job(programId, 1, address(0), 1 ether);

        vm.warp(bridge.jobs(jobId).deadline + 1);
        bridge.refundOnTimeout(jobId);

        vm.expectRevert(
            abi.encodeWithSelector(
                DiscaBridge.JobNotOpen.selector, jobId, IDiscaBridge.JobState.Refunded
            )
        );
        bridge.refundOnTimeout(jobId);
    }

    function test_refundOnTimeout_revertsForAFulfilledJob() public {
        uint256 programId = _program(2);
        uint256 jobId = _job(programId, 1, address(0), 1 ether);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        vm.prank(coordinator);
        bridge.fulfillJob(
            jobId, resultHash, resultBlob, _quorum(2, jobId, bytecodeHash, resultHash)
        );

        vm.warp(bridge.jobs(jobId).deadline + 1);
        vm.expectRevert(
            abi.encodeWithSelector(
                DiscaBridge.JobNotOpen.selector, jobId, IDiscaBridge.JobState.Fulfilled
            )
        );
        bridge.refundOnTimeout(jobId);
    }

    function test_refundOnTimeout_revertsForAnUnknownJob() public {
        vm.expectRevert(abi.encodeWithSelector(DiscaBridge.UnknownJob.selector, 1));
        bridge.refundOnTimeout(1);
    }

    /// @dev A refunded job cannot then be settled: the escrow is already gone,
    /// so a later fulfillment would pay the coordinator out of the next job's
    /// money.
    function test_fulfillJob_revertsForARefundedJob() public {
        uint256 programId = _program(1);
        uint256 jobId = _job(programId, 1, address(0), 1 ether);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);
        IDiscaBridge.Attestation[] memory quorum = _quorum(1, jobId, bytecodeHash, resultHash);

        vm.warp(bridge.jobs(jobId).deadline + 1);
        bridge.refundOnTimeout(jobId);

        // Warp back inside the window so the state check, not the deadline
        // check, is what refuses this.
        vm.warp(bridge.jobs(jobId).deadline - 1);
        vm.prank(coordinator);
        vm.expectRevert(
            abi.encodeWithSelector(
                DiscaBridge.JobNotOpen.selector, jobId, IDiscaBridge.JobState.Refunded
            )
        );
        bridge.fulfillJob(jobId, resultHash, resultBlob, quorum);
    }

    // ---------------------------------------------------------------------
    // Registry access. §2: owner-gated in the demo.
    // ---------------------------------------------------------------------

    function test_registerWorker_isOwnerGated() public {
        vm.prank(poster);
        vm.expectRevert(DiscaBridge.NotOwner.selector);
        bridge.registerWorker(makeAddr("a worker the poster likes"));
    }

    function test_setCoordinator_isOwnerGated() public {
        vm.prank(poster);
        vm.expectRevert(DiscaBridge.NotOwner.selector);
        bridge.setCoordinator(poster);
    }

    /// @dev Registering a program grants nothing, so it is open. The gate that
    /// matters is the worker registry.
    function test_registerProgram_isOpen() public {
        vm.prank(poster);
        uint256 programId = bridge.registerProgram(keccak256("bc"), keccak256("sk"), 2);
        (,, uint8 required) = bridge.programs(programId);
        assertEq(required, 2);
    }

    // ---------------------------------------------------------------------
    // The preimage layout. §2a.
    // ---------------------------------------------------------------------

    /// @dev Pins the 94-byte layout without needing the Rust vector present.
    /// `test/AttestationVector.t.sol` is the real cross-language check; this is
    /// what fails first, and locally, if `abi.encodePacked` is ever fed the
    /// wrong types — a `uint256` job id instead of a `uint64` would silently
    /// produce a 118-byte preimage that no worker has ever signed.
    function test_attestationPreimage_matchesTheDocumentedLayout() public view {
        bytes32 bytecodeHash = bytes32(uint256(0x1111) * (2 ** 240));
        bytes32 rHash = bytes32(uint256(0x2222) * (2 ** 240));
        bytes memory preimage = bridge.attestationPreimage(0x0102030405060708, bytecodeHash, rHash);

        assertEq(preimage.length, 94);
        assertEq(keccak256(_slice(preimage, 0, 22)), keccak256(bytes("DISCA/attest/result/v1")));
        assertEq(
            keccak256(_slice(preimage, 22, 8)),
            keccak256(abi.encodePacked(hex"0102030405060708")),
            "the job id is 8 big-endian bytes"
        );
        assertEq(keccak256(_slice(preimage, 30, 32)), keccak256(abi.encodePacked(bytecodeHash)));
        assertEq(keccak256(_slice(preimage, 62, 32)), keccak256(abi.encodePacked(rHash)));
    }

    /// @dev The EIP-191 prefix must actually be applied, not merely present.
    function test_attestationDigest_appliesTheEip191Prefix() public view {
        bytes memory preimage = bridge.attestationPreimage(1, keccak256("bc"), keccak256("r"));
        bytes32 inner = keccak256(preimage);
        bytes32 digest = bridge.attestationDigest(1, keccak256("bc"), keccak256("r"));

        assertEq(digest, keccak256(abi.encodePacked("\x19Ethereum Signed Message:\n32", inner)));
        assertTrue(digest != inner);
    }

    function _slice(bytes memory data, uint256 offset, uint256 length)
        private
        pure
        returns (bytes memory out)
    {
        out = new bytes(length);
        for (uint256 i = 0; i < length; ++i) {
            out[i] = data[offset + i];
        }
    }
}
