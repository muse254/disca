// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {BridgeHarness} from "./BridgeHarness.sol";
import {CommitteeTally} from "../src/CommitteeTally.sol";
import {DiscaBridge} from "../src/DiscaBridge.sol";
import {IDiscaBridge, IDiscaConsumer} from "../src/IDiscaBridge.sol";

/// @notice A consumer that always reverts, to pin what `fulfillJob` does about
/// it (`DiscaBridge.fulfillJob`: the settlement fails and the job takes the
/// refund path, rather than the escrow releasing to a consumer that never
/// learned the job finished).
contract RevertingConsumer is IDiscaConsumer {
    error Nope();

    function onJobFulfilled(uint256, bytes32) external pure {
        revert Nope();
    }
}

/// @notice The demo consumer from `docs/bridge.md` §5, end to end.
contract CommitteeTallyTest is BridgeHarness {
    CommitteeTally internal tally;
    address internal committee = makeAddr("committee");
    uint256 internal programId;
    bytes32 internal bytecodeHash;

    bytes internal resultBlob;
    bytes32 internal resultHash;

    function setUp() public override {
        super.setUp();

        programId = _program(2);
        (bytecodeHash,,) = bridge.programs(programId);
        tally = new CommitteeTally(bridge, committee, programId);

        resultBlob = _blob(RESULT_BLOB_BYTES, "tally result");
        resultHash = keccak256(resultBlob);
        vm.deal(committee, 10 ether);
    }

    /// @dev The whole §5 lifecycle in one test: encrypted ballots committed
    /// on-chain, a 2-of-N attested result settled, and the consumer holding a
    /// commitment to a ciphertext it cannot read.
    function test_theTallyLifecycleRunsEndToEnd() public {
        uint256 jobId = _startTally(3, 1 ether);

        IDiscaBridge.Job memory job = bridge.jobs(jobId);
        assertEq(job.callback, address(tally));
        assertEq(job.poster, address(tally));
        assertEq(job.escrow, 1 ether);
        assertEq(tally.jobId(), jobId);
        assertEq(tally.resultCommit(), bytes32(0));

        vm.expectEmit(true, false, false, true, address(tally));
        emit CommitteeTally.TallySettled(jobId, resultHash);

        vm.prank(coordinator);
        bridge.fulfillJob(
            jobId, resultHash, resultBlob, _quorum(2, jobId, bytecodeHash, resultHash)
        );

        assertEq(tally.resultCommit(), resultHash, "the callback must fire on fulfillment");
        assertEq(coordinator.balance, 1 ether);

        // The trust boundary. Nothing on-chain can check this number against
        // the ciphertext; only the key holder can decrypt it.
        vm.prank(committee);
        tally.reveal(7);
        assertEq(tally.winner(), 7);
        assertTrue(tally.revealed());
    }

    /// @dev `onJobFulfilled` is the only path by which a result enters this
    /// contract, and the bridge is the only party that has checked a quorum. If
    /// anyone could call it, the consumer would accept a `resultCommit` no
    /// worker ever signed — which is the §2b failure again, one contract along.
    function test_onJobFulfilled_revertsWhenTheCallerIsNotTheBridge() public {
        uint256 jobId = _startTally(1, 0);

        vm.prank(makeAddr("anyone at all"));
        vm.expectRevert(CommitteeTally.NotBridge.selector);
        tally.onJobFulfilled(jobId, keccak256("a result I made up"));
    }

    /// @dev A callback for a job this contract is not currently running is a
    /// stale callback, and accepting it would overwrite the live tally.
    function test_onJobFulfilled_revertsForADifferentJob() public {
        uint256 jobId = _startTally(1, 0);

        vm.prank(address(bridge));
        vm.expectRevert(abi.encodeWithSelector(CommitteeTally.WrongJob.selector, jobId + 1, jobId));
        tally.onJobFulfilled(jobId + 1, resultHash);
    }

    function test_startTally_isCommitteeGated() public {
        bytes32[] memory commits = new bytes32[](0);
        bytes[] memory blobs = new bytes[](0);

        vm.prank(poster);
        vm.expectRevert(CommitteeTally.NotCommittee.selector);
        tally.startTally(commits, blobs);
    }

    /// @dev One `jobId` and one `resultCommit`, so a second tally started
    /// before the first settles would make the first callback unroutable.
    function test_startTally_revertsWhileATallyIsInFlight() public {
        uint256 jobId = _startTally(1, 0);

        bytes32[] memory commits = new bytes32[](0);
        bytes[] memory blobs = new bytes[](0);
        vm.prank(committee);
        vm.expectRevert(abi.encodeWithSelector(CommitteeTally.TallyInFlight.selector, jobId));
        tally.startTally(commits, blobs);
    }

    function test_reveal_revertsBeforeAResultHasSettled() public {
        _startTally(1, 0);

        vm.prank(committee);
        vm.expectRevert(CommitteeTally.NoResultYet.selector);
        tally.reveal(1);
    }

    function test_reveal_isCommitteeGatedAndHappensOnce() public {
        uint256 jobId = _startTally(1, 0);
        vm.prank(coordinator);
        bridge.fulfillJob(
            jobId, resultHash, resultBlob, _quorum(2, jobId, bytecodeHash, resultHash)
        );

        vm.prank(poster);
        vm.expectRevert(CommitteeTally.NotCommittee.selector);
        tally.reveal(1);

        vm.prank(committee);
        tally.reveal(1);

        vm.prank(committee);
        vm.expectRevert(CommitteeTally.AlreadyRevealed.selector);
        tally.reveal(2);
    }

    /// @dev Documented behaviour, not an accident: a consumer that reverts
    /// takes the settlement with it and the job then times out and refunds.
    /// Swallowing the revert would release the escrow while the contract that
    /// paid for the computation believes nothing happened.
    function test_fulfillJob_revertsWhenTheConsumerReverts() public {
        RevertingConsumer consumer = new RevertingConsumer();
        uint256 jobId = _job(programId, 1, address(consumer), 1 ether);
        IDiscaBridge.Attestation[] memory quorum = _quorum(2, jobId, bytecodeHash, resultHash);

        vm.prank(coordinator);
        vm.expectRevert(RevertingConsumer.Nope.selector);
        bridge.fulfillJob(jobId, resultHash, resultBlob, quorum);

        // And the escrow is still recoverable by the poster, which is what
        // makes the failure survivable.
        assertEq(uint256(bridge.jobs(jobId).state), uint256(IDiscaBridge.JobState.Open));
        vm.warp(bridge.jobs(jobId).deadline + 1);
        bridge.refundOnTimeout(jobId);
        assertEq(uint256(bridge.jobs(jobId).state), uint256(IDiscaBridge.JobState.Refunded));
    }

    function _startTally(uint256 ballots, uint256 escrow) private returns (uint256 jobId) {
        bytes[] memory blobs = new bytes[](ballots);
        bytes32[] memory commits = new bytes32[](ballots);
        for (uint256 i = 0; i < ballots; ++i) {
            blobs[i] = _blob(INPUT_BLOB_BYTES, bytes32(i));
            commits[i] = keccak256(blobs[i]);
        }

        vm.prank(committee);
        return tally.startTally{value: escrow}(commits, blobs);
    }
}
