// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {BridgeHarness} from "./BridgeHarness.sol";
import {CommitteeTally} from "../src/CommitteeTally.sol";
import {DiscaBridge} from "../src/DiscaBridge.sol";
import {IDiscaBridge} from "../src/IDiscaBridge.sol";

/// @notice What `refundOnTimeout` does when the poster is a contract that
/// cannot receive ether — which, today, is `CommitteeTally`.
///
/// @dev Found by driving `scripts/run-anvil.sh` against a real chain (task
/// 3.3), and worth a file of its own because it is not a bug in the code under
/// test so much as a gap between two documents.
///
/// `docs/bridge.md` §6 lists a refund as the handling for every liveness
/// failure: coordinator silence, a withheld result, workers who never agree.
/// §5's demo consumer posts the job itself — `startTally` calls
/// `bridge.submitJob{value: msg.value}(...)`, so `job.poster` is the
/// `CommitteeTally` address — and neither §5's sketch nor `CommitteeTally.sol`
/// has a `receive` or a `fallback`. `refundOnTimeout` pays with
/// `poster.call{value: escrow}("")`, that call fails, and the whole refund
/// reverts on `EscrowTransferFailed`.
///
/// The escrow is then stuck. Not merely delayed: `refundOnTimeout` is the only
/// path out of `Open` besides `fulfillJob`, and the job is past its deadline so
/// `fulfillJob` reverts with `JobExpired`. Every retry fails the same way,
/// forever.
///
/// **The bridge is not the thing that is wrong here.** Reverting rather than
/// swallowing a failed transfer is the correct choice — silently marking a job
/// `Refunded` while the money stayed put would be worse, and
/// `EscrowTransferFailed` exists precisely so this cannot happen quietly. The
/// fix belongs in the consumer: `CommitteeTally` needs a `receive() external
/// payable`, and the tests below are written so that adding one turns
/// `test_aTallyWithEscrowCanNeverBeRefunded` red rather than leaving it
/// passing on stale reasoning.
///
/// A zero-escrow job is unaffected, because `DiscaBridge` skips the transfer
/// entirely when there is nothing to send — which is why this has not shown up
/// in `CommitteeTally.t.sol`, where every refund path runs at zero value or
/// through an EOA poster.
contract RefundToContractPosterTest is BridgeHarness {
    CommitteeTally internal tally;
    address internal committee = makeAddr("committee");
    uint256 internal programId;

    function setUp() public override {
        super.setUp();
        programId = _program(2);
        tally = new CommitteeTally(bridge, committee, programId);
        vm.deal(committee, 10 ether);
    }

    /// @dev The finding, stated as an assertion so that fixing it is a visible
    /// change rather than a quiet one.
    function test_aTallyWithEscrowCanNeverBeRefunded() public {
        uint256 jobId = _startTally(1 ether);
        vm.warp(bridge.jobs(jobId).deadline + 1);

        vm.expectRevert(
            abi.encodeWithSelector(
                DiscaBridge.EscrowTransferFailed.selector, address(tally), 1 ether
            )
        );
        bridge.refundOnTimeout(jobId);

        // Still Open, and permanently so: the deadline has passed, so the only
        // other exit is closed too.
        assertEq(
            uint256(bridge.jobs(jobId).state),
            uint256(IDiscaBridge.JobState.Open),
            "the job should not have changed state"
        );
        assertEq(bridge.jobs(jobId).escrow, 1 ether, "the escrow is still held by the bridge");
        assertEq(address(bridge).balance, 1 ether, "and the ether is still in the bridge");

        vm.expectRevert(
            abi.encodeWithSelector(
                DiscaBridge.JobExpired.selector, jobId, bridge.jobs(jobId).deadline
            )
        );
        vm.prank(coordinator);
        bridge.fulfillJob(
            jobId, keccak256("anything"), "anything", new IDiscaBridge.Attestation[](0)
        );
    }

    /// @dev The other half, and the reason the failure above has stayed
    /// invisible: with nothing to send, `DiscaBridge` never attempts the
    /// transfer, so the same consumer refunds cleanly.
    function test_aTallyWithNoEscrowRefundsCleanly() public {
        uint256 jobId = _startTally(0);
        vm.warp(bridge.jobs(jobId).deadline + 1);

        bridge.refundOnTimeout(jobId);
        assertEq(uint256(bridge.jobs(jobId).state), uint256(IDiscaBridge.JobState.Refunded));
    }

    /// @dev And the control: an EOA poster, same bridge, same escrow, refunds.
    /// Without this the test above would be consistent with `refundOnTimeout`
    /// being broken outright.
    function test_anEoaPosterWithTheSameEscrowIsRefunded() public {
        uint256 jobId = _job(programId, 1, address(0), 1 ether);
        uint256 before = poster.balance;
        vm.warp(bridge.jobs(jobId).deadline + 1);

        bridge.refundOnTimeout(jobId);
        assertEq(uint256(bridge.jobs(jobId).state), uint256(IDiscaBridge.JobState.Refunded));
        assertEq(poster.balance, before + 1 ether);
    }

    function _startTally(uint256 escrow) private returns (uint256) {
        bytes[] memory blobs = new bytes[](1);
        bytes32[] memory commits = new bytes32[](1);
        blobs[0] = _blob(INPUT_BLOB_BYTES, bytes32(0));
        commits[0] = keccak256(blobs[0]);

        vm.prank(committee);
        return tally.startTally{value: escrow}(commits, blobs);
    }
}
