// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Test} from "forge-std/Test.sol";
import {stdJson} from "forge-std/StdJson.sol";
import {console2} from "forge-std/console2.sol";

import {DiscaBridge} from "../src/DiscaBridge.sol";
import {IDiscaBridge} from "../src/IDiscaBridge.sol";

/// @notice A whole settlement driven from the files `scripts/run-anvil.sh`
/// carries between the two halves of the system — the `--attestations` JSON and
/// the result blob — with no Rust process and no chain in the loop.
///
/// @dev Task 3.3. `AttestationVector.t.sol` proves one signature over one
/// synthetic claim recovers to the address Rust says it should. This proves the
/// next thing along: that a *file* in the shape the coordinator writes, holding
/// a quorum over a result an actual DISCA network actually produced, is a file
/// `fulfillJob` accepts.
///
/// **Where the fixture came from.** `script/fixtures/result.hex` is the
/// 12,075-byte compressed ciphertext a real 2-of-3 run of `tally4_select` over
/// encrypted `71,93,42,88` settled on — the bytes `--result` wrote.
/// `script/fixtures/attestations.json` holds signatures over the §2a claim
/// naming that result, produced by `script/fixtures/sign-attestations.sh` under
/// the *worker* keys: `keccak256("DISCA/dev-key/v1" || id)`, which is what
/// `primitives::attest::WorkerKey::derive` computes, checked at generation time
/// against what `node worker-address` printed.
///
/// **What that does not make it.** The signatures were produced by `cast wallet
/// sign` rather than by a worker that had just evaluated the circuit, because
/// the coordinator has no `--attestations` flag yet and, more fundamentally,
/// signs over a job id it invents rather than the one `submitJob` assigns
/// (task 2.9f). So this is evidence about the contract and the file format. It
/// is not evidence that two independent evaluators agreed, which is the only
/// thing M-of-N buys (`docs/bridge.md` §2). When the flag lands, the same
/// assertions run against a file the network wrote, and the fixture becomes the
/// offline copy of a thing that happened.
contract FixtureSettlementTest is Test {
    using stdJson for string;

    /// @dev Mirrors one element of the `attesters` array. `forge` decodes a
    /// JSON object into a struct by sorting the object's keys alphabetically
    /// and matching them positionally, so the *order* of these fields is load
    /// bearing and their names are not: address, r, s, v happens to be both the
    /// alphabetical order of the JSON keys and the order they are read in.
    struct FixtureAttester {
        address attester;
        bytes32 r;
        bytes32 s;
        uint8 v;
    }

    string internal constant ATTESTATIONS_PATH = "./script/fixtures/attestations.json";
    string internal constant RESULT_PATH = "./script/fixtures/result.hex";

    /// @dev The quorum the demo runs (`docs/bridge.md` §4: 1 coordinator, 3
    /// workers, 2-of-3).
    uint8 internal constant ATTESTERS_REQUIRED = 2;

    uint64 internal constant JOB_TIMEOUT = 1 hours;

    DiscaBridge internal bridge;
    address internal coordinator = makeAddr("coordinator");
    address internal poster = makeAddr("poster");

    /// @dev Storage rather than locals throughout. The fixture carries eight
    /// values that all have to be live at once and solc runs out of stack
    /// slots; splitting the test into functions that each hold a few of them
    /// would hide which assertion depends on which field.
    uint64 internal jobId;
    bytes32 internal bytecodeHash;
    bytes32 internal resultHash;
    bytes internal resultBlob;
    FixtureAttester[] internal attesters;

    function setUp() public {
        string memory fixture = vm.readFile(ATTESTATIONS_PATH);

        jobId = uint64(fixture.readUint(".jobId"));
        bytecodeHash = fixture.readBytes32(".bytecodeHash");
        resultHash = fixture.readBytes32(".resultHash");

        // `vm.parseBytes` over a hex file rather than `vm.readFileBinary`,
        // because the blob is committed as text: 24 KB of hex is greppable, and
        // a diff on it is a diff rather than "binary files differ".
        resultBlob = vm.parseBytes(vm.trim(vm.readFile(RESULT_PATH)));

        FixtureAttester[] memory parsed =
            abi.decode(fixture.parseRaw(".attesters"), (FixtureAttester[]));
        for (uint256 i = 0; i < parsed.length; ++i) {
            attesters.push(parsed[i]);
        }

        bridge = new DiscaBridge(JOB_TIMEOUT);
        bridge.setCoordinator(coordinator);
        vm.deal(poster, 10 ether);
    }

    /// @dev Checked before anything uses them, because every assertion below is
    /// about a claim naming `resultHash`. A blob that no longer hashes to it
    /// would fail as `ResultBlobMismatch` — a true failure with a cause that
    /// points at the contract rather than at the fixture.
    function test_theFixtureIsInternallyConsistent() public view {
        assertEq(keccak256(resultBlob), resultHash, "result.hex does not hash to the attested hash");

        // `docs/architecture.md` §2 measures a computed `CompressedFheInt32` at
        // 11.8 KB. A fixture that had shrunk to a few hundred bytes would still
        // satisfy every other assertion here while quietly ceasing to be a
        // realistic settlement.
        assertGt(resultBlob.length, 10_000, "the fixture blob is not a real compressed result");
        assertGe(attesters.length, ATTESTERS_REQUIRED, "the fixture carries less than a quorum");

        bytes32 digest = bridge.attestationDigest(jobId, bytecodeHash, resultHash);
        address previous = address(0);

        for (uint256 i = 0; i < attesters.length; ++i) {
            address recovered = ecrecover(digest, attesters[i].v, attesters[i].r, attesters[i].s);
            assertEq(
                recovered, attesters[i].attester, "a signature does not recover to its address"
            );

            // `docs/bridge.md` §2a step 4. The coordinator is specified to emit
            // this array already sorted; an unsorted file reaches the chain as
            // `AttestersOutOfOrder`, so the file format is checked here rather
            // than diagnosed from a revert.
            assertGt(
                uint160(recovered), uint160(previous), "attesters must ascend strictly by address"
            );
            previous = recovered;
        }
    }

    function test_theFixtureQuorumSettlesARealResultBlob() public {
        // The registry is built from the addresses the *file* names, and the
        // file's signatures are then what settle the job. Registering
        // `ecrecover`'s output instead would make this pass by construction
        // whatever had been signed; `test_theFixtureIsInternallyConsistent`
        // above is what ties the two together.
        for (uint256 i = 0; i < attesters.length; ++i) {
            bridge.registerWorker(attesters[i].attester);
        }

        uint256 programId = bridge.registerProgram(
            bytecodeHash, keccak256("a server key this test never sees"), ATTESTERS_REQUIRED
        );
        uint256 posted = _submit(programId);

        // The claim binds the job id, so a fixture generated for job 1 settles
        // job 1 and nothing else. On a fresh `DiscaBridge` the first
        // `submitJob` is job 1, which is why the fixture pins that value — and
        // why this is an assertion rather than a comment.
        assertEq(posted, jobId, "the fixture is bound to a different job id");

        vm.expectEmit(true, false, false, true, address(bridge));
        emit IDiscaBridge.JobFulfilled(jobId, resultHash, resultBlob);

        uint256 spent = _fulfill();

        assertEq(
            uint256(bridge.jobs(jobId).state),
            uint256(IDiscaBridge.JobState.Fulfilled),
            "the job did not settle"
        );
        assertEq(coordinator.balance, 1 ether, "the escrow did not reach the coordinator");
        assertEq(bridge.jobs(jobId).escrow, 0, "the escrow was not cleared");

        // Not asserted against a threshold: `FulfillGas.t.sol` owns the gas
        // table and its measurements are controlled. This is the same call over
        // bytes a real network produced, printed so the two can be compared
        // when one of them moves.
        console2.log("fulfillJob execution gas over the real fixture blob:", spent);
        console2.log("  result blob bytes:", resultBlob.length);
        console2.log("  attesters:", attesters.length);
    }

    /// @dev Four ballots, because the fixture's result came from
    /// `tally4_select`. Their contents cannot affect settlement — the contract
    /// checks each against its own commitment and nothing else — but the count
    /// and the size keep the gas figure above comparable to a real job.
    function _submit(uint256 programId) private returns (uint256) {
        bytes[] memory blobs = new bytes[](4);
        bytes32[] memory commits = new bytes32[](4);
        for (uint256 i = 0; i < 4; ++i) {
            blobs[i] = new bytes(2355);
            commits[i] = keccak256(blobs[i]);
        }

        vm.prank(poster);
        return bridge.submitJob{value: 1 ether}(programId, commits, blobs, address(0));
    }

    function _fulfill() private returns (uint256 spent) {
        IDiscaBridge.Attestation[] memory quorum = new IDiscaBridge.Attestation[](attesters.length);
        for (uint256 i = 0; i < attesters.length; ++i) {
            quorum[i] =
                IDiscaBridge.Attestation({r: attesters[i].r, s: attesters[i].s, v: attesters[i].v});
        }

        uint256 before = gasleft();
        vm.prank(coordinator);
        bridge.fulfillJob(jobId, resultHash, resultBlob, quorum);
        spent = before - gasleft();
    }
}
