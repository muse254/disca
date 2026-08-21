// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Script} from "forge-std/Script.sol";
import {console2} from "forge-std/console2.sol";

import {CommitteeTally} from "../src/CommitteeTally.sol";
import {DiscaBridge} from "../src/DiscaBridge.sol";
import {IDiscaBridge} from "../src/IDiscaBridge.sol";

/// @title Deploy — the one transaction bundle `scripts/run-anvil.sh` cannot do
/// with `cast`.
/// @notice Task 3.3, and step 4 of `docs/bridge.md` §8: stand up a bridge, pin
/// the program the DISCA workers are about to run, and bind the demo consumer
/// to it. Everything after this — `registerWorker`, `submitJob`, `fulfillJob`,
/// `refundOnTimeout` — is an ordinary call and lives in the shell script, where
/// a reader can see each step as a separate transaction.
///
/// @dev **Why `registerProgram` is in here and not in the shell.**
/// `CommitteeTally.programId` is `immutable`, so it has to be known at
/// construction, and it is not known until `registerProgram` has returned. A
/// shell doing this with `cast` would have to send the transaction, read the
/// `ProgramRegistered` event back, and then deploy — three round trips whose
/// middle step is a log parse that fails silently when it goes wrong. One
/// script gets the return value directly.
///
/// The hashes are still the real ones: they arrive as environment variables
/// that `run-anvil.sh` fills from what `disca-cli compile` and `disca-cli
/// keygen` print. Nothing here invents a value.
///
/// **Every input is required.** `vm.envBytes32` and friends revert on a missing
/// variable rather than substituting a zero, which is the behaviour this whole
/// exercise is about: a bridge deployed against `bytes32(0)` as its bytecode
/// hash would accept attestations over a program that does not exist, and would
/// look exactly like a working deployment until someone checked what the
/// workers had signed.
contract Deploy is Script {
    /// @notice Deploys the bridge and the demo consumer, and pins the program.
    /// @return bridge The `DiscaBridge` deployment.
    /// @return tally The `CommitteeTally` bound to `programId` on it.
    /// @return programId The program id `registerProgram` issued.
    /// @dev Reads:
    ///
    /// | variable             | meaning                                        |
    /// |----------------------|------------------------------------------------|
    /// | `JOB_TIMEOUT`        | seconds a job stays fulfillable                |
    /// | `BYTECODE_HASH`      | `disca-cli compile` -> `bytecode_hash=`        |
    /// | `SERVER_KEY_HASH`    | `disca-cli keygen` -> `server_key_hash=`       |
    /// | `ATTESTERS_REQUIRED` | M, the quorum size the workers run under       |
    /// | `COORDINATOR`        | the only address allowed to call `fulfillJob`  |
    /// | `COMMITTEE`          | key holder; starts and reveals the tally       |
    function run() external returns (DiscaBridge bridge, CommitteeTally tally, uint256 programId) {
        uint64 jobTimeout = uint64(vm.envUint("JOB_TIMEOUT"));
        bytes32 bytecodeHash = vm.envBytes32("BYTECODE_HASH");
        bytes32 serverKeyHash = vm.envBytes32("SERVER_KEY_HASH");
        uint8 attestersRequired = uint8(vm.envUint("ATTESTERS_REQUIRED"));
        address coordinator = vm.envAddress("COORDINATOR");
        address committee = vm.envAddress("COMMITTEE");

        // `DiscaBridge` rejects both of these itself, but it does so from
        // inside a broadcast, where the revert reason arrives as a failed
        // transaction rather than as a sentence. Checking here costs nothing
        // and names the variable that was wrong.
        require(jobTimeout > 0, "JOB_TIMEOUT must be non-zero: every job would be born expired");
        require(attestersRequired > 0, "ATTESTERS_REQUIRED must be non-zero: see QuorumTooSmall");

        vm.startBroadcast();

        bridge = new DiscaBridge(jobTimeout);

        // The constructor makes the deployer the coordinator. Leaving it there
        // would make the escrow assertion in `run-anvil.sh` untestable — the
        // deployer already holds the gas budget, so "the escrow moved" would be
        // a delta against a moving target. Separating the two is also what
        // `docs/bridge.md` §2 describes: the coordinator is a distinct party
        // that is paid for settling.
        bridge.setCoordinator(coordinator);

        programId = bridge.registerProgram(bytecodeHash, serverKeyHash, attestersRequired);
        tally = new CommitteeTally(IDiscaBridge(address(bridge)), committee, programId);

        vm.stopBroadcast();

        // Shell-parseable, one `KEY=value` per line, no other `=` on the line.
        // `forge script` indents its log output under `== Logs ==`, so the
        // consumer greps rather than reads positionally — see `_deploy` in
        // `scripts/run-anvil.sh`.
        console2.log(string.concat("BRIDGE_ADDRESS=", vm.toString(address(bridge))));
        console2.log(string.concat("TALLY_ADDRESS=", vm.toString(address(tally))));
        console2.log(string.concat("PROGRAM_ID=", vm.toString(programId)));
        console2.log(string.concat("JOB_TIMEOUT=", vm.toString(uint256(jobTimeout))));
    }
}
