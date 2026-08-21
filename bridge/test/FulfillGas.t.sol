// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {console2} from "forge-std/console2.sol";
import {BridgeHarness} from "./BridgeHarness.sol";
import {DiscaBridge} from "../src/DiscaBridge.sol";
import {IDiscaBridge} from "../src/IDiscaBridge.sol";

/// @notice What `fulfillJob` actually costs, against a realistic 11.8 KB result
/// blob (`docs/architecture.md` §2).
///
/// @dev `docs/bridge.md` §1 and §2b state "order 250-350k gas" total and
/// "~3.5k gas per attester". Both were arithmetic, not measurement. Measured
/// here, solc 0.8.26, optimizer on, 1000 runs, 3 input commitments, escrow
/// paid to an already-funded coordinator (run
/// `forge test --match-contract FulfillGasTest -vv` to reprint):
///
/// ```text
///  M  calldata B   execution   calldata gas   total Cancun   total Prague
///  1      12,388     129,962        195,844        346,806        510,610
///  2      12,484     136,369        197,008        354,377        513,520
///  3      12,580     142,251        198,160        361,411        516,400
///  5      12,772     155,255        200,488        376,743        522,220
/// ```
///
/// Totals include the 21,000 intrinsic. "Cancun" prices calldata at EIP-2028's
/// 4/16 gas per zero/non-zero byte; "Prague" applies EIP-7623's floor of 10 gas
/// per token, a non-zero byte being 4 tokens.
///
/// Three things the doc does not say, none of which change a decision in it.
///
/// **The 250-350k range understates even the Cancun cost.** A 2-of-3
/// settlement — the demo's shape (§4) — measures 354k, above the top of the
/// range at every M. The gap is not the signatures: §1's own sketch counts the
/// blob only as "calldata at ~16 gas/byte and event data at ~8", and misses the
/// ~25k of everything else the transaction does (the 21,000 intrinsic, hashing
/// 11.8 KB twice, copying it into memory, the escrow transfer).
///
/// **On mainnet as it runs today the real number is ~510k, not ~350k.** EIP-7623
/// (Prague/Pectra) put a floor under transactions that are mostly calldata, and
/// `fulfillJob` is exactly that shape: 12 KB of almost entirely non-zero
/// calldata against ~140k of execution. The floor binds at every M measured, so
/// the attester count barely moves the total — 510,610 at M=1 against 516,400
/// at M=3. §5a's conclusion survives (option B is still an ordinary
/// transaction, and §7's targets are Anvil and an L2 where neither figure is
/// felt), but §1's range should be read as pre-Pectra.
///
/// **~3.5k per attester is about half the measured cost.** The marginal Cancun
/// cost of one more attester is 7.0-7.7k: `ecrecover` at 3,000 and 1,164 of
/// calldata, as §2b reasons, plus ~2,100 for the cold `isRegisteredWorker`
/// SLOAD that §2b does not count and ~800 of decode and loop. §2b's per-attester
/// calldata arithmetic also assumes 65 bytes; an `Attestation` in a dynamic
/// array is ABI-encoded as three 32-byte words, so it is 96 bytes on the wire,
/// of which 31 are the zero padding around `v`. So the signature scheme costs
/// ~22k for a 3-of-N job rather than §2b's ~10k — 6% of the transaction rather
/// than 3%. The conclusion is unchanged and the reasoning behind it is
/// strengthened: the address-list design of §2b was "cheaper" by 6%, in
/// exchange for the only property the contract provides.
///
/// The result blob dominates, as §1 says: ~198k of the 361k Cancun total at
/// M=3, and the sole reason the Prague floor binds. §5a's warning that this is
/// the cost of a *single* `i32` is the number to watch — ten output values
/// would be ~118 KB, and under EIP-7623 that is ~4.8M gas of floor alone.
contract FulfillGasTest is BridgeHarness {
    /// @dev Per EIP-2028 and EIP-7623 respectively.
    uint256 private constant GAS_PER_ZERO_BYTE = 4;
    uint256 private constant GAS_PER_NONZERO_BYTE = 16;
    uint256 private constant TOKENS_PER_NONZERO_BYTE = 4;
    uint256 private constant FLOOR_GAS_PER_TOKEN = 10;
    uint256 private constant TX_BASE_GAS = 21000;

    function setUp() public override {
        super.setUp();
        // A real coordinator has paid for gas and therefore has a balance.
        // Without this the escrow transfer lands on an *empty* account and
        // costs an extra 25,000 gas for creating it (EIP-161), which would show
        // up in every row of the table as a one-off that no production
        // settlement pays.
        vm.deal(coordinator, 1 wei);
    }

    function test_gas_oneAttester() public {
        _measure(1);
    }

    function test_gas_twoAttesters() public {
        _measure(2);
    }

    /// @dev The demo's shape: 1 coordinator, 3 workers, 2-of-3 attestation
    /// (`docs/bridge.md` §4). Three attesters is the upper end of it.
    function test_gas_threeAttesters() public {
        _measure(3);
    }

    function test_gas_fiveAttesters() public {
        _measure(5);
    }

    /// @dev One measurement per test function, deliberately. Forge runs each
    /// test as one transaction, so a loop inside a single function would leave
    /// the coordinator account and the bridge's storage slots warm from the
    /// previous iteration and report every M after the first as ~2-5k cheaper
    /// than it is.
    function _measure(uint8 m) private {
        bytes memory data = _settlementCalldata(m);

        vm.prank(coordinator);
        uint256 before = gasleft();
        (bool ok,) = address(bridge).call(data);
        uint256 execution = before - gasleft();
        assertTrue(ok, "the measured call must be the succeeding one");

        _report(m, data, execution);
    }

    /// @dev Split out only to keep `_measure` inside the EVM's stack limit; the
    /// measured call has to be the last thing in a function that holds as
    /// little as possible.
    function _settlementCalldata(uint8 m) private returns (bytes memory) {
        uint256 programId = _program(m);
        uint256 jobId = _job(programId, 3, address(0), 1 ether);
        (bytes32 bytecodeHash,,) = bridge.programs(programId);

        bytes memory resultBlob = _blob(RESULT_BLOB_BYTES, "result");
        bytes32 resultHash = keccak256(resultBlob);

        return abi.encodeCall(
            DiscaBridge.fulfillJob,
            (jobId, resultHash, resultBlob, _quorum(m, jobId, bytecodeHash, resultHash))
        );
    }

    function _report(uint8 m, bytes memory data, uint256 execution) private pure {
        (uint256 calldataGas, uint256 floorTotal) = _calldataCost(data);
        uint256 cancun = TX_BASE_GAS + calldataGas + execution;
        uint256 prague = cancun > floorTotal ? cancun : floorTotal;

        console2.log("attesters                ", m);
        console2.log("  calldata bytes         ", data.length);
        console2.log("  execution gas          ", execution);
        console2.log("  calldata gas           ", calldataGas);
        console2.log("  total, Cancun          ", cancun);
        console2.log("  total, Prague/EIP-7623 ", prague);

        // Bounds, not exact values: the point is to fail loudly if `fulfillJob`
        // ever leaves the range `docs/bridge.md` reasons about, not to pin a
        // number that a compiler release will move by 40 gas.
        require(execution < 200_000, "execution gas left the range in bridge.md 1");
        require(prague < 600_000, "a settlement stopped being an ordinary transaction");
    }

    /// @return calldataGas EIP-2028 pricing: 4 gas per zero byte, 16 per
    /// non-zero.
    /// @return floorTotal EIP-7623's floor, which is what a calldata-heavy
    /// transaction like this one actually pays post-Prague.
    function _calldataCost(bytes memory data)
        private
        pure
        returns (uint256 calldataGas, uint256 floorTotal)
    {
        uint256 zeros;
        for (uint256 i = 0; i < data.length; ++i) {
            if (data[i] == 0) ++zeros;
        }
        uint256 nonzeros = data.length - zeros;

        calldataGas = zeros * GAS_PER_ZERO_BYTE + nonzeros * GAS_PER_NONZERO_BYTE;
        uint256 tokens = zeros + nonzeros * TOKENS_PER_NONZERO_BYTE;
        floorTotal = TX_BASE_GAS + tokens * FLOOR_GAS_PER_TOKEN;
    }
}
