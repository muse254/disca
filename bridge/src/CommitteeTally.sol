// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {IDiscaBridge, IDiscaConsumer} from "./IDiscaBridge.sol";

/// @title CommitteeTally — the demo consumer from `docs/bridge.md` §5.
/// @notice A committee posts encrypted ballots, DISCA workers run the tally
/// circuit over the ciphertexts, and the winning candidate comes back as a
/// ciphertext only the committee can decrypt. This contract is what a judge
/// watching the demo sees: private inputs committed on-chain, distributed
/// evaluation attested M-of-N, settlement, and one explicit trust boundary at
/// reveal time.
///
/// @dev The trust boundary is `reveal`, and it is not hidden. The chain holds
/// `keccak256` of a ciphertext; the plaintext exists only after the key holder
/// decrypts with the client key (`docs/architecture.md` §2: single-key model,
/// the key holder is the data owner). Nothing on-chain can check that the
/// revealed winner is what the ciphertext contained. Proving that needs a
/// verifiable decryption or a threshold KMS, both roadmap.
contract CommitteeTally is IDiscaConsumer {
    /// @notice The bridge this tally settles through.
    IDiscaBridge public immutable bridge;

    /// @notice The committee, and the holder of the FHE client key.
    /// @dev The same party in the demo: whoever can decrypt the result is the
    /// only party that could honestly reveal it.
    address public immutable committee;

    /// @notice The registered tally circuit.
    uint256 public immutable programId;

    /// @notice The job posted by the most recent `startTally`, or zero.
    uint256 public jobId;

    /// @notice `keccak256` of the compressed result ciphertext, set by the
    /// bridge callback.
    bytes32 public resultCommit;

    /// @notice The decrypted winner, once the committee has revealed it.
    uint32 public winner;

    /// @notice Whether `reveal` has been called for the current job.
    bool public revealed;

    /// @notice A tally was posted to the bridge.
    event TallyStarted(uint256 indexed jobId, uint256 ballots);

    /// @notice The bridge reported a settled result.
    event TallySettled(uint256 indexed jobId, bytes32 resultCommit);

    /// @notice A refunded escrow left the contract.
    event EscrowWithdrawn(address indexed to, uint256 amount);

    /// @notice The committee published the plaintext winner.
    /// @dev Trusted. See the contract-level note.
    event TallyRevealed(uint256 indexed jobId, uint32 winner);

    error NotCommittee();
    error NotBridge();
    error WrongJob(uint256 reported, uint256 expected);
    error TallyInFlight(uint256 jobId);
    error NoResultYet();
    error AlreadyRevealed();
    error WithdrawFailed();

    /// @param bridge_ The `DiscaBridge` deployment.
    /// @param committee_ The key holder, allowed to start a tally and to reveal.
    /// @param programId_ A program already registered on `bridge_`.
    constructor(IDiscaBridge bridge_, address committee_, uint256 programId_) {
        bridge = bridge_;
        committee = committee_;
        programId = programId_;
    }

    /// @notice Posts the encrypted ballots to the bridge and escrows
    /// `msg.value` for the coordinator.
    /// @param commits `keccak256` of each compressed input ciphertext.
    /// @param blobs The ciphertexts themselves; the bridge emits them for
    /// workers and checks them against `commits`.
    /// @return id The new job id.
    /// @dev One tally at a time. The contract keeps a single `jobId` and a
    /// single `resultCommit`, so starting a second job while the first is
    /// unsettled would silently make the first one's callback unroutable —
    /// `onJobFulfilled` would reject it as the wrong job and take the whole
    /// settlement down with it.
    function startTally(bytes32[] calldata commits, bytes[] calldata blobs)
        external
        payable
        returns (uint256 id)
    {
        if (msg.sender != committee) revert NotCommittee();
        if (jobId != 0 && resultCommit == bytes32(0)) revert TallyInFlight(jobId);

        resultCommit = bytes32(0);
        revealed = false;
        winner = 0;

        id = bridge.submitJob{value: msg.value}(programId, commits, blobs, address(this));
        jobId = id;

        emit TallyStarted(id, commits.length);
    }

    /// @inheritdoc IDiscaConsumer
    /// @dev Both checks matter. `msg.sender == bridge` is what stops anyone
    /// writing a `resultCommit` the workers never signed — this function is the
    /// only path by which a result enters this contract, and the bridge is the
    /// only party that has verified a quorum. `_jobId == jobId` stops a stale
    /// callback from an earlier tally overwriting the current one.
    function onJobFulfilled(uint256 _jobId, bytes32 _resultHash) external {
        if (msg.sender != address(bridge)) revert NotBridge();
        if (_jobId != jobId) revert WrongJob(_jobId, jobId);

        resultCommit = _resultHash;
        emit TallySettled(_jobId, _resultHash);
    }

    /// @notice The committee publishes the decrypted winner.
    /// @param winner_ The winning candidate index.
    /// @dev **Trusted.** The chain cannot check this against `resultCommit`;
    /// see the contract-level note. What it can check, and does, is that a
    /// result was attested at all — revealing before the bridge has settled a
    /// quorum would be a claim about a computation that never finished.
    function reveal(uint32 winner_) external {
        if (msg.sender != committee) revert NotCommittee();
        if (resultCommit == bytes32(0)) revert NoResultYet();
        if (revealed) revert AlreadyRevealed();

        winner = winner_;
        revealed = true;
        emit TallyRevealed(jobId, winner_);
    }

    /// @notice Accepts a refunded escrow back from the bridge.
    /// @dev Without this the escrow is stuck forever, and the bug is quiet
    /// enough to be worth spelling out. `startTally` posts the job, so
    /// `job.poster` is this contract. `refundOnTimeout` pays the poster with a
    /// bare `call`, which fails against a contract that cannot receive value —
    /// and the bridge reverts rather than swallowing it, correctly, since
    /// releasing a job's escrow into nothing would be worse. But the job is
    /// past its deadline by then, so `fulfillJob` refuses too. Both exits are
    /// closed and `bridge.md` §6 promises a refund for exactly this case.
    ///
    /// `CommitteeTally.t.sol` did not catch it because every refund path there
    /// is zero-value or posted by an EOA; `RefundToContractPoster.t.sol` pins
    /// it, with an EOA control so a failure cannot be misread as
    /// `refundOnTimeout` being broken.
    receive() external payable {}

    /// @notice The committee withdraws a refunded escrow.
    /// @param to Where to send it.
    /// @dev `receive` alone would move the problem rather than fix it: the
    /// refund would land here and stay, which is stuck in a nicer place. Gated
    /// on the committee because they funded the job in the first place.
    function withdraw(address payable to) external {
        if (msg.sender != committee) revert NotCommittee();

        uint256 amount = address(this).balance;
        (bool ok,) = to.call{value: amount}("");
        if (!ok) revert WithdrawFailed();

        emit EscrowWithdrawn(to, amount);
    }
}
