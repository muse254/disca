# DISCA / Ethereum Bridge Design

Status: draft (pre-hackathon planning for ETHOnline 2026)

The bridge is what makes DISCA an **FHE coprocessor for Ethereum**: contracts
request confidential computation; DISCA nodes execute it on ciphertexts; result
commitments settle back on-chain. This document specifies the on-chain/off-chain
boundary, the contract interface, key lifecycle, and settlement mechanics.

Parent document: [architecture.md](architecture.md) (measured constraints, trust
model, scope fence). All numbers below come from the tfhe-rs 1.5.0 measurements
recorded there.

## 1. Boundary rule

**On-chain: commitments, metadata, escrow, attestations.**
**Off-chain: keys and (decompressed) ciphertext bodies.**

| Data | Placement | Why |
|---|---|---|
| Program bytecode (DISCA ops) | Off-chain store; `keccak256` hash on-chain | Arbitrary size; nodes need the full circuit |
| Server key (114.8 MB, or 28.8 MB compressed) | Off-chain (coordinator-served compressed, workers pull by hash); `keccak256` on-chain | Impossible on-chain |
| Client key (23.5 KB, secret) | Key holder only, never transmitted | It is the privacy boundary |
| Input ciphertexts | `CompressedFheInt32` (2.3 KB/value) in calldata/event; `keccak256` commitment in storage | Calldata-viable; on-chain availability proves inputs unchanged |
| Working ciphertexts (257.9 KB/value, decompressed) | Node memory only | ~4.1M gas/value on-chain; pointless |
| Result ciphertext | Compressed form **emitted on-chain**; `keccak256` of those same bytes stored as the attestation (see §5a) | Contract can verify blob against hash; key holder fetches and decrypts locally |
| Escrow, job state, attester set | On-chain | It is the settlement layer |

Gas sketch for a 3-input job (L1 constants; cheaper on L2):

- Input blobs via event data: 3 x 2.3 KB x ~8 gas/byte = ~57k gas
- Commitments in storage: 3 x 32 B slots = ~60k gas
- `fulfillJob` (result hash + attester list + compressed result blob **11.8 KB**,
  paid once as calldata at ~16 gas/byte and once as event data at ~8 gas/byte):
  order 250-350k gas

The result blob dominates, and it is 5x larger than an input blob: a freshly
encrypted ciphertext compresses to a replayable PRNG seed, whereas a computed
one has no seed and must carry real coefficients (11.8 KB measured, see
architecture.md §2). That still lands inside the range of an ordinary DeFi
transaction on L1, and is negligible on the L2 and Anvil targets in §7 — but it
is the number to watch if job results ever grow beyond a single `i32`.

## 2. Contract interface (DiscaBridge.sol)

```solidity
// SPDX-License-Identifier: MIT OR Apache-2.0
pragma solidity ^0.8.26;

/// Minimal sketch - final signatures may shift during implementation.
interface IDiscaBridge {
    struct Job {
        uint256 programId;
        address poster;
        address callback;        // consumer contract, or zero
        bytes32[] inputCommits;  // keccak256 of each compressed input ct
        uint64 deadline;
        uint256 escrow;
        JobState state;          // Open | Fulfilled | Refunded | Disputed
    }

    // --- registration ---
    function registerProgram(bytes32 bytecodeHash, bytes32 serverKeyHash,
                             uint8 attestersRequired) external returns (uint256 programId);
    function registerWorker(address worker) external;             // owner-gated in demo

    // --- job lifecycle ---
    /// inputBlobs are the compressed ciphertexts; emitted in JobRequested so
    /// nodes can fetch them; only commitments persist in storage.
    function submitJob(uint256 programId, bytes32[] inputCommits, bytes[] inputBlobs,
                       address callback) external payable returns (uint256 jobId);

    /// Callable by the coordinator once >= attestersRequired workers report the
    /// same resultHash. Releases escrow, callbacks the consumer.
    function fulfillJob(uint256 jobId, bytes32 resultHash, bytes resultBlob,
                        address[] attesters) external;

    function refundOnTimeout(uint256 jobId) external;

    // --- events ---
    event JobRequested(uint256 indexed jobId, uint256 indexed programId,
                       bytes32[] inputCommits, bytes[] inputBlobs, address callback);
    event JobFulfilled(uint256 indexed jobId, bytes32 resultHash, bytes resultBlob);
}
```

Design notes:

- **Attestation = registered address list, not ECDSA signatures.** The demo
  keeps an owner-gated worker registry; `fulfillJob` requires `attesters` to be
  distinct registered workers, length >= `attestersRequired`. Simpler and
  cheaper than on-chain signature verification. Signature-based attestation is
  an L1-ladder upgrade.
- **The contract never verifies computation cryptography.** It verifies that M
  of N registered workers agreed on `keccak256(resultCiphertext)`. Because FHE
  evaluation is deterministic, agreement implies correct evaluation with at most
  N-M Byzantine workers tolerated (for the parts each worker ran).
- **Escrow pays the coordinator on fulfillment** and refunds the poster on
  timeout. No slashing in the demo; dishonest workers only waste their own time
  since they cannot forge agreement.

## 3. Key lifecycle

One program, one keypair (demo model):

1. Key holder runs `disca-cli keygen --program <bytecode>` producing
   (client key, server key) locally. 685 ms measured in release.
2. Server key is compressed (28.8 MB, 438 ms) and pushed to the coordinator
   off-band; coordinator exposes it at `GET /keys/<serverKeyHash>` for workers
   to pull. Workers decompress once at startup.
3. `registerProgram(bytecodeHash, serverKeyHash, M)` pins both hashes on-chain.
4. Key holder encrypts inputs with the client key into
   `CompressedFheInt32` blobs (2.3 KB each) - these are the only bytes that
   ever cross the bridge.
5. Nodes decompress with the server key, evaluate, and the coordinator
   re-compresses the result for the return trip.
6. Key holder decrypts locally. Nothing else in the system can.

Explicitly rejected for the demo:

- **Public-key encryption mode** (2.00 GB key measured): unusable. Multi-party
  inputs from mutually distrusting parties therefore require multi-key or
  threshold FHE - roadmap, alongside Zama-style threshold KMS.
- **Server key on-chain or in IPFS-as-only-source**: too large for the former;
  hash-pinned coordinator storage is sufficient for the latter's guarantees.

## 4. Off-chain protocol (coordinator <-> workers)

Minimal HTTP/JSON control plane with bincode bodies:

- `POST /jobs` (coordinator -> worker): program bytecode, server key hash,
  decompressed input ciphertexts, job id.
- `POST /results` (worker -> coordinator): job id, `keccak256(result ct)`,
  worker id.
- `GET /result/<jobId>` (key holder -> coordinator): compressed result
  ciphertext after `JobFulfilled`.

Worker flow: pull server key once by hash -> for each job, decompress inputs,
execute the `CircuitOp` sequence, hash result, report. Identical binaries run
with `--role coordinator|worker`; the demo runs 1 coordinator + 3 workers as
local processes (2-of-3 attestation).

Chain watcher (coordinator): subscribes to `JobRequested`, validates blob
count/commitments, dispatches, then submits `fulfillJob`. Implementation:
alloy (Rust) with a simple polling loop - no indexing infra needed.

## 5. Consumer contract (demo): committee tally

```solidity
contract CommitteeTally {
    IDiscaBridge bridge;
    uint256 public programId;      // registered tally circuit
    uint256 public jobId;
    bytes32 public resultCommit;   // set via callback

    function startTally(bytes32[] memory commits, bytes[] memory blobs)
        external payable
    {
        jobId = bridge.submitJob{value: msg.value}(
            programId, commits, blobs, address(this));
    }

    function onJobFulfilled(uint256 _jobId, bytes32 resultHash) external {
        require(msg.sender == address(bridge));
        require(_jobId == jobId);
        resultCommit = resultHash;   // committee decrypts off-chain, reveals
    }
}
```

The committee (key holder) then decrypts the result ciphertext and calls a
plain `reveal(winner)` marked as a trusted reveal in the demo script. Judges
see: private inputs committed on-chain, distributed FHE execution attested
2-of-3, settlement, and an explicit trust boundary at reveal time.

## 5a. Which bytes the result hash covers

**Decided: the attested hash covers the compressed result** — the same blob the
contract emits. `fulfillJob` therefore requires

```solidity
require(keccak256(resultBlob) == resultHash, "blob does not match attestation");
```

so the ciphertext the key holder retrieves is provably the one M-of-N workers
attested to.

The alternative — hashing the uncompressed result — commits to bytes that never
leave a worker and that no verifying party can obtain, which leaves a
coordinator free to publish a genuinely-attested hash beside a substituted or
garbage blob. Nothing on-chain would contradict it, and only the key holder
would ever find out, after decrypting, with no evidence to dispute with.

The cost is that attestation now depends on compression being deterministic as
well as evaluation. Two points make that acceptable:

- Both properties are verified in `primitives/src/wire.rs`
  (`results_are_deterministic`, `compression_is_deterministic`).
- If either broke, honest workers would report different hashes and the job
  would fail to reach agreement — a timeout and refund, caught the first time
  three workers run. It cannot silently yield a wrong answer.

`primitives::wire::SealedResult` bundles the blob with its hash so the two
cannot be handled separately.

### What B costs, and when to revisit it

Emitting the blob costs roughly 100-200k gas that option A would have skipped.
On the Anvil and L2 targets in §7 that is zero or cents, and on L1 it keeps
`fulfillJob` within the range of an ordinary transaction (a Uniswap V3 swap is
~150k), so the guarantee is worth paying for today.

**The constraint to watch: 11.8 KB is the cost of a single `i32` result.**
Results are compressed ciphertexts, so calldata grows linearly with the number
of output values. A job returning ten values would carry ~118 KB, which is not
an ordinary transaction on any chain. B holds while results stay small; it is
not a design that scales to large outputs.

If that changes — multi-value results, ranked outputs, anything beyond a handful
of ciphertexts — the fallback is:

**Option C: attest to `keccak256(compressed result)` as in B, but do not emit
the blob.** The key holder fetches it from the coordinator and checks the hash
against the on-chain value themselves. That keeps A's gas cost and most of B's
verifiability. What it gives up is atomicity: the key holder can *detect*
substitution but cannot prove on-chain which blob they were handed, so escrow
still releases before result availability is established. C is right only once
gas becomes binding, which it is not at present.

The three options in one line each:

| | Contract can verify blob | Escrow release atomic with availability | `fulfillJob` gas |
|---|---|---|---|
| A | no | no | ~70-120k |
| B (current) | yes | yes | ~250-350k |
| C | no (key holder can) | no | ~70-120k |

**Consequence for §1:** emitting the compressed result on-chain is now
*required*, not optional. The guarantee is the contract checking the emitted
blob against the attested hash, which does nothing if the blob is never
emitted.

## 6. Failure modes

| Failure | Handling (demo) |
|---|---|
| Coordinator goes silent | `refundOnTimeout` returns escrow |
| Worker hash mismatch | Job marked Disputed; escrow refunded; off-chain rerun |
| Malformed input blob | Coordinator rejects pre-dispatch; commitment check on-chain prevents substitution |
| Key holder loses client key | Result undecryptable by design; documented, not "handled" |
| Result withheld by coordinator | Key holder cannot decrypt - same path as coordinator silence (refund); ciphertext availability via event emission mitigates |

## 7. Chain targets

- **Development + demo video: local Anvil** (Foundry). Deterministic, fast,
  scriptable end-to-end in the demo video.
- **Stretch: an L2 testnet deploy** (cheap calldata makes the input-blob design
  essentially free). Nice for the submission; not required.

## 8. Build order (bridge workstream)

1. Foundry scaffold in `bridge/`: `DiscaBridge.sol` with registry, job
   lifecycle, escrow, events; unit tests (forge).
2. Anvil end-to-end with a mocked coordinator submitting `fulfillJob`.
3. Alloy watcher in `node` (coordinator role) wired to real events.
4. `CommitteeTally.sol` + demo script (cast/forge script) driving the full
   lifecycle for the video.
5. Stretch: L2 testnet deploy + verified contracts.
