# DISCA / Ethereum Bridge Design

Status: draft.

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
| Escrow, job state, attester signatures | On-chain | It is the settlement layer; the signatures are what make the attester set checkable rather than asserted (§2a) |

Gas sketch for a 3-input job (L1 constants; cheaper on L2):

- Input blobs via event data: 3 x 2.3 KB x ~8 gas/byte = ~57k gas
- Commitments in storage: 3 x 32 B slots = ~60k gas
- `fulfillJob` (result hash + attester signatures + compressed result blob
  **11.8 KB**, paid once as calldata at ~16 gas/byte and once as event data at
  ~8 gas/byte): order 250-350k gas. The signatures add ~10k of that for a 3-of-N
  job — 65 bytes and one `ecrecover` (~3k gas) each — which is what §2b is about

The result blob dominates, and it is 5x larger than an input blob: a freshly
encrypted ciphertext compresses to a replayable PRNG seed, whereas a computed
one has no seed and must carry real coefficients (11.8 KB measured, see
architecture.md §2). That still lands inside the range of an ordinary DeFi
transaction on L1, and is negligible on the L2 and Anvil targets in §7 — but it
is the number to watch if job results ever grow beyond a single `i32`.

## 2. Contract interface (DiscaBridge.sol)

```solidity
// SPDX-License-Identifier: MIT
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

    /// One worker's signature over the claim in §2a. `v` is 27 or 28.
    struct Attestation { bytes32 r; bytes32 s; uint8 v; }

    /// Callable by the coordinator once >= attestersRequired workers have each
    /// SIGNED for the same resultHash. Releases escrow, callbacks the consumer.
    ///
    /// The caller supplies signatures, not addresses: the contract recovers
    /// each signer itself. See §2a for what it must check.
    function fulfillJob(uint256 jobId, bytes32 resultHash, bytes resultBlob,
                        Attestation[] attestations) external;

    function refundOnTimeout(uint256 jobId) external;

    // --- events ---
    event JobRequested(uint256 indexed jobId, uint256 indexed programId,
                       bytes32[] inputCommits, bytes[] inputBlobs, address callback);
    event JobFulfilled(uint256 indexed jobId, bytes32 resultHash, bytes resultBlob);
}
```

Design notes:

- **Attestation = recovered ECDSA signatures, not a supplied address list.**
  *Changed by task 2.10i; the original design and why it was wrong are recorded
  in §2b.* Each worker holds a secp256k1 keypair and signs the claim in §2a.
  `fulfillJob` recovers a signer per attestation and requires the recovered
  addresses to be distinct, registered, and at least `attestersRequired` in
  number. The registry stays owner-gated in the demo.
- **The contract never verifies computation cryptography.** It verifies that M
  of N registered workers agreed on `keccak256(resultCiphertext)`. Because FHE
  evaluation is byte-reproducible *under the conditions in `architecture.md`
  §3*, agreement implies correct evaluation with at most N-M Byzantine workers
  tolerated (for the parts each worker ran).

  Those conditions are load-bearing and belong in the worker registry, not just
  in prose: the FFT plan must be pinned, every worker must share one CPU
  architecture (x86 and ARM produce different bytes), and evaluation must be on
  CPU rather than GPU. A worker that violates any of them disagrees with honest
  workers while behaving honestly. Registration should record architecture and
  reject a mismatch (task 2.10b); until it does, disagreement is evidence of
  *divergence*, not of dishonesty, and must not feed slashing.
- **Escrow pays the coordinator on fulfillment** and refunds the poster on
  timeout. No slashing in the demo; dishonest *workers* only waste their own
  time, since they cannot make other workers agree with them.

- **A dishonest coordinator can forge agreement outright, and this interface
  does not prevent it.** Attestations are unsigned: a `SealedResult` is a blob
  and `keccak256(blob)`, computable by anyone. The coordinator supplies the
  `attesters` array, so the contract can verify only that those addresses are
  registered and distinct — not that they ever saw the job. Nothing contradicts
  a fabricated set: the workers signed nothing, and the key holder cannot tell a
  wrong plaintext from a right one. **`fulfillJob` must not be implemented
  against this signature.** Workers need signing keys and the contract needs to
  `ecrecover` each attestation (task 2.10i); the cost is roughly 3.5k gas per
  attester against a 250-350k transaction, which reverses the reasoning in
  architecture.md §11 Q3.

## 2a. What a worker signs, and what the contract must check

Implemented in `primitives/src/attest.rs` (task 2.10i). The coordinator performs
exactly the same recovery off-chain before it settles, so a coordinator that
submits a bad attester set produces a transaction that reverts rather than a
claim nothing can check.

**A worker is an Ethereum address.** The last 20 bytes of `keccak256` over the
uncompressed public key, exactly as an EOA is, so `registerWorker(address)`
needs no mapping table and an operator can hold the key in any Ethereum wallet.

**The signed claim.** Signing a bare `resultHash` would let a signature be
lifted from one job onto another that produced the same bytes — with a
deterministic evaluator over a small result space, not a remote coincidence.
The preimage is fixed-width throughout, so concatenation is injective and no
length prefixes are needed:

```text
offset  len  field
     0   22  "DISCA/attest/result/v1"   ASCII domain tag, no length prefix
    22    8  jobId                      big-endian uint64
    30   32  bytecodeHash               keccak256(DISCA bytecode)
    62   32  resultHash                 keccak256(compressed result), i.e. §5a
    94       total
```

```solidity
bytes32 inner = keccak256(abi.encodePacked(
    "DISCA/attest/result/v1", uint64(jobId), bytecodeHash, resultHash));
bytes32 digest = MessageHashUtils.toEthSignedMessageHash(inner);   // EIP-191
address signer = ecrecover(digest, a.v, a.r, a.s);
```

`abi.encodePacked` writes a string literal without a length prefix and a
`uint64` as 8 big-endian bytes, which is why the layout above is reproducible
on-chain in one line.

**EIP-191, not EIP-712.** The `0x19` prefix means an attestation is structurally
not an RLP-encoded transaction, so a worker key signing attestations cannot be
tricked into signing a transfer; it also lets an operator keep the key behind a
`personal_sign` interface (wallet, HSM, KMS) rather than one that signs raw
digests. EIP-712 is the better answer and is the intended successor, because it
binds a chain id and the verifying contract address — which is what stops an
attestation minted for one `DiscaBridge` deployment being replayed against
another, or against the same contract on a forked chain. Neither value exists
until step 1 of §8 lands. The `/v1` suffix is how that migration stays loud.

**What `fulfillJob` must do.** In order:

1. `require(keccak256(resultBlob) == resultHash)` — §5a, unchanged.
2. Reconstruct `digest` as above from the job's stored `programId ->
   bytecodeHash`, the `jobId`, and the submitted `resultHash`. **Never from
   anything in the calldata that the coordinator could vary.**
3. For each attestation: `require(a.v == 27 || a.v == 28)` and reject high `s`
   (EIP-2), then `ecrecover`. Reject `address(0)` — `ecrecover` returns it on
   failure rather than reverting, and treating that as a signer is the classic
   way to accept a forged signature.
4. Require recovered addresses to be **distinct** and each `isRegisteredWorker`.
   Requiring strictly ascending addresses makes distinctness an O(n) check with
   no storage; the off-chain coordinator already sorts its attester set for this
   reason.
5. `require(count >= attestersRequired)`.

Note what step 4 does *not* require: that the bridge dispatched to those
workers. It never knew. An attestation is valid because of who signed it, not
because of who was asked.

**What this still does not prove.** That the signer evaluated anything. A worker
can sign a wrong answer, and this is exactly what M-of-N is for; and because
divergence is currently indistinguishable from dishonesty (see the note above
and task 2.10b), a signed minority answer is evidence of *disagreement*, not of
fraud, and must not feed slashing. What signatures buy is that the disagreement
now has a name attached that nobody else could have written.

## 2b. Superseded: the address-list design

Recorded because the reasoning that produced it is easy to re-derive.

The original interface was `fulfillJob(..., address[] attesters)`, justified as
"simpler and cheaper than on-chain signature verification", with signature-based
attestation deferred as an L1-ladder upgrade (`architecture.md` §11 Q3 leaned
the same way).

It does not work. The attester array came from the coordinator, so the contract
could verify those addresses were distinct and registered — and nothing else.
Any M registered addresses could be named beside any `resultHash`. The check
looked like verification and was a formatting requirement.

Nothing downstream catches it either: the key holder decrypts a plaintext and
cannot tell a wrong one from a right one, which is the whole reason the job was
worth paying for. So the failure is silent and permanent. That is a strictly
worse property than having no attestation field at all, which would at least be
honest about what is being trusted.

The cost of fixing it is one `ecrecover` (~3,000 gas) plus 65 bytes of calldata
per attester — order 10k gas for a 3-of-N job, against the ~250-350k `fulfillJob`
already costs for the result blob. The "cheaper" option was cheaper by about 3%,
in exchange for the property the contract exists to provide.

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
- `POST /results` (worker -> coordinator): job id, the sealed result, and the
  worker's `(r, s, v)` over the §2a claim. The worker id in the message is a log
  label; the coordinator counts the address it recovers, not the name it is
  given.
- `GET /result/<jobId>` (key holder -> coordinator): compressed result
  ciphertext after `JobFulfilled`.

Worker flow: pull server key once by hash -> for each job, decompress inputs,
execute the `CircuitOp` sequence, seal the result, sign the §2a claim, report.
Identical binaries run with `--role coordinator|worker`; the demo runs 1
coordinator + 3 workers as local processes (2-of-3 attestation).

A failure report is signed too, under `"DISCA/attest/failed/v1"` with
`keccak256(reason)` in the result-hash slot. It never counts towards a quorum —
a worker that could not evaluate has attested to nothing — but an *unsigned*
failure would let anyone who can reach the coordinator manufacture evidence
against an honest operator, which matters as soon as registration or reputation
depends on who failed. The two tags are the same length and differ in one word,
so neither claim can ever be read as the other.

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
