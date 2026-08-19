# DISCA Architecture — FHE Coprocessor for Ethereum

Status: draft (pre-hackathon planning for ETHOnline 2026, Sept 4–16, Continuity track)

This document defines the target architecture for turning DISCA from a
single-process FHE evaluator into a **distributed FHE coprocessor for Ethereum**:
smart contracts offload confidential computation to a network of DISCA nodes, and
encrypted results settle back on-chain.

See [bridge.md](bridge.md) for the Ethereum bridge design in depth.

## 1. Positioning

DISCA's core thesis (per the whitepaper) is unchanged: a distributed computer using
FHE, not a blockchain. The Ethereum integration is an **adapter at the edge** — a
bridge contract plus a chain watcher — not a change to the execution core.

Pitch: *"Ethereum contracts can't keep secrets. DISCA lets a contract request
computation over data that no node, validator, or observer ever sees."*

## 2. Measured constraints (tfhe-rs 1.5.0, default params, **release** build)

Reproduce with `cargo run --release -p primitives --example size_probe`
(add `--public-key` for the public-key measurement, which is slow and
memory-hungry). Sizes are `safe_serialize` wire sizes, not in-memory footprints.

| Artifact | Size / latency | Consequence |
|---|---|---|
| Client key (secret) | 23.5 KB | Lives exclusively with the key holder. Never transmitted. |
| Server key (public eval key) | 114.8 MB | Cannot go on-chain. Distributed to nodes out-of-band, registered on-chain by hash. |
| Server key (compressed) | **28.8 MB** | 4x smaller; compresses in 438 ms. **This is what the coordinator should serve to workers.** |
| Public key (public-key encryption mode) | 2.00 GB | **Impractical. Public-key mode is out of scope.** Multi-party input requires multi-key/threshold FHE → roadmap. |
| `FheInt32` ciphertext (uncompressed) | 257.9 KB | Too big for calldata (~4.1M gas/value). Only exists inside the node network. |
| `CompressedFheInt32`, freshly encrypted | 2.3 KB | The input wire format. Calldata-viable (~38k gas on L1, cheaper on L2). Compress/decompress ~1 ms each — negligible. |
| `CompressedFheInt32`, **computed** | **11.8 KB** | A result that has been through the evaluator compresses ~5x worse than a fresh encryption: a fresh ciphertext compresses to a replayable PRNG seed, a computed one has no seed and must carry real coefficients. This is the size of the result blob `fulfillJob` carries. |
| `FheBool` (comparison result) | 16.2 KB | Intermediate only; never crosses the boundary. |
| Key generation | 685 ms | Per-program keygen is effectively free; do it at registration time. |
| i32 add | 225 ms | |
| i32 sub | 265 ms | |
| i32 mul | 2.04 s | ~9x an add. Still worth avoiding in hot loops, but no longer disqualifying. |
| i32 compare (`gt`) | 141 ms | Cheapest op measured. |
| `select` (`if_then_else`) | 200 ms | |

A compare+select pair costs ~341 ms, so a tally over N candidates (N-1 pairs)
runs in roughly `(N-1) x 0.34 s` — a 16-candidate tally is ~5 s of evaluation.
Circuit size is not a binding constraint on the demo.

Hard consequences:

1. **Single-key model for the hackathon.** One key holder per program. The key
   holder is the data owner (or a committee acting as one party). Multi-party
   inputs from mutually distrusting parties require multi-key FHE — explicitly
   future work (cf. Zama's threshold-KMS approach).
2. **Compressed ciphertexts at the boundary, uncompressed inside.** Clients submit
   `CompressedFheInt32`; nodes decompress server-side before evaluation. At ~1 ms
   per conversion this costs nothing.
3. **Privacy guarantee = data privacy only.** Nodes and the chain never see input
   or output plaintext. The *algorithm* (DISCA bytecode) is visible to executing
   nodes; algorithm privacy is roadmap (private function evaluation).
4. **Always build and demo in release.** Debug is 87–98x slower on evaluation
   (i32 add 22 s → 225 ms; mul 178 s → 2.04 s). Every earlier estimate in this
   document was made against debug numbers and understated what fits in a demo.

## 2a. Program constraints (validated against rustc 1.96)

The IR is a **straight-line circuit**: a linear `CircuitOp` sequence over an
operand stack, with no control flow and no addressable memory. FHE cannot branch
on an encrypted condition — a node holds ciphertexts and cannot learn which way
a test went — so this is a property of the model, not a gap in the front end.

Two consequences, both confirmed empirically by `committee-tally/` (see
`primitives/tests/tally_circuit.rs`):

1. **Programs must be compiled with optimizations on.** At `--release` rustc
   flattens `if`/`else`, mutable accumulators, and fixed-size loops into
   `select`, emitting exactly `local.get`, `local.tee`, `i32.gt_s`, `i32.add`
   and `select` — all supported. The same source built with `--debug` is
   *rejected*: unoptimized rustc spills locals to a linear-memory stack frame
   via `global.get $__stack_pointer` and `i32.store`, which a circuit model has
   no way to represent.
2. **Idiomatic Rust is fine, within limits.** A `for` loop over a fixed-size
   array unrolls completely; the loop-written and select-written tallies compile
   to the same circuit. What will *not* work is anything whose trip count or
   memory access depends on a runtime value: a loop over a slice of unknown
   length, indexing by a computed offset, or heap allocation.

The practical rule for demo programs: fixed arity, fixed-size data, no
allocation, built in release.

## 3. Actors and trust model

| Actor | Sees | Does not see | Trust assumption (demo) |
|---|---|---|---|
| Key holder (data owner) | Everything (holds client key) | — | Trusted to custody its own key |
| Consumer contract / job poster | Commitments, result commitment | Plaintext | None needed |
| Coordinator | Ciphertexts, bytecode | Plaintext | Honest liveness (refundable escrow covers failure) |
| Worker nodes | Ciphertexts, circuit segment | Plaintext | ≤ threshold may lie about results; M-of-N attestation catches it |
| Chain / observers | Commitments, bytecode hash, result commitment | Plaintext | — |

### What "M-of-N" means here, and what it does not

**Both letters are counts, not parties.** `N` is how many workers are given the
job. `M` is how many of them must return the *same* answer before the
coordinator accepts it. Nobody is "the M" — it is a threshold.

In `scripts/run-local.sh`: `N = 3` (three worker processes), `M = 2`
(`--attesters 2`). Two workers return hash `0xda91…`, the third returns
`0x5dad…`, two is enough, the job settles on `0xda91…` and the third is logged
as disagreeing. Raise it to `--attesters 3` and that same run fails, because
three workers no longer agree.

Choosing `M` is a trade between two failure modes:

- **Too low** — with `M <= N/2`, two different answers can each reach the
  threshold. The coordinator refuses rather than picking one, because that state
  means more workers are faulty than the scheme can tolerate.
- **Too high** — `M = N` means one slow, crashed or unreachable worker blocks
  every job. The scheme tolerates `N - M` faulty or missing workers, so `M = 2`
  of `N = 3` tolerates exactly one.

The two mechanisms that share this name:

| | **Replicate and vote** (DISCA today) | **Split a secret** (roadmap) |
|---|---|---|
| Idea | All `N` do the whole job; accept an answer once `M` of them return it | Work is divided so fewer than `M` parties can do nothing at all |
| Answers | "Did the worker compute correctly?" | "Can one party decrypt on its own?" |
| Cost | `N`× compute | Coordination |
| Familiar as | triple modular redundancy in avionics, Byzantine fault tolerance, oracle networks | Shamir's secret sharing, threshold signatures, Bitcoin multisig |

L0 (§7) is the first kind. Zama's fhEVM uses the same construction —
`sns-worker` hashes the serialized ciphertext and a contract majority-votes the
digest.

The multi-key / threshold FHE in §2's roadmap is the second kind. It would
remove the single key holder so no one party can decrypt, and it says **nothing**
about whether a worker evaluated correctly. It is not an upgrade path from L0; a
system with threshold decryption still needs L0, L1 or L2 on top. The rungs that
actually replace replication are L1 (optimistic challenge) and L2 (ZK proof),
because those verify computation rather than counting agreement.

**Deterministic evaluation property — holds, but only once the FFT plan is
pinned.**

M-of-N in §7 assumes two honest workers executing the same circuit over the same
input ciphertexts produce *byte-identical* results, so agreement on
`keccak256(result)` is evidence of correct evaluation. Out of the box that is
false, and the reason is not what it looks like.

**It is not randomness.** `Fft::new` builds every plan with
`Method::Measure(Duration::from_millis(10))`: tfhe-rs times several
numerically-equivalent FFT algorithms at first use and keeps whichever won *on
that process, under that machine load*. The algorithms associate the
floating-point butterflies differently, a few torus coefficients round the other
way, and the ciphertext differs while the plaintext does not. The chosen plan is
cached in a `OnceLock`, so a process is self-consistent forever — which is
exactly why an in-process determinism test passes while separate workers
disagree, and why a whole round of concurrent workers can drift together (they
benchmark under the same contention).

**The fix is one call before anything touches a key**, `pin_fft_plan` in
`node/src/main.rs`, using the public `setup_custom_fft_plan` demonstrated in
tfhe-rs's own `examples/manual_fft.rs`. Measured on the demo circuit, three
concurrent processes with identical keys and inputs:

| | rounds unanimous (of 6) | demo settle rate |
|---|---|---|
| unpinned | 1 | 2 of 8 |
| **pinned** | **6** | **6 of 6** |

No measurable slowdown. With it pinned, the `attestation disagreement` warning
fires only on the genuinely faulty worker; unpinned it accused honest workers
more often than the faulty one.

Three limits to keep in view:

1. **Per polynomial size.** The call covers the 2048 used by
   `ConfigBuilder::default()`. Changing parameters means pinning again.
2. **Same architecture only.** Zama document that outputs differ between x86 and
   ARM, so byte equality is an ISA-homogeneity assumption, not a theorem. A
   mixed-ISA worker fleet will disagree no matter what is pinned — treat
   homogeneity as a registration requirement (task 2.10d).
3. **Call it early.** `setup_custom_fft_plan` panics if the plan for that size is
   already initialised, and decompressing a server key initialises it.

Compression is deterministic (`compression_is_deterministic`). The attested hash
covers the compressed blob so the bridge contract can verify what it emits
(bridge.md §5a).

Prior art worth knowing: Zama's own fhEVM does exactly this — `sns-worker`
hashes the serialized ciphertext and `CiphertextCommits.sol` majority-votes the
digest, with drift detection and N≥3. Our L0 is their scheme.

## 4. System overview

```
            ┌─────────────────────────── Ethereum ───────────────────────────┐
            │                                                                │
            │   DiscaBridge.sol                Consumer contract (demo)      │
            │   - program registry             - submits job, escrows fee    │
            │   - job escrow + lifecycle  ◄─── - receives JobFulfilled cb    │
            │   - M-of-N result attestation    - stores result commitment    │
            │                                                                │
            └───────────────▲────────────────────────────▲───────────────────┘
                     events │                            │ fulfillJob tx
                            │                            │
   off-chain store  ┌───────┴────────────────────────────┴────────┐
   (compressed cts) │                COORDINATOR                  │
   server keys ────►│  - watches JobRequested events              │
                    │  - validates ct commitments vs blobs        │
                    │  - dispatches circuit to N workers          │
                    │  - collects result hashes, submits result   │
                    └───────▲──────────────▲──────────────▲───────┘
                            │              │              │
                        ┌───┴───┐      ┌───┴───┐      ┌───┴───┐
                        │WORKER │      │WORKER │      │WORKER │
                        │srv key│      │srv key│      │srv key│
                        │eval   │      │eval   │      │eval   │
                        └───────┘      └───────┘      └───────┘

   key holder (off-chain): keygen, encrypts inputs (compressed), decrypts result
```

## 5. Job lifecycle

1. **Register program.** Developer compiles Rust → WASM → DISCA bytecode via
   `disca-cli`; calls `registerProgram(bytecodeHash)`. Program ID assigned.
2. **Register key.** Key holder generates (client key, server key) for the
   program; uploads the 120 MB server key to the coordinator off-band; registers
   `serverKeyHash` on-chain.
3. **Submit job.** Data owner encrypts inputs → `CompressedFheInt32` blobs; job
   poster calls `submitJob(programId, inputCommitments[], inputBlobs, callback)`
   with fee escrow. Contract emits `JobRequested`. (Blobs in calldata/events for
   the demo; off-chain store + CID as scaling path.)
4. **Coordinate.** Coordinator validates each blob against its on-chain
   commitment, decompresses, dispatches (circuit, inputs) to N workers.
5. **Evaluate.** Each worker runs the full circuit deterministically; returns
   `keccak256(resultCiphertext)` to the coordinator.
6. **Aggregate.** If ≥ M of N hashes match, coordinator calls
   `fulfillJob(jobId, resultHash, resultBlob)`; contract verifies the attester
   set, releases escrow, callbacks the consumer.
7. **Decrypt.** Key holder fetches the result ciphertext, decrypts locally.
   Plaintext never touches the chain or any node. (Optional trusted `reveal()`
   for demos needing public output — signed by the key holder, clearly labeled.)

Failure handling: coordinator/ worker liveness failure → `refundOnTimeout`.
Hash disagreement → job marked disputed, escrow refunded (slashing is roadmap).

## 6. Distribution model

- **Phase 1 (hackathon): redundant full execution.** Every worker evaluates the
  whole circuit. Distribution exists for *verifiability* (M-of-N attestation),
  not scale. Simple, honest, demoable on one machine with 3 processes.
- **Phase 2 (stretch): circuit partitioning.** Split the linear `CircuitOp`
  sequence at stack-depth-zero points; intermediate ciphertexts flow between
  workers through the coordinator. Distribution for *scale*. Only if time allows.

## 7. Verification ladder (trust roadmap)

| Level | Mechanism | Status |
|---|---|---|
| L0 | M-of-N worker attestation on deterministic execution + trusted key-holder reveal | Hackathon deliverable |
| L1 | Optimistic challenge window: anyone re-executes and disputes a result hash | Stretch |
| L2 | ZK proof of correct homomorphic evaluation; threshold-FHE decryption; stake/slashing | Roadmap (whitepaper alignment) |

## 8. Workspace layout

```
disca/
  primitives/           # IR, CircuitOp set, evaluator, bytecode, wire format
    src/program.rs      #   WASM -> IR -> CircuitOp, stack-machine evaluation
    src/validate.rs     #   static circuit checks + partitioning split points
    src/bytecode.rs     #   canonical encoding + keccak256 program hash
    src/wire.rs         #   ciphertext boundary, commitments, SealedResult
    examples/           #   size_probe, inspect, cross_process, key_probe
  node/                 # coordinator | worker | demo roles
    src/coordinator.rs  #   job prep, fan-out, M-of-N aggregation, key holder
    src/worker.rs       #   validate, evaluate, seal, report
    src/protocol.rs     #   messages; src/transport.rs — sync HTTP
  disca-cli/            # parse wasm->bytecode; keygen (still stubs)
  committee-tally/      # demo circuit; simple-arithmetic/ — sample program
  scripts/run-local.sh  # 1 coordinator + 3 workers, one deliberately faulty
  docs/                 # this file, bridge.md, attestation.md, tasks.md,
                        # tfhe-determinism-request.md
```

Not yet built: `bridge/` (Foundry project — Track 3) and the chain watcher.

## 9. Scope fence for the 12 days

**In:** opcode expansion (comparisons, select, `local.set/tee`); coordinator +
worker roles with minimal transport; bridge contract with escrow + M-of-N
attestation; one end-to-end demo; release-build measurements; demo video.

**Out:** public-key/multi-key/threshold FHE; ZK evaluation proofs; circuit
partitioning (stretch); slashing economics; algorithm privacy; production
transport (QUIC/libp2p); persistent storage layer.

## 10. Demo candidate

**Confidential committee tally** (recommended): a grants/DAO committee (single
key holder) encrypts N vendor scores as compressed ciphertexts, posts the job
on-chain; the network evaluates a compare/select tally circuit compiled from
Rust→WASM; result commitment settles on-chain; committee decrypts the winner.
Multi-input, visually clear, honest single-key privacy story.

Alternative: **confidential data-owner scoring** — a user encrypts private
attributes; a registered scoring program runs over them; only the user decrypts.
Purest privacy story, less visual.

## 11. Open questions (resolve week 1)

1. Release-mode op latency → max circuit size for a ≤5 min demo.
2. Server key distribution: coordinator-served HTTP pull by workers (by hash) —
   confirm 120 MB is manageable in the demo environment (yes on LAN/local).
3. Attestation scheme: on-chain registered worker address list (simpler) vs
   ECDSA signature aggregation (cheaper calldata). Leaning: address list.
4. Transport: HTTP/JSON with bincode payloads (simplest) vs gRPC. Leaning: HTTP.
5. Chain target for the video: local Anvil (deterministic) vs L2 testnet
   (more impressive). Plan: Anvil for dev, testnet deploy as stretch.
