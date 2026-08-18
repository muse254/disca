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

## 2. Measured constraints (tfhe-rs 1.5.0, default params, debug build)

These numbers drive every placement decision below. Re-measure in release mode
during hackathon week 1 (release is typically 10–100x faster for evaluation).

| Artifact | Size / latency | Consequence |
|---|---|---|
| Client key (secret) | ~24 KB | Lives exclusively with the key holder. Never transmitted. |
| Server key (public eval key) | ~120 MB | Cannot go on-chain. Distributed to nodes out-of-band, registered on-chain by hash. |
| Public key (public-key encryption mode) | ~2.1 GB | **Impractical. Public-key mode is out of scope.** Multi-party input requires multi-key/threshold FHE → roadmap. |
| `FheInt32` ciphertext (uncompressed) | ~263 KB | Too big for calldata (~4.2M gas/value). Only exists inside the node network. |
| `CompressedFheInt32` ciphertext | ~1.6 KB | Calldata-viable (~26k gas on L1, cheaper on L2). **This is the on-chain wire format.** |
| Key generation | ~7 s | Per-program keygen is cheap enough to do at registration time. |
| i32 add (debug) | ~22 s | Demo circuits must be small; build demos in release mode. |
| i32 mul (debug) | ~178 s | Avoid multiplication in demo circuits; prefer compare/select patterns. |

Hard consequences:

1. **Single-key model for the hackathon.** One key holder per program. The key
   holder is the data owner (or a committee acting as one party). Multi-party
   inputs from mutually distrusting parties require multi-key FHE — explicitly
   future work (cf. Zama's threshold-KMS approach).
2. **Compressed ciphertexts at the boundary, uncompressed inside.** Clients submit
   `CompressedFheInt32`; nodes decompress server-side before evaluation.
3. **Privacy guarantee = data privacy only.** Nodes and the chain never see input
   or output plaintext. The *algorithm* (DISCA bytecode) is visible to executing
   nodes; algorithm privacy is roadmap (private function evaluation).

## 3. Actors and trust model

| Actor | Sees | Does not see | Trust assumption (demo) |
|---|---|---|---|
| Key holder (data owner) | Everything (holds client key) | — | Trusted to custody its own key |
| Consumer contract / job poster | Commitments, result commitment | Plaintext | None needed |
| Coordinator | Ciphertexts, bytecode | Plaintext | Honest liveness (refundable escrow covers failure) |
| Worker nodes | Ciphertexts, circuit segment | Plaintext | ≤ threshold may lie about results; M-of-N attestation catches it |
| Chain / observers | Commitments, bytecode hash, result commitment | Plaintext | — |

**Deterministic evaluation property:** tfhe-rs evaluation is deterministic given
the same input ciphertexts (randomness exists only at encryption time). Two honest
workers executing the same circuit on the same inputs produce *byte-identical*
result ciphertexts. This makes M-of-N result-hash matching a meaningful, cheap
correctness check without any ZK machinery.

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

## 8. Workspace layout (post-hackathon)

```
disca/
  primitives/     # IR, CircuitOp set, (stretch) partitioning — pure, no I/O
  node/           # binary: --role coordinator|worker; transport + chain watcher
  bridge/         # Foundry project: DiscaBridge.sol, demo consumer, scripts
  disca-cli/      # parse wasm→bytecode; keygen; register helpers
  simple-arithmetic/  # sample program (add real demo programs alongside)
  docs/           # this file, bridge.md
```

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
