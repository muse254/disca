# DISCA Task Checklist

Status: working checklist derived from [architecture.md](architecture.md) and
[bridge.md](bridge.md) against the code as merged in #2.

Timeline: today is 2026-08-18. ETHOnline runs **Sept 4–16** (12 days), so there
are ~17 days of prep first. Track 0 is prep work that de-risks the hackathon;
tracks 1–4 are the build.

Dependency spine — nothing in the bridge track can start until **0.2** lands,
and the demo in architecture.md §10 can't be built until **1.1** lands:

```
0.1 measure ──► sizes the circuits everything else assumes
0.2 bytecode serialization ──► 3.x bridge (needs bytecodeHash)
1.1 opcode expansion ──────► 4.x demo (needs compare/select)
1.2 compressed ct boundary ─► 2.x transport, 3.x calldata
```

---

## Track 0 — Prep (do before Sept 4)

Cheap, unblocks estimation, and none of it burns hackathon hours.

- [ ] **0.1 Commit the measurement harness.** The numbers in architecture.md §2
      came from a `size_probe` example that was never checked in — only its
      build artifacts survive in `target/`. Recreate it as `examples/size_probe.rs`
      so the table is reproducible.
- [ ] **0.2 Re-measure in release mode** and update architecture.md §2. Debug
      i32 mul is ~178 s; release is typically 10–100x faster. This number sets
      the max circuit size for a ≤5 min demo (architecture.md §11 Q1).
- [ ] **0.3 Serialization format for `CircuitOp` / `DiscaFunction`.** Currently
      there is none, so `keccak256(bytecode)` — which the entire bridge design
      pins on-chain — is not computable. Add `serde` + a stable binary encoding
      and a `bytecode_hash()` helper. **Blocks all of Track 3.**
- [ ] **0.4 Speed up the test loop.** `cargo test -p primitives` takes ~8 min in
      debug. Add a `[profile.test]` opt-level bump, or gate the adder truth-table
      tests behind a feature so the fast parser tests stay fast.
- [ ] **0.5 Decide the open questions** in architecture.md §11 (3: attester
      scheme, 4: transport, 5: chain target). The doc already leans address-list
      / HTTP / Anvil — just confirm and strike them.

## Track 1 — Execution core (`primitives/`)

- [ ] **1.1 Opcode expansion.** Add `I32Eq`, `I32Ne`, `I32LtS`, `I32GtS`,
      `I32GeS`, `I32LeS`, `Select`, `LocalSet`, `LocalTee`. FHE comparisons
      return `FheBool`, so `CircuitOp::run` needs a stack value type that is
      either `FheInt32` or `FheBool` rather than today's `Vec<FheInt32>`.
      **Blocks the demo.**
- [ ] **1.2 Compressed ciphertext boundary.** `CompressedFheInt32` is the
      documented wire format (architecture.md §2, bridge.md §1) but appears
      nowhere in code. Add encode/decode helpers at the crate edge; keep
      evaluation on decompressed values.
- [ ] **1.3 Locals as real storage.** `Function.locals` is parsed and then
      discarded; `circuit_sequence()` maps `LocalGet(i)` straight to input index
      `i`. Once `local.set` exists, the evaluator needs a locals frame distinct
      from the input vector.
- [ ] **1.4 Validate stack discipline at compile time.** `run()` errors at
      execution time on underflow. Cheaper to reject a malformed circuit when
      building `DiscaFunction` — and stack-depth-zero points are exactly what
      Phase 2 partitioning (architecture.md §6) needs anyway.
- [ ] **1.5 Tests for the new opcodes** against plaintext reference semantics.

## Track 2 — Node roles (`node/`)

- [ ] **2.1 `--role coordinator|worker`.** `node/src/main.rs` is currently a
      hardcoded single-process demo. Split into role dispatch behind clap.
- [ ] **2.2 Worker HTTP server** — `POST /jobs`, `POST /results` per bridge.md §4.
- [ ] **2.3 Coordinator dispatch** — fan a job to N workers, collect
      `keccak256(result ct)`, compare M-of-N.
- [ ] **2.4 Server key distribution** — coordinator serves `GET /keys/<hash>`,
      workers pull once and cache. Confirm 120 MB is fine locally
      (architecture.md §11 Q2).
- [ ] **2.5 Local 1-coordinator + 3-worker run** with 2-of-3 attestation, no
      chain yet.

## Track 3 — Bridge (`bridge/`, new Foundry project) — needs 0.3

- [ ] **3.1 Foundry scaffold** + `DiscaBridge.sol`: program registry, worker
      registry, job escrow, lifecycle, events (bridge.md §2).
- [ ] **3.2 Forge unit tests** for the job state machine and the M-of-N
      attester check.
- [ ] **3.3 Anvil end-to-end** with a mocked coordinator calling `fulfillJob`.
- [ ] **3.4 Alloy chain watcher** in the coordinator role — subscribe to
      `JobRequested`, validate blob commitments, dispatch, submit `fulfillJob`.
      Adds `alloy` as the first non-tfhe dependency.
- [ ] **3.5 `refundOnTimeout` + disputed-job path** (bridge.md §6).

## Track 4 — Demo + submission — needs 1.1

- [ ] **4.1 Write the tally circuit** in Rust, compile to WASM, confirm it
      parses into `CircuitOp`s and fits the release-mode latency budget from 0.2.
- [ ] **4.2 `CommitteeTally.sol`** consumer contract (bridge.md §5).
- [ ] **4.3 `disca-cli` for real** — `parse` is a no-op (`main.rs:29`) and
      `parser.rs:3` is `todo!()`. Needs at minimum: wasm → bytecode + hash, and
      `keygen`.
- [ ] **4.4 End-to-end demo script** driving register → submit → evaluate →
      fulfill → decrypt.
- [ ] **4.5 Demo video** + README rewrite (it currently documents only the
      whitepaper build, not the project).

## Stretch (only if the above is done)

- [ ] Circuit partitioning at stack-depth-zero points (architecture.md §6 Phase 2).
- [ ] L2 testnet deploy with verified contracts.
- [ ] Optimistic challenge window (architecture.md §7, L1).

## Explicitly out of scope

Public-key / multi-key / threshold FHE; ZK evaluation proofs; slashing
economics; algorithm privacy; QUIC/libp2p; persistent storage.
