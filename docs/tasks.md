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

- [x] **0.1 Commit the measurement harness.** The numbers in architecture.md §2
      came from a `size_probe` example that was never checked in — only its
      build artifacts survive in `target/`. Recreate it as `examples/size_probe.rs`
      so the table is reproducible.
- [x] **0.2 Re-measure in release mode** — done; architecture.md §2 and
      bridge.md §1/§3 now carry release numbers. Release is 87–98x faster
      (add 22 s → 225 ms, mul 178 s → 2.04 s). **Circuit size is no longer a
      binding constraint on the demo**; architecture.md §11 Q1 is answered.
- [x] **0.3 Serialization format for `CircuitOp` / `DiscaFunction`.** Done:
      `primitives/src/bytecode.rs` — magic+version header, pinned bincode
      config (fixint / little-endian / reject-trailing), `bytecode_hash()`
      returning keccak256. **Track 3 is unblocked.**
- [x] **0.4 Speed up the test loop.** Done: `[profile.dev.package."*"]` and
      `[profile.test.package."*"]` at `opt-level = 3` in the root manifest.
      Test suite went from **464.70 s to 4.70 s**.
- [ ] **0.5 Decide the open questions** in architecture.md §11 (3: attester
      scheme, 4: transport, 5: chain target). The doc already leans address-list
      / HTTP / Anvil — just confirm and strike them. Q1 (circuit size) is now
      answered by 0.2; Q2 (server key distribution) is easier than assumed —
      the compressed server key is 28.8 MB, not 114.8 MB.

## Track 1 — Execution core (`primitives/`)

- [x] **1.1 Opcode expansion.** Add `I32Eq`, `I32Ne`, `I32LtS`, `I32GtS`,
      `I32GeS`, `I32LeS`, `Select`, `LocalSet`, `LocalTee`. FHE comparisons
      return `FheBool`, so `DiscaFunction::run` needs a stack value type that is
      either `FheInt32` or `FheBool` rather than today's `Vec<FheInt32>`.
      WASM `select` maps onto tfhe's `IfThenElse::if_then_else` (confirmed in
      the size probe). Compare is 141 ms, select 200 ms — both cheap.
      **Blocks the demo. Highest-value item on the list.**
- [x] **1.2 Compressed ciphertext boundary.** `primitives/src/wire.rs` owns the
      conversion, plus the two protocol hashes: `commitment()` over an encoded
      input (bridge.md's `inputCommits`) and `result_hash()` over an evaluated
      result (the M-of-N attestation value). Decoding is size-bounded against
      untrusted peers. `results_are_deterministic` verifies the assumption the
      whole attestation scheme rests on.
- [ ] **1.2a Decide what `resultHash` covers** — uncompressed (current) or
      compressed, which would let the contract verify the emitted blob on-chain.
      Trade-off written up in bridge.md §5a.
- [x] **1.3 Locals as real storage.** Done alongside 1.1, which needed it:
      `DiscaFunction` carries its declared locals and the evaluator builds a
      frame of parameters followed by trivially encrypted zeros.
- [x] **1.4 Validate stack discipline at compile time.** `primitives/src/validate.rs`
      walks a lowered circuit checking arity, local addressing and final stack
      depth, and returns a `CircuitLayout` carrying peak stack depth and the
      split points Phase 2 partitioning needs. `bytecode::deserialize` runs it
      over every blob it accepts, so a worker rejects a bad circuit from the
      network before evaluating a gate rather than minutes in.
- [x] **1.5 Tests for the new opcodes** against plaintext reference semantics.
      Six execution tests run circuits under real encryption and check
      decrypted results.

## Track 1b — Observability

- [x] **1b.1 Replace `println!` with `tracing`.** The node emits structured,
      nested spans: `INFO` for phase timings (load / keygen / encrypt /
      evaluate), `DEBUG` for a `circuit.run` span per circuit, `TRACE` for
      per-opcode timings and stack depth. `RUST_LOG` controls verbosity. These
      are the same measurements a worker will report to a coordinator in
      Track 2, so the instrumentation is deliberately shaped like job telemetry
      rather than debug printing.
- [ ] **1b.2 Wire telemetry into the coordinator/worker roles** once 2.1 lands —
      a job id field on every span, and worker-reported evaluation durations.
- [ ] **1b.3 Decide on a machine-readable sink** (JSON layer, or OTLP export)
      before the demo, so the video can show real job traces.

## Track 2 — Node roles (`node/`)

- [ ] **2.1 `--role coordinator|worker`.** `node/src/main.rs` is currently a
      hardcoded single-process demo. Split into role dispatch behind clap.
- [ ] **2.2 Worker HTTP server** — `POST /jobs`, `POST /results` per bridge.md §4.
- [ ] **2.3 Coordinator dispatch** — fan a job to N workers, collect
      `keccak256(result ct)`, compare M-of-N.
- [ ] **2.4 Server key distribution** — coordinator serves `GET /keys/<hash>`,
      workers pull once and cache. Serve the **compressed** key: 28.8 MB, not
      114.8 MB (architecture.md §11 Q2 — comfortably fine locally).
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

- [x] **4.1 Write the tally circuit** — done as a validation spike.
      `committee-tally/` holds four ways of writing the tally; all parse, and
      `primitives/tests/tally_circuit.rs` evaluates them under real encryption.
      Findings are in architecture.md §2a: release builds are required (debug
      output is rejected), and fixed-size loops unroll cleanly.
      `primitives/examples/inspect.rs` dumps a module's lowered opcodes.
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
