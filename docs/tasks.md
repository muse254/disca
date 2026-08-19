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
      untrusted peers. `results_are_deterministic` and `compression_is_deterministic`
      verify the assumptions the whole attestation scheme rests on.
- [x] **1.2a Decide what `resultHash` covers** — decided: the **compressed**
      result, so `fulfillJob` can verify `keccak256(resultBlob) == resultHash`
      and a coordinator cannot pair a real attestation with a substituted blob.
      Emitting the result blob on-chain becomes required rather than optional.
      Reasoning, cost, and the fallback (option C) in bridge.md §5a.
- [ ] **1.2b Revisit 1.2a if results stop being single-valued.** 11.8 KB is the
      cost of one `i32`; calldata grows linearly with output count, and ~10
      values would already be too large for an ordinary transaction.
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
- [x] **1b.2 Wire telemetry into the coordinator/worker roles** — folded into
      Track 2 as 2.7, where it is actionable rather than blocked.
- [ ] **1b.3 Decide on a machine-readable sink** (JSON layer, or OTLP export)
      before the demo, so the video can show real job traces.

## Track 2 — Node roles and transport (`node/`)

This is the first work that adds real network surface. The execution core it
sits on is finished: bytecode encodes and decodes with validation on receipt
(1.4), the ciphertext boundary and the hash workers attest to are in place
(1.2), and deterministic evaluation is verified rather than assumed. What
remains is moving those artifacts between processes.

**Not in this track:** the chain watcher (that is 3.4, and needs the contract to
exist first), persistence, and any authentication beyond the worker registry.
Track 2 ends at three local processes agreeing on a result with no chain
involved.

### 2.0 Decisions to settle before writing code

- [x] **2.0a HTTP library.** Evaluation is CPU-bound and blocking — a tally is
      ~1 s, a multiply ~2 s — so async buys very little at three workers.
      Recommend a threaded sync server (`tiny_http`) over `axum` + `tokio`:
      far fewer dependencies, and blocking FHE work inside a handler is honest
      rather than something to design around. Revisit only if the coordinator
      needs many concurrent connections.
- [x] **2.0b Payload encoding.** We already carry two encoders — `wincode` for
      bytecode, tfhe's `safe_serialize` for ciphertexts. Reuse both rather than
      adding a third: JSON for control fields, length-prefixed raw bytes for
      ciphertext and bytecode blobs.
- [x] **2.0c Worker identity.** architecture.md §11 Q3 leans an on-chain
      registered address list over signature aggregation. Confirm it — it
      decides whether a worker needs a keypair at all in Phase 1, and therefore
      whether `POST /results` needs to be authenticated.
- [x] **2.0d Job identifier.** Chain-assigned `jobId` does not exist until
      Track 3. Use a coordinator-local monotonic id now and make it the
      correlation key everywhere (see 2.7), so swapping in the on-chain id later
      touches one place.

### 2.1 Protocol types (`node/src/protocol.rs`)

- [x] Define the messages in one module, independent of the transport that
      carries them: `JobDispatch` (job id, bytecode, input blobs), `JobReport`
      (job id, worker id, result blob + attestation hash, or a failure), and the
      key-fetch response.
- [x] Reuse `primitives::wire::SealedResult` for what a worker reports, so blob
      and hash cannot be handled apart even across the wire.
- [x] Round-trip tests per message.

### 2.2 Role dispatch (`node/src/main.rs`)

- [x] `--role coordinator|worker` behind clap, replacing the hardcoded demo.
- [x] Coordinator flags: bind address, worker addresses, `--attesters M`.
- [x] Worker flags: bind address, coordinator address, worker id.
- [x] Keep the current single-process demo reachable as `--role demo`, since it
      is the fastest way to check the core still works without standing up a
      network.

### 2.3 Worker

- [x] Serve `POST /jobs`: decode bytecode, **validate before evaluating**
      (`bytecode::deserialize` already does this — surface the rejection as a
      failure report rather than a dropped connection).
- [x] Decompress inputs, evaluate, `seal_result`, report to the coordinator.
- [x] Verify each input blob against its commitment on receipt; a worker should
      not evaluate over bytes the coordinator altered in transit.
- [x] Report failures explicitly. A worker that cannot evaluate must say so —
      silence is indistinguishable from being slow, and stalls the job.

### 2.4 Coordinator

- [x] Serve `POST /results` (worker reports).
- [ ] `GET /result/<jobId>` for the key holder. Deferred: the coordinator still
      stands in for the key holder (`KeyHolder` in `coordinator.rs`), so there
      is nobody to serve yet. Needed when they split in Track 3.
- [x] Dispatch a job to N workers concurrently.
- [x] Collect reports until M agree on an attestation hash, or the deadline
      passes.

### 2.5 Server key distribution

- [x] Coordinator serves `GET /keys/<serverKeyHash>`; workers pull once and
      cache by hash.
- [x] Serve the **compressed** key: 28.8 MB, not 114.8 MB (measured — this
      answers architecture.md §11 Q2 and makes the pull comfortable locally).
- [x] Workers verify the hash of what they received before installing it.

### 2.6 M-of-N aggregation

- [x] Group reports by attestation hash; succeed at the first hash reaching M.
- [x] **Disagreement is a finding, not noise.** Determinism is verified
      (1.2), so honest workers cannot disagree — a mismatch means a faulty or
      dishonest worker. Log which worker reported what, and mark the job
      disputed rather than quietly retrying.
- [x] Timeout → job fails, refundable (mirrors `refundOnTimeout` in bridge.md §6).

### 2.7 Job-scoped telemetry (absorbs 1b.2)

- [x] Put `job_id` on every span on both sides, so one job's path through the
      coordinator and three workers is a single filterable trace.
- [x] Coordinator logs the agreed hash, the attester set, and per-worker
      latency — the same fields `fulfillJob` will eventually take on-chain.

### 2.8 Local end-to-end run

- [x] Script one coordinator + three workers as local processes, 2-of-3
      attestation, no chain.
- [x] Include a deliberately faulty worker in the script — a wrong result is the
      only way to demonstrate that M-of-N does anything, and it is the most
      convincing thing in the eventual demo video.
- [x] Assert the key holder decrypts the expected plaintext at the end.

### 2.9 Review follow-ups (PR #9)

- [x] **2.9a Bind attestations to dispatches.** Agreement was counted over
      self-declared worker names with no registry check and no de-duplication,
      so one worker could settle a job alone. Per-(job, worker) tokens now
      attribute each report to the worker it was dispatched to.
- [x] **2.9b Bound the worker job queue** and refuse with 503 when full.
- [x] **2.9c Refuse when two groups both reach quorum** rather than letting
      `HashMap` order pick a winner. Reachable whenever `M <= N/2`.
- [x] **2.9d Check HTTP status and fail loudly on truncation.** A 404 body was
      being handed back to the caller as the server key.
- [x] **2.9e Stop overclaiming the commitment checks.** The commitment travels
      in the same message as the bytes it commits to, so it detects corruption,
      not a malicious coordinator. Real once the commitment comes from the
      chain (Track 3).
- [ ] **2.9f Re-verify input commitments against the chain**, not against the
      dispatch, once `submitJob` exists. This is what makes 2.9e's check
      adversarial rather than merely diagnostic.

### 2.10 Byte-equality attestation — fixed by pinning the FFT plan

The divergence was never randomness: tfhe-rs benchmarks FFT algorithms for 10 ms
at first use and caches the winner, so different processes pick different
numerically-equivalent algorithms and round a few coefficients differently.
`pin_fft_plan` in `node/src/main.rs` fixes it. Demo settle rate went from 2 of 8
to 6 of 6. Full write-up in architecture.md §3.

- [x] **2.10a Pin the FFT plan at node startup**, before anything touches a key.
- [ ] **2.10b Enforce the reproducibility preconditions at registration.** Byte
      equality holds only for workers sharing one CPU architecture (x86 and ARM
      diverge), with the FFT plan pinned, evaluating on CPU (the `gpu` feature
      selects multi-bit parameters that are non-deterministic without
      `with_deterministic_execution()`). Registration should record and check
      these; until it does, disagreement means divergence, not dishonesty, and
      must not feed slashing. `bridge.md` §2 now states this.
- [x] **2.10c Guard reproducibility across processes.**
      `primitives/tests/determinism_under_concurrency.rs` re-executes its own
      binary as concurrent children sharing one key and inputs, and fails if
      they disagree. It has to be cross-process: tfhe caches the FFT plan in a
      process-global `OnceLock`, so an in-process test passes whether or not the
      plan is pinned and would not have caught this. Verified to fail when
      pinning is skipped (`DISCA_SKIP_PIN=1`). It also covers the polynomial
      size, which has no public accessor — if `ConfigBuilder::default()` ever
      moves off 2048 the pin covers nothing and this test fails.
- [ ] **2.10f Keep the divergence path anyway.** Agreement can still fail for
      reasons we have not seen; a disputed job must stay a first-class outcome
      rather than an assertion.
- [ ] **2.10d Raise it upstream.** `setup_custom_fft_plan` is public but
      `#![doc(hidden)]`, absent from the docs and release notes, and panics if
      called late. Draft issue in `docs/tfhe-determinism-request.md`.
- [ ] **2.10e Re-check the parallel carry-propagation path.** `add.rs` selects an
      algorithm from `rayon::current_num_threads()`, which would diverge across
      machines with different core counts. Not reproduced on our circuit shape;
      confirm it cannot bite before relying on cross-machine agreement.

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
