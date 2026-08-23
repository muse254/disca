# DISCA Task Checklist

Status: working checklist derived from [architecture.md](architecture.md) and
[bridge.md](bridge.md) against the code as merged in #2.

Track 0 is groundwork — measurement, encoding, tooling — that everything else
depends on. Tracks 1–4 are the build, roughly in dependency order.

Dependency spine — nothing in the bridge track can start until **0.2** lands,
and the demo in architecture.md §10 can't be built until **1.1** lands:

```
0.1 measure ──► sizes the circuits everything else assumes
0.2 bytecode serialization ──► 3.x bridge (needs bytecodeHash)
1.1 opcode expansion ──────► 4.x demo (needs compare/select)
1.2 compressed ct boundary ─► 2.x transport, 3.x calldata
```

---

## Track 0 — Groundwork

Cheap, and unblocks estimating everything downstream.

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
- [ ] **0.5 Decide the open questions** in architecture.md §11. Q1 (circuit
      size) answered by 0.2. Q2 (server key distribution) easier than assumed —
      the compressed key is 28.8 MB, not 114.8 MB. **Q3 (attester scheme) is
      settled, and settled *against* the doc's leaning**: signatures, not an
      address list, because an address list the coordinator supplies is not
      something a contract can check (2.10i, bridge.md §2b). The measured cost
      is 7.0-7.7k gas per attester on a 354k transaction — the leaning rested on
      signatures being expensive, and they are not. Q4 (HTTP) and Q5 (Anvil)
      still want confirming and striking.

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
- [ ] `GET /result/<jobId>` for the key holder. **No longer blocked** — the
      reason it was deferred was that the coordinator *was* the key holder, and
      task 4.3 split them: `disca-cli` holds the client key and never talks to a
      worker. The coordinator writes the winning blob to `--result` today, which
      is enough for a local run and not enough for a key holder on another
      machine.
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

### 2.11 Housekeeping

- [x] **2.11a Keep `logic_gates` out of the binary.** The gate-composition route
      to FHE arithmetic has no callers — the evaluator uses tfhe's integer API.
      Now behind the `boolean-circuits` feature, off by default, documented as
      the deliberate alternative rather than deleted (the research paper describes
      that approach and this is its only implementation). Its truth-table tests
      were ~7.3 s of the primitives suite, about a third, guarding unused code.
- [x] **2.11b Build with `--all-features` in CI.** `boolean-circuits` is not
      compiled by an ordinary build, so nothing type-checks it and it will rot.
- [x] **2.11c Implement `disca-cli parse` (see 4.3).** Done as `compile`, which
      lowers wasm and validates every function before writing bytecode. The stub
      it replaced exited 2 with a
      pointer to the `inspect` example instead of accepting a file, doing
      nothing, and reporting success.

### 2.10 Byte-equality attestation — fixed by pinning the FFT plan

Investigation and measurements: [PR #9](https://github.com/muse254/disca/pull/9).

The divergence was never randomness: tfhe-rs benchmarks FFT algorithms for 10 ms
at first use and caches the winner, so different processes pick different
numerically-equivalent algorithms and round a few coefficients differently.
`pin_fft_plan` in `node/src/main.rs` fixes it. Demo settle rate went from 2 of 8
to 6 of 6. Full write-up in architecture.md §3.

- [x] **2.10a Pin the FFT plan at node startup**, before anything touches a key.
- [x] **2.10i Sign attestations per worker. Blocks all of Track 3.** Attestations
      are unsigned: a `SealedResult` is a blob and `keccak256(blob)`, computable
      by anyone. The coordinator supplies the attester list to `fulfillJob`, so
      the contract can check only that the addresses are registered and distinct
      — a dishonest coordinator can name any two registered workers beside any
      result, and nothing contradicts it. The attestation tokens added in 2.9a
      fix the equivalent hole *inside* the coordinator and are invisible to a
      contract. Give workers secp256k1 keys and have the contract `ecrecover`
      each attestation. Cost is ~3.5k gas per attester on a 250–350k
      transaction, which reverses architecture.md §11 Q3's reasoning. The token
      then becomes redundant and `/results` gets authentication free.

      **Sign a digest, not the bare result hash.** `resultHash` is
      `keccak256(blob)` and commits to no job, program or worker, so a signature
      over it alone is replayable onto any other job with the same output — the
      hole moves rather than closes. The raw `resultHash` stays as the grouping
      key for M-of-N.

      **Decided: v1 binds `(domain, jobId, bytecodeHash, resultHash)` under
      EIP-191; `chainId` and the verifying contract arrive as `/v2` under
      EIP-712 when the bridge is deployed.** Those two values do not exist
      until then, and signing placeholders for them produces signatures that
      have to be reinterpreted later — silent breakage of exactly the kind the
      version tag exists to make loud. EIP-191 already buys what matters now:
      `0x19` is not a legal leading byte for an RLP transaction, so an
      attestation cannot be replayed as one, and the key stays usable behind
      any `personal_sign` interface. Implemented in #13. Recoverable (r, s, v)
      signatures, so a contract can `ecrecover` without being handed a pubkey.
      The preimage layout is a schema decision Solidity has to reconstruct byte
      for byte, and `SealedResult` is already threaded through
      `primitives::wire`, `node::protocol` and both roles — see
      [next-architecture.md](next-architecture.md) §2.4 for why this is the type
      most expensive to change late.

      Do the ten lines in §2.5 at the same time: `JobDispatch` has no
      `bytecode_hash`, so a worker never checks *which* program it ran and
      attests to less than the contract will assume. Add `program_id` and
      `bytecode_hash` to the dispatch, verify before evaluating, and put
      `programId` in the digest above.

      **Do not implement `fulfillJob` before deciding this.**
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
- [ ] **2.10g Model the faults a deployment would actually hit.** `--faulty`
      injects one mode: a well-formed wrong answer, i.e. the adversarial case.
      The divergence we have actually observed was *misconfiguration* — honest
      workers disagreeing 6 of 12 runs before the FFT plan was pinned. Add fault
      modes for the realistic causes (mismatched architecture, tfhe version
      skew, unpinned plan, GPU build) plus crash, hang and garbage output, so
      the local run exercises what registration will have to reject.
- [x] **2.10h Keep fault injection and the demo role out of release builds.**
      `--faulty` is behind the `fault-injection` feature, off by default; the
      `demo` role is `#[cfg(debug_assertions)]`. A default release build has
      neither. `scripts/run-local.sh` opts in explicitly.
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

## Track 2c — Formal specification (`spec/`)

Not on the original list. It exists because the paper claimed "a formal
verification proof for the distributed computer" and there was none; rather than
only delete the claim, the thing it would have to rest on got built.

- [x] **2c.1 TLA+ model of M-of-N attestation and settlement.**
      `spec/DiscaAttestation.tla`, 30 TLC configurations, ~45 s. Checks
      `QuorumIsReal`, `EscrowPaidOnce`, `NoSettleOnSplit`, `VoteNotDisplaced`,
      `SomeoneActuallyEvaluated` and liveness. Half the configurations are
      counterexamples that are *supposed* to violate a named invariant, and
      `check.sh` fails if one stops finding its counterexample — a spec that
      only checks "no errors" cannot tell a proof from a model that stopped
      modelling anything.
- [x] **2c.2 Two findings, both fixed.** A minority quorum could settle during
      the straggler grace before honest workers reported (`MC_GraceRace_N4M2`);
      `2M > N` is now required at startup. And a constant `job_id` made
      attestations interchangeable between runs, so replays could *pre-empt*
      every honest worker rather than displace them — first-write-wins does not
      help against arriving first (`MC_ReplayPreempt_N3M2`); job ids are now
      per-run.
- [x] **2c.3 Pin the model to the code it describes.** `spec/models.toml` hashes
      each modelled function; `make -C spec drift` fails when one changes and
      names what the spec uses it for. Runs in CI before TLC, because the
      failure it catches is the one TLC structurally cannot: a model that is
      internally consistent and externally wrong.
- [ ] **2c.4 Bind the input commitments into the signed digest.** `Claim::preimage`
      binds job id, bytecode hash and result hash but not the inputs. With
      per-run job ids that is currently sound, since the id identifies the input
      set — but the soundness is incidental rather than stated, and it breaks
      quietly the moment an id becomes predictable or shared.
- [ ] **2c.5 The model is single-job, N ≤ 4, and unverified against the Rust.**
      There is no extraction or refinement: the correspondence is a careful
      reading, and 2c.3 is a tripwire rather than a proof of it. Concurrent
      jobs, multiple chain ids, registration and gas are not modelled.

## Track 2d — The coordinator as a job service

From `next-architecture.md` §2.2, which showed task 2.0d's "swapping in the
on-chain job id touches one place" was not true: the id was never the problem,
the absence of per-job state was.

- [x] **2d.1 Per-job state.** `Coordinator { jobs: HashMap<u64, Arc<Job>> }`,
      each job owning its verifier, inbox, dispatch set, deadline and outcome.
      `accept_job(JobSpec) -> jobId` is the entry point and the CLI is now one
      caller of it. `POST /results` routes by job id; an unknown job is a named
      404 rather than a silent drop.
- [x] **2d.2 `JobSpec::job_id: Option<u64>`.** `None` mints one, `Some(id)` is
      the id the chain assigned — which is what makes on-chain settlement
      possible at all, since a worker signs a digest binding it.
- [x] **2d.3 `--attestations <path>`.** The winning group's signatures in the
      shape `fulfillJob` takes, sorted ascending because the contract requires
      strictly increasing addresses. Each exported signature is asserted to
      recover to the address printed beside it.
- [ ] **2d.4 The model is still single-job.** Routing by id and unknown-job
      rejection are behaviour no TLC configuration checks. `spec/models.toml`
      records this; a multi-job model needs new variables and new invariants and
      is work in its own right.

## Track 3 — Bridge (`bridge/`, new Foundry project) — needs 0.3

- [x] **3.1 Foundry scaffold** + `DiscaBridge.sol`: program registry, worker
      registry, job escrow, lifecycle, events (bridge.md §2).
- [x] **3.2 Forge unit tests** for the job state machine and the M-of-N
      attester check.
- [x] **3.3 Anvil end-to-end** — `scripts/run-anvil.sh`, and not with a mocked
      coordinator: the workers' own signatures settle it. Deploy, register,
      `submitJob` with real commitments, evaluate under FHE, `fulfillJob`, then
      decrypt the ciphertext taken **out of the on-chain event** to 93, plus the
      refund path and a pass where the liar is decisive so no quorum forms.

      Two things it found that reading the contracts had not. **A
      `CommitteeTally` holding escrow could never be refunded** — it posts its
      own job, so `job.poster` is a contract with no `receive`; the bridge's
      refund `call` fails, the bridge reverts (correctly), and by then the
      deadline has passed so `fulfillJob` refuses too. Both exits closed against
      a §6 that promises a refund. Fixed with `receive` and a committee-gated
      `withdraw`; the unit suite missed it because every refund path there was
      zero-value or through an EOA. And **the coordinator was signing under an
      id it invented** while `fulfillJob` rebuilt the digest from the id
      `submitJob` assigned, so correct settlements were rejected as
      `NotRegisteredWorker` — `--job-id` closes it.
- [ ] **3.4 Alloy chain watcher** in the coordinator role — subscribe to
      `JobRequested`, validate blob commitments, dispatch, submit `fulfillJob`.
      Adds `alloy` as the first non-tfhe dependency.
- [x] **3.5 `refundOnTimeout`** — done, with the deadline a per-deployment
      immutable and fulfilment after it refused, so settlement is not a race
      between coordinator and poster over one escrow. **The disputed-job path is
      not pending, it is unreachable**: divergence is invisible on-chain, so a
      disagreeing worker simply fails to contribute to a quorum and the job
      expires into a refund. `JobState.Disputed` exists and nothing sets it.
      bridge.md §6 row 2 now says so rather than describing a path that cannot
      be taken.

## Track 4 — Demo + submission — needs 1.1

- [x] **4.1 Write the tally circuit** — done as a validation spike.
      `committee-tally/` holds four ways of writing the tally; all parse, and
      `primitives/tests/tally_circuit.rs` evaluates them under real encryption.
      Findings are in architecture.md §2a: release builds are required (debug
      output is rejected), and fixed-size loops unroll cleanly.
      `primitives/examples/inspect.rs` dumps a module's lowered opcodes.
- [x] **4.2 `CommitteeTally.sol`** consumer contract (bridge.md §5).
- [x] **4.3 `disca-cli` for real** — `parse` is a no-op (`main.rs:29`) and
      `parser.rs:3` is `todo!()`. Needs at minimum: wasm → bytecode + hash, and
      `keygen`.
- [ ] **4.4 End-to-end demo script** driving register → submit → evaluate →
      fulfill → decrypt.
- [ ] **4.5 Demo video** + README rewrite (it currently documents only the
      research-paper build, not the project).

## Stretch (only if the above is done)

- [ ] Circuit partitioning at stack-depth-zero points (architecture.md §6 Phase 2).
- [ ] L2 testnet deploy with verified contracts.
- [ ] Optimistic challenge window (architecture.md §7, L1).

## Explicitly out of scope

Public-key / multi-key / threshold FHE; ZK evaluation proofs; slashing
economics; algorithm privacy; QUIC/libp2p; persistent storage.
