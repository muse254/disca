# What to build next, and what the next stage will break

Status: assessment. Written against the tree at `f5e936d` (plus the unmerged
`ci/coverage` work), reading `architecture.md`, `bridge.md`, `attestation.md`,
`tasks.md`, and `primitives/` + `node/` as they stand.

Claims below are marked **[verified]** where they come from reading the code (file
and line given) and **[inferred]** where they are a judgement about what would
happen next. Nothing here is a restatement of `tasks.md`.

---

## 1. The recommendation in one paragraph

**Track 3 is the right destination and the wrong first step.** The bridge is what
turns DISCA from a demo into a coprocessor, so building it next is correct. But
`bridge.md` §2 specifies `fulfillJob(jobId, resultHash, resultBlob, address[]
attesters)` — an attester list the coordinator supplies and the contract cannot
check. Implementing that interface makes on-chain M-of-N decorative: a
coordinator can name any two registered addresses beside any blob it likes, and
nothing on-chain, off-chain, or in the key holder's plaintext contradicts it. So
the highest-value next thing is not 3.1. It is **the decision 3.1 would
foreclose** — revisiting `architecture.md` §11 Q3 in favour of per-worker signing
keys — followed by turning the coordinator into a job service that does not hold
the client key. Both are small now (a few hundred lines in `node/`) and are
rewrites of the contract, the watcher, the registry and the demo script if
deferred. Then build the contract, and build it once.

---

## 2. What will not survive contact with the next stage

### 2.1 The attester list makes on-chain attestation unverifiable

This is the finding that reorders everything else.

Today a worker's attestation is `keccak256(compressed result)` and nothing more
(`primitives/src/wire.rs:126-130`, `seal_result`) **[verified]**. It is not
signed, not bound to a job, and not bound to a worker. The only thing tying a
report to a worker is the attestation token, which the coordinator minted and
therefore already knows (`node/src/coordinator.rs:131-136`, `296-310`)
**[verified]**.

That is defensible while the coordinator is the only party settling jobs, because
the coordinator is also the audience. It stops being defensible the moment the
contract becomes the audience. `fulfillJob` receives `address[] attesters` from
the coordinator and can check only that they are distinct and registered
(`bridge.md` §2) **[verified in the doc]**. It has no evidence any of them
computed anything. The `require(keccak256(resultBlob) == resultHash)` in §5a ties
the blob to the hash, not the hash to any worker.

The consequence is precise and worth stating plainly, because three documents
currently claim the opposite:

- `node/src/coordinator.rs:5-8`: the coordinator "cannot forge a result, because
  the attestation hash it submits has to be one M registered workers
  independently reported." Once settlement is on-chain with an address list, that
  is false — nothing forces the hash to have come from anywhere.
- `bridge.md` §2: "dishonest workers only waste their own time since they cannot
  forge agreement." True of workers. The coordinator is the party that can, and
  it is absent from the §6 failure table, which lists only coordinator *silence*.
- `README.md:4-5`: "agreement between independent nodes is what makes a result
  trustworthy." That is the product claim, and the specified bridge does not
  deliver it.

And the failure is silent. A forging coordinator produces a settled job, a
correct-looking event, and a ciphertext the key holder decrypts to *some* number.
`attestation.md` §1 is explicit that the key holder cannot tell a wrong answer
from a right one. There is no test that fails, which is exactly why this must be
fixed before the contract exists rather than after — a working demo video is not
evidence the property is present. **[inferred, but the mechanism is fully
determined by the interface in `bridge.md` §2]**

`architecture.md` §11 Q3 framed this as "address list (simpler) vs ECDSA
signature aggregation (cheaper calldata)". Two things are wrong with that
framing. First, signatures are *more* calldata than addresses, not less —
presumably the note meant BLS aggregation, which is a different and much larger
project. Second, and more importantly, it is not a cost trade at all: the address
list does not do the job. The real trade is:

| | Contract can attribute an attestation | Extra gas vs address list | Extra transactions |
|---|---|---|---|
| Address list (`bridge.md` §2) | **no** | — | — |
| M ECDSA signatures, coordinator submits | yes | ~3.5k/attester (1040 calldata + 3000 `ecrecover`) | 0 |
| Each attester transacts itself | yes | 21k base each | M |

Roughly 7k gas for M=2, against a `fulfillJob` the doc already sizes at 250-350k
because of the 11.8 KB blob — 2-3%. Measure it rather than trust my arithmetic
(§4, step 0), but the order of magnitude is not in doubt.

Note that the third row is what the prior art the docs already cite actually
does. `architecture.md` §3 says "Zama's fhEVM uses the same construction —
`sns-worker` hashes the serialized ciphertext and a contract majority-votes the
digest." It works there because each coprocessor sends its own transaction; the
contract counts `msg.sender`. DISCA's bridge has the coordinator vote on the
workers' behalf, which is where the construction is lost.

**Recommendation: per-worker secp256k1 keys, coordinator aggregates, contract
`ecrecover`s.** Concretely, keep `resultHash = keccak256(blob)` as the value
agreement is counted over — the coordinator's grouping logic
(`coordinator.rs:410-427`) is unchanged — and have each worker additionally sign

```
digest = keccak256(DOMAIN_SEP, chainId, bridgeAddress, jobId, programId, resultHash)
```

`fulfillJob(jobId, resultHash, resultBlob, bytes[] signatures)` recovers each
signer, requires them distinct and registered, and requires
`keccak256(resultBlob) == resultHash`. One transaction, coordinator still the
only chain-facing party, and the forgery is gone.

Two things fall out for free, which is how you know it is the right seam:

- The attestation token (task 2.9a, `protocol.rs:33-49`) becomes redundant as an
  authentication mechanism. A signature proves authorship; a token the
  coordinator minted cannot. Keep the token as a dispatch nonce or delete it, but
  stop describing it as identity.
- `POST /results` gains authentication it does not have (§2.7), because the
  report carries a signature over content the coordinator cannot produce.

### 2.2 The coordinator is a one-shot job runner, not a service

`coordinator::run` generates keys, compiles one program, hardcodes `let job_id =
1` (`coordinator.rs:149`), dispatches, collects, prints, and returns
**[verified]**. There is no job loop. The state a job needs is process-global:

- `Inbox` is one `HashMap<worker, JobReport>` for the process, not per job
  (`coordinator.rs:104, 138`) **[verified]**.
- `Tokens` is minted once before the job, not per job (`coordinator.rs:131-136`)
  **[verified]** — so "a fresh unguessable token per (job, worker)"
  (`protocol.rs:41`) is currently per *process*, per worker.
- `JobReport.job_id` is never read by the coordinator. `serve` resolves the token
  and inserts; the field appears only in log lines and tests **[verified: grep
  across `node/src`]**.

Task 2.0d says the coordinator-local job id was chosen so "swapping in the
on-chain id later touches one place." That promise does not hold. The id is not
the problem — the absence of per-job state is. A chain watcher receives
`JobRequested` events concurrently and must keep N jobs in flight; every
structure above has to become keyed by job id first, and only then does the id's
provenance matter. **[inferred, but directly implied by the structures above]**

Related, and cheap to get right now: because `job_id` is unchecked, a worker can
answer job 7 with job 3's report and the coordinator will count it. Harmless with
one job in flight; a replay bug the day there are two.

**Recommendation: make the coordinator a service before the watcher exists.** A
`Jobs: HashMap<u64, JobState>` with the inbox, tokens, deadline and dispatch set
inside it, an `accept_job(JobDispatch) -> jobId` entry point, and a `collect`
that runs per job. Then the watcher is a thin caller and the demo path is one
more caller. Doing this after the watcher means debugging concurrency and chain
plumbing at the same time.

### 2.3 The coordinator holds the client key *and* the plaintext

`KeyHolder` (`coordinator.rs:50-96`) is a well-marked seam, and the marking is
honest. But the split is larger than moving a struct, because of what
`KeyHolder::new()` does: it *generates a fresh keypair on every coordinator
start* (`coordinator.rs:55-77`) **[verified]**. Three consequences:

1. `serverKeyHash` changes every run, so the worker's cache-by-hash
   (`worker.rs:151, 158`) never hits across runs and every worker re-pulls 28.8
   MB **[verified]**.
2. `registerProgram(bytecodeHash, serverKeyHash, M)` pins the server key hash
   on-chain at registration time (`bridge.md` §3). A coordinator that generates
   its own key cannot participate in that lifecycle at all. It has to be *handed*
   a server key and a program, not produce them.
3. The coordinator's CLI takes plaintext inputs: `--inputs 71,93,42,88`
   (`main.rs:91-94`, `run-local.sh`) **[verified]**. For a system whose pitch is
   that no node ever sees plaintext, the secret values are currently an argv
   string on the party that fans the job out. This is the single most visible
   thing wrong with the demo, and a judge reading `ps` would find it.

**Recommendation: `disca-cli` is the key-holder role, and therefore belongs on
the critical path, not in Track 4.** Task 4.3 ("`disca-cli` for real — wasm →
bytecode + hash, and `keygen`") is filed as demo polish; it is actually the other
half of the split. Add `encrypt` and `decrypt` to it and the key holder becomes a
process that never talks to a worker: it produces the bytecode blob and hash for
`registerProgram`, the server key blob for the coordinator, the compressed input
blobs and their commitments for `submitJob`, and consumes the result blob from
`GET /result/<jobId>` (task 2.4, currently deferred *because there is nobody to
serve*). The coordinator then loses `KeyHolder` entirely rather than relocating
it.

### 2.4 The attested hash binds to nothing but the bytes

`SealedResult.hash = keccak256(blob)` and nothing else **[verified,
`wire.rs:126-137`]**. It does not commit to the job, the program, or the inputs.

Today that is fine — one job per process. On-chain it means an attestation is a
statement about a ciphertext, not about a computation. The same result over the
same inputs and circuit produces the same attestation forever, so an attestation
is replayable across jobs by anyone who has seen it. With signatures (§2.1) this
is solved as a side effect, provided the *signed digest* includes `jobId` and
`programId` and the raw `resultHash` stays as the grouping key. If §2.1 is not
adopted, this has to be solved separately and worse.

Getting the digest layout right early matters because `SealedResult` is already
threaded through `primitives::wire`, `node::protocol`, both roles, and the
planned Solidity. It is the type most expensive to change later.

### 2.5 Commitments are diagnostic, and the program hash is not checked at all

Task 2.9e/2.9f already state the input-commitment problem correctly: the
commitment travels in the same message as the bytes (`protocol.rs:24-30`,
`worker.rs:258-272`), so it detects corruption, not a lying sender
**[verified]**. Nothing to add there except the ordering: 2.9f is not a separate
task, it is a consequence of the watcher existing, and the check code is already
in the right place to become adversarial.

The gap the checklist does *not* record is the symmetric one for the program.
`JobDispatch` carries `server_key_hash` and the worker verifies the key it pulls
against it (`worker.rs:212-236`) **[verified]** — but there is no
`bytecode_hash` field, and the worker never hashes what it was told to run. The
coordinator computes `hash_bytecode` once, for a log line
(`coordinator.rs:126`), and it goes nowhere **[verified: grep for
`hash_bytecode` across the workspace]**.

So a worker attests to having run *some* circuit. With `programId →
bytecodeHash` on-chain, a worker that cannot state which registered program it
executed is attesting to less than the contract will assume. Add `program_id` and
`bytecode_hash` to `JobDispatch`, have the worker check
`keccak256(bytecode) == bytecode_hash` before evaluating, and include
`programId` in the signed digest. This is ten lines now.

Related and worth catching before Solidity is written: `bridge.md` §2's
`submitJob(programId, inputCommits, inputBlobs, callback)` has **no function
selector**, but DISCA bytecode is a *program* — `serialize` encodes every
function (`bytecode.rs:46`) and the dispatch names one by string
(`protocol.rs:53`) **[verified]**. Either `registerProgram` pins a
(bytecodeHash, functionName) pair, or a program registers one entry point. The
second is simpler and I would take it; either way it is a schema decision that is
annoying to change after a registry has entries.

### 2.6 Reproducibility is an architectural constraint, not a registration checkbox

Task 2.10b is filed as an enforcement gap. It is bigger than that, and the honest
version belongs in `architecture.md` §1 rather than in a checklist.

Byte equality holds within one CPU architecture, one tfhe version, one parameter
set, CPU not GPU, FFT plan pinned (`architecture.md` §3; `main.rs:234-250`)
**[verified in docs and code]**. Two honest workers in different classes
disagree. The coordinator's disagreement warning cannot distinguish that from
dishonesty (`worker.rs:36-50` says so directly).

The implication for the question "what does this mean for a permissionless worker
set" is: **there is no permissionless worker set at L0.** Not "not yet" —
replicate-and-vote over exact bytes is definitionally a closed, homogeneous
fleet. A permissionless set needs either L1 (challenge on re-execution, where a
divergent honest worker is a false accusation you must adjudicate) or adjudicating
on decrypted plaintext (`attestation.md` §5d, ~7% cost, puts the key holder on
the critical path and gives up `bridge.md` §5a's on-chain verifiability). That is
a positioning statement the whitepaper should carry, because a reader will
otherwise assume the oracle-network shape implies an open one.

The concrete design consequence, and the reason to act now rather than at 2.10b's
leisure: the *contract* should carry the class, not just the node.
`registerWorker(address, bytes32 envDigest)` and `registerProgram(...,  bytes32
envDigest)`, with `fulfillJob` requiring every attester to share the program's
digest. `envDigest` covers `(arch, tfhe version, param set id, cpu|gpu, fft plan
id)`.

A worker can lie about its class, and that is fine — a liar simply disagrees and
is outvoted, exactly as today. The field is not there to be trusted; it is there
so a mixed attester set is rejected structurally rather than surfacing as a
mysterious quorum failure, and so the contract has a defensible reason to refuse
an attestation set that could never have agreed. Adding it later means migrating
a registry.

### 2.7 The transport is unauthenticated in ways the next stage makes worse

Small, but concrete, and all in `node/src/transport.rs` and the two role servers
**[verified]**:

- `POST /jobs` on a worker takes a dispatch from anyone who can reach the port
  (`worker.rs:99-126`). The bound queue (2.9b) caps memory at 16 jobs but not
  CPU: 16 × ~3 s of free homomorphic evaluation per fill, indefinitely.
- `GET /keys/<hash>` is unauthenticated and serves 28.8 MB
  (`coordinator.rs:244-248`). It is a bandwidth amplifier with a 30 MB response
  to a ~40 byte request.
- `MAX_RESPONSE_BYTES` is 256 MB (`transport.rs:27`), so a hostile
  "coordinator" can make a worker buffer a quarter gigabyte before the hash check
  rejects it. The hash check is correct; the allocation happens first.

None of this is urgent while everything is on loopback. It becomes urgent the
moment a worker is a separate machine, which is the same moment the bridge makes
the system worth pointing at. Signed reports (§2.1) fix `/results`; the other two
want a shared-secret or mTLS bearer between coordinator and its registered
workers. Do not build a full auth story — a static token per worker in the
registry file is proportionate and takes an afternoon.

---

## 3. What is over-built or premature

- **`logic_gates` behind a feature flag, kept alive by an `--all-features` CI job
  (2.11a, 2.11b).** 190 lines with no callers, an alternative implementation
  strategy the evaluator does not use, and ~7.3 s of truth-table tests. The
  README's justification — the whitepaper describes that approach and this is its
  only implementation — is real but is served just as well by git history and
  PR #1. **Recommendation: delete it and cite the commit from the whitepaper.**
  If it stays, do not let it gate CI; a build job for dead code is a maintenance
  tax paid to keep a number flattering. This is the maintainer's call and I would
  not spend long on it either way, but 2.11b as written is work in the wrong
  direction.
- **`CircuitLayout::split_points`** (`validate.rs:26-27, 85-87`) is computed on
  every `deserialize` for Phase 2 partitioning that `architecture.md` §9 puts
  explicitly out of scope. It costs nothing at runtime and it is *nice* that
  validation produces it. Leave it, but do not extend it; partitioning is not the
  constraint (§2's measurements removed circuit size as a binding limit).
- **Task 2.10g — more fault modes.** Premature in its current form and for a
  specific reason: the faults it wants to model (mismatched architecture, version
  skew, unpinned plan, GPU build) are all things *registration* should reject,
  and registration does not exist. Written now they become worker behaviours
  nobody checks against anything. Written after §2.6 they become registration
  test cases, which is what they actually are. **Defer, then re-file under the
  registry.**
- **Task 1b.3 — machine-readable telemetry sink.** The stated reason is "so the
  video can show real job traces." The `tracing` output already reads well
  (PR #9's log excerpt is the demo's most convincing artefact). Defer.
- **Task 1.2b — multi-value results.** `DiscaFunction::run` returns a single
  `FheInt32` (`program.rs:144, 260-269`) and `validate` enforces
  `depth == results.len()` **[verified]**. Multi-output would break the
  signature, `SealedResult`, and `bridge.md` §5a's option B simultaneously.
  **Recommendation: do not do 1.2b. Instead write single-output down as a design
  constraint** in `architecture.md` §2a alongside "fixed arity, fixed-size data,
  no allocation, built in release." It is the same class of constraint and it is
  currently only implied.

---

## 4. Sequencing

Ordered by what each step unblocks and what it forecloses. Steps 0-3 are roughly
a week; they are what makes step 4 buildable once.

**Step 0 — gas spike (throwaway, half a day).** A Foundry project containing
nothing but a `fulfillJob` stub, measuring three variants against a real 11.8 KB
blob: address list, M `ecrecover`s, and each-attester-transacts. Delete it
afterwards. *Why a spike:* this is the `size_probe` move — `bridge.md` §1's
250-350k is a sketch built from per-byte constants, and §11 Q3 is being decided
on a cost claim that is currently arithmetic rather than measurement. The tally
spike (PR #5) was worth its day because it tested an assumption the demo depended
on; this tests the assumption §2.1's recommendation depends on. If signatures
turn out to cost 30% rather than 3%, the honest answer changes to
each-attester-transacts, and you want to know that before writing the contract,
not after.

**Step 1 — signed attestations (`primitives` + `node`).** Per-worker secp256k1
keys, signed digest over `(domain, chainId, bridge, jobId, programId,
resultHash)`, coordinator collects signatures alongside hashes, `run-local.sh`
verifies them locally with no chain present. *Unblocks:* an honest contract
interface, authenticated `/results`, meaningful worker registration.
*Forecloses if skipped:* everything in §2.1, plus a Solidity rewrite and a
registry migration.

**Step 2 — split the key holder out into `disca-cli` (task 4.3, promoted).**
`keygen`, `compile` (wasm → bytecode + hash), `encrypt`, `decrypt`. Coordinator
loses `KeyHolder` and takes a server key blob and a bytecode blob as inputs.
`run-local.sh` becomes: cli generates keys and encrypts → coordinator runs the
job → cli decrypts. *Unblocks:* task 2.4's `GET /result/<jobId>` (which is
blocked today only because there is no second party), a coordinator that can
participate in the on-chain key lifecycle at all, and a demo where plaintext is
not in the coordinator's argv. *Forecloses if skipped:* the coordinator cannot be
given a registered server key, so `registerProgram` cannot be implemented
honestly.

**Step 3 — coordinator becomes a job service.** Per-job state, `accept_job`,
concurrent jobs, `job_id` validated on reports. *Unblocks:* the watcher, and the
"one place" swap 2.0d promised. *Forecloses if skipped:* the watcher lands on top
of process-global state and the first two concurrent jobs corrupt each other's
inbox.

**Step 4 — the bridge, for real (3.1, 3.2, 3.5, then 3.3).** `DiscaBridge.sol`
with the registry carrying `envDigest` (§2.6), a program entry point (§2.5),
signature-checked `fulfillJob`, escrow, `refundOnTimeout`, disputed path. Forge
tests for the state machine and — the test that matters — *a coordinator that
submits a well-formed result with a forged attester set must revert*. That test
is the whole point of steps 0-1 and should be written first.

**Step 5 — the watcher (3.4), with one spike inside it.** Before wiring alloy
into the coordinator, spike the event round-trip alone: emit a `JobRequested`
carrying `bytes[] inputBlobs` with three real 2.3 KB blobs on Anvil, decode it in
alloy, reconstruct the commitments, compare. *Why a spike:* dynamic `bytes[]` in
event data is the one part of `bridge.md` §1's blob-availability design nobody
has exercised, and if it is awkward the fallback (blobs in calldata, commitments
in the event) changes the contract, which you would rather learn before step 4's
tests exist. Small enough to be an `examples/` binary.

**Step 6 — consumer contract and demo (4.2, 4.4, 4.5).**

Two things deliberately *not* in this order: `2.10g` fault modes (see §3, they
belong after the registry) and worker-to-worker anything (Phase 2 partitioning is
still out of scope and the measurements say it is not the constraint).

---

## 5. The calls that are genuinely the maintainer's

Presented with a recommendation rather than neutrally, per the ground rules.

1. **Signatures vs each-attester-transacts.** I recommend signatures: one
   transaction, coordinator remains the only chain-facing party, and workers need
   no gas or chain connectivity — which matters because a worker is otherwise a
   pure compute node. Each-attester-transacts is more faithful to the fhEVM prior
   art and removes the coordinator from the trust path entirely, at M× the base
   transaction cost and a much more complex liveness story. Take the measurement
   from step 0 before committing. What is *not* a live option is the address
   list.
2. **Delete `logic_gates` or keep it behind CI.** I recommend deleting. Low
   stakes either way; the reason to decide is that 2.11b is otherwise on the
   list as work.
3. **Whether the whitepaper should say outright that L0 cannot be permissionless
   (§2.6).** I recommend yes, and early — it is a sharper and more defensible
   claim than leaving the reader to assume otherwise, and it motivates L1 on the
   ladder rather than making it look like polish.

---

## 6. Summary of what was verified in code

For the reader who wants to check rather than take:

| Claim | Where |
|---|---|
| Attestation is `keccak256(blob)`, unsigned, unbound to job or worker | `primitives/src/wire.rs:126-137` |
| Coordinator runs exactly one job, `job_id` hardcoded to 1 | `node/src/coordinator.rs:149` |
| Inbox and tokens are process-global, not per job | `node/src/coordinator.rs:104, 131-138` |
| `JobReport.job_id` is never validated | grep across `node/src`; `serve` uses only the token |
| Coordinator generates a fresh keypair per start | `node/src/coordinator.rs:55-77` |
| Plaintext inputs are a coordinator CLI argument | `node/src/main.rs:91-94`, `scripts/run-local.sh` |
| No `bytecode_hash` in `JobDispatch`; worker never checks the program | `node/src/protocol.rs:33-57`, `node/src/coordinator.rs:126` |
| Worker *does* verify the server key against its hash | `node/src/worker.rs:212-236` |
| `run` returns a single `FheInt32` | `primitives/src/program.rs:144, 260-269` |
| `POST /jobs` and `GET /keys/<hash>` are unauthenticated | `node/src/worker.rs:99-126`, `node/src/coordinator.rs:244-248` |
| No `bridge/` directory exists | `ls bridge` |
