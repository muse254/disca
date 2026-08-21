# A TLA+ specification of M-of-N attestation and job settlement

`DiscaAttestation.tla` models one job, N workers, M required attesters, a
coordinator, and the bridge contract, and TLC checks it exhaustively at
N = 3 and N = 4.

Read this before trusting any of it: **half the configurations in this
directory are supposed to fail.** A model that reports "no errors" while
modelling something other than the real protocol is worse than no model,
because it launders an assumption into an apparent proof. Every property
below is one TLC actually checked, every configuration reports the size of
the state space it checked, and every safety check is paired with a
configuration where the corresponding decision is removed and TLC produces
the counterexample it prevents. If a check cannot fail, it is not evidence.

```
make -C spec          # fetch tla2tools.jar if absent, check all 30 configurations
spec/check.sh         # the same thing
spec/check.sh MC_N3M2 # one configuration
```

`check.sh` exits non-zero if a configuration that should pass does not, **and
if a configuration that should produce a named counterexample does not.**
The jar is fetched from the tlaplus releases and pinned by SHA-256; it is
gitignored rather than committed. Java 11 or later. The whole suite takes
about 45 seconds on eight cores.

---

## Why TLA+, and what this is not

The whitepaper claimed "a formal verification proof for the distributed
computer" and there was none. That claim has been removed. This directory is
not a replacement for it and does not restore it: it is a model of one
protocol, checked exhaustively at two small sizes, resting on a stack of
assumptions listed below. What it buys is a precise answer to questions the
Rust unit tests cannot answer, because those questions are about
*interleavings* rather than about functions:

- The coordinator's inbox is filled by concurrent HTTP handlers while
  `collect` polls it. `node/src/coordinator.rs` has tests for `record`,
  `tally` and `group` in isolation. It has no test for "what orders can
  reports arrive in, and is there one that settles the wrong result".
- The bridge contract does not exist yet (`docs/bridge.md` §8 step 1). Its
  behaviour is prose in §2/§2a. A specification of a contract nobody has
  written is exactly what a model is good for, and `EscrowPaidOnce` here is
  a requirement on `DiscaBridge.sol` rather than a verification of it.
- The known limitations — the constant job id, `M <= N/2` — are documented
  in comments as things that "would be" a problem. A model turns "would be"
  into a numbered sequence of steps.

TLC does not prove anything about the Rust. It proves things about this
module. The module was written against the code (`Verifier::attribute`,
`record`, `group`, `tally`, `collect`, `Behaviour::Faulty`, `Claim`,
`recover`, and `fulfillJob`/`refundOnTimeout` as specified), every
divergence is marked `DIVERGENCE` in the source and listed below, and that
correspondence is reviewed by hand. It is the weakest link.

---

## What is assumed rather than proved

Each of these is marked `ASSUMED` at the head of `DiscaAttestation.tla`.

**Unforgeability of secp256k1.** A signature bearing a party's name enters
the model only through an action that party takes. There is no action
anywhere that lets one party mint another's signature. This is the boundary
between what TLC establishes and what cryptography provides, and TLC does
not check it — it is true of the module by construction, and the
construction *is* the assumption.

**Recovery on a mismatched claim yields an unrelated address.** ECDSA
recovery essentially never fails; a signature over job id J presented
against job id K ≠ J returns *some* address, just not the signer's. The
model renders that as "refused", which is the registry check in
`Verifier::attribute` rejecting the unrelated address that comes back. The
chance that the unrelated address is a registered worker's is negligible and
is not modelled.

**Attestations are public.** Not an assumption in the cryptographic sense —
a consequence of the design, and the code says so: "`/results` accepts a
report from any registered address and an attestation is not a secret". Any
party can relay any signature it has seen. This is what makes replay
expressible at all; a model where only the signer can post its own signature
would prove the replay away by fiat.

**Deterministic honest evaluation.** Every honest worker computes the same
result. This is `architecture.md` §3's deterministic-evaluation property and
it is not free: it holds only with the FFT plan pinned (`pin_fft_plan` in
`node/src/main.rs`), on one CPU architecture (x86 and ARM differ), on CPU
rather than GPU, at one pinned tfhe version, at one polynomial size. Before
the plan was pinned, honest workers disagreed in 6 of 12 measured runs. A
worker violating any of those conditions diverges *while behaving honestly*
and is indistinguishable in this model from `Behaviour::Faulty` — which is
correct, because it is indistinguishable to the coordinator too. **Every
"honest" in every property below silently carries these preconditions.**

**keccak256 is collision-resistant.** Distinct results have distinct hashes,
so `Results` is a set of distinguishable values and
`require(keccak256(resultBlob) == resultHash)` pins the blob. The blob is
not a variable in the model.

**The contract behaves as `docs/bridge.md` §2/§2a specify.** `bridge/` does
not exist in this repository. `fulfillJob` and `refundOnTimeout` here are
models of a design document, not of code. If the contract is written
differently, this model says nothing about it.

### Where the model diverges from the implementation

| Divergence | Why it is safe |
|---|---|
| `Compute` evaluates and signs in one step | No party can observe the gap: nothing is sent between them, and a worker that dies in the gap is indistinguishable from one that never started. Halves the state count. |
| `agreement_still_possible` is defined but not used as a guard on settlement | It is an early-exit optimisation — it decides when to stop waiting, never what to settle on. Omitting it leaves the model an over-approximation of the code's timing, which is the safe direction for safety checking. It *is* used to define `PatientDeadline`. |
| The "sealed result matches its own hash" check is not modelled | It rejects a report before any counting happens and cannot admit one, so it can only remove behaviours. |
| `require(keccak256(resultBlob) == resultHash)` is not modelled | The blob is not a variable; hashing is assumed injective, so the blob is determined by the result. This check is what makes the *emitted ciphertext* the attested one (§5a) — a property about bytes, not about quorum. |
| The EIP-2 low-`s` and `v ∈ {27,28}` checks are not modelled | Byte-level well-formedness with no state-machine content. Both are tested directly in `primitives/src/attest.rs`. |
| One signature per worker per job unless `EquivocatingFaults` | Equivocation is a strictly stronger fault than `Behaviour::Faulty`, which reports once. It is available as a flag and exercised in `MC_LWW_Equivocation_N3M2`. |
| Two possible results (`good`, `bad`) | Faulty workers agreeing on one wrong answer is the *worst* case for splits, and the realistic one: `worker.rs` says the likely divergence is misconfiguration (wrong FFT plan, wrong ISA), and misconfigured workers agree with each other. Independent wrong answers are strictly weaker. |

### What is not modelled at all

- Multiple concurrent jobs. One job, one program.
- The key holder and decryption. The model never mentions plaintext.
- The FHE evaluation itself, input commitments, the server key, its hash, or
  the `GET /keys/<hash>` fetch.
- Network partitions distinct from crashes, message reordering below the
  level of "a report arrives", duplicate delivery beyond replay.
- Gas, reentrancy, or anything else EVM-specific.
- Worker registration and deregistration. The registry is fixed.

---

## The properties

Each is stated in `DiscaAttestation.tla`. "In terms of the running system"
is what it means when the processes are `node --role coordinator` and three
`node --role worker`.

### 1. `QuorumIsReal`

> If the contract fulfilled the job with result R, then at least M distinct
> registered workers signed R **for this job**.

The property the whole design exists to provide, and the one `bridge.md` §2b
records the earlier design failing to have. The words that carry weight are
*distinct* (one worker cannot be two), *registered* (a valid signature from
a stranger is worth nothing), and **for this job** — which is the phrase the
constant job id breaks.

"For this job" is not observable by any party in the protocol, so the model
carries a ghost variable, `current`, holding the signatures that were
produced by a worker running *this* job instance. Nothing reads it except
the invariant. A replayed signature is in `sigs` and not in `current`, and
no check anywhere in the system can tell them apart. That is the point.

Checked with `AdversarialFulfill = TRUE`, meaning *anyone at all* may call
`fulfillJob` with any set of signatures they have seen. So this is a
property of the contract's `ecrecover` loop, not of the coordinator being
well-behaved — which is exactly the reversal task 2.10i made.

**Result: holds** at N=3 M=2, N=3 M=3, N=4 M=2, N=4 M=3, with and without
strangers present, and with an earlier run's attestations in circulation
*provided job ids are unique*. **Fails** under a constant job id
(`MC_Replay_N3M2`) and with the registry check removed (`MC_Sybil_N3M2`).

### 2. `EscrowPaidOnce`

> Fulfilment and refund are mutually exclusive and each happens at most once.

`fulfillJob` releasing escrow to the coordinator and `refundOnTimeout`
returning it to the poster must not both happen, and neither twice. In the
model, two counters and `fulfilCount + refundCount <= 1`.

This is close to true by construction, which is a fair thing to be
suspicious of, so `MC_UnguardedRefund_N3M2` removes the one guard that makes
it true — `refundOnTimeout` requiring `state == Open` — and TLC immediately
finds the trace where a job is fulfilled, the deadline then passes, and the
escrow is paid a second time.

**Result: holds.** But note what it is: `DiscaBridge.sol` does not exist, so
this is a *specification of what the contract must do*, not a verification
of what it does.

### 3. `NoSettleOnSplit`

> When two result groups both reach M — possible whenever M ≤ N/2 — the
> coordinator refuses rather than picking one.

`tally` returns `None` when `quorums.len() > 1`. The job then does not settle
at all and the escrow refund path closes it.

**Result: holds.** `MC_Split_witness` confirms non-vacuously that split
states are reachable at N=4, M=2 (a 9-state trace).
`MC_NoSplitRefusal_N4M2` removes the refusal and TLC settles a split.

**Is refusing necessary or merely cautious?** Necessary, but the reason is
not the one you would guess, and the model is the thing that makes the
difference visible.

`MC_SplitQuorumStillReal_N4M2` checks `QuorumIsReal` against a coordinator
that settles splits, and it **holds**. Both groups genuinely signed. The
signers are real, registered and distinct, the contract's `ecrecover` loop
passes, and it would pass on-chain. So settling either group does *not*
violate `QuorumIsReal`.

What dies is the inference from agreement to correctness, which is the only
reason anyone wanted agreement. `MC_SplitCost_N4M2` states that narrowly —
"when you settled a split, was the group you picked the right one?" — and TLC
answers no with a 10-state trace: w1 and w2 (honest) attest `good`, w3 and w4
(faulty) attest `bad`, both groups are at M = 2, the coordinator picks `bad`,
and the job settles. `coordinator.rs` calls the alternative what it is, "a
coin flip decided by hash iteration order"; this is the flip losing.

Nothing downstream catches it. The contract cannot: the quorum is real. The
key holder cannot: `attestation.md` §1, "decrypts to a number. A wrong
ciphertext decrypts to a wrong number just as cleanly." So refusing is the
only answer that does not hand somebody a wrong plaintext they have no reason
to doubt.

### 4. `VoteNotDisplaced` — first-write-wins

> No report ever overwrites a vote already cast.

`record` keeps the first attestation per recovered address. See the trace
below.

**Result: holds** under first-write-wins, **fails** under last-write-wins.

### 5. `SomeoneActuallyEvaluated` — replay under a constant job id

> If the contract fulfilled the job, at least M workers evaluated something
> for it.

The second face of `QuorumIsReal`, and the sharper one. See the trace below.

**Result: fails** under a constant job id, **holds** once the id is unique.

### 6. `Liveness`

> The job does not hang: the escrow is always eventually settled, by
> fulfilment or by refund.

Weak fairness on every protocol action — each one, once enabled, stays
enabled until taken, so WF is the right strength and SF would assume more
than the implementation provides. Crashed and failing workers are left
unconstrained, because a job that hangs when a worker dies is exactly the
failure this excludes.

**Result: holds** at N=3 M=2 with fewer than M faulty, and at N=4 M=3 with
*two* faulty (beyond the tolerance N−M=1, so most of those runs cannot settle
at all — they must still not hang).

`EventuallyFulfils` is the stronger claim, and it needs three hypotheses:
a deadline generous enough not to cut a viable job short, at least M honest
workers, and fewer than M faulty ones. **Result: holds** at N=3 M=2 and
N=4 M=3. `MC_Hangs_witness` drops the first hypothesis and TLC produces the
temporal counterexample, which both shows the liveness checking has teeth and
shows which hypothesis does the work: a short deadline does not break the
system, it turns every job into a refund — `bridge.md` §6 working as designed
and a coprocessor that computes nothing.

---

## Counterexample 4, in prose: last-write-wins loses a unanimous job

Configuration `MC_LWW_Settles_N3M2`. Three workers, **all honest**. No faulty
worker, no crash, an honest coordinator that refuses splits, and nobody but
the coordinator calling `fulfillJob`. The only two things wrong are that the
job id is the constant 1 (so an earlier run's attestations verify against
this job) and that the inbox keeps the *last* report per address instead of
the first.

`LateReplayOnly = TRUE` forbids a relayed signature from arriving until every
live worker has already reported, so this is unambiguously the attack the
code comment is about — a **late** report displacing a vote already cast —
and not a relay that simply got there first.

TLC's trace, ten states:

1. w1 evaluates the circuit and signs `good`.
2. w2 evaluates and signs `good`.
3. w3 evaluates and signs `good`. All three honest workers agree, as
   deterministic evaluation says they must.
4. w1's report arrives. Inbox: `w1 → good`.
5. w3's report arrives. Inbox: `w1 → good, w3 → good`.
6. w2's report arrives. Inbox: **all three `good`**. A unanimous job.
7. A relayer posts w2's attestation **from an earlier run**, over `bad`. It
   is a real secp256k1 signature by w2 over (job 1, this bytecode hash,
   `bad`), so `Verifier::attribute` recovers w2's address and the registry
   accepts it. Last-write-wins overwrites the vote w2 cast ten milliseconds
   ago. Inbox: `w1 → good, w2 → bad, w3 → good`.
8. The relayer posts w3's earlier-run attestation over `bad`. Inbox:
   `w1 → good, w2 → bad, w3 → bad`.
9. `tally` groups the inbox: `bad` has two members, `good` has one. Exactly
   one group at M = 2 — **not** a split, so the refusal in property 3 never
   fires and there is no warning to log.
10. The job settles on `bad` and the contract fulfils it.

Every check in the system passed. Two distinct registered workers signed
`bad` for job 1, so `QuorumIsReal` holds and `fulfillJob`'s `ecrecover` loop
is satisfied. The key holder decrypts a number and has no reason to doubt it.

`MC_FWW_Late_N3M2` is the same configuration with one Boolean flipped back to
what `coordinator.rs` does. Steps 7 and 8 are refused —
`Entry::Occupied` returns `AlreadyVoted`, logged as "attester reported more
than once; keeping the first" — the inbox stays unanimous, and every
invariant holds. 118 distinct states, no error.

**Two bounds on that claim, both checked.**

`MC_LWW_UniqueId_N3M2` runs last-write-wins with *unique* job ids and
non-equivocating workers, and `VoteNotDisplaced` **holds**. The inbox is
keyed by the recovered address, so the only thing that can overwrite a
party's slot is a second signature *by that same party* over a different
result, and there are exactly two sources for one: an earlier run under a job
id that repeats, or a signer that equivocates. Remove both and the two
policies are indistinguishable.

`MC_LWW_Equivocation_N3M2` removes only the first — unique job ids, no
replay, one worker that signs two different results for one job — and
last-write-wins is displaced again. So first-write-wins is not made redundant
by fixing the job id: it is the defence that still stands when a signer
equivocates, and it is the one that is load-bearing *today*, because job ids
are the constant 1.

---

## Counterexample 5, in prose: a quorum nobody computed

Configuration `MC_ReplayNobodyRan_N3M2`. The coordinator hardcodes
`let job_id = 1;` — there is no chain to take an id from — and the signed
digest binds that id, so an attestation from an earlier run of the same
program verifies against the job running now. The code says so in a comment
above the constant. This is what it costs.

TLC's trace, five states:

1. w2 and w3 are crashed. w1 is honest and has not started evaluating — the
   coordinator has just fanned the job out. **No worker signs anything at any
   point in this trace**; `evaluated` is `unstarted` for all three in the
   final state.
2. A relayer posts w1's attestation from an earlier run: a real signature by
   w1 over ("DISCA/attest/result/v1", job **1**, this bytecode hash, `good`).
   `report.job_id` matches. The claim the coordinator reconstructs from *its
   own* job id and *its own* bytecode hash is byte-identical to the one w1
   signed last time. Recovery returns w1's address. The registry contains it.
   Accepted. Inbox: `w1 → good`.
3. The same for w2. Inbox: `w1 → good, w2 → good`.
4. The same for w3. Inbox: all three.
5. `tally` finds one group of three at M = 2. The job settles. `fulfillJob`
   recovers three distinct registered signers over the submitted result hash
   and fulfils. Escrow is released.

`SomeoneActuallyEvaluated` is violated: zero workers evaluated anything.
`QuorumIsReal` is violated in the same configuration by a **two-state** trace
— the adversary needs only to relay M signatures and call `fulfillJob`
directly; it does not need the coordinator's participation at all.

Note what the settled result was in that trace: `good`, the right answer.
The invariant that fails is not "the result is wrong" — it is "anybody ran
the job". A replay of the correct answer settles a job and releases escrow
for work nobody did, and the same three steps with `bad` in place of `good`
settle a wrong one. Both are available; TLC happened to find the first.

Nothing here is a bug in any check. `Verifier::attribute` did its job.
`ecrecover` did its job. The registry did its job. The signatures are
genuine. A constant job id is not a weak check — it is the *absence* of the
only value that would have distinguished this run from the last one.

**The second half of it, which the code comment does not mention.** The
signed preimage (`Claim::preimage`, all 94 bytes) binds the domain tag, the
job id, the bytecode hash and the result hash — and **not the input
commitments**. So the earlier run does not even have to be the same job. Any
earlier run of the same program, over *any* inputs, produced attestations
that verify against this one. The job id is the only field that was ever
going to separate them.

**The fix, checked.** `MC_UniqueJobId_N3M2` is `MC_Replay_N3M2` with one
constant changed: the earlier run used a different job id, which is
`submitJob` assigning a globally unique one (`bridge.md` §2, task 2.9f). The
same attestations are in circulation, the same adversary may relay them and
call `fulfillJob`. Every invariant holds — `QuorumIsReal`,
`SomeoneActuallyEvaluated`, `ResultIsCorrect`, `VoteNotDisplaced`,
`NoSettleOnSplit`, `EscrowPaidOnce` — across all 2,842 reachable states. The
stated fix is sufficient.

Binding a chain id and a verifying contract address (EIP-712, `attest.rs`
`Claim::digest`) is a *separate* replay boundary — across deployments and
forks — and is not modelled here, because a second deployment is not modelled
here.

---

## Two findings that were not being looked for

TLC produced both of these as unexpected failures of configurations written
to be controls. They are the most useful things in the directory, and both
are about behaviour that is in the tree today.

### First-write-wins stops displacement, not pre-emption

`MC_ReplayPreempt_N3M2`. Three honest workers, first-write-wins exactly as
`coordinator.rs` has it, an honest coordinator, no faults, no adversarial
`fulfillJob`. The only defect is the constant job id.

A relayer posts three earlier-run attestations — w1's over `good`, w2's and
w3's over `bad` — **before any worker has finished evaluating**. All three
slots are now filled with the relayer's choices. `tally` finds one group at
M = 2 on `bad` and the job settles. The honest workers' real attestations
arrive afterwards to find their slots taken, and are refused by the very rule
that was protecting them.

`displaced` is `FALSE` in every state of that trace: `VoteNotDisplaced`
holds. Nothing was displaced, because nothing had been cast.

First-write-wins defends a vote already cast; it cannot defend a vote not yet
cast. Under a repeated job id, an attacker who is faster than an FHE
evaluation — a bar measured in seconds, `architecture.md` §2: 225 ms per
add, 2.04 s per multiply — votes on the workers' behalf before they can. The
two are not alternatives: first-write-wins is a partial mitigation and only
unique job ids close the hole. (`MC_UniqueJobId_N3M2` passes.)

### The straggler grace can settle a faulty group before the honest one reports

`MC_GraceRace_N4M2`. `RefuseOnSplit = TRUE` — the shipping code. Honest
coordinator, no replay, no adversarial `fulfillJob`. N = 4, M = 2, two faulty
workers, which `M <= N/2` permits.

1. w3 (faulty) evaluates and reports `bad`.
2. w4 (faulty) evaluates and reports `bad`.
3. `STRAGGLER_GRACE` expires. w1 and w2, the honest workers, have not
   reported: FHE evaluation is seconds of work and the grace is five.
4. `tally` sees exactly one group at M and returns it. The job settles on
   `bad`.

`settleSplit` is `FALSE` in every state — the refusal branch is never
reached, because from the coordinator's point of view there is nothing to
refuse. It never sees the honest group.

This is `collect` behaving as written: it settles when a quorum exists **and**
(everyone has reported **or** the grace has expired), and the second disjunct
is the race. It is not a soundness failure of attestation — `QuorumIsReal`
holds; two registered workers really did sign `bad` for this job — it is the
same statement as the split: with `M <= N/2` the fault threshold has been
exceeded and agreement stops implying correctness.

Worth being clear about the fix, because the obvious one does not work. A
longer grace period does not help: a faulty worker can always be faster than
an honest one, because it need not evaluate. What closes it is choosing
`M > N/2`, which makes a second group arithmetically impossible — and at
N = 4, M = 3 (`MC_N4M3`) and N = 3, M = 2 (`MC_N3M2`) every invariant
including `ResultIsCorrect` holds.

---

## State-space sizes

Exhaustive breadth-first search, no state constraints, no symmetry
reduction. "Distinct states" is the full reachable state space and
"diameter" is the depth of the complete search — both are what TLC prints
for a completed run, and they are stable across runs.

| Configuration | N | M | Distinct states | Diameter | Result |
|---|---|---|---|---|---|
| `MC_N3M2` | 3 | 2 | 2,842 | 10 | all invariants hold |
| `MC_N3M3` | 3 | 3 | 1,054 | 10 | all invariants hold |
| `MC_N4M2_tolerated` | 4 | 2 | 28,916 | 12 | all invariants hold |
| `MC_N4M2` (2 faulty) | 4 | 2 | 42,224 | 13 | quorum/escrow/split hold |
| `MC_N4M3` | 4 | 3 | 20,540 | 12 | all invariants hold |
| `MC_Registry_N3M2` (2 strangers) | 3 | 2 | 2,842 | 10 | all invariants hold |
| `MC_UniqueJobId_N3M2` | 3 | 2 | 2,842 | 10 | all invariants hold |
| `MC_FWW_N3M2` | 3 | 2 | 1,336 | 10 | `VoteNotDisplaced` holds |
| `MC_FWW_Late_N3M2` | 3 | 2 | 118 | 10 | all invariants hold |
| `MC_LWW_UniqueId_N3M2` | 3 | 2 | 2,506 | 10 | `VoteNotDisplaced` holds |
| `MC_SplitQuorumStillReal_N4M2` | 4 | 2 | 31,670 | 12 | `QuorumIsReal` holds |
| `MC_Liveness_N3M2` | 3 | 2 | 2,506 | 10 | `Liveness` holds |
| `MC_Liveness_N4M3` | 4 | 3 | 27,692 | 12 | `Liveness` holds |
| `MC_Fulfils_N3M2` | 3 | 2 | 314 | 10 | `EventuallyFulfils` holds |
| `MC_Fulfils_N4M3` | 4 | 3 | 966 | 12 | `EventuallyFulfils` holds |

The counterexample configurations halt at the first violation, so they have
no state-space size to report — only how deep the trace was. Both the number
of states explored before halting and, occasionally, the trace length by a
state or two, vary between runs: TLC's parallel breadth-first search does not
fix the order in which a level is expanded, so which of several equally short
counterexamples it reports first is not deterministic. The lengths below are
what one run produced; `check.sh` asserts the named invariant is violated,
not that the trace has a particular length.

| Configuration | Violates | Trace length |
|---|---|---|
| `MC_Fulfil_witness` | `NeverFulfils` (witness) | 7 |
| `MC_Refund_witness` | `NeverRefunds` (witness) | 3 |
| `MC_Split_witness` | `NoSplitEver` (witness) | 9 |
| `MC_Hangs_witness` | `EventuallyFulfils` (witness) | temporal |
| `MC_NoSplitRefusal_N4M2` | `NoSettleOnSplit` | 10 |
| `MC_SplitCost_N4M2` | `SplitSettlementWasCorrect` | 10 |
| `MC_GraceRace_N4M2` | `ResultIsCorrect` | 7 |
| `MC_LWW_N3M2` | `VoteNotDisplaced` | 8 |
| `MC_LWW_Settles_N3M2` | `ResultIsCorrect` | 10 |
| `MC_ReplayPreempt_N3M2` | `ResultIsCorrect` | 5 |
| `MC_LWW_Equivocation_N3M2` | `VoteNotDisplaced` | 5 |
| `MC_Replay_N3M2` | `QuorumIsReal` | 2 |
| `MC_ReplayNobodyRan_N3M2` | `SomeoneActuallyEvaluated` | 5 |
| `MC_Sybil_N3M2` | `QuorumIsReal` | 5 |
| `MC_UnguardedRefund_N3M2` | `EscrowPaidOnce` | 4 |

### What the sizes do and do not license

**These are proofs at N = 3 and N = 4, and nothing else.** Nothing here is
an induction and no property is established for general N or general M. The
sizes are given so a reader can see the checks were not vacuous — 42,224
states with a diameter of 13 is a search that went somewhere — not to imply
that a larger deployment inherits them.

Two properties are checked at one size only and should be read that way:

- The **straggler-grace race** (`MC_GraceRace_N4M2`) is exhibited at N = 4,
  M = 2. The argument that it generalises to any `M <= N/2` is prose, not a
  model check.
- The **split** (`MC_Split_witness`, `MC_SplitCost_N4M2`,
  `MC_NoSplitRefusal_N4M2`) is exercised at N = 4, M = 2 only, because it is
  the smallest configuration in which `M <= N/2` holds with M ≥ 2. At N = 3,
  M = 2 a split is arithmetically impossible, so `NoSettleOnSplit` holds
  there **vacuously** and the passing result for `MC_N3M2` should not be read
  as evidence about split handling.

`SplitSettlementWasCorrect` is likewise vacuous in every configuration with
`RefuseOnSplit = TRUE`, since there are no split settlements to be correct
about. It has content only in `MC_SplitCost_N4M2`, where it fails.

---

## What could not be verified

- **That the Rust matches this model.** There is no extraction, no
  refinement proof and no trace validation between TLC and `node`. The
  correspondence is a careful reading, recorded in the comments of
  `DiscaAttestation.tla`, and it is the weakest link in the chain.
- **The contract.** `bridge/DiscaBridge.sol` does not exist. `fulfillJob`
  and `refundOnTimeout` here are models of `docs/bridge.md` §2/§2a, so
  `EscrowPaidOnce` and half of `QuorumIsReal` are requirements on code
  nobody has written rather than checks on code somebody has.
- **Anything about FHE.** Determinism of evaluation and of compression is
  assumed, not modelled; it is measured in
  `primitives/src/wire.rs` (`results_are_deterministic`,
  `compression_is_deterministic`) and in
  `primitives/tests/determinism_under_concurrency.rs`, and it is the
  assumption most likely to be violated in a real deployment, because a
  single ARM machine in an x86 fleet breaks it silently.
- **General N and M.** See above.
- **Concurrent jobs.** One job. Whether the constant job id causes
  cross-*talk* between two jobs running at the same time — as opposed to
  between one job and an earlier one — is not modelled, and is plausibly
  worse.
