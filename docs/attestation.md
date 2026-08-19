# M-of-N attestation

Why DISCA replicates every job across workers, what agreement is supposed to
prove, and what it costs measured against the alternatives.

Companion to `architecture.md` §3 (trust model) and §7 (verification ladder),
and `bridge.md` §5a (which bytes the hash covers). Read this one first if the
question is *why does this mechanism exist at all*.

## 1. The problem attestation solves

A worker returns a ciphertext. Nobody who receives it can tell whether it is the
right one.

That sentence is the whole justification, and it is worth being precise about
why each party is helpless:

| Party | Why it cannot check the result |
|---|---|
| The consumer contract | It holds a ciphertext and no key. There is nothing to compare against. |
| The chain | Re-executing the circuit on-chain is the thing DISCA exists to avoid — an 18-op FHE circuit is ~3 s of native CPU and gigabytes of key material. |
| The coordinator | Sees only ciphertext. It cannot distinguish a correct result from a plausible forgery. |
| **The key holder** | **Decrypts to a number. A wrong ciphertext decrypts to a wrong number just as cleanly — no error, no signal.** |

The last row is the one that surprises people. FHE gives confidentiality, not
integrity. `decrypt` on a corrupted-but-well-formed ciphertext returns an
integer, not a failure. If a worker evaluates `a + a` where the circuit said
`max(a, b)`, the key holder gets `142` instead of `93` and has no way to know.

This is not hypothetical: the `--faulty` worker flag in `node` does exactly that
(`&result + &result`), and its output is indistinguishable from an honest one
until something compares it to another worker's.

So an unverified FHE coprocessor is a machine that returns confidential answers
you have no reason to believe. **Verification is not a hardening feature here;
without it there is no product.**

## 2. What M-of-N attestation is

The cheapest verification that needs no proof machinery: run the job more than
once, on machines that do not trust each other, and require them to arrive at
the same answer independently.

As implemented:

1. The coordinator dispatches the identical job — same bytecode, same input
   ciphertexts, same server key hash — to N workers.
2. Each worker evaluates, compresses the result, and hashes the compressed
   bytes: `keccak256(compressed_result)`. That hash is its **attestation**.
3. Each worker reports its attestation to the coordinator, echoing the
   per-worker **attestation token** it was dispatched with. The token is what
   makes a report attributable to a worker the coordinator actually dispatched
   to; without it one worker can report M times under M invented names and
   settle a job single-handedly.
4. The coordinator settles on the first hash M distinct workers report. At most
   one report per worker counts.
5. `bridge.md` §5a: because the hash covers the *compressed* result — the same
   blob emitted on-chain — the contract can recompute
   `keccak256(resultBlob) == resultHash` and confirm the ciphertext it publishes
   is the one that was attested to.

The security claim is then: to forge a result you must control M workers, not
one. With M = 2, N = 3, a single dishonest worker is outvoted and visible.

**The load-bearing assumption is byte reproducibility.** Two honest workers must
produce byte-identical compressed ciphertexts, or their hashes differ and
agreement never happens. §5 measures whether that holds. It does not.

## 3. What "without it" would have to mean

Verification has to compare a result against something. The options are limited,
and eliminating them is what leaves replication:

- **Re-execute at the verifier.** Whoever checks must run the same FHE circuit —
  the exact cost DISCA offloads. If the key holder can afford to re-execute, it
  can afford to execute, and the coprocessor has no reason to exist.
- **Prove correct evaluation.** A ZK proof of correct homomorphic evaluation is
  the real answer (`architecture.md` §7, L2). It does not exist in usable form
  for tfhe-rs today.
- **Trust hardware.** A TEE moves trust rather than removing it, and is out of
  scope for the design.
- **Replicate and compare.** No proofs, no re-execution by the verifier, no new
  trusted hardware. The verifier compares 32-byte strings. This is L0.

Replication buys verification at the price of doing the work N times. That is
the trade, and it is why the mechanism is worth measuring rather than assuming.

## 4. Benchmark method

All figures measured on the machine below, at commit `9cfde8f`, release builds.

| | |
|---|---|
| Hardware | Apple M1 Pro, 8 cores, 16 GB |
| OS | macOS 15.6.1 |
| tfhe-rs | 1.5 (`boolean`, `shortint`, `integer`) |
| Circuit | `tally4_select`, 18 ops, peak stack 5 — the demo circuit, not a toy |
| Inputs | `71, 93, 42, 88`; correct answer `93` |

Two harnesses:

- `scripts/run-local.sh` — the real system: one coordinator, three worker
  processes, HTTP transport, 2-of-3.
- `primitives/examples/cross_process.rs` — three concurrent processes reading a
  *fixed* server key and *fixed* input ciphertexts from disk, transport removed,
  so divergence cannot be blamed on the network or on differing inputs.

## 5. Results

### 5a. What each part of the mechanism costs

| Step | Measured | Note |
|---|---|---|
| Evaluate the circuit, one process alone | **1,412 ms** | The unavoidable work |
| Evaluate, 3 workers concurrently on 8 cores | **2,858–3,992 ms** | Contention; this is what the demo actually sees |
| Evaluate, `RAYON_NUM_THREADS=1`, alone | **5,427 ms** | 3.8× slower than default |
| Hash a result blob (attestation) | **~48 µs** | 12,183-byte blob; SHA3-256 as a Keccak-speed proxy |
| Compare hashes at the coordinator | Nanoseconds | `HashMap` bucketing of ≤ N entries |
| Key holder decompress + decrypt one result | **~75 ms** | Coordinator's last-report → `job settled` gap |
| Server key served per worker | 30,146,955 bytes | GET, once per worker per key |
| Job dispatch body | 10,301 bytes | Per worker |
| Job report body | 12,183 bytes | Per worker |

**The verification step is free.** Hashing and comparing cost ~48 µs against a
~3,000 ms job — around 0.002%. The entire cost of M-of-N is the replication:
N× the CPU, and nothing else worth counting.

### 5b. Does byte equality actually hold?

`cross_process`, three concurrent processes, byte-identical key and inputs from
disk, 6 rounds each:

| Configuration | Rounds where all 3 agreed | ms/round |
|---|---|---|
| Default threading | 5 / 6 | 2,996 |
| `RAYON_NUM_THREADS=1` | 4 / 6 | 5,999 |

Single-threading does not fix it and costs 2× wall clock. Worse, with the key
and inputs *fixed on disk across rounds*, different rounds unanimously agreed on
**different** hashes (`0x333cf70…` in most rounds, `0x17764b7…` in another).
There is no canonical encoding that a correct worker converges on — the whole
population drifts. This is not one flaky machine.

### 5c. What that does to real jobs

`scripts/run-local.sh`, 2-of-3, **all three workers honest**, 12 runs:

| Outcome | Runs | |
|---|---|---|
| All 3 agreed | 3 | Settled cleanly |
| 2 agreed, 1 diverged | 6 | Settled — **and logged an honest worker as an attestation disagreement** |
| All 3 differed | 3 | **Job failed**, no quorum |

So with zero dishonest participants: **25% of jobs fail outright, and 50% of
jobs falsely accuse an honest worker of faulty attestation.** The disagreement
warning — the mechanism's entire diagnostic output — is wrong half the time.

The shipped demo is worse, because it deliberately runs one faulty worker, which
leaves exactly two honest workers who must agree. `scripts/run-local.sh` at its
defaults, 8 runs:

| Outcome | Runs |
|---|---|
| `job settled result=93` | **2** |
| `did not reach 2-of-3 agreement` | **6** |

**The demo in the pull request description succeeds 25% of the time.** The
sample log in that description is a real run; it is also the minority case.

### 5d. The alternatives, costed against the same job

| Approach | Verification cost | Replication cost | Works today? |
|---|---|---|---|
| Trust one worker | 0 | 1× (1.4 s) | Returns wrong answers silently. Not verification. |
| Verifier re-executes | 1 extra evaluation (1.4 s) | 1× | Defeats the purpose — the verifier could have run the job |
| **M-of-N on result bytes (current)** | **~48 µs** | **3× (≈3.0 s)** | **No — settles 25% of demo runs** |
| M-of-N on decrypted plaintext | ~75 ms × N ≈ **225 ms** | 3× (≈3.0 s) | Yes, if divergent results decrypt identically — see caveat below |
| Optimistic challenge (L1) | 0 normally; 1 evaluation per challenge | 1× + challenges | Not built; needs a dispute window and stake |
| ZK proof of evaluation (L2) | Not measurable | 1× | Not available for tfhe-rs |

The row that matters: **plaintext adjudication costs ~225 ms on a ~3,000 ms job
— about 7%.** Every one of the nine runs in §5c that reached a quorum decrypted
to the correct `93`, including the six that settled on a 2-of-3 majority over a
diverging third worker.

**Caveat, and the next thing worth measuring.** Those runs only ever decrypt the
*winning* ciphertext. That the *diverging* worker's bytes also decrypt to `93` is
`architecture.md` §3's finding, not something re-measured here — no current
tooling exposes the client key to a losing blob. Plaintext adjudication rests
entirely on it, so it should be pinned by a test that decrypts every worker's
result, before the scheme is chosen on the strength of this table.

## 6. Verdict

Replication is justified. It is the only verification available at L0, and the
comparison step is genuinely free — 0.002% of a job. The mechanism is not fluff;
an unverified FHE coprocessor returns numbers nobody should believe (§1).

**The comparison key is sound once the FFT plan is pinned.** The numbers below
were measured *unpinned* and are the reason this document exists: 25% of
all-honest jobs failed, 50% slandered an honest worker, and the demo settled
twice in eight attempts. The cause turned out to be tfhe-rs benchmarking FFT
algorithms at first use rather than any randomness in evaluation. With
`pin_fft_plan` (architecture.md §3) the demo settles 6 of 6 and disagreement
fires only on the genuinely faulty worker. Byte equality *is* available; it just
is not the default.

Two consequences that are easy to miss:

1. **The disagreement warning was not evidence, and now is.** Unpinned it fired
   on honest workers more often than on the faulty one, which would have made
   any slashing or reputation logic punish honest participants. Pinned, it
   fires only on the faulty worker — but see architecture.md §3 on mixed-ISA
   fleets before treating it as proof of dishonesty.
2. **`bridge.md` §2 is now stale.** It states "Because FHE evaluation is
   deterministic, agreement implies correct evaluation." That premise is false
   as measured, and the sentence should be corrected along with whichever
   replacement scheme is chosen.

Adjudicating on the decrypted plaintext remains the fallback if byte equality
ever proves unreliable again (mixed architectures being the known risk) (options and costs in `architecture.md` §3, recorded as tasks
2.10a–c). It costs ~7% of a job and puts the key holder on the critical path,
weakening `bridge.md` §5a's on-chain verifiability — that is a real trade, not a
free fix, and it is the decision this document exists to inform.

## 7. Reproducing

```sh
# Divergence with the transport removed, fixed key and inputs.
cargo build --release -p primitives --example cross_process
target/release/examples/cross_process setup /tmp/disca-bench
for i in 1 2 3; do target/release/examples/cross_process eval /tmp/disca-bench & done; wait
# Three hashes. They are not reliably equal.

# End-to-end settle rate, all workers honest.
for i in $(seq 1 12); do HONEST=1 ./scripts/run-local.sh 2>&1 | grep -E "settled|did not reach"; done

# End-to-end settle rate, shipped demo defaults (one faulty worker).
for i in $(seq 1 8); do ./scripts/run-local.sh 2>&1 | grep -E "settled|did not reach"; done
```
