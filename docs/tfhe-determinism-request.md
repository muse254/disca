# Byte-reproducible evaluation in tfhe-rs: root cause, prior art, and what to ask for

Context: found while building M-of-N attestation in
[PR #9](https://github.com/muse254/disca/pull/9), which carries the end-to-end
measurements and the DISCA-side fix.

Status: research note. Written to answer one question — is "reproducible/deterministic
homomorphic evaluation" a sensible feature request to make to Zama, and if so, what
exactly should the request say?

**Headline: the root cause is not randomness. It is the FFT plan, which tfhe-rs picks
by running a 10 ms micro-benchmark at process start. A supported fix already exists in
1.5.0 and restores byte equality completely, at no measurable cost. `architecture.md`
§3 and `attestation.md` §6 are wrong about the mechanism and too pessimistic about the
conclusion: L0 is salvageable.**

Everything below is marked **verified** (read in the vendored source or measured on this
machine), **reported** (retrieved from Zama's repos, docs, forum, or published papers),
or **inferred** / **unknown**.

---

## 1. What we observed

Reproduced clean-room, in pure tfhe-rs with no DISCA code involved, so a maintainer can
run it without knowing anything about this project.

**Environment.** tfhe 1.5.0, features `boolean,shortint,integer`; `ConfigBuilder::default()`;
`FheInt32`; release build; Apple M1 Pro (8 cores), macOS 15.6.1, aarch64.

**Program.** Two modes.

- `setup`: `generate_keys(ConfigBuilder::default().build())`, then `safe_serialize` a
  `CompressedServerKey` and four `CompressedFheInt32` inputs (71, 93, 42, 88) to disk.
- `eval`: read those bytes back, `safe_deserialize`, `decompress()`, `set_server_key`,
  then evaluate max-of-four as three `FheOrd::gt` + `IfThenElse::if_then_else` pairs, and
  print a digest of `safe_serialize` of the result.

Every process therefore starts from **byte-identical key material and byte-identical
input ciphertexts read from the same files**.

**Result — default configuration.** Six rounds, three concurrent processes per round:

| Round | Distinct result digests among the 3 processes |
|---|---|
| 1 | 2 |
| 2 | **3** |
| 3 | 2 |
| 4 | 1 |
| 5 | 2 |
| 6 | 1 |

Two rounds of six were unanimous. Across the 18 evaluations, four distinct result
digests appeared (`8cefdb92…` in 11 of 18, then `24cf277b…`, `032509a1…`, `29542e41…`).
**Every one decrypted to 93.** The digests of the decompressed input ciphertexts were
identical in all 18 runs, so decompression is not the source.

That is the same phenomenon `attestation.md` §5b records against the DISCA harness, at
about the same rate, with our evaluator removed from the picture. It is a tfhe-rs
property, not ours.

**Result — with the FFT plan pinned.** Same binary, same six-round protocol, with one
call added before anything else runs:

```rust
use tfhe::core_crypto::fft_impl::fft64::math::fft::{
    setup_custom_fft_plan, FftAlgo, Method, Plan, PolynomialSize,
};

let n = PolynomialSize(2048);              // default params
let fourier = n.to_fourier_polynomial_size();
setup_custom_fft_plan(Plan::new(
    fourier.0,
    Method::UserProvided { base_algo: FftAlgo::Dif4, base_n: fourier.0 },
));
```

**18 of 18 evaluations produced the identical result digest `f8f89dbf…`.** Repeating with
`RAYON_NUM_THREADS` ∈ {1, 2, 4, 8, 16, 32, 64} gave that same digest every time. Wall
clock was unchanged (~2.4–2.6 s per evaluation either way).

So on one machine, byte equality is available today and is free.

---

## 2. Root cause

**Verified in `tfhe-1.5.0/src/core_crypto/fft_impl/fft64/math/fft/mod.rs`.**

`Fft::new` lazily builds a plan per polynomial size and caches it in a process-global
map (`PLANS`, line 104–108). The plan is built like this (line 180–187):

```rust
#[cfg(not(feature = "experimental-force_fft_algo_dif4"))]
{
    Arc::new((
        Twisties::new(n / 2),
        Plan::new(n / 2, Method::Measure(Duration::from_millis(10))),
    ))
}
```

`Method::Measure` asks `tfhe-fft` to **time several candidate FFT algorithms for 10 ms
and keep whichever ran fastest on this process, at this moment, under this machine
load.** The algorithms are numerically equivalent but not bit-identical: they associate
the floating-point butterflies differently, so the `f64` accumulations round differently.
After the inverse transform rounds back to `u64` torus coefficients, a handful of
coefficients land on the other side of a rounding boundary. The ciphertext differs; the
plaintext does not, because the difference is far below the noise budget.

This accounts for every feature of the observation:

- **Deterministic within a process, divergent across processes.** The plan is measured
  once and cached in a `OnceLock`, so a single process is self-consistent forever. That
  is why `primitives/src/wire.rs::results_are_deterministic` — two evaluations in one
  process — has always passed while cross-process agreement fails.
- **Concurrent processes drifting together per round.** `attestation.md` §5b flags as
  suspicious that whole rounds agreed unanimously on *different* hashes. That is exactly
  what a load-sensitive benchmark predicts: three processes started together benchmark
  under the same contention and tend to pick the same winner; a round started under
  different load picks a different winner in all three.
- **Single-threading does not help.** The benchmark still runs, and still races.
- **Divergence grows with circuit length.** Each PBS is another opportunity for at least
  one coefficient to cross a rounding boundary. A 6-op circuit often survives; an 18-op
  one usually does not.

### 2a. What it is *not*

Each of these was checked and ruled out. This matters, because §3 of `architecture.md`
guesses "randomness drawn during evaluation (noise management)", and that guess is wrong.

- **No RNG in the evaluation path (verified).** Grepping `src/shortint/server_key/`,
  `src/shortint/atomic_pattern/`, and `src/integer/server_key/` for `RandomGenerator`,
  `rand::`, `thread_rng`, or `encryption_generator` yields hits only inside `mod tests`
  and inside **key generation**: `ModulusSwitchNoiseReductionKey::new` and its compressed
  twin (`src/shortint/server_key/modulus_switch_noise_reduction.rs:199,209`). The
  thread-local `ShortintEngine` (`src/shortint/engine/mod.rs:31`) holds the encryption
  and secret generators, and evaluation never touches them.
- **The modulus-switch drift technique is deterministic (verified).**
  `choose_candidate_to_improve_modulus_switch_noise_for_binary_key`
  (`src/core_crypto/algorithms/modulus_switch_noise_reduction.rs`) is a plain linear scan
  over encryptions of zero that were generated at keygen, keeping the first candidate
  under `ms_bound`. No randomness. Moreover the default parameter set does not even use
  it: `IntegerConfig::default()` selects `PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128`
  (`src/high_level_api/keys/inner.rs:116–120`), which resolves through
  `src/shortint/parameters/aliases.rs:74–80` to
  `V1_4_PARAM_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128` — `polynomial_size: 2048`,
  `modulus_switch_noise_reduction_params: ModulusSwitchType::CenteredMeanNoiseReduction`
  (`src/shortint/parameters/v1_4/classic/tuniform/p_fail_2_minus_128/ks_pbs.rs:31–46`), which
  draws nothing. Note for later: with the `gpu` feature that same `default()` selects a
  **multi-bit** parameter set instead, and multi-bit PBS is the one path Zama documents as
  non-deterministic unless you append `with_deterministic_execution()`. A CPU-tested
  determinism assumption does not survive turning the GPU feature on.
- **Re-randomization is opt-in and off (verified).** `src/high_level_api/re_randomization.rs`
  requires `ConfigBuilder::enable_ciphertext_re_randomization(...)`; without it
  `ServerKey::re_randomization_support()` is `NoSupport`. Its own doc comment says the
  process "is seeded using the `ReRandomizationContext` and thus **can be made
  deterministic**".
- **No circuit privacy / sanitization in tfhe-rs (reported, source-checked).** Nothing
  implements Ducas–Stehlé or Bourse–Del Pino–Minelli–Wee. Evaluation randomness is not
  serving circuit privacy here, because there is neither.
- **Parallel reduction order is not the cause here.** The blind rotate is sequential;
  the parallel paths in `core_crypto` chunk element-wise work whose per-element result
  does not depend on the chunking, and the integer layer's parallel sums are exact
  modulo 2^64, hence order-independent.

### 2b. A second, separate hazard we did *not* trigger

**Verified in source, not observed in our test.**
`src/integer/server_key/radix_parallel/add.rs:425` and `:518` select between two
different carry-propagation algorithms by calling `rayon::current_num_threads()`:

```rust
if should_parallel_propagation_be_faster(
    self.message_modulus().0 * self.carry_modulus().0,
    lhs.len(),
    rayon::current_num_threads(),
) { CarryPropagationAlgorithm::Parallel } else { CarryPropagationAlgorithm::Sequential }
```

The two algorithms produce ciphertexts that decrypt alike but need not be byte-equal. So
in principle, **two machines with different core counts can diverge even with the FFT
plan pinned.** Our max-of-four circuit gave the same bytes for thread counts 1 through
64, so the branch either did not flip or the outcomes coincided for this shape. Whether
it flips for wider integers or longer circuits is **unknown** — worth a test before
relying on a heterogeneous fleet.

### 2c. And a third, which no amount of pinning fixes

**Reported.** Zama's own NIST Threshold Call writeup ("TFHE, ZHEnith and Nexus", v0.1,
2026-01-09, §3.4.1) states plainly:

> The algorithms in TFHE require floating point FFT operations, which inherently have
> numerical errors. These numerical errors can be different on different machines due to
> micro-architectural differences in the underlying hardware. […] a ciphertext produced
> by a homomorphic evaluation on one machine may be different from that on another
> machine, even though they both decrypt to the correct evaluation. There is little we
> can do to mitigate such problems without producing a crypto system which is terribly
> inefficient. It would appear from experimenting with various x86-based platforms, that
> across x86-based machines the results are indeed the same. But when moving (say) to
> ARM-based platforms the outputs are indeed different.

So byte equality is achievable **within** an ISA family and **not** across x86 ↔ ARM.
Any design that hashes result ciphertexts is implicitly assuming a homogeneous fleet.

---

## 3. Does a supported path already exist?

**Partly — yes for the root cause, no as a documented guarantee.**

- `setup_custom_fft_plan` is `pub` in 1.5.0
  (`src/core_crypto/fft_impl/fft64/math/fft/mod.rs:110`) and is demonstrated in the
  shipped `examples/manual_fft.rs`. We measured that it works (§1).
- `DeterministicSeeder` does **not** reach evaluation. **Verified**: every use is in
  keygen, seeded encryption, the seeded-key expansion paths, the WASM API, or
  `integer/oprf.rs` (the deliberately-random OPRF, which takes its seed from the caller).
  The high-level entry point is `ClientKey::generate_with_seed(config, Seed)`
  (`src/high_level_api/keys/client.rs:69`). `architecture.md` §3 option 2 is right that
  `DeterministicSeeder` is wired to keygen and low-level encryption — but it is looking
  in the wrong place, because there is no evaluation randomness for it to seed.

Three caveats keep this from being a *supported* path in any strong sense:

1. **It lives behind `#![doc(hidden)]`.** `src/core_crypto/fft_impl/fft64/mod.rs:1` hides
   the whole module, so `setup_custom_fft_plan` appears in no rendered documentation and
   carries no stability promise.
2. **It panics if you are late.** The setter is
   `Entry::Occupied(mut e) => e.get_mut().set(plan).unwrap()` (line 120–121). If any FFT
   has already run for that polynomial size — decompressing a server key is enough — the
   `OnceLock` is initialised and `set` returns `Err`, so `unwrap` panics.
3. **It is per polynomial size, and you must know your sizes.** Compute params, compression
   params, and noise-squashing params can differ; each needs its own plan, and the caller
   has to work out which sizes a given config will touch.

**Cargo feature alternative.** `experimental-force_fft_algo_dif4` pins the same choice at
compile time (the `#[cfg]` shown in §2). It is named "experimental" and is not documented
as the determinism switch, but Zama's own draft reproducibility test depends on it (§4).

---

## 4. Prior art and existing reports

**All reported, with URLs, retrieved rather than recalled.**

### In tfhe-rs

- **Issue #1765**, "Have a method to select deterministic FFT algorithms for all supported
  PolynomialSize" (opened 2024-11-08 by `furkanturan` of Belfort, closed 2026-01-16) —
  https://github.com/zama-ai/tfhe-rs/issues/1765. This is our bug, reported eighteen
  months ago, from the hardware-accelerator side rather than the consensus side. The
  reporter saw **differing bootstrapping keys** across runs with a fixed
  `DeterministicSeeder` — "roughly 1 in 10,000 executions" on AMD EPYC, "about 1 in 10 of
  executions" on Intel Xeon — and correctly named the 10 ms benchmark. Zama's reply:
  "this is a known behavior for now, we are considering options at this point", then
  re-triaged "as not a bug but a feature request/improvement request". Closed with: "we
  now have the possibility to select the FFT algorithm manually, see the example in main:
  `tfhe/examples/manual_fft.rs`. The feature is available in v1.5.0 onward."
- **Issue #177**, "Deterministic seeder" (2023, closed) —
  https://github.com/zama-ai/tfhe-rs/issues/177. About keygen/encryption determinism, not
  evaluation. Zama: "Using a deterministic seed was not intended to be an high-level
  feature for obvious security reasons".
- **PR #2627**, "add reproducibility_test", **open draft since 2025-08-01, no description,
  no comments** — https://github.com/zama-ai/tfhe-rs/pull/2627. By a Zama engineer. It
  seeds `DeterministicSeeder`, generates 60 keysets, runs `apply_lookup_table` over 30,000
  ciphertexts each, bincode-serializes, SHA-256s, and folds to one hash — i.e. exactly the
  test this note argues for. It depends on `features = ["shortint",
  "experimental-force_fft_algo_dif4"]`, so Zama internally knows the forced FFT algorithm
  is a precondition for byte-reproducible output. It has sat untouched for a year.
- **No issue anywhere in the repo asks for byte-reproducible evaluation as a guarantee.**
  Searches for `nondeterministic`, `seeded evaluation`, `bitwise identical`,
  `multiple parties` return zero results. GitHub Discussions are disabled on the repo.
- **No 1.x release note mentions determinism or reproducibility** — including 1.5.0's,
  which does not mention the manual FFT selection that closed #1765.

### In the docs

Only one page documents evaluation-level non-determinism, and it is scoped to a feature
we do not use — `configuration/parallelized-pbs.md`:

> By nature, the parallelized PBS might not be deterministic: while the resulting
> ciphertext will always decrypt to the correct plaintext, the order of the operations
> could vary, resulting in different output ciphertext. To ensure a consistent ciphertext
> output regardless of execution order, add the `with_deterministic_execution()` suffix
> to the parameters.

The encrypted-PRF page hedges reproducibility with "**assuming the same hardware**".
Nothing states, either way, whether ordinary evaluation is byte-reproducible.

### Zama's own products have exactly our problem, and solve it exactly our way

This is the most important finding in this section, and it contradicts `attestation.md` §6.

- **fhEVM agrees by hashing the serialized result ciphertext and majority-voting.**
  `coprocessor/fhevm-engine/sns-worker/src/aws_upload.rs` defines
  `compute_digest(ct: &[u8]) -> Vec<u8>` as a plain `Keccak256` over the compressed
  ciphertext bytes. `gateway-contracts/contracts/CiphertextCommits.sol` has each
  coprocessor call `addCiphertextMaterial(ctHandle, keyId, ciphertextDigest,
  snsCiphertextDigest)`, counts identical submissions, and declares consensus at
  `getCoprocessorMajorityThreshold()`. That is DISCA L0, in production, from Zama.
- **They handle the residual divergence operationally, not cryptographically.**
  `docs/protocol/architecture/coprocessor.md` documents *drift auto-reversal*: a
  coprocessor whose digest disagrees with the consensus digest "treats itself as the
  drifted node and automatically reverts its own state". It requires "at least 3
  registered coprocessors with the threshold set to a strict majority" and must **not**
  be enabled with two or fewer. `docs/metrics/metrics.md` ships
  `coprocessor_gw_listener_drift_detected_counter`, `drift_revert_success/failure_counter`,
  and `drift_revert_too_many_attempts_counter`.
- **They derive re-randomization seeds by hashing the inputs, so evaluation stays
  deterministic.** `coprocessor/fhevm-engine/scheduler/src/dfg/scheduler.rs` builds a
  `ReRandomizationContext` from the domain separator `b"TFHE_Rrd"`, the opcode, and every
  input ciphertext, then re-randomizes from the derived seed. No node-local RNG anywhere
  in the evaluation path.
- **They do not use ZK proofs of evaluation** (ZKPoK is for input validity only), and they
  *deprecated* their designated-executor fast path — the dead
  `priorityConsensusTxSender` mapping is still in `CiphertextCommits.sol` with a comment
  saying so.
- On the forum, asked "How do coprocessors achieve consensus?", Zama's `benoit` replied:
  "yes, multiple copro compute the same thing since it's completely deterministic. Then,
  there is a consensus." (https://community.zama.org/t/how-do-coprocessors-achieve-consensus/3768)
  — with none of the preconditions from §2 stated.

**Consequence for us:** `attestation.md` §6's claim that "byte equality of compressed
ciphertexts is not a property tfhe-rs provides" is too strong. It is a property tfhe-rs
provides *conditionally*, the conditions are knowable, and Zama's own protocol depends on
them. What tfhe-rs does not provide is the *documentation* of those conditions.

---

## 5. The security tension, honestly

The brief asked me to engage with this rather than assume it. The honest answer is that
the tension is real in the literature but **does not apply to the request we would make**,
and it is worth being precise about why.

**Where the worry comes from.** Li–Micciancio (Eurocrypt 2021) introduced IND-CPA^D and
proved IND-CPA ⟺ IND-CPA^D only for schemes with negligible decryption failure
probability. Cheon–Choe–Passelègue–Stehlé–Suvanto (eprint 2024/127, CCS 2024) and
Checri–Sirdey–Boudguiga–Bultel (eprint 2024/116, CRYPTO 2024) then attacked *exact*
schemes, TFHE included, through imperfect correctness.

**What those attacks actually exploit — and it is not evaluation randomness.** The
TFHE-specific attack in 2024/127 §4.2 targets "the large rounding error in the ModSwitch
step", noting that "information concerning this error is publicly available". Each
decryption failure yields an inequality on ⟨ẽ, s⟩ + e, i.e. an LWE-with-hints instance;
collect enough and recover the key. The attack works *because* the modulus-switch error
is a deterministic, publicly computable function of the ciphertext — not because
evaluation lacks entropy. The BFV/BGV attack in the same paper amplifies **encryption**
noise via `ct ← Eval(Add, ct, ct)`; homomorphic addition contributes no noise of its own.
Checri et al.'s prescribed smudging is "**after partial decryption**", i.e. in the
threshold-decryption layer, not in `Eval`. They note their attack hits tfhe-rs only "when
the public 'unchecked addition' function […] is used" — an API-misuse path.

**What Zama actually defends with.** Their default parameter sets target
`p_fail = 2^-128`, and the security page says a "failure probability below 2^−128 ensures
that our implementation is resilient against attacks in the IND-CPA-D model". The
mechanism is failure probability, not evaluation randomness. Separately, the drift
technique (Bernard–Joye–Smart–Walter, "Drifting Towards Better Error Probabilities in
Fully Homomorphic Encryption Schemes", Eurocrypt 2025 — all four authors at Zama) is a
*countermeasure to* the ModSwitch attack, and the paper explicitly says the transformation
"can be deterministic, random, or pseudo-random" and that in their scheme "what matters is
to keep the deterministic nature of the process".

**Where randomness genuinely is load-bearing, and how Zama already reconciles it.**
sIND-CPA^D in the public-key setting does need randomized evaluation. Smart & Walter,
"Reactive Correctness, sIND-CPA^D-Security and **Deterministic Evaluation** for TFHE"
(eprint 2025/2005, IACR CiC 3(1), 2026) is about precisely this conflict, and its
motivation is our use case verbatim:

> In [Sma23] a more lightweight proposal to obtain verifiable evaluation is proposed;
> namely the homomorphic evaluation is performed by (say) n independent parties, and the
> result is taken to be the values which are output by a majority of the parties.
> **However, this requires the evaluation operation to be deterministic.** Thus the above
> randomization […] is in direct conflict with the deterministic evaluation needed to
> obtain cheap verifiability…

Their resolution is to de-randomize by deriving the seed from a random oracle over the
inputs: `seed_i ← Hash(ct_0, …, ct_{m−1}, aux_0, …, aux_{m−1}, F, i)`. That is exactly
what tfhe-rs's `ReRandomizationContext` implements and what fhEVM's scheduler calls.

**Conclusion on the tension.** Asking for byte-reproducible evaluation is **not** asking
Zama to weaken anything, because (a) by default there is no evaluation randomness to
remove, (b) the defences that matter are parameter choice and decryption-time smudging,
and (c) where randomness *is* needed, the sanctioned construction is already deterministic
by design. The one line that must not be crossed: a "deterministic mode" that worked by
**deleting** re-randomization would lose sIND-CPA^D in the public-key setting. The request
must therefore be scoped to *making the existing determinism explicit and controllable*,
never to *removing entropy*.

---

## 6. Verdict

**(c), with a small piece of (b).**

Not (a): a supported path exists for the root cause, but it is `#[doc(hidden)]`, panics if
called late, is per-polynomial-size, and is mentioned in no release note or doc page.
Calling that "supported" would be generous.

Not (d): there is no conflict with security goals (§5), and Zama's own protocol depends on
the property.

The genuine gap is **documentation of a guarantee, plus a safe way to opt into it**.
Concretely, tfhe-rs should say what is and is not reproducible, and offer one call that
turns on everything needed rather than making each user rediscover `setup_custom_fft_plan`
from a `doc(hidden)` module. That is worth filing, and issue #1765's history suggests Zama
will be receptive — they already shipped the primitive, they just never framed it as the
determinism story or wrote it down.

### What DISCA should do, independent of whether Zama acts

1. **Call `setup_custom_fft_plan` at worker start-up**, before touching the server key.
   Measured: 18/18 agreement, no slowdown. This restores L0 as designed and makes
   `architecture.md` §3's options 1 and 3 unnecessary for the hackathon.
2. **Require a homogeneous worker fleet** (same ISA, and until §2b is tested, same core
   count), and say so in `architecture.md` §3 as an explicit trust-model assumption. This
   is what Zama does.
3. **Keep a divergence path anyway.** Even Zama's production system has
   `drift_detected_counter`. Byte equality is an operational property, not a theorem, and
   should not be the *only* thing standing between a job and settlement.
4. **Correct the two documents.** `architecture.md` §3's "likely mechanism is randomness
   drawn during evaluation (noise management)" is wrong; `attestation.md` §6's "byte
   equality […] is not a property tfhe-rs provides" is too strong.

---

## 7. Ready-to-file issue draft

Written for tfhe-rs maintainers. Leads with the minimal reproduction; says nothing about
DISCA.

---

**Title:** Document (and provide a supported switch for) byte-reproducible evaluation: the default FFT plan is chosen by a runtime benchmark

**Body:**

Evaluating the same circuit over byte-identical input ciphertexts with a byte-identical
server key, in separate processes on one machine, intermittently produces result
ciphertexts that decrypt to the same plaintext but serialize to different bytes.

This matters for any deployment where independent parties evaluate the same circuit and
compare results — including Zama's own fhEVM coprocessors, which reach consensus by
majority vote over `Keccak256` of the serialized ciphertext (`sns-worker/src/aws_upload.rs`,
`gateway-contracts/contracts/CiphertextCommits.sol`).

**Reproduction** (tfhe 1.5.0, `boolean,shortint,integer`, release build; observed on
aarch64 macOS, and consistent with #1765's reports on x86):

```rust
// setup: write a CompressedServerKey and four CompressedFheInt32 to disk with
// safe_serialize, from ConfigBuilder::default().

// eval (run this as N concurrent processes, all reading the same files):
let csk: CompressedServerKey = safe_deserialize(sk_bytes, LIMIT)?;
set_server_key(csk.decompress());
let inputs: Vec<FheInt32> = /* safe_deserialize + decompress the four blobs */;

let mut best = inputs[0].clone();
for c in &inputs[1..] {
    best = c.gt(&best).if_then_else(c, &best);
}

let mut out = Vec::new();
safe_serialize(&best, &mut out, LIMIT)?;
println!("{}", sha3_256_hex(&out));
```

Six rounds x three concurrent processes = 18 evaluations: **four distinct result digests**,
unanimous in only 2 of 6 rounds. All 18 decrypted correctly. Digests of the decompressed
inputs were identical in all 18, so this is not deserialization or decompression.

**Cause.** `Fft::new` in `core_crypto/fft_impl/fft64/math/fft/mod.rs` builds each plan with
`Method::Measure(Duration::from_millis(10))` unless `experimental-force_fft_algo_dif4` is
set. The winning algorithm therefore depends on machine load at first use. Different
algorithms are numerically equivalent but not bit-identical, so a few torus coefficients
round differently after the inverse transform. Because the plan is cached in a `OnceLock`
per process, a single process is self-consistent — which is why in-process determinism
tests pass and cross-process comparison fails. This is the same root cause as #1765.

**Fix that works today.** Calling
`core_crypto::fft_impl::fft64::math::fft::setup_custom_fft_plan` with
`Method::UserProvided { base_algo: FftAlgo::Dif4, .. }` before anything else, as in
`examples/manual_fft.rs`, gave **18/18 identical digests**, unchanged for
`RAYON_NUM_THREADS` in {1, 2, 4, 8, 16, 32, 64}, with no measurable change in wall clock.

**What I'd like from tfhe-rs.** The primitive exists; what's missing is that it is
discoverable, safe, and documented as the determinism story.

1. **Document the guarantee.** State in the docs whether evaluation is byte-reproducible,
   under what conditions, and what breaks it. Today the only statement anywhere is
   `configuration/parallelized-pbs.md`, scoped to multi-bit PBS. Meanwhile the NIST
   Threshold Call writeup (§3.4.1) says outputs differ across x86 and ARM, and a forum
   answer says coprocessor evaluation is "completely deterministic" — those should be
   reconciled in one place.
2. **Make opting in safe and single-call.** `setup_custom_fft_plan` sits behind
   `#![doc(hidden)]` (`fft_impl/fft64/mod.rs:1`), must be called once per polynomial size,
   requires the caller to know which sizes a config will touch, and **panics** if any FFT
   has already run for that size, because the setter is `.set(plan).unwrap()` on an
   already-initialised `OnceLock` — decompressing a server key is enough to trip it.
   Something like a config-level `deterministic_fft()` that pins every size the key set
   will use, and is idempotent rather than panicking, would remove the whole footgun.
3. **Say where the remaining divergence sources are.** At minimum:
   `integer/server_key/radix_parallel/add.rs` selects a carry-propagation algorithm via
   `rayon::current_num_threads()`, so machines with different core counts can in principle
   diverge even with the FFT pinned; multi-bit PBS needs `with_deterministic_execution()`;
   and CPU/GPU/HPU bit-exactness is unstated.
4. Optionally: land something like the reproducibility harness in draft PR #2627, so the
   property is covered by CI rather than being incidental.

Happy to open a PR against the docs if that is the easier path.

---

## 8. Sources

**Source read locally** (`~/.cargo/registry/.../tfhe-1.5.0/`):
`src/core_crypto/fft_impl/fft64/math/fft/mod.rs` (:104–123 plan cache and
`setup_custom_fft_plan`, :180–200 `Method::Measure`), `src/core_crypto/fft_impl/fft64/mod.rs:1`
(`#![doc(hidden)]`), `src/core_crypto/algorithms/modulus_switch_noise_reduction.rs`,
`src/shortint/server_key/modulus_switch_noise_reduction.rs`, `src/shortint/engine/mod.rs`,
`src/shortint/parameters/aliases.rs`,
`src/shortint/parameters/v1_4/classic/tuniform/p_fail_2_minus_128/ks_pbs.rs`,
`src/integer/server_key/radix_parallel/add.rs`, `src/high_level_api/re_randomization.rs`,
`src/high_level_api/keys/client.rs`, `src/high_level_api/config.rs`, `examples/manual_fft.rs`.

**Measured locally**: clean-room harness described in §1 (pure tfhe-rs, no DISCA code).

**Retrieved**: tfhe-rs issues [#1765](https://github.com/zama-ai/tfhe-rs/issues/1765),
[#177](https://github.com/zama-ai/tfhe-rs/issues/177), draft PR
[#2627](https://github.com/zama-ai/tfhe-rs/pull/2627); tfhe-rs docs
`configuration/parallelized-pbs.md`, `get-started/security-and-cryptography`; fhEVM
`CiphertextCommits.sol`, `sns-worker/src/aws_upload.rs`, `scheduler/src/dfg/scheduler.rs`,
`docs/protocol/architecture/coprocessor.md`, `docs/metrics/metrics.md`; Zama forum
[thread 3768](https://community.zama.org/t/how-do-coprocessors-achieve-consensus/3768);
Zama NIST Threshold Call writeup "TFHE, ZHEnith and Nexus" v0.1 (2026-01-09) §3.4.1.

**Papers**: Li & Micciancio, Eurocrypt 2021; Cheon–Choe–Passelègue–Stehlé–Suvanto,
[eprint 2024/127](https://eprint.iacr.org/2024/127) (CCS 2024);
Checri–Sirdey–Boudguiga–Bultel, [eprint 2024/116](https://eprint.iacr.org/2024/116)
(CRYPTO 2024); Bernard–Joye–Smart–Walter, "Drifting Towards Better Error Probabilities…",
Eurocrypt 2025; Smart & Walter, "Reactive Correctness, sIND-CPA^D-Security and
Deterministic Evaluation for TFHE", [eprint 2025/2005](https://eprint.iacr.org/2025/2005),
IACR CiC 3(1) 2026.

**Unknown / not established**: whether the `current_num_threads` carry-propagation branch
(§2b) actually flips for realistic circuit shapes; whether GPU/HPU backends are bit-exact
with CPU; whether Zama's production coprocessor fleet is architecturally homogeneous by
policy or only in practice.
