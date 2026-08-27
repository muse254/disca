# Getting started

Running DISCA on your own machine: what the parts are, the one command that
stands all of them up, and how to do the same thing by hand when you want to see
where each piece of data goes.

This is the operator's document. The *why* lives in
[architecture.md](architecture.md) (constraints and trust model) and
[attestation.md](attestation.md) (why results are replicated at all); read those
if a step here looks arbitrary. Before running this on more than one machine,
read [architecture.md](architecture.md) §3: moving workers onto separate hosts is
the one change that can break correctness while every process stays honest, and
what has to match is not obvious.

## What you are standing up

Four parties, and they are separate processes because they are separate parties:

| Party | Binary | Holds | Sees |
|---|---|---|---|
| Key holder | `disca-cli` | the client key | plaintext — the only party that does |
| Coordinator | `node coordinator` | the server key blob, the bytecode | ciphertext |
| Worker ×3 | `node worker` | the server key, its own secp256k1 key | ciphertext |
| (optional) Watcher | `node watcher` | a coordinator key for the chain | ciphertext |

Each worker evaluates the whole circuit and reports `keccak256` of its
compressed result, signed with its own key. The coordinator settles on the first
hash that **M** distinct registered workers signed for. Nothing in that loop can
decrypt anything; only `disca-cli` can, and it never talks to a worker.

The demo runs N = 3 workers with M = 2, and **the third worker is deliberately
wrong**. That is the point: a run where everyone agrees looks identical to a run
with no verification at all.

## Prerequisites

- **Rust, stable.** `rust-toolchain.toml` pins `channel = "stable"`, so
  `rustup` will fetch whatever stable is current. Nothing here needs nightly and
  nothing here needs a C toolchain — `k256` is pure Rust, chosen partly for that
  reason (see the comment on it in the workspace `Cargo.toml`).
- **`bash`, `lsof`, `nc`, `mktemp`.** `scripts/run-local.sh` uses all four:
  `lsof` to kill a worker left over from a previous run, `nc` to wait for the
  three workers to bind before the coordinator fans out to them.
- **Time and disk for the first build.** `alloy` puts ~280 crates in `node`'s
  tree and `tfhe` is the expensive one. CI measures the whole cold build plus the
  full test suite at about **12 minutes** on a GitHub `ubuntu-latest` runner
  (`.github/workflows/ci.yml`, `test` job — the comment there records the run
  ids). Subsequent builds are seconds.
- **A CPU you are not sharing with the demo.** Three workers evaluating the same
  circuit concurrently on 8 cores took 2,858–3,992 ms per job against 1,412 ms
  for one worker alone ([attestation.md](attestation.md) §5a). It still settles;
  it is just slower than the single-process number suggests.

No chain is needed. `anvil`, `cast` and `forge` are only for the
[chain path](#the-chain-path) below.

## The one command

```sh
./scripts/run-local.sh
```

It builds `node` (with `fault-injection`) and `disca-cli` in release, generates
a keypair into a temporary directory, compiles `committee-tally/committee_tally.wasm`,
encrypts the scores `71,93,42,88`, starts three workers on ports 8081–8083 and a
coordinator on 8080, waits for a 2-of-3 settlement, and decrypts the winner.

Three modes, and you want at least the first two:

```sh
./scripts/run-local.sh              # 2-of-3, one worker deliberately faulty
HONEST=1 ./scripts/run-local.sh     # all three honest; nobody accused
ATTESTERS=3 ./scripts/run-local.sh  # unanimity against a liar — fails, by design
```

Run the first two as a pair. A detector that never fires and one that always
fires would each pass a single test; only the pair says the mechanism tracks
reality. The third is the honest demonstration that M is a real threshold rather
than decoration.

Other knobs, all environment variables read at the top of the script:
`PROGRAM`, `FUNCTION`, `INPUTS`, `ATTESTERS`, `DEADLINE`, `EXPECT`, `HONEST`.

### What to look for in the output

Four things, in order:

1. **`worker listening`**, three times, each with an `address=0x…`. That address
   is the worker's identity — it is what `--registered-worker` must contain and
   what `registerWorker` would pin on-chain. The demo derives it from `--id`, and
   each worker says so at startup: *"no --key given; attesting under a key
   derived from the worker id, which anyone can recompute."* That warning is
   correct and you should never see it on a real deployment.
2. **`attestation disagreement`**, once *per distinct result hash*, each naming
   the addresses that signed for it. The coordinator does not accuse anybody —
   it prints the split and lets you see who is in the minority
   (`node/src/coordinator.rs`, the `groups.len() > 1` branch). In the demo you
   get two lines: one address on one hash, two on another. The lone address is
   the faulty worker, and you can check that against the `address=` it logged at
   startup. With the FFT plan pinned that minority is only ever the genuinely
   faulty worker; before it was pinned, honest workers landed there more often
   than the faulty one ([attestation.md](attestation.md) §5c) — which is why the
   warning is evidence now and was not before.
3. **`job settled`**, with the hash and the attester addresses in ascending
   order. Ascending because `fulfillJob` requires strictly increasing addresses,
   so duplicate detection costs one comparison each (`bridge.md` §2a).
4. **`key holder decrypted: 93`**. The coordinator wrote a blob it could not
   read; only the last step can, and only because it holds the client key. The
   script asserts the value, because every earlier check is about workers
   *agreeing*, and workers agreeing on the wrong answer would satisfy all of
   them.

If you want the log stream as JSON instead of formatted text, set
`DISCA_LOG_FORMAT=json`. It goes to stderr, leaving stdout for what a command
returns.

### What a run costs

One run of the defaults, taken from the log lines above on an 8-core Apple
M-series laptop with an already-warm `target/`. Your absolute numbers will
differ; the *shape* is the useful part.

| | |
|---|---|
| Whole script, including the incremental build | 15.1 s wall |
| Server key served, per worker | 30,146,955 bytes |
| Fetch + verify + decompress + install that key | 402–433 ms per worker, over loopback |
| Job dispatch body, per worker | 10,269 bytes |
| Evaluate `tally4_select`, three workers at once | 2,844–3,057 ms |
| Result blob | 12,075 bytes |
| Dispatch to settled | 3,487 ms |

Two things worth noticing. The server key is ~2,900× the size of everything else
on the wire combined, and it is fetched **once per worker per key** — the worker
caches it and only refetches when a dispatch names a different hash
(`node/src/worker.rs`). And the whole job is CPU, not network: 3.0 s of
evaluation against ~30 KB of messages.

## The single-process path

The fastest way to tell whether a failure is in the execution core or in the
transport:

```sh
cargo run -p node -- demo
RUST_LOG=debug cargo run -p node -- demo   # per-circuit spans
RUST_LOG=trace cargo run -p node -- demo   # per-opcode timings
```

**Debug builds only**, deliberately: `demo` doubles as the key holder, encrypting
and decrypting in the same process, which is exactly the separation a deployment
has to keep. `#[cfg(debug_assertions)]` keeps the role out of a release binary.

Expect it to be slow. Debug evaluation is 87–98× slower than release
([architecture.md](architecture.md) §2) — an i32 add goes from 225 ms to 22 s.
The `[profile.dev.package."*"] opt-level = 3` in the workspace manifest already
optimises dependencies; it is the workspace's own crates that stay unoptimised.

## Doing it by hand

`run-local.sh` is the same seven commands with the bookkeeping done for you.
Running them yourself is worth doing once, because it makes the party boundaries
visible: notice that no command below hands a private key or a plaintext to
anything but `disca-cli`.

```sh
cargo build --release -p node -p disca-cli --features node/fault-injection
NODE=target/release/node
CLI=target/release/disca-cli
work=$(mktemp -d)
```

**1. The key holder generates a keypair.** `client.key` (23.5 KB) never leaves
this directory. `server.key` is the compressed evaluation key — 28.8 MB — and is
public.

```sh
$CLI keygen --out-dir "$work/keys"        # prints server_key_hash=0x…
```

`keygen` refuses to overwrite an existing `client.key` without `--force`, because
that key is the only thing that can ever decrypt a result and jobs settle
asynchronously.

**2. Compile the program.**

```sh
$CLI compile --input committee-tally/committee_tally.wasm \
             --output "$work/program.bytecode"           # prints bytecode_hash=0x…
```

The `.wasm` is committed, so this works without a WASM toolchain. If you build
your own, build it **`--release`**: at `-O` rustc flattens `if`/`else` and
fixed-size loops into `select`, which the circuit model can represent, and an
unoptimised build spills locals to linear memory, which it cannot
([architecture.md](architecture.md) §2a). The compiler rejects the debug build
rather than lowering it wrongly.

**3. Encrypt the inputs.**

```sh
$CLI encrypt --client-key "$work/keys/client.key" \
             --values 71,93,42,88 --out-dir "$work/inputs"   # prints commitment=0x… each
```

Order matters and nothing downstream can notice a transposition — the inputs are
ciphertext, so the wrong order produces a plausible answer to a different
question.

**4. Learn the workers' addresses.** The coordinator has to be told whose
signatures count *before* anything runs.

```sh
for id in worker-1 worker-2 worker-3; do RUST_LOG=off $NODE worker-address --id "$id"; done
```

`RUST_LOG=off` because that command prints its answer with `println!` and a stray
log line on the same stream would be substituted into `--registered-worker` and
rejected as a malformed address.

**5. Start the workers.**

```sh
$NODE worker --id worker-1 --bind 127.0.0.1:8081 &
$NODE worker --id worker-2 --bind 127.0.0.1:8082 &
$NODE worker --id worker-3 --bind 127.0.0.1:8083 --faulty &
```

`--faulty` needs the `fault-injection` feature and corrupts only the answer, at
the last step before sealing: the worker still fetches and verifies the real
server key, validates the bytecode, checks input commitments and performs the
real homomorphic evaluation. From outside it is indistinguishable from an honest
worker.

Omit `--key` and each worker derives one from `--id`. **That is not a secret.**
It exists so a shell script needs no key-distribution step. A deployment passes
`--key`.

**6. Run the job.**

```sh
$NODE coordinator \
  --worker 127.0.0.1:8081 --worker 127.0.0.1:8082 --worker 127.0.0.1:8083 \
  --registered-worker 0x… --registered-worker 0x… --registered-worker 0x… \
  --attesters 2 \
  --server-key "$work/keys/server.key" \
  --bytecode "$work/program.bytecode" \
  --function tally4_select \
  --input "$work/inputs/input-0.ct" --input "$work/inputs/input-1.ct" \
  --input "$work/inputs/input-2.ct" --input "$work/inputs/input-3.ct" \
  --result "$work/result.blob" \
  --attestations "$work/attestations.json" \
  --deadline-secs 120
```

`--worker` and `--registered-worker` are separate flags on purpose, and they are
different questions: one says *where to send work*, the other says *whose
signature is worth counting*. Dispatching to a machine does not entitle it to
vote.

Each worker pulls the server key once from `GET /keys/<hash>` on the coordinator,
verifies the bytes hash to the advertised value, and installs it. That is 28.8 MB
per worker, once per key.

**7. Decrypt.** The coordinator wrote a file it cannot read.

```sh
$CLI decrypt --client-key "$work/keys/client.key" \
             --server-key "$work/keys/server.key" \
             --input "$work/result.blob"       # -> 93
```

The server key is needed here only because expanding a *compressed* result is a
server-key operation; decrypting is not.

## Build shapes

A default release build has no `demo` role and no `--faulty` flag — neither
belongs in a production binary:

```sh
cargo build --release -p node                              # what you would ship
cargo build --release -p node --features fault-injection   # adds --faulty
cargo build -p node                                        # debug; adds `demo`
```

Build both binaries in **one** `cargo` invocation when you need the feature.
Two calls with different feature sets resolve the graph differently, so each
invalidates the other's `primitives` and every run recompiles from scratch —
minutes of it. `--features node/fault-injection` names the feature on the
package rather than on the current one, which is what lets a single invocation
carry it.

## Things that will bite you

- **A leftover worker on 8081–8083.** The new one fails to bind and the *old*
  one — holding an old server key — answers this run's jobs. `run-local.sh` kills
  them first; by hand, check with `lsof -ti tcp:8081`.
- **Running the debug binary for anything timed.** 87–98× is enough to make a
  working system look broken.
- **`--attesters 3` with the faulty worker.** It fails, and that is the correct
  outcome, not a bug. `M = N` tolerates no faulty or missing worker.
- **A wrong `--function` or transposed `--input` order.** Both produce a
  perfectly valid ciphertext for a different question. FHE gives
  confidentiality, not integrity: `decrypt` on a wrong-but-well-formed ciphertext
  returns a number, not an error ([attestation.md](attestation.md) §1).
- **Memory.** Each worker installs the server key *decompressed*, at 114.8 MB, in
  tfhe's thread-local storage, on top of the working set of an evaluation. Three
  workers on one laptop is fine; it is worth knowing before you run thirty.
- **Anything that makes two workers differ.** Same binary, same CPU, same `tfhe`
  version, CPU not GPU, FFT plan pinned. On one machine you get all of that for
  free, which is exactly why the constraint is invisible locally. The list, and
  the one item on it that is still unverified, is [architecture.md](architecture.md)
  §3.

## The chain path

The same thing against a real chain — Anvil, the contracts, escrow, and a
`fulfillJob` that counts the workers' own signatures. Needs `anvil`, `cast` and
`forge` on the path (`foundryup`):

```sh
./scripts/run-anvil.sh              # the script drives the lifecycle with cast
./scripts/run-anvil.sh --watcher    # `node watcher` settles it; no cast send
./scripts/run-anvil.sh --synthetic  # contracts and chain only, no Rust, no FHE
```

`--watcher` is the one that shows DISCA settling by itself. The script sends no
`fulfillJob` in that mode and proves it: it records every transaction hash it
produces and asserts the settling transaction is not one of them
([bridge.md](bridge.md) §4, §8 step 3).

One constraint to know before pointing `--rpc` anywhere real: **the watcher
refuses an `https://` URL at startup** (`node/src/watcher.rs`). This build has no
TLS transport — the `alloy` features in the workspace manifest deliberately omit
`reqwest-rustls-tls`, and the manifest says what adding it costs. So the watcher
talks to Anvil, a local node, or something on the other end of an ssh tunnel, and
not to a hosted RPC provider.

## Checks

The same commands CI runs, so a green laptop and a red pipeline cannot disagree
about why:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
cargo check --workspace --all-targets --locked
cargo test -p node --locked
scripts/check-deps.sh
scripts/coverage.sh --check
```

`--locked` is not decoration: it is the other half of the exact `tfhe` pin.
Building against a lockfile cargo was allowed to rewrite would defeat the pin
silently, and the symptom is honest workers no longer agreeing.

`pre-commit install` wires the first group to **commit** and the coverage floor
to **push**. The hook deliberately never runs `cargo update`;
`scripts/check-deps.sh --report` shows what an update *would* change without
changing anything.

## Where to go next

- [architecture.md](architecture.md) — the constraints, the measured FHE costs,
  and §3 on why byte-reproducibility is the whole security model.
- [attestation.md](attestation.md) — why replication, and what it costs.
- [bridge.md](bridge.md) — the Ethereum boundary.
- [tasks.md](tasks.md) Track 5 — what running this somewhere other than a
  laptop requires. A cloud deployment plan exists but is kept out of the
  repository, because it is specific to one account's machine types and prices;
  the part of it the rest of these docs rely on is written out in 5.3.
- [tasks.md](tasks.md) — what is built and what is next, kept current rather than
  aspirational.
