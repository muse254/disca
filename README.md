# disca

A distributed computer that executes programs under fully homomorphic
encryption: nodes evaluate circuits over ciphertexts they cannot decrypt, and
agreement between independent nodes is what makes a result trustworthy.

Design docs: [architecture.md](docs/architecture.md) (constraints, trust model)
· [bridge.md](docs/bridge.md) (Ethereum boundary)
· [attestation.md](docs/attestation.md) (why M-of-N, and what it costs)
· [tasks.md](docs/tasks.md) (what is built and what is next)
· [tfhe-determinism-request.md](docs/tfhe-determinism-request.md) (why evaluation
must be pinned to be reproducible)

Design decisions carry a pointer to the pull request that produced them, so the
reasoning and the dead ends stay recoverable rather than only the conclusion.

## Running it

Three worker processes evaluate an encrypted committee tally; the coordinator
settles on the result two of them independently attest to. The third worker is
deliberately faulty — a run where everyone agrees demonstrates nothing.

```sh
./scripts/run-local.sh              # 2-of-3, one faulty worker
HONEST=1 ./scripts/run-local.sh     # all three honest
ATTESTERS=3 ./scripts/run-local.sh  # unanimity required; fails, by design
```

The scores `71,93,42,88` are encrypted before they leave the key holder. No
worker sees a plaintext, and the winning score comes back correct.

Single process, no network — the quickest way to tell whether a failure is in
the execution core or the transport. **Debug builds only**: it doubles as the
key holder, encrypting and decrypting in the same process, which is exactly the
separation a deployment has to keep.

```sh
cargo run -p node -- demo
RUST_LOG=debug cargo run -p node -- demo   # per-circuit spans
RUST_LOG=trace cargo run -p node -- demo   # per-opcode timings
```

Inspect what a WASM module lowers to, or re-measure the FHE constants the design
rests on:

```sh
cargo run -p primitives --example inspect -- committee-tally/committee_tally.wasm
cargo run --release -p primitives --example size_probe
```

Tests: `cargo test --workspace`. Note that
`primitives/tests/determinism_under_concurrency.rs` spawns child processes — it
guards byte-reproducibility of evaluation, which M-of-N depends on and which an
in-process test cannot check (see [architecture.md](docs/architecture.md) §3).

## Build shapes

A default release build has no `demo` role and no `--faulty` flag — neither
belongs in a production binary:

```sh
cargo build --release -p node                              # what you would ship
cargo build --release -p node --features fault-injection   # adds --faulty
cargo build -p node                                        # debug; adds `demo`
```

`scripts/run-local.sh` opts into `fault-injection` explicitly, because
demonstrating M-of-N requires a worker that disagrees.

`node/src/main.rs` lists the assumptions the binary makes but does not enforce —
worth reading before trusting a result, since a violation shows up as workers
disagreeing rather than as anything failing loudly.

## Checks

CI runs the first five, and so can you — same commands, same flags, so a green
laptop and a red pipeline cannot disagree about why. The last one runs only
here; see [Coverage](#coverage):

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
cargo check --workspace --all-targets --locked   # ...and default features still build
cargo test -p node --locked                      # ...and behave (no --faulty flag)
scripts/check-deps.sh          # lockfile in sync; tfhe still pinned exactly
scripts/coverage.sh --check    # local only: per-crate coverage floors
```

`--all-features` is not decoration. A module behind an off-by-default feature is
not compiled by an ordinary build, so nothing type-checks it and nothing runs
its tests — it rots quietly while the coverage number stays flattering
([tasks.md](docs/tasks.md) 2.11b).

Neither is the pass *without* it. `--all-features` cannot see a missing feature
gate, because under it everything is enabled; a default build is what a release
actually compiles, and it is the one that has to prove there is no way to ask a
worker for a wrong answer.

`--locked` is the other half of the `tfhe` pin. The exact version in the
workspace manifest is what makes evaluation byte-reproducible
([architecture.md](docs/architecture.md) §3); building against a lockfile cargo
was allowed to rewrite would defeat it.

### Coverage

`scripts/coverage.sh` wraps [`cargo-llvm-cov`][llvm-cov] and writes
`target/llvm-cov/html/index.html`.

The floor is per crate, not one number for the workspace, and each floor carries
its reason in `scripts/lib/coverage_report.py`. `primitives` is the execution
core — pure functions with a checkable answer, and the thing an attestation is a
claim about — so it is held high. Much of `node` is socket binding, thread
spawning and blocking receive loops that only `scripts/run-local.sh` exercises;
tests written to walk those lines would raise the number and assert nothing.
`scripts/coverage.sh --check` applies the floors. It runs on **`pre-push`**, and
not in CI.

That is a deliberate trade. Coverage needs its own cold build of tfhe — a
second one, because `cargo-llvm-cov` builds into a different target directory
with `RUSTC_WRAPPER` set and can share nothing with the test job — and on this
repo's 2-vCPU runner that is hours of billable time on every push, to produce
numbers you can produce here in about a minute warm. What it costs is
enforcement: the floors hold on the machine of whoever is pushing, and a push
made with `--no-verify` is not checked at all. If that stops being an
acceptable trade, the job to restore is a `push:`-on-`main` one — per merge
rather than per PR push.

`--open` instead of `--check` writes and opens the browsable HTML report.

[llvm-cov]: https://github.com/taiki-e/cargo-llvm-cov

### Pre-commit hook

Formatting, linting and dependency hygiene on **commit**; coverage floors on
**push**, because that check measures the whole workspace and takes about a
minute — a per-commit cost that size is one people turn off, and a disabled
hook enforces nothing. Install [`pre-commit`](https://pre-commit.com/) (`brew
install pre-commit` on macOS), then:

```sh
pre-commit install   # installs both the pre-commit and pre-push hooks
```

Both, because `default_install_hook_types` asks for both — a plain
`pre-commit install` without it would leave the coverage gate uninstalled and
silent.

The hook deliberately does **not** run `cargo update`. Rewriting `Cargo.lock` on
every commit is exactly how the exact `tfhe` pin gets lost, and losing it breaks
byte-reproducible evaluation with no error — honest workers simply stop
agreeing. `scripts/check-deps.sh` verifies the pin instead of moving it; run it
with `--report` to see what an update *would* change, without changing anything.

## Status

The execution core and the distributed layer work. The Ethereum bridge is
designed but not built — see [tasks.md](docs/tasks.md).

## White Paper: The Disca Specification

The description and formal specification of the Disca protocol.

Built with XeLaTex.

### Setup

#### Pre-commit

The same hook as [above](#pre-commit-hook) — it spell-checks the document as
well as the Rust.

#### XeLaTex

Install XeLaTex, available via brew on macOS:

```sh
brew install basictex
```

### Build

```sh
make
```

### Clean

```sh
make clean
```

## License

[MIT](LICENSE).