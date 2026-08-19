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

## Status

The execution core and the distributed layer work. The Ethereum bridge is
designed but not built — see [tasks.md](docs/tasks.md).

## White Paper: The Disca Specification

The description and formal specification of the Disca protocol.

Built with XeLaTex.

### Setup

#### Pre-commit

Install the [`pre-commit` CLI tool](https://pre-commit.com/), available via brew on macOS:

```sh
brew install pre-commit
```

Install the pre-commit hooks:

```sh
pre-commit install
```

This hook is necessary to ensure that the document is properly formatted and spell-checked.

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