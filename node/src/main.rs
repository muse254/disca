//! DISCA node.
//!
//! One binary, four roles. `worker` evaluates; `coordinator` and `watcher` are
//! the two ways a job reaches one, taking it from a command line and from a
//! chain respectively; `demo` is the original single-process evaluator, kept
//! because it is the fastest way to check the execution core still works
//! without standing up a network.
//!
//! `coordinator` and `watcher` share everything below `accept_job` and differ
//! only in where a job comes from and where its answer goes — argv and a file,
//! or `JobRequested` and `fulfillJob`. They are separate roles rather than one
//! role with chain flags because half of each one's arguments are meaningless
//! to the other, and because keeping them apart is what guarantees that adding
//! a chain could not change what `node coordinator` does. See `watcher.rs`.
//!
//! All output is [`tracing`] rather than `println!`: these are the measurements
//! a worker reports to a coordinator, so they are shaped like job telemetry
//! from the start. `RUST_LOG` sets verbosity — `debug` adds per-circuit
//! evaluation spans, `trace` adds per-opcode timings.
//!
//! # Assumptions this binary makes
//!
//! None of these are enforced yet. They are listed because a violation shows up
//! as workers disagreeing or a job settling on something it should not, rather
//! than as anything failing loudly.
//!
//! * **Every worker runs the same build on the same CPU architecture.** Byte
//!   equality of results is what M-of-N compares, and it holds within an
//!   architecture, not across one — Zama document that x86 and ARM diverge. A
//!   different tfhe version, a `gpu` build, or a binary that never pinned the
//!   FFT plan diverges the same way. Registration should check this and does
//!   not (task 2.10b), so **disagreement currently means divergence, not
//!   dishonesty**, and must not feed slashing.
//! * **The `ConfigBuilder::default()` polynomial size is 2048.** `pin_fft_plan`
//!   hardcodes it and there is no public accessor to check against, which is
//!   why `tfhe` is pinned to an exact version.
//! * **The coordinator is honest about liveness but not trusted for results.**
//!   It can stall a job — the escrow refund path covers that — but cannot forge
//!   one, because the attestation hash has to be one M workers independently
//!   produced.
//! * **A worker is its secp256k1 key, and the registry says which keys count.**
//!   Every report is signed over a claim binding the job id, the program hash
//!   and the result (task 2.10i), and the coordinator recovers the signer
//!   rather than being told who it is. What is *not* enforced is that the
//!   registry passed on the command line matches any on-chain
//!   `registerWorker` set — there is no chain yet (`bridge.md` §2), so the
//!   registry is only as good as the operator who typed it.
//! * **Job ids are unique.** They are bound into every signature, which is what
//!   stops an attestation being lifted onto another job. The coordinator now
//!   mints one per job (`fresh_job_id`), so ids are unique *per coordinator*
//!   and separate concurrent jobs from each other and from earlier runs. They
//!   are not globally unique and they commit to nothing: two coordinators can
//!   still collide, and neither can show a contract that its id was ever
//!   issued. `submitJob` assigning the id is what makes this sound rather than
//!   merely unlikely (task 2.9f), and `JobSpec::job_id` is where that id
//!   arrives.

mod coordinator;
#[cfg(debug_assertions)]
mod demo;
mod protocol;
mod transport;
mod watcher;
mod worker;

use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::{Parser, Subcommand};
use tfhe::core_crypto::fft_impl::fft64::math::fft::{
    FftAlgo, Method, Plan, PolynomialSize, setup_custom_fft_plan,
};
use tracing::{Level, error};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(author, version, about = "DISCA node")]
struct Cli {
    #[command(subcommand)]
    role: Role,
}

#[derive(Subcommand, Debug)]
enum Role {
    /// Fan a prepared job out to workers and settle on an M-of-N result.
    ///
    /// Takes blobs, not secrets: a server key and bytecode to distribute, and
    /// inputs that are already ciphertext. The key holder is `disca-cli`, which
    /// produced all three and is the only party that can read the result.
    Coordinator {
        /// Address to serve the server key and worker reports on.
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,

        /// Worker address. Repeat for each worker.
        #[arg(long = "worker", required = true)]
        workers: Vec<String>,

        /// Ethereum address whose attestations count. Repeat for each.
        ///
        /// This is the worker registry, standing in for `registerWorker` on the
        /// bridge contract (`bridge.md` §2). It is separate from `--worker`
        /// deliberately: that flag says where to send work, this one says whose
        /// signature is worth counting, and the two are different questions.
        /// Dispatching to a machine does not entitle it to vote, and an address
        /// can be registered without this coordinator ever dispatching to it.
        ///
        /// A worker prints its address at startup, and `node worker-address`
        /// computes it without starting one.
        #[arg(long = "registered-worker", required = true)]
        registry: Vec<String>,

        /// How many workers must report the same result (the M in M-of-N).
        #[arg(long, default_value_t = 2)]
        attesters: usize,

        /// Compressed server key to distribute, from `disca-cli keygen`.
        ///
        /// The coordinator serves these bytes and hashes them to name the key.
        /// It never installs one: nothing here evaluates or decrypts.
        #[arg(long)]
        server_key: PathBuf,

        /// DISCA bytecode to run, from `disca-cli compile`.
        #[arg(long)]
        bytecode: PathBuf,

        /// Exported function within that program.
        #[arg(long)]
        function: String,

        /// An encrypted input, from `disca-cli encrypt`. Repeat in argument
        /// order.
        ///
        /// Ciphertext, not values. Plaintext inputs used to be a flag here,
        /// which put the secrets in this process's `argv` — visible to anyone
        /// who can run `ps` on the machine that fans the job out, on a system
        /// whose claim is that no node sees them.
        #[arg(long = "input", required = true)]
        inputs: Vec<PathBuf>,

        /// Run under a job id a chain already assigned, rather than minting one.
        ///
        /// Workers sign a digest binding the job id, and `fulfillJob` rebuilds
        /// that digest from the id `submitJob` returned. Settling on-chain
        /// therefore requires the two to be the same id — without this the
        /// signatures recover to addresses the registry has never heard of, and
        /// a correct result is rejected as if the workers were impostors.
        #[arg(long)]
        job_id: Option<u64>,

        /// Where to write the winning result blob. Still encrypted; feed it to
        /// `disca-cli decrypt`.
        #[arg(long = "result")]
        result_out: Option<PathBuf>,

        /// Where to write the winning group's signatures, as JSON.
        ///
        /// The evidence, beside the answer. `--result` is the ciphertext a key
        /// holder decrypts; this is what a contract is handed to check that the
        /// ciphertext was agreed by M registered workers rather than asserted
        /// by this process — the `Attestation[]` argument `fulfillJob` takes
        /// (`bridge.md` §2, task 3.3).
        ///
        /// Attesters are written in ascending address order, because
        /// `fulfillJob` requires strictly increasing addresses so that
        /// duplicate detection costs one comparison each.
        #[arg(long = "attestations")]
        attestations_out: Option<PathBuf>,

        /// How long to wait for agreement before giving up.
        #[arg(long, default_value_t = 120)]
        deadline_secs: u64,
    },

    /// Take jobs from a `DiscaBridge` contract and settle them back to it.
    ///
    /// The `coordinator` role with the chain attached (task 3.4,
    /// `bridge.md` §8 step 3): it subscribes to `JobRequested`, verifies each
    /// input blob against the commitment the chain is holding, runs the job
    /// through the same job service, and submits `fulfillJob`. Runs until
    /// killed, and keeps several jobs in flight.
    ///
    /// What it is *not* given is a job. There is no `--input`, no `--job-id`
    /// and no `--result`: the inputs, the id and the quorum all come from the
    /// chain, and the result goes back to it. What the operator supplies is the
    /// deployment (`--rpc`, `--bridge`, `--coordinator-key`), the program they
    /// registered, and the workers they run.
    Watcher {
        /// JSON-RPC endpoint of the chain to watch.
        ///
        /// **http only.** This build has no TLS transport — see the `alloy`
        /// features in the workspace manifest for what that saves and the one
        /// line that changes it. An `https://` URL is refused at startup rather
        /// than failing inside the transport once a job has been posted.
        #[arg(long)]
        rpc: String,

        /// The deployed `DiscaBridge`.
        #[arg(long)]
        bridge: String,

        /// secp256k1 private key of the address the bridge holds as
        /// `coordinator()`, 32 bytes of hex.
        ///
        /// Only that address may call `fulfillJob` (`bridge.md` §2), and the
        /// gate is about payment rather than trust: an attestation is valid
        /// because of who signed it, so a stranger relaying a genuine
        /// settlement would be harmless to correctness — and would collect the
        /// escrow, which is a free front-run on every settlement. The watcher
        /// checks this key against `coordinator()` before it waits for a single
        /// event.
        ///
        /// Passing a key on a command line puts it in the process table and the
        /// shell history, exactly as `worker --key` does, and is acceptable for
        /// the same reason and no other: there is nothing here yet to protect.
        #[arg(long)]
        coordinator_key: String,

        /// Address to serve the server key and worker reports on.
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,

        /// Worker address. Repeat for each worker.
        #[arg(long = "worker", required = true)]
        workers: Vec<String>,

        /// Ethereum address whose attestations count. Repeat for each.
        ///
        /// Still supplied rather than read from the chain, because the bridge's
        /// registry is a mapping and a mapping cannot be enumerated. What the
        /// chain *can* answer is whether a given address is in it, so every
        /// entry here is checked against `isRegisteredWorker` at startup — the
        /// alternative is a `NotRegisteredWorker` revert after three workers
        /// have finished evaluating.
        #[arg(long = "registered-worker", required = true)]
        registry: Vec<String>,

        /// The `programId` this watcher runs, as `registerProgram` returned it.
        ///
        /// One program per process, and jobs for any other are skipped. The
        /// coordinator serves exactly one server key at `/keys/<hash>` and
        /// `registerProgram` pins one `serverKeyHash` per program
        /// (`bridge.md` §3), so a watcher fronting two programs would need the
        /// job service to hold two keys — a lifecycle nothing here exercises,
        /// and the honest way to front a second program today is a second
        /// process on a second port.
        #[arg(long)]
        program_id: u64,

        /// DISCA bytecode to run, from `disca-cli compile`.
        ///
        /// Checked against the `bytecodeHash` the program is registered under.
        #[arg(long)]
        bytecode: PathBuf,

        /// Exported function within that program.
        #[arg(long)]
        function: String,

        /// Compressed server key to distribute, from `disca-cli keygen`.
        ///
        /// Checked against the `serverKeyHash` the program is registered under.
        #[arg(long)]
        server_key: PathBuf,

        /// How long to wait for agreement on one job before giving up on it.
        ///
        /// Giving up means doing nothing: the job's escrow returns to its
        /// poster through `refundOnTimeout`, which is where `bridge.md` §6
        /// routes every liveness failure. Keep it comfortably under the
        /// bridge's own `jobTimeout`, or the watcher is still collecting when
        /// the job stops being fulfillable.
        #[arg(long, default_value_t = 120)]
        deadline_secs: u64,

        /// How many blocks to let a job age before dispatching it.
        ///
        /// The reorg mitigation. Zero is the default because Anvil mines only
        /// when a transaction arrives, so a job in the newest block would
        /// otherwise wait for an unrelated transaction to bury it. On a chain
        /// that mines on a timer, set it to that chain's reorg depth — and read
        /// the "Reorgs and restarts" section of `watcher.rs` for what it does
        /// and does not buy.
        #[arg(long, default_value_t = 0)]
        confirmations: u64,

        /// How long to wait between `eth_getLogs` calls.
        #[arg(long, default_value_t = 250)]
        poll_ms: u64,

        /// First block to scan.
        ///
        /// There is no cursor on disk: a restart rescans from here, and jobs
        /// that are no longer `Open` are skipped by reading their state. The
        /// chain is the cursor, which is why a stale file cannot make this
        /// watcher miss a job.
        #[arg(long, default_value_t = 0)]
        from_block: u64,
    },

    /// Evaluate circuits dispatched by a coordinator.
    Worker {
        #[arg(long, default_value = "127.0.0.1:8081")]
        bind: String,

        /// Coordinator address, for pulling the server key and reporting back.
        #[arg(long, default_value = "127.0.0.1:8080")]
        coordinator: String,

        /// Names this worker in logs. Not its identity — see `--key`.
        #[arg(long)]
        id: String,

        /// secp256k1 private key, 32 bytes of hex, `0x` optional.
        ///
        /// **This is the worker's identity.** Its Ethereum address is what the
        /// coordinator's `--registered-worker` list must contain and what an
        /// on-chain registry would hold, and only the holder of this key can
        /// produce an attestation under that address.
        ///
        /// Omit it and the worker derives a key from `--id` instead. That is
        /// **not a secret** — the id is in every log line, so anyone can
        /// recompute it — and it exists so `scripts/run-local.sh` can stand up
        /// a coordinator that already knows three workers' addresses without a
        /// key-distribution step in a shell script. The worker says so at
        /// startup. Never do this in a deployment.
        ///
        /// Passing a key on a command line puts it in the process table and the
        /// shell history. That is acceptable for a demo and not for anything
        /// else; the real answer is a file or an agent socket, and it is not
        /// built because there is nothing yet to protect.
        #[arg(long)]
        key: Option<String>,

        /// Deliberately return a wrong result.
        ///
        /// This is fault injection, not mocking: the worker still fetches and
        /// verifies the real server key, validates the real bytecode, checks
        /// input commitments and performs the real homomorphic evaluation. Only
        /// the answer is corrupted, at the last step before sealing. From
        /// outside it is indistinguishable from an honest worker — same
        /// timings, same well-formed report — which is the point.
        ///
        /// **What it buys.** A run where every worker agrees is
        /// indistinguishable from a run with no verification at all: you see
        /// `job settled` either way. Only a disagreeing worker shows that the
        /// mechanism does anything. Pairing this with `HONEST=1` gives both
        /// halves — a detector that never fires and one that always fires would
        /// each pass a single test.
        ///
        /// **What stays honest.** The fault is injected; the detection is not.
        /// Nothing tells the coordinator which worker is faulty — this flag is
        /// local to one process and appears nowhere in the protocol. The
        /// coordinator groups reports by hash and names the odd one out from
        /// the evidence. Before the FFT plan was pinned it regularly accused
        /// *honest* workers, which is how that bug surfaced; a coordinator with
        /// privileged knowledge of the answer could not have made that mistake.
        ///
        /// **What it does not cover.** Exactly one fault mode: a well-formed
        /// but wrong answer. Not a crash, a hang, garbage bytes, or a worker
        /// that diverges only on some jobs. And notably not the fault a real
        /// deployment would hit first — see `Behaviour` in `worker.rs`.
        ///
        /// Requires the `fault-injection` feature, so a default release build
        /// has no way to return a wrong answer on purpose.
        #[cfg(feature = "fault-injection")]
        #[arg(long)]
        faulty: bool,
    },

    /// Print the Ethereum address a worker would attest under, and exit.
    ///
    /// The address is what a coordinator registers (`--registered-worker`) and
    /// what `registerWorker` would pin on-chain, and it is not something an
    /// operator can compute by hand from a private key. Takes the same `--key`
    /// and `--id` a worker does, so it answers for the exact key that worker
    /// will run with.
    ///
    /// Prints one line to stdout and nothing else, because it exists to be
    /// captured by a script. Everywhere else in this binary output is
    /// `tracing`; this is a value, not telemetry.
    WorkerAddress {
        #[arg(long)]
        id: String,

        /// See `worker --key`. The key is never printed, only its address.
        #[arg(long)]
        key: Option<String>,
    },

    /// Run the execution core in a single process, no network.
    ///
    /// Debug builds only. It is a development aid — the quickest way to tell
    /// whether a failure is in the execution core or in the transport — and it
    /// also acts as the key holder, encrypting and decrypting in the same
    /// process. That is exactly the separation a deployment must keep, so the
    /// role has no business existing in a release binary.
    #[cfg(debug_assertions)]
    Demo,
}

/// Reads one of the blobs `disca-cli` produced, naming the file if it cannot.
///
/// Every input to a coordinator is now a file rather than a value, so "wrong
/// path" is the most likely way to start a job badly. An error that names the
/// path is the difference between fixing a typo and reading a stack trace.
fn read_blob(path: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

fn main() {
    init_telemetry();
    pin_fft_plan();

    let result = match Cli::parse().role {
        Role::Coordinator {
            bind,
            workers,
            registry,
            attesters,
            job_id,
            server_key,
            bytecode,
            function,
            inputs,
            result_out,
            attestations_out,
            deadline_secs,
        } => coordinator::parse_registry(&registry).and_then(|registry| {
            let server_key = read_blob(&server_key)?;
            let bytecode = read_blob(&bytecode)?;
            let inputs = inputs
                .iter()
                .map(|path| read_blob(path))
                .collect::<Result<Vec<_>, _>>()?;

            coordinator::run(coordinator::Config {
                bind,
                workers,
                registry,
                attesters,
                job_id,
                server_key,
                bytecode,
                function,
                inputs,
                result_out,
                attestations_out,
                deadline: Duration::from_secs(deadline_secs),
            })
        }),

        Role::Watcher {
            rpc,
            bridge,
            coordinator_key,
            bind,
            workers,
            registry,
            program_id,
            bytecode,
            function,
            server_key,
            deadline_secs,
            confirmations,
            poll_ms,
            from_block,
        } => coordinator::parse_registry(&registry).and_then(|registry| {
            let bridge = bridge
                .parse()
                .map_err(|e| format!("--bridge is not an Ethereum address: {e}"))?;
            let bytecode = read_blob(&bytecode)?;
            let server_key = read_blob(&server_key)?;

            watcher::run(watcher::Config {
                rpc,
                bridge,
                coordinator_key,
                bind,
                workers,
                registry,
                program_id: alloy::primitives::U256::from(program_id),
                bytecode,
                function,
                server_key,
                deadline: Duration::from_secs(deadline_secs),
                confirmations,
                poll: Duration::from_millis(poll_ms),
                from_block,
            })
        }),

        Role::Worker {
            bind,
            coordinator,
            id,
            key,
            #[cfg(feature = "fault-injection")]
            faulty,
        } => {
            #[cfg(feature = "fault-injection")]
            let behaviour = if faulty {
                worker::Behaviour::Faulty
            } else {
                worker::Behaviour::Honest
            };
            #[cfg(not(feature = "fault-injection"))]
            let behaviour = worker::Behaviour::Honest;

            worker::resolve_key(key.as_deref(), &id).and_then(|key| {
                worker::run(worker::Config {
                    bind,
                    coordinator,
                    id,
                    key,
                    behaviour,
                })
            })
        }

        Role::WorkerAddress { id, key } => worker::resolve_key(key.as_deref(), &id).map(|key| {
            // The one `println!` in this binary: a script substitutes this into
            // the coordinator's `--registered-worker`, and a tracing line would
            // arrive wrapped in a timestamp and a level.
            println!("{}", primitives::attest::hex_address(&key.address()));
        }),

        #[cfg(debug_assertions)]
        Role::Demo => demo::run(),
    };

    if let Err(error) = result {
        error!(%error, "node exited with an error");
        std::process::exit(1);
    }
}

/// Pins the FFT plan so evaluation is byte-reproducible across nodes.
///
/// **This is what makes M-of-N attestation work.** By default `Fft::new` picks
/// between numerically-equivalent FFT algorithms by benchmarking them for 10 ms
/// at first use, so the winner depends on machine load at that instant. The
/// algorithms associate the floating-point butterflies differently, a few torus
/// coefficients round the other way, and two honest workers produce ciphertexts
/// that decrypt identically but differ byte for byte. Agreement then fails at
/// random. Measured on the demo circuit: 1 of 6 rounds unanimous unpinned,
/// 6 of 6 pinned, with no measurable slowdown.
///
/// Called before anything else touches a plan — `setup_custom_fft_plan` panics
/// if the plan for that polynomial size is already initialised, and merely
/// decompressing a server key initialises it.
///
/// Three ways this stops working, all silent:
///
/// * **Different polynomial size.** The plan is per size; 2048 is what
///   `ConfigBuilder::default()` selects for the parameters this build pins.
///   Changing parameters, or a tfhe-rs version that changes the default, leaves
///   evaluation unpinned with no error — which is why `tfhe` is pinned to an
///   exact version in the workspace manifest.
/// * **Mixed CPU architectures.** Zama document that outputs differ between x86
///   and ARM, so byte equality holds within an architecture and not across one.
///   A mixed fleet disagrees no matter what is pinned.
/// * **GPU evaluation.** The `gpu` feature switches the default to multi-bit
///   parameters, which are documented as non-deterministic unless
///   `with_deterministic_execution()` is set. This build is CPU-only.
///
/// A worker violating any of these disagrees with honest workers while behaving
/// honestly, so disagreement is evidence of divergence rather than dishonesty
/// until worker registration enforces them (task 2.10b).
fn pin_fft_plan() {
    let fourier = PolynomialSize(2048).to_fourier_polynomial_size();
    setup_custom_fft_plan(Plan::new(
        fourier.0,
        Method::UserProvided {
            base_algo: FftAlgo::Dif4,
            base_n: fourier.0,
        },
    ));
}

/// Installs the tracing subscriber. `RUST_LOG` wins when set; otherwise we
/// default to `INFO`, which covers phase timings without per-op noise.
///
/// `DISCA_LOG_FORMAT=json` switches to one JSON object per line, on **stderr**.
/// That is task 1b.3, and the reason it took a consumer to land: every field
/// these roles already emit — a worker's `address`, `elapsed_ms` on an
/// evaluation, the `attesters` a job settled on — exists only inside a
/// human-formatted line today. `scripts/run-anvil.sh` reads three of them with
/// `sed` and needs a `strip_ansi` helper to do it, and says so.
///
/// **Why stderr, and only in this mode.** `worker-address` writes its answer
/// with `println!`, and the text formatter defaults to stdout — so today the
/// answer and the log arrive in one stream, which is why `run-local.sh` wraps
/// that call in `RUST_LOG=off`. Sending the JSON stream to stderr splits
/// answers from events: stdout is what a command returns, stderr is what it
/// did. The redirect lives only in this branch, so no existing script sees a
/// byte of difference.
///
/// **Why an environment variable rather than a flag.** This runs before
/// `Cli::parse()`, and deliberately: `pin_fft_plan` must run before anything
/// touches an FFT plan, and clap's own errors should be formatted by a
/// subscriber that already exists. A flag would mean reordering the one
/// function in this binary whose ordering is load-bearing. An environment
/// variable also composes with `RUST_LOG`, which is how this binary is already
/// configured.
fn init_telemetry() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(Level::INFO.to_string()));

    match std::env::var("DISCA_LOG_FORMAT").as_deref() {
        Ok("json") => {
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                // `with_target` stays on here, unlike the text format. A human
                // reading a terminal does not want the module path; a program
                // switching on events does.
                .json()
                .with_writer(std::io::stderr)
                // Both default to on for this formatter; named because a
                // consumer depends on them. `job_id` reaches an event through
                // these, and note it is not always in `span`: `install_server_key`
                // opens a nested `keys.fetch`, so for "server key installed" the
                // current span is that one and `job_id` appears only in the
                // list. A parser must read the list, not the innermost span.
                .with_current_span(true)
                .with_span_list(true)
                .init();
        }
        Ok("text") | Err(_) => text_format(filter),
        Ok(other) => {
            // Deliberately not silent, and deliberately not fatal. A typo that
            // fell through to the text format would produce perfectly readable
            // output that a consumer parses as zero events — a failure that
            // looks like "the system did nothing" rather than like a mistake.
            eprintln!(
                "warning: DISCA_LOG_FORMAT={other:?} is not recognised; using the text format. \
                 Valid values are `json` and `text`."
            );
            text_format(filter);
        }
    }
}

/// The human format, unchanged: stdout, no target, colour when attached.
fn text_format(filter: EnvFilter) {
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    use super::*;

    #[test]
    fn the_cli_definition_is_internally_consistent() {
        // clap validates its own configuration at runtime and panics on a
        // duplicate short flag or a default that does not parse. Doing it here
        // means a broken definition fails a test instead of failing the first
        // person to run the binary.
        Cli::command().debug_assert();
    }

    #[test]
    fn coordinator_inputs_are_comma_separated() {
        // scripts/run-local.sh passes `--inputs 71,93,42,88` as one argument.
        // Lose the delimiter and every invocation in the README and the demo
        // breaks, with an error about an invalid integer rather than about the
        // flag.
        let cli = Cli::try_parse_from([
            "node",
            "coordinator",
            "--worker",
            "127.0.0.1:8081",
            "--registered-worker",
            "0x0000000000000000000000000000000000000001",
            "--server-key",
            "keys/server.key",
            "--bytecode",
            "build/tally.bytecode",
            "--function",
            "tally4_select",
            "--input",
            "build/input-0.ct",
            "--input",
            "build/input-1.ct",
        ])
        .unwrap();

        let Role::Coordinator {
            inputs,
            attesters,
            deadline_secs,
            bind,
            registry,
            ..
        } = cli.role
        else {
            panic!("parsed as the wrong role");
        };

        // Paths, in the order given. Argument order is the call's argument
        // order, and nothing downstream can notice if it is wrong: the inputs
        // are ciphertext, so a transposition produces a plausible answer to a
        // different question.
        assert_eq!(
            inputs,
            vec![
                PathBuf::from("build/input-0.ct"),
                PathBuf::from("build/input-1.ct"),
            ]
        );
        assert_eq!(registry.len(), 1);
        // The defaults run-local.sh and the README rely on.
        assert_eq!(attesters, 2);
        assert_eq!(deadline_secs, 120);
        assert_eq!(bind, "127.0.0.1:8080");
    }

    #[test]
    fn a_coordinator_with_no_workers_is_refused_by_the_parser() {
        // Also checked in `coordinator::run`, but a job with nobody to dispatch
        // to should not get as far as reading a key off disk.
        // Every other flag is valid, so this fails for the reason under test
        // rather than because an argument was renamed out from under it.
        assert!(
            Cli::try_parse_from([
                "node",
                "coordinator",
                "--registered-worker",
                "0x0000000000000000000000000000000000000001",
                "--server-key",
                "keys/server.key",
                "--bytecode",
                "build/tally.bytecode",
                "--function",
                "f",
                "--input",
                "build/input-0.ct",
            ])
            .is_err()
        );
    }

    #[test]
    fn a_coordinator_can_be_told_which_job_id_the_chain_assigned() {
        // The whole of on-chain settlement turns on this: a worker signs a
        // digest binding the job id, `fulfillJob` rebuilds that digest from the
        // id `submitJob` returned, and if the coordinator minted its own
        // instead, every signature recovers to an address the registry has
        // never seen. A correct result is then rejected as if the workers were
        // impostors, which is the least informative possible failure.
        let cli = Cli::try_parse_from([
            "node",
            "coordinator",
            "--worker",
            "127.0.0.1:8081",
            "--registered-worker",
            "0x0000000000000000000000000000000000000001",
            "--server-key",
            "keys/server.key",
            "--bytecode",
            "build/tally.bytecode",
            "--function",
            "tally4_select",
            "--input",
            "build/input-0.ct",
            "--job-id",
            "42",
        ])
        .unwrap();

        let Role::Coordinator { job_id, .. } = cli.role else {
            panic!("parsed as the wrong role");
        };
        assert_eq!(job_id, Some(42));

        // Absent means "no chain assigned one", not zero — a coordinator that
        // defaulted to 0 would sign under an id `submitJob` never issues.
        let local = Cli::try_parse_from([
            "node",
            "coordinator",
            "--worker",
            "127.0.0.1:8081",
            "--registered-worker",
            "0x0000000000000000000000000000000000000001",
            "--server-key",
            "keys/server.key",
            "--bytecode",
            "build/tally.bytecode",
            "--function",
            "tally4_select",
            "--input",
            "build/input-0.ct",
        ])
        .unwrap();
        let Role::Coordinator { job_id, .. } = local.role else {
            panic!("parsed as the wrong role");
        };
        assert_eq!(job_id, None, "no chain, no id");
    }

    #[test]
    fn a_coordinator_cannot_be_handed_plaintext_inputs() {
        // The flag is gone, not merely discouraged. Plaintext inputs on this
        // process's command line put the secret values in `argv` of the party
        // that fans the job out — readable by anyone who can run `ps` — on a
        // system whose claim is that no node ever sees them. Refusing is the
        // difference between a fixed leak and one that returns the next time
        // somebody finds the old invocation in their shell history.
        assert!(
            Cli::try_parse_from([
                "node",
                "coordinator",
                "--worker",
                "127.0.0.1:8081",
                "--registered-worker",
                "0x0000000000000000000000000000000000000001",
                "--server-key",
                "keys/server.key",
                "--bytecode",
                "build/tally.bytecode",
                "--function",
                "tally4_select",
                "--inputs",
                "71,93,42,88",
            ])
            .is_err(),
            "--inputs must not be accepted in any form"
        );
    }

    #[test]
    fn a_coordinator_with_an_empty_registry_is_refused_by_the_parser() {
        // A coordinator with nobody registered can never settle a job: every
        // report it receives recovers to an address it will reject. Making the
        // flag required means that is a usage error at startup rather than a
        // two-minute deadline followed by "no agreement".
        assert!(
            Cli::try_parse_from([
                "node",
                "coordinator",
                "--worker",
                "127.0.0.1:8081",
                "--program",
                "p.wasm",
                "--function",
                "f",
                "--inputs",
                "1",
            ])
            .is_err()
        );
    }

    #[test]
    fn a_worker_address_can_be_computed_without_starting_a_worker() {
        // scripts/run-local.sh substitutes the output of this into the
        // coordinator's registry, so the two roles have to agree on what a
        // given --id/--key implies.
        let cli = Cli::try_parse_from(["node", "worker-address", "--id", "worker-1"]).unwrap();
        let Role::WorkerAddress { id, key } = cli.role else {
            panic!("parsed as the wrong role");
        };

        assert_eq!(id, "worker-1");
        assert!(key.is_none());

        // The address this role prints is the one the worker will attest
        // under. If the two ever diverged, the local run would fail with every
        // report rejected as "not a registered worker" and nothing pointing at
        // why.
        use primitives::attest::{WorkerKey, hex_address};

        let printed = worker::resolve_key(None, &id).unwrap();
        assert_eq!(
            hex_address(&printed.address()),
            hex_address(&WorkerKey::derive("worker-1").address())
        );
    }

    #[test]
    fn a_worker_takes_its_signing_key_from_the_command_line() {
        let cli = Cli::try_parse_from([
            "node",
            "worker",
            "--id",
            "worker-1",
            "--key",
            "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318",
        ])
        .unwrap();

        let Role::Worker { key, .. } = cli.role else {
            panic!("parsed as the wrong role");
        };
        assert!(key.is_some(), "--key must reach the worker");
    }

    // `--faulty` is a `fault-injection` flag: it does not exist in a default
    // build, so neither can a test that parses it. The default build gets its
    // own assertion below — that the flag is *refused* — which is the property
    // the gate exists to provide.
    #[cfg(feature = "fault-injection")]
    #[test]
    fn a_worker_takes_its_identity_and_its_behaviour_from_the_command_line() {
        let cli = Cli::try_parse_from(["node", "worker", "--id", "worker-3", "--faulty"]).unwrap();

        let Role::Worker {
            id,
            faulty,
            coordinator,
            ..
        } = cli.role
        else {
            panic!("parsed as the wrong role");
        };

        assert_eq!(id, "worker-3");
        assert!(
            faulty,
            "--faulty must reach the worker; run-local.sh needs it"
        );
        assert_eq!(coordinator, "127.0.0.1:8080");
    }

    #[cfg(feature = "fault-injection")]
    #[test]
    fn an_honest_worker_is_the_default() {
        // Faulty must be something you ask for. A worker that returns wrong
        // answers by default would be indistinguishable from a broken build.
        let cli = Cli::try_parse_from(["node", "worker", "--id", "worker-1"]).unwrap();
        let Role::Worker { faulty, .. } = cli.role else {
            panic!("parsed as the wrong role");
        };
        assert!(!faulty);
    }

    #[cfg(not(feature = "fault-injection"))]
    #[test]
    fn a_default_build_has_no_way_to_be_told_to_lie() {
        // The stronger half of the pair above. Without the feature there is no
        // flag, so `--faulty` is an unknown argument and the worker refuses to
        // start rather than starting honest and ignoring what it was asked —
        // which would look identical to a build that had quietly lost the gate.
        assert!(Cli::try_parse_from(["node", "worker", "--id", "worker-1", "--faulty"]).is_err());

        // ...and the role still parses without it, so the refusal is about the
        // flag and not about the role.
        assert!(Cli::try_parse_from(["node", "worker", "--id", "worker-1"]).is_ok());
    }
}
