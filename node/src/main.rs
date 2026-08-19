//! DISCA node.
//!
//! One binary, three roles. `coordinator` and `worker` are the distributed
//! system; `demo` is the original single-process evaluator, kept because it is
//! the fastest way to check the execution core still works without standing up
//! a network.
//!
//! All output is [`tracing`] rather than `println!`: these are the measurements
//! a worker reports to a coordinator, so they are shaped like job telemetry
//! from the start. `RUST_LOG` sets verbosity — `debug` adds per-circuit
//! evaluation spans, `trace` adds per-opcode timings.

mod coordinator;
mod demo;
mod protocol;
mod transport;
mod worker;

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
    /// Prepare a job, fan it out to workers, and settle on an M-of-N result.
    ///
    /// Also stands in for the key holder until job submission moves on-chain.
    Coordinator {
        /// Address to serve the server key and worker reports on.
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,

        /// Worker address. Repeat for each worker.
        #[arg(long = "worker", required = true)]
        workers: Vec<String>,

        /// How many workers must report the same result (the M in M-of-N).
        #[arg(long, default_value_t = 2)]
        attesters: usize,

        /// WASM module to run. Must be built with optimizations — see
        /// architecture.md §2a.
        #[arg(long)]
        program: String,

        /// Exported function within that module.
        #[arg(long)]
        function: String,

        /// Comma-separated plaintext inputs. The key holder encrypts these; no
        /// worker ever sees them.
        #[arg(long, value_delimiter = ',', required = true)]
        inputs: Vec<i32>,

        /// How long to wait for agreement before giving up.
        #[arg(long, default_value_t = 120)]
        deadline_secs: u64,
    },

    /// Evaluate circuits dispatched by a coordinator.
    Worker {
        #[arg(long, default_value = "127.0.0.1:8081")]
        bind: String,

        /// Coordinator address, for pulling the server key and reporting back.
        #[arg(long, default_value = "127.0.0.1:8080")]
        coordinator: String,

        /// Identifies this attester. Stands in for the address it is registered
        /// under on-chain.
        #[arg(long)]
        id: String,

        /// Deliberately return a wrong result. Exists so the local run can show
        /// M-of-N rejecting something — a job where every worker agrees
        /// demonstrates nothing about the mechanism.
        #[arg(long)]
        faulty: bool,
    },

    /// Run the execution core in a single process, no network.
    Demo,
}

fn main() {
    init_telemetry();
    pin_fft_plan();

    let result = match Cli::parse().role {
        Role::Coordinator {
            bind,
            workers,
            attesters,
            program,
            function,
            inputs,
            deadline_secs,
        } => coordinator::run(coordinator::Config {
            bind,
            workers,
            attesters,
            program,
            function,
            inputs,
            deadline: Duration::from_secs(deadline_secs),
        }),

        Role::Worker {
            bind,
            coordinator,
            id,
            faulty,
        } => worker::run(worker::Config {
            bind,
            coordinator,
            id,
            behaviour: if faulty {
                worker::Behaviour::Faulty
            } else {
                worker::Behaviour::Honest
            },
        }),

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
fn init_telemetry() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(Level::INFO.to_string()));

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
            "--program",
            "committee-tally/committee_tally.wasm",
            "--function",
            "tally4_select",
            "--inputs",
            "71,93,42,88",
        ])
        .unwrap();

        let Role::Coordinator {
            inputs,
            attesters,
            deadline_secs,
            bind,
            ..
        } = cli.role
        else {
            panic!("parsed as the wrong role");
        };

        assert_eq!(inputs, vec![71, 93, 42, 88]);
        // The defaults run-local.sh and the README rely on.
        assert_eq!(attesters, 2);
        assert_eq!(deadline_secs, 120);
        assert_eq!(bind, "127.0.0.1:8080");
    }

    #[test]
    fn a_coordinator_with_no_workers_is_refused_by_the_parser() {
        // Also checked in `coordinator::run`, but a job with nobody to dispatch
        // to should not get as far as generating a keypair.
        assert!(
            Cli::try_parse_from([
                "node",
                "coordinator",
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
}
