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
/// Two limits worth knowing. The plan is per polynomial size, so this covers
/// the 2048 used by `ConfigBuilder::default()` and nothing else. And it makes
/// results reproducible across *machines of the same architecture* — Zama
/// document that outputs differ between x86 and ARM, so a mixed-ISA fleet will
/// still disagree.
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
