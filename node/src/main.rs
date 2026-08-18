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
