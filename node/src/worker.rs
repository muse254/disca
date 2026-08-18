//! The worker role: evaluate dispatched circuits and report what you got.
//!
//! A worker holds the server key and nothing else of value. It sees ciphertexts
//! and the circuit, never plaintext, and it cannot decrypt its own output.
//!
//! Evaluation happens on one dedicated thread rather than per request. The
//! server key is installed in tfhe's thread-local storage and is 114.8 MB
//! decompressed, so cloning it per job would dominate the cost of the job
//! itself. Serial evaluation is also honest about what a worker is: one
//! machine's worth of CPU.

use std::sync::mpsc::{Receiver, channel};
use std::thread;
use std::time::Instant;

use primitives::bytecode;
use primitives::wire;
use tfhe::set_server_key;
use tracing::{error, info, info_span, warn};

use crate::protocol::{JobDispatch, JobOutcome, JobReport};
use crate::transport;

/// How a worker was told to behave. `Faulty` exists so the local end-to-end run
/// can demonstrate that M-of-N attestation actually rejects something — a job
/// where every worker agrees proves nothing about the mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Behaviour {
    Honest,
    Faulty,
}

pub struct Config {
    pub bind: String,
    pub coordinator: String,
    pub id: String,
    pub behaviour: Behaviour,
}

/// Pins evaluation to a single thread.
///
/// **This is a correctness requirement, not a tuning knob.** tfhe-rs
/// multi-threaded evaluation is not bit-reproducible: the same circuit over the
/// same ciphertexts, evaluated with more than one thread, yields results that
/// decrypt identically but differ byte for byte. M-of-N attestation compares
/// hashes of those bytes, so two honest workers would disagree and no job would
/// ever settle.
///
/// Measured: pinning to one thread costs roughly 3x on evaluation
/// (0.65 s to 2.14 s for a compare-and-select circuit on 8 cores). That is the
/// price of a result two workers can independently arrive at. Lifting it needs
/// a verification scheme that does not depend on byte equality — the L1/L2 rungs
/// in `architecture.md` §7.
fn pin_evaluation_to_one_thread() {
    if let Err(error) = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
    {
        // Only fails if something already initialised the pool, which would
        // mean evaluation is about to be non-reproducible.
        warn!(%error, "could not pin evaluation to one thread; results may not be reproducible");
    }
}

/// Runs the worker until the process is killed.
pub fn run(config: Config) -> Result<(), String> {
    pin_evaluation_to_one_thread();

    let server = tiny_http::Server::http(&config.bind)
        .map_err(|e| format!("cannot bind {}: {e}", config.bind))?;

    info!(
        worker = %config.id,
        bind = %config.bind,
        coordinator = %config.coordinator,
        behaviour = ?config.behaviour,
        "worker listening"
    );

    if config.behaviour == Behaviour::Faulty {
        warn!(
            worker = %config.id,
            "running FAULTY: results will be deliberately wrong"
        );
    }

    let (tx, rx) = channel::<JobDispatch>();
    let evaluator = spawn_evaluator(&config, rx);

    for mut request in server.incoming_requests() {
        // Accept and acknowledge immediately. Evaluation takes seconds, and a
        // coordinator waiting on the POST would serialise its own fan-out.
        match transport::read_body(&mut request) {
            Ok(body) => match crate::protocol::decode::<JobDispatch>(&body) {
                Ok(dispatch) => {
                    info!(job_id = dispatch.job_id, worker = %config.id, "job accepted");
                    let _ = tx.send(dispatch);
                    transport::respond(request, 202, b"accepted");
                }
                Err(error) => {
                    warn!(%error, "rejecting malformed dispatch");
                    transport::respond(request, 400, error.as_bytes());
                }
            },
            Err(error) => {
                warn!(%error, "cannot read request body");
                transport::respond(request, 400, error.as_bytes());
            }
        }
    }

    drop(tx);
    let _ = evaluator.join();
    Ok(())
}

fn spawn_evaluator(config: &Config, rx: Receiver<JobDispatch>) -> thread::JoinHandle<()> {
    let id = config.id.clone();
    let coordinator = config.coordinator.clone();
    let behaviour = config.behaviour;

    thread::spawn(move || {
        // The key is fetched on first need and kept for the process lifetime;
        // it is installed in this thread's tfhe storage, which is why every job
        // runs here.
        let mut installed: Option<[u8; 32]> = None;

        for dispatch in rx {
            let span = info_span!("job", job_id = dispatch.job_id, worker = %id);
            let _enter = span.enter();

            if installed != Some(dispatch.server_key_hash) {
                match install_server_key(&coordinator, &dispatch.server_key_hash) {
                    Ok(()) => installed = Some(dispatch.server_key_hash),
                    Err(error) => {
                        error!(%error, "cannot obtain server key");
                        report(&coordinator, failure(&dispatch, &id, error, 0));
                        continue;
                    }
                }
            }

            let started = Instant::now();
            match evaluate(&dispatch, behaviour) {
                Ok(sealed) => {
                    let elapsed_ms = started.elapsed().as_millis() as u64;
                    info!(
                        elapsed_ms,
                        result_hash = %bytecode::hex(&sealed.hash),
                        "job evaluated"
                    );
                    report(
                        &coordinator,
                        JobReport {
                            job_id: dispatch.job_id,
                            worker: id.clone(),
                            outcome: JobOutcome::Evaluated(sealed),
                            elapsed_ms,
                        },
                    );
                }
                Err(error) => {
                    // Reporting failure is not optional: silence is
                    // indistinguishable from slowness and stalls the job.
                    warn!(%error, "job failed");
                    let elapsed_ms = started.elapsed().as_millis() as u64;
                    report(&coordinator, failure(&dispatch, &id, error, elapsed_ms));
                }
            }
        }
    })
}

fn failure(dispatch: &JobDispatch, id: &str, reason: String, elapsed_ms: u64) -> JobReport {
    JobReport {
        job_id: dispatch.job_id,
        worker: id.to_string(),
        outcome: JobOutcome::Failed(reason),
        elapsed_ms,
    }
}

/// Pulls the server key by hash and installs it, verifying before trusting.
fn install_server_key(coordinator: &str, hash: &[u8; 32]) -> Result<(), String> {
    let span = info_span!("keys.fetch", hash = %bytecode::hex(hash));
    let _enter = span.enter();
    let started = Instant::now();

    let bytes = transport::get(&format!(
        "http://{coordinator}/keys/{}",
        bytecode::hex(hash)
    ))?;

    // Addressing the key by hash is only worth anything if we check it.
    let actual = wire::commitment(&bytes);
    if actual != *hash {
        return Err(format!(
            "server key hash mismatch: asked for {}, got {}",
            bytecode::hex(hash),
            bytecode::hex(&actual)
        ));
    }

    let key = wire::decode_server_key(&bytes).map_err(|e| e.to_string())?;
    set_server_key(key);

    info!(
        bytes = bytes.len(),
        elapsed_ms = started.elapsed().as_millis(),
        "server key installed"
    );
    Ok(())
}

/// Validates and runs one job. Everything that can be checked before spending
/// CPU on homomorphic work is checked first.
fn evaluate(dispatch: &JobDispatch, behaviour: Behaviour) -> Result<wire::SealedResult, String> {
    // Decoding validates the circuit (arity, local addressing, final depth), so
    // a malformed program fails here rather than minutes into evaluation.
    let program = bytecode::deserialize(&dispatch.bytecode).map_err(|e| e.to_string())?;

    let func = program
        .function(&dispatch.function)
        .ok_or_else(|| format!("no exported function named {}", dispatch.function))?;

    let mut inputs = Vec::with_capacity(dispatch.inputs.len());
    for (index, blob) in dispatch.inputs.iter().enumerate() {
        // A worker must not evaluate over bytes the coordinator altered between
        // the chain and here. The commitment is what the contract pinned.
        let actual = wire::commitment(&blob.bytes);
        if actual != blob.commitment {
            return Err(format!(
                "input {index} does not match its commitment: expected {}, got {}",
                bytecode::hex(&blob.commitment),
                bytecode::hex(&actual)
            ));
        }

        let ciphertext = wire::decode(&blob.bytes).map_err(|e| e.to_string())?;
        inputs.push(wire::decompress(&ciphertext));
    }

    let result = func.run(&inputs).map_err(|e| e.to_string())?;

    let result = match behaviour {
        Behaviour::Honest => result,
        // Corrupt the value, not the encoding: the point is to produce a
        // well-formed result that disagrees, which is exactly what a subtly
        // broken or dishonest worker would emit.
        Behaviour::Faulty => &result + &result,
    };

    wire::seal_result(&result).map_err(|e| e.to_string())
}

fn report(coordinator: &str, report: JobReport) {
    let Ok(body) = crate::protocol::encode(&report) else {
        error!("cannot encode own report");
        return;
    };

    if let Err(error) = transport::post(&format!("http://{coordinator}/results"), body) {
        // Nothing to retry against: if the coordinator is gone the job is
        // already lost to its deadline.
        error!(%error, "cannot deliver report");
    }
}
