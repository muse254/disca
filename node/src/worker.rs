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

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::thread;
use std::time::Instant;

use primitives::bytecode;
use primitives::wire;
use tfhe::set_server_key;
use tracing::{error, info, info_span, warn};

use crate::protocol::{JobDispatch, JobOutcome, JobReport};
use crate::transport;

/// How a worker was told to behave.
///
/// Set once at startup from `--faulty` and never changed. It is deliberately
/// local: it is not in `JobDispatch`, not in `JobReport`, and not inferable
/// from anything on the wire, so the coordinator has to catch a faulty worker
/// from its output alone. That is also how a real dishonest operator would
/// work — they would change their own binary, not announce it — so the shape of
/// the test matches the shape of the threat.
///
/// **The fault this models is the less likely one.** A wrong answer from a
/// dishonest worker is the adversarial case, and today it is barely rational:
/// there is no reward for participating and no slashing for lying, so a liar
/// just wastes CPU and gets outvoted.
///
/// The divergence a deployment would actually hit first is *misconfiguration*,
/// and we have already seen it: before the FFT plan was pinned, honest workers
/// disagreed in 6 of 12 runs. To the coordinator that was indistinguishable
/// from this enum's `Faulty` — same well-formed report, same wrong hash, same
/// warning — but nobody was malicious. The realistic causes are all boring: a
/// worker on ARM among x86 machines, a different tfhe version, a `gpu` build
/// selecting non-deterministic parameters, or an older binary that never pinned
/// the plan. None is checked at registration yet (task 2.10b), which is why
/// `bridge.md` §2 treats disagreement as evidence of divergence rather than
/// dishonesty, and why slashing must wait for those checks — otherwise it would
/// punish the operator with the wrong CPU instead of the liar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Behaviour {
    Honest,
    /// Only exists when the `fault-injection` feature is on, so a default
    /// release build cannot be told to return a wrong answer.
    #[cfg(feature = "fault-injection")]
    Faulty,
}

pub struct Config {
    pub bind: String,
    pub coordinator: String,
    pub id: String,
    pub behaviour: Behaviour,
}

/// Runs the worker until the process is killed.
///
/// Reproducible results depend on the FFT plan having been pinned before this
/// process touched a server key — see `pin_fft_plan` in `main.rs`. Without it,
/// two honest workers disagree at random and no job settles.
pub fn run(config: Config) -> Result<(), String> {
    let server = tiny_http::Server::http(&config.bind)
        .map_err(|e| format!("cannot bind {}: {e}", config.bind))?;

    info!(
        worker = %config.id,
        bind = %config.bind,
        coordinator = %config.coordinator,
        behaviour = ?config.behaviour,
        "worker listening"
    );

    #[cfg(feature = "fault-injection")]
    if config.behaviour == Behaviour::Faulty {
        warn!(
            worker = %config.id,
            "running FAULTY: results will be deliberately wrong"
        );
    }

    // Dispatches arrive faster than they can be evaluated, and any peer that
    // can reach this port can send them. Without a ceiling the queue is an
    // unbounded allocation an attacker controls.
    let queued = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = channel::<JobDispatch>();
    let evaluator = spawn_evaluator(&config, rx, queued.clone());

    for mut request in server.incoming_requests() {
        // Accept and acknowledge immediately. Evaluation takes seconds, and a
        // coordinator waiting on the POST would serialise its own fan-out.
        match transport::read_body(&mut request) {
            Ok(body) => match crate::protocol::decode::<JobDispatch>(&body) {
                Ok(dispatch) => {
                    if queued.load(Ordering::Acquire) >= MAX_QUEUED_JOBS {
                        warn!(job_id = dispatch.job_id, "queue full, refusing job");
                        transport::respond(request, 503, b"queue full");
                        continue;
                    }

                    info!(job_id = dispatch.job_id, worker = %config.id, "job accepted");
                    queued.fetch_add(1, Ordering::AcqRel);
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

/// How many dispatches may wait to be evaluated. Each holds its inputs and
/// bytecode in memory, and evaluation is seconds per job, so a deep queue is
/// latency nobody wants and memory nobody bounded.
const MAX_QUEUED_JOBS: usize = 16;

fn spawn_evaluator(
    config: &Config,
    rx: Receiver<JobDispatch>,
    queued: Arc<AtomicUsize>,
) -> thread::JoinHandle<()> {
    let id = config.id.clone();
    let coordinator = config.coordinator.clone();
    let behaviour = config.behaviour;

    thread::spawn(move || {
        // The key is fetched on first need and kept for the process lifetime;
        // it is installed in this thread's tfhe storage, which is why every job
        // runs here.
        let mut installed: Option<[u8; 32]> = None;

        for dispatch in rx {
            queued.fetch_sub(1, Ordering::AcqRel);
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
                            attestation_token: dispatch.attestation_token,
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
        attestation_token: dispatch.attestation_token,
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

    // Confirms the coordinator served the key it advertised in the dispatch,
    // and catches a truncated or corrupted transfer. Both the hash and the key
    // come from the coordinator, so this does not make the coordinator
    // trustworthy — it makes it consistent.
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
        // Detects corruption in transit only. The commitment travels in the
        // same message as the bytes it commits to, so a malicious coordinator
        // simply recomputes it — this cannot police the sender. It becomes a
        // real check once the commitment is read from the chain rather than
        // taken from the dispatch (Track 3).
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
        // Corrupt the value, not the encoding, and do it here rather than
        // earlier: every check above has already passed and the real evaluation
        // has already run, so what leaves this function is a perfectly
        // well-formed result that happens to be wrong. That is what a subtly
        // broken or dishonest worker emits, and it is the case attestation has
        // to catch — a malformed report would be rejected by decoding long
        // before any voting happened, proving nothing.
        #[cfg(feature = "fault-injection")]
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
