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

#[cfg(test)]
mod tests {
    use primitives::program::{DiscaProgram, Program};
    use tfhe::prelude::FheDecrypt;
    use tfhe::{ClientKey, ConfigBuilder, generate_keys};

    use super::*;
    use crate::protocol::InputBlob;

    const MAX: &str = r#"
    (module
        (func $max (param i32 i32) (result i32)
          local.get 0
          local.get 1
          local.get 0
          local.get 1
          i32.gt_s
          select
        )
        (export "max" (func $max))
    )
    "#;

    fn bytecode_for(wat: &str) -> Vec<u8> {
        let program = DiscaProgram::from_program(&Program::from_wat(wat).unwrap());
        bytecode::serialize(&program).unwrap()
    }

    /// A dispatch whose inputs are opaque bytes. Enough for every check that
    /// happens before a ciphertext is decoded, which is all of them except
    /// evaluation itself.
    fn dispatch_with(bytecode: Vec<u8>, function: &str, inputs: Vec<InputBlob>) -> JobDispatch {
        JobDispatch {
            job_id: 1,
            attestation_token: [0u8; 32],
            bytecode,
            function: function.into(),
            inputs,
            server_key_hash: [0u8; 32],
        }
    }

    fn committed(bytes: Vec<u8>) -> InputBlob {
        let commitment = wire::commitment(&bytes);
        InputBlob { bytes, commitment }
    }

    #[test]
    fn bytecode_that_is_not_bytecode_is_refused_before_any_evaluation() {
        // A worker validates before it spends CPU (task 2.3). This is the
        // cheapest place the check can fail and the only one where failing is
        // free: past here, a bad circuit costs minutes of homomorphic work
        // before anyone finds out.
        let dispatch = dispatch_with(b"not a disca blob".to_vec(), "max", vec![]);

        let error = evaluate(&dispatch, Behaviour::Honest).unwrap_err();
        assert!(error.contains("DISCA"), "says what it rejected: {error}");
    }

    #[test]
    fn a_dispatch_naming_a_function_the_program_does_not_export_is_refused() {
        let dispatch = dispatch_with(bytecode_for(MAX), "tally4_select", vec![]);

        let error = evaluate(&dispatch, Behaviour::Honest).unwrap_err();
        assert!(
            error.contains("tally4_select"),
            "names the function it could not find: {error}"
        );
    }

    #[test]
    fn an_input_that_does_not_match_its_commitment_is_refused() {
        // Task 2.9e is honest about what this buys today: the commitment
        // travels with the bytes it commits to, so this detects corruption in
        // transit rather than a malicious coordinator. It becomes adversarial
        // once the commitment is read from the chain (2.9f), and the check has
        // to still be here when that happens.
        let mut blob = committed(vec![1, 2, 3, 4]);
        blob.bytes[0] ^= 0xff;

        let dispatch = dispatch_with(bytecode_for(MAX), "max", vec![blob]);

        let error = evaluate(&dispatch, Behaviour::Honest).unwrap_err();
        assert!(
            error.contains("does not match its commitment"),
            "got: {error}"
        );
        assert!(error.contains("input 0"), "names which input: {error}");
    }

    #[test]
    fn an_input_that_matches_its_commitment_but_is_not_a_ciphertext_is_refused() {
        // Passing the commitment check only proves the bytes arrived intact.
        // They still have to be a ciphertext, and a peer can send bytes that
        // are self-consistently committed to and still garbage.
        let dispatch = dispatch_with(bytecode_for(MAX), "max", vec![committed(vec![0u8; 32])]);

        assert!(evaluate(&dispatch, Behaviour::Honest).is_err());
    }

    #[test]
    fn a_faulty_worker_attests_to_a_different_hash_than_an_honest_one() {
        // This is what makes scripts/run-local.sh mean anything. If injection
        // ever stopped changing the result, the local run would show three
        // workers agreeing and be read as a demonstration of M-of-N — when in
        // fact it would demonstrate nothing at all.
        //
        // Real encryption, because the divergence has to survive compression
        // and sealing to reach the coordinator as a different attestation.
        let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
        set_server_key(server_key);

        let inputs = vec![encrypted(17, &client_key), encrypted(42, &client_key)];
        let dispatch = dispatch_with(bytecode_for(MAX), "max", inputs);

        let honest = evaluate(&dispatch, Behaviour::Honest).unwrap();
        let faulty = evaluate(&dispatch, Behaviour::Faulty).unwrap();

        assert_ne!(
            honest.hash, faulty.hash,
            "a faulty worker must not agree with an honest one"
        );

        // Well-formed, not corrupt: the coordinator has to be unable to tell
        // the two apart by inspection, which is the whole point of M-of-N.
        let decrypt = |sealed: &wire::SealedResult| -> i32 {
            wire::decompress(&wire::decode(&sealed.blob).unwrap()).decrypt(&client_key)
        };
        assert_eq!(decrypt(&honest), 42, "max(17, 42)");
        assert_eq!(
            decrypt(&faulty),
            84,
            "the injected fault doubles the result"
        );
    }

    fn encrypted(value: i32, client_key: &ClientKey) -> InputBlob {
        committed(wire::encode(&wire::encrypt_input(value, client_key).unwrap()).unwrap())
    }
}
