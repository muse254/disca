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
//!
//! A worker also holds a secp256k1 key and signs every report with it (task
//! 2.10i). That key is the worker's identity — the Ethereum address it would be
//! registered under on-chain — and it is the only thing that distinguishes this
//! worker's attestation from anyone else's. It is never logged, never sent, and
//! never derived from anything the coordinator supplies.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::thread;
use std::time::Instant;

use primitives::attest::{self, Claim, WorkerKey};
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
    /// The key this worker attests with. See [`resolve_key`].
    pub key: WorkerKey,
    pub behaviour: Behaviour,
}

/// Picks the signing key a worker will run with: the one it was given, or a
/// deterministic development key derived from its id.
///
/// The fallback exists so `scripts/run-local.sh` can start three workers and a
/// coordinator that already knows their addresses without a key-distribution
/// step in a shell script. It is not a secret — the id is in every log line, so
/// anyone can recompute it — and a worker using it says so at startup. A
/// deployment passes `--key`.
///
/// Note what this function does *not* do: it never falls back silently on a bad
/// `--key`. An operator who meant to supply a key and typo'd it must not end up
/// attesting under a publicly-derivable address that happens to be registered
/// on someone's testnet.
pub fn resolve_key(key: Option<&str>, id: &str) -> Result<WorkerKey, String> {
    match key {
        Some(hex) => WorkerKey::from_hex(hex).map_err(|e| e.to_string()),
        None => {
            let key = WorkerKey::derive(id);
            warn!(
                worker = %id,
                address = %attest::hex_address(&key.address()),
                "no --key given; attesting under a key derived from the worker id, \
                 which anyone can recompute. Fine locally, never in a deployment"
            );
            Ok(key)
        }
    }
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
        // The address, never the key. This is the value an operator registers
        // on-chain and the value the coordinator's registry has to contain, so
        // it is worth one line at startup: a worker whose attestations are all
        // being rejected is otherwise indistinguishable from one nobody is
        // dispatching to.
        address = %attest::hex_address(&config.key.address()),
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

    // The signing key moves to the evaluator thread and stays there: it is used
    // at exactly one point, sealing a report, and the fewer places hold it the
    // fewer there are to leak it. The accept loop keeps only the id it logs.
    let id = config.id.clone();
    let evaluator = spawn_evaluator(config, rx, queued.clone());

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

                    info!(job_id = dispatch.job_id, worker = %id, "job accepted");
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
    config: Config,
    rx: Receiver<JobDispatch>,
    queued: Arc<AtomicUsize>,
) -> thread::JoinHandle<()> {
    let Config {
        id,
        coordinator,
        key,
        behaviour,
        ..
    } = config;

    thread::spawn(move || {
        // The key is fetched on first need and kept for the process lifetime;
        // it is installed in this thread's tfhe storage, which is why every job
        // runs here.
        let mut installed: Option<[u8; 32]> = None;

        for dispatch in rx {
            queued.fetch_sub(1, Ordering::AcqRel);
            let span = info_span!("job", job_id = dispatch.job_id, worker = %id);
            let _enter = span.enter();

            // Both sides derive the program's identity from the bytecode they
            // hold rather than being told it, so a coordinator cannot get a
            // worker to sign an attestation naming a program it did not run.
            let bytecode_hash = bytecode::hash_bytecode(&dispatch.bytecode);

            if installed != Some(dispatch.server_key_hash) {
                match install_server_key(&coordinator, &dispatch.server_key_hash) {
                    Ok(()) => installed = Some(dispatch.server_key_hash),
                    Err(error) => {
                        error!(%error, "cannot obtain server key");
                        report(
                            &coordinator,
                            failure(&dispatch, bytecode_hash, &key, &id, error, 0),
                        );
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

                    // Signed *after* sealing and over the sealed hash, so what
                    // is attested to is exactly the bytes that go on-chain
                    // (`bridge.md` §5a). Signing anything earlier would commit
                    // to a value the contract cannot recompute.
                    let attestation = key.attest(&Claim::Result {
                        job_id: dispatch.job_id,
                        bytecode_hash,
                        result_hash: sealed.hash,
                    });

                    report(
                        &coordinator,
                        JobReport {
                            job_id: dispatch.job_id,
                            attestation,
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
                    report(
                        &coordinator,
                        failure(&dispatch, bytecode_hash, &key, &id, error, elapsed_ms),
                    );
                }
            }
        }
    })
}

/// Builds a signed "I could not run this" report.
///
/// A failure attests to nothing and is never counted towards a quorum, but it
/// is signed under its own domain tag all the same: an unsigned failure is
/// something anyone who can reach the coordinator could forge in an honest
/// worker's name, and the moment registration or reputation depends on who
/// failed, that is a way to discredit a working operator for free.
fn failure(
    dispatch: &JobDispatch,
    bytecode_hash: [u8; 32],
    key: &WorkerKey,
    id: &str,
    reason: String,
    elapsed_ms: u64,
) -> JobReport {
    let attestation = key.attest(&Claim::Failure {
        job_id: dispatch.job_id,
        bytecode_hash,
        // The reason is unbounded operator-controlled text; its hash is what
        // goes in the fixed-width preimage, and the text travels beside it so
        // the coordinator can still recompute the hash and check the two agree.
        reason_hash: wire::commitment(reason.as_bytes()),
    });

    JobReport {
        job_id: dispatch.job_id,
        attestation,
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

#[cfg(test)]
mod tests {
    use primitives::program::{DiscaProgram, Program};
    // Only the fault-injection test encrypts and decrypts for real; a default
    // build would carry these as unused imports.
    #[cfg(feature = "fault-injection")]
    use tfhe::prelude::FheDecrypt;
    #[cfg(feature = "fault-injection")]
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

    // `Behaviour::Faulty` only exists behind `fault-injection`, so this test
    // only exists there too. Nothing is lost in a default build: the behaviour
    // under test is not compiled into it.
    #[cfg(feature = "fault-injection")]
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

    #[cfg(feature = "fault-injection")]
    fn encrypted(value: i32, client_key: &ClientKey) -> InputBlob {
        committed(wire::encode(&wire::encrypt_input(value, client_key).unwrap()).unwrap())
    }

    #[test]
    fn a_worker_without_a_key_falls_back_to_one_derived_from_its_id() {
        // What scripts/run-local.sh relies on: the coordinator derives the
        // addresses it will accept from the same ids, in another process.
        let key = resolve_key(None, "worker-1").unwrap();
        assert_eq!(
            key.address(),
            attest::WorkerKey::derive("worker-1").address()
        );
    }

    #[test]
    fn a_supplied_key_wins_over_the_derived_one() {
        let hex = "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318";
        let supplied = resolve_key(Some(hex), "worker-1").unwrap();

        assert_eq!(
            supplied.address(),
            WorkerKey::from_hex(hex).unwrap().address()
        );
        assert_ne!(
            supplied.address(),
            WorkerKey::derive("worker-1").address(),
            "--key must not be silently ignored"
        );
    }

    #[test]
    fn a_malformed_key_stops_the_worker_rather_than_falling_back() {
        // The dangerous version of this bug is the quiet one: an operator
        // typos --key, the worker starts anyway under a publicly-derivable
        // address, and every attestation it makes is one anybody could have
        // made. Failing to start is the only safe answer.
        let error = resolve_key(Some("0xnot-a-key"), "worker-1").unwrap_err();
        assert!(error.contains("32 hex-encoded bytes"), "got: {error}");
    }

    #[test]
    fn a_failure_report_is_signed_under_the_failure_tag_and_names_its_reason() {
        // Two things at once: the report is attributable to this worker, and
        // the reason text the coordinator receives is the one whose hash was
        // signed — otherwise a relay could rewrite the reason and leave the
        // signature valid.
        let key = WorkerKey::derive("worker-1");
        let dispatch = dispatch_with(bytecode_for(MAX), "max", vec![]);
        let bytecode_hash = bytecode::hash_bytecode(&dispatch.bytecode);

        let report = failure(
            &dispatch,
            bytecode_hash,
            &key,
            "worker-1",
            "stack underflow at op 3".into(),
            7,
        );

        let JobOutcome::Failed(reason) = &report.outcome else {
            panic!("expected a failure report");
        };

        let claim = Claim::Failure {
            job_id: dispatch.job_id,
            bytecode_hash,
            reason_hash: wire::commitment(reason.as_bytes()),
        };
        assert_eq!(
            attest::recover(&claim, &report.attestation).unwrap(),
            key.address()
        );

        // ...and it is not a result attestation, however the coordinator
        // reconstructs the claim.
        let as_result = Claim::Result {
            job_id: dispatch.job_id,
            bytecode_hash,
            result_hash: wire::commitment(reason.as_bytes()),
        };
        assert_ne!(
            attest::recover(&as_result, &report.attestation).unwrap_or_default(),
            key.address()
        );
    }
}
