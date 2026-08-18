//! The coordinator role: hand a job to N workers and settle on what M of them
//! agree it produced.
//!
//! The coordinator sees ciphertexts and bytecode, never plaintext. It is
//! trusted only for liveness — it can stall a job, and the escrow refund path
//! in `bridge.md` §6 covers that, but it cannot forge a result, because the
//! attestation hash it submits has to be one M registered workers independently
//! reported.
//!
//! For now the coordinator also stands in for the **key holder**: it generates
//! the keypair, encrypts the inputs and decrypts the winning result. Those are
//! separate parties in the real design (`architecture.md` §3) and separate
//! processes once job submission moves on-chain in Track 3. Everything the key
//! holder does here is confined to [`KeyHolder`] so the seam is visible.

use std::collections::HashMap;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use primitives::program::{DiscaProgram, Program};
use primitives::wire::{self, SealedResult};
use primitives::{bytecode, validate};
use tfhe::prelude::FheDecrypt;
use tfhe::{ClientKey, CompressedServerKey, ConfigBuilder, generate_keys, set_server_key};
use tracing::{info, info_span, warn};

use crate::protocol::{InputBlob, JobDispatch, JobOutcome, JobReport};
use crate::transport;

pub struct Config {
    pub bind: String,
    pub workers: Vec<String>,
    pub attesters: usize,
    pub program: String,
    pub function: String,
    pub inputs: Vec<i32>,
    pub deadline: Duration,
}

/// Once M workers agree the job could settle immediately, but a straggler that
/// is about to disagree is the most interesting report of the run — it is how a
/// faulty or dishonest worker becomes visible. Wait this long for the remaining
/// workers before settling. Bounded, so one hung worker cannot hold a job open.
const STRAGGLER_GRACE: Duration = Duration::from_secs(5);

/// The party that holds the client key. Kept as its own type because it is the
/// privacy boundary: nothing outside this struct can decrypt anything.
struct KeyHolder {
    client_key: ClientKey,
}

impl KeyHolder {
    fn new() -> (Self, Vec<u8>, [u8; 32]) {
        let started = Instant::now();
        let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());

        // The key holder needs the server key too: decompressing the returned
        // result is a server-key operation, even though decrypting it is not.
        // The server key is public, and this party generated it.
        set_server_key(server_key);

        // Workers receive the compressed key: 28.8 MB rather than 114.8 MB.
        let compressed = CompressedServerKey::new(&client_key);
        let encoded = wire::encode_server_key(&compressed).expect("encode server key");
        let hash = wire::commitment(&encoded);

        info!(
            server_key_bytes = encoded.len(),
            server_key_hash = %bytecode::hex(&hash),
            elapsed_ms = started.elapsed().as_millis(),
            "keys generated"
        );

        (Self { client_key }, encoded, hash)
    }

    fn encrypt(&self, values: &[i32]) -> Result<Vec<InputBlob>, String> {
        values
            .iter()
            .map(|value| {
                let compressed =
                    wire::encrypt_input(*value, &self.client_key).map_err(|e| e.to_string())?;
                let bytes = wire::encode(&compressed).map_err(|e| e.to_string())?;
                let commitment = wire::commitment(&bytes);
                Ok(InputBlob { bytes, commitment })
            })
            .collect()
    }

    fn decrypt(&self, sealed: &SealedResult) -> Result<i32, String> {
        let ciphertext = wire::decode(&sealed.blob).map_err(|e| e.to_string())?;
        Ok(wire::decompress(&ciphertext).decrypt(&self.client_key))
    }
}

/// Reports collected for the job in flight.
type Inbox = Arc<Mutex<Vec<JobReport>>>;

pub fn run(config: Config) -> Result<(), String> {
    if config.attesters == 0 || config.attesters > config.workers.len() {
        return Err(format!(
            "--attesters {} is impossible with {} worker(s)",
            config.attesters,
            config.workers.len()
        ));
    }

    let (key_holder, server_key_bytes, server_key_hash) = KeyHolder::new();
    let bytecode_blob = compile(&config.program, &config.function)?;
    let program_hash = bytecode::hash_bytecode(&bytecode_blob);

    let inbox: Inbox = Arc::new(Mutex::new(Vec::new()));
    let (wake, woken) = channel::<()>();
    serve(
        &config.bind,
        server_key_bytes,
        server_key_hash,
        inbox.clone(),
        wake,
    )?;

    let job_id = 1;
    let span = info_span!("job", job_id, function = %config.function);
    let _enter = span.enter();

    let inputs = key_holder.encrypt(&config.inputs)?;
    info!(
        program_hash = %bytecode::hex(&program_hash),
        inputs = inputs.len(),
        workers = config.workers.len(),
        attesters = config.attesters,
        "job prepared"
    );

    let dispatch = JobDispatch {
        job_id,
        bytecode: bytecode_blob,
        function: config.function.clone(),
        inputs,
        server_key_hash,
    };
    dispatch_to_workers(&config.workers, &dispatch);

    let started = Instant::now();
    let outcome = collect(
        &inbox,
        &woken,
        config.attesters,
        config.workers.len(),
        config.deadline,
    );
    report_outcome(&inbox);

    match outcome {
        Some((sealed, attesters)) => {
            let value = key_holder.decrypt(&sealed)?;
            info!(
                result = value,
                result_hash = %bytecode::hex(&sealed.hash),
                attesters = ?attesters,
                elapsed_ms = started.elapsed().as_millis(),
                "job settled"
            );
            Ok(())
        }
        None => Err(format!(
            "job {job_id} did not reach {}-of-{} agreement within {:?}",
            config.attesters,
            config.workers.len(),
            config.deadline
        )),
    }
}

/// Compiles a WASM module to bytecode, checking up front that the function the
/// job names exists and is runnable.
fn compile(path: &str, function: &str) -> Result<Vec<u8>, String> {
    let wasm = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let parsed = Program::from_wasm(&wasm).map_err(|e| e.to_string())?;
    let program = DiscaProgram::from_program(&parsed);

    let func = program
        .function(function)
        .ok_or_else(|| format!("{path} exports no function named {function}"))?;

    // Fail here rather than after fanning a doomed job out to every worker.
    let layout = validate::validate(func).map_err(|e| e.to_string())?;
    info!(
        ops = func.body.len(),
        peak_stack = layout.max_depth,
        "program compiled"
    );

    bytecode::serialize(&program).map_err(|e| e.to_string())
}

/// Starts the HTTP surface: the server key by hash, and worker reports.
fn serve(
    bind: &str,
    server_key: Vec<u8>,
    server_key_hash: [u8; 32],
    inbox: Inbox,
    wake: Sender<()>,
) -> Result<(), String> {
    let server = tiny_http::Server::http(bind).map_err(|e| format!("cannot bind {bind}: {e}"))?;
    info!(bind = %bind, "coordinator listening");

    let key_path = format!("/keys/{}", bytecode::hex(&server_key_hash));

    thread::spawn(move || {
        for mut request in server.incoming_requests() {
            let url = request.url().to_string();

            if url == key_path {
                info!(bytes = server_key.len(), "serving server key");
                transport::respond(request, 200, &server_key);
                continue;
            }

            if url == "/results" {
                match transport::read_body(&mut request)
                    .and_then(|body| crate::protocol::decode::<JobReport>(&body))
                {
                    Ok(report) => {
                        transport::respond(request, 200, b"ok");
                        inbox.lock().expect("inbox poisoned").push(report);
                        let _ = wake.send(());
                    }
                    Err(error) => {
                        warn!(%error, "rejecting malformed report");
                        transport::respond(request, 400, error.as_bytes());
                    }
                }
                continue;
            }

            transport::respond(request, 404, b"not found");
        }
    });

    Ok(())
}

fn dispatch_to_workers(workers: &[String], dispatch: &JobDispatch) {
    let body = crate::protocol::encode(dispatch).expect("encode dispatch");

    for worker in workers {
        let url = format!("http://{worker}/jobs");
        match transport::post(&url, body.clone()) {
            Ok(()) => info!(worker = %worker, bytes = body.len(), "dispatched"),
            // A worker that cannot be reached simply never reports; the
            // deadline covers it.
            Err(error) => warn!(worker = %worker, %error, "cannot dispatch"),
        }
    }
}

/// Waits for `required` workers to report the same attestation hash.
///
/// After agreement is reached, keeps listening briefly for the workers that
/// have not reported yet: a straggler that disagrees is a faulty worker, and
/// settling the instant M agree would leave that undetected.
fn collect(
    inbox: &Inbox,
    woken: &std::sync::mpsc::Receiver<()>,
    required: usize,
    dispatched: usize,
    deadline: Duration,
) -> Option<(SealedResult, Vec<String>)> {
    let expiry = Instant::now() + deadline;
    let mut settle_by: Option<Instant> = None;

    loop {
        let agreed = tally(inbox, required);

        if agreed.is_some() && settle_by.is_none() {
            settle_by = Some(Instant::now() + STRAGGLER_GRACE);
        }

        // Everyone has spoken; nothing further can change the picture.
        if agreed.is_some() && reported(inbox) >= dispatched {
            return agreed;
        }

        let until = match settle_by {
            Some(grace) => grace.min(expiry),
            None => expiry,
        };

        let Some(remaining) = until.checked_duration_since(Instant::now()) else {
            return agreed.or_else(|| tally(inbox, required));
        };

        if woken.recv_timeout(remaining).is_err() {
            // Grace or deadline elapsed. One last look in case a report landed
            // in the gap.
            return tally(inbox, required);
        }
    }
}

fn reported(inbox: &Inbox) -> usize {
    inbox.lock().expect("inbox poisoned").len()
}

/// Groups reports by attestation hash and returns the first group to reach
/// `required` members.
fn tally(inbox: &Inbox, required: usize) -> Option<(SealedResult, Vec<String>)> {
    group(inbox)
        .into_values()
        .find(|(_, workers)| workers.len() >= required)
}

/// Buckets the reports received so far by attestation hash.
fn group(inbox: &Inbox) -> HashMap<[u8; 32], (SealedResult, Vec<String>)> {
    let reports = inbox.lock().expect("inbox poisoned");

    let mut groups: HashMap<[u8; 32], (SealedResult, Vec<String>)> = HashMap::new();
    for report in reports.iter() {
        if let JobOutcome::Evaluated(sealed) = &report.outcome {
            groups
                .entry(sealed.hash)
                .or_insert_with(|| (sealed.clone(), Vec::new()))
                .1
                .push(report.worker.clone());
        }
    }
    groups
}

/// Says out loud what the reports amounted to, once, when the job settles.
///
/// Evaluation and compression are both deterministic (verified in
/// `primitives::wire`), so honest workers cannot disagree. More than one group
/// means a faulty or dishonest worker, and that is a finding rather than noise
/// to be quietly resolved by taking the majority.
fn report_outcome(inbox: &Inbox) {
    for report in inbox.lock().expect("inbox poisoned").iter() {
        if let JobOutcome::Failed(reason) = &report.outcome {
            warn!(worker = %report.worker, reason = %reason, "worker reported failure");
        }
    }

    let groups = group(inbox);
    if groups.len() > 1 {
        for (hash, (_, workers)) in &groups {
            warn!(
                hash = %bytecode::hex(hash),
                workers = ?workers,
                "attestation disagreement"
            );
        }
    }
}
