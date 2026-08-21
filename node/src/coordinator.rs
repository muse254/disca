//! The coordinator role: hand a job to N workers and settle on what M of them
//! agree it produced.
//!
//! The coordinator sees ciphertexts and bytecode, never plaintext. It is
//! trusted only for liveness *within this process*: attestation tokens bind each
//! report to the dispatch that authorised it, so one worker cannot report M
//! times and the coordinator cannot count a report it never dispatched for.
//!
//! **That property does not extend to a third party, and the on-chain design in
//! `bridge.md` §2 currently assumes it does.** A `SealedResult` is a blob and
//! `keccak256(blob)` — nothing signed. Anyone can compute it for any blob. The
//! tokens are visible only to this process, so a contract handed
//! `fulfillJob(jobId, resultHash, resultBlob, attesters)` can check that the
//! attester addresses are registered and distinct, and nothing more: a
//! dishonest coordinator can name any two registered workers beside any result.
//! No party contradicts it — the workers signed nothing, and the key holder
//! cannot distinguish a wrong plaintext from a right one (`attestation.md` §1).
//!
//! Fixing this needs per-worker signing keys, so that what a worker returns is
//! evidence rather than a claim (task 2.10i). Zama's fhEVM avoids the problem by
//! having each coprocessor transact for itself and counting `msg.sender`; the
//! moment one party votes on behalf of others, unsigned attestations stop
//! meaning anything off-box.
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

/// Reports collected for the job in flight, keyed by the worker each one is
/// attributable to.
///
/// A map rather than a list, because agreement must be counted per worker: two
/// reports from the same worker are one attestation, and a report that cannot
/// be attributed to a dispatched worker is not an attestation at all.
type Inbox = Arc<Mutex<HashMap<String, JobReport>>>;

/// Resolves an attestation token to the worker the coordinator dispatched it
/// to.
///
/// This is what makes a report attributable. The alternative — trusting the
/// `worker` field a reporter puts in its own message — lets one worker report M
/// times under M invented names and settle a job single-handedly, since nothing
/// else distinguishes the reports.
type Tokens = Arc<HashMap<[u8; 32], String>>;

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

    // One unguessable token per worker. Only the worker it is dispatched to
    // ever sees it, so echoing it back is what proves a report answers a
    // dispatch this coordinator actually made.
    let assignments: Vec<([u8; 32], String)> = config
        .workers
        .iter()
        .map(|worker| (mint_token(), worker.clone()))
        .collect();
    let tokens: Tokens = Arc::new(assignments.iter().cloned().collect());

    let inbox: Inbox = Arc::new(Mutex::new(HashMap::new()));
    let (wake, woken) = channel::<()>();
    serve(
        &config.bind,
        server_key_bytes,
        server_key_hash,
        inbox.clone(),
        tokens,
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
        // Replaced per worker below; each gets its own token.
        attestation_token: [0u8; 32],
        bytecode: bytecode_blob,
        function: config.function.clone(),
        inputs,
        server_key_hash,
    };
    dispatch_to_workers(&assignments, &dispatch);

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
    tokens: Tokens,
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
                        // Attribute the report to the worker this coordinator
                        // dispatched that token to. Anything else is not an
                        // attestation, however well-formed it looks.
                        let Some(worker) = tokens.get(&report.attestation_token) else {
                            warn!(
                                claimed = %report.worker,
                                "discarding report with an unrecognised attestation token"
                            );
                            transport::respond(request, 403, b"unrecognised attestation token");
                            continue;
                        };

                        transport::respond(request, 200, b"ok");

                        // One dispatch, one attestation. A repeat overwrites
                        // rather than accumulating, so a worker cannot inflate
                        // its own weight by reporting twice.
                        let mut inbox = inbox.lock().expect("inbox poisoned");
                        if inbox.insert(worker.clone(), report).is_some() {
                            warn!(worker = %worker, "worker reported more than once");
                        }
                        drop(inbox);

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

/// Sends each worker its own copy, carrying its own attestation token.
fn dispatch_to_workers(assignments: &[([u8; 32], String)], dispatch: &JobDispatch) {
    for (token, worker) in assignments {
        let mut addressed = dispatch.clone();
        addressed.attestation_token = *token;

        let body = crate::protocol::encode(&addressed).expect("encode dispatch");
        let url = format!("http://{worker}/jobs");
        match transport::post(&url, body.clone()) {
            Ok(()) => info!(worker = %worker, bytes = body.len(), "dispatched"),
            // A worker that cannot be reached simply never reports; the
            // deadline covers it.
            Err(error) => warn!(worker = %worker, %error, "cannot dispatch"),
        }
    }
}

/// Mints an unguessable attestation token.
fn mint_token() -> [u8; 32] {
    let mut token = [0u8; 32];
    getrandom::fill(&mut token).expect("system randomness");
    token
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

        // No outstanding worker could still form a quorum, so waiting out the
        // deadline would only delay a failure we can already call.
        if agreed.is_none() && !agreement_still_possible(inbox, required, dispatched) {
            return None;
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

/// Whether any worker yet to report could still bring a group up to `required`.
///
/// Once the answer is no, waiting out the deadline tells us nothing.
fn agreement_still_possible(inbox: &Inbox, required: usize, dispatched: usize) -> bool {
    let best = group(inbox)
        .values()
        .map(|(_, workers)| workers.len())
        .max()
        .unwrap_or(0);
    let outstanding = dispatched.saturating_sub(reported(inbox));
    best + outstanding >= required
}

/// Groups reports by attestation hash and returns the first group to reach
/// `required` members.
fn tally(inbox: &Inbox, required: usize) -> Option<(SealedResult, Vec<String>)> {
    let mut quorums: Vec<(SealedResult, Vec<String>)> = group(inbox)
        .into_values()
        .filter(|(_, workers)| workers.len() >= required)
        .collect();

    // With `required <= dispatched / 2`, two groups can both reach quorum. That
    // means the fault threshold has been exceeded and neither answer is
    // trustworthy, so refusing is the honest outcome — picking one would be a
    // coin flip decided by hash iteration order.
    if quorums.len() > 1 {
        warn!(
            quorums = quorums.len(),
            "more than one group reached quorum; refusing to choose between them"
        );
        return None;
    }

    quorums.pop()
}

/// Buckets the reports received so far by attestation hash.
fn group(inbox: &Inbox) -> HashMap<[u8; 32], (SealedResult, Vec<String>)> {
    let reports = inbox.lock().expect("inbox poisoned");

    let mut groups: HashMap<[u8; 32], (SealedResult, Vec<String>)> = HashMap::new();
    for (worker, report) in reports.iter() {
        if let JobOutcome::Evaluated(sealed) = &report.outcome {
            // Keyed by the worker the coordinator dispatched to, not by the
            // name in the message: the map holds at most one report per worker,
            // so each contributes at most one attestation.
            groups
                .entry(sealed.hash)
                .or_insert_with(|| (sealed.clone(), Vec::new()))
                .1
                .push(worker.clone());
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
    for (worker, report) in inbox.lock().expect("inbox poisoned").iter() {
        if let JobOutcome::Failed(reason) = &report.outcome {
            warn!(worker = %worker, reason = %reason, "worker reported failure");
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Stands in for a worker's sealed result. Only the hash matters to
    /// aggregation — the blob is opaque to the coordinator, which is the point:
    /// it settles on bytes it cannot read.
    fn sealed(marker: u8) -> SealedResult {
        SealedResult {
            blob: vec![marker; 16],
            hash: [marker; 32],
        }
    }

    fn evaluated(worker: &str, marker: u8) -> (String, JobReport) {
        report(worker, JobOutcome::Evaluated(sealed(marker)))
    }

    fn failed(worker: &str, reason: &str) -> (String, JobReport) {
        report(worker, JobOutcome::Failed(reason.into()))
    }

    fn report(worker: &str, outcome: JobOutcome) -> (String, JobReport) {
        (
            worker.to_string(),
            JobReport {
                job_id: 1,
                attestation_token: [0u8; 32],
                worker: worker.to_string(),
                outcome,
                elapsed_ms: 1,
            },
        )
    }

    fn inbox(reports: Vec<(String, JobReport)>) -> Inbox {
        Arc::new(Mutex::new(reports.into_iter().collect()))
    }

    #[test]
    fn a_majority_that_agrees_settles_and_names_its_attesters() {
        let inbox = inbox(vec![
            evaluated("w1", 0xaa),
            evaluated("w2", 0xaa),
            evaluated("w3", 0xbb),
        ]);

        let (result, mut attesters) = tally(&inbox, 2).expect("two workers agreed");
        assert_eq!(result.hash, [0xaa; 32]);

        // The attester set is what `fulfillJob` will take on-chain, so it must
        // be the workers that actually reported that hash and no one else.
        attesters.sort();
        assert_eq!(attesters, vec!["w1".to_string(), "w2".to_string()]);
    }

    #[test]
    fn two_groups_at_quorum_refuse_to_settle() {
        // Reachable whenever M <= N/2 (task 2.9c). Both groups are "valid" by
        // the counting rule, which means the fault threshold has been exceeded
        // and neither answer is trustworthy. Returning either one would be a
        // coin flip decided by HashMap iteration order — and it would look, to
        // everyone downstream, exactly like a settled job.
        let inbox = inbox(vec![
            evaluated("w1", 0xaa),
            evaluated("w2", 0xaa),
            evaluated("w3", 0xbb),
            evaluated("w4", 0xbb),
        ]);

        assert!(
            tally(&inbox, 2).is_none(),
            "a split quorum must be refused, not resolved by iteration order"
        );
    }

    #[test]
    fn a_failure_report_is_not_an_attestation() {
        // A worker that says "I could not evaluate" has attested to nothing.
        // Counting failures towards a quorum would let two broken workers
        // settle a job between them.
        let inbox = inbox(vec![
            evaluated("w1", 0xaa),
            failed("w2", "stack underflow at op 3"),
            failed("w3", "stack underflow at op 3"),
        ]);

        assert!(tally(&inbox, 2).is_none());
        assert_eq!(
            group(&inbox).len(),
            1,
            "only the evaluated report forms a group"
        );
    }

    #[test]
    fn agreement_stops_being_possible_once_everyone_has_disagreed() {
        // Three workers, three different answers, two required. Nobody is left
        // to break the tie, so waiting out the deadline tells us nothing —
        // this is what turns a 120 s timeout into an immediate failure.
        let all_disagreed = inbox(vec![
            evaluated("w1", 0xaa),
            evaluated("w2", 0xbb),
            evaluated("w3", 0xcc),
        ]);
        assert!(!agreement_still_possible(&all_disagreed, 2, 3));

        // Same shape one report earlier: the outstanding worker could still
        // join either group, so the job must stay open.
        let one_outstanding = inbox(vec![evaluated("w1", 0xaa), evaluated("w2", 0xbb)]);
        assert!(agreement_still_possible(&one_outstanding, 2, 3));
    }

    #[test]
    fn a_hopeless_job_fails_immediately_rather_than_waiting_out_the_deadline() {
        let inbox = inbox(vec![
            evaluated("w1", 0xaa),
            evaluated("w2", 0xbb),
            evaluated("w3", 0xcc),
        ]);
        let (_wake, woken) = channel::<()>();

        let started = Instant::now();
        let outcome = collect(&inbox, &woken, 2, 3, Duration::from_secs(120));

        assert!(outcome.is_none());
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "collect waited {:?} for a quorum that can no longer form",
            started.elapsed()
        );
    }

    #[test]
    fn a_settled_job_does_not_sit_out_the_straggler_grace_once_everyone_has_reported() {
        // The grace period exists to catch a worker that has not spoken yet.
        // When all of them have, holding the job open for another five seconds
        // is latency bought for nothing.
        let inbox = inbox(vec![
            evaluated("w1", 0xaa),
            evaluated("w2", 0xaa),
            evaluated("w3", 0xbb),
        ]);
        let (_wake, woken) = channel::<()>();

        let started = Instant::now();
        let (result, _) =
            collect(&inbox, &woken, 2, 3, Duration::from_secs(120)).expect("two workers agreed");

        assert_eq!(result.hash, [0xaa; 32]);
        assert!(
            started.elapsed() < STRAGGLER_GRACE,
            "collect waited {:?} after every worker had reported",
            started.elapsed()
        );
    }

    #[test]
    fn every_token_a_coordinator_mints_is_distinct() {
        // Tokens are what bind a report to a dispatch (task 2.9a). Two workers
        // sharing one would be able to answer for each other, which is the
        // failure the token exists to prevent.
        let tokens: std::collections::HashSet<[u8; 32]> = (0..64).map(|_| mint_token()).collect();
        assert_eq!(tokens.len(), 64);
        assert!(
            !tokens.contains(&[0u8; 32]),
            "an all-zero token is not random"
        );
    }

    const TALLY_WASM: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../committee-tally/committee_tally.wasm"
    );

    #[test]
    fn a_compiled_program_is_something_a_worker_can_decode_and_run() {
        // The coordinator compiles once and every worker decodes the same
        // blob. If what `compile` emits were not accepted by
        // `bytecode::deserialize` -- which validates as well as decodes -- the
        // job would fan out to every worker and fail on all of them at once.
        let blob = compile(TALLY_WASM, "tally4_select").unwrap();

        let program = bytecode::deserialize(&blob).expect("a worker must accept this");
        let func = program
            .function("tally4_select")
            .expect("the function the job named");
        assert_eq!(func.sig.params.len(), 4);
        assert_eq!(func.sig.results.len(), 1);
    }

    #[test]
    fn compiling_refuses_a_function_the_program_does_not_export() {
        // Cheap here, expensive later: this runs before keygen and before any
        // dispatch, so a typo costs a millisecond instead of fanning a doomed
        // job out to three workers and waiting for the deadline.
        let error = compile(TALLY_WASM, "tally5_select").unwrap_err();
        assert!(error.contains("tally5_select"), "names it: {error}");
    }

    #[test]
    fn compiling_refuses_a_program_that_is_not_there() {
        let error = compile("committee-tally/does-not-exist.wasm", "max2").unwrap_err();
        assert!(error.contains("does-not-exist.wasm"), "names it: {error}");
    }
}
