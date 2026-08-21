//! The coordinator role: hand a job to N workers and settle on what M of them
//! agree it produced.
//!
//! The coordinator sees ciphertexts and bytecode, never plaintext. It is
//! trusted only for liveness — it can stall a job, and the escrow refund path
//! in `bridge.md` §6 covers that. It cannot forge agreement: since task 2.10i
//! every report carries a secp256k1 signature over a digest binding the job,
//! the program and the result, so the coordinator *recovers* each attester's
//! address rather than being told it, and counts agreement over distinct
//! addresses that appear in a configured registry.
//!
//! That is deliberately the same operation `fulfillJob` performs on-chain, and
//! the property is only verifiable by a third party once it does. A contract
//! handed a bare `address[]` would be taking the coordinator's word for who
//! attested, which is what made the earlier design unsound (`bridge.md` §2a,
//! §2b). Until that contract exists, the check here is enforced by this
//! process's own code: the signatures are evidence anyone *could* verify, not
//! evidence anyone has.
//!
//! Zama's fhEVM sidesteps the question entirely by having each coprocessor
//! transact for itself and counting `msg.sender`. The moment one party votes on
//! behalf of others, unsigned attestations stop meaning anything off-box.
//!
//! For now the coordinator also stands in for the **key holder**: it generates
//! the keypair, encrypts the inputs and decrypts the winning result. Those are
//! separate parties in the real design (`architecture.md` §3) and separate
//! processes once job submission moves on-chain in Track 3. Everything the key
//! holder does here is confined to [`KeyHolder`] so the seam is visible.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use primitives::attest::{self, Address, Claim};
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
    /// Addresses whose attestations count. Stands in for the on-chain worker
    /// registry `bridge.md` §2 describes — `registerWorker(address)` — and is
    /// the reason a recovered address is worth anything: recovery alone tells
    /// you *who* signed, not whether that party is anyone this network has
    /// agreed to accept work from. Without it, anybody with a keyboard can
    /// generate M keypairs and out-vote the honest workers for free.
    pub registry: Vec<Address>,
    pub attesters: usize,
    pub program: String,
    pub function: String,
    pub inputs: Vec<i32>,
    pub deadline: Duration,
}

/// Turns `--registered-worker` arguments into the registry, refusing anything
/// that is not an address and anything named twice.
///
/// A duplicate is rejected rather than deduplicated: it is silently halving the
/// number of independent parties the operator thinks they configured, and
/// `--attesters 2` against a registry that is one address written twice would
/// be a quorum of one. Better to make them look at it.
pub fn parse_registry(entries: &[String]) -> Result<Vec<Address>, String> {
    let mut registry = Vec::with_capacity(entries.len());
    for entry in entries {
        let address = attest::parse_address(entry).map_err(|e| e.to_string())?;
        if registry.contains(&address) {
            return Err(format!(
                "{} is registered more than once; each attester must be a \
                 distinct party",
                attest::hex_address(&address)
            ));
        }
        registry.push(address);
    }
    Ok(registry)
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

/// Reports collected for the job in flight, keyed by the address that signed
/// each one.
///
/// A map rather than a list, because agreement must be counted per attester:
/// two reports from the same address are one attestation, and a report whose
/// signature does not recover to a registered address is not an attestation at
/// all. The key is the *recovered* address, so the map's own shape enforces
/// one-attestation-per-party — there is no field a reporter can set to occupy a
/// second slot.
type Inbox = Arc<Mutex<HashMap<Address, JobReport>>>;

/// Everything needed to decide whether an arriving report is an attestation:
/// the job and program it must be about, and who is allowed to speak.
///
/// Held by the HTTP thread and never mutated. The two hashes are what make the
/// check adversarial rather than cosmetic — they come from what *this*
/// coordinator dispatched, so a signature for any other job or any other
/// program recovers to some address that is not the sender's and is rejected by
/// the registry a line later.
struct Verifier {
    job_id: u64,
    bytecode_hash: [u8; 32],
    registry: HashSet<Address>,
}

impl Verifier {
    /// Decides who, if anyone, this report is from.
    ///
    /// Kept separate from the HTTP plumbing because this is the whole security
    /// decision and it should be testable without a socket.
    fn attribute(&self, report: &JobReport) -> Result<Address, String> {
        if report.job_id != self.job_id {
            return Err(format!(
                "report is for job {}, not job {}",
                report.job_id, self.job_id
            ));
        }

        // The claim is reconstructed from what the coordinator knows plus the
        // outcome in the report. Nothing here is taken from a field the sender
        // could set to steer the recovery — job id and bytecode hash are the
        // coordinator's, and the result hash is the one being voted on.
        let claim = match &report.outcome {
            JobOutcome::Evaluated(sealed) => {
                // A sealed result whose hash does not cover its blob is not a
                // result; catching it here means the hash that is about to be
                // counted, and later compared against `keccak256(resultBlob)`
                // on-chain (`bridge.md` §5a), is one the contract will agree
                // with.
                let actual = wire::commitment(&sealed.blob);
                if actual != sealed.hash {
                    return Err(format!(
                        "sealed result does not match its own hash: claims {}, is {}",
                        bytecode::hex(&sealed.hash),
                        bytecode::hex(&actual)
                    ));
                }
                Claim::Result {
                    job_id: self.job_id,
                    bytecode_hash: self.bytecode_hash,
                    result_hash: sealed.hash,
                }
            }
            JobOutcome::Failed(reason) => Claim::Failure {
                job_id: self.job_id,
                bytecode_hash: self.bytecode_hash,
                reason_hash: wire::commitment(reason.as_bytes()),
            },
        };

        let address = attest::recover(&claim, &report.attestation).map_err(|e| e.to_string())?;

        // Recovery says who signed; the registry says whether that party counts.
        // Both are needed: without the signature the address is a claim, and
        // without the registry the address is a stranger's.
        if !self.registry.contains(&address) {
            return Err(format!(
                "{} is not a registered worker",
                attest::hex_address(&address)
            ));
        }

        Ok(address)
    }
}

pub fn run(config: Config) -> Result<(), String> {
    // A quorum has to be a majority of the workers dispatched to, and refusing
    // anything less is not caution — a minority quorum can settle on a wrong
    // answer with nothing left to contradict it.
    //
    // The TLA+ model exhibits it (`spec/`, `MC_GraceRace_N4M2`): with N = 4 and
    // M = 2, the two faulty workers report, `STRAGGLER_GRACE` expires while the
    // honest two are still evaluating — FHE is seconds of work and the grace is
    // five — and `tally` sees exactly one group at quorum and settles on it.
    // The split refusal never fires, because at that moment there is no split
    // to see. A longer grace does not help: a faulty worker need not evaluate
    // at all, so it can always report first.
    //
    // `2M > N` makes two disjoint quorums impossible, so the honest workers can
    // always outvote a faulty group unless they are outnumbered — which is the
    // fault threshold being exceeded, not a race being lost.
    if let Some(error) = quorum_error(config.attesters, config.workers.len()) {
        return Err(error);
    }

    // A quorum can only ever be formed by registered addresses, so a registry
    // smaller than M is a job that cannot settle. Say so before generating a
    // keypair and fanning out, rather than after the deadline.
    if config.registry.len() < config.attesters {
        return Err(format!(
            "--attesters {} is impossible with {} registered worker(s)",
            config.attesters,
            config.registry.len()
        ));
    }

    // Fixed at 1 while there is no chain to take it from. It is bound into
    // every signature, so it is also the reason an attestation cannot be lifted
    // onto another job — which is a real guarantee only once job ids are
    // globally unique. Two runs of this binary over the same program and inputs
    // both use job 1 today, and their attestations are interchangeable. That
    // becomes sound, not merely conventional, when `submitJob` assigns the id
    // (`bridge.md` §2, task 2.9f).
    let job_id = fresh_job_id();

    let (key_holder, server_key_bytes, server_key_hash) = KeyHolder::new();
    let bytecode_blob = compile(&config.program, &config.function)?;
    let program_hash = bytecode::hash_bytecode(&bytecode_blob);

    let verifier = Arc::new(Verifier {
        job_id,
        bytecode_hash: program_hash,
        registry: config.registry.iter().copied().collect(),
    });

    let inbox: Inbox = Arc::new(Mutex::new(HashMap::new()));
    let (wake, woken) = channel::<()>();
    serve(
        &config.bind,
        server_key_bytes,
        server_key_hash,
        inbox.clone(),
        verifier,
        wake,
    )?;

    let span = info_span!("job", job_id, function = %config.function);
    let _enter = span.enter();

    let inputs = key_holder.encrypt(&config.inputs)?;
    info!(
        program_hash = %bytecode::hex(&program_hash),
        inputs = inputs.len(),
        workers = config.workers.len(),
        registered = config.registry.len(),
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
                // Recovered, not asserted. `fulfillJob` will take the
                // *signatures* rather than these addresses (`bridge.md` §2) and
                // recover them itself; the signatures are the ones sitting in
                // the inbox under exactly these keys.
                attesters = ?attesters.iter().map(attest::hex_address).collect::<Vec<_>>(),
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
/// A job id that no earlier run of this binary has used.
///
/// The signed digest binds the job id (`primitives::attest`), so this is what
/// stops an attestation being lifted from one job onto another. While the id
/// was the constant 1 it stopped nothing: two runs over the same program and
/// inputs produced byte-identical claims, so attestations were interchangeable
/// across runs — and `/results` accepts a report from any registered address.
///
/// The TLA+ model shows what that costs (`spec/`, `MC_ReplayPreempt_N3M2`). A
/// relayer holding attestations from an earlier run does not have to win a race
/// to *displace* a vote, because first-write-wins only protects a vote already
/// cast. It only has to arrive before the workers do, and FHE evaluation is
/// seconds. Every inbox slot fills with replayed attestations, nothing is
/// displaced, and the job settles on an old answer with three honest workers
/// and no faults.
///
/// Wall-clock nanoseconds rather than a counter, because a counter restarts
/// with the process — which is exactly the failure being fixed. The bound is
/// honest: this makes ids unique *per coordinator*, which is all that is needed
/// while the coordinator is the only party assigning them. It is not global
/// uniqueness and it commits to nothing — two coordinators can still collide,
/// and neither can show a contract that its id was ever issued. `submitJob`
/// assigning the id on-chain is what makes this sound rather than merely
/// unlikely (`bridge.md` §2, task 2.9f).
/// Whether `attesters`-of-`workers` is a quorum this coordinator will run.
///
/// Split out so the rule can be tested at its boundary without binding sockets
/// or generating a keypair; `run` calls it before doing either.
fn quorum_error(attesters: usize, workers: usize) -> Option<String> {
    if attesters == 0 || attesters > workers {
        return Some(format!(
            "--attesters {attesters} is impossible with {workers} worker(s)"
        ));
    }

    if attesters * 2 <= workers {
        return Some(format!(
            "--attesters {attesters} is not a majority of {workers} worker(s); a \
             minority quorum can settle before the rest report (see spec/, \
             MC_GraceRace_N4M2)"
        ));
    }

    None
}

fn fresh_job_id() -> u64 {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    // Two parts, because neither is sufficient alone. The clock separates this
    // process from earlier ones; the counter separates jobs within it.
    //
    // The counter is not belt and braces. A wall clock read twice in quick
    // succession can return the same value — `two_runs_do_not_share_a_job_id`
    // failed on exactly that when this function was the timestamp alone, which
    // is the whole reason the test reads it twice rather than once.
    static BASE: OnceLock<u64> = OnceLock::new();
    static ISSUED: AtomicU64 = AtomicU64::new(0);

    let base = *BASE.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            // A clock before the epoch is not a reason to run a job under an id
            // some earlier run may also have used.
            .expect("system clock is before the unix epoch")
            .as_nanos() as u64
    });

    base.wrapping_add(ISSUED.fetch_add(1, Ordering::Relaxed))
}

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
    verifier: Arc<Verifier>,
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
                        // Attribute the report to whoever signed it. Anything
                        // that does not recover to a registered address is not
                        // an attestation, however well-formed it looks.
                        //
                        // Rejections are logged rather than dropped: a worker
                        // whose signature is being refused looks exactly like a
                        // worker that never reported, and the two have entirely
                        // different fixes (a misconfigured key versus a dead
                        // process).
                        let attester = match verifier.attribute(&report) {
                            Ok(address) => address,
                            Err(error) => {
                                warn!(
                                    claimed = %report.worker,
                                    %error,
                                    "discarding report that is not a valid attestation"
                                );
                                transport::respond(request, 403, error.as_bytes());
                                continue;
                            }
                        };

                        transport::respond(request, 200, b"ok");

                        // One attester, one attestation, and the *first* one
                        // counts. Keying by the recovered address already stops
                        // a party voting twice; keeping the first also stops a
                        // later message displacing a vote already cast, which
                        // matters because `/results` accepts a report from any
                        // registered address and an attestation is not a secret.
                        //
                        // A second attestation over a different result is
                        // signed evidence that one signer said two things about
                        // one job, and with unique job ids the right response
                        // would be to discard that signer's vote entirely.
                        // `submitJob` does not assign them yet, so `job_id` is
                        // the constant 1 and a replayed attestation from an
                        // earlier run is indistinguishable from equivocation.
                        // Dropping the vote would turn a replay anyone can
                        // mount into a denial of quorum. Keep the first, say so
                        // loudly, and revisit when job ids are real.
                        if let Recorded::AlreadyVoted { conflicting } =
                            record(&inbox, attester, report)
                        {
                            warn!(
                                attester = %attest::hex_address(&attester),
                                conflicting,
                                "attester reported more than once; keeping the first"
                            );
                        }

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

/// Fans one dispatch out to every worker.
///
/// Every worker receives byte-identical bytes. Before task 2.10i each copy
/// carried a per-worker secret, which meant the dispatch was also the thing
/// that authorised a reply; now a worker's authority to attest comes from a key
/// it already holds, so there is nothing to personalise and nothing lost if a
/// dispatch is relayed, cached or replayed.
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

/// Waits for `required` attesters to report the same attestation hash.
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
) -> Option<(SealedResult, Vec<Address>)> {
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
        .map(|(_, attesters)| attesters.len())
        .max()
        .unwrap_or(0);

    // `reported` counts attesters, and an attester is anyone the registry
    // accepts — not necessarily one of the workers this job was dispatched to.
    // Once the registry is larger than the dispatch set, which is the on-chain
    // shape (a global registry, a job sent to a subset of it), more parties can
    // report than were dispatched to and `dispatched - reported` stops meaning
    // "workers that might still answer". Saturating it to zero would call a job
    // dead while a dispatched worker was still evaluating, so when the count is
    // not authoritative, keep waiting and let the deadline decide. Early exit
    // is an optimisation; being wrong about it costs a correct result.
    let reported = reported(inbox);
    if reported > dispatched {
        return true;
    }

    best + (dispatched - reported) >= required
}

/// What happened to a report the verifier already accepted.
#[derive(Debug, PartialEq, Eq)]
enum Recorded {
    /// First attestation from this address; it is the vote.
    Counted,
    /// This address had already voted. `conflicting` distinguishes a duplicate
    /// from one signer saying two different things about one job.
    AlreadyVoted { conflicting: bool },
}

/// Files an attested report under the address that signed it, first one wins.
///
/// Split out of `serve` because this is where a vote is decided, and a decision
/// worth making is a decision worth testing without standing up a socket.
fn record(inbox: &Inbox, attester: Address, report: JobReport) -> Recorded {
    let mut inbox = inbox.lock().expect("inbox poisoned");
    match inbox.entry(attester) {
        Entry::Vacant(slot) => {
            slot.insert(report);
            Recorded::Counted
        }
        Entry::Occupied(held) => Recorded::AlreadyVoted {
            conflicting: outcome_key(&held.get().outcome) != outcome_key(&report.outcome),
        },
    }
}

/// What two reports have to share to be the same answer.
///
/// Only used to say whether a repeat attestation contradicts the one already
/// held, so failures compare by reason: two workers that both gave up for the
/// same stated reason said the same thing.
fn outcome_key(outcome: &JobOutcome) -> [u8; 32] {
    match outcome {
        JobOutcome::Evaluated(sealed) => sealed.hash,
        JobOutcome::Failed(reason) => wire::commitment(reason.as_bytes()),
    }
}

/// Groups reports by attestation hash and returns the first group to reach
/// `required` members.
fn tally(inbox: &Inbox, required: usize) -> Option<(SealedResult, Vec<Address>)> {
    let mut quorums: Vec<(SealedResult, Vec<Address>)> = group(inbox)
        .into_values()
        .filter(|(_, attesters)| attesters.len() >= required)
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
///
/// The members of a group are addresses recovered from signatures, so a group
/// of M is M distinct parties who each independently signed for that hash —
/// which is exactly the statement `fulfillJob` needs to be able to check.
fn group(inbox: &Inbox) -> HashMap<[u8; 32], (SealedResult, Vec<Address>)> {
    let reports = inbox.lock().expect("inbox poisoned");

    let mut groups: HashMap<[u8; 32], (SealedResult, Vec<Address>)> = HashMap::new();
    for (attester, report) in reports.iter() {
        if let JobOutcome::Evaluated(sealed) = &report.outcome {
            // Keyed by the recovered address, not by the name in the message:
            // the inbox holds at most one report per address, so each attester
            // contributes at most one attestation and cannot be two of them.
            groups
                .entry(sealed.hash)
                .or_insert_with(|| (sealed.clone(), Vec::new()))
                .1
                .push(*attester);
        }
    }

    // Deterministic order, so the attester list a job settles on does not
    // depend on `HashMap` iteration — it is destined for calldata, where two
    // runs producing different orderings would look like two different results.
    for (_, attesters) in groups.values_mut() {
        attesters.sort_unstable();
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
    for (attester, report) in inbox.lock().expect("inbox poisoned").iter() {
        if let JobOutcome::Failed(reason) = &report.outcome {
            warn!(
                attester = %attest::hex_address(attester),
                worker = %report.worker,
                reason = %reason,
                "worker reported failure"
            );
        }
    }

    let groups = group(inbox);
    if groups.len() > 1 {
        for (hash, (_, attesters)) in &groups {
            warn!(
                hash = %bytecode::hex(hash),
                // Named by address, because that is who signed. The
                // self-declared `worker` name is a label; this is evidence, and
                // disagreement is the one place the difference matters most.
                attesters = ?attesters.iter().map(attest::hex_address).collect::<Vec<_>>(),
                "attestation disagreement"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use primitives::attest::{Attestation, WorkerKey};

    use super::*;

    /// The program every test report claims to have run. Only its hash matters
    /// here — it is one of the two values a coordinator binds into the claim it
    /// reconstructs, and what makes a signature job- and program-specific.
    const BYTECODE_HASH: [u8; 32] = [0x77; 32];

    /// Stands in for a worker's sealed result. The blob is what the hash covers
    /// — `Verifier::attribute` re-checks that pairing, and the blob is opaque to
    /// the coordinator otherwise, which is the point: it settles on bytes it
    /// cannot read.
    fn sealed(marker: u8) -> SealedResult {
        let blob = vec![marker; 16];
        let hash = wire::commitment(&blob);
        SealedResult { blob, hash }
    }

    /// A worker with a real signing key. Derived from the label so a test can
    /// name the same worker twice and get the same address.
    fn worker(label: &str) -> WorkerKey {
        WorkerKey::derive(label)
    }

    fn evaluated(label: &str, marker: u8) -> (Address, JobReport) {
        let key = worker(label);
        let sealed = sealed(marker);
        let attestation = key.attest(&Claim::Result {
            job_id: 1,
            bytecode_hash: BYTECODE_HASH,
            result_hash: sealed.hash,
        });
        (
            key.address(),
            report(label, attestation, JobOutcome::Evaluated(sealed)),
        )
    }

    fn failed(label: &str, reason: &str) -> (Address, JobReport) {
        let key = worker(label);
        let attestation = key.attest(&Claim::Failure {
            job_id: 1,
            bytecode_hash: BYTECODE_HASH,
            reason_hash: wire::commitment(reason.as_bytes()),
        });
        (
            key.address(),
            report(label, attestation, JobOutcome::Failed(reason.into())),
        )
    }

    fn report(label: &str, attestation: Attestation, outcome: JobOutcome) -> JobReport {
        JobReport {
            job_id: 1,
            attestation,
            worker: label.to_string(),
            outcome,
            elapsed_ms: 1,
        }
    }

    fn inbox(reports: Vec<(Address, JobReport)>) -> Inbox {
        Arc::new(Mutex::new(reports.into_iter().collect()))
    }

    /// A verifier that accepts the named workers and nobody else.
    fn verifier(registered: &[&str]) -> Verifier {
        Verifier {
            job_id: 1,
            bytecode_hash: BYTECODE_HASH,
            registry: registered.iter().map(|l| worker(l).address()).collect(),
        }
    }

    #[test]
    fn a_majority_that_agrees_settles_and_names_its_attesters() {
        let inbox = inbox(vec![
            evaluated("w1", 0xaa),
            evaluated("w2", 0xaa),
            evaluated("w3", 0xbb),
        ]);

        let (result, attesters) = tally(&inbox, 2).expect("two workers agreed");
        assert_eq!(result.hash, sealed(0xaa).hash);

        // The attester set backs what `fulfillJob` will take on-chain, so it
        // must be the workers that actually signed for that hash and no one
        // else. `group` sorts, so this is order-independent by construction.
        let mut expected = vec![worker("w1").address(), worker("w2").address()];
        expected.sort_unstable();
        assert_eq!(attesters, expected);
    }

    #[test]
    fn a_signed_report_from_a_registered_worker_is_attributed_to_its_signer() {
        let (address, report) = evaluated("w1", 0xaa);

        assert_eq!(verifier(&["w1", "w2"]).attribute(&report).unwrap(), address);
    }

    #[test]
    fn a_report_from_a_signer_outside_the_registry_is_refused() {
        // The Sybil case. Generating a keypair is free, so a valid signature by
        // itself buys nothing: the registry is what makes an address mean "a
        // party this network agreed to accept work from" (`bridge.md` §2,
        // `registerWorker`).
        let (address, report) = evaluated("stranger", 0xaa);

        let error = verifier(&["w1", "w2", "w3"])
            .attribute(&report)
            .expect_err("an unregistered signer must be refused");
        assert!(error.contains("not a registered worker"), "got: {error}");
        assert!(
            error.contains(&attest::hex_address(&address)),
            "names who it refused, so a misconfigured registry is diagnosable: {error}"
        );
    }

    #[test]
    fn an_attestation_from_another_job_is_refused() {
        // The replay task 2.10i exists to close: a signature harvested from a
        // settled job, presented against a different one. The coordinator
        // reconstructs the claim from *its* job id, so the recovered address is
        // not the signer's and the registry check catches it.
        let key = worker("w1");
        let sealed = sealed(0xaa);
        let report = report(
            "w1",
            key.attest(&Claim::Result {
                job_id: 99,
                bytecode_hash: BYTECODE_HASH,
                result_hash: sealed.hash,
            }),
            JobOutcome::Evaluated(sealed),
        );

        let error = verifier(&["w1"])
            .attribute(&report)
            .expect_err("a signature for job 99 must not count for job 1");
        assert!(error.contains("not a registered worker"), "got: {error}");
    }

    #[test]
    fn an_attestation_for_another_program_is_refused() {
        // Same job, same answer, different circuit. Without the bytecode hash
        // in the claim, an attestation earned on a trivial program would be a
        // valid attestation for the one the job actually paid for.
        let key = worker("w1");
        let sealed = sealed(0xaa);
        let report = report(
            "w1",
            key.attest(&Claim::Result {
                job_id: 1,
                bytecode_hash: [0x11; 32],
                result_hash: sealed.hash,
            }),
            JobOutcome::Evaluated(sealed),
        );

        assert!(verifier(&["w1"]).attribute(&report).is_err());
    }

    #[test]
    fn an_attestation_lifted_onto_a_different_result_is_refused() {
        // A coordinator (or anyone on the path) swapping the result under a
        // genuine signature. This is the substitution `bridge.md` §5a is about,
        // seen one layer earlier.
        let (_, mut report) = evaluated("w1", 0xaa);
        report.outcome = JobOutcome::Evaluated(sealed(0xbb));

        assert!(verifier(&["w1"]).attribute(&report).is_err());
    }

    #[test]
    fn a_report_whose_blob_does_not_match_its_own_hash_is_refused() {
        // The hash is what gets counted and what `fulfillJob` compares against
        // `keccak256(resultBlob)` on-chain. A report where the two already
        // disagree would settle here and revert there.
        let (_, mut report) = evaluated("w1", 0xaa);
        if let JobOutcome::Evaluated(sealed) = &mut report.outcome {
            sealed.blob[0] ^= 0xff;
        }

        let error = verifier(&["w1"]).attribute(&report).unwrap_err();
        assert!(
            error.contains("does not match its own hash"),
            "got: {error}"
        );
    }

    #[test]
    fn a_report_answering_a_job_this_coordinator_is_not_running_is_refused() {
        let (_, mut report) = evaluated("w1", 0xaa);
        report.job_id = 2;

        let error = verifier(&["w1"]).attribute(&report).unwrap_err();
        assert!(error.contains("for job 2, not job 1"), "got: {error}");
    }

    #[test]
    fn a_garbled_signature_is_refused_rather_than_recovering_to_someone() {
        let (_, mut report) = evaluated("w1", 0xaa);
        report.attestation.s = [0u8; 32];

        assert!(verifier(&["w1"]).attribute(&report).is_err());
    }

    #[test]
    fn renaming_yourself_in_the_report_changes_nothing() {
        // The `worker` field is a log label. If it were load-bearing, one
        // worker could report M times under M invented names — which is the
        // hole task 2.9a's tokens patched and 2.10i's signatures close for
        // good, since the identity now falls out of the signature.
        let (address, mut report) = evaluated("w1", 0xaa);
        report.worker = "w2".into();

        assert_eq!(
            verifier(&["w1", "w2"]).attribute(&report).unwrap(),
            address,
            "attribution must follow the key, not the name"
        );
    }

    #[test]
    fn a_signed_failure_is_attributable_but_never_an_attestation() {
        // Both halves matter. A failure has to be attributable, or anyone can
        // manufacture evidence against an honest operator; and it must not
        // count towards a quorum, or two broken workers settle a job between
        // them.
        let (address, report) = failed("w2", "stack underflow at op 3");
        assert_eq!(verifier(&["w1", "w2"]).attribute(&report).unwrap(), address);

        let inbox = inbox(vec![
            evaluated("w1", 0xaa),
            failed("w2", "stack underflow at op 3"),
            failed("w3", "stack underflow at op 3"),
        ]);
        assert!(tally(&inbox, 2).is_none());
    }

    #[test]
    fn a_failure_whose_reason_was_rewritten_in_flight_is_refused() {
        // The reason travels as text beside a signature over its hash, so
        // altering it invalidates the report rather than putting words in an
        // operator's mouth.
        let (_, mut report) = failed("w2", "stack underflow at op 3");
        report.outcome = JobOutcome::Failed("I am dishonest".into());

        assert!(verifier(&["w1", "w2"]).attribute(&report).is_err());
    }

    #[test]
    fn one_worker_cannot_reach_quorum_by_reporting_twice() {
        // M-of-N counts *distinct* addresses. The inbox is keyed by recovered
        // address precisely so that this is true by construction rather than by
        // a check somebody could forget: a second report from w1 replaces the
        // first instead of adding to it.
        let first = evaluated("w1", 0xaa);
        let second = evaluated("w1", 0xaa);
        assert_eq!(first.0, second.0, "the same key gives the same address");

        let inbox = inbox(vec![first, second]);
        assert_eq!(reported(&inbox), 1);
        assert!(
            tally(&inbox, 2).is_none(),
            "one worker reporting twice is one attestation"
        );

        // ...and a genuinely second party tips it over.
        inbox
            .lock()
            .unwrap()
            .extend(std::iter::once(evaluated("w2", 0xaa)));
        assert!(tally(&inbox, 2).is_some());
    }

    #[test]
    fn a_registry_is_parsed_from_addresses_and_rejects_anything_else() {
        let one = attest::hex_address(&worker("w1").address());
        let two = attest::hex_address(&worker("w2").address());

        assert_eq!(
            parse_registry(&[one.clone(), two.clone()]).unwrap(),
            vec![worker("w1").address(), worker("w2").address()]
        );

        // A duplicate is refused rather than deduplicated: silently accepting
        // it would let `--attesters 2` be satisfied by one party.
        let error = parse_registry(&[one.clone(), two, one]).unwrap_err();
        assert!(error.contains("more than once"), "got: {error}");

        let error = parse_registry(&["0xnot-an-address".into()]).unwrap_err();
        assert!(error.contains("20-byte hex address"), "got: {error}");
    }

    #[test]
    fn a_minority_quorum_is_refused_before_the_job_starts() {
        // 2-of-4 is the shape the TLA+ model settles wrongly
        // (`spec/`, `MC_GraceRace_N4M2`): two faulty workers report, the
        // straggler grace expires while the honest two are still evaluating,
        // and `tally` sees one group at quorum with no split to refuse.
        //
        // Checked at startup rather than at settlement because by settlement
        // there is nothing left to notice — the wrong answer looks exactly like
        // the right one, and `attestation.md` §1 says the key holder cannot
        // tell them apart either.
        let error = quorum_error(2, 4).expect("2-of-4 must be refused");
        assert!(
            error.contains("majority"),
            "the refusal must say why: {error}"
        );

        // The demo's own shape, and the boundary either side of it.
        assert!(quorum_error(2, 3).is_none(), "2-of-3 is a majority");
        assert!(quorum_error(3, 4).is_none(), "3-of-4 is a majority");
        assert!(quorum_error(2, 5).is_some(), "2-of-5 is not");
        assert!(quorum_error(3, 5).is_none(), "3-of-5 is");
    }

    #[test]
    fn two_runs_do_not_share_a_job_id() {
        // The signed digest binds the job id, so equal ids across runs make
        // attestations interchangeable between them — which is what lets a
        // relayer settle this job with an earlier job's signatures
        // (`spec/`, `MC_ReplayPreempt_N3M2`). Nanosecond wall clock, so this is
        // about the id not being a constant rather than about entropy.
        let first = fresh_job_id();
        let second = fresh_job_id();
        assert_ne!(first, second, "a fresh id per job, not a fixed one");
        assert_ne!(first, 1, "and specifically not the old constant");
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
    fn a_repeat_attestation_does_not_displace_the_vote_already_cast() {
        // `/results` takes a report from any registered address, and an
        // attestation is not secret, so "arrived later" must not mean "wins".
        // Last-write-wins would let a replayed attestation from an earlier run
        // overwrite the vote a worker actually cast for this job.
        let (address, first) = evaluated("w1", 0xaa);
        let (_, contradiction) = evaluated("w1", 0xbb);
        let (_, repeat) = evaluated("w1", 0xaa);
        let inbox = inbox(vec![]);

        assert_eq!(record(&inbox, address, first), Recorded::Counted);
        assert_eq!(
            record(&inbox, address, contradiction),
            Recorded::AlreadyVoted { conflicting: true },
            "a second, different answer from one signer is a contradiction"
        );
        assert_eq!(
            record(&inbox, address, repeat),
            Recorded::AlreadyVoted { conflicting: false },
            "the same answer twice is a duplicate, not a contradiction"
        );

        let groups = group(&inbox);
        assert_eq!(groups.len(), 1, "neither later report forms a group");
        assert!(
            groups.contains_key(&sealed(0xaa).hash),
            "the first attestation is the one that counts"
        );
    }

    #[test]
    fn a_job_stays_open_when_more_parties_report_than_were_dispatched_to() {
        // The registry is global and a job goes to a subset of it, so a
        // registered worker that was not dispatched to can still report. When
        // that happens the coordinator cannot tell which inbox entries came
        // from workers it is waiting on, and must not conclude the job is dead:
        // here two of three dispatched workers disagree, a fourth party has
        // also reported, and the third dispatched worker is still evaluating.
        let crowded = inbox(vec![
            evaluated("w1", 0xaa),
            evaluated("w2", 0xbb),
            evaluated("w4", 0xcc),
        ]);
        assert_eq!(reported(&crowded), 3);
        assert!(
            agreement_still_possible(&crowded, 2, 2),
            "an outstanding dispatched worker could still form a quorum"
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

        assert_eq!(result.hash, sealed(0xaa).hash);
        assert!(
            started.elapsed() < STRAGGLER_GRACE,
            "collect waited {:?} after every worker had reported",
            started.elapsed()
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
