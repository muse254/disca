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
//! The coordinator does not hold the client key and never sees a plaintext.
//! It is handed a server key blob, a bytecode blob and already-encrypted input
//! blobs, and it hands back the result blob still encrypted. The key holder is
//! `disca-cli` — a separate process that generates the keypair, encrypts the
//! inputs and decrypts the result, and that never talks to a worker
//! (`architecture.md` §3, task 4.3).
//!
//! That split is not tidiness. `registerProgram` pins a server key hash
//! on-chain at registration (`bridge.md` §3), so a coordinator that minted its
//! own keypair on every start could never match a registered program — it has
//! to be *given* a key, not produce one. And a coordinator that took plaintext
//! inputs on its command line put the secret values in `argv` of the party that
//! fans the job out, on a system whose whole claim is that no node sees them.
//!
//! # One process, many jobs
//!
//! This is a job service rather than a one-shot runner
//! (`next-architecture.md` §2.2, §4 step 3). Every piece of state a job needs —
//! its inbox, the dispatch set it is waiting on, the verifier that decides who
//! may speak for it, its deadline, and what it came to — lives in a [`Job`]
//! keyed by job id in [`Coordinator::jobs`], and `POST /results` routes an
//! arriving report to one of them by the id the report names.
//!
//! That was a prerequisite for task 3.4 rather than a convenience. The chain
//! watcher ([`crate::watcher`]) receives `JobRequested` events concurrently and
//! keeps several jobs in flight; landing it on top of one process-global inbox
//! would have meant debugging concurrency and chain plumbing at once, and the
//! first two concurrent jobs would have eaten each other's votes — the inbox is
//! keyed by attester address, so two jobs sharing one would let a worker's
//! report for job A occupy that worker's only slot in job B.
//!
//! Task 2.0d claimed the coordinator-local job id was chosen so that swapping
//! in the on-chain one would "touch one place". It was never the id that was in
//! the way; it was the absence of per-job state. [`Coordinator::accept_job`]
//! takes the id as an argument for exactly that reason — `None` mints one
//! locally, and the watcher passes the one `submitJob` assigned.
//!
//! There are two callers now and they share everything here. [`run`] is the
//! one-shot command line: one job from argv, its answer written to files, its
//! failure the process's exit code. `watcher::run` is the chain: every job a
//! contract posts, each on its own thread, each answer a `fulfillJob`
//! transaction. Nothing below this line knows which of the two it is serving.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use std::path::PathBuf;

use primitives::attest::{self, Address, Attestation, Claim};
use primitives::wire::{self, SealedResult};
use primitives::{bytecode, validate};
use tracing::{Span, info, info_span, warn};

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
    /// The compressed server key, as `disca-cli keygen` wrote it. The
    /// coordinator serves these bytes to workers and hashes them to name the
    /// key; it never installs one, because nothing here evaluates or decrypts.
    pub server_key: Vec<u8>,
    /// DISCA bytecode, as `disca-cli compile` wrote it.
    pub bytecode: Vec<u8>,
    /// Which exported function of that program to run.
    pub function: String,
    /// Already-encrypted inputs, in argument order. The coordinator cannot read
    /// them and does not need to.
    pub inputs: Vec<Vec<u8>>,
    /// Where to write the result blob once the job settles. The coordinator
    /// cannot decrypt it; `disca-cli decrypt` can.
    pub result_out: Option<PathBuf>,
    /// The job id to run under, when a chain has already assigned one.
    ///
    /// `None` mints one locally, which is right for a run with no chain. It is
    /// wrong the moment there *is* one: a worker signs a digest binding the job
    /// id, and `fulfillJob` rebuilds that digest from the id `submitJob`
    /// assigned. Two different ids means signatures that recover to nothing the
    /// registry knows, and the contract rejects a settlement that is in every
    /// other respect correct — `NotRegisteredWorker`, for a worker that is
    /// registered.
    ///
    /// That is not hypothetical: it is what `scripts/run-anvil.sh` hit before
    /// this field existed, and why it settled from a fixture rather than from
    /// the coordinator. The flag remains for that script's `cast`-driven mode,
    /// where a shell reads the id off the chain and passes it in; the watcher
    /// needs no flag, because it has the event.
    pub job_id: Option<u64>,
    /// Where to write the winning group's signatures, in the shape
    /// `fulfillJob` takes. See [`attestations_json`].
    ///
    /// Optional because a run with no chain to submit to has nothing to do with
    /// them, and the local demo is exactly that run.
    pub attestations_out: Option<PathBuf>,
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

/// Reports collected for **one** job, keyed by the address that signed each one.
///
/// A map rather than a list, because agreement must be counted per attester:
/// two reports from the same address are one attestation, and a report whose
/// signature does not recover to a registered address is not an attestation at
/// all. The key is the *recovered* address, so the map's own shape enforces
/// one-attestation-per-party — there is no field a reporter can set to occupy a
/// second slot.
///
/// One of these per [`Job`], not one per process. That shape is what makes the
/// paragraph above true across jobs as well as within one: an attester votes
/// once per *job*, and a single map keyed by address alone cannot say that —
/// w1's report for job A and w1's report for job B are the same key, so the
/// second is refused as a duplicate of the first and job B silently loses a
/// vote (`next-architecture.md` §2.2).
type Inbox = Mutex<HashMap<Address, JobReport>>;

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

/// Every job this coordinator has accepted, keyed by the id it was accepted
/// under.
///
/// Two levels of lock rather than one, and the split is the point: this one is
/// held just long enough to look a job up and clone its handle, so a report
/// arriving for job A never blocks job B while an ECDSA recovery runs. All the
/// contended state — the inbox — is behind each [`Job`]'s own lock, which is
/// also the lock `collect` waits under.
type Jobs = Mutex<HashMap<u64, Arc<Job>>>;

/// A coordinator process: the jobs it is running, and the server key it serves.
pub struct Coordinator {
    jobs: Jobs,
    /// The compressed server key, served by hash at `/keys/<hash>`.
    ///
    /// Process-wide rather than per job, because it is the key *workers* are
    /// told to hold: `registerProgram` pins one `serverKeyHash` per program
    /// (`bridge.md` §3), so every job over that program wants the same bytes. A
    /// service fronting several programs would key this by hash as well; there
    /// is one program today and inventing that lifecycle would be inventing a
    /// thing nothing exercises.
    server_key: Vec<u8>,
    server_key_hash: [u8; 32],
}

/// One job in flight, and everything deciding it needs.
///
/// Every field here used to be a local of `run` or a process-global
/// (`next-architecture.md` §2.2). Owned per job so that two jobs settling at
/// the same time share the map they are looked up in and nothing else.
struct Job {
    /// Who may speak for this job, and about what.
    ///
    /// Per job because the two hashes it holds are *this* job's. "A signature
    /// for any other job or any other program recovers to some address that is
    /// not the sender's" is only true if the values recovery is performed
    /// against belong to the job the report was routed to; a process-global
    /// verifier makes it true of one job and vacuous for the rest.
    verifier: Verifier,
    /// M — how many agreeing attesters this job settles on.
    required: usize,
    /// N — how many workers this job was dispatched to.
    ///
    /// The dispatch set's size, not the registry's: `collect` uses it to decide
    /// when everyone has spoken and when nobody outstanding could still form a
    /// quorum, and both questions are about the parties this job is waiting on.
    dispatched: usize,
    deadline: Duration,
    inbox: Inbox,
    /// Nudges whoever is collecting this job that a report landed, so `collect`
    /// reacts to arrivals instead of polling.
    ///
    /// A buffered channel rather than a condvar because a report can land in
    /// the window between `collect` tallying and `collect` going back to sleep.
    /// A send survives that window; a `notify_one` is lost in it, and the job
    /// would then sit out the grace or the deadline for no reason.
    wake: Sender<()>,
    /// The other end, handed to the first caller of [`Coordinator::settle`].
    ///
    /// In an `Option` so that taking it is what makes "this job is being
    /// collected" a fact rather than a convention: a `Receiver` has one
    /// consumer, and two threads collecting one job would each see half of its
    /// wakeups and each conclude the other half never arrived.
    woken: Mutex<Option<Receiver<()>>>,
    /// What this job came to. Written once, by [`Coordinator::settle`].
    outcome: Mutex<Outcome>,
    /// Tags this job's log lines with its id.
    ///
    /// Carried on the job rather than entered once in `run`, because the
    /// threads that log about a job are no longer one thread: the HTTP handler
    /// files reports for every job in flight, and with several running its
    /// warnings are unreadable without saying which job they are about.
    span: Span,
}

impl Job {
    /// A job with an empty inbox, ready to be reported to.
    ///
    /// Used by `accept_job` and by the tests, so that what the tests exercise
    /// is the job a real dispatch would create rather than a hand-built
    /// lookalike that could drift from it.
    fn new(verifier: Verifier, required: usize, dispatched: usize, deadline: Duration) -> Job {
        let (wake, woken) = channel::<()>();
        Job {
            span: info_span!("job", job_id = verifier.job_id),
            verifier,
            required,
            dispatched,
            deadline,
            inbox: Mutex::new(HashMap::new()),
            wake,
            woken: Mutex::new(Some(woken)),
            outcome: Mutex::new(Outcome::Collecting),
        }
    }
}

/// Where a job is in its life, and what it came to once it is over.
///
/// `pub(crate)` from here down to [`Attester`] because the chain watcher
/// (`crate::watcher`, task 3.4) is the second caller of
/// [`Coordinator::settle`] and has to turn a settlement into `fulfillJob`
/// calldata. The CLI path writes the same values to a file through
/// [`attestations_json`]; a watcher submitting a transaction needs them as
/// values, not as JSON it would then have to parse back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Outcome {
    /// Accepted and collecting. No quorum yet, and the deadline has not passed.
    Collecting,
    /// `required` distinct registered attesters signed the same result.
    Settled(Settlement),
    /// This job will not settle: the deadline passed, agreement stopped being
    /// possible, or two groups reached quorum and `tally` refused to choose
    /// between them.
    Unsettled,
}

/// What a settled job amounts to, and everything `fulfillJob` needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Settlement {
    pub(crate) result: SealedResult,
    /// The winning group, ascending by address. See [`attesters_of`] for why
    /// the ordering is part of the value rather than a presentation detail.
    pub(crate) attesters: Vec<Attester>,
}

/// One member of the winning group: who signed, and what they signed with.
///
/// The signature travels beside the address because on-chain the address is
/// *recovered* from it. A contract handed a bare address list is taking the
/// coordinator's word for who attested, which is the unsoundness `bridge.md`
/// §2b records; handed the signatures, it recovers the addresses itself and the
/// coordinator's word is not part of the check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Attester {
    pub(crate) address: Address,
    pub(crate) attestation: Attestation,
}

/// Why a report that arrived on `/results` was not counted.
///
/// Two shapes rather than one string because they have different causes,
/// different fixes and different status codes: "nothing here knows about job 7"
/// is a sender talking to a coordinator that never accepted that job, while
/// "this is not an attestation for job 7" is a sender whose key or whose claim
/// is wrong. Collapsing them would make a misrouted worker and a misconfigured
/// one look identical in a log.
#[derive(Debug, PartialEq, Eq)]
enum Rejection {
    /// No job with this id has been accepted here.
    UnknownJob(u64),
    /// The report named a job this coordinator is running, and did not verify
    /// against it.
    NotAnAttestation(String),
}

impl Rejection {
    /// Not found versus forbidden, because that is the distinction above and
    /// the status code is the only part of it a script sees.
    fn status(&self) -> u16 {
        match self {
            Rejection::UnknownJob(_) => 404,
            Rejection::NotAnAttestation(_) => 403,
        }
    }
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Rejection::UnknownJob(job_id) => {
                write!(f, "no job {job_id} was accepted by this coordinator")
            }
            Rejection::NotAnAttestation(reason) => f.write_str(reason),
        }
    }
}

/// One job as a caller asks for it.
///
/// Separate from [`Config`], which is the shape of a command line. The CLI is
/// one caller of [`Coordinator::accept_job`]; task 3.4's chain watcher is the
/// other, and it builds this from a `JobRequested` event rather than from argv.
pub struct JobSpec {
    /// The id to run this job under, or `None` to mint one.
    ///
    /// This is the seam task 2.0d promised. The watcher passes `Some(id)` — the
    /// id `submitJob` assigned on-chain, which is what makes uniqueness a
    /// property anyone can check rather than a convention this process keeps
    /// (`bridge.md` §2, task 2.9f). The CLI passes `None` and gets
    /// [`fresh_job_id`], which is unique per coordinator and commits to nothing.
    pub job_id: Option<u64>,
    pub workers: Vec<String>,
    pub registry: Vec<Address>,
    pub attesters: usize,
    pub bytecode: Vec<u8>,
    pub function: String,
    pub inputs: Vec<Vec<u8>>,
    pub deadline: Duration,
}

impl Coordinator {
    /// A coordinator holding the server key it will serve to workers.
    ///
    /// The key is a blob to this process. Naming it by hash is what lets a
    /// worker verify what it pulled (`worker.rs`), and is the same value
    /// `registerProgram` pins on-chain (`bridge.md` §3).
    pub fn new(server_key: Vec<u8>) -> Coordinator {
        let server_key_hash = wire::commitment(&server_key);
        Coordinator {
            jobs: Mutex::new(HashMap::new()),
            server_key,
            server_key_hash,
        }
    }

    /// The job with this id, if this coordinator accepted one.
    ///
    /// Clones the handle rather than lending it, so the registry lock covers a
    /// map lookup and nothing else. Everything slow that happens to a report —
    /// signature recovery above all — happens after this returns.
    fn job(&self, job_id: u64) -> Option<Arc<Job>> {
        self.jobs
            .lock()
            .expect("jobs poisoned")
            .get(&job_id)
            .cloned()
    }

    /// Puts a job in the registry, refusing an id that is already there.
    ///
    /// A duplicate id is refused rather than replacing what is there.
    /// `fresh_job_id` makes a collision within one process unlikely and an
    /// on-chain id makes it impossible, but "unlikely" quietly overwriting a
    /// running job's inbox — dropping every vote already cast for it, with no
    /// error anywhere — is not a failure mode worth keeping for the sake of one
    /// less branch.
    fn register(&self, job: Job) -> Result<Arc<Job>, String> {
        let job_id = job.verifier.job_id;
        match self.jobs.lock().expect("jobs poisoned").entry(job_id) {
            Entry::Vacant(slot) => Ok(slot.insert(Arc::new(job)).clone()),
            Entry::Occupied(_) => Err(format!("job {job_id} is already running here")),
        }
    }

    /// Validates a job, registers it, fans it out, and returns the id it runs
    /// under.
    ///
    /// The entry point (`next-architecture.md` §4, step 3). The one-shot CLI
    /// path in [`run`] is one caller; the chain watcher of task 3.4 is the
    /// other, and the only difference between them is where `spec.job_id` comes
    /// from.
    ///
    /// Returns as soon as the job is dispatched. Waiting for it is
    /// [`Coordinator::settle`], and the split is what lets a caller keep
    /// several jobs in flight without this module deciding how.
    pub fn accept_job(&self, spec: JobSpec) -> Result<u64, String> {
        // A quorum has to be a majority of the workers dispatched to, and
        // refusing anything less is not caution — a minority quorum can settle
        // on a wrong answer with nothing left to contradict it.
        //
        // The TLA+ model exhibits it (`spec/`, `MC_GraceRace_N4M2`): with N = 4
        // and M = 2, the two faulty workers report, `STRAGGLER_GRACE` expires
        // while the honest two are still evaluating — FHE is seconds of work
        // and the grace is five — and `tally` sees exactly one group at quorum
        // and settles on it. The split refusal never fires, because at that
        // moment there is no split to see. A longer grace does not help: a
        // faulty worker need not evaluate at all, so it can always report
        // first.
        //
        // `2M > N` makes two disjoint quorums impossible, so the honest workers
        // can always outvote a faulty group unless they are outnumbered — which
        // is the fault threshold being exceeded, not a race being lost.
        if let Some(error) = quorum_error(spec.attesters, spec.workers.len()) {
            return Err(error);
        }

        // A quorum can only ever be formed by registered addresses, so a
        // registry smaller than M is a job that cannot settle. Say so before
        // fanning out, rather than after the deadline.
        if spec.registry.len() < spec.attesters {
            return Err(format!(
                "--attesters {} is impossible with {} registered worker(s)",
                spec.attesters,
                spec.registry.len()
            ));
        }

        let program_hash = bytecode::hash_bytecode(&spec.bytecode);

        let arity = check_program(&spec.bytecode, &spec.function)?;
        if spec.inputs.len() != arity {
            return Err(format!(
                "{} takes {arity} input(s), but {} were supplied",
                spec.function,
                spec.inputs.len()
            ));
        }

        let job_id = spec.job_id.unwrap_or_else(fresh_job_id);

        let job = self.register(Job::new(
            Verifier {
                job_id,
                bytecode_hash: program_hash,
                registry: spec.registry.iter().copied().collect(),
            },
            spec.attesters,
            spec.workers.len(),
            spec.deadline,
        ))?;
        let _enter = job.span.enter();

        // Commitments are computed here rather than trusted from the caller:
        // they are `keccak256` of bytes this process is holding, so recomputing
        // costs nothing and removes a way for them to disagree. Once
        // `submitJob` exists these come from the chain instead, which is what
        // makes the worker's check adversarial rather than diagnostic
        // (task 2.9f).
        let inputs: Vec<InputBlob> = spec
            .inputs
            .iter()
            .map(|bytes| InputBlob {
                commitment: wire::commitment(bytes),
                bytes: bytes.clone(),
            })
            .collect();

        info!(
            function = %spec.function,
            program_hash = %bytecode::hex(&program_hash),
            inputs = inputs.len(),
            workers = spec.workers.len(),
            registered = spec.registry.len(),
            attesters = spec.attesters,
            "job prepared"
        );

        let dispatch = JobDispatch {
            job_id,
            bytecode: spec.bytecode,
            function: spec.function,
            inputs,
            server_key_hash: self.server_key_hash,
        };

        // Registered above and dispatched here, never the other way round. A
        // worker can report before this call has finished walking the list, and
        // a report for a job not yet in the registry would be turned away as
        // unknown — surfacing as one attestation mysteriously missing from an
        // otherwise healthy run.
        dispatch_to_workers(&spec.workers, &dispatch);

        Ok(job_id)
    }

    /// Files a report against the job it names.
    ///
    /// Routing on `report.job_id` is what makes `/results` a service endpoint
    /// rather than a mailbox for the one job in flight
    /// (`next-architecture.md` §2.2). Until now the field was read only by the
    /// equality check inside [`Verifier::attribute`], against *the* job,
    /// because there was only one; a worker answering job 7 with job 3's report
    /// was counted, which is harmless with one job and a replay bug with two.
    ///
    /// That check stays where it is. It is unreachable through this path, since
    /// the job was looked up by the very field it compares — and it is worth
    /// keeping anyway, because it is the property the model states as
    /// `Accepted(s): s.job = JobId` (`spec/DiscaAttestation.tla`) and because
    /// attribution should not become correct only by virtue of its caller.
    ///
    /// A report for a job nobody here accepted is a named rejection: not a
    /// panic, which would let any peer stop the process, and not a silent drop,
    /// because a coordinator that quietly discards reports is indistinguishable
    /// from a worker that never sent them and the two have entirely different
    /// fixes.
    fn deliver(&self, report: JobReport) -> Result<(Address, Recorded), Rejection> {
        let job = self
            .job(report.job_id)
            .ok_or(Rejection::UnknownJob(report.job_id))?;
        let _enter = job.span.enter();

        // Attribute the report to whoever signed it, against *this* job's id
        // and program hash. Anything that does not recover to a registered
        // address is not an attestation, however well-formed it looks.
        let attester = job
            .verifier
            .attribute(&report)
            .map_err(Rejection::NotAnAttestation)?;

        // One attester, one attestation, and the *first* one counts. Keying by
        // the recovered address already stops a party voting twice; keeping the
        // first also stops a later message displacing a vote already cast,
        // which matters because `/results` accepts a report from any registered
        // address and an attestation is not a secret.
        //
        // A second attestation over a different result is signed evidence that
        // one signer said two things about one job, and with unique job ids the
        // right response would be to discard that signer's vote entirely. Ids
        // are unique per coordinator now and not yet globally (`fresh_job_id`),
        // so a replayed attestation from an earlier run of *another*
        // coordinator is still indistinguishable from equivocation. Dropping
        // the vote would turn a replay anyone can mount into a denial of
        // quorum. Keep the first, say so loudly, and revisit when `submitJob`
        // assigns the ids.
        let recorded = record(&job.inbox, attester, report);

        // Woken whether or not the report changed anything: `collect` re-tallies
        // cheaply, and a duplicate is still news that the sender is alive.
        let _ = job.wake.send(());

        Ok((attester, recorded))
    }

    /// Waits for a job to settle, and records what it came to.
    ///
    /// Separate from [`Coordinator::accept_job`] so that a caller picks its own
    /// concurrency: the CLI accepts one job and blocks on it, and the watcher
    /// will hand each `JobRequested` to a thread that does the same. Two such
    /// threads share the registry lookup and nothing else — each job's grace,
    /// deadline and quorum are its own.
    pub(crate) fn settle(&self, job_id: u64) -> Result<Outcome, String> {
        let job = self
            .job(job_id)
            .ok_or_else(|| format!("no job {job_id} was accepted by this coordinator"))?;
        let _enter = job.span.enter();

        let woken = job
            .woken
            .lock()
            .expect("job poisoned")
            .take()
            .ok_or_else(|| format!("job {job_id} is already being collected"))?;

        let agreed = collect(
            &job.inbox,
            &woken,
            job.required,
            job.dispatched,
            job.deadline,
        );
        report_outcome(&job.inbox);

        let outcome = match agreed {
            Some((result, attesters)) => Outcome::Settled(Settlement {
                attesters: attesters_of(&job.inbox, &attesters),
                result,
            }),
            None => Outcome::Unsettled,
        };

        // The job keeps its own answer. Nothing reads it back today — `run`
        // uses the value returned here — but a watcher that has to resubmit a
        // `fulfillJob` transaction needs the settlement without re-collecting,
        // and a job that does not remember what it decided cannot provide it.
        *job.outcome.lock().expect("job poisoned") = outcome.clone();
        Ok(outcome)
    }
}

/// The one-shot command line path: accept exactly one job and wait for it.
///
/// One caller of [`Coordinator::accept_job`] among the two there will be
/// (`next-architecture.md` §4, step 3). Everything specific to *this* caller
/// lives here — a single job, taken from argv, whose settlement is written to
/// files and whose failure is the process's exit code.
pub fn run(config: Config) -> Result<(), String> {
    // Kept before the job is moved into its spec: the settled job's program is
    // named in the attestations file, and the failure message needs the shape
    // of the quorum that was not reached.
    let program_hash = bytecode::hash_bytecode(&config.bytecode);
    let attesters_required = config.attesters;
    let dispatched = config.workers.len();

    let coordinator = Arc::new(Coordinator::new(config.server_key));

    // The listener is up before any job is accepted, because a dispatch is
    // immediately followed by workers pulling `/keys/<hash>` from it. A
    // validation failure below therefore costs a bound port for the few
    // milliseconds before the process exits, which is the right way round: the
    // alternative is a worker fetching a key from a coordinator that is not yet
    // listening.
    serve(&config.bind, coordinator.clone())?;

    let started = Instant::now();
    let job_id = coordinator.accept_job(JobSpec {
        // Supplied when a chain has already assigned one, minted otherwise.
        // Task 3.4 is the watcher passing it rather than a caller.
        job_id: config.job_id,
        workers: config.workers,
        registry: config.registry,
        attesters: config.attesters,
        bytecode: config.bytecode,
        function: config.function,
        inputs: config.inputs,
        deadline: config.deadline,
    })?;

    match coordinator.settle(job_id)? {
        Outcome::Settled(settlement) => {
            let sealed = &settlement.result;
            if let Some(path) = &config.result_out {
                std::fs::write(path, &sealed.blob)
                    .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            }

            // The evidence, beside the answer. Written only when asked for, and
            // written before "job settled" is logged, so a run that claims to
            // have settled has already produced the thing a contract would be
            // handed (task 3.3, task 3.4).
            if let Some(path) = &config.attestations_out {
                let json =
                    attestations_json(job_id, &program_hash, &sealed.hash, &settlement.attesters);
                std::fs::write(path, json)
                    .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
            }

            info!(
                result_bytes = sealed.blob.len(),
                result_out = ?config.result_out,
                attestations_out = ?config.attestations_out,
                result_hash = %bytecode::hex(&sealed.hash),
                // Recovered, not asserted. `fulfillJob` takes the *signatures*
                // rather than these addresses (`bridge.md` §2) and recovers
                // them itself; the signatures are the ones `--attestations`
                // writes, under exactly these addresses.
                attesters = ?settlement
                    .attesters
                    .iter()
                    .map(|attester| attest::hex_address(&attester.address))
                    .collect::<Vec<_>>(),
                elapsed_ms = started.elapsed().as_millis(),
                "job settled"
            );
            Ok(())
        }
        // `settle` writes one of the two terminal outcomes, so `Collecting`
        // cannot come back from it; matched rather than ignored so that adding
        // a third state is a compile error here.
        Outcome::Unsettled | Outcome::Collecting => Err(format!(
            "job {job_id} did not reach {attesters_required}-of-{dispatched} agreement within {:?}",
            config.deadline
        )),
    }
}

/// Whether `attesters`-of-`workers` is a quorum this coordinator will run.
///
/// Split out so the rule can be tested at its boundary without binding sockets
/// or fanning a job out; `accept_job` calls it before doing either.
///
/// The chain watcher calls it a second time, at startup rather than per job:
/// it takes M from `registerProgram` and N from `--worker`, so a deployment
/// whose registered quorum is not a majority of the workers this operator runs
/// is a configuration that can never settle anything. Better said once at
/// startup than once per job, after the fan-out, forever.
pub(crate) fn quorum_error(attesters: usize, workers: usize) -> Option<String> {
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

/// A job id that no earlier run of this binary, and no other job in this one,
/// has used.
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
/// unlikely (`bridge.md` §2, task 2.9f), and [`JobSpec::job_id`] is where that
/// id will arrive.
///
/// Now that the coordinator runs several jobs at once, this separates them from
/// each other as well as from earlier runs — two jobs sharing an id would share
/// a registry slot, and `accept_job` refuses the second rather than letting the
/// counter's uniqueness be the only thing standing between them.
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

/// Checks up front that the blob is a program and that the function the job
/// names exists and is runnable, and returns how many inputs it takes.
fn check_program(bytecode_blob: &[u8], function: &str) -> Result<usize, String> {
    // `deserialize` already validates every circuit it accepts (task 1.4), so
    // this rejects a malformed program before a worker is asked to spend
    // minutes on it. The coordinator is no longer the compiler — `disca-cli
    // compile` is — but it is still the last party that can fail cheaply.
    let program = bytecode::deserialize(bytecode_blob).map_err(|e| e.to_string())?;

    let func = program
        .function(function)
        .ok_or_else(|| format!("the program exports no function named {function}"))?;

    let layout = validate::validate(func).map_err(|e| e.to_string())?;
    info!(
        ops = func.body.len(),
        peak_stack = layout.max_depth,
        params = func.sig.params.len(),
        "program accepted"
    );

    Ok(func.sig.params.len())
}

/// Starts the HTTP surface: the server key by hash, and worker reports.
///
/// One listener for the process, serving every job the coordinator has
/// accepted. `/results` is no longer wired to a single job's inbox — it hands
/// the report to [`Coordinator::deliver`], which routes on the id the report
/// names. That is what lets a second job be accepted while a first is still
/// collecting, and it is all this layer knows about jobs.
///
/// `pub(crate)` because the chain watcher stands the same surface up: a worker
/// pulls the server key and posts its report over it whether the job came from
/// argv or from a `JobRequested` event. A second copy in `watcher.rs` would be
/// a second place for `/results` routing to drift from what workers send.
pub(crate) fn serve(bind: &str, coordinator: Arc<Coordinator>) -> Result<(), String> {
    let server = tiny_http::Server::http(bind).map_err(|e| format!("cannot bind {bind}: {e}"))?;
    info!(bind = %bind, "coordinator listening");

    let key_path = format!("/keys/{}", bytecode::hex(&coordinator.server_key_hash));

    thread::spawn(move || {
        for mut request in server.incoming_requests() {
            let url = request.url().to_string();

            if url == key_path {
                info!(bytes = coordinator.server_key.len(), "serving server key");
                transport::respond(request, 200, &coordinator.server_key);
                continue;
            }

            if url == "/results" {
                match transport::read_body(&mut request)
                    .and_then(|body| crate::protocol::decode::<JobReport>(&body))
                {
                    Ok(report) => {
                        // Kept before the report is handed over, so the log
                        // line for a rejection can still say who claimed to
                        // send it and which job they named.
                        let job_id = report.job_id;
                        let claimed = report.worker.clone();

                        match coordinator.deliver(report) {
                            Ok((attester, recorded)) => {
                                if let Recorded::AlreadyVoted { conflicting } = recorded {
                                    warn!(
                                        attester = %attest::hex_address(&attester),
                                        conflicting,
                                        "attester reported more than once; keeping the first"
                                    );
                                }
                                transport::respond(request, 200, b"ok");
                            }
                            // Rejections are logged rather than dropped: a
                            // worker whose report is being refused looks
                            // exactly like a worker that never reported, and
                            // the two have entirely different fixes (a
                            // misconfigured key, a job this coordinator never
                            // accepted, or a dead process).
                            Err(rejection) => {
                                warn!(
                                    job_id,
                                    claimed = %claimed,
                                    error = %rejection,
                                    "discarding report that was not counted"
                                );
                                let status = rejection.status();
                                transport::respond(
                                    request,
                                    status,
                                    rejection.to_string().as_bytes(),
                                );
                            }
                        }
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
///
/// Every argument belongs to one job — its inbox, its quorum, its dispatch set,
/// its deadline — which is why this needed no change when the coordinator
/// became a service. It was already a function of one job's state; there was
/// simply only ever one job's state to hand it.
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

/// Pairs each address in the winning group with the signature it voted with.
///
/// `tally` counts addresses, because agreement is over parties. A contract
/// cannot be counted at: it is handed the signatures and recovers the addresses
/// itself, which is the whole reason task 2.10i exists (`bridge.md` §2b). So
/// the group's membership is looked back up in the inbox, where the reports it
/// was counted from are still sitting.
///
/// Sorted ascending by address, and sorted *here* rather than trusted from
/// `group`: `fulfillJob` requires strictly increasing addresses so that
/// duplicate detection costs one comparison per attester instead of a set. Out
/// of order is not a warning there, it is a revert — and the revert names the
/// caller rather than the ordering, so a wrong sort would surface as a
/// coordinator that cannot settle anything for no visible reason.
///
/// Strictly increasing rather than merely sorted, because the inbox holds at
/// most one report per address and `group` pushes each key once, so the
/// addresses are already distinct before they are ordered.
fn attesters_of(inbox: &Inbox, attesters: &[Address]) -> Vec<Attester> {
    let reports = inbox.lock().expect("inbox poisoned");

    let mut group: Vec<Attester> = attesters
        .iter()
        .filter_map(|address| {
            reports.get(address).map(|report| Attester {
                address: *address,
                attestation: report.attestation.clone(),
            })
        })
        .collect();

    group.sort_unstable_by_key(|attester| attester.address);
    group
}

/// The winning group's signatures, as the JSON a contract caller is handed.
///
/// The shape is an interface rather than a convenience: the Anvil round trip
/// (task 3.3) reads this file and passes its `attesters` array to `fulfillJob`,
/// and the chain watcher (task 3.4) is the same submission with the file
/// replaced by a struct. Every choice here is one the other side depends on —
/// lowercase `0x` hex, `v` as a number rather than a string, and the array in
/// ascending address order, which `fulfillJob` requires and does not explain
/// when it reverts.
///
/// Written by hand rather than serialised, matching
/// `primitives/examples/attestation_vector.rs`, which produces the neighbouring
/// cross-language vector the same way. This is one flat object; a JSON
/// dependency added for it would show up in `Cargo.lock` beside a `tfhe` pin
/// that exists to keep the build reproducible.
fn attestations_json(
    job_id: u64,
    bytecode_hash: &[u8; 32],
    result_hash: &[u8; 32],
    attesters: &[Attester],
) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"jobId\": {job_id},\n"));
    out.push_str(&format!(
        "  \"bytecodeHash\": \"{}\",\n",
        bytecode::hex(bytecode_hash)
    ));
    out.push_str(&format!(
        "  \"resultHash\": \"{}\",\n",
        bytecode::hex(result_hash)
    ));
    out.push_str("  \"attesters\": [\n");

    for (index, attester) in attesters.iter().enumerate() {
        // No trailing comma after the last element: JSON has no tolerance for
        // one, and the consumer is a Solidity test harness rather than a
        // forgiving parser.
        let separator = if index + 1 == attesters.len() {
            ""
        } else {
            ","
        };
        out.push_str(&format!(
            "    {{ \"address\": \"{}\", \"r\": \"{}\", \"s\": \"{}\", \"v\": {} }}{separator}\n",
            attest::hex_address(&attester.address),
            bytecode::hex(&attester.attestation.r),
            bytecode::hex(&attester.attestation.s),
            attester.attestation.v,
        ));
    }

    out.push_str("  ]\n");
    out.push_str("}\n");
    out
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
        (
            worker(label).address(),
            evaluated_for(1, BYTECODE_HASH, label, marker),
        )
    }

    /// A signed report for a *named* job and program.
    ///
    /// The single-job helpers above pin job 1 and one bytecode hash, which is
    /// all a single-job coordinator could tell apart. Telling two jobs apart is
    /// the property under test now, so the routing tests need both to vary.
    fn evaluated_for(job_id: u64, bytecode_hash: [u8; 32], label: &str, marker: u8) -> JobReport {
        let key = worker(label);
        let sealed = sealed(marker);
        let attestation = key.attest(&Claim::Result {
            job_id,
            bytecode_hash,
            result_hash: sealed.hash,
        });

        JobReport {
            job_id,
            attestation,
            worker: label.to_string(),
            outcome: JobOutcome::Evaluated(sealed),
            elapsed_ms: 1,
        }
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
        Mutex::new(reports.into_iter().collect())
    }

    /// A verifier that accepts the named workers and nobody else.
    fn verifier(registered: &[&str]) -> Verifier {
        Verifier {
            job_id: 1,
            bytecode_hash: BYTECODE_HASH,
            registry: registered.iter().map(|l| worker(l).address()).collect(),
        }
    }

    /// A coordinator with no server key, which no test here serves.
    fn coordinator() -> Coordinator {
        Coordinator::new(Vec::new())
    }

    /// Registers a 2-of-3 job the way `accept_job` would, minus the socket and
    /// the fan-out.
    ///
    /// Goes through `Job::new` and `Coordinator::register`, so what these tests
    /// exercise is the job a real dispatch builds rather than a lookalike that
    /// could drift from it. What is skipped is dispatching to workers that do
    /// not exist in a unit test, and the `check_program` that precedes it.
    fn register(
        coordinator: &Coordinator,
        job_id: u64,
        bytecode_hash: [u8; 32],
        registered: &[&str],
    ) -> Arc<Job> {
        coordinator
            .register(Job::new(
                Verifier {
                    job_id,
                    bytecode_hash,
                    registry: registered.iter().map(|l| worker(l).address()).collect(),
                },
                2,
                3,
                Duration::from_secs(120),
            ))
            .expect("a job id nothing else is using")
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

    #[test]
    fn two_jobs_in_flight_settle_independently() {
        // The property task 3.4 needs and the one-shot coordinator did not have
        // (`next-architecture.md` §2.2). Under one process-global inbox this
        // fails and fails quietly: the inbox is keyed by attester address, so
        // w1's report for job 11 and w1's report for job 22 are the same slot,
        // first-write-wins keeps whichever landed first, and the other job ends
        // with one attestation instead of three. It never settles, and the log
        // says only that some workers did not report.
        let coordinator = coordinator();
        let first = register(&coordinator, 11, [0x77; 32], &["w1", "w2", "w3"]);
        let second = register(&coordinator, 22, [0x88; 32], &["w1", "w2", "w3"]);

        // Different programs and different answers, so neither job's reports
        // could be mistaken for the other's even by accident.
        let for_first: Vec<JobReport> = ["w1", "w2", "w3"]
            .iter()
            .map(|label| evaluated_for(11, [0x77; 32], label, 0xaa))
            .collect();
        let for_second: Vec<JobReport> = ["w1", "w2", "w3"]
            .iter()
            .map(|label| evaluated_for(22, [0x88; 32], label, 0xbb))
            .collect();

        // Delivered from two threads rather than in sequence, because "keyed by
        // job id" is a claim about concurrent access: a sequential test would
        // pass against a registry with no locking at all.
        let coordinator = &coordinator;
        thread::scope(|scope| {
            scope.spawn(move || {
                for report in for_first {
                    coordinator.deliver(report).expect("job 11 takes its own");
                }
            });
            scope.spawn(move || {
                for report in for_second {
                    coordinator.deliver(report).expect("job 22 takes its own");
                }
            });
        });

        assert_eq!(reported(&first.inbox), 3, "job 11 kept all three votes");
        assert_eq!(reported(&second.inbox), 3, "job 22 kept all three votes");

        let (first_result, first_attesters) = tally(&first.inbox, 2).expect("job 11 has a quorum");
        let (second_result, second_attesters) =
            tally(&second.inbox, 2).expect("job 22 has a quorum");

        assert_eq!(first_result.hash, sealed(0xaa).hash);
        assert_eq!(second_result.hash, sealed(0xbb).hash);
        assert_eq!(first_attesters.len(), 3);
        assert_eq!(second_attesters.len(), 3);
        assert_eq!(
            first_attesters, second_attesters,
            "the same three parties attested to both, which is exactly why one \
             inbox cannot hold both jobs"
        );
    }

    #[test]
    fn a_report_for_a_job_this_coordinator_never_accepted_is_refused_by_name() {
        // Not a panic, which would let any peer that can reach the port stop
        // the process, and not a silent drop, which is indistinguishable from a
        // worker that never reported.
        let coordinator = coordinator();
        register(&coordinator, 11, [0x77; 32], &["w1"]);

        let rejection = coordinator
            .deliver(evaluated_for(99, [0x77; 32], "w1", 0xaa))
            .expect_err("job 99 was never accepted here");

        assert_eq!(rejection, Rejection::UnknownJob(99));
        assert_eq!(
            rejection.status(),
            404,
            "not found rather than forbidden: the sender's key is fine, the job is not here"
        );
        assert!(
            rejection.to_string().contains("99"),
            "names the job, so a misrouted worker is diagnosable: {rejection}"
        );
    }

    #[test]
    fn an_attestation_for_the_other_job_in_flight_does_not_count_for_this_one() {
        // Routing puts this report in front of job 22's verifier, which
        // reconstructs the claim from *its* job id and *its* bytecode hash. The
        // signature was made over job 11's, so recovery returns an address that
        // is not w1's and the registry refuses it. With a process-global
        // verifier there is nothing to compare against and this is a vote — the
        // replay `next-architecture.md` §2.2 calls "harmless with one job in
        // flight; a replay bug the day there are two".
        let coordinator = coordinator();
        register(&coordinator, 11, [0x77; 32], &["w1", "w2", "w3"]);
        let second = register(&coordinator, 22, [0x88; 32], &["w1", "w2", "w3"]);

        let mut lifted = evaluated_for(11, [0x77; 32], "w1", 0xaa);
        lifted.job_id = 22;

        let rejection = coordinator
            .deliver(lifted)
            .expect_err("job 11's signature must not vote in job 22");

        assert_eq!(rejection.status(), 403);
        assert!(
            rejection.to_string().contains("not a registered worker"),
            "the claim recovered to a stranger: {rejection}"
        );
        assert_eq!(
            reported(&second.inbox),
            0,
            "nothing was recorded against job 22"
        );
    }

    #[test]
    fn a_job_id_already_in_flight_is_refused_rather_than_replacing_it() {
        // `fresh_job_id` makes this unlikely and an on-chain id makes it
        // impossible, but the failure it prevents is silent: the replaced job's
        // inbox goes with it, so every vote already cast disappears and the
        // first caller waits out its deadline against an inbox nobody is
        // filling.
        let coordinator = coordinator();
        let first = register(&coordinator, 11, [0x77; 32], &["w1"]);
        coordinator
            .deliver(evaluated_for(11, [0x77; 32], "w1", 0xaa))
            .unwrap();

        let error = coordinator
            .register(Job::new(
                Verifier {
                    job_id: 11,
                    bytecode_hash: [0x88; 32],
                    registry: HashSet::new(),
                },
                2,
                3,
                Duration::from_secs(1),
            ))
            // Discarded rather than named: `Job` has no `Debug`, because a
            // derived one would print the registry and the inbox on every
            // failed assertion.
            .map(|_| ())
            .expect_err("job 11 is already running");

        assert!(error.contains("11"), "names the job: {error}");
        assert_eq!(reported(&first.inbox), 1, "the running job kept its vote");
    }

    #[test]
    fn collecting_a_job_twice_is_refused_rather_than_splitting_its_wakeups() {
        // A `Receiver` has one consumer. Two collectors would each see half the
        // arrivals, and each would conclude the workers it missed had gone
        // quiet — a job that fails its deadline with a full inbox.
        let coordinator = coordinator();
        register(&coordinator, 11, [0x77; 32], &["w1", "w2", "w3"]);
        for label in ["w1", "w2", "w3"] {
            coordinator
                .deliver(evaluated_for(11, [0x77; 32], label, 0xaa))
                .unwrap();
        }

        assert!(matches!(
            coordinator.settle(11).unwrap(),
            Outcome::Settled(_)
        ));
        let error = coordinator.settle(11).expect_err("collected once already");
        assert!(error.contains("already being collected"), "got: {error}");

        let error = coordinator.settle(12).expect_err("never accepted");
        assert!(error.contains("12"), "names the job: {error}");
    }

    #[test]
    fn a_settled_job_remembers_what_it_decided() {
        // The outcome lives on the job rather than only in `settle`'s return
        // value, because a watcher that has to resubmit a `fulfillJob`
        // transaction (task 3.4) needs the settlement without re-collecting.
        let coordinator = coordinator();
        let job = register(&coordinator, 11, [0x77; 32], &["w1", "w2", "w3"]);
        assert_eq!(*job.outcome.lock().unwrap(), Outcome::Collecting);

        for label in ["w1", "w2", "w3"] {
            coordinator
                .deliver(evaluated_for(11, [0x77; 32], label, 0xaa))
                .unwrap();
        }

        let settled = coordinator.settle(11).unwrap();
        assert_eq!(*job.outcome.lock().unwrap(), settled);

        let Outcome::Settled(settlement) = settled else {
            panic!("three agreeing attesters is a quorum");
        };
        assert_eq!(settlement.result.hash, sealed(0xaa).hash);
    }

    #[test]
    fn a_job_nobody_could_settle_is_recorded_as_unsettled_rather_than_left_open() {
        let coordinator = coordinator();
        let job = register(&coordinator, 11, [0x77; 32], &["w1", "w2", "w3"]);

        // Three parties, three different answers: agreement has stopped being
        // possible, so this returns immediately rather than at the deadline.
        for (label, marker) in [("w1", 0xaa), ("w2", 0xbb), ("w3", 0xcc)] {
            coordinator
                .deliver(evaluated_for(11, [0x77; 32], label, marker))
                .unwrap();
        }

        assert_eq!(coordinator.settle(11).unwrap(), Outcome::Unsettled);
        assert_eq!(*job.outcome.lock().unwrap(), Outcome::Unsettled);
    }

    #[test]
    fn the_exported_attestation_set_is_what_fulfilljob_takes() {
        // The file `bridge/script` and the Anvil round trip (task 3.3) read.
        // Three things have to hold, and only the first is visible in a diff:
        // the addresses ascend, every signature is the one that attester
        // actually sent, and each recovers to the address it is printed beside.
        // `fulfillJob` checks all three and reverts without saying which.
        let coordinator = coordinator();
        let job = register(&coordinator, 11, [0x77; 32], &["w1", "w2", "w3"]);
        for label in ["w1", "w2", "w3"] {
            coordinator
                .deliver(evaluated_for(11, [0x77; 32], label, 0xaa))
                .unwrap();
        }

        let (result, attesters) = tally(&job.inbox, 2).expect("three agreed");
        let group = attesters_of(&job.inbox, &attesters);
        assert_eq!(group.len(), 3);

        for pair in group.windows(2) {
            assert!(
                pair[0].address < pair[1].address,
                "strictly increasing, or `fulfillJob`'s O(n) duplicate check \
                 rejects an honest set: {} then {}",
                attest::hex_address(&pair[0].address),
                attest::hex_address(&pair[1].address)
            );
        }

        // The signature is the evidence, so it must recover to the address it
        // is exported beside — over this job's id, this program and this
        // result, which is the digest the contract reconstructs.
        for attester in &group {
            let claim = Claim::Result {
                job_id: 11,
                bytecode_hash: [0x77; 32],
                result_hash: result.hash,
            };
            assert_eq!(
                attest::recover(&claim, &attester.attestation).unwrap(),
                attester.address,
                "an exported signature that does not recover to its own address \
                 would revert on-chain and look like a dishonest worker"
            );
        }

        let json = attestations_json(11, &[0x77; 32], &result.hash, &group);
        assert!(json.contains("\"jobId\": 11,"), "{json}");
        assert!(
            json.contains(&format!(
                "\"resultHash\": \"{}\"",
                bytecode::hex(&result.hash)
            )),
            "{json}"
        );
        for attester in &group {
            assert!(
                json.contains(&format!(
                    "\"address\": \"{}\"",
                    attest::hex_address(&attester.address)
                )),
                "every attester appears: {json}"
            );
        }
        assert!(
            !json.contains("},\n  ]"),
            "no trailing comma; the consumer is a Solidity harness, not a \
             forgiving parser: {json}"
        );
        assert!(
            json.contains(&format!("\"v\": {}", group[0].attestation.v)),
            "v is a number, as `ecrecover` takes it: {json}"
        );
    }

    #[test]
    fn the_exported_order_does_not_depend_on_the_order_reports_arrived() {
        // Arrival order is a network fact and the ordering `fulfillJob` demands
        // is a calldata fact. Two runs of one job that settle on the same result
        // must produce the same file, or a resubmission looks like a different
        // settlement.
        let forwards = coordinator();
        let a = register(&forwards, 11, [0x77; 32], &["w1", "w2", "w3"]);
        for label in ["w1", "w2", "w3"] {
            forwards
                .deliver(evaluated_for(11, [0x77; 32], label, 0xaa))
                .unwrap();
        }

        let backwards = coordinator();
        let b = register(&backwards, 11, [0x77; 32], &["w1", "w2", "w3"]);
        for label in ["w3", "w2", "w1"] {
            backwards
                .deliver(evaluated_for(11, [0x77; 32], label, 0xaa))
                .unwrap();
        }

        let (_, one) = tally(&a.inbox, 2).unwrap();
        let (_, other) = tally(&b.inbox, 2).unwrap();
        assert_eq!(attesters_of(&a.inbox, &one), attesters_of(&b.inbox, &other));
    }

    #[test]
    fn an_accepted_job_is_dispatched_and_registered_under_the_id_it_returns() {
        // The one place the whole entry point runs: validation, the minted id,
        // the registry entry, and a fan-out to workers that are not there.
        // Port 1 on loopback refuses immediately, so an unreachable worker is a
        // logged failure rather than a hang — the same path a dead worker takes.
        let coordinator = coordinator();
        let registry: Vec<Address> = ["w1", "w2", "w3"]
            .iter()
            .map(|l| worker(l).address())
            .collect();

        let spec = || JobSpec {
            job_id: None,
            workers: vec![
                "127.0.0.1:1".into(),
                "127.0.0.1:1".into(),
                "127.0.0.1:1".into(),
            ],
            registry: registry.clone(),
            attesters: 2,
            bytecode: tally_bytecode(),
            function: "tally4_select".into(),
            inputs: vec![vec![1], vec![2], vec![3], vec![4]],
            deadline: Duration::from_millis(1),
        };

        let first = coordinator.accept_job(spec()).unwrap();
        let second = coordinator.accept_job(spec()).unwrap();

        assert_ne!(
            first, second,
            "two jobs accepted by one coordinator must not share an id"
        );
        assert!(coordinator.job(first).is_some());
        assert!(coordinator.job(second).is_some());

        // And the id the watcher will supply is honoured rather than replaced.
        let mut chain_job = spec();
        chain_job.job_id = Some(4242);
        assert_eq!(coordinator.accept_job(chain_job).unwrap(), 4242);
        assert_eq!(
            coordinator.job(4242).unwrap().verifier.job_id,
            4242,
            "the verifier has to bind the id the job was accepted under, or \
             every report for it recovers to a stranger"
        );
    }

    #[test]
    fn an_impossible_job_is_refused_before_anything_is_dispatched() {
        // The checks that used to live at the top of `run`. They belong on the
        // entry point, not on one caller of it: the watcher gets its quorum and
        // its registry from a chain and can be handed a job just as impossible.
        let coordinator = coordinator();
        let base = || JobSpec {
            job_id: None,
            workers: vec![
                "127.0.0.1:1".into(),
                "127.0.0.1:1".into(),
                "127.0.0.1:1".into(),
            ],
            registry: ["w1", "w2", "w3"]
                .iter()
                .map(|l| worker(l).address())
                .collect(),
            attesters: 2,
            bytecode: tally_bytecode(),
            function: "tally4_select".into(),
            inputs: vec![vec![1], vec![2], vec![3], vec![4]],
            deadline: Duration::from_millis(1),
        };

        let mut minority = base();
        minority.workers.push("127.0.0.1:1".into());
        let error = coordinator.accept_job(minority).unwrap_err();
        assert!(error.contains("majority"), "2-of-4 is refused: {error}");

        let mut short_registry = base();
        short_registry.registry.truncate(1);
        let error = coordinator.accept_job(short_registry).unwrap_err();
        assert!(error.contains("registered worker"), "got: {error}");

        let mut wrong_arity = base();
        wrong_arity.inputs.pop();
        let error = coordinator.accept_job(wrong_arity).unwrap_err();
        assert!(error.contains("4 input(s)"), "got: {error}");

        let mut missing_function = base();
        missing_function.function = "tally5_select".into();
        let error = coordinator.accept_job(missing_function).unwrap_err();
        assert!(error.contains("tally5_select"), "got: {error}");

        assert!(
            coordinator.jobs.lock().unwrap().is_empty(),
            "a refused job must not be left in the registry, where a report \
             would be counted against a job nobody is collecting"
        );
    }

    const TALLY_WASM: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../committee-tally/committee_tally.wasm"
    );

    /// The tally program as `disca-cli compile` would hand it over.
    fn tally_bytecode() -> Vec<u8> {
        use primitives::program::{DiscaProgram, Program};
        let wasm = std::fs::read(TALLY_WASM).expect("the committed demo circuit");
        let program = DiscaProgram::from_program(&Program::from_wasm(&wasm).unwrap());
        bytecode::serialize(&program).unwrap()
    }

    #[test]
    fn a_program_blob_is_checked_before_a_worker_is_asked_to_run_it() {
        // The coordinator no longer compiles -- `disca-cli compile` does -- but
        // it is still the last party that can fail cheaply. Every worker will
        // decode this same blob, so a blob that does not decode here would fan
        // out and fail on all of them at once.
        let arity = check_program(&tally_bytecode(), "tally4_select").unwrap();
        assert_eq!(arity, 4, "the caller checks its input count against this");
    }

    #[test]
    fn checking_refuses_a_function_the_program_does_not_export() {
        // Cheap here, expensive later: this runs before any dispatch, so a typo
        // costs a millisecond instead of three workers and a deadline.
        let error = check_program(&tally_bytecode(), "tally5_select").unwrap_err();
        assert!(error.contains("tally5_select"), "names it: {error}");
    }

    #[test]
    fn checking_refuses_bytes_that_are_not_bytecode() {
        // The blob arrives as a file path from the command line, so "not
        // bytecode at all" is a typo away and must be an error rather than
        // something a worker discovers.
        let error = check_program(b"not bytecode", "tally4_select").unwrap_err();
        assert!(!error.is_empty(), "the rejection has to say something");
    }
}
