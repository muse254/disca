//! The chain watcher: take jobs from `DiscaBridge`, settle them back to it.
//!
//! Task 3.4, and step 3 of `docs/bridge.md` §8. What `scripts/run-anvil.sh`
//! does by hand with `cast` — read `JobRequested`, run the job, send
//! `fulfillJob` — this does as a process. The shell script is the
//! specification; the point of this module is to make its `cast send
//! fulfillJob` step unnecessary.
//!
//! # What this is, next to the `coordinator` role
//!
//! Both roles run the same job service. [`crate::coordinator::run`] accepts one
//! job from argv, blocks on it, writes the result to a file and exits; this
//! accepts every job a contract posts, keeps several in flight, and writes the
//! result to a transaction. `Coordinator::accept_job` is the seam both sit on
//! (`next-architecture.md` §4 step 3), and the only difference between the two
//! callers is where a job's id, inputs and quorum come from — argv there, the
//! chain here.
//!
//! It is a separate subcommand rather than chain flags on `coordinator` because
//! the two have disjoint arguments and opposite lifetimes. A `coordinator` with
//! `--rpc` would have to make `--input`, `--job-id`, `--result` and
//! `--attestations` conditionally forbidden and `--bridge` conditionally
//! required, and its `--help` would then describe two programs. Keeping them
//! apart is also what guarantees the chainless path is untouched: nothing in
//! this file is reachable from `node coordinator`, so `scripts/run-local.sh`
//! cannot be broken by anything here.
//!
//! # The commitment check is adversarial here for the first time
//!
//! [`verify_inputs`] is task 2.9f. Task 2.9e recorded why the worker's own
//! check ([`crate::worker`]) is diagnostic rather than adversarial: the
//! commitment travels in the same dispatch as the bytes it commits to, so a
//! coordinator that substitutes an input simply recomputes the commitment
//! beside it. It detects corruption, not a lying sender.
//!
//! From the chain that stops being true. The commitment is written by
//! `submitJob`, which hashes each blob itself before storing it
//! (`DiscaBridge.sol`, `InputCommitmentMismatch`), and it is *storage* — the
//! same consensus state every node agrees on. The blobs are event data, which
//! is not. So this check compares bytes an RPC endpoint handed over against a
//! hash the chain is holding, and neither the poster, nor the relay, nor the
//! node's own operator can move both sides. That is the first commitment check
//! in this system that an adversary cannot satisfy by recomputation, and it is
//! the reason the worker's copy of it is worth keeping: the worker checks the
//! dispatch it was given, this checks the dispatch it is about to give.
//!
//! # Reorgs and restarts
//!
//! Handled: a `--confirmations` delay, so nothing is dispatched until the
//! chain has buried it; and a `jobs(jobId).state` read before dispatch and a
//! `fulfillJob` that reverts on a job that is no longer `Open`, so a job
//! settled by an earlier run of this watcher is skipped rather than settled
//! twice. There is no cursor on disk on purpose — a stale cursor is a way to
//! miss jobs, and the chain already holds the only authoritative answer to
//! "which jobs still need work". A restart rescans from `--from-block` and the
//! state check filters out everything already done.
//!
//! **Not handled**, and worth stating rather than discovering:
//!
//! * **A reorg deeper than `--confirmations`.** A job dispatched and then
//!   reorganised out has already cost worker time, and if the same id is
//!   re-mined for a *different* job the attestations this watcher collected
//!   still bind that id and that program hash — so a `fulfillJob` for the old
//!   job could settle the new one, if the two happen to share a program. The
//!   confirmations delay makes it unlikely and nothing here makes it
//!   impossible. The fix is a per-job record of the block hash the event came
//!   from, re-read before submitting; it is not built because there is no
//!   chain in this project's targets (`bridge.md` §7 — Anvil and an L2) where
//!   a reorg past even one confirmation is a live concern.
//! * **A dropped or replaced `fulfillJob` transaction.** It is sent once and
//!   waited on; if it is evicted from the mempool the job is not resubmitted,
//!   and it expires into `refundOnTimeout` like any other liveness failure
//!   (`bridge.md` §6). The settlement is kept on the job
//!   ([`crate::coordinator::Outcome`]) so a resubmission would not have to
//!   re-collect, which is what that field was put there for.
//! * **Unbounded concurrency.** One thread per job, and nothing caps how many
//!   jobs a chain can post. Each thread is almost entirely blocked — the FHE
//!   work is on the workers — but a chain that posted ten thousand jobs would
//!   still spawn ten thousand threads.
//!
//! # What it deliberately does not do
//!
//! There is no dispute path. A job that reaches no quorum is left alone, and
//! its escrow goes back to the poster through `refundOnTimeout`. That is not a
//! gap: `JobState.Disputed` is unreachable by design (`bridge.md` §6 row 2,
//! task 3.5) because divergence is invisible on-chain — a worker that disagrees
//! contributes nothing to a quorum, and the contract cannot tell that from a
//! worker being slow. Inventing a transaction to send here would be inventing a
//! claim the contract has no way to check.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use alloy::network::TransactionBuilder;
use alloy::primitives::{Address as EthAddress, B256, Bytes, U256};
use alloy::providers::{DynProvider, Provider, ProviderBuilder};
use alloy::rpc::types::{Filter, TransactionRequest};
use alloy::signers::local::PrivateKeySigner;
use alloy::sol;
use alloy::sol_types::{SolCall, SolEvent};
use primitives::attest::{self, Address};
use primitives::{bytecode, wire};
use tokio::runtime::{Handle, Runtime};
use tracing::{info, info_span, warn};

use crate::coordinator::{Attester, Coordinator, JobSpec, Outcome, quorum_error};

// The slice of `IDiscaBridge` this watcher speaks, transcribed from
// `bridge/src/IDiscaBridge.sol`. Only the four members it uses are here, and
// they are copied rather than generated from the Foundry artefacts on purpose:
// a build that shells out to `forge` to produce Rust would make `cargo check`
// depend on a Solidity toolchain, and every one of these signatures is checked
// against the real contract by `scripts/run-anvil.sh --watcher` on every run.
//
// `Job.state` is `uint8` rather than the `JobState` enum: the ABI encoding is
// the same and `sol!` would otherwise need the enum declared for a value this
// module only ever compares against `Open`.
sol! {
    #[derive(Debug)]
    event JobRequested(
        uint256 indexed jobId,
        uint256 indexed programId,
        bytes32[] inputCommits,
        bytes[] inputBlobs,
        address callback
    );

    struct Attestation {
        bytes32 r;
        bytes32 s;
        uint8 v;
    }

    struct Job {
        uint256 programId;
        address poster;
        address callback;
        bytes32[] inputCommits;
        uint64 deadline;
        uint256 escrow;
        uint8 state;
    }

    function fulfillJob(
        uint256 jobId,
        bytes32 resultHash,
        bytes resultBlob,
        Attestation[] attestations
    ) external;

    function jobs(uint256 jobId) external view returns (Job memory);

    function programs(uint256 programId)
        external
        view
        returns (bytes32 bytecodeHash, bytes32 serverKeyHash, uint8 attestersRequired);

    function isRegisteredWorker(address worker) external view returns (bool);

    function coordinator() external view returns (address);
}

/// `IDiscaBridge.JobState.Open`. Named because the numbering is load-bearing —
/// `None` is reserved as zero so an unwritten mapping entry cannot read as a
/// real job — and a bare `1` in a comparison says none of that.
const JOB_STATE_OPEN: u8 = 1;

/// How many blocks one `eth_getLogs` may span.
///
/// A first scan from genesis on anything but a fresh Anvil would otherwise ask
/// for the whole chain in one call, which public endpoints refuse (usually with
/// a range limit in the error, sometimes with a timeout). Chunking makes the
/// first poll slower and every poll after it a no-op.
const MAX_SCAN_SPAN: u64 = 2_000;

pub struct Config {
    /// JSON-RPC endpoint. **`http://` only** — see the `alloy` features in the
    /// workspace manifest; [`run`] refuses `https://` at startup rather than
    /// letting it arrive as a transport error.
    pub rpc: String,
    /// The deployed `DiscaBridge`.
    pub bridge: EthAddress,
    /// The key that signs `fulfillJob`. Must be the address the bridge holds as
    /// `coordinator()`, which [`run`] checks before waiting for a single event:
    /// the alternative is discovering it as a `NotCoordinator` revert after a
    /// job has already been evaluated by three workers.
    pub coordinator_key: String,
    /// Where the job service listens: the server key by hash, and worker
    /// reports. Same surface as the `coordinator` role.
    pub bind: String,
    pub workers: Vec<String>,
    /// Addresses whose attestations count. Checked against the chain's
    /// `isRegisteredWorker` at startup — see [`run`].
    pub registry: Vec<Address>,
    /// The registered program this watcher runs. Events for any other
    /// `programId` are logged and skipped.
    pub program_id: U256,
    /// DISCA bytecode, from `disca-cli compile`. Its hash must equal the
    /// `bytecodeHash` `registerProgram` pinned.
    pub bytecode: Vec<u8>,
    /// Which exported function of that program a job runs.
    pub function: String,
    /// The compressed server key, from `disca-cli keygen`. Its hash must equal
    /// the `serverKeyHash` `registerProgram` pinned.
    pub server_key: Vec<u8>,
    /// How long to wait for worker agreement on one job.
    pub deadline: Duration,
    /// How many blocks to let a job age before dispatching it.
    pub confirmations: u64,
    /// How long to wait between `eth_getLogs` calls.
    pub poll: Duration,
    /// First block to scan.
    pub from_block: u64,
}

/// Everything a job thread needs to talk to the chain.
///
/// Cloned per job rather than shared behind a lock: [`DynProvider`] is a
/// reference-counted handle over one HTTP client, so two jobs settling at the
/// same time queue on the transport rather than on this process.
#[derive(Clone)]
struct Chain {
    provider: DynProvider,
    bridge: EthAddress,
    /// The runtime the provider's futures run on. Held as a [`Handle`] so a job
    /// thread can block on a chain call without this module having to make the
    /// rest of the node async.
    runtime: Handle,
}

impl Chain {
    /// An `eth_call` against the bridge, decoded as the call's return type.
    ///
    /// Every read in this module goes through here so that "the address has no
    /// code" and "the view reverted" are one error shape with the function
    /// named in it, rather than an ABI decode failure on an empty buffer —
    /// which is what a wrong `--bridge` produces and is close to unreadable.
    fn read<C: SolCall>(&self, call: C) -> Result<C::Return, String> {
        let request = TransactionRequest::default()
            .with_to(self.bridge)
            .with_input(Bytes::from(call.abi_encode()));

        let returned = self
            .runtime
            .block_on(async { self.provider.call(request).await })
            .map_err(|e| format!("{} on {} failed: {e}", C::SIGNATURE, self.bridge))?;

        C::abi_decode_returns(&returned).map_err(|e| {
            format!(
                "{} on {} returned {} byte(s) that do not decode: {e} — is that a \
                 DiscaBridge?",
                C::SIGNATURE,
                self.bridge,
                returned.len()
            )
        })
    }
}

/// Runs until killed: scan, dispatch, settle, repeat.
pub fn run(config: Config) -> Result<(), String> {
    // Checked here rather than left to the transport. Without
    // `alloy/reqwest-rustls-tls` (see the workspace manifest for what that
    // costs and why it is off) an `https://` URL fails somewhere inside
    // reqwest with an error about a builder, which reads as a bug in this
    // program rather than as a missing feature.
    if config.rpc.starts_with("https://") {
        return Err(format!(
            "{} is https, and this build has no TLS transport. Either point \
             --rpc at an http endpoint (Anvil, a local node, an ssh tunnel) or \
             rebuild with the `reqwest-rustls-tls` feature of `alloy` enabled \
             in the workspace manifest.",
            config.rpc
        ));
    }

    let signer: PrivateKeySigner = config
        .coordinator_key
        .parse()
        .map_err(|e| format!("--coordinator-key is not a secp256k1 private key: {e}"))?;
    let submitter = signer.address();

    let url = config
        .rpc
        .parse()
        .map_err(|e| format!("--rpc is not a URL: {e}"))?;

    // One runtime for the process. `rt-multi-thread` because the poll loop and
    // every job thread block on it at once; a current-thread runtime would let
    // a settlement in progress stop the loop from ever scanning again.
    let runtime = Runtime::new().map_err(|e| format!("cannot start a tokio runtime: {e}"))?;
    let chain = Chain {
        provider: ProviderBuilder::new()
            .wallet(signer)
            .connect_http(url)
            .erased(),
        bridge: config.bridge,
        runtime: runtime.handle().clone(),
    };

    // --- everything that can be wrong about this deployment, before any job ---
    //
    // All four of these surface as a revert or a silence *after* three workers
    // have spent seconds on FHE, and each of them reverts with an error that
    // points somewhere else: a wrong key is `NotCoordinator`, an unregistered
    // worker is `NotRegisteredWorker` for a worker that looks registered, and a
    // bytecode hash that does not match what `registerProgram` pinned produces
    // signatures over a digest the contract never builds — which lands as
    // `NotRegisteredWorker` too.

    let on_chain_coordinator = chain.read(coordinatorCall {})?;
    if on_chain_coordinator != submitter {
        return Err(format!(
            "--coordinator-key is {submitter}, but {} holds {on_chain_coordinator} as its \
             coordinator; only that address may call fulfillJob (bridge.md §2)",
            config.bridge
        ));
    }

    let program = chain.read(programsCall {
        programId: config.program_id,
    })?;
    let attesters = check_registered_program(
        &config.bytecode,
        &config.server_key,
        program.bytecodeHash,
        program.serverKeyHash,
        program.attestersRequired,
    )?;

    if let Some(error) = quorum_error(attesters, config.workers.len()) {
        return Err(format!(
            "program {} requires {attesters} attester(s) on-chain: {error}",
            config.program_id
        ));
    }

    for address in &config.registry {
        let worker = EthAddress::from(*address);
        if !chain.read(isRegisteredWorkerCall { worker })? {
            return Err(format!(
                "--registered-worker {worker} is not registered on {}; its \
                 attestations would be counted here and rejected there, and \
                 `fulfillJob` reports that as NotRegisteredWorker rather than as \
                 a misconfigured watcher",
                config.bridge
            ));
        }
    }

    info!(
        rpc = %config.rpc,
        bridge = %config.bridge,
        coordinator = %submitter,
        program_id = %config.program_id,
        function = %config.function,
        attesters,
        workers = config.workers.len(),
        registered = config.registry.len(),
        confirmations = config.confirmations,
        from_block = config.from_block,
        "watching"
    );

    let coordinator = Arc::new(Coordinator::new(config.server_key));
    // Up before the first event, for the same reason `coordinator::run` binds
    // before it accepts: a dispatch is immediately followed by workers pulling
    // `/keys/<hash>` from this process.
    crate::coordinator::serve(&config.bind, coordinator.clone())?;

    // Nothing below returns `Err`. A watcher that exits on a failed poll takes
    // every job currently evaluating down with it — the threads are this
    // process's — and those jobs then expire into a refund because their
    // settlement never gets sent. Every configuration error that this endpoint
    // could have has already been raised above, by four `eth_call`s made before
    // a single job was accepted, so a failure from here on is a transient one
    // and the right response to a transient failure is the next poll. It is
    // logged at `warn` every time rather than counted or backed off, because a
    // watcher that has gone quiet about a chain it cannot reach is the failure
    // this whole role exists to avoid.
    let mut next = config.from_block;
    loop {
        let head = match chain.runtime.block_on(chain.provider.get_block_number()) {
            Ok(head) => head,
            Err(error) => {
                warn!(rpc = %config.rpc, %error, "cannot read the block number; retrying");
                thread::sleep(config.poll);
                continue;
            }
        };

        let Some((from, to)) = scan_range(next, head, config.confirmations, MAX_SCAN_SPAN) else {
            thread::sleep(config.poll);
            continue;
        };

        let filter = Filter::new()
            .address(config.bridge)
            .event_signature(JobRequested::SIGNATURE_HASH)
            .from_block(from)
            .to_block(to);

        let logs = match chain.runtime.block_on(chain.provider.get_logs(&filter)) {
            Ok(logs) => logs,
            // `next` is deliberately not advanced: the range is asked for again
            // on the next poll. A watcher that skipped a range it failed to read
            // would lose every job posted in it, permanently and silently.
            Err(error) => {
                warn!(from, to, %error, "eth_getLogs failed; retrying the same range");
                thread::sleep(config.poll);
                continue;
            }
        };

        for log in logs {
            let event = match JobRequested::decode_log_data(&log.inner.data) {
                Ok(event) => event,
                // A log that matched the topic and did not decode is either a
                // different contract at this address or an ABI that has moved
                // out from under this file. Neither is a reason to stop
                // watching, and both need saying.
                Err(error) => {
                    warn!(block = ?log.block_number, %error, "skipping a JobRequested that did not decode");
                    continue;
                }
            };

            let coordinator = coordinator.clone();
            let chain = chain.clone();
            let job = PendingJob {
                event,
                program_id: config.program_id,
                bytecode: config.bytecode.clone(),
                function: config.function.clone(),
                workers: config.workers.clone(),
                registry: config.registry.clone(),
                attesters,
                deadline: config.deadline,
            };

            // A thread per job. `accept_job` returns as soon as a job is
            // dispatched and `settle` blocks until it is decided, so doing both
            // on the poll loop would stop the watcher scanning for the length
            // of every job — and `JobRequested` events arrive while a job is
            // evaluating, which is the case the job service was built for
            // (`next-architecture.md` §2.2).
            //
            // The handle is dropped rather than joined: one job failing is a
            // job that will be refunded, not a reason for this process to stop
            // watching, and `handle_job` logs every path it can take.
            thread::spawn(move || {
                let job_id = job.event.jobId;
                let span = info_span!("chain_job", %job_id);
                let _enter = span.enter();
                if let Err(error) = handle_job(&coordinator, &chain, job) {
                    warn!(%error, "job not settled; its escrow refunds on timeout (bridge.md §6)");
                }
            });
        }

        next = to + 1;
    }
}

/// One `JobRequested` and the local artefacts it will be run with.
///
/// A struct rather than eight arguments because every one of them is cloned per
/// job and the compiler is better than a reviewer at noticing a missing one.
struct PendingJob {
    event: JobRequested,
    program_id: U256,
    bytecode: Vec<u8>,
    function: String,
    workers: Vec<String>,
    registry: Vec<Address>,
    attesters: usize,
    deadline: Duration,
}

/// Verify, dispatch, wait, submit. One job, start to finish, on its own thread.
fn handle_job(coordinator: &Coordinator, chain: &Chain, job: PendingJob) -> Result<(), String> {
    let event = &job.event;

    // Not an error and not a warning: a bridge fronts many programs and this
    // watcher holds the bytecode for one of them. Somebody else's job is the
    // normal case.
    if event.programId != job.program_id {
        info!(
            program_id = %event.programId,
            ours = %job.program_id,
            "skipping a job for another program"
        );
        return Ok(());
    }

    let job_id = job_id_of(event.jobId)?;

    // The chain's own copy of the job, which is where three things come from
    // that the event cannot be trusted for on its own: the stored commitments
    // (see `verify_inputs`), the state, and the deadline.
    let on_chain = chain.read(jobsCall { jobId: event.jobId })?;
    if on_chain.state != JOB_STATE_OPEN {
        // The restart path. A rescan from `--from-block` re-delivers every job
        // this watcher ever settled, and each of them is filtered out here
        // rather than re-run and re-submitted into a `JobNotOpen` revert.
        info!(
            state = on_chain.state,
            "skipping a job that is no longer open"
        );
        return Ok(());
    }

    let inputs = verify_inputs(
        &event.inputCommits,
        &event.inputBlobs,
        &on_chain.inputCommits,
    )?;

    info!(
        inputs = inputs.len(),
        callback = %event.callback,
        escrow_wei = %on_chain.escrow,
        deadline = on_chain.deadline,
        "accepting a job from the chain"
    );

    // `Some(job_id)`, and this is the whole of task 2.9f's other half. A worker
    // signs a digest binding the job id and `fulfillJob` rebuilds that digest
    // from the id `submitJob` assigned, so a coordinator minting its own
    // produces signatures that recover to addresses no registry holds — and the
    // contract rejects a settlement that is correct in every other respect,
    // with `NotRegisteredWorker`, for workers that are registered.
    let accepted = coordinator.accept_job(JobSpec {
        job_id: Some(job_id),
        workers: job.workers,
        registry: job.registry,
        attesters: job.attesters,
        bytecode: job.bytecode,
        function: job.function,
        inputs,
        deadline: job.deadline,
    })?;

    match coordinator.settle(accepted)? {
        Outcome::Settled(settlement) => {
            let tx = submit_fulfillment(
                chain,
                event.jobId,
                &settlement.result,
                &settlement.attesters,
            )?;
            info!(
                result_hash = %bytecode::hex(&settlement.result.hash),
                result_bytes = settlement.result.blob.len(),
                attesters = ?settlement
                    .attesters
                    .iter()
                    .map(|attester| attest::hex_address(&attester.address))
                    .collect::<Vec<_>>(),
                %tx,
                "job settled on-chain"
            );
            Ok(())
        }
        // No quorum, so nothing to submit and nothing to dispute. `bridge.md`
        // §6 routes every liveness failure — a silent coordinator, a withheld
        // result, workers that never agree — to `refundOnTimeout`, because none
        // of them produce a quorum and the contract cannot tell them apart.
        // Sending anything here would be asserting something the contract has
        // no way to check.
        Outcome::Unsettled | Outcome::Collecting => Err(format!(
            "no quorum of {} within {:?}",
            job.attesters, job.deadline
        )),
    }
}

/// Sends `fulfillJob` and waits for it to be mined, returning its hash.
fn submit_fulfillment(
    chain: &Chain,
    job_id: U256,
    result: &wire::SealedResult,
    attesters: &[Attester],
) -> Result<B256, String> {
    let call = fulfillJobCall {
        jobId: job_id,
        resultHash: B256::from(result.hash),
        resultBlob: Bytes::from(result.blob.clone()),
        attestations: attestations_for(attesters),
    };

    let request = TransactionRequest::default()
        .with_to(chain.bridge)
        .with_input(Bytes::from(call.abi_encode()));

    let receipt = chain
        .runtime
        .block_on(async {
            chain
                .provider
                .send_transaction(request)
                .await?
                .get_receipt()
                .await
        })
        .map_err(|e| format!("fulfillJob did not land: {e}"))?;

    // A reverted transaction is mined, so the send succeeds and only the
    // receipt says otherwise. Reporting success here would leave the job
    // heading for a refund while the log says it settled.
    if !receipt.status() {
        return Err(format!(
            "fulfillJob reverted in {} ({} gas)",
            receipt.transaction_hash, receipt.gas_used
        ));
    }

    Ok(receipt.transaction_hash)
}

/// The winning group as `fulfillJob` takes it.
///
/// Order is carried through untouched: `attesters_of` in
/// [`crate::coordinator`] sorts ascending because `fulfillJob` requires
/// strictly increasing recovered addresses, and re-sorting here would put the
/// same rule in two places where only one of them is tested against the
/// contract.
fn attestations_for(attesters: &[Attester]) -> Vec<Attestation> {
    attesters
        .iter()
        .map(|attester| Attestation {
            r: B256::from(attester.attestation.r),
            s: B256::from(attester.attestation.s),
            v: attester.attestation.v,
        })
        .collect()
}

/// The job id as the protocol carries it.
///
/// `submitJob` returns a `uint256` and `fulfillJob` casts it to `uint64` to
/// rebuild the digest, so this conversion has to agree with that cast rather
/// than merely succeed. Refused rather than truncated: a truncating watcher
/// would have its workers sign under an id the event does not name, which
/// recovers to nothing the registry holds. It cannot happen — `jobCount` is a
/// counter incremented once per job — which is exactly why an error is
/// affordable here.
fn job_id_of(job_id: U256) -> Result<u64, String> {
    u64::try_from(job_id).map_err(|_| {
        format!(
            "job id {job_id} does not fit in the 8-byte field the attestation \
             preimage reserves for it (bridge.md §2a)"
        )
    })
}

/// Checks the blobs an event carried against the commitments the chain holds.
///
/// **Task 2.9f.** Two comparisons, and they buy different things:
///
/// * Each blob against the commitment beside it, which is what makes the bytes
///   about to be dispatched the bytes the poster paid to have computed on.
/// * The event's commitments against the ones in contract *storage*. The blobs
///   and the commitments in a log are both event data, so on their own they
///   prove only that whoever wrote the log was consistent; storage is consensus
///   state. An endpoint that fabricated a `JobRequested` — or replayed a real
///   one from a chain this watcher is not on — has to also fabricate the job it
///   names, and `jobs(jobId)` is the read that catches it.
///
/// This is the check `docs/tasks.md` 2.9e says the worker's copy is not.
/// There, the commitment travels in the same message as the bytes, so a
/// dishonest sender recomputes it and the check detects corruption rather than
/// substitution. Here the two sides come from different places and only the
/// chain writes one of them, so recomputation is not available to anybody.
///
/// Returns the blobs in the order the chain lists them, which is the order they
/// are passed to the program. Nothing downstream can notice a transposition —
/// the inputs are ciphertext, so the wrong order is a plausible answer to a
/// different question.
fn verify_inputs(
    commits: &[B256],
    blobs: &[Bytes],
    stored: &[B256],
) -> Result<Vec<Vec<u8>>, String> {
    if commits.len() != blobs.len() {
        return Err(format!(
            "the event carries {} commitment(s) and {} blob(s)",
            commits.len(),
            blobs.len()
        ));
    }

    if commits != stored {
        return Err(format!(
            "the event's {} commitment(s) are not the {} the contract stored; \
             the log and the chain disagree about this job",
            commits.len(),
            stored.len()
        ));
    }

    let mut inputs = Vec::with_capacity(blobs.len());
    for (index, (blob, commit)) in blobs.iter().zip(commits).enumerate() {
        let actual = wire::commitment(blob);
        if &B256::from(actual) != commit {
            return Err(format!(
                "input {index} does not match the commitment the chain holds: \
                 expected {commit}, got {}",
                bytecode::hex(&actual)
            ));
        }
        inputs.push(blob.to_vec());
    }

    Ok(inputs)
}

/// Checks the local artefacts against what `registerProgram` pinned, and
/// returns the quorum the chain asks for.
///
/// M comes from the chain rather than from a flag on purpose. `registerProgram`
/// pins `attestersRequired` per program and `fulfillJob` enforces it, so a
/// watcher with its own idea of the number would either collect fewer
/// signatures than the contract accepts — a `QuorumNotMet` revert after the
/// work is done — or more than it needs, which is worker time spent on nothing.
///
/// The two hash checks are the same argument one step earlier. A bytecode blob
/// that is not the registered one produces attestations over a digest the
/// contract never reconstructs, and the revert names the *worker*
/// (`NotRegisteredWorker`) rather than the program, which sends whoever reads
/// it to the registry.
fn check_registered_program(
    bytecode_blob: &[u8],
    server_key: &[u8],
    registered_bytecode: B256,
    registered_server_key: B256,
    attesters_required: u8,
) -> Result<usize, String> {
    if attesters_required == 0 {
        return Err(
            "no program is registered under that id: attestersRequired is 0, and \
             `registerProgram` refuses a quorum of zero (QuorumTooSmall)"
                .to_string(),
        );
    }

    // `hash_bytecode` rather than `keccak256` of the file: it is what the
    // coordinator binds into every claim, so this compares the value that will
    // actually be signed against the value the contract will reconstruct with.
    let local = B256::from(bytecode::hash_bytecode(bytecode_blob));
    if local != registered_bytecode {
        return Err(format!(
            "--bytecode hashes to {local}, but the program is registered as \
             {registered_bytecode}"
        ));
    }

    let local = B256::from(wire::commitment(server_key));
    if local != registered_server_key {
        return Err(format!(
            "--server-key hashes to {local}, but the program is registered as \
             {registered_server_key}; workers verify what they pull from \
             /keys/<hash> against the registered value"
        ));
    }

    Ok(usize::from(attesters_required))
}

/// The next block range to ask for, or `None` when there is nothing new yet.
///
/// `confirmations` is subtracted from the head rather than compared against a
/// log's own depth, so a job is never dispatched until the chain has buried it
/// by that much. Zero is a legitimate value and the default: Anvil mines only
/// when a transaction arrives, so a job in the newest block would otherwise
/// wait for some *unrelated* transaction to bury it — a demo that hangs for a
/// reason nothing on screen explains.
///
/// Saturating rather than wrapping at both ends: a chain shorter than
/// `confirmations` is a fresh Anvil, not an error, and it simply has nothing to
/// report yet.
fn scan_range(next: u64, head: u64, confirmations: u64, max_span: u64) -> Option<(u64, u64)> {
    let safe = head.checked_sub(confirmations)?;
    if next > safe {
        return None;
    }
    Some((next, safe.min(next.saturating_add(max_span - 1))))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A commitment/blob pair the chain would have accepted: `submitJob`
    /// rejects any other (`InputCommitmentMismatch`), so a test that built them
    /// independently would be testing a job that cannot exist.
    fn committed(bytes: &[u8]) -> (B256, Bytes) {
        (
            B256::from(wire::commitment(bytes)),
            Bytes::from(bytes.to_vec()),
        )
    }

    #[test]
    fn inputs_that_match_the_chains_commitments_are_accepted_in_order() {
        let (first_commit, first) = committed(b"one");
        let (second_commit, second) = committed(b"two");
        let commits = vec![first_commit, second_commit];

        let inputs = verify_inputs(&commits, &[first, second], &commits).unwrap();

        // Order, not just membership. The inputs are ciphertext, so a
        // transposition produces a plausible answer to a different question and
        // nothing downstream — not the workers, not the contract, not the key
        // holder — can notice.
        assert_eq!(inputs, vec![b"one".to_vec(), b"two".to_vec()]);
    }

    #[test]
    fn a_blob_that_does_not_hash_to_the_chains_commitment_is_refused() {
        // Task 2.9f, and the reason this module exists rather than the worker's
        // copy of the check being enough. The commitment here is the chain's;
        // whoever substituted the blob cannot recompute it, because they did not
        // write it.
        let (commit, _) = committed(b"the input the poster paid for");
        let substituted = Bytes::from_static(b"something else entirely");

        let error = verify_inputs(&[commit], &[substituted], &[commit]).unwrap_err();
        assert!(
            error.contains("does not match the commitment the chain holds"),
            "got: {error}"
        );
        assert!(error.contains("input 0"), "names which one: {error}");
    }

    #[test]
    fn the_second_of_two_inputs_is_checked_too() {
        // A loop that returned after the first match would pass every
        // single-input test in this file and substitute every later argument.
        let (good_commit, good) = committed(b"one");
        let (bad_commit, _) = committed(b"two");
        let commits = vec![good_commit, bad_commit];

        let error =
            verify_inputs(&commits, &[good, Bytes::from_static(b"not two")], &commits).unwrap_err();
        assert!(error.contains("input 1"), "got: {error}");
    }

    #[test]
    fn an_event_whose_commitments_are_not_the_chains_is_refused() {
        // The half of the check that a self-consistent forgery would otherwise
        // pass: blobs and commitments in a log come from the same place, so
        // hashing one against the other proves only that whoever wrote the log
        // was consistent. Storage is what the chain agreed on.
        let (commit, blob) = committed(b"one");
        let (stored, _) = committed(b"the input the chain actually holds");

        let error = verify_inputs(&[commit], &[blob], &[stored]).unwrap_err();
        assert!(
            error.contains("the log and the chain disagree"),
            "got: {error}"
        );
    }

    #[test]
    fn an_event_with_more_blobs_than_commitments_is_refused() {
        // `submitJob` refuses this (`InputLengthMismatch`), so reaching it means
        // the log is not describing a job this contract accepted. Checked before
        // the zip below, which would otherwise silently ignore the extra.
        let (commit, blob) = committed(b"one");

        let error = verify_inputs(
            &[commit],
            &[blob.clone(), blob],
            std::slice::from_ref(&commit),
        )
        .unwrap_err();
        assert!(
            error.contains("1 commitment(s) and 2 blob(s)"),
            "got: {error}"
        );
    }

    #[test]
    fn a_job_with_no_inputs_is_not_special_cased() {
        assert_eq!(verify_inputs(&[], &[], &[]).unwrap(), Vec::<Vec<u8>>::new());
    }

    #[test]
    fn the_quorum_and_the_program_hashes_come_from_the_chain() {
        let bytecode = b"not really bytecode, but hashed the same way";
        let server_key = b"nor is this a server key";
        let bytecode_hash = B256::from(bytecode::hash_bytecode(bytecode));
        let server_key_hash = B256::from(wire::commitment(server_key));

        assert_eq!(
            check_registered_program(bytecode, server_key, bytecode_hash, server_key_hash, 2)
                .unwrap(),
            2,
            "M is the chain's, not a flag's: `fulfillJob` enforces the registered \
             value and a watcher that disagreed would collect the wrong number"
        );

        // An unregistered program reads back as a zero-valued struct, and its
        // quorum of zero is the only field that can distinguish that from a real
        // one — `registerProgram` refuses zero (QuorumTooSmall).
        let error =
            check_registered_program(bytecode, server_key, B256::ZERO, B256::ZERO, 0).unwrap_err();
        assert!(error.contains("no program is registered"), "got: {error}");

        let error = check_registered_program(
            bytecode,
            server_key,
            B256::repeat_byte(0x11),
            server_key_hash,
            2,
        )
        .unwrap_err();
        assert!(error.contains("--bytecode hashes to"), "got: {error}");

        let error = check_registered_program(
            bytecode,
            server_key,
            bytecode_hash,
            B256::repeat_byte(0x22),
            2,
        )
        .unwrap_err();
        assert!(error.contains("--server-key hashes to"), "got: {error}");
    }

    #[test]
    fn a_job_id_that_would_truncate_is_refused_rather_than_signed_under() {
        assert_eq!(job_id_of(U256::from(7)).unwrap(), 7);
        assert_eq!(job_id_of(U256::from(u64::MAX)).unwrap(), u64::MAX);

        // `fulfillJob` casts to `uint64` to rebuild the digest. Truncating to
        // match would mean workers signing under an id the event does not name,
        // and every signature recovering to an address no registry holds.
        let error = job_id_of(U256::from(u64::MAX) + U256::from(1)).unwrap_err();
        assert!(error.contains("8-byte field"), "got: {error}");
    }

    #[test]
    fn a_scan_never_reaches_past_the_confirmations_depth() {
        // The reorg mitigation, such as it is. Nothing is dispatched until the
        // chain has buried it by `confirmations` blocks.
        assert_eq!(scan_range(0, 10, 0, 2_000), Some((0, 10)));
        assert_eq!(scan_range(0, 10, 3, 2_000), Some((0, 7)));
        assert_eq!(scan_range(8, 10, 3, 2_000), None, "8 is not yet buried");
        assert_eq!(scan_range(7, 10, 3, 2_000), Some((7, 7)));

        // A chain shorter than the confirmations depth is a fresh Anvil, not an
        // error: it has nothing to report yet and will have in a moment.
        assert_eq!(scan_range(0, 1, 5, 2_000), None);

        // Chunked, so a first scan from genesis does not ask a public endpoint
        // for the whole chain in one call.
        assert_eq!(scan_range(0, 10_000, 0, 2_000), Some((0, 1_999)));
        assert_eq!(scan_range(2_000, 10_000, 0, 2_000), Some((2_000, 3_999)));
    }

    #[test]
    fn the_calldata_attester_order_is_the_one_the_coordinator_settled_on() {
        // `fulfillJob` requires strictly increasing recovered addresses, and
        // out of order is a revert rather than a warning — one that names the
        // caller rather than the ordering. The coordinator already sorts; this
        // must not re-sort, or the rule would live in two places and only one
        // of them would be tested against the contract.
        let attesters = [
            attester(0x11, 0xaa, 0xbb, 27),
            attester(0x22, 0xcc, 0xdd, 28),
        ];

        let calldata = attestations_for(&attesters);
        assert_eq!(calldata.len(), 2);
        assert_eq!(calldata[0].r, B256::repeat_byte(0xaa));
        assert_eq!(calldata[0].s, B256::repeat_byte(0xbb));
        assert_eq!(calldata[0].v, 27, "27 or 28, never the bare 0/1 k256 uses");
        assert_eq!(calldata[1].r, B256::repeat_byte(0xcc));
        assert_eq!(calldata[1].v, 28);
    }

    fn attester(address: u8, r: u8, s: u8, v: u8) -> Attester {
        Attester {
            address: [address; 20],
            attestation: primitives::attest::Attestation {
                r: [r; 32],
                s: [s; 32],
                v,
            },
        }
    }

    #[test]
    fn an_https_endpoint_is_refused_at_startup_by_name() {
        // This build has no TLS transport (see the `alloy` features in the
        // workspace manifest). Without this the failure is a reqwest builder
        // error raised from inside the provider, which reads as a bug in this
        // program rather than as a missing feature — and it would be raised
        // only once a job had already been posted.
        let error = run(config("https://eth.example/rpc")).unwrap_err();
        assert!(error.contains("reqwest-rustls-tls"), "got: {error}");

        // ...and an http URL gets past this check, failing later for a reason
        // that is about the chain rather than about the scheme. Port 1 on
        // loopback refuses immediately, so this does not wait on a network.
        let error = run(config("http://127.0.0.1:1")).unwrap_err();
        assert!(!error.contains("reqwest-rustls-tls"), "got: {error}");
    }

    #[test]
    fn a_coordinator_key_that_is_not_a_key_is_refused_before_anything_connects() {
        let mut config = config("http://127.0.0.1:1");
        config.coordinator_key = "hunter2".into();

        let error = run(config).unwrap_err();
        assert!(
            error.contains("--coordinator-key is not a secp256k1 private key"),
            "got: {error}"
        );
    }

    // --- the startup checks, against a chain that is not there ---------------
    //
    // Every check in `run` before the poll loop is an `eth_call`, and every one
    // of them exists to turn a revert that arrives *after* three workers have
    // spent seconds on FHE into a refusal that arrives before any of them
    // start. Each also reverts on-chain with an error that points somewhere
    // else — a wrong key is `NotCoordinator`, a wrong bytecode hash is
    // `NotRegisteredWorker` for workers that are registered — so a check that
    // has quietly stopped firing is not something the next Anvil run makes
    // obvious. It shows up as a settlement that does not happen.
    //
    // Answered by a `tiny_http` server, which the coordinator already uses, and
    // encoded with the same `sol!` types that decode it. That makes these tests
    // about the checks rather than about the ABI — the ABI is what
    // `scripts/run-anvil.sh --watcher` exercises, against the real contract.

    struct FakeBridge {
        coordinator: EthAddress,
        program: programsReturn,
        registered: bool,
        /// Answer every call with empty data, which is what an address holding
        /// no code returns for all of them.
        no_code: bool,
    }

    /// A bridge that agrees with [`config`] about everything.
    fn fake_bridge() -> FakeBridge {
        let config = config("http://127.0.0.1:1");
        FakeBridge {
            coordinator: config
                .coordinator_key
                .parse::<PrivateKeySigner>()
                .unwrap()
                .address(),
            program: programsReturn {
                bytecodeHash: B256::from(bytecode::hash_bytecode(&config.bytecode)),
                serverKeyHash: B256::from(wire::commitment(&config.server_key)),
                attestersRequired: 1,
            },
            registered: true,
            no_code: false,
        }
    }

    /// Serves `bridge` as a JSON-RPC endpoint and returns its URL.
    fn serve_fake_bridge(bridge: FakeBridge) -> String {
        let server = tiny_http::Server::http("127.0.0.1:0").expect("a loopback port");
        let url = format!(
            "http://{}",
            server.server_addr().to_ip().expect("a TCP address")
        );

        thread::spawn(move || {
            for mut request in server.incoming_requests() {
                let mut body = String::new();
                let _ = std::io::Read::read_to_string(request.as_reader(), &mut body);

                // Matched on `"0x` + selector rather than on the selector alone,
                // so an argument that happens to contain another function's four
                // bytes cannot answer the wrong call.
                let called = |selector: [u8; 4]| {
                    body.contains(&format!("0x{}", alloy::primitives::hex::encode(selector)))
                };

                let returned = if bridge.no_code {
                    Vec::new()
                } else if called(coordinatorCall::SELECTOR) {
                    coordinatorCall::abi_encode_returns(&bridge.coordinator)
                } else if called(programsCall::SELECTOR) {
                    programsCall::abi_encode_returns(&bridge.program)
                } else if called(isRegisteredWorkerCall::SELECTOR) {
                    isRegisteredWorkerCall::abi_encode_returns(&bridge.registered)
                } else {
                    // Refused rather than answered with a zero. A check that
                    // starts calling something new should fail here, loudly,
                    // instead of reading a plausible default out of this stub.
                    let _ = request
                        .respond(tiny_http::Response::from_string("no").with_status_code(400));
                    continue;
                };

                let id = body
                    .split("\"id\":")
                    .nth(1)
                    .map(|rest| {
                        rest.chars()
                            .take_while(char::is_ascii_digit)
                            .collect::<String>()
                    })
                    .filter(|digits| !digits.is_empty())
                    .unwrap_or_else(|| "0".into());

                let json = format!(
                    "{{\"jsonrpc\":\"2.0\",\"id\":{id},\"result\":\"0x{}\"}}",
                    alloy::primitives::hex::encode(&returned)
                );
                let response = tiny_http::Response::from_string(json).with_header(
                    tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                        .expect("a valid header"),
                );
                let _ = request.respond(response);
            }
        });

        url
    }

    /// Runs a watcher against a stubbed bridge and returns why it refused.
    fn refusal(bridge: FakeBridge, adjust: impl FnOnce(&mut Config)) -> String {
        let mut config = config(&serve_fake_bridge(bridge));
        adjust(&mut config);
        run(config).expect_err("this watcher must not start")
    }

    #[test]
    fn a_bridge_address_with_no_code_says_so_rather_than_failing_to_decode() {
        // The most likely thing to be wrong on a fresh chain: an address copied
        // from the wrong deployment. `eth_call` against it succeeds and returns
        // nothing, so without this the first sign of trouble is an ABI decoder
        // complaining about a zero-length buffer.
        let error = refusal(
            FakeBridge {
                no_code: true,
                ..fake_bridge()
            },
            |_| {},
        );
        assert!(error.contains("is that a"), "got: {error}");
        assert!(
            error.contains("coordinator()"),
            "names the call that came back empty: {error}"
        );
    }

    #[test]
    fn a_key_that_is_not_the_bridges_coordinator_is_refused_before_any_job() {
        // On-chain this is `NotCoordinator`, raised after the workers have
        // finished. The gate is about payment rather than trust (`bridge.md`
        // §2) — anyone relaying a genuine settlement would be harmless to
        // correctness and would collect the escrow — but a watcher that cannot
        // collect is a watcher that cannot settle.
        let stranger = EthAddress::repeat_byte(0x99);
        let error = refusal(
            FakeBridge {
                coordinator: stranger,
                ..fake_bridge()
            },
            |_| {},
        );
        assert!(error.contains(&stranger.to_string()), "got: {error}");
        assert!(
            error.contains("fulfillJob"),
            "says what the address is for: {error}"
        );
    }

    #[test]
    fn bytecode_that_is_not_the_registered_program_is_refused() {
        // The nastiest of the four to diagnose on-chain. Workers sign a digest
        // built from the hash of the bytecode they were dispatched, `fulfillJob`
        // rebuilds it from the hash `registerProgram` pinned, and if the two
        // differ every signature recovers to a stranger — so the contract
        // reports `NotRegisteredWorker` for workers that are registered, and
        // sends whoever reads it to the registry.
        let mut bridge = fake_bridge();
        bridge.program.bytecodeHash = B256::repeat_byte(0x11);

        let error = refusal(bridge, |_| {});
        assert!(error.contains("--bytecode hashes to"), "got: {error}");
    }

    #[test]
    fn a_server_key_that_is_not_the_registered_one_is_refused() {
        // Workers verify what they pull from `/keys/<hash>` against the hash the
        // dispatch names, so serving the wrong key fails every worker at once —
        // as a job where nobody reported, which is also what a network partition
        // looks like.
        let mut bridge = fake_bridge();
        bridge.program.serverKeyHash = B256::repeat_byte(0x22);

        let error = refusal(bridge, |_| {});
        assert!(error.contains("--server-key hashes to"), "got: {error}");
    }

    #[test]
    fn a_registered_quorum_that_is_not_a_majority_of_the_workers_is_refused() {
        // M comes from `registerProgram` and N from `--worker`, so this is a
        // deployment-shaped version of the rule `accept_job` applies per job:
        // 2-of-4 lets a minority settle during the straggler grace
        // (`spec/`, MC_GraceRace_N4M2). Refusing at startup means the operator
        // hears about it once rather than once per job, forever.
        let mut bridge = fake_bridge();
        bridge.program.attestersRequired = 2;

        let error = refusal(bridge, |config| {
            config.workers = vec!["127.0.0.1:1".into(); 4];
        });
        assert!(error.contains("majority"), "got: {error}");
        assert!(
            error.contains("2 attester(s) on-chain"),
            "says whose number it is: {error}"
        );
    }

    #[test]
    fn a_worker_the_chain_has_never_registered_is_refused() {
        // The registry is a mapping and cannot be enumerated, so the addresses
        // are supplied and each is checked. Without this, a typo in
        // `--registered-worker` is a job that reaches quorum here and reverts
        // there — as `NotRegisteredWorker`, which is accurate and arrives far
        // too late to be useful.
        let error = refusal(
            FakeBridge {
                registered: false,
                ..fake_bridge()
            },
            |_| {},
        );
        assert!(error.contains("is not registered on"), "got: {error}");
        assert!(
            error.contains("NotRegisteredWorker"),
            "names the revert it is standing in front of: {error}"
        );
    }

    /// A watcher configuration that is valid apart from having no chain to talk
    /// to. Anvil's first account key, which is public knowledge and funds
    /// nothing outside a throwaway devnet.
    fn config(rpc: &str) -> Config {
        Config {
            rpc: rpc.into(),
            bridge: EthAddress::repeat_byte(0xbb),
            coordinator_key: "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
                .into(),
            bind: "127.0.0.1:0".into(),
            workers: vec!["127.0.0.1:1".into()],
            registry: vec![[1u8; 20]],
            program_id: U256::from(1),
            bytecode: b"not bytecode".to_vec(),
            function: "tally4_select".into(),
            server_key: b"not a server key".to_vec(),
            deadline: Duration::from_millis(1),
            confirmations: 0,
            poll: Duration::from_millis(1),
            from_block: 0,
        }
    }
}
