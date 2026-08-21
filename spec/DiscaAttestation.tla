--------------------------- MODULE DiscaAttestation ---------------------------
(***************************************************************************)
(* DISCA M-of-N attestation and job settlement.                            *)
(*                                                                         *)
(* One job, N workers, M required attesters, one coordinator, and the      *)
(* bridge contract.  The model is written against the implementation, not  *)
(* against an idealisation of it:                                          *)
(*                                                                         *)
(*   node/src/coordinator.rs   Verifier::attribute, record, group, tally,  *)
(*                             agreement_still_possible, collect           *)
(*   node/src/worker.rs        Behaviour::Honest / Behaviour::Faulty       *)
(*   primitives/src/attest.rs  Claim, Claim::preimage, recover             *)
(*   docs/bridge.md  2, 2a, 5a, 6   fulfillJob / refundOnTimeout          *)
(*                                                                         *)
(* Every deviation from the code is flagged in a comment beginning         *)
(* "DIVERGENCE".  Every thing that is assumed rather than checked is       *)
(* flagged in a comment beginning "ASSUMED".  spec/README.md collects both *)
(* lists; if you change this file, change that one.                        *)
(*                                                                         *)
(***************************************************************************)
(* THE CRYPTOGRAPHIC BOUNDARY --- what TLC proves and what it does not.    *)
(*                                                                         *)
(* An attestation is a recoverable secp256k1 signature over the claim in   *)
(* primitives/src/attest.rs:                                               *)
(*                                                                         *)
(*     keccak256(EIP-191 prefix || keccak256(                              *)
(*         "DISCA/attest/result/v1" || jobId || bytecodeHash || resultHash))*)
(*                                                                         *)
(* and a verifier does not check it, it *recovers* from it: the address    *)
(* falls out of (claim, r, s, v).  This model represents a signature as    *)
(* the record  [signer, job, res]  --- the triple that determines which    *)
(* address recovery yields --- and represents "the set of signatures that  *)
(* exist anywhere in the world" as the variable `sigs`.                    *)
(*                                                                         *)
(* ASSUMED (unforgeability).  A record [signer |-> w, ...] enters `sigs`   *)
(* only through an action taken by w itself.  No action anywhere in this   *)
(* module adds a signature attributed to a party other than the one        *)
(* taking the step.  This is the assumption that secp256k1 is unforgeable  *)
(* under chosen-message attack, and it is exactly the boundary between     *)
(* what TLC establishes and what cryptography is relied on to provide.     *)
(* TLC does not verify it; it is true of this module by construction, and  *)
(* the construction is the assumption.                                     *)
(*                                                                         *)
(* ASSUMED (recovery on a mismatched claim yields an unrelated address).   *)
(* A signature made over job id J presented against job id K /= J still    *)
(* recovers to *some* address --- ECDSA recovery essentially never fails   *)
(* --- just not to the signer's.  The model renders this as "a signature   *)
(* with s.job /= JobId is refused", which is the registry check in         *)
(* Verifier::attribute rejecting the unrelated address that comes back.    *)
(* The probability that the unrelated address happens to be a registered   *)
(* worker's is negligible and is not modelled.                             *)
(*                                                                         *)
(* ASSUMED (attestations are public).  A signature is not a secret.  Any   *)
(* party that has seen one can relay it.  `Post` is therefore enabled for  *)
(* *any* signature in `sigs`, not only for the worker that made it, which  *)
(* is what makes replay expressible at all.  The code says the same thing  *)
(* in the comment above `record`: "/results accepts a report from any      *)
(* registered address and an attestation is not a secret".                 *)
(*                                                                         *)
(* ASSUMED (deterministic honest evaluation).  Two honest workers running  *)
(* the same circuit over the same input ciphertexts produce byte-identical *)
(* compressed results, so they hash to the same attestation.  In the model *)
(* every honest worker computes TrueResult.  This is architecture.md 3's  *)
(* deterministic-evaluation property and it is NOT free: it holds only     *)
(* with the FFT plan pinned, on one CPU architecture, on CPU rather than   *)
(* GPU, at one tfhe version.  A worker violating any of those diverges     *)
(* while behaving honestly and is indistinguishable here from "faulty".    *)
(* That is deliberate --- it is also indistinguishable to the coordinator. *)
(*                                                                         *)
(* ASSUMED (keccak256 is collision-resistant).  Distinct results have      *)
(* distinct hashes, so `Results` is a set of distinguishable values and    *)
(* `keccak256(resultBlob) == resultHash` in fulfillJob pins the blob.  Not *)
(* modelled further; the blob is not a variable.                           *)
(***************************************************************************)

EXTENDS Naturals, FiniteSets

CONSTANTS
    (* The registered worker set --- the on-chain registry of bridge.md 2, *)
    (* `registerWorker(address)`, and Config::registry in coordinator.rs.   *)
    Workers,

    (* Signers holding their own keys who are NOT in the registry.  Any     *)
    (* number of these can be conjured for free, which is the whole reason  *)
    (* the registry exists (coordinator.rs: "anybody with a keyboard can    *)
    (* generate M keypairs and out-vote the honest workers").  Usually {}.  *)
    Strangers,

    (* M, `--attesters`, `attestersRequired`.                               *)
    M,

    (* The distinguishable sealed-result hashes a job can produce.          *)
    Results,
    TrueResult,

    (* The job id bound into every signature.  The deployed coordinator     *)
    (* hardcodes `let job_id = 1;` and there is no chain to take one from,  *)
    (* so an earlier run of the same program used the same id.  Set         *)
    (* PriorJobId = JobId to model that; set it different to model          *)
    (* `submitJob` assigning a globally unique id (bridge.md 2, task 2.9f).*)
    JobId,
    PriorJobId,

    (* Variant flags.  Each one turns a decision the implementation made    *)
    (* into a knob, so that the counterexample for taking the other branch  *)
    (* is a thing TLC produces rather than a thing this file asserts.       *)
    FirstWriteWins,      \* coordinator.rs `record`: Entry::Occupied keeps the first
    RefuseOnSplit,       \* coordinator.rs `tally`: quorums.len() > 1 -> None
    RegistryEnforced,    \* coordinator.rs `attribute` + bridge.md 2a step 4
    RefundGuarded,       \* refundOnTimeout requires state == Open
    AdversarialFulfill,  \* anyone may call fulfillJob with any signatures they hold
    ReplayAvailable,     \* an earlier run's attestations are in circulation
    EquivocatingFaults,  \* a faulty worker will sign two different results
    PatientDeadline,     \* the deadline does not fire before the workers answer
    LateReplayOnly,      \* a relayed signature arrives only after every live
                         \* worker has already reported --- the "late report"
                         \* the first-write-wins comment is about, isolated
                         \* from the pre-emption case

    (* Bounds on the fault assignment, so a configuration can say "within   *)
    (* the tolerance N - M" or "beyond it".                                 *)
    MaxFaults,
    MinHonest

(***************************************************************************)
(* Static sanity.  A configuration that fails these is checking nothing.   *)
(***************************************************************************)
ASSUME M \in Nat /\ M > 0
ASSUME IsFiniteSet(Workers) /\ Cardinality(Workers) >= M
ASSUME TrueResult \in Results /\ Cardinality(Results) >= 2
ASSUME Workers \cap Strangers = {}

----------------------------------------------------------------------------

(* Claim::Failure occupies the result slot with keccak256(reason) under a   *)
(* different 22-byte domain tag, so it can never be read as a result        *)
(* attestation (attest.rs, DOMAIN_FAILURE).  Modelled as a value that is    *)
(* simply not in Results --- which is what makes it structurally impossible *)
(* for a failure to reach a quorum, since `Quorums` below ranges over       *)
(* Results only.  That mirrors `group()`, which buckets                     *)
(* JobOutcome::Evaluated and drops JobOutcome::Failed on the floor.         *)
Failure  == "failure"
Empty    == "empty"
Unstarted == "unstarted"

Signers  == Workers \cup Strangers
Registry == Workers
Outcomes == Results \cup {Failure}
Behaviours == {"honest", "faulty", "crashed", "failing"}

Sig(w, j, r) == [signer |-> w, job |-> j, res |-> r]

(* Every signature that could ever exist, used as a constant quantifier     *)
(* domain.  Nothing is in `sigs` because it is in here; things get into     *)
(* `sigs` only by being signed.                                             *)
SigSpace == [signer: Signers, job: {JobId, PriorJobId}, res: Outcomes]

(* What an earlier run of the same program leaves lying around.             *)
(*                                                                          *)
(* The claim binds the job id, the bytecode hash and the result hash --- it *)
(* does NOT bind the input commitments (attest.rs Claim::preimage, all 94   *)
(* bytes of it).  So a previous job over the *same program* with *different *)
(* inputs produced a different result, and its attestations are signatures  *)
(* over (JobId, bytecodeHash, thatResult).  With a constant job id those    *)
(* are, byte for byte, valid attestations for the job running now.  The     *)
(* adversary is given the strongest such set: every worker's signature over *)
(* every result.                                                            *)
PriorSigs == IF ReplayAvailable
               THEN [signer: Workers, job: {PriorJobId}, res: Results]
               ELSE {}

(* A stranger holds its own key and can sign whatever it likes, at any      *)
(* time, for the current job.  Seeded at Init rather than given an action,  *)
(* because "can sign at will" and "has already signed" are the same thing   *)
(* for a party that is never waited on.                                     *)
StrangerSigs == [signer: Strangers, job: {JobId}, res: Results]

----------------------------------------------------------------------------

VARIABLES
    behaviour,      \* Workers -> Behaviours.  Set at Init, never observable
                    \* on the wire: worker.rs keeps `--faulty` local on purpose.
    evaluated,      \* Signers -> Outcomes \cup {Unstarted}: what this party
                    \* actually computed in THIS job.
    sigs,           \* the signatures that exist anywhere in the world
    current,        \* GHOST: sigs \cap "produced by a worker running THIS job".
                    \* Not observable by any party --- that is the point.  A
                    \* replayed signature is in `sigs` and not in `current`,
                    \* and no protocol action reads this variable.
    inbox,          \* Signers -> Outcomes \cup {Empty}.  The coordinator's
                    \* HashMap<Address, JobReport>, keyed by RECOVERED address.
    displaced,      \* GHOST: has a report ever overwritten a slot already cast
    coord,          \* "collecting" | "settled" | "refused"
    settleSplit,    \* GHOST: were two groups at quorum when the coordinator settled
    grace,          \* has STRAGGLER_GRACE expired
    deadline,       \* has the job deadline passed
    chain,          \* "open" | "fulfilled" | "refunded"
    chainResult,
    chainAttesters,
    fulfilCount,    \* capped at 2; one is all that may ever happen
    refundCount

vars == << behaviour, evaluated, sigs, current, inbox, displaced, coord,
           settleSplit, grace, deadline, chain, chainResult, chainAttesters,
           fulfilCount, refundCount >>

TypeOK ==
    /\ behaviour \in [Workers -> Behaviours]
    /\ evaluated \in [Signers -> Outcomes \cup {Unstarted}]
    /\ sigs \subseteq SigSpace
    /\ current \subseteq sigs
    /\ inbox \in [Signers -> Outcomes \cup {Empty}]
    /\ displaced \in BOOLEAN
    /\ coord \in {"collecting", "settled", "refused"}
    /\ settleSplit \in BOOLEAN
    /\ grace \in BOOLEAN
    /\ deadline \in BOOLEAN
    /\ chain \in {"open", "fulfilled", "refunded"}
    /\ chainResult \in Results \cup {"none"}
    /\ chainAttesters \subseteq Signers
    /\ fulfilCount \in 0..2
    /\ refundCount \in 0..2

----------------------------------------------------------------------------
(***************************************************************************)
(* The coordinator's view.                                                 *)
(***************************************************************************)

(* coordinator.rs `group`: bucket the inbox by attestation hash.  Keyed by  *)
(* the recovered address, so a party occupies at most one slot and cannot   *)
(* be two members of a group.  Here that is a property of `inbox` being a   *)
(* function on Signers rather than a check anyone could forget --- the same *)
(* reason the code uses a HashMap keyed by Address.                         *)
Group(r) == {w \in Signers : inbox[w] = r}

(* coordinator.rs `tally`: the groups that have reached M.  Ranges over     *)
(* Results, so a signed failure is never a member of any group.             *)
Quorums == {r \in Results : Cardinality(Group(r)) >= M}

Reported == {w \in Signers : inbox[w] # Empty}

(* coordinator.rs `agreement_still_possible`.  DIVERGENCE: modelled but not *)
(* used as a guard.  It is an early-exit optimisation --- it decides when   *)
(* to stop waiting, never what to settle on --- so including it as a guard  *)
(* would only remove behaviours from the model.  Leaving it out keeps the   *)
(* model an over-approximation of the code's timing, which is the safe      *)
(* direction for safety checking.  It is defined here so the README can     *)
(* point at it and so its "reported > dispatched -> keep waiting" clause is *)
(* on the record.                                                           *)
AgreementStillPossible ==
    LET sizes == {Cardinality(Group(r)) : r \in Results}
        best  == CHOOSE b \in sizes : \A o \in sizes : o <= b
    IN  \/ Cardinality(Reported) > Cardinality(Workers)
        \/ best + (Cardinality(Workers) - Cardinality(Reported)) >= M

----------------------------------------------------------------------------
(***************************************************************************)
(* Actions.                                                               *)
(***************************************************************************)

(* worker.rs: what this worker's evaluation produces.                       *)
(*                                                                          *)
(* "faulty" is Behaviour::Faulty --- `&result + &result`, corrupting the    *)
(* value and not the encoding, so what leaves the worker is a perfectly     *)
(* well-formed result that happens to be wrong.  Modelled as any result     *)
(* other than the true one, which also covers the misconfiguration case     *)
(* worker.rs says is the likelier one (wrong FFT plan, wrong ISA): honest   *)
(* code, divergent bytes, indistinguishable to the coordinator.             *)
(*                                                                          *)
(* "failing" is JobOutcome::Failed --- a signed failure report.             *)
(* "crashed" is silence, which the deadline covers.                         *)
OutcomeOf(w) ==
    CASE behaviour[w] = "honest"  -> {TrueResult}
      [] behaviour[w] = "faulty"  -> Results \ {TrueResult}
      [] behaviour[w] = "failing" -> {Failure}
      [] OTHER                    -> {}

(* Evaluate and sign, in one step.                                          *)
(*                                                                          *)
(* DIVERGENCE: worker.rs evaluates and then signs as two statements.  They  *)
(* are merged here because no other party can observe the gap: nothing is   *)
(* sent between them, and a worker that dies in the gap is indistinguishable *)
(* from one that never started.  Merging halves the reachable states with   *)
(* no behaviour lost.                                                       *)
(*                                                                          *)
(* Note which signature is produced: Sig(w, JobId, o) where `o` is what THIS *)
(* worker computed.  A worker cannot sign a result it did not compute,      *)
(* because there is no other action that puts a signature bearing its name  *)
(* into `sigs`.  That is the unforgeability assumption, made structural.    *)
Compute(w) ==
    /\ behaviour[w] # "crashed"
    /\ evaluated[w] = Unstarted
    /\ \E o \in OutcomeOf(w) :
         /\ evaluated' = [evaluated EXCEPT ![w] = o]
         /\ sigs'    = sigs    \cup {Sig(w, JobId, o)}
         /\ current' = current \cup {Sig(w, JobId, o)}
    /\ UNCHANGED << behaviour, inbox, displaced, coord, settleSplit, grace,
                    deadline, chain, chainResult, chainAttesters, fulfilCount,
                    refundCount >>

(* One signer saying two different things about one job.                    *)
(*                                                                          *)
(* coordinator.rs calls this out by name: "A second attestation over a      *)
(* different result is signed evidence that one signer said two things      *)
(* about one job, and with unique job ids the right response would be to    *)
(* discard that signer's vote entirely."  It is a strictly stronger fault   *)
(* than Behaviour::Faulty, which reports once, so it is off unless a        *)
(* configuration asks for it.                                               *)
Equivocate(w) ==
    /\ EquivocatingFaults
    /\ behaviour[w] = "faulty"
    /\ evaluated[w] \in Results
    /\ Cardinality({s \in current : s.signer = w}) = 1
    /\ \E r \in Results \ {evaluated[w]} :
         /\ sigs'    = sigs    \cup {Sig(w, JobId, r)}
         /\ current' = current \cup {Sig(w, JobId, r)}
    /\ UNCHANGED << behaviour, evaluated, inbox, displaced, coord, settleSplit,
                    grace, deadline, chain, chainResult, chainAttesters,
                    fulfilCount, refundCount >>

(* coordinator.rs `Verifier::attribute`.                                    *)
(*                                                                          *)
(*   - report.job_id must equal the coordinator's job id, AND the claim is  *)
(*     reconstructed from the coordinator's own job id and bytecode hash,   *)
(*     never from a field the sender can steer.  A signature made over any  *)
(*     other job therefore recovers to an address that is not the sender's. *)
(*   - the recovered address must be in the registry.                       *)
(*                                                                          *)
(* The sealed-blob-matches-its-own-hash check is not modelled: it rejects a *)
(* report before any counting happens and cannot admit one, so it can only  *)
(* remove behaviours.  DIVERGENCE, in the safe direction.                   *)
Accepted(s) ==
    /\ s.job = JobId
    /\ RegistryEnforced => s.signer \in Registry

(* Anyone may POST any signature they hold.  See the "attestations are      *)
(* public" assumption at the head of the module.                            *)
Post(s) ==
    /\ coord = "collecting"
    /\ s \in sigs
    /\ Accepted(s)
    (* Restricts a relayed signature --- one not produced by a worker in     *)
    (* this job --- to arriving after every live worker has already spoken.  *)
    (* Without it, TLC's shortest counterexample for anything involving      *)
    (* replay is always pre-emption (relay first, before any worker starts), *)
    (* which is a different attack and hides the one first-write-wins is     *)
    (* about.  This flag pins the arrival order to "late" so the two can be  *)
    (* separated.                                                            *)
    /\ (LateReplayOnly /\ s \notin current) =>
         (\A w \in Workers : behaviour[w] = "crashed" \/ inbox[w] # Empty)
    (* coordinator.rs `record`: first write wins.  Guarded so that a report *)
    (* which would change nothing is not a step --- it keeps the state      *)
    (* graph free of self-loops without changing what is reachable.         *)
    /\ \/ inbox[s.signer] = Empty
       \/ (~FirstWriteWins /\ inbox[s.signer] # s.res)
    /\ inbox' = [inbox EXCEPT ![s.signer] = s.res]
    /\ displaced' = (displaced \/ inbox[s.signer] # Empty)
    /\ UNCHANGED << behaviour, evaluated, sigs, current, coord, settleSplit,
                    grace, deadline, chain, chainResult, chainAttesters,
                    fulfilCount, refundCount >>

(* coordinator.rs `collect`: STRAGGLER_GRACE starts once a quorum exists    *)
(* and expiring it is what allows settling before every worker has spoken.  *)
GraceExpires ==
    /\ ~grace
    /\ Quorums # {}
    /\ grace' = TRUE
    /\ UNCHANGED << behaviour, evaluated, sigs, current, inbox, displaced,
                    coord, settleSplit, deadline, chain, chainResult,
                    chainAttesters, fulfilCount, refundCount >>

(* bridge.md 2a: what fulfillJob must do, in order.                        *)
(*                                                                          *)
(*   1. keccak256(resultBlob) == resultHash.  Not modelled: the blob is not *)
(*      a variable and hashing is assumed collision-resistant, so the blob  *)
(*      is determined by `r`.  DIVERGENCE, and it is the check that makes   *)
(*      the emitted ciphertext the attested one (5a) rather than anything  *)
(*      about quorum.                                                       *)
(*   2. Reconstruct the digest from the job's STORED bytecode hash and job  *)
(*      id, never from calldata.  Modelled: the guard demands a signature   *)
(*      over JobId, the contract's own, not over anything `A` carries.      *)
(*   3. ecrecover per attestation; reject v outside {27,28}, reject high s, *)
(*      reject address(0).  Modelled: `A \subseteq Signers` --- recovery    *)
(*      yields a real party, never the zero address.  The malleability and  *)
(*      recovery-id checks are byte-level and have no state-machine content.*)
(*   4. distinct and registered.  Distinct: `A` is a set.  Registered:      *)
(*      RegistryEnforced.                                                   *)
(*   5. count >= attestersRequired.                                         *)
(*                                                                          *)
(* The `Sig(w, JobId, r) \in sigs` conjunct is the whole point: the caller  *)
(* supplies signatures, and it can only supply signatures that exist.  It   *)
(* cannot name an address; the address comes back out of the signature.     *)
FulfillWith(r, A) ==
    /\ chain = "open"
    /\ fulfilCount < 2
    /\ A \subseteq Signers
    /\ RegistryEnforced => A \subseteq Registry
    /\ Cardinality(A) >= M
    /\ \A w \in A : Sig(w, JobId, r) \in sigs
    /\ chain' = "fulfilled"
    /\ chainResult' = r
    /\ chainAttesters' = A
    /\ fulfilCount' = fulfilCount + 1
    /\ refundCount' = refundCount

(* coordinator.rs `tally` + `collect` + the settlement that follows.        *)
(*                                                                          *)
(* Settling requires a quorum AND either every worker having reported or    *)
(* the straggler grace having expired --- `collect` returns early only when *)
(* `reported >= dispatched`, and otherwise waits out STRAGGLER_GRACE and    *)
(* re-tallies.  That window is real: a quorum can be settled before a slow  *)
(* worker's disagreeing report arrives.                                     *)
(*                                                                          *)
(* With two groups at quorum, `tally` returns None and the job does not     *)
(* settle at all; the escrow refund path (bridge.md 6) is what closes it.  *)
Settle ==
    /\ coord = "collecting"
    /\ chain = "open"
    /\ Quorums # {}
    /\ (Workers \subseteq Reported) \/ grace
    /\ IF Cardinality(Quorums) > 1 /\ RefuseOnSplit
         THEN /\ coord' = "refused"
              /\ UNCHANGED << settleSplit, chain, chainResult, chainAttesters,
                              fulfilCount, refundCount >>
         ELSE \E r \in Quorums :
                /\ coord' = "settled"
                /\ settleSplit' = (Cardinality(Quorums) > 1)
                /\ FulfillWith(r, Group(r))
    /\ UNCHANGED << behaviour, evaluated, sigs, current, inbox, displaced,
                    grace, deadline >>

(* Anyone at all calling fulfillJob with any signatures they have seen.     *)
(*                                                                          *)
(* This is what makes QuorumIsReal a property OF THE CONTRACT rather than a *)
(* property of the coordinator being nice.  bridge.md 2b is the record of  *)
(* what happens when the contract trusts the caller's word instead: "Any M  *)
(* registered addresses could be named beside any resultHash.  The check    *)
(* looked like verification and was a formatting requirement."              *)
AdversaryFulfill ==
    /\ AdversarialFulfill
    /\ \E r \in Results, A \in SUBSET Signers :
         /\ FulfillWith(r, A)
    /\ UNCHANGED << behaviour, evaluated, sigs, current, inbox, displaced,
                    coord, settleSplit, grace, deadline >>

(* The job deadline expiring.  Unguarded, it may fire at any moment --- the *)
(* coordinator is trusted only for liveness (architecture.md 3) and a job  *)
(* whose deadline passes before anyone answers is exactly the case the      *)
(* refundable escrow exists for.                                            *)
(*                                                                          *)
(* PatientDeadline models a deadline generous enough to be irrelevant: it   *)
(* does not expire while the coordinator is still collecting AND a quorum   *)
(* could still form.  `AgreementStillPossible` is coordinator.rs's own      *)
(* early-exit predicate, so this says precisely "the deadline never beats a *)
(* job that was still going to succeed".  It is what turns the liveness     *)
(* question from "settles or refunds" into "settles".                       *)
Timeout ==
    /\ ~deadline
    /\ PatientDeadline => (coord # "collecting" \/ ~AgreementStillPossible)
    /\ deadline' = TRUE
    /\ UNCHANGED << behaviour, evaluated, sigs, current, inbox, displaced,
                    coord, settleSplit, grace, chain, chainResult,
                    chainAttesters, fulfilCount, refundCount >>

(* bridge.md 6: coordinator goes silent -> refundOnTimeout returns escrow. *)
(* RefundGuarded is the `state == Open` requirement.  Turning it off is how *)
(* the model shows the guard is load-bearing rather than decorative.        *)
Refund ==
    /\ deadline
    /\ refundCount < 2
    /\ RefundGuarded => chain = "open"
    /\ chain' = "refunded"
    /\ refundCount' = refundCount + 1
    /\ UNCHANGED << behaviour, evaluated, sigs, current, inbox, displaced,
                    coord, settleSplit, grace, deadline, chainResult,
                    chainAttesters, fulfilCount >>

(* A terminal stutter so that a finished job is not reported as a deadlock. *)
(* Guarded on the escrow having been settled one way or the other, so a     *)
(* genuine mid-protocol deadlock is still caught.                           *)
Done ==
    /\ chain \in {"fulfilled", "refunded"}
    /\ UNCHANGED vars

----------------------------------------------------------------------------

Init ==
    /\ behaviour \in [Workers -> Behaviours]
    /\ Cardinality({w \in Workers : behaviour[w] = "faulty"}) <= MaxFaults
    /\ Cardinality({w \in Workers : behaviour[w] = "honest"}) >= MinHonest
    /\ evaluated = [w \in Signers |-> Unstarted]
    /\ sigs = PriorSigs \cup StrangerSigs
    /\ current = {}
    /\ inbox = [w \in Signers |-> Empty]
    /\ displaced = FALSE
    /\ coord = "collecting"
    /\ settleSplit = FALSE
    /\ grace = FALSE
    /\ deadline = FALSE
    /\ chain = "open"
    /\ chainResult = "none"
    /\ chainAttesters = {}
    /\ fulfilCount = 0
    /\ refundCount = 0

Next ==
    \/ \E w \in Workers : Compute(w)
    \/ \E w \in Workers : Equivocate(w)
    \/ \E s \in SigSpace : Post(s)
    \/ GraceExpires
    \/ Settle
    \/ AdversaryFulfill
    \/ Timeout
    \/ Refund
    \/ Done

Fairness ==
    /\ WF_vars(\E w \in Workers : Compute(w))
    /\ WF_vars(\E s \in SigSpace : Post(s))
    /\ WF_vars(GraceExpires)
    /\ WF_vars(Settle)
    /\ WF_vars(Timeout)
    /\ WF_vars(Refund)

Spec == Init /\ [][Next]_vars /\ Fairness

----------------------------------------------------------------------------
(***************************************************************************)
(* PROPERTIES                                                             *)
(***************************************************************************)

(*-------------------------------------------------------------------------*)
(* 1. QuorumIsReal.                                                        *)
(*                                                                         *)
(* If the contract fulfilled the job with result R, then at least M         *)
(* distinct REGISTERED workers signed R for THIS job.                       *)
(*                                                                         *)
(* "for this job" is the load-bearing phrase and it is why the ghost        *)
(* `current` exists.  A signature in `sigs \ current` is one that some      *)
(* worker really made, over this job id, over this program --- but in an    *)
(* earlier run.  No party in the protocol can tell the two apart; the       *)
(* invariant can, which is precisely how the cost of a constant job id      *)
(* becomes visible.                                                         *)
(*-------------------------------------------------------------------------*)
QuorumIsReal ==
    chain = "fulfilled" =>
        Cardinality({w \in Registry : Sig(w, JobId, chainResult) \in current}) >= M

(*-------------------------------------------------------------------------*)
(* The other face of the same failure: a job can be fulfilled without a     *)
(* single party having evaluated anything for it.                           *)
(*-------------------------------------------------------------------------*)
SomeoneActuallyEvaluated ==
    chain = "fulfilled" =>
        Cardinality({w \in Workers : evaluated[w] \in Results}) >= M

(*-------------------------------------------------------------------------*)
(* 2. EscrowPaidOnce.  Fulfilment and refund are mutually exclusive and     *)
(* each happens at most once.                                              *)
(*-------------------------------------------------------------------------*)
EscrowPaidOnce == fulfilCount + refundCount <= 1

(*-------------------------------------------------------------------------*)
(* 3. NoSettleOnSplit.  When two groups both reach M the coordinator        *)
(* refuses rather than picking one.                                        *)
(*-------------------------------------------------------------------------*)
NoSettleOnSplit == coord = "settled" => ~settleSplit

(*-------------------------------------------------------------------------*)
(* What a split actually costs, and it is NOT QuorumIsReal.                *)
(*                                                                         *)
(* Both groups genuinely signed, so QuorumIsReal survives either choice.    *)
(* What dies is the inference from agreement to correctness, which is the   *)
(* only reason anyone wanted agreement.  This invariant is that inference,  *)
(* stated so it can fail.                                                   *)
(*                                                                         *)
(* Expected to hold only where a configuration keeps the fault count inside *)
(* the tolerance; where it does not, its counterexample IS the argument     *)
(* that refusing is necessary rather than merely cautious.                  *)
(*-------------------------------------------------------------------------*)
ResultIsCorrect == chain = "fulfilled" => chainResult = TrueResult

(* The same inference, restricted to settlements that WERE splits.  Under   *)
(* RefuseOnSplit this is vacuous, because there are no such settlements ---  *)
(* and saying so is the honest way to report it.  Under a coordinator that  *)
(* picks a group, it is the direct statement that the pick can be wrong,    *)
(* which is what makes refusal necessary rather than cautious.              *)
SplitSettlementWasCorrect ==
    (coord = "settled" /\ settleSplit) => chainResult = TrueResult

(*-------------------------------------------------------------------------*)
(* 4. FirstWriteWins.  No report ever overwrites a vote already cast.       *)
(*-------------------------------------------------------------------------*)
VoteNotDisplaced == ~displaced

(*-------------------------------------------------------------------------*)
(* 6. Liveness.  The job does not hang: the escrow is always eventually     *)
(* settled, by fulfilment or by refund.                                    *)
(*-------------------------------------------------------------------------*)
Liveness == <>(chain \in {"fulfilled", "refunded"})

(* The stronger statement, which needs a deadline that does not fire before *)
(* the workers answer (PatientDeadline), at least M honest workers, and     *)
(* fewer than M faulty ones so no second group can reach quorum.           *)
EventuallyFulfils == <>(chain = "fulfilled")

(*-------------------------------------------------------------------------*)
(* Non-vacuity witnesses.  Each of these is checked as an invariant in a    *)
(* configuration that EXPECTS it to be violated; the violation is the       *)
(* evidence that the scenario the property is about is actually reachable   *)
(* in that configuration, rather than the property holding because nothing  *)
(* interesting ever happens.                                               *)
(*-------------------------------------------------------------------------*)
NoSplitEver == Cardinality(Quorums) <= 1
NeverFulfils == chain # "fulfilled"
NeverRefunds == chain # "refunded"

============================================================================
