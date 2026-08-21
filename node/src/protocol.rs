//! Messages the coordinator and workers exchange.
//!
//! Kept independent of the transport that carries them, so the HTTP layer in
//! [`crate::transport`] can be replaced without touching the protocol.
//!
//! Bodies are `wincode`-encoded rather than JSON. Every message carries
//! ciphertext or bytecode blobs, which JSON can only hold base64-inflated, and
//! the crate already encodes bytecode with `wincode` — one encoder is better
//! than two plus a padding scheme.
//!
//! Job ids are assigned by the coordinator for now. They become the on-chain
//! `jobId` once the bridge exists (Track 3); everything correlates on this
//! field so that swap touches one place.

use primitives::attest::Attestation;
use primitives::wire::SealedResult;
use wincode::config::DefaultConfig;
use wincode::{SchemaRead, SchemaWrite};

/// One encrypted argument, with the commitment that pins it.
///
/// The commitment is what the bridge contract stores as `inputCommits`. A
/// worker recomputes it on receipt, so a coordinator cannot alter inputs
/// between the chain and the evaluation.
#[derive(Debug, Clone, PartialEq, Eq, SchemaWrite, SchemaRead)]
pub struct InputBlob {
    /// An encoded `CompressedFheInt32`.
    pub bytes: Vec<u8>,
    /// `keccak256(bytes)`.
    pub commitment: [u8; 32],
}

/// A unit of work handed to a worker.
#[derive(Debug, Clone, PartialEq, Eq, SchemaWrite, SchemaRead)]
pub struct JobDispatch {
    pub job_id: u64,
    /// DISCA bytecode. The worker validates it before evaluating anything.
    ///
    /// Also the program's identity: both sides take `keccak256` of these bytes
    /// and bind it into the signed attestation (`primitives::attest::Claim`),
    /// so neither has to be told what program it is running.
    pub bytecode: Vec<u8>,
    /// Which exported function of that program to run.
    pub function: String,
    pub inputs: Vec<InputBlob>,
    /// Hash of the server key the worker must be holding.
    pub server_key_hash: [u8; 32],
}

/// What a worker made of a job.
#[derive(Debug, Clone, PartialEq, Eq, SchemaWrite, SchemaRead)]
pub enum JobOutcome {
    /// The sealed result: the blob that goes on-chain and the hash committing
    /// to it, produced together so they cannot disagree.
    Evaluated(SealedResult),
    /// The worker could not evaluate. Reporting this is not optional — silence
    /// is indistinguishable from being slow, and stalls the whole job.
    Failed(String),
}

/// A worker's response to a dispatch.
#[derive(Debug, Clone, PartialEq, Eq, SchemaWrite, SchemaRead)]
pub struct JobReport {
    pub job_id: u64,
    /// The worker's signature over the claim its `outcome` amounts to (task
    /// 2.10i).
    ///
    /// This is the whole of the report's authority. Everything else in the
    /// message is either covered by the signature or is decoration: the
    /// coordinator reconstructs the claim from what it already knows (the job
    /// id it dispatched, the hash of the bytecode it compiled) plus the outcome
    /// carried here, recovers the signing address, and counts agreement over
    /// distinct *registered* addresses. Nothing a sender puts in this message
    /// can change which address comes out.
    ///
    /// This replaces the per-(job, worker) attestation token of task 2.9a. The
    /// token was a bearer secret standing in for exactly this signing key
    /// (`architecture.md` §11 Q3 said so at the time), and it could not travel
    /// past the coordinator: a contract has no way to check a secret the
    /// coordinator minted. Keeping both would leave two attribution mechanisms
    /// that can disagree, and an attacker attacks the weaker one.
    pub attestation: Attestation,
    /// What the worker calls itself. Useful in logs, and nothing more — it is
    /// self-declared, so it must never be what agreement is counted by. The
    /// address recovered from `attestation` is the identity that counts.
    pub worker: String,
    pub outcome: JobOutcome,
    /// Evaluation wall-clock, so the coordinator can see which worker is slow
    /// without correlating logs.
    pub elapsed_ms: u64,
}

/// Encodes a message for an HTTP body.
pub fn encode<T>(message: &T) -> Result<Vec<u8>, String>
where
    T: SchemaWrite<DefaultConfig, Src = T>,
{
    wincode::serialize(message).map_err(|e| format!("failed to encode message: {e:?}"))
}

/// Decodes a message from an HTTP body, rejecting trailing bytes.
pub fn decode<T>(bytes: &[u8]) -> Result<T, String>
where
    T: for<'a> SchemaRead<'a, DefaultConfig, Dst = T>,
{
    wincode::deserialize_exact(bytes).map_err(|e| format!("failed to decode message: {e:?}"))
}

#[cfg(test)]
mod tests {
    use primitives::attest::{Claim, WorkerKey};

    use super::*;

    fn dispatch() -> JobDispatch {
        JobDispatch {
            job_id: 42,
            bytecode: vec![1, 2, 3, 4],
            function: "tally4_select".into(),
            inputs: vec![InputBlob {
                bytes: vec![9; 64],
                commitment: [7u8; 32],
            }],
            server_key_hash: [3u8; 32],
        }
    }

    #[test]
    fn dispatch_round_trips() {
        let bytes = encode(&dispatch()).unwrap();
        assert_eq!(decode::<JobDispatch>(&bytes).unwrap(), dispatch());
    }

    #[test]
    fn a_successful_report_round_trips() {
        let sealed = SealedResult {
            blob: vec![5; 128],
            hash: [1u8; 32],
        };
        let key = WorkerKey::derive("worker-1");
        let report = JobReport {
            job_id: 42,
            attestation: key.attest(&Claim::Result {
                job_id: 42,
                bytecode_hash: [9u8; 32],
                result_hash: sealed.hash,
            }),
            worker: "worker-1".into(),
            outcome: JobOutcome::Evaluated(sealed),
            elapsed_ms: 1234,
        };

        let bytes = encode(&report).unwrap();
        assert_eq!(decode::<JobReport>(&bytes).unwrap(), report);
    }

    #[test]
    fn a_report_survives_the_wire_still_recovering_to_its_signer() {
        // Equality after a round trip is not quite the property that matters:
        // the signature has to still recover to the same address on the far
        // side, because that is the only thing the coordinator counts. A
        // field-order or endianness change in the codec would pass the equality
        // check above (both sides use the same codec) and break this one.
        let key = WorkerKey::derive("worker-1");
        let claim = Claim::Result {
            job_id: 42,
            bytecode_hash: [9u8; 32],
            result_hash: [1u8; 32],
        };

        let report = JobReport {
            job_id: 42,
            attestation: key.attest(&claim),
            worker: "worker-1".into(),
            outcome: JobOutcome::Evaluated(SealedResult {
                blob: vec![5; 128],
                hash: [1u8; 32],
            }),
            elapsed_ms: 1234,
        };

        let decoded: JobReport = decode(&encode(&report).unwrap()).unwrap();
        assert_eq!(
            primitives::attest::recover(&claim, &decoded.attestation).unwrap(),
            key.address()
        );
    }

    #[test]
    fn a_failure_report_round_trips() {
        let key = WorkerKey::derive("worker-2");
        let report = JobReport {
            job_id: 7,
            attestation: key.attest(&Claim::Failure {
                job_id: 7,
                bytecode_hash: [9u8; 32],
                reason_hash: [4u8; 32],
            }),
            worker: "worker-2".into(),
            outcome: JobOutcome::Failed("stack underflow at op 3".into()),
            elapsed_ms: 12,
        };

        let bytes = encode(&report).unwrap();
        assert_eq!(decode::<JobReport>(&bytes).unwrap(), report);
    }

    #[test]
    fn a_truncated_body_is_rejected() {
        // Bodies arrive over a network; a short read must not decode into
        // something that looks plausible.
        let bytes = encode(&dispatch()).unwrap();
        assert!(decode::<JobDispatch>(&bytes[..bytes.len() / 2]).is_err());
    }
}
