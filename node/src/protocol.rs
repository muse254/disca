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
    /// Identifies the attester. Phase 1 uses the address the worker is
    /// registered under on-chain rather than a signature, per
    /// `architecture.md` §11 Q3, so this field is the whole identity.
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
        let report = JobReport {
            job_id: 42,
            worker: "worker-1".into(),
            outcome: JobOutcome::Evaluated(SealedResult {
                blob: vec![5; 128],
                hash: [1u8; 32],
            }),
            elapsed_ms: 1234,
        };

        let bytes = encode(&report).unwrap();
        assert_eq!(decode::<JobReport>(&bytes).unwrap(), report);
    }

    #[test]
    fn a_failure_report_round_trips() {
        let report = JobReport {
            job_id: 7,
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
