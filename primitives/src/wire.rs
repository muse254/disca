//! The ciphertext boundary between DISCA and everything outside it.
//!
//! `architecture.md` §2 fixes the rule: **compressed at the boundary,
//! uncompressed inside**. An `FheInt32` is ~258 KB, which is ~4.1M gas to put
//! on chain and pointless to ship over a wire; the same value as a
//! `CompressedFheInt32` is ~2.3 KB, and compressing or decompressing costs
//! about a millisecond. So compressed ciphertexts are the only ciphertext bytes
//! that ever cross the bridge, and nodes decompress on receipt.
//!
//! This module owns that conversion in one place, along with the two hashes the
//! protocol is built on:
//!
//! * [`commitment`] over an encoded input — what `bridge.md` §2 stores as
//!   `inputCommits`, so a coordinator cannot substitute inputs after the fact.
//! * [`SealedResult::hash`] over an evaluated result — what workers report and
//!   what `fulfillJob` compares M-of-N of.
//!
//! # Why the result hash covers the compressed form
//!
//! The attested bytes are the *compressed* result, which is also the blob the
//! contract emits. That lets `fulfillJob` require
//! `keccak256(resultBlob) == resultHash` on-chain, so the ciphertext the key
//! holder retrieves is provably the one the workers attested to. Hashing the
//! uncompressed result instead would commit to bytes that never leave the
//! worker and that no verifying party can obtain — leaving a coordinator free
//! to publish a genuinely-attested hash beside a substituted blob.
//!
//! This makes attestation depend on compression being deterministic as well as
//! evaluation. Both are verified here (`results_are_deterministic`,
//! `compression_is_deterministic`), and the failure mode if either broke is a
//! job that never reaches agreement — a timeout and refund — rather than a
//! wrong answer. See `bridge.md` §5a.
//!
//! [`SealedResult`] exists so a hash cannot be handled apart from the bytes it
//! covers: a worker seals once and reports both together.

use sha3::{Digest, Keccak256};
use tfhe::prelude::FheTryEncrypt;
use tfhe::safe_serialization::{safe_deserialize, safe_serialize};
use tfhe::{ClientKey, CompressedFheInt32, FheInt32};

use crate::program::ProgramError;

type Result<T> = std::result::Result<T, ProgramError>;

/// Upper bound accepted when decoding a ciphertext from an untrusted source.
///
/// Compressed ciphertexts measure ~2.3 KB, so this leaves ample headroom while
/// refusing to let a peer name an allocation big enough to kill the process.
pub const MAX_CIPHERTEXT_BYTES: u64 = 4 * 1024 * 1024;

/// An evaluated result together with the hash that commits to it.
///
/// The pairing is the point: `blob` is what the contract emits and the key
/// holder decrypts, `hash` is what workers attest to, and because the hash is
/// only ever derived from the blob here, the two cannot be made to disagree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealedResult {
    /// The compressed result ciphertext, as it goes on-chain.
    pub blob: Vec<u8>,
    /// `keccak256(blob)` — the attestation value, verifiable by the contract.
    pub hash: [u8; 32],
}

/// Encrypts a value into the compressed form that crosses the boundary.
///
/// Only the key holder can call this: it needs the client key, which by
/// construction never leaves them.
pub fn encrypt_input(value: i32, client_key: &ClientKey) -> Result<CompressedFheInt32> {
    CompressedFheInt32::try_encrypt(value, client_key)
        .map_err(|e| ProgramError(format!("failed to encrypt input: {e:?}")))
}

/// Encodes a compressed ciphertext for calldata, an event, or the wire.
pub fn encode(ciphertext: &CompressedFheInt32) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    safe_serialize(ciphertext, &mut out, MAX_CIPHERTEXT_BYTES)
        .map_err(|e| ProgramError(format!("failed to encode ciphertext: {e}")))?;
    Ok(out)
}

/// Decodes a compressed ciphertext received from an untrusted source.
pub fn decode(bytes: &[u8]) -> Result<CompressedFheInt32> {
    safe_deserialize(bytes, MAX_CIPHERTEXT_BYTES)
        .map_err(|e| ProgramError(format!("failed to decode ciphertext: {e}")))
}

/// Expands a boundary ciphertext into the form the evaluator operates on.
pub fn decompress(ciphertext: &CompressedFheInt32) -> FheInt32 {
    ciphertext.decompress()
}

/// Compresses and commits to an evaluated result in one step.
///
/// Two honest workers evaluating the same circuit over the same inputs seal to
/// the same bytes and therefore the same hash, which is what makes M-of-N
/// agreement meaningful without any proof machinery.
pub fn seal_result(result: &FheInt32) -> Result<SealedResult> {
    let blob = encode(&result.compress())?;
    let hash = commitment(&blob);
    Ok(SealedResult { blob, hash })
}

/// `keccak256` over encoded bytes — the EVM hash, so a contract can recompute it.
pub fn commitment(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    use tfhe::prelude::FheDecrypt;
    use tfhe::{ConfigBuilder, generate_keys, set_server_key};

    use crate::program::{DiscaProgram, Program};

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

    /// Encrypts, encodes, decodes and expands — the trip an input makes from the
    /// key holder to a worker.
    fn deliver(value: i32, client_key: &ClientKey) -> FheInt32 {
        let compressed = encrypt_input(value, client_key).unwrap();
        let encoded = encode(&compressed).unwrap();
        decompress(&decode(&encoded).unwrap())
    }

    #[test]
    fn input_survives_the_round_trip_the_bridge_puts_it_through() {
        let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
        set_server_key(server_key);

        let plain: i32 = deliver(-4321, &client_key).decrypt(&client_key);
        assert_eq!(plain, -4321);
    }

    #[test]
    fn compressed_inputs_are_calldata_sized() {
        let (client_key, _) = generate_keys(ConfigBuilder::default().build());

        let compressed = encode(&encrypt_input(7, &client_key).unwrap()).unwrap();

        // architecture.md §2 records ~2.3 KB, and the gas estimates in
        // bridge.md §1 are built on it. Bound it generously but do bound it:
        // silently drifting into six figures would break the on-chain design.
        assert!(
            compressed.len() < 8 * 1024,
            "compressed input grew to {} bytes",
            compressed.len()
        );
    }

    #[test]
    fn a_commitment_is_stable_and_input_specific() {
        let (client_key, _) = generate_keys(ConfigBuilder::default().build());

        let bytes = encode(&encrypt_input(11, &client_key).unwrap()).unwrap();
        assert_eq!(
            commitment(&bytes),
            commitment(&bytes),
            "the same bytes must commit identically"
        );

        let other = encode(&encrypt_input(12, &client_key).unwrap()).unwrap();
        assert_ne!(
            commitment(&bytes),
            commitment(&other),
            "distinct ciphertexts must not share a commitment"
        );
    }

    #[test]
    fn results_are_deterministic() {
        // The assumption the whole M-of-N attestation scheme rests on
        // (architecture.md §3): evaluation is deterministic given the same input
        // ciphertexts, so two honest workers agree byte for byte and agreement
        // is evidence of correct evaluation. Worth checking rather than
        // asserting in prose.
        let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
        set_server_key(server_key);

        let program = DiscaProgram::from_program(&Program::from_wat(MAX).unwrap());
        let func = program.function("max").unwrap();

        let inputs = vec![deliver(17, &client_key), deliver(42, &client_key)];

        // Two independent evaluations stand in for two honest workers handed
        // the same job.
        let first = seal_result(&func.run(&inputs).unwrap()).unwrap();
        let second = seal_result(&func.run(&inputs).unwrap()).unwrap();

        assert_eq!(
            first.hash, second.hash,
            "two evaluations of one circuit over one set of inputs must agree"
        );

        let plain: i32 = decompress(&decode(&first.blob).unwrap()).decrypt(&client_key);
        assert_eq!(plain, 42);
    }

    #[test]
    fn compression_is_deterministic() {
        // Load-bearing: the attested hash covers the compressed blob, so
        // compressing one result twice has to give one answer. If this ever
        // failed, honest workers would report different hashes and jobs would
        // time out rather than settle -- loud, but fatal to liveness.
        let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
        set_server_key(server_key);

        let program = DiscaProgram::from_program(&Program::from_wat(MAX).unwrap());
        let func = program.function("max").unwrap();

        let inputs = vec![deliver(3, &client_key), deliver(8, &client_key)];
        let result = func.run(&inputs).unwrap();

        assert_eq!(
            seal_result(&result).unwrap(),
            seal_result(&result).unwrap(),
            "sealing one result twice diverged"
        );
    }

    #[test]
    fn the_contract_can_verify_the_blob_against_the_attested_hash() {
        // This is what option B buys, and the check fulfillJob will perform:
        // the emitted ciphertext is provably the one the workers attested to,
        // so a coordinator cannot pair a real hash with a substituted blob.
        let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
        set_server_key(server_key);

        let program = DiscaProgram::from_program(&Program::from_wat(MAX).unwrap());
        let func = program.function("max").unwrap();

        let sealed = seal_result(
            &func
                .run(&[deliver(5, &client_key), deliver(9, &client_key)])
                .unwrap(),
        )
        .unwrap();

        assert_eq!(
            commitment(&sealed.blob),
            sealed.hash,
            "the attested hash must be recomputable from the emitted blob alone"
        );

        // A substituted blob fails that check.
        let substituted = encode(&encrypt_input(0, &client_key).unwrap()).unwrap();
        assert_ne!(commitment(&substituted), sealed.hash);

        let plain: i32 = decompress(&decode(&sealed.blob).unwrap()).decrypt(&client_key);
        assert_eq!(
            plain, 9,
            "the key holder decrypts what the network computed"
        );
    }

    #[test]
    fn result_blobs_are_larger_than_input_blobs() {
        // A fresh ciphertext compresses to a replayable PRNG seed; a computed
        // one has no seed and carries real coefficients, so it lands ~5x bigger
        // (11.8 KB measured). The gas sketch in bridge.md §1 is built on this,
        // so pin it rather than rediscovering it on-chain.
        let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
        set_server_key(server_key);

        let program = DiscaProgram::from_program(&Program::from_wat(MAX).unwrap());
        let func = program.function("max").unwrap();

        let inputs = vec![deliver(2, &client_key), deliver(6, &client_key)];
        let sealed = seal_result(&func.run(&inputs).unwrap()).unwrap();
        let input_blob = encode(&encrypt_input(2, &client_key).unwrap()).unwrap();

        assert!(
            sealed.blob.len() > input_blob.len(),
            "result {} vs input {}",
            sealed.blob.len(),
            input_blob.len()
        );
        assert!(
            sealed.blob.len() < 32 * 1024,
            "result blob grew to {} bytes, which would change the gas sketch",
            sealed.blob.len()
        );
    }

    #[test]
    fn different_inputs_produce_different_attestations() {
        let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
        set_server_key(server_key);

        let program = DiscaProgram::from_program(&Program::from_wat(MAX).unwrap());
        let func = program.function("max").unwrap();

        let run_with = |a: i32, b: i32| {
            let inputs = vec![deliver(a, &client_key), deliver(b, &client_key)];
            seal_result(&func.run(&inputs).unwrap()).unwrap().hash
        };

        assert_ne!(
            run_with(1, 2),
            run_with(3, 4),
            "a worker must not be able to attest to the wrong job's result"
        );
    }
}
