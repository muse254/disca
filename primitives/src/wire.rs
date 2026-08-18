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
//! * [`result_hash`] over an evaluated result — what workers report and what
//!   `fulfillJob` compares M-of-N of.
//!
//! [`result_hash`] deliberately hashes the *uncompressed* result. The
//! attestation scheme rests on two honest workers producing byte-identical
//! results (`architecture.md` §3), which is a property of evaluation; folding
//! compression into the hashed bytes would make correctness depend on
//! compression being deterministic too. `results_are_deterministic` in the
//! tests pins the property the scheme actually needs.

use sha3::{Digest, Keccak256};
use tfhe::prelude::FheTryEncrypt;
use tfhe::safe_serialization::{safe_deserialize, safe_serialize};
use tfhe::{ClientKey, CompressedFheInt32, FheInt32};

use crate::program::ProgramError;

type Result<T> = std::result::Result<T, ProgramError>;

/// Upper bound accepted when decoding a ciphertext from an untrusted source.
///
/// Compressed inputs measure ~2.3 KB and uncompressed results ~258 KB, so this
/// leaves ample headroom while refusing to let a peer name an allocation big
/// enough to kill the process.
pub const MAX_CIPHERTEXT_BYTES: u64 = 4 * 1024 * 1024;

/// Encrypts a value into the compressed form that crosses the boundary.
///
/// Only the key holder can call this: it needs the client key, which by
/// construction never leaves them.
pub fn encrypt_input(value: i32, client_key: &ClientKey) -> Result<CompressedFheInt32> {
    CompressedFheInt32::try_encrypt(value, client_key)
        .map_err(|e| ProgramError(format!("failed to encrypt input: {e:?}")))
}

/// Encodes a compressed input for calldata, an event, or the wire.
pub fn encode_input(input: &CompressedFheInt32) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    safe_serialize(input, &mut out, MAX_CIPHERTEXT_BYTES)
        .map_err(|e| ProgramError(format!("failed to encode input ciphertext: {e}")))?;
    Ok(out)
}

/// Decodes a compressed input received from an untrusted source.
pub fn decode_input(bytes: &[u8]) -> Result<CompressedFheInt32> {
    safe_deserialize(bytes, MAX_CIPHERTEXT_BYTES)
        .map_err(|e| ProgramError(format!("failed to decode input ciphertext: {e}")))
}

/// Expands a boundary ciphertext into the form the evaluator operates on.
pub fn decompress(input: &CompressedFheInt32) -> FheInt32 {
    input.decompress()
}

/// Encodes an evaluated result. This is the uncompressed form, which is what
/// [`result_hash`] covers; compress separately if the result is going on-chain.
pub fn encode_result(result: &FheInt32) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    safe_serialize(result, &mut out, MAX_CIPHERTEXT_BYTES)
        .map_err(|e| ProgramError(format!("failed to encode result ciphertext: {e}")))?;
    Ok(out)
}

/// Compresses an evaluated result for the trip back across the boundary.
pub fn compress_result(result: &FheInt32) -> CompressedFheInt32 {
    result.compress()
}

/// `keccak256` over encoded bytes — the EVM hash, so a contract can recompute it.
pub fn commitment(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// The value a worker attests to: `keccak256` of the encoded result ciphertext.
///
/// Two honest workers evaluating the same circuit over the same inputs produce
/// the same bytes here, which is what makes M-of-N agreement meaningful without
/// any proof machinery.
pub fn result_hash(result: &FheInt32) -> Result<[u8; 32]> {
    Ok(commitment(&encode_result(result)?))
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

    #[test]
    fn input_survives_the_round_trip_the_bridge_puts_it_through() {
        let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
        set_server_key(server_key);

        // encrypt -> encode -> (calldata) -> decode -> decompress -> evaluate
        let compressed = encrypt_input(-4321, &client_key).unwrap();
        let bytes = encode_input(&compressed).unwrap();
        let decoded = decode_input(&bytes).unwrap();
        let expanded = decompress(&decoded);

        let plain: i32 = expanded.decrypt(&client_key);
        assert_eq!(plain, -4321);
    }

    #[test]
    fn compressed_inputs_are_calldata_sized() {
        let (client_key, _) = generate_keys(ConfigBuilder::default().build());

        let compressed = encode_input(&encrypt_input(7, &client_key).unwrap()).unwrap();

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

        let bytes = encode_input(&encrypt_input(11, &client_key).unwrap()).unwrap();
        assert_eq!(
            commitment(&bytes),
            commitment(&bytes),
            "the same bytes must commit identically"
        );

        let other = encode_input(&encrypt_input(12, &client_key).unwrap()).unwrap();
        assert_ne!(
            commitment(&bytes),
            commitment(&other),
            "distinct ciphertexts must not share a commitment"
        );
    }

    #[test]
    fn results_are_deterministic() {
        // This is the assumption the whole M-of-N attestation scheme rests on
        // (architecture.md §3): evaluation is deterministic given the same input
        // ciphertexts, so two honest workers agree byte for byte and agreement
        // is evidence of correct evaluation. Worth checking rather than
        // asserting in prose.
        let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
        set_server_key(server_key);

        let program = DiscaProgram::from_program(&Program::from_wat(MAX).unwrap());
        let func = program.function("max").unwrap();

        let inputs: Vec<FheInt32> = [17, 42]
            .iter()
            .map(|v| decompress(&encrypt_input(*v, &client_key).unwrap()))
            .collect();

        // Two independent evaluations stand in for two honest workers handed
        // the same job.
        let first = func.run(&inputs).unwrap();
        let second = func.run(&inputs).unwrap();

        assert_eq!(
            result_hash(&first).unwrap(),
            result_hash(&second).unwrap(),
            "two evaluations of one circuit over one set of inputs must agree"
        );

        let plain: i32 = first.decrypt(&client_key);
        assert_eq!(plain, 42);
    }

    #[test]
    fn compression_is_not_relied_on_being_deterministic() {
        // result_hash covers the uncompressed result specifically so that
        // attestation depends only on evaluation being deterministic. This test
        // records what compression actually does, so the choice is grounded:
        // if compressing the same result twice ever diverges, hashing the
        // compressed form would silently break M-of-N agreement.
        let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
        set_server_key(server_key);

        let program = DiscaProgram::from_program(&Program::from_wat(MAX).unwrap());
        let func = program.function("max").unwrap();

        let inputs: Vec<FheInt32> = [3, 8]
            .iter()
            .map(|v| decompress(&encrypt_input(*v, &client_key).unwrap()))
            .collect();
        let result = func.run(&inputs).unwrap();

        let a = encode_input(&compress_result(&result)).unwrap();
        let b = encode_input(&compress_result(&result)).unwrap();

        // Measured: compression is deterministic today. That is recorded as a
        // canary rather than depended on -- if it ever stops holding, the
        // alternative design noted in bridge.md (hash the compressed blob so
        // the contract can verify it against the emitted calldata) becomes
        // unsafe, while result_hash keeps working.
        assert_eq!(a, b, "compressing one result twice diverged");

        assert_eq!(
            result_hash(&result).unwrap(),
            result_hash(&result).unwrap(),
            "the attested hash must not depend on compression either way"
        );
    }

    #[test]
    fn different_inputs_produce_different_attestations() {
        let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
        set_server_key(server_key);

        let program = DiscaProgram::from_program(&Program::from_wat(MAX).unwrap());
        let func = program.function("max").unwrap();

        let run_with = |a: i32, b: i32| {
            let inputs: Vec<FheInt32> = [a, b]
                .iter()
                .map(|v| decompress(&encrypt_input(*v, &client_key).unwrap()))
                .collect();
            result_hash(&func.run(&inputs).unwrap()).unwrap()
        };

        assert_ne!(
            run_with(1, 2),
            run_with(3, 4),
            "a worker must not be able to attest to the wrong job's result"
        );
    }

    #[test]
    fn results_compress_for_the_trip_back() {
        let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
        set_server_key(server_key);

        let program = DiscaProgram::from_program(&Program::from_wat(MAX).unwrap());
        let func = program.function("max").unwrap();

        let inputs: Vec<FheInt32> = [5, 9]
            .iter()
            .map(|v| decompress(&encrypt_input(*v, &client_key).unwrap()))
            .collect();
        let result = func.run(&inputs).unwrap();

        let compressed = compress_result(&result);
        let round_tripped = decode_input(&encode_input(&compressed).unwrap()).unwrap();
        let plain: i32 = decompress(&round_tripped).decrypt(&client_key);

        assert_eq!(
            plain, 9,
            "the key holder decrypts what the network computed"
        );
        assert!(
            encode_input(&compressed).unwrap().len() < encode_result(&result).unwrap().len(),
            "compression must actually shrink the result"
        );
    }
}
