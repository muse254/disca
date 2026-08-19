//! The original single-process evaluator, kept as a role.
//!
//! It exercises the whole execution core — parse, compile, keygen, the
//! ciphertext boundary, evaluation, attestation — without any network, which
//! makes it the quickest way to tell whether a failure is in the core or in the
//! transport.

use std::time::Instant;

use primitives::program::{DiscaProgram, Program};
use primitives::{bytecode, wire};
use tfhe::prelude::FheDecrypt;
use tfhe::{ConfigBuilder, FheInt32, generate_keys, set_server_key};
use tracing::{info, info_span, warn};

pub fn run() -> Result<(), String> {
    let wasm_path = format!(
        "{}/../simple-arithmetic/simple_arithmetic.wasm",
        env!("CARGO_MANIFEST_DIR")
    );

    let program = {
        let _span = info_span!("program.load", path = %wasm_path).entered();
        let started = Instant::now();

        let wasm = std::fs::read(&wasm_path).map_err(|e| format!("cannot read wasm: {e}"))?;
        let parsed = Program::from_wasm(&wasm).map_err(|e| e.to_string())?;
        let program = DiscaProgram::from_program(&parsed);
        let hash = bytecode::bytecode_hash(&program).map_err(|e| e.to_string())?;

        info!(
            bytes = wasm.len(),
            functions = program.functions().len(),
            bytecode_hash = %bytecode::hex(&hash),
            elapsed_ms = started.elapsed().as_millis(),
            "program loaded"
        );
        program
    };

    let (client_key, server_key) = {
        let _span = info_span!("keys.generate").entered();
        let started = Instant::now();
        let keys = generate_keys(ConfigBuilder::default().build());
        info!(
            elapsed_ms = started.elapsed().as_millis(),
            "keypair generated"
        );
        keys
    };
    set_server_key(server_key);

    // Inputs cross the boundary compressed and committed, exactly as they will
    // when they arrive as calldata rather than as local values.
    let inputs: Vec<FheInt32> = {
        let _span = info_span!("inputs.encrypt", count = 2).entered();
        let started = Instant::now();

        let expanded = [4i32, 7i32]
            .iter()
            .map(|value| {
                let compressed = wire::encrypt_input(*value, &client_key).expect("encrypt input");
                let encoded = wire::encode(&compressed).expect("encode input");
                let commit = wire::commitment(&encoded);

                info!(
                    bytes = encoded.len(),
                    commitment = %bytecode::hex(&commit),
                    "input committed"
                );

                let received = wire::decode(&encoded).expect("decode input");
                wire::decompress(&received)
            })
            .collect();

        info!(
            elapsed_ms = started.elapsed().as_millis(),
            "inputs encrypted"
        );
        expanded
    };

    for func in program {
        let Some(name) = func.name.as_deref() else {
            continue;
        };

        let span = info_span!("function.evaluate", function = name);
        let _enter = span.enter();
        let started = Instant::now();

        match func.run(&inputs) {
            Ok(output) => {
                let sealed = wire::seal_result(&output).map_err(|e| e.to_string())?;
                let value: i32 =
                    wire::decompress(&wire::decode(&sealed.blob).map_err(|e| e.to_string())?)
                        .decrypt(&client_key);

                info!(
                    result = value,
                    result_bytes = sealed.blob.len(),
                    result_hash = %bytecode::hex(&sealed.hash),
                    elapsed_ms = started.elapsed().as_millis(),
                    "function evaluated"
                );
            }
            Err(error) => warn!(%error, "function failed to evaluate"),
        }
    }

    Ok(())
}
