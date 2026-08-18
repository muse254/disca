//! DISCA node.
//!
//! Currently a single-process demo: it loads a compiled program, generates a
//! keypair, evaluates every exported function homomorphically, and decrypts the
//! results. The coordinator/worker split this eventually grows into is task 2.1.
//!
//! All output is [`tracing`] rather than `println!`, because these are the same
//! measurements a worker will have to report to a coordinator once the node is
//! distributed: what it ran, over how many ops, and how long each phase took.
//! Set `RUST_LOG` to change verbosity — `RUST_LOG=debug` adds per-circuit
//! evaluation timings, `RUST_LOG=trace` adds per-opcode timings.

use std::time::Instant;

use primitives::program::{DiscaProgram, Program};
use primitives::{bytecode, wire};
use tfhe::prelude::FheDecrypt;
use tfhe::{ConfigBuilder, FheInt32, generate_keys, set_server_key};
use tracing::{Level, info, info_span, warn};
use tracing_subscriber::EnvFilter;

fn main() {
    init_telemetry();

    let wasm_path = format!(
        "{}/../simple-arithmetic/simple_arithmetic.wasm",
        env!("CARGO_MANIFEST_DIR")
    );

    let program = {
        let _span = info_span!("program.load", path = %wasm_path).entered();
        let started = Instant::now();

        let wasm = std::fs::read(&wasm_path).expect("failed to read wasm");
        let parsed = Program::from_wasm(&wasm).expect("failed to parse wasm");
        let program = DiscaProgram::from_program(&parsed);

        // The bytecode hash is what a bridge contract pins on-chain, so it is
        // the identity a worker should log alongside anything it executes.
        let hash = bytecode::bytecode_hash(&program).expect("encode bytecode");

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
                let encoded = wire::encode_input(&compressed).expect("encode input");
                let commit = wire::commitment(&encoded);

                info!(
                    bytes = encoded.len(),
                    commitment = %bytecode::hex(&commit),
                    "input committed"
                );

                let received = wire::decode_input(&encoded).expect("decode input");
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
            // Unexported functions have no stable name to address them by, so
            // there is nothing a caller could ask us to run.
            continue;
        };

        let span = info_span!("function.evaluate", function = name);
        let _enter = span.enter();
        let started = Instant::now();

        match func.run(&inputs) {
            Ok(output) => {
                // The hash a worker would report for M-of-N attestation, and
                // the compressed blob the key holder would fetch and decrypt.
                let attestation = wire::result_hash(&output).expect("hash result");
                let returned = wire::compress_result(&output);
                let value: i32 = wire::decompress(&returned).decrypt(&client_key);

                info!(
                    result = value,
                    result_hash = %bytecode::hex(&attestation),
                    elapsed_ms = started.elapsed().as_millis(),
                    "function evaluated"
                );
            }
            Err(error) => {
                // A worker that cannot evaluate a circuit has to report the
                // failure, not abort the whole job.
                warn!(%error, "function failed to evaluate");
            }
        }
    }
}

/// Installs the tracing subscriber. `RUST_LOG` wins when set; otherwise we
/// default to `INFO`, which covers phase timings without per-op noise.
fn init_telemetry() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(Level::INFO.to_string()));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();
}
