//! Validates the confidential committee tally end to end.
//!
//! `architecture.md` §10 proposes a committee tally as the demo, on the
//! assumption that the opcode set can express it. This test checks that
//! assumption against a real Rust program compiled by a real rustc: it loads
//! `committee-tally/committee_tally.wasm`, lowers it, evaluates it under
//! encryption, and compares the decrypted result to plain Rust.
//!
//! Regenerate the fixture with `cargo build --release` from `committee-tally/`.

use primitives::program::{DiscaProgram, Program};
use tfhe::prelude::{FheDecrypt, FheTryEncrypt};
use tfhe::{ClientKey, ConfigBuilder, FheInt32, generate_keys, set_server_key};

fn load() -> DiscaProgram {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../committee-tally/committee_tally.wasm"
    );
    let wasm = std::fs::read(path).expect("committee_tally.wasm fixture");
    DiscaProgram::from_program(&Program::from_wasm(&wasm).expect("parse tally module"))
}

fn run(program: &DiscaProgram, client_key: &ClientKey, name: &str, inputs: &[i32]) -> i32 {
    let func = program
        .function(name)
        .unwrap_or_else(|| panic!("no exported function {name}"));

    let encrypted: Vec<FheInt32> = inputs
        .iter()
        .map(|v| FheInt32::try_encrypt(*v, client_key).expect("encrypt score"))
        .collect();

    func.run(&encrypted)
        .expect("evaluate tally circuit")
        .decrypt(client_key)
}

#[test]
fn tally_circuit_runs_under_encryption() {
    let program = load();
    let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
    set_server_key(server_key);

    // Committee scores for four vendors. The winning score must come back
    // correct without any node having seen a single input.
    let scores = [71, 93, 42, 88];
    let expected = *scores.iter().max().unwrap();

    assert_eq!(
        run(&program, &client_key, "tally4_select", &scores),
        expected,
        "select-tree tally"
    );

    // The idiomatic loop-and-mutate version compiles to the same straight-line
    // circuit, so it must agree.
    assert_eq!(
        run(&program, &client_key, "tally_loop", &scores),
        expected,
        "loop-written tally"
    );

    // Winner in first and last position, to catch operand-order mistakes that a
    // middle-position winner would hide.
    assert_eq!(
        run(&program, &client_key, "tally4_select", &[99, 1, 2, 3]),
        99
    );
    assert_eq!(
        run(&program, &client_key, "tally4_select", &[1, 2, 3, 99]),
        99
    );

    // Negative scores exercise signed comparison.
    assert_eq!(
        run(&program, &client_key, "tally4_select", &[-5, -20, -3, -40]),
        -3
    );
}

#[test]
fn threshold_count_runs_under_encryption() {
    let program = load();
    let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
    set_server_key(server_key);

    // Counting is the other demo shape: it sums comparison results, so it
    // exercises the Bool -> Int coercion rather than selection.
    assert_eq!(
        run(&program, &client_key, "count_above", &[71, 93, 42, 88, 70]),
        3
    );
    assert_eq!(
        run(&program, &client_key, "count_above", &[1, 2, 3, 4, 100]),
        0
    );
    assert_eq!(
        run(&program, &client_key, "count_above", &[1, 2, 3, 4, 0]),
        4
    );
}
