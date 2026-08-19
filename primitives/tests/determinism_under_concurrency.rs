//! Does concurrent evaluation still produce byte-identical results?
//!
//! `results_are_deterministic` in `primitives::wire` evaluates twice in
//! sequence on one thread and they agree. The M-of-N attestation scheme needs
//! something stronger: workers evaluate *at the same time*, on machines under
//! load, and must still land on identical bytes.
//!
//! The answer is no. tfhe-rs evaluation is **not** byte-reproducible: processes
//! given byte-identical keys and inputs intermittently produce results that
//! decrypt to the same value but differ byte for byte. Restricting evaluation
//! to one thread was tried and rejected — it holds on a six-op circuit and
//! fails on the real eighteen-op one, at ~3x the cost.
//!
//! This test states the property M-of-N attestation requires. It is `#[ignore]`d
//! because that property does not currently hold: it is a specification of what
//! a replacement scheme has to deliver (architecture.md §3, tasks 2.10a–c), not
//! a regression guard. Run it with `--ignored` to see the current behaviour;
//! `primitives/examples/cross_process.rs` reproduces it across real processes.

use std::sync::Arc;
use std::thread;

use primitives::program::{DiscaProgram, Program};
use primitives::wire;
use tfhe::{
    CompressedServerKey, ConfigBuilder, FheInt32, ServerKey, generate_keys, set_server_key,
};

const TALLY: &str = r#"
(module
    (func $max (param i32 i32 i32 i32) (result i32)
      local.get 0
      local.get 1
      local.get 0
      local.get 1
      i32.gt_s
      select
      local.get 2
      local.get 2
      local.get 2
      i32.gt_s
      select
      local.get 3
      local.get 3
      local.get 3
      i32.gt_s
      select
    )
    (export "max" (func $max))
)
"#;

#[test]
#[ignore = "tfhe-rs evaluation is not byte-reproducible; see architecture.md §3"]
fn concurrent_workers_agree_byte_for_byte() {
    // The requirement the worker role enforces at startup. Without it this test
    // is flaky by design rather than by accident.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();

    let (client_key, local_key) = generate_keys(ConfigBuilder::default().build());
    set_server_key(local_key);

    // Mirror what a worker actually installs: the compressed key is built once
    // by the coordinator, served to everyone, and each worker decompresses its
    // own copy of those same bytes.
    let served = wire::encode_server_key(&CompressedServerKey::new(&client_key)).unwrap();
    let server_key = wire::decode_server_key(&served).unwrap();

    let program = Arc::new(DiscaProgram::from_program(
        &Program::from_wat(TALLY).unwrap(),
    ));

    // Encrypt once. Every "worker" evaluates over the identical ciphertexts,
    // exactly as they would after decompressing the same dispatched blobs.
    let inputs: Arc<Vec<FheInt32>> = Arc::new(
        [71, 93, 42, 88]
            .iter()
            .map(|v| wire::decompress(&wire::encrypt_input(*v, &client_key).expect("encrypt")))
            .collect(),
    );

    let key = Arc::new(server_key);
    const WORKERS: usize = 4;
    const ROUNDS: usize = 5;

    for round in 0..ROUNDS {
        let hashes: Vec<[u8; 32]> = (0..WORKERS)
            .map(|_| {
                let program = Arc::clone(&program);
                let inputs = Arc::clone(&inputs);
                let key: ServerKey = (*key).clone();
                thread::spawn(move || {
                    // tfhe holds the server key in thread-local storage, so each
                    // worker thread installs its own.
                    set_server_key(key);
                    let func = program.function("max").unwrap();
                    let result = func.run(&inputs).expect("evaluate");
                    wire::seal_result(&result).expect("seal").hash
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().expect("worker panicked"))
            .collect();

        let first = hashes[0];
        let disagreeing: Vec<usize> = hashes
            .iter()
            .enumerate()
            .filter(|(_, h)| **h != first)
            .map(|(i, _)| i)
            .collect();

        assert!(
            disagreeing.is_empty(),
            "round {round}: workers {disagreeing:?} of {WORKERS} disagreed; M-of-N \
         attestation requires concurrent evaluation to be byte-identical.\nhashes: {}",
            hashes
                .iter()
                .map(primitives::bytecode::hex)
                .collect::<Vec<_>>()
                .join("\n        ")
        );
    }
}
