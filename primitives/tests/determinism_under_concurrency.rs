//! Guards the property M-of-N attestation depends on: two workers evaluating
//! the same circuit over the same ciphertexts produce byte-identical results.
//!
//! This has to run **across processes**. tfhe-rs caches its chosen FFT plan in a
//! process-global `OnceLock`, so any in-process test is self-consistent by
//! construction and passes whether or not the plan was pinned — it would not
//! have caught the bug this test exists to prevent (`architecture.md` §3).
//!
//! So the test re-executes its own binary as several concurrent child
//! processes, each loading the same key and input bytes from disk and reporting
//! the hash of its result. It fails if they disagree.
//!
//! It also catches a failure the version pin cannot: if `ConfigBuilder::default()`
//! ever selects a polynomial size other than the 2048 that `pin_fft_plan`
//! pins, the pin silently covers a size nothing uses and divergence returns.
//! There is no public accessor for the configured size, so this asserts the
//! consequence rather than the value.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use primitives::program::{DiscaProgram, Program};
use primitives::wire;
use tfhe::core_crypto::fft_impl::fft64::math::fft::{
    FftAlgo, Method, Plan, PolynomialSize, setup_custom_fft_plan,
};
use tfhe::{CompressedServerKey, ConfigBuilder, FheInt32, generate_keys};

/// Set on the children to tell them which role to play, and where the fixtures
/// are.
const ROLE: &str = "DISCA_DETERMINISM_CHILD";

/// Marks the child's answer in among the test harness's own output.
const MARKER: &str = "DISCA_RESULT ";

const WORKERS: usize = 3;

/// Divergence is probabilistic — unpinned, a single round of three workers
/// agreed by luck in roughly one attempt in four. Two rounds put detection near
/// 95% for about three extra seconds, which is worth it for a guard on the
/// property the whole attestation scheme rests on.
const ROUNDS: usize = 2;

/// Set this to skip pinning and watch the test fail, which is the quickest way
/// to confirm the guard still guards something. Test-only; the node has no such
/// switch.
const SKIP_PIN: &str = "DISCA_SKIP_PIN";

/// Mirrors `pin_fft_plan` in `node/src/main.rs`. Must run before anything
/// touches a key.
fn pin_fft_plan() {
    let fourier = PolynomialSize(2048).to_fourier_polynomial_size();
    setup_custom_fft_plan(Plan::new(
        fourier.0,
        Method::UserProvided {
            base_algo: FftAlgo::Dif4,
            base_n: fourier.0,
        },
    ));
}

const TALLY: &str = r#"
(module
    (func $max (param i32 i32) (result i32)
      local.get 0
      local.get 1
      local.get 0
      local.get 1
      i32.gt_s
      select
      local.get 0
      local.get 1
      local.get 0
      i32.gt_s
      select
    )
    (export "max" (func $max))
)
"#;

#[test]
fn concurrent_workers_agree_byte_for_byte() {
    if let Some(dir) = std::env::var_os(ROLE) {
        // Child: evaluate and report, then leave without running anything else.
        let hash = evaluate(Path::new(&dir));
        println!("{MARKER}{hash}");
        std::process::exit(0);
    }

    let dir = fixtures();
    let exe = std::env::current_exe().expect("test binary path");

    for round in 0..ROUNDS {
        // Spawn first, wait after, so the children genuinely overlap — the
        // divergence only appears under concurrency.
        let children: Vec<_> = (0..WORKERS)
            .map(|_| {
                let mut command = Command::new(&exe);
                command
                    .env(ROLE, &dir)
                    .args([
                        "--exact",
                        "concurrent_workers_agree_byte_for_byte",
                        "--nocapture",
                    ])
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null());
                if let Some(skip) = std::env::var_os(SKIP_PIN) {
                    command.env(SKIP_PIN, skip);
                }
                command.spawn().expect("spawn worker")
            })
            .collect();

        let hashes: Vec<String> = children
            .into_iter()
            .map(|child| {
                let out = child.wait_with_output().expect("worker finished");
                assert!(out.status.success(), "worker failed: {:?}", out.status);
                String::from_utf8_lossy(&out.stdout)
                    .lines()
                    .find_map(|line| line.strip_prefix(MARKER).map(str::to_owned))
                    .expect("worker reported no result")
            })
            .collect();

        let first = &hashes[0];
        if hashes.iter().any(|h| h != first) {
            let _ = std::fs::remove_dir_all(&dir);
            panic!(
                "round {round}: workers disagreed, so M-of-N attestation cannot \
                 settle. Either the FFT plan is no longer pinned for the \
                 polynomial size in use (see pin_fft_plan in node/src/main.rs), \
                 or evaluation has stopped being reproducible. \
                 See architecture.md §3.\nhashes:\n  {}",
                hashes.join("\n  ")
            );
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Generates one key and one set of inputs for every child to share.
fn fixtures() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("disca-determinism-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create fixture dir");

    let (client_key, _) = generate_keys(ConfigBuilder::default().build());

    let server_key = wire::encode_server_key(&CompressedServerKey::new(&client_key)).unwrap();
    std::fs::write(dir.join("server_key.bin"), &server_key).unwrap();

    for (i, value) in [71i32, 93i32].iter().enumerate() {
        let blob = wire::encode(&wire::encrypt_input(*value, &client_key).unwrap()).unwrap();
        std::fs::write(dir.join(format!("input{i}.bin")), &blob).unwrap();
    }

    dir
}

/// The child's work: install the shared key, evaluate, report the hash a worker
/// would attest to.
fn evaluate(dir: &Path) -> String {
    if std::env::var_os(SKIP_PIN).is_none() {
        pin_fft_plan();
    }

    let server_key = std::fs::read(dir.join("server_key.bin")).expect("server key fixture");
    tfhe::set_server_key(wire::decode_server_key(&server_key).unwrap());

    let inputs: Vec<FheInt32> = (0..2)
        .map(|i| {
            let blob = std::fs::read(dir.join(format!("input{i}.bin"))).unwrap();
            wire::decompress(&wire::decode(&blob).unwrap())
        })
        .collect();

    let program = DiscaProgram::from_program(&Program::from_wat(TALLY).unwrap());
    let result = program.function("max").unwrap().run(&inputs).unwrap();

    primitives::bytecode::hex(&wire::seal_result(&result).unwrap().hash)
}
