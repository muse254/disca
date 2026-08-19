//! Is evaluation reproducible across processes?
//!
//! Workers are separate processes. Everything they share arrives as bytes: the
//! same server key blob, the same input ciphertexts. If two processes given
//! byte-identical inputs produce different results, M-of-N attestation cannot
//! work, so this pins the question down with the transport removed.
//!
//! ```sh
//! cargo run --release -p primitives --example cross_process -- setup /tmp/d
//! cargo run --release -p primitives --example cross_process -- eval /tmp/d
//! cargo run --release -p primitives --example cross_process -- eval /tmp/d
//! ```

use std::path::Path;

use primitives::program::{DiscaProgram, Program};
use primitives::{bytecode, wire};
use tfhe::core_crypto::fft_impl::fft64::math::fft::{
    FftAlgo, Method, Plan, PolynomialSize, setup_custom_fft_plan,
};
use tfhe::{CompressedServerKey, ConfigBuilder, FheInt32, generate_keys, set_server_key};

/// Pins the FFT plan instead of letting tfhe benchmark one at first use.
///
/// By default `Fft::new` picks between numerically-equivalent FFT algorithms by
/// timing them for 10 ms, so the winner depends on machine load at that moment.
/// Different algorithms associate the floating-point butterflies differently,
/// which changes the result ciphertext's bytes without changing the plaintext.
/// Must run before anything touches a plan -- the setter panics if the plan for
/// that polynomial size is already initialised.
fn pin_fft_plan() {
    let n = PolynomialSize(2048);
    let fourier = n.to_fourier_polynomial_size();
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
    )
    (export "max" (func $max))
)
"#;

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let dir = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "/tmp/disca".into());
    let dir = Path::new(&dir);

    match mode.as_str() {
        "setup" => setup(dir),
        "eval" => eval(dir),
        _ => {
            eprintln!("usage: cross_process <setup|eval> <dir>");
            std::process::exit(2);
        }
    }
}

fn setup(dir: &Path) {
    std::fs::create_dir_all(dir).expect("create dir");
    let (client_key, _) = generate_keys(ConfigBuilder::default().build());

    let server_key = wire::encode_server_key(&CompressedServerKey::new(&client_key)).unwrap();
    std::fs::write(dir.join("server_key.bin"), &server_key).unwrap();

    for (i, value) in [71i32, 93i32, 42i32, 88i32].iter().enumerate() {
        let blob = wire::encode(&wire::encrypt_input(*value, &client_key).unwrap()).unwrap();
        std::fs::write(dir.join(format!("input{i}.bin")), &blob).unwrap();
    }

    println!("setup: server_key {} bytes", server_key.len());
}

fn eval(dir: &Path) {
    if std::env::var_os("PIN_FFT").is_some() {
        pin_fft_plan();
    }

    let server_key = std::fs::read(dir.join("server_key.bin")).expect("run setup first");
    set_server_key(wire::decode_server_key(&server_key).unwrap());

    let inputs: Vec<FheInt32> = (0..4)
        .map(|i| {
            let blob = std::fs::read(dir.join(format!("input{i}.bin"))).unwrap();
            wire::decompress(&wire::decode(&blob).unwrap())
        })
        .collect();

    // Use the real demo circuit when it is available; a six-op toy is not
    // representative of what workers actually evaluate.
    let wasm = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../committee-tally/committee_tally.wasm"
    );
    let (program, name) = match std::fs::read(wasm) {
        Ok(bytes) => (
            DiscaProgram::from_program(&Program::from_wasm(&bytes).unwrap()),
            "tally4_select",
        ),
        Err(_) => (
            DiscaProgram::from_program(&Program::from_wat(TALLY).unwrap()),
            "max",
        ),
    };
    let result = program.function(name).unwrap().run(&inputs).unwrap();
    let sealed = wire::seal_result(&result).unwrap();

    println!("{}", bytecode::hex(&sealed.hash));
}
