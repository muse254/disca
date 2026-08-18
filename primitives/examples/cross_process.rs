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
use tfhe::{CompressedServerKey, ConfigBuilder, FheInt32, generate_keys, set_server_key};

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

    for (i, value) in [71i32, 93i32].iter().enumerate() {
        let blob = wire::encode(&wire::encrypt_input(*value, &client_key).unwrap()).unwrap();
        std::fs::write(dir.join(format!("input{i}.bin")), &blob).unwrap();
    }

    println!("setup: server_key {} bytes", server_key.len());
}

fn eval(dir: &Path) {
    let server_key = std::fs::read(dir.join("server_key.bin")).expect("run setup first");
    set_server_key(wire::decode_server_key(&server_key).unwrap());

    let inputs: Vec<FheInt32> = (0..2)
        .map(|i| {
            let blob = std::fs::read(dir.join(format!("input{i}.bin"))).unwrap();
            wire::decompress(&wire::decode(&blob).unwrap())
        })
        .collect();

    let program = DiscaProgram::from_program(&Program::from_wat(TALLY).unwrap());
    let result = program.function("max").unwrap().run(&inputs).unwrap();
    let sealed = wire::seal_result(&result).unwrap();

    println!("{}", bytecode::hex(&sealed.hash));
}
