use primitives::program::{DiscaProgram, Program};
use tfhe::{
    ConfigBuilder, FheInt32, generate_keys,
    prelude::{FheDecrypt, FheTryEncrypt},
    set_server_key,
};

fn main() {
    let wasm_path = format!(
        "{}/../simple-arithmetic/simple_arithmetic.wasm",
        env!("CARGO_MANIFEST_DIR")
    );
    let wasm = std::fs::read(wasm_path).expect("failed to read wasm");
    let program = Program::from_wasm(&wasm).expect("failed to parse wasm");
    let disca = DiscaProgram::from_program(&program);

    let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
    set_server_key(server_key);

    let a = FheInt32::try_encrypt(4i32, &client_key).expect("encrypt a");
    let b = FheInt32::try_encrypt(7i32, &client_key).expect("encrypt b");

    for func in disca {
        if let Some(name) = func.name.as_deref() {
            let out = func.run(&[a.clone(), b.clone()]).expect("program run");
            let value: i32 = out.decrypt(&client_key);
            println!("{name}: {value}");
        }
    }
}
