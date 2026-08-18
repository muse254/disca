//! Inspects a compiled WASM module through the DISCA front end.
//!
//! Reports, per exported function, the lowered `CircuitOp` sequence and the
//! program's bytecode hash — or the parse error, which is the more useful
//! output when checking whether a candidate demo program is expressible at all.
//!
//! ```sh
//! cargo run -p primitives --example inspect -- path/to/module.wasm
//! ```

use primitives::program::{DiscaProgram, Program};
use primitives::{bytecode, validate};

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: inspect <module.wasm>");
        std::process::exit(2);
    };

    let wasm = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(2);
    });

    let parsed = match Program::from_wasm(&wasm) {
        Ok(program) => program,
        Err(error) => {
            // The failure is the finding: it names the opcode that the IR does
            // not yet cover.
            println!("{path}: REJECTED\n  {error}");
            std::process::exit(1);
        }
    };

    let program = DiscaProgram::from_program(&parsed);
    let hash = bytecode::bytecode_hash(&program).expect("encode bytecode");
    let bytecode_len = bytecode::serialize(&program)
        .expect("encode bytecode")
        .len();

    println!("{path}: ACCEPTED");
    println!("  wasm          {} bytes", wasm.len());
    println!("  bytecode      {bytecode_len} bytes");
    println!("  bytecode_hash {}", bytecode::hex(&hash));
    println!("  functions     {}", program.functions().len());

    for func in program.functions() {
        let name = func.name.as_deref().unwrap_or("<not exported>");
        println!(
            "\n  {name}({} param) -> {} | {} local, {} ops",
            func.sig.params.len(),
            func.sig.results.len(),
            func.locals.len(),
            func.body.len()
        );

        match validate::validate(func) {
            Ok(layout) => println!(
                "    peak stack {} ciphertext(s), {} split point(s)",
                layout.max_depth,
                layout.split_points.len()
            ),
            Err(error) => println!("    INVALID: {error}"),
        }

        for (i, op) in func.body.iter().enumerate() {
            println!("    {i:>3}  {op:?}");
        }
    }
}
