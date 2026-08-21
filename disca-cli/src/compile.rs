//! wasm in, bytecode out — and the last place a bad circuit can be stopped.
//!
//! `bytecode::serialize` will happily encode a program that cannot run: it
//! writes down opcodes and signatures, and asks no questions about whether the
//! stack balances or a local index is in range. `bytecode::deserialize` *does*
//! validate, which means the first party to find out is a worker, minutes into
//! a job, after a coordinator has already fanned it out and (once there is a
//! chain) after `registerProgram` has pinned the hash of the broken blob
//! forever. So this command validates before it writes. It is the last point
//! before a circuit reaches the network at which failing is cheap.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use primitives::bytecode::{self, hex};
use primitives::program::{DiscaProgram, Program};
use primitives::validate;

/// What a compilation produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compiled {
    pub output: PathBuf,
    /// `keccak256(bytecode)` — the `bytecodeHash` `registerProgram` pins, and
    /// the value every worker independently recomputes from the bytes it was
    /// handed.
    pub bytecode_hash: [u8; 32],
    /// How many functions the module lowered to. Reported because a module that
    /// silently lowered to fewer functions than its author wrote is a real
    /// failure mode of the wasm frontend.
    pub functions: usize,
}

impl Compiled {
    /// The stdout contract. One line, greppable, because registering a program
    /// means pasting this hash into a transaction.
    pub fn write_report(&self, out: &mut impl Write) -> io::Result<()> {
        writeln!(out, "bytecode_hash={}", hex(&self.bytecode_hash))
    }
}

/// Lowers a wasm module to DISCA bytecode and writes it to `output`.
pub fn compile(input: &Path, output: &Path) -> Result<Compiled> {
    let wasm = fs::read(input).with_context(|| format!("cannot read {}", input.display()))?;

    let parsed = Program::from_wasm(&wasm)
        .with_context(|| format!("cannot lower {} to a circuit", input.display()))?;
    let program = DiscaProgram::from_program(&parsed);

    if program.functions().is_empty() {
        bail!(
            "{} defines no functions; there is nothing to evaluate",
            input.display()
        );
    }

    // Every function, not just the exported ones. `bytecode::deserialize`
    // validates the whole function list, so a worker rejects the entire program
    // over one unrunnable circuit — including a private helper the module never
    // exported and nobody would ever ask for by name. Checking the same set
    // here is what makes "it compiled" mean "a worker will accept it".
    for (index, func) in program.functions().iter().enumerate() {
        let name = func.name.as_deref().unwrap_or("<anonymous>");
        validate::validate(func).map_err(|e| {
            anyhow!(
                "{}: function {index} ({name}) is not a valid circuit: {e}",
                input.display()
            )
        })?;
    }

    let bytecode = bytecode::serialize(&program)
        .with_context(|| format!("cannot encode {} as bytecode", input.display()))?;
    fs::write(output, &bytecode).with_context(|| format!("cannot write {}", output.display()))?;

    Ok(Compiled {
        output: output.to_path_buf(),
        bytecode_hash: bytecode::hash_bytecode(&bytecode),
        functions: program.functions().len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testing::{TempDir, wasm};

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
    fn a_compiled_module_is_bytecode_a_worker_would_accept() {
        let dir = TempDir::new("compile");
        let module = dir.write("max.wasm", &wasm(MAX));
        let output = dir.path().join("max.disca");

        let compiled = compile(&module, &output).expect("compile");
        assert_eq!(compiled.functions, 1);

        let bytes = fs::read(&output).expect("bytecode written");
        assert_eq!(bytecode::hash_bytecode(&bytes), compiled.bytecode_hash);

        // `deserialize` is the worker's side of this exchange: it re-validates
        // everything, so a round trip through it is the real assertion that the
        // blob is usable rather than merely well formed.
        let program = bytecode::deserialize(&bytes).expect("a worker must accept this");
        assert!(program.function("max").is_some());

        let mut report = Vec::new();
        compiled.write_report(&mut report).unwrap();
        assert_eq!(
            String::from_utf8(report).unwrap(),
            format!("bytecode_hash={}\n", hex(&compiled.bytecode_hash))
        );
    }

    #[test]
    fn the_hash_is_the_one_registerprogram_would_pin() {
        // Two compilations of one module must agree, or the hash pinned at
        // registration is not the hash the next build produces and no program
        // could ever be re-registered from source.
        let dir = TempDir::new("compile-stable");
        let module = dir.write("max.wasm", &wasm(MAX));

        let first = compile(&module, &dir.path().join("a.disca")).unwrap();
        let second = compile(&module, &dir.path().join("b.disca")).unwrap();

        assert_eq!(first.bytecode_hash, second.bytecode_hash);
        assert_eq!(hex(&first.bytecode_hash).len(), 66);
    }

    #[test]
    fn an_unrunnable_circuit_is_refused_by_name_and_nothing_is_written() {
        // `i32.add` with one operand on the stack: encodes fine, validates
        // never. Without the check here this reaches a worker as bytecode and
        // fails there, after a job has been dispatched.
        let underflow = r#"
        (module
            (func $lopsided (param i32) (result i32)
              local.get 0
              i32.add
            )
            (export "lopsided" (func $lopsided))
        )
        "#;

        let dir = TempDir::new("compile-invalid");
        let module = dir.write("bad.wasm", &wasm(underflow));
        let output = dir.path().join("bad.disca");

        let err = compile(&module, &output).expect_err("must refuse");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains("lopsided"),
            "the error must name the offending function: {rendered}"
        );
        assert!(
            !output.exists(),
            "a refused compile must not leave a blob behind for someone to ship"
        );
    }

    #[test]
    fn a_private_helper_is_validated_too() {
        // Not exported, so nobody can name it in a job -- but `deserialize`
        // validates the whole function list, so a worker rejects the program
        // because of it. Catching it here is the difference between a build
        // error and a job that dies on every worker at once.
        let hidden = r#"
        (module
            (func $helper (param i32) (result i32)
              local.get 0
              i32.add
            )
            (func $visible (param i32) (result i32)
              local.get 0
            )
            (export "visible" (func $visible))
        )
        "#;

        let dir = TempDir::new("compile-private");
        let module = dir.write("hidden.wasm", &wasm(hidden));

        let err = compile(&module, &dir.path().join("out.disca")).expect_err("must refuse");
        assert!(
            format!("{err:#}").contains("<anonymous>"),
            "an unexported function has no name to report: {err:#}"
        );
    }

    #[test]
    fn errors_name_the_file_that_failed() {
        let dir = TempDir::new("compile-missing");
        let absent = dir.path().join("nowhere.wasm");
        let err = compile(&absent, &dir.path().join("out.disca")).expect_err("no such file");
        assert!(
            format!("{err:#}").contains(&absent.display().to_string()),
            "got: {err:#}"
        );

        let garbage = dir.write("garbage.wasm", b"this is not a wasm module");
        let err = compile(&garbage, &dir.path().join("out.disca")).expect_err("not wasm");
        assert!(
            format!("{err:#}").contains(&garbage.display().to_string()),
            "got: {err:#}"
        );
    }
}
