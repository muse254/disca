//! The key holder, as a program you run rather than a struct inside a node.
//!
//! Until task 4.3 the coordinator *was* the key holder: `KeyHolder::new` in
//! `node/src/coordinator.rs` minted a keypair on every start, took plaintext
//! inputs on the command line as `--inputs 71,93,42,88`, encrypted them itself
//! and decrypted the winning result. Two separate things are wrong with that:
//!
//! * **A node saw plaintext.** `architecture.md` §3 makes the key holder a
//!   party distinct from every node in the network, and the claim that no node
//!   sees plaintext is worth nothing when the process fanning the job out is
//!   holding the client key.
//! * **A coordinator that mints its own key cannot join the on-chain key
//!   lifecycle.** `registerProgram` pins a `serverKeyHash` once, before any job
//!   exists (`bridge.md` §2). A process that generates a fresh keypair on every
//!   start produces a different hash every start, so it could never match the
//!   one that was pinned.
//!
//! This crate is that separate party, and it never talks to a worker. It only
//! reads and writes files: a keypair, a compiled program, encrypted inputs, and
//! a plaintext read back out of a result blob. Whatever carries those bytes to
//! the network — a coordinator process today, `submitJob` calldata once there
//! is a chain — does so without ever holding the client key.
//!
//! # Why the commands are separate invocations
//!
//! The key holder's work is spread across time: inputs are encrypted when a job
//! is submitted, and the result is decrypted whenever the network settles it,
//! which may be minutes or hours later. Nothing should have to stay running in
//! between. That is why `primitives::wire` had to learn to encode a client key
//! at all — a key that could not survive between two `disca-cli` runs could not
//! decrypt the result it encrypted the inputs for.
//!
//! # What is on stdout
//!
//! Each command prints machine-readable lines and nothing else, because the
//! things it prints — a server key hash, input commitments, a bytecode hash —
//! are exactly the values `registerProgram` and `submitJob` carry, and a demo
//! script has to be able to capture them. Nothing here ever prints key
//! material, and no error message quotes a plaintext value.

pub mod compile;
pub mod keys;

/// Scratch-directory and wat-to-wasm helpers shared by the module test suites.
///
/// Deliberately an inline `#[cfg(test)]` module rather than a `tests/` file or
/// a `dev-dependency` on `tempfile`: `scripts/lib/coverage_report.py` drops
/// everything below the first `#[cfg(test)]` in a file, so helpers that live
/// here cannot pad the covered-lines numerator, and a helper in its own file
/// would.
#[cfg(test)]
mod testing {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A directory under the system temp root that removes itself on drop.
    pub struct TempDir(PathBuf);

    impl TempDir {
        pub fn new(label: &str) -> Self {
            // The pid keeps concurrent `cargo test` runs apart and the counter
            // keeps parallel test threads apart; both are needed, because the
            // suite is threaded and CI is not the only thing running it.
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let unique = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("disca-cli-{label}-{}-{unique}", std::process::id()));
            std::fs::create_dir_all(&path).expect("create scratch directory");
            Self(path)
        }

        pub fn path(&self) -> &Path {
            &self.0
        }

        /// Writes a fixture into the directory and returns its path.
        pub fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).expect("write fixture");
            path
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Assembles a `.wat` fixture into the wasm bytes `compile` expects.
    ///
    /// Fixtures are written as text so a reviewer can see the circuit under
    /// test; `compile` takes a binary module because that is what a toolchain
    /// emits.
    pub fn wasm(wat: &str) -> Vec<u8> {
        let buffer = wast::parser::ParseBuffer::new(wat).expect("parse wat");
        let mut module: wast::Wat<'_> = wast::parser::parse(&buffer).expect("invalid wat");
        module.encode().expect("encode wat")
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use primitives::bytecode;
    use primitives::wire;
    use tfhe::set_server_key;

    use crate::testing::{TempDir, wasm};
    use crate::{compile, keys};

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

    /// The whole point of task 4.3 in one test: a key holder that generates a
    /// keypair, compiles a program, encrypts inputs, hands only public bytes to
    /// something that evaluates, and reads the answer back — without that
    /// evaluator ever holding the client key.
    ///
    /// The middle step stands in for the network using `primitives` directly.
    /// It is not a mock: it is exactly what a worker does — verify the server
    /// key against the hash it was given, decode the bytecode, expand the
    /// inputs, evaluate, seal. If any of the file formats this crate writes
    /// were wrong, this is where it would show.
    #[test]
    fn a_job_round_trips_without_the_evaluator_holding_the_client_key() {
        let dir = TempDir::new("end-to-end");

        let keys = keys::keygen(&dir.path().join("keys"), false).expect("keygen");
        let module = dir.write("max.wasm", &wasm(MAX));
        let compiled = compile::compile(&module, &dir.path().join("max.disca")).expect("compile");
        let inputs = keys::encrypt(&keys.client_key_path, &[71, 93], &dir.path().join("inputs"))
            .expect("encrypt");

        // --- everything below here is what a worker sees. No client key. ---

        let server_bytes = fs::read(&keys.server_key_path).unwrap();
        assert_eq!(
            wire::commitment(&server_bytes),
            keys.server_key_hash,
            "a worker addresses the key by hash and checks before installing"
        );
        set_server_key(wire::decode_server_key(&server_bytes).unwrap());

        let program = bytecode::deserialize(&fs::read(&compiled.output).unwrap())
            .expect("a worker must accept the compiled blob");
        let func = program.function("max").unwrap();

        let ciphertexts: Vec<_> = inputs
            .iter()
            .map(|input| {
                let bytes = fs::read(&input.path).unwrap();
                assert_eq!(
                    wire::commitment(&bytes),
                    input.commitment,
                    "the reported commitment must be the one submitJob would carry"
                );
                wire::decompress(&wire::decode(&bytes).unwrap())
            })
            .collect();

        let sealed = wire::seal_result(&func.run(&ciphertexts).unwrap()).unwrap();
        let result = dir.write("result.blob", &sealed.blob);

        // --- and back to the key holder, in what is a separate process. ---

        let plain =
            keys::decrypt(&keys.client_key_path, &keys.server_key_path, &result).expect("decrypt");
        assert_eq!(plain, 93);
    }
}
