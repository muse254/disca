//! Argument dispatch, and nothing else.
//!
//! Every decision this binary makes lives in the library beside it, so that the
//! key holder's behaviour can be tested without spawning a process and without
//! a shell. What is left here is the mapping from flags to a function call and
//! from a result to stdout — see `lib.rs` for what the commands are for.

use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use disca_cli::{compile, keys};

#[derive(Parser, Debug)]
#[command(author, version, about)]
#[non_exhaustive]
enum Commands {
    /// Generate a keypair: the client key stays here, the server key goes to
    /// the network.
    Keygen {
        /// Directory to write `client.key` and `server.key` into.
        #[arg(long)]
        out_dir: PathBuf,

        /// Replace an existing `client.key`. Without this the command refuses,
        /// because the key is the only thing that can decrypt results already
        /// computed under it.
        #[arg(long)]
        force: bool,
    },

    /// Lower a WASM module to DISCA bytecode, validating every circuit in it.
    Compile {
        /// Input WASM module.
        #[arg(short, long)]
        input: PathBuf,

        /// Where to write the bytecode.
        #[arg(short, long)]
        output: PathBuf,
    },

    /// Encrypt plaintext inputs for a job. Nothing downstream sees these
    /// values.
    Encrypt {
        /// The `client.key` written by `keygen`.
        #[arg(long)]
        client_key: PathBuf,

        /// Comma-separated plaintext inputs, in the order the circuit takes its
        /// parameters.
        ///
        /// `allow_hyphen_values` because the domain is `i32`, not `u32`:
        /// without it clap reads `--values -5,7` as an unknown short flag and
        /// the key holder cannot express half its input space. The cost is that
        /// a mistyped flag after `--values` becomes a value, which then fails
        /// to parse as an integer — a worse message, but not a wrong answer.
        #[arg(
            long,
            value_delimiter = ',',
            required = true,
            allow_hyphen_values = true
        )]
        values: Vec<i32>,

        /// Directory to write `input-0.ct`, `input-1.ct`, … into.
        #[arg(long)]
        out_dir: PathBuf,
    },

    /// Decrypt a result blob the network settled on.
    Decrypt {
        /// The `client.key` written by `keygen`.
        #[arg(long)]
        client_key: PathBuf,

        /// The `server.key` written by `keygen`. Needed because expanding the
        /// compressed result is a server-key operation; decrypting is not.
        #[arg(long)]
        server_key: PathBuf,

        /// The result blob, as emitted by `fulfillJob`.
        #[arg(short, long)]
        input: PathBuf,
    },

    #[command(about = "Prints the version of the application.")]
    Version,
}

fn run(command: Commands, out: &mut impl Write) -> Result<()> {
    match command {
        Commands::Keygen { out_dir, force } => keys::keygen(&out_dir, force)?.write_report(out)?,

        Commands::Compile { input, output } => {
            compile::compile(&input, &output)?.write_report(out)?
        }

        Commands::Encrypt {
            client_key,
            values,
            out_dir,
        } => {
            for input in keys::encrypt(&client_key, &values, &out_dir)? {
                input.write_report(out)?;
            }
        }

        // The one command whose output is a value rather than a report: it
        // prints the plaintext and nothing else, so `$(disca-cli decrypt ...)`
        // is the answer.
        Commands::Decrypt {
            client_key,
            server_key,
            input,
        } => writeln!(out, "{}", keys::decrypt(&client_key, &server_key, &input)?)?,

        Commands::Version => writeln!(out, "{}", env!("CARGO_PKG_VERSION"))?,
    }

    Ok(())
}

fn main() {
    // Errors go to stderr with their full context chain flattened onto one
    // line, so that stdout stays exactly the machine-readable contract the
    // commands document even when something fails mid-way.
    if let Err(err) = run(Commands::parse(), &mut io::stdout()) {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;

    use clap::CommandFactory;

    /// A scratch directory that removes itself on drop.
    ///
    /// The library has one of these too, in its `testing` module, but that
    /// module is `#[cfg(test)]` and this file is a separate crate that links
    /// the library's *non*-test build. Ten duplicated lines is the cheaper of
    /// the two ways out; the other is exporting test scaffolding from the
    /// library under a feature, where it would count as production code.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(label: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("disca-cli-bin-{label}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create scratch directory");
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Assembles a `.wat` fixture into a wasm module on disk.
    fn wasm_module(dir: &Path, name: &str, wat: &str) -> PathBuf {
        let buffer = wast::parser::ParseBuffer::new(wat).expect("parse wat");
        let mut module: wast::Wat<'_> = wast::parser::parse(&buffer).expect("invalid wat");
        let path = dir.join(name);
        std::fs::write(&path, module.encode().expect("encode wat")).expect("write module");
        path
    }

    /// Runs all four working commands through `run` the way `main` does, and
    /// asserts what lands on stdout line for line.
    ///
    /// This is the contract the node side is written against: `keygen` emits
    /// the `serverKeyHash` a `registerProgram` transaction carries, `encrypt`
    /// emits the `inputCommits` array in order, and `decrypt` emits a bare
    /// integer so `$(disca-cli decrypt ...)` is the answer and not the answer
    /// plus a banner. Testing the library functions alone would leave the
    /// dispatch free to wire the wrong one to the wrong flag.
    #[test]
    fn every_command_prints_the_contract_the_node_side_parses() {
        let scratch = Scratch::new("surface");
        let dir = &scratch.0;

        let mut out = Vec::new();
        run(
            Commands::Keygen {
                out_dir: dir.join("keys"),
                force: false,
            },
            &mut out,
        )
        .expect("keygen");
        let printed = String::from_utf8(out).unwrap();
        let hash = printed
            .strip_prefix("server_key_hash=0x")
            .and_then(|rest| rest.strip_suffix('\n'))
            .unwrap_or_else(|| panic!("keygen printed {printed:?}"));
        assert_eq!(hash.len(), 64, "a keccak256 in hex");

        let module = wasm_module(
            dir,
            "identity.wasm",
            r#"
            (module
                (func $identity (param i32) (result i32)
                  local.get 0
                )
                (export "identity" (func $identity))
            )
            "#,
        );
        let mut out = Vec::new();
        run(
            Commands::Compile {
                input: module,
                output: dir.join("identity.disca"),
            },
            &mut out,
        )
        .expect("compile");
        let printed = String::from_utf8(out).unwrap();
        assert!(
            printed.starts_with("bytecode_hash=0x") && printed.len() == "bytecode_hash=".len() + 67,
            "compile printed {printed:?}"
        );

        let mut out = Vec::new();
        run(
            Commands::Encrypt {
                client_key: dir.join("keys").join("client.key"),
                values: vec![71, -93],
                out_dir: dir.join("inputs"),
            },
            &mut out,
        )
        .expect("encrypt");
        let printed = String::from_utf8(out).unwrap();
        let lines: Vec<&str> = printed.lines().collect();
        assert_eq!(lines.len(), 2, "one line per input: {printed:?}");
        for line in &lines {
            assert!(
                line.starts_with("commitment=0x") && line.len() == 13 + 64,
                "{line}"
            );
        }
        assert_ne!(lines[0], lines[1]);

        // Decrypting `input-1.ct` and getting -93 back is what makes the
        // commitment lines above meaningful as an *ordered* array: file N holds
        // argument N.
        let mut out = Vec::new();
        run(
            Commands::Decrypt {
                client_key: dir.join("keys").join("client.key"),
                server_key: dir.join("keys").join("server.key"),
                input: dir.join("inputs").join("input-1.ct"),
            },
            &mut out,
        )
        .expect("decrypt");
        assert_eq!(
            String::from_utf8(out).unwrap(),
            "-93\n",
            "decrypt must print the plaintext and nothing else"
        );
    }

    #[test]
    fn the_argument_definition_is_well_formed() {
        // clap's own consistency checks -- duplicate flags, a long name that
        // collides, an argument that is required and defaulted at once. Running
        // them here means a broken definition fails a test rather than the
        // first person to run the binary.
        Commands::command().debug_assert();
    }

    #[test]
    fn version_prints_only_the_version() {
        let mut out = Vec::new();
        run(Commands::Version, &mut out).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            format!("{}\n", env!("CARGO_PKG_VERSION"))
        );
    }

    #[test]
    fn comma_separated_values_parse_into_one_list_per_flag() {
        // `--values 71,93,42,88` is a single flag carrying four inputs, matching
        // how the coordinator's `--inputs` reads today. Worth pinning: if this
        // silently became four separate values the ordering contract that
        // `inputCommits` depends on would be the thing that broke.
        let parsed = Commands::parse_from([
            "disca-cli",
            "encrypt",
            "--client-key",
            "k",
            "--values",
            "71,93,42,88",
            "--out-dir",
            "d",
        ]);

        match parsed {
            Commands::Encrypt { values, .. } => assert_eq!(values, vec![71, 93, 42, 88]),
            other => panic!("parsed as {other:?}"),
        }
    }

    #[test]
    fn negative_inputs_survive_argument_parsing() {
        // The evaluator's domain is `i32`. A CLI that could only express the
        // non-negative half of it would silently put half the input space out
        // of reach of the only party allowed to encrypt.
        let parsed = Commands::parse_from([
            "disca-cli",
            "encrypt",
            "--client-key",
            "k",
            "--values",
            "-5,7,-2147483648",
            "--out-dir",
            "d",
        ]);

        match parsed {
            Commands::Encrypt { values, .. } => assert_eq!(values, vec![-5, 7, i32::MIN]),
            other => panic!("parsed as {other:?}"),
        }
    }
}
