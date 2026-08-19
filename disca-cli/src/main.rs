use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about)]
#[non_exhaustive]
enum Commands {
    /// Parse a WASM file and convert to DISCA VM compatible bytecode
    Parse {
        /// Input WASM file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file (optional, prints to stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    #[command(about = "Prints the version of the application.")]
    Version,
}

fn main() {
    match Commands::parse() {
        Commands::Parse { .. } => {
            // Previously this matched to `{}`: it accepted a file, did nothing,
            // and exited successfully. Failing is the honest behaviour until
            // task 4.3 implements it.
            eprintln!(
                "disca-cli parse is not implemented yet (task 4.3).\n\
                 To inspect a module today:\n  \
                 cargo run -p primitives --example inspect -- <module.wasm>"
            );
            std::process::exit(2);
        }
        Commands::Version => {
            println!("{}", env!("CARGO_PKG_VERSION"));
        }
    }
}
