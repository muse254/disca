use std::path::PathBuf;

use clap::Parser;

mod parser;

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
        Commands::Parse { .. } => {}
        Commands::Version => {
            println!("{}", env!("CARGO_PKG_VERSION"));
        }
    }
}
