//! DISCA Parser CLI
//!
//! Command-line interface for parsing WASM bytecode and converting to FHE circuits.

use clap::{Parser, Subcommand};
use disca_parser::{prelude::*, OutputFormat, WasmParser};
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "disca-parser")]
#[command(about = "WASM to FHE circuit parser and optimizer")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Parse a WASM file and convert to circuit
    Parse {
        /// Input WASM file
        #[arg(short, long)]
        input: PathBuf,

        /// Output file (optional, prints to stdout if not specified)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format
        #[arg(short, long, default_value = "json")]
        format: OutputFormat,

        /// Optimization level
        #[arg(long, default_value = "basic")]
        optimization: String,

        /// Use binary encoding
        #[arg(long)]
        binary: bool,

        /// Print statistics
        #[arg(long)]
        stats: bool,
    },

    /// Analyze a WASM file and show statistics
    Analyze {
        /// Input WASM file
        #[arg(short, long)]
        input: PathBuf,

        /// Show detailed analysis
        #[arg(long)]
        detailed: bool,
    },

    /// Validate a circuit file
    Validate {
        /// Circuit file to validate
        #[arg(short, long)]
        input: PathBuf,

        /// Input format of the circuit
        #[arg(short, long, default_value = "json")]
        format: OutputFormat,
    },

    /// Execute a circuit with test inputs
    Execute {
        /// Circuit file
        #[arg(short, long)]
        circuit: PathBuf,

        /// Input values (comma-separated)
        #[arg(short, long)]
        inputs: String,

        /// Circuit format
        #[arg(short, long, default_value = "json")]
        format: OutputFormat,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Parse {
            input,
            output,
            format,
            optimization,
            binary,
            stats,
        } => {
            let optimization_level = optimization
                .parse()
                .map_err(|e: DiscaError| anyhow::anyhow!("Invalid optimization level: {}", e))?;

            let parser = WasmParser::with_optimization(optimization_level);

            let result = if binary {
                let circuit = parser.parse_file_to_binary(&input)?;
                match format {
                    OutputFormat::Json => circuit.to_json()?,
                    OutputFormat::Binary => format!("{:?}", circuit.to_storage_bytes()),
                    OutputFormat::Text => format!("{:#?}", circuit),
                }
            } else {
                let circuit = parser.parse_file(&input)?;
                disca_parser::utils::convert_format(&circuit, format)?
            };

            if let Some(output_path) = output {
                fs::write(output_path, result)?;
            } else {
                println!("{}", result);
            }

            if stats {
                let circuit = parser.parse_file(&input)?;
                println!("\n{}", disca_parser::utils::analyze_circuit(&circuit));
            }
        }

        Commands::Analyze { input, detailed } => {
            let parser = WasmParser::new();
            let circuit = parser.parse_file(&input)?;

            println!("{}", disca_parser::utils::analyze_circuit(&circuit));

            if detailed {
                // TODO: Add more detailed analysis
                println!("\nDetailed analysis not yet implemented");
            }
        }

        Commands::Validate { input, format } => match format {
            OutputFormat::Json => {
                let content = fs::read_to_string(&input)?;
                let circuit = BinaryCircuit::from_json(&content)?;
                println!("✓ Circuit file is valid");
                println!("  - Gates: {}", circuit.gates.len());
                println!("  - Wires: {}", circuit.wire_count);
                println!("  - Inputs: {}", circuit.input_wires.len());
                println!("  - Outputs: {}", circuit.output_wires.len());
            }
            _ => {
                println!("Validation for format {:?} not yet implemented", format);
            }
        },

        Commands::Execute {
            circuit: circuit_path,
            inputs,
            format,
        } => {
            let input_values: std::result::Result<Vec<i64>, _> =
                inputs.split(',').map(|s| s.trim().parse()).collect();
            let input_values =
                input_values.map_err(|e| anyhow::anyhow!("Invalid input format: {}", e))?;

            match format {
                OutputFormat::Json => {
                    let content = fs::read_to_string(&circuit_path)?;
                    let circuit = BinaryCircuit::from_json(&content)?;
                    let outputs = circuit
                        .execute(&input_values)
                        .map_err(|e| anyhow::anyhow!("Execution error: {}", e))?;
                    println!("Outputs: {:?}", outputs);
                }
                _ => {
                    println!("Execution for format {:?} not yet implemented", format);
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_parsing() {
        // Basic smoke test for CLI argument parsing
        let cli = Cli::try_parse_from(&[
            "disca-parser",
            "parse",
            "--input",
            "test.wasm",
            "--format",
            "json",
        ]);
        assert!(cli.is_ok());
    }
}
