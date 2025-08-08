//! DISCA Parser Library
//!
//! A library for parsing WASM bytecode and converting it to FHE-compatible circuits.
//! Provides both legacy text-based gates and optimized binary encoding for blockchain runtimes.
//!
//! # Example
//!
//! ```no_run
//! use disca_parser::{WasmParser, OptimizationLevel, OutputFormat};
//! use std::fs;
//!
//! // Create a parser with optimization
//! let parser = WasmParser::with_optimization(OptimizationLevel::Basic);
//!
//! // Load WASM bytecode
//! let wasm_bytes = fs::read("program.wasm")?;
//!
//! // Parse to binary circuit for FHE execution
//! let binary_circuit = parser.parse_to_binary_circuit(&wasm_bytes)?;
//! println!("Generated circuit with {} gates", binary_circuit.gates.len());
//!
//! // Parse to logic circuit for analysis
//! let logic_circuit = parser.parse_to_logic_circuit(&wasm_bytes)?;
//!
//! // Convert to different output formats
//! let json_output = disca_parser::utils::convert_format(&logic_circuit, OutputFormat::Json)?;
//! let binary_output = disca_parser::utils::convert_format(&logic_circuit, OutputFormat::Binary)?;
//!
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub use crate::errors::{DiscaError, Result};
pub use crate::homomorphic::{
    BinaryCircuit, BinaryLogicGate, GateOpcode, LogicCircuit, LogicGate, WireId,
};
pub use crate::optimizer::{CircuitAnalyzer, CircuitOptimizer, OptimizationLevel};
pub use crate::wasm::parser::{utils, OutputFormat, WasmParser};
pub use crate::wasm::wasm_module::{WasmModule, WasmOperation};

mod errors;
mod homomorphic;
mod optimizer;
mod wasm;

/// Re-export commonly used types
pub mod prelude {
    pub use crate::{
        BinaryCircuit, BinaryLogicGate, DiscaError, LogicCircuit, LogicGate, OutputFormat, Result,
        WasmParser, WireId,
    };
}
