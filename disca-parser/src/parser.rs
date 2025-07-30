//! WASM parser and circuit conversion functionality

use crate::errors::{DiscaError, Result};
use crate::homomorphic::{BinaryCircuit, LogicCircuit};
use crate::optimizer::{CircuitOptimizer, OptimizationLevel};
use crate::wasm_module::WasmModule;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Output format options for circuit export
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// Human-readable JSON format
    Json,
    /// Compact binary format
    Binary,
    /// Text-based gate representation
    Text,
}

impl std::str::FromStr for OutputFormat {
    type Err = DiscaError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "json" => Ok(OutputFormat::Json),
            "binary" => Ok(OutputFormat::Binary),
            "text" => Ok(OutputFormat::Text),
            _ => Err(DiscaError::invalid_input(format!(
                "Unknown output format: {}. Supported: json, binary, text",
                s
            ))),
        }
    }
}

/// Main WASM parser for converting WASM bytecode to FHE circuits
#[derive(Debug, Default)]
pub struct WasmParser {
    optimization_level: OptimizationLevel,
}

impl WasmParser {
    /// Create a new WASM parser with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a parser with specific optimization level
    pub fn with_optimization(optimization_level: OptimizationLevel) -> Self {
        Self {
            optimization_level,
        }
    }

    /// Parse WASM bytes and return a legacy logic circuit
    pub fn parse_to_logic_circuit(&self, wasm_bytes: &[u8]) -> Result<LogicCircuit> {
        let module = self.parse_wasm_module(wasm_bytes)?;
        let mut circuit = module.to_logic_circuit();

        // Apply optimization if requested
        if self.optimization_level != OptimizationLevel::None {
            let optimizer = CircuitOptimizer::new(self.optimization_level);
            circuit = optimizer.optimize(circuit)?;
        }

        Ok(circuit)
    }

    /// Parse WASM bytes and return an optimized binary circuit
    pub fn parse_to_binary_circuit(&self, wasm_bytes: &[u8]) -> Result<BinaryCircuit> {
        let logic_circuit = self.parse_to_logic_circuit(wasm_bytes)?;
        BinaryCircuit::from_logic_circuit(&logic_circuit)
    }

    /// Parse WASM file from path
    pub fn parse_file<P: AsRef<Path>>(&self, path: P) -> Result<LogicCircuit> {
        let wasm_bytes = std::fs::read(path)?;
        self.parse_to_logic_circuit(&wasm_bytes)
    }

    /// Parse WASM file to binary circuit
    pub fn parse_file_to_binary<P: AsRef<Path>>(&self, path: P) -> Result<BinaryCircuit> {
        let wasm_bytes = std::fs::read(path)?;
        self.parse_to_binary_circuit(&wasm_bytes)
    }

    /// Get optimization statistics for the last parsed circuit
    pub fn get_optimization_stats(&self) -> Option<String> {
        // TODO: Implement statistics tracking
        None
    }

    /// Parse raw WASM module (internal implementation)
    fn parse_wasm_module(&self, wasm_bytes: &[u8]) -> Result<WasmModule> {
        let mut module = WasmModule::new();
        module.parse(wasm_bytes)?;
        Ok(module)
    }
}

/// Utility functions for circuit analysis and conversion
pub mod utils {
    use super::*;

    /// Get basic circuit statistics
    pub fn analyze_circuit(circuit: &LogicCircuit) -> String {
        format!(
            "Circuit Statistics:\n\
             - Total gates: {}\n\
             - Add gates: {}\n\
             - Multiply gates: {}\n\
             - Other gates: {}",
            circuit.gates.len(),
            circuit
                .gates
                .iter()
                .filter(|g| matches!(g, crate::homomorphic::LogicGate::Add { .. }))
                .count(),
            circuit
                .gates
                .iter()
                .filter(|g| matches!(g, crate::homomorphic::LogicGate::Multiply { .. }))
                .count(),
            circuit
                .gates
                .iter()
                .filter(|g| !matches!(
                    g,
                    crate::homomorphic::LogicGate::Add { .. }
                        | crate::homomorphic::LogicGate::Multiply { .. }
                ))
                .count(),
        )
    }

    /// Validate circuit integrity
    pub fn validate_circuit(circuit: &LogicCircuit) -> Result<()> {
        if circuit.gates.is_empty() {
            return Err(DiscaError::circuit_validation("Circuit is empty"));
        }

        // TODO: Add more validation logic
        // - Check wire connectivity
        // - Verify gate dependencies
        // - Ensure no cycles in directed graph

        Ok(())
    }

    /// Convert between different circuit formats
    pub fn convert_format(
        circuit: &LogicCircuit,
        format: OutputFormat,
    ) -> Result<String> {
        match format {
            OutputFormat::Json => {
                let binary = BinaryCircuit::from_logic_circuit(circuit)?;
                Ok(binary.to_json()?)
            }
            OutputFormat::Binary => {
                let binary = BinaryCircuit::from_logic_circuit(circuit)?;
                Ok(format!("{:?}", binary.to_storage_bytes()))
            }
            OutputFormat::Text => Ok(format!("{:#?}", circuit)),
        }
    }
}
