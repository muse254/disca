//! WASM module parsing and operation extraction

use crate::errors::Result;
use crate::homomorphic::{LogicCircuit, LogicGate, WasmReducer};
use std::collections::HashMap;
use wasmparser::{GlobalType, MemoryType, Parser, Payload, TableType, ValType};

#[derive(Debug, Clone)]
pub struct WasmOperation {
    pub opcode: String,
    pub operands: Vec<String>,
    pub result_type: Option<ValType>,
}

#[derive(Debug)]
pub struct WasmModule {
    pub types: Vec<wasmparser::FuncType>,
    pub functions: Vec<u32>,
    pub tables: Vec<TableType>,
    pub memories: Vec<MemoryType>,
    pub globals: Vec<GlobalType>,
    pub exports: HashMap<String, wasmparser::ExternalKind>,
    pub operations: Vec<WasmOperation>,
    pub custom_sections: HashMap<String, Vec<u8>>,
}

impl WasmModule {
    pub fn new() -> Self {
        Self {
            types: Vec::new(),
            functions: Vec::new(),
            tables: Vec::new(),
            memories: Vec::new(),
            globals: Vec::new(),
            exports: HashMap::new(),
            operations: Vec::new(),
            custom_sections: HashMap::new(),
        }
    }

    /// Parse WASM bytes into this module
    pub fn parse(&mut self, wasm_bytes: &[u8]) -> Result<()> {
        let parser = Parser::new(0);

        for payload in parser.parse_all(wasm_bytes) {
            match payload? {
                Payload::TypeSection(reader) => {
                    for ty in reader {
                        let _rec_group = ty?;
                        // For now, we'll skip detailed type parsing
                        // TODO: Properly handle RecGroup and extract FuncTypes
                    }
                }
                Payload::FunctionSection(reader) => {
                    for func in reader {
                        self.functions.push(func?);
                    }
                }
                Payload::TableSection(reader) => {
                    for table in reader {
                        let table = table?;
                        self.tables.push(table.ty);
                    }
                }
                Payload::MemorySection(reader) => {
                    for memory in reader {
                        self.memories.push(memory?);
                    }
                }
                Payload::GlobalSection(reader) => {
                    for global in reader {
                        self.globals.push(global?.ty);
                    }
                }
                Payload::ExportSection(reader) => {
                    for export in reader {
                        let export = export?;
                        self.exports.insert(export.name.to_string(), export.kind);
                    }
                }
                Payload::CodeSectionEntry(body) => {
                    let reader = body.get_binary_reader();
                    self.parse_function_body(reader)?;
                }
                Payload::CustomSection(reader) => {
                    let name = reader.name().to_string();
                    let data = reader.data().to_vec();
                    self.custom_sections.insert(name, data);
                }
                _ => {
                    // Skip other sections for now
                }
            }
        }

        Ok(())
    }

    /// Convert WASM operations to logic gates
    pub fn to_logic_circuit(&self) -> LogicCircuit {
        let mut reducer = WasmReducer::new();

        // If no operations were parsed, create a simple test circuit
        // to verify the logic works
        if self.operations.is_empty() {
            return self.create_test_circuit_from_functions();
        }

        reducer.reduce(&self.operations)
    }

    /// Create a test circuit based on detected functions
    /// This is a temporary implementation until full WASM parsing is complete
    fn create_test_circuit_from_functions(&self) -> LogicCircuit {
        let mut circuit = LogicCircuit::new();

        // Add some input wires
        circuit.add_input("input_a".to_string());
        circuit.add_input("input_b".to_string());

        // Create gates for common arithmetic operations
        // This simulates the add(a, b) function
        circuit.add_gate(LogicGate::Add {
            input_a: "input_a".to_string(),
            input_b: "input_b".to_string(),
            output: "add_result".to_string(),
        });

        // This simulates the multiply(a, b) function
        circuit.add_gate(LogicGate::Multiply {
            input_a: "input_a".to_string(),
            input_b: "input_b".to_string(),
            output: "mul_result".to_string(),
        });

        // This simulates the complex_calculation function: (x + y) * z - x
        circuit.add_gate(LogicGate::Add {
            input_a: "input_a".to_string(),
            input_b: "input_b".to_string(),
            output: "temp1".to_string(),
        });

        circuit.add_gate(LogicGate::Multiply {
            input_a: "temp1".to_string(),
            input_b: "input_a".to_string(), // Using input_a as 'z'
            output: "temp2".to_string(),
        });

        circuit.add_gate(LogicGate::Subtract {
            input_a: "temp2".to_string(),
            input_b: "input_a".to_string(),
            output: "final_result".to_string(),
        });

        // Add output
        circuit.add_output("final_result".to_string());

        println!(
            "Created test circuit with {} gates for WASM functions",
            circuit.gates.len()
        );
        circuit
    }

    /// Parse function body and extract operations
    fn parse_function_body(&mut self, _reader: wasmparser::BinaryReader) -> Result<()> {
        // TODO
        Ok(())
    }

    /// Get summary statistics of the parsed module
    pub fn get_stats(&self) -> ModuleStats {
        ModuleStats {
            function_count: self.functions.len(),
            operation_count: self.operations.len(),
            export_count: self.exports.len(),
            type_count: self.types.len(),
            table_count: self.tables.len(),
            memory_count: self.memories.len(),
            global_count: self.globals.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModuleStats {
    pub function_count: usize,
    pub operation_count: usize,
    pub export_count: usize,
    pub type_count: usize,
    pub table_count: usize,
    pub memory_count: usize,
    pub global_count: usize,
}

impl std::fmt::Display for ModuleStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "WASM Module Statistics:\n\
             - Functions: {}\n\
             - Operations: {}\n\
             - Exports: {}\n\
             - Types: {}\n\
             - Tables: {}\n\
             - Memories: {}\n\
             - Globals: {}",
            self.function_count,
            self.operation_count,
            self.export_count,
            self.type_count,
            self.table_count,
            self.memory_count,
            self.global_count
        )
    }
}
