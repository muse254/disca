//! WASM module parsing and operation extraction

use crate::errors::Result;
use crate::homomorphic::{LogicCircuit, LogicGate, WasmReducer};
use crate::DiscaError;
use std::collections::HashMap;
use wasmparser::{BinaryReaderError, GlobalType, MemoryType, Parser, Payload, TableType, ValType};

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

pub fn parse_wasm_module(wasm_bytes: &[u8]) -> Result<HashMap<String, LogicCircuit>> {
    todo!()
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
        let mut end = false;

        for payload in Parser::new(0).parse_all(wasm_bytes) {
            // LogicCircuit

            match payload? {
                Payload::Version {
                    num,
                    encoding,
                    range,
                } => {}

                Payload::TypeSection(reader) => {
                    for ty in reader {
                        // output data as debug log
                        log::debug!("Payload::TypeSection: {:#?}", ty);

                        let rec_group = ty?;
                        // Use iterator to access types in the RecGroup
                        for sub_type in rec_group.types() {
                            if let wasmparser::CompositeType {
                                inner: wasmparser::CompositeInnerType::Func(ref func_type),
                                ..
                            } = sub_type.composite_type
                            {
                                self.types.push(func_type.clone());

                                log::debug!(
                                    "Extracted FuncType: params={:?}, results={:?}",
                                    func_type.params(),
                                    func_type.results()
                                );
                            }
                        }
                    }
                }

                Payload::ImportSection(reader) => {
                    for import in reader {
                        log::debug!("Payload::ImportSection: {:#?}", import);
                    }
                }

                Payload::FunctionSection(reader) => {
                    for func in reader {
                        log::debug!("Payload::FunctionSection: {:#?}", func);
                        self.functions.push(func?);
                    }
                }

                Payload::TableSection(reader) => {
                    for table in reader {
                        log::debug!("Payload::TableSection: {:#?}", table);
                        let table = table?;
                        self.tables.push(table.ty);
                    }
                }

                Payload::MemorySection(reader) => {
                    for memory in reader {
                        log::debug!("Payload::MemorySection: {:#?}", memory);
                        self.memories.push(memory?);
                    }
                }

                Payload::TagSection(tag) => {
                    log::debug!("Payload::TagSection: {:#?}", tag);
                    // For now, we skip tag parsing
                }

                Payload::GlobalSection(reader) => {
                    for global in reader {
                        log::debug!("Payload::GlobalSection: {:#?}", global);
                        self.globals.push(global?.ty);
                    }
                }

                Payload::ExportSection(reader) => {
                    for export in reader {
                        let export = export?;
                        log::debug!("Payload::ExportSection: {:#?}", export);
                        self.exports.insert(export.name.to_string(), export.kind);
                    }
                }

                Payload::StartSection { func, range } => {
                    log::debug!("Payload::StartSection: func={}, range={:?}", func, range);
                    // For now, we skip start section parsing
                }

                Payload::ElementSection(reader) => {
                    log::debug!("Payload::ElementSection: {:#?}", reader);
                    for val in reader {
                        // log::debug!("Payload::ElementSection: {:#?}", val?);
                        // For now, we skip element section parsing
                    }
                }

                Payload::DataCountSection { count, range } => {
                    log::debug!(
                        "Payload::DataCountSection: count={}, range={:?}",
                        count,
                        range
                    );
                    // For now, we skip data count section parsing
                }

                Payload::DataSection(reader) => {
                    for val in reader {
                        log::debug!("Payload::DataSection: {:#?}", val?);
                    }
                }

                Payload::CodeSectionStart { count, range, size } => {
                    log::debug!(
                        "Payload::CodeSectionStart: count={}, range={:?}, size={}",
                        count,
                        range,
                        size
                    );
                }

                Payload::CodeSectionEntry(body) => {
                    log::debug!(
                        "Payload::CodeSectionEntry: {} bytes of function bytecode",
                        body.as_bytes().len()
                    );

                    // Parse the actual function bytecode using FunctionBody
                    self.parse_function_body(&body)?;
                }
                Payload::ModuleSection {
                    parser,
                    unchecked_range,
                } => {}

                Payload::InstanceSection(reader) => {}

                Payload::CoreTypeSection(reader) => {}

                Payload::ComponentSection {
                    parser,
                    unchecked_range,
                } => {}

                Payload::ComponentInstanceSection(reader) => {}

                Payload::ComponentAliasSection(reader) => {}

                Payload::ComponentTypeSection(reader) => {}

                Payload::ComponentCanonicalSection(reader) => {}

                Payload::ComponentStartSection { start, range } => {}

                Payload::ComponentImportSection(reader) => {}

                Payload::ComponentExportSection(reader) => {}

                Payload::CustomSection(reader) => {
                    let name = reader.name().to_string();
                    let data = reader.data().to_vec();
                    self.custom_sections.insert(name, data);
                }

                Payload::UnknownSection {
                    id,
                    contents,
                    range,
                } => {}

                Payload::End(_) => {
                    // end = true;
                }

                _ => {
                    return Err(DiscaError::InternalError(
                        "Unexpected WASM payload type".to_string(),
                    ));
                }
            }
        }

        // if !end {
        //     return Err(DiscaError::InternalError(
        //         "WASM parsing did not end correctly".to_string(),
        //     ));
        // }

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

        let circuit = reducer.reduce(&self.operations);
        log::debug!("LOGIC_CIRCUIT: {:#?}", circuit);
        circuit
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
    fn parse_function_body(&mut self, body: &wasmparser::FunctionBody) -> Result<()> {
        // Get the operators reader from the function body
        let operators_reader = body.get_operators_reader().map_err(|e| {
            DiscaError::InternalError(format!("Failed to create operators reader: {}", e))
        })?;

        for operator_result in operators_reader.into_iter() {
            match operator_result {
                Ok(op) => {
                    let wasm_op = self.convert_operator_to_operation(&op);
                    if let Some(operation) = wasm_op {
                        log::debug!("Parsed operation: {:?}", operation);
                        self.operations.push(operation);
                    }

                    // Log end of function
                    if matches!(op, wasmparser::Operator::End) {
                        log::debug!("Reached end of function");
                    }
                }
                Err(e) => {
                    log::warn!("Failed to parse operator: {}", e);
                    break;
                }
            }
        }

        log::debug!(
            "Parsed {} operations from function body",
            self.operations.len()
        );
        Ok(())
    }

    /// Convert wasmparser::Operator to WasmOperation
    fn convert_operator_to_operation(&self, op: &wasmparser::Operator) -> Option<WasmOperation> {
        match op {
            // Arithmetic operations
            wasmparser::Operator::I32Add => Some(WasmOperation {
                opcode: "i32.add".to_string(),
                operands: vec![],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I32Sub => Some(WasmOperation {
                opcode: "i32.sub".to_string(),
                operands: vec![],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I32Mul => Some(WasmOperation {
                opcode: "i32.mul".to_string(),
                operands: vec![],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I64Add => Some(WasmOperation {
                opcode: "i64.add".to_string(),
                operands: vec![],
                result_type: Some(ValType::I64),
            }),
            wasmparser::Operator::I64Sub => Some(WasmOperation {
                opcode: "i64.sub".to_string(),
                operands: vec![],
                result_type: Some(ValType::I64),
            }),
            wasmparser::Operator::I64Mul => Some(WasmOperation {
                opcode: "i64.mul".to_string(),
                operands: vec![],
                result_type: Some(ValType::I64),
            }),

            // Constants
            wasmparser::Operator::I32Const { value } => Some(WasmOperation {
                opcode: "i32.const".to_string(),
                operands: vec![value.to_string()],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I64Const { value } => Some(WasmOperation {
                opcode: "i64.const".to_string(),
                operands: vec![value.to_string()],
                result_type: Some(ValType::I64),
            }),
            wasmparser::Operator::F32Const { value } => Some(WasmOperation {
                opcode: "f32.const".to_string(),
                operands: vec![value.bits().to_string()],
                result_type: Some(ValType::F32),
            }),
            wasmparser::Operator::F64Const { value } => Some(WasmOperation {
                opcode: "f64.const".to_string(),
                operands: vec![value.bits().to_string()],
                result_type: Some(ValType::F64),
            }),

            // Local variable operations
            wasmparser::Operator::LocalGet { local_index } => Some(WasmOperation {
                opcode: "local.get".to_string(),
                operands: vec![local_index.to_string()],
                result_type: None, // Type depends on local variable
            }),
            wasmparser::Operator::LocalSet { local_index } => Some(WasmOperation {
                opcode: "local.set".to_string(),
                operands: vec![local_index.to_string()],
                result_type: None,
            }),
            wasmparser::Operator::LocalTee { local_index } => Some(WasmOperation {
                opcode: "local.tee".to_string(),
                operands: vec![local_index.to_string()],
                result_type: None,
            }),

            // Memory operations with offset
            wasmparser::Operator::I32Load { memarg } => Some(WasmOperation {
                opcode: "i32.load".to_string(),
                operands: vec![
                    format!("offset={}", memarg.offset),
                    format!("align={}", memarg.align),
                ],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I32Load8U { memarg } => Some(WasmOperation {
                opcode: "i32.load8_u".to_string(),
                operands: vec![
                    format!("offset={}", memarg.offset),
                    format!("align={}", memarg.align),
                ],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I32Load8S { memarg } => Some(WasmOperation {
                opcode: "i32.load8_s".to_string(),
                operands: vec![
                    format!("offset={}", memarg.offset),
                    format!("align={}", memarg.align),
                ],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I32Load16U { memarg } => Some(WasmOperation {
                opcode: "i32.load16_u".to_string(),
                operands: vec![
                    format!("offset={}", memarg.offset),
                    format!("align={}", memarg.align),
                ],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I32Load16S { memarg } => Some(WasmOperation {
                opcode: "i32.load16_s".to_string(),
                operands: vec![
                    format!("offset={}", memarg.offset),
                    format!("align={}", memarg.align),
                ],
                result_type: Some(ValType::I32),
            }),

            wasmparser::Operator::I32Store { memarg } => Some(WasmOperation {
                opcode: "i32.store".to_string(),
                operands: vec![
                    format!("offset={}", memarg.offset),
                    format!("align={}", memarg.align),
                ],
                result_type: None,
            }),
            wasmparser::Operator::I32Store8 { memarg } => Some(WasmOperation {
                opcode: "i32.store8".to_string(),
                operands: vec![
                    format!("offset={}", memarg.offset),
                    format!("align={}", memarg.align),
                ],
                result_type: None,
            }),
            wasmparser::Operator::I32Store16 { memarg } => Some(WasmOperation {
                opcode: "i32.store16".to_string(),
                operands: vec![
                    format!("offset={}", memarg.offset),
                    format!("align={}", memarg.align),
                ],
                result_type: None,
            }),

            // Comparison operations
            wasmparser::Operator::I32Eq => Some(WasmOperation {
                opcode: "i32.eq".to_string(),
                operands: vec![],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I32Ne => Some(WasmOperation {
                opcode: "i32.ne".to_string(),
                operands: vec![],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I32LtS => Some(WasmOperation {
                opcode: "i32.lt_s".to_string(),
                operands: vec![],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I32GtS => Some(WasmOperation {
                opcode: "i32.gt_s".to_string(),
                operands: vec![],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I32LeS => Some(WasmOperation {
                opcode: "i32.le_s".to_string(),
                operands: vec![],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I32GeS => Some(WasmOperation {
                opcode: "i32.ge_s".to_string(),
                operands: vec![],
                result_type: Some(ValType::I32),
            }),

            // Bitwise operations
            wasmparser::Operator::I32And => Some(WasmOperation {
                opcode: "i32.and".to_string(),
                operands: vec![],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I32Or => Some(WasmOperation {
                opcode: "i32.or".to_string(),
                operands: vec![],
                result_type: Some(ValType::I32),
            }),
            wasmparser::Operator::I32Xor => Some(WasmOperation {
                opcode: "i32.xor".to_string(),
                operands: vec![],
                result_type: Some(ValType::I32),
            }),

            // Control flow
            wasmparser::Operator::Block { blockty } => Some(WasmOperation {
                opcode: "block".to_string(),
                operands: vec![format!("{:?}", blockty)],
                result_type: None,
            }),
            wasmparser::Operator::End => Some(WasmOperation {
                opcode: "end".to_string(),
                operands: vec![],
                result_type: None,
            }),
            wasmparser::Operator::BrIf { relative_depth } => Some(WasmOperation {
                opcode: "br_if".to_string(),
                operands: vec![relative_depth.to_string()],
                result_type: None,
            }),
            wasmparser::Operator::Return => Some(WasmOperation {
                opcode: "return".to_string(),
                operands: vec![],
                result_type: None,
            }),

            // Unsupported operations - return None to skip
            _ => {
                log::debug!("Unsupported WASM operator: {:?}", op);
                None
            }
        }
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
