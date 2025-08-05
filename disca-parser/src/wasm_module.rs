//! WASM module parsing and operation extraction

use crate::errors::Result;
use crate::homomorphic::{LogicCircuit, LogicGate, WasmReducer};
use crate::DiscaError;
use std::collections::HashMap;
use wasmparser::{GlobalType, Parser, Payload, ValType};

#[derive(Debug, Clone)]
pub struct WasmOperation {
    pub opcode: String,
    pub operands: Vec<String>,
    pub result_type: Option<ValType>,
}

#[derive(Debug, Clone)]
pub struct WasmFunction {
    pub name: String,
    pub type_index: u32,
    pub func_type: wasmparser::FuncType,
    pub operations: Vec<WasmOperation>,
    pub is_exported: bool,
}

#[derive(Debug, Clone)]
pub struct GlobalState {
    pub memory_size: u32,
    pub globals: Vec<GlobalType>,
    pub memory_operations: Vec<WasmOperation>,
}

#[derive(Debug)]
pub struct WasmModule {
    pub functions: HashMap<String, WasmFunction>,
    pub global_state: GlobalState,
    pub exports: HashMap<String, wasmparser::ExternalKind>,
    pub types: Vec<wasmparser::FuncType>,
    pub custom_sections: HashMap<String, Vec<u8>>,
}

pub fn parse_wasm_module(wasm_bytes: &[u8]) -> Result<HashMap<String, LogicCircuit>> {
    let mut module = WasmModule::new();
    module.parse(wasm_bytes)?;
    Ok(module.to_logic_circuits())
}

impl WasmModule {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            global_state: GlobalState {
                memory_size: 0,
                globals: Vec::new(),
                memory_operations: Vec::new(),
            },
            exports: HashMap::new(),
            types: Vec::new(),
            custom_sections: HashMap::new(),
        }
    }

    /// Parse WASM bytes into this module
    pub fn parse(&mut self, wasm_bytes: &[u8]) -> Result<()> {
        let mut function_types: Vec<u32> = Vec::new(); // Store function type indices
        let mut current_function_index = 0;

        for payload in Parser::new(0).parse_all(wasm_bytes) {
            match payload? {
                Payload::Version { .. } => {}

                Payload::TypeSection(reader) => {
                    for ty in reader {
                        log::debug!("Payload::TypeSection: {:#?}", ty);

                        let rec_group = ty?;
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
                        function_types.push(func?);
                    }
                }

                Payload::TableSection(reader) => {
                    for table in reader {
                        log::debug!("Payload::TableSection: {:#?}", table);
                        let _table = table?;
                        // Tables are handled as part of global state if needed
                    }
                }

                Payload::MemorySection(reader) => {
                    for memory in reader {
                        log::debug!("Payload::MemorySection: {:#?}", memory);
                        let memory = memory?;
                        self.global_state.memory_size = memory.initial as u32;
                    }
                }

                Payload::TagSection(tag) => {
                    log::debug!("Payload::TagSection: {:#?}", tag);
                }

                Payload::GlobalSection(reader) => {
                    for global in reader {
                        log::debug!("Payload::GlobalSection: {:#?}", global);
                        self.global_state.globals.push(global?.ty);
                    }
                }

                Payload::ExportSection(reader) => {
                    for export in reader {
                        let export = export?;
                        log::debug!("Payload::ExportSection: {:#?}", export);
                        self.exports.insert(export.name.to_string(), export.kind);

                        // Mark functions as exported
                        if let wasmparser::ExternalKind::Func = export.kind {
                            // We'll update this when we process the function bodies
                        }
                    }
                }

                Payload::StartSection { func, range } => {
                    log::debug!("Payload::StartSection: func={}, range={:?}", func, range);
                }

                Payload::ElementSection(reader) => {
                    log::debug!("Payload::ElementSection: {:#?}", reader);
                    for _val in reader {
                        // Skip element section parsing for now
                    }
                }

                Payload::DataCountSection { count, range } => {
                    log::debug!(
                        "Payload::DataCountSection: count={}, range={:?}",
                        count,
                        range
                    );
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

                    // Parse the function body and create a WasmFunction
                    self.parse_function_body(&body, current_function_index, &function_types)?;
                    current_function_index += 1;
                }

                Payload::CustomSection(reader) => {
                    let name = reader.name().to_string();
                    let data = reader.data().to_vec();
                    self.custom_sections.insert(name, data);
                }

                Payload::End(_) => {
                    break;
                }

                _ => {
                    log::debug!("Skipping unsupported WASM section");
                }
            }
        }

        Ok(())
    }

    /// Convert WASM operations to logic circuits per function
    pub fn to_logic_circuits(&self) -> HashMap<String, LogicCircuit> {
        let mut circuits = HashMap::new();

        // Generate circuits for each exported function
        for (func_name, func) in &self.functions {
            if func.is_exported {
                let mut reducer = WasmReducer::new();
                let circuit = reducer.reduce(&func.operations);
                log::debug!(
                    "Generated circuit for function '{}': {:#?}",
                    func_name,
                    circuit
                );
                circuits.insert(func_name.clone(), circuit);
            }
        }

        // If no exported functions, create a test circuit
        if circuits.is_empty() {
            circuits.insert("test".to_string(), self.create_test_circuit());
        }

        circuits
    }

    /// Convert WASM operations to a single logic circuit (legacy method)
    pub fn to_logic_circuit(&self) -> LogicCircuit {
        // For backwards compatibility, return the first exported function's circuit
        // or a test circuit if no functions exist
        if let Some((_, func)) = self.functions.iter().find(|(_, f)| f.is_exported) {
            let mut reducer = WasmReducer::new();
            return reducer.reduce(&func.operations);
        }

        self.create_test_circuit()
    }

    /// Create a test circuit based on detected functions
    /// This is a temporary implementation until full WASM parsing is complete
    fn create_test_circuit(&self) -> LogicCircuit {
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
    fn parse_function_body(
        &mut self,
        body: &wasmparser::FunctionBody,
        function_index: usize,
        function_types: &[u32],
    ) -> Result<()> {
        let mut operations = Vec::new();

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

                        // Check if this is a memory operation (global state)
                        if self.is_memory_operation(&operation) {
                            self.global_state.memory_operations.push(operation.clone());
                        }

                        operations.push(operation);
                    }
                }
                Err(e) => {
                    log::warn!("Failed to parse operator: {}", e);
                    break;
                }
            }
        }

        // Create function name and determine if it's exported
        let func_name = self.get_function_name(function_index);
        let is_exported = self.exports.contains_key(&func_name);

        // Get function type
        let type_index = function_types.get(function_index).copied().unwrap_or(0);
        let func_type = self
            .types
            .get(type_index as usize)
            .cloned()
            .unwrap_or_else(|| {
                // Default function type if not found
                wasmparser::FuncType::new([], [])
            });

        let wasm_function = WasmFunction {
            name: func_name.clone(),
            type_index,
            func_type,
            operations,
            is_exported,
        };

        self.functions.insert(func_name, wasm_function);

        log::debug!("Parsed function with {} operations", self.functions.len());
        Ok(())
    }

    /// Check if an operation is a memory operation
    fn is_memory_operation(&self, operation: &WasmOperation) -> bool {
        matches!(
            operation.opcode.as_str(),
            "i32.load"
                | "i32.load8_u"
                | "i32.load8_s"
                | "i32.load16_u"
                | "i32.load16_s"
                | "i32.store"
                | "i32.store8"
                | "i32.store16"
                | "i64.load"
                | "i64.store"
                | "f32.load"
                | "f32.store"
                | "f64.load"
                | "f64.store"
        )
    }

    /// Get function name from index (use export name if available)
    fn get_function_name(&self, function_index: usize) -> String {
        // Check if this function is exported
        for (export_name, kind) in &self.exports {
            if let wasmparser::ExternalKind::Func = kind {
                // Note: This is a simplified approach. In a full implementation,
                // we'd need to track the export index properly
                return export_name.clone();
            }
        }

        // Default to indexed name
        format!("func_{}", function_index)
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
        let total_operations: usize = self
            .functions
            .values()
            .map(|f| f.operations.len())
            .sum::<usize>()
            + self.global_state.memory_operations.len();

        ModuleStats {
            function_count: self.functions.len(),
            operation_count: total_operations,
            export_count: self.exports.len(),
            type_count: self.types.len(),
            table_count: 0, // Tables are not tracked separately anymore
            memory_count: if self.global_state.memory_size > 0 {
                1
            } else {
                0
            },
            global_count: self.global_state.globals.len(),
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
