use crate::WasmOperation;
use serde::{Deserialize, Serialize};

/// Wire identifier for circuit connections
pub type WireId = u32;

/// Compact binary-encoded logic gate for blockchain runtime
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryLogicGate {
    /// Operation type (4 bits: 0-15 operations supported)
    pub opcode: u8,
    /// Input wire A (28 bits: supports up to 268M wires)
    pub input_a: WireId,
    /// Input wire B (28 bits: supports up to 268M wires)
    pub input_b: WireId,
    /// Output wire (28 bits: supports up to 268M wires)
    pub output: WireId,
    /// Immediate value for constants (64 bits)
    pub immediate: Option<i64>,
}

/// Circuit opcodes for binary encoding
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateOpcode {
    Add = 0x00,      // 0000
    Multiply = 0x01, // 0001
    Subtract = 0x02, // 0002
    Constant = 0x03, // 0003
    Load = 0x04,     // 0004
    Store = 0x05,    // 0005
    Equal = 0x06,    // 0006
    LessThan = 0x07, // 0007
    And = 0x08,      // 0008
    Or = 0x09,       // 0009
    Xor = 0x0A,      // 000A
    Not = 0x0B,      // 000B
                     // Reserved for future operations: 0x0C - 0x0F
}

/// Legacy string-based logic gate representation
#[derive(Debug, Clone, PartialEq)]
pub enum LogicGate {
    /// Addition gate - compatible with homomorphic addition
    Add {
        input_a: String,
        input_b: String,
        output: String,
    },
    /// Multiplication gate - compatible with homomorphic multiplication  
    Multiply {
        input_a: String,
        input_b: String,
        output: String,
    },
    /// Subtraction gate
    Subtract {
        input_a: String,
        input_b: String,
        output: String,
    },
    /// Constant input gate
    Constant {
        value: i64,
        output: String,
    },
    /// Load from memory/variable
    Load {
        address: String,
        output: String,
    },
    /// Store to memory/variable
    Store {
        input: String,
        address: String,
    },
    /// Comparison gates
    Equal {
        input_a: String,
        input_b: String,
        output: String,
    },
    LessThan {
        input_a: String,
        input_b: String,
        output: String,
    },
    /// Bitwise operations
    And {
        input_a: String,
        input_b: String,
        output: String,
    },
    Or {
        input_a: String,
        input_b: String,
        output: String,
    },
    Xor {
        input_a: String,
        input_b: String,
        output: String,
    },
    Not {
        input: String,
        output: String,
    },
}

/// Represents a circuit of logic gates
#[derive(Debug, Clone)]
pub struct LogicCircuit {
    pub gates: Vec<LogicGate>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub wire_count: u32,
}

impl LogicCircuit {
    pub fn new() -> Self {
        Self {
            gates: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            wire_count: 0,
        }
    }

    /// Generate a new unique wire name
    pub fn new_wire(&mut self) -> String {
        let wire_name = format!("wire_{}", self.wire_count);
        self.wire_count += 1;
        wire_name
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: LogicGate) {
        self.gates.push(gate);
    }

    /// Add an input wire
    pub fn add_input(&mut self, name: String) {
        self.inputs.push(name);
    }

    /// Add an output wire
    pub fn add_output(&mut self, name: String) {
        self.outputs.push(name);
    }

    /// Generate a textual representation of the circuit
    pub fn to_string(&self) -> String {
        let mut result = String::new();

        result.push_str("Logic Circuit:\n");
        result.push_str(&format!("Inputs: {:?}\n", self.inputs));
        result.push_str(&format!("Outputs: {:?}\n", self.outputs));
        result.push_str("Gates:\n");

        for (i, gate) in self.gates.iter().enumerate() {
            result.push_str(&format!("  {}: {}\n", i, gate.to_string()));
        }

        result
    }

    /// Generate Boolean circuit representation for FHE
    pub fn to_fhe_circuit(&self) -> String {
        let mut result = String::new();

        result.push_str("// FHE-compatible Boolean circuit\n");
        result.push_str("// Generated from WASM operations\n\n");

        // Input declarations
        for input in &self.inputs {
            result.push_str(&format!("input {}\n", input));
        }
        result.push_str("\n");

        // Gate implementations
        for gate in &self.gates {
            result.push_str(&format!("{}\n", gate.to_fhe_gate()));
        }
        result.push_str("\n");

        // Output declarations
        for output in &self.outputs {
            result.push_str(&format!("output {}\n", output));
        }

        result
    }
}

/// Binary encoding implementations for blockchain runtime
impl BinaryLogicGate {
    /// Create a new binary logic gate
    pub fn new(opcode: GateOpcode, input_a: WireId, input_b: WireId, output: WireId) -> Self {
        Self {
            opcode: opcode as u8,
            input_a,
            input_b,
            output,
            immediate: None,
        }
    }

    /// Create a constant gate with immediate value
    pub fn constant(value: i64, output: WireId) -> Self {
        Self {
            opcode: GateOpcode::Constant as u8,
            input_a: 0,
            input_b: 0,
            output,
            immediate: Some(value),
        }
    }

    /// Convert to compact byte representation for blockchain storage
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16); // Fixed 16-byte encoding

        // Byte 0: Opcode (4 bits) + Reserved (4 bits)
        bytes.push(self.opcode & 0x0F);

        // Bytes 1-4: Input A wire ID (32 bits)
        bytes.extend_from_slice(&self.input_a.to_le_bytes());

        // Bytes 5-8: Input B wire ID (32 bits)
        bytes.extend_from_slice(&self.input_b.to_le_bytes());

        // Bytes 9-12: Output wire ID (32 bits)
        bytes.extend_from_slice(&self.output.to_le_bytes());

        // Bytes 13-16: Immediate value (32 bits, truncated from i64)
        if let Some(immediate) = self.immediate {
            bytes.extend_from_slice(&(immediate as i32).to_le_bytes());
        } else {
            bytes.extend_from_slice(&[0, 0, 0, 0]);
        }

        bytes
    }

    /// Create from byte representation
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() != 16 {
            return Err("Invalid byte length for BinaryLogicGate");
        }

        let opcode = bytes[0] & 0x0F;
        let input_a = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
        let input_b = u32::from_le_bytes([bytes[5], bytes[6], bytes[7], bytes[8]]);
        let output = u32::from_le_bytes([bytes[9], bytes[10], bytes[11], bytes[12]]);
        let immediate_raw = i32::from_le_bytes([bytes[13], bytes[14], bytes[15], bytes[16]]);

        let immediate = if opcode == GateOpcode::Constant as u8 {
            Some(immediate_raw as i64)
        } else {
            None
        };

        Ok(Self {
            opcode,
            input_a,
            input_b,
            output,
            immediate,
        })
    }

    /// Get the operation type
    pub fn get_operation(&self) -> Result<GateOpcode, &'static str> {
        match self.opcode {
            0x00 => Ok(GateOpcode::Add),
            0x01 => Ok(GateOpcode::Multiply),
            0x02 => Ok(GateOpcode::Subtract),
            0x03 => Ok(GateOpcode::Constant),
            0x04 => Ok(GateOpcode::Load),
            0x05 => Ok(GateOpcode::Store),
            0x06 => Ok(GateOpcode::Equal),
            0x07 => Ok(GateOpcode::LessThan),
            0x08 => Ok(GateOpcode::And),
            0x09 => Ok(GateOpcode::Or),
            0x0A => Ok(GateOpcode::Xor),
            0x0B => Ok(GateOpcode::Not),
            _ => Err("Invalid opcode"),
        }
    }

    /// Execute the gate operation (for runtime evaluation)
    pub fn execute(&self, wire_values: &mut [i64]) -> Result<(), &'static str> {
        let op = self.get_operation()?;

        match op {
            GateOpcode::Add => {
                let a = wire_values[self.input_a as usize];
                let b = wire_values[self.input_b as usize];
                wire_values[self.output as usize] = a.wrapping_add(b);
            }
            GateOpcode::Multiply => {
                let a = wire_values[self.input_a as usize];
                let b = wire_values[self.input_b as usize];
                wire_values[self.output as usize] = a.wrapping_mul(b);
            }
            GateOpcode::Subtract => {
                let a = wire_values[self.input_a as usize];
                let b = wire_values[self.input_b as usize];
                wire_values[self.output as usize] = a.wrapping_sub(b);
            }
            GateOpcode::Constant => {
                if let Some(value) = self.immediate {
                    wire_values[self.output as usize] = value;
                } else {
                    return Err("Constant gate missing immediate value");
                }
            }
            GateOpcode::Load => {
                // Load from memory address (input_a) to output
                let address = self.input_a as usize;
                if address < wire_values.len() {
                    wire_values[self.output as usize] = wire_values[address];
                } else {
                    return Err("Load address out of bounds");
                }
            }
            GateOpcode::Store => {
                // Store input value to memory address (output)
                let value = wire_values[self.input_a as usize];
                let address = self.output as usize;
                if address < wire_values.len() {
                    wire_values[address] = value;
                } else {
                    return Err("Store address out of bounds");
                }
            }
            GateOpcode::Equal => {
                let a = wire_values[self.input_a as usize];
                let b = wire_values[self.input_b as usize];
                wire_values[self.output as usize] = if a == b { 1 } else { 0 };
            }
            GateOpcode::LessThan => {
                let a = wire_values[self.input_a as usize];
                let b = wire_values[self.input_b as usize];
                wire_values[self.output as usize] = if a < b { 1 } else { 0 };
            }
            GateOpcode::And => {
                let a = wire_values[self.input_a as usize];
                let b = wire_values[self.input_b as usize];
                wire_values[self.output as usize] = a & b;
            }
            GateOpcode::Or => {
                let a = wire_values[self.input_a as usize];
                let b = wire_values[self.input_b as usize];
                wire_values[self.output as usize] = a | b;
            }
            GateOpcode::Xor => {
                let a = wire_values[self.input_a as usize];
                let b = wire_values[self.input_b as usize];
                wire_values[self.output as usize] = a ^ b;
            }
            GateOpcode::Not => {
                let a = wire_values[self.input_a as usize];
                wire_values[self.output as usize] = !a;
            }
        }

        Ok(())
    }
}

/// Compact binary circuit representation for blockchain storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryCircuit {
    /// Circuit metadata
    pub wire_count: u32,
    pub input_wires: Vec<WireId>,
    pub output_wires: Vec<WireId>,
    pub multiplicative_depth: u32,

    /// Compact gate encoding (16 bytes per gate)
    pub gates: Vec<BinaryLogicGate>,
}

impl BinaryCircuit {
    /// Convert to bytes for blockchain storage
    pub fn to_storage_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Header: 16 bytes
        bytes.extend_from_slice(&self.wire_count.to_le_bytes()); // 4 bytes
        bytes.extend_from_slice(&(self.input_wires.len() as u32).to_le_bytes()); // 4 bytes
        bytes.extend_from_slice(&(self.output_wires.len() as u32).to_le_bytes()); // 4 bytes
        bytes.extend_from_slice(&self.multiplicative_depth.to_le_bytes()); // 4 bytes

        // Input wire IDs
        for wire_id in &self.input_wires {
            bytes.extend_from_slice(&wire_id.to_le_bytes());
        }

        // Output wire IDs
        for wire_id in &self.output_wires {
            bytes.extend_from_slice(&wire_id.to_le_bytes());
        }

        // Gates (16 bytes each)
        for gate in &self.gates {
            bytes.extend_from_slice(&gate.to_bytes());
        }

        bytes
    }

    /// Execute the entire circuit
    pub fn execute(&self, inputs: &[i64]) -> Result<Vec<i64>, &'static str> {
        if inputs.len() != self.input_wires.len() {
            return Err("Input count mismatch");
        }

        // Initialize wire values
        let mut wire_values = vec![0i64; self.wire_count as usize];

        // Set input values
        for (i, &wire_id) in self.input_wires.iter().enumerate() {
            wire_values[wire_id as usize] = inputs[i];
        }

        // Execute gates in order
        for gate in &self.gates {
            gate.execute(&mut wire_values)?;
        }

        // Extract outputs
        let mut outputs = Vec::with_capacity(self.output_wires.len());
        for &wire_id in &self.output_wires {
            outputs.push(wire_values[wire_id as usize]);
        }

        Ok(outputs)
    }

    /// Convert to JSON representation
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Create from JSON representation
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Convert from legacy LogicCircuit
    pub fn from_logic_circuit(circuit: &LogicCircuit) -> Result<Self, crate::errors::DiscaError> {
        let binary_gates: Vec<BinaryLogicGate> =
            circuit.gates.iter().map(|gate| gate.into()).collect();

        // Analyze circuit to determine wire count and input/output wires
        let wire_count = Self::calculate_wire_count(&binary_gates);
        let input_wires = Self::extract_input_wires(&binary_gates);
        let output_wires = Self::extract_output_wires(&binary_gates);
        let multiplicative_depth = Self::calculate_multiplicative_depth(&binary_gates);

        Ok(BinaryCircuit {
            wire_count,
            input_wires,
            output_wires,
            multiplicative_depth,
            gates: binary_gates,
        })
    }

    /// Calculate the number of unique wires in the circuit
    fn calculate_wire_count(gates: &[BinaryLogicGate]) -> u32 {
        use std::collections::HashSet;
        let mut wires = HashSet::new();

        for gate in gates {
            wires.insert(gate.input_a);
            wires.insert(gate.input_b);
            wires.insert(gate.output);
        }

        wires.len() as u32
    }

    /// Extract input wires (wires that are not outputs of any gate)
    fn extract_input_wires(gates: &[BinaryLogicGate]) -> Vec<WireId> {
        use std::collections::HashSet;
        let mut all_wires = HashSet::new();
        let mut output_wires = HashSet::new();

        for gate in gates {
            all_wires.insert(gate.input_a);
            all_wires.insert(gate.input_b);
            all_wires.insert(gate.output);
            output_wires.insert(gate.output);
        }

        all_wires.difference(&output_wires).cloned().collect()
    }

    /// Extract output wires (heuristic: wires that are outputs but not inputs to other gates)
    fn extract_output_wires(gates: &[BinaryLogicGate]) -> Vec<WireId> {
        use std::collections::HashSet;
        let mut output_wires = HashSet::new();
        let mut input_wires = HashSet::new();

        for gate in gates {
            output_wires.insert(gate.output);
            input_wires.insert(gate.input_a);
            input_wires.insert(gate.input_b);
        }

        // Output wires are those that are never used as inputs
        output_wires.difference(&input_wires).cloned().collect()
    }

    /// Calculate multiplicative depth of the circuit
    fn calculate_multiplicative_depth(gates: &[BinaryLogicGate]) -> u32 {
        // Simplified implementation - count multiply gates
        gates
            .iter()
            .filter(|gate| gate.opcode == GateOpcode::Multiply as u8)
            .count() as u32
    }
}

/// Convert legacy LogicGate to BinaryLogicGate
impl From<&LogicGate> for BinaryLogicGate {
    fn from(gate: &LogicGate) -> Self {
        match gate {
            LogicGate::Add {
                input_a,
                input_b,
                output,
            } => BinaryLogicGate::new(
                GateOpcode::Add,
                parse_wire_id(input_a),
                parse_wire_id(input_b),
                parse_wire_id(output),
            ),
            LogicGate::Multiply {
                input_a,
                input_b,
                output,
            } => BinaryLogicGate::new(
                GateOpcode::Multiply,
                parse_wire_id(input_a),
                parse_wire_id(input_b),
                parse_wire_id(output),
            ),
            LogicGate::Subtract {
                input_a,
                input_b,
                output,
            } => BinaryLogicGate::new(
                GateOpcode::Subtract,
                parse_wire_id(input_a),
                parse_wire_id(input_b),
                parse_wire_id(output),
            ),
            LogicGate::Constant { value, output } => {
                BinaryLogicGate::constant(*value, parse_wire_id(output))
            }
            LogicGate::Load { address, output } => BinaryLogicGate::new(
                GateOpcode::Load,
                parse_wire_id(address),
                0,
                parse_wire_id(output),
            ),
            LogicGate::Store { input, address } => BinaryLogicGate::new(
                GateOpcode::Store,
                parse_wire_id(input),
                0,
                parse_wire_id(address),
            ),
            LogicGate::Equal {
                input_a,
                input_b,
                output,
            } => BinaryLogicGate::new(
                GateOpcode::Equal,
                parse_wire_id(input_a),
                parse_wire_id(input_b),
                parse_wire_id(output),
            ),
            LogicGate::LessThan {
                input_a,
                input_b,
                output,
            } => BinaryLogicGate::new(
                GateOpcode::LessThan,
                parse_wire_id(input_a),
                parse_wire_id(input_b),
                parse_wire_id(output),
            ),
            LogicGate::And {
                input_a,
                input_b,
                output,
            } => BinaryLogicGate::new(
                GateOpcode::And,
                parse_wire_id(input_a),
                parse_wire_id(input_b),
                parse_wire_id(output),
            ),
            LogicGate::Or {
                input_a,
                input_b,
                output,
            } => BinaryLogicGate::new(
                GateOpcode::Or,
                parse_wire_id(input_a),
                parse_wire_id(input_b),
                parse_wire_id(output),
            ),
            LogicGate::Xor {
                input_a,
                input_b,
                output,
            } => BinaryLogicGate::new(
                GateOpcode::Xor,
                parse_wire_id(input_a),
                parse_wire_id(input_b),
                parse_wire_id(output),
            ),
            LogicGate::Not { input, output } => BinaryLogicGate::new(
                GateOpcode::Not,
                parse_wire_id(input),
                0,
                parse_wire_id(output),
            ),
        }
    }
}

/// Helper function to parse wire ID from string (e.g., "wire_5" -> 5)
fn parse_wire_id(wire_str: &str) -> WireId {
    if let Some(id_str) = wire_str.strip_prefix("wire_") {
        id_str.parse().unwrap_or(0)
    } else {
        // Fallback: hash the string to get a numeric ID
        wire_str
            .chars()
            .fold(0u32, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u32))
    }
}

impl LogicGate {
    /// Convert gate to string representation
    pub fn to_string(&self) -> String {
        match self {
            LogicGate::Add {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = ADD({}, {})", output, input_a, input_b)
            }
            LogicGate::Multiply {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = MUL({}, {})", output, input_a, input_b)
            }
            LogicGate::Subtract {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = SUB({}, {})", output, input_a, input_b)
            }
            LogicGate::Constant { value, output } => {
                format!("{} = CONST({})", output, value)
            }
            LogicGate::Load { address, output } => {
                format!("{} = LOAD({})", output, address)
            }
            LogicGate::Store { input, address } => {
                format!("STORE({}, {})", input, address)
            }
            LogicGate::Equal {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = EQ({}, {})", output, input_a, input_b)
            }
            LogicGate::LessThan {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = LT({}, {})", output, input_a, input_b)
            }
            LogicGate::And {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = AND({}, {})", output, input_a, input_b)
            }
            LogicGate::Or {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = OR({}, {})", output, input_a, input_b)
            }
            LogicGate::Xor {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = XOR({}, {})", output, input_a, input_b)
            }
            LogicGate::Not { input, output } => {
                format!("{} = NOT({})", output, input)
            }
        }
    }

    /// Convert gate to FHE-compatible representation
    pub fn to_fhe_gate(&self) -> String {
        match self {
            LogicGate::Add {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = fhe.add({}, {})", output, input_a, input_b)
            }
            LogicGate::Multiply {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = fhe.multiply({}, {})", output, input_a, input_b)
            }
            LogicGate::Subtract {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = fhe.subtract({}, {})", output, input_a, input_b)
            }
            LogicGate::Constant { value, output } => {
                format!("{} = fhe.encrypt({})", output, value)
            }
            LogicGate::Load { address, output } => {
                format!("{} = fhe.load({})", output, address)
            }
            LogicGate::Store { input, address } => {
                format!("fhe.store({}, {})", input, address)
            }
            LogicGate::Equal {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = fhe.equal({}, {})", output, input_a, input_b)
            }
            LogicGate::LessThan {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = fhe.less_than({}, {})", output, input_a, input_b)
            }
            LogicGate::And {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = fhe.and({}, {})", output, input_a, input_b)
            }
            LogicGate::Or {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = fhe.or({}, {})", output, input_a, input_b)
            }
            LogicGate::Xor {
                input_a,
                input_b,
                output,
            } => {
                format!("{} = fhe.xor({}, {})", output, input_a, input_b)
            }
            LogicGate::Not { input, output } => {
                format!("{} = fhe.not({})", output, input)
            }
        }
    }
}

/// WASM to Logic Gate Reducer
pub struct WasmReducer {
    stack: Vec<String>,
    locals: Vec<String>,
    wire_counter: u32,
}

impl WasmReducer {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            locals: Vec::new(),
            wire_counter: 0,
        }
    }

    /// Generate a new unique wire name
    fn new_wire(&mut self) -> String {
        let wire_name = format!("wire_{}", self.wire_counter);
        self.wire_counter += 1;
        wire_name
    }

    /// Reduce WASM operations to logic gates
    pub fn reduce(&mut self, operations: &[WasmOperation]) -> LogicCircuit {
        let mut circuit = LogicCircuit::new();

        for operation in operations {
            self.process_operation(operation, &mut circuit);
        }

        // Add final stack values as outputs
        for wire in &self.stack {
            circuit.add_output(wire.clone());
        }

        circuit
    }

    /// Process a single WASM operation and generate corresponding logic gates
    fn process_operation(&mut self, operation: &WasmOperation, circuit: &mut LogicCircuit) {
        match operation.opcode.as_str() {
            // Arithmetic operations
            "i32.add" | "i64.add" | "f32.add" | "f64.add" => {
                if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) {
                    let output = self.new_wire();
                    circuit.add_gate(LogicGate::Add {
                        input_a: a,
                        input_b: b,
                        output: output.clone(),
                    });
                    self.stack.push(output);
                }
            }

            "i32.sub" | "i64.sub" | "f32.sub" | "f64.sub" => {
                if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) {
                    let output = self.new_wire();
                    circuit.add_gate(LogicGate::Subtract {
                        input_a: a,
                        input_b: b,
                        output: output.clone(),
                    });
                    self.stack.push(output);
                }
            }

            "i32.mul" | "i64.mul" | "f32.mul" | "f64.mul" => {
                if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) {
                    let output = self.new_wire();
                    circuit.add_gate(LogicGate::Multiply {
                        input_a: a,
                        input_b: b,
                        output: output.clone(),
                    });
                    self.stack.push(output);
                }
            }

            // Constants
            "i32.const" | "i64.const" | "f32.const" | "f64.const" => {
                let value = operation
                    .operands
                    .first()
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(0);
                let output = self.new_wire();
                circuit.add_gate(LogicGate::Constant {
                    value,
                    output: output.clone(),
                });
                self.stack.push(output);
            }

            // Local variable operations
            "local.get" => {
                if let Some(index_str) = operation.operands.first() {
                    if let Ok(index) = index_str.parse::<usize>() {
                        // Ensure locals vector is large enough
                        while self.locals.len() <= index {
                            let new_wire = format!("wire_{}", self.wire_counter);
                            self.wire_counter += 1;
                            self.locals.push(new_wire);
                        }

                        let output = self.new_wire();
                        circuit.add_gate(LogicGate::Load {
                            address: format!("local_{}", index),
                            output: output.clone(),
                        });
                        self.stack.push(output);
                    }
                }
            }

            "local.set" => {
                if let Some(index_str) = operation.operands.first() {
                    if let Ok(index) = index_str.parse::<usize>() {
                        if let Some(value) = self.stack.pop() {
                            // Ensure locals vector is large enough
                            while self.locals.len() <= index {
                                let new_wire = format!("wire_{}", self.wire_counter);
                                self.wire_counter += 1;
                                self.locals.push(new_wire);
                            }

                            circuit.add_gate(LogicGate::Store {
                                input: value,
                                address: format!("local_{}", index),
                            });
                        }
                    }
                }
            }

            // Comparison operations
            "i32.eq" | "i64.eq" | "f32.eq" | "f64.eq" => {
                if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) {
                    let output = self.new_wire();
                    circuit.add_gate(LogicGate::Equal {
                        input_a: a,
                        input_b: b,
                        output: output.clone(),
                    });
                    self.stack.push(output);
                }
            }

            "i32.lt_s" | "i64.lt_s" | "f32.lt" | "f64.lt" => {
                if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) {
                    let output = self.new_wire();
                    circuit.add_gate(LogicGate::LessThan {
                        input_a: a,
                        input_b: b,
                        output: output.clone(),
                    });
                    self.stack.push(output);
                }
            }

            // Bitwise operations
            "i32.and" | "i64.and" => {
                if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) {
                    let output = self.new_wire();
                    circuit.add_gate(LogicGate::And {
                        input_a: a,
                        input_b: b,
                        output: output.clone(),
                    });
                    self.stack.push(output);
                }
            }

            "i32.or" | "i64.or" => {
                if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) {
                    let output = self.new_wire();
                    circuit.add_gate(LogicGate::Or {
                        input_a: a,
                        input_b: b,
                        output: output.clone(),
                    });
                    self.stack.push(output);
                }
            }

            "i32.xor" | "i64.xor" => {
                if let (Some(b), Some(a)) = (self.stack.pop(), self.stack.pop()) {
                    let output = self.new_wire();
                    circuit.add_gate(LogicGate::Xor {
                        input_a: a,
                        input_b: b,
                        output: output.clone(),
                    });
                    self.stack.push(output);
                }
            }

            _ => {
                // For unsupported operations, we'll create a placeholder
                // In a real implementation, you might want to handle more operations
            }
        }
    }
}
