use disca_parser::prelude::*;
use disca_parser::{OptimizationLevel, OutputFormat};

const WASM_PATH: &str = "wasm_program/wasm_program.wasm";

#[test]
fn test_wasm_parsing() {
    let wasm_bytes = load_test_wasm();
    let parser = WasmParser::new();

    // Test logic circuit parsing
    let logic_result = parser.parse_to_logic_circuit(&wasm_bytes);
    assert!(logic_result.is_ok(), "Should parse WASM to logic circuit");
    let logic_circuit = logic_result.unwrap();
    assert!(
        !logic_circuit.gates.is_empty(),
        "Logic circuit should have gates"
    );

    // Test binary circuit parsing
    let binary_result = parser.parse_to_binary_circuit(&wasm_bytes);
    assert!(binary_result.is_ok(), "Should parse WASM to binary circuit");
    let binary_circuit = binary_result.unwrap();
    assert!(
        !binary_circuit.gates.is_empty(),
        "Binary circuit should have gates"
    );
    assert!(
        binary_circuit.wire_count > 0,
        "Binary circuit should have wires"
    );

    println!(
        "Parsed {} logic gates, {} binary gates",
        logic_circuit.gates.len(),
        binary_circuit.gates.len()
    );
}

#[test]
fn test_circuit_properties() {
    let wasm_bytes = load_test_wasm();
    let parser = WasmParser::new();
    let circuit = parser.parse_to_binary_circuit(&wasm_bytes).unwrap();

    // Basic structure checks
    assert!(circuit.wire_count > 0, "Should have wires");
    assert!(!circuit.gates.is_empty(), "Should have gates");
    assert!(!circuit.input_wires.is_empty(), "Should have input wires");
    assert!(!circuit.output_wires.is_empty(), "Should have output wires");

    // Gate type analysis
    let add_gates = circuit.gates.iter().filter(|g| g.opcode == 0x00).count();
    let mul_gates = circuit.gates.iter().filter(|g| g.opcode == 0x01).count();

    println!(
        "Circuit has {} add gates, {} multiply gates",
        add_gates, mul_gates
    );
    assert!(
        add_gates > 0 || mul_gates > 0,
        "Should have arithmetic operations"
    );
}

#[test]
fn test_gate_execution() {
    let mut wire_values = vec![5, 3, 0, 0];

    // Test addition gate
    let add_gate = BinaryLogicGate {
        opcode: 0x00,
        input_a: 0,
        input_b: 1,
        output: 2,
        immediate: None,
    };
    assert!(add_gate.execute(&mut wire_values).is_ok());
    assert_eq!(wire_values[2], 8, "5 + 3 = 8");

    // Test multiplication gate
    let mul_gate = BinaryLogicGate {
        opcode: 0x01,
        input_a: 2,
        input_b: 1,
        output: 3,
        immediate: None,
    };
    assert!(mul_gate.execute(&mut wire_values).is_ok());
    assert_eq!(wire_values[3], 24, "8 * 3 = 24");

    println!("Gate execution test passed: {:?}", wire_values);
}

#[test]
fn test_serialization() {
    let wasm_bytes = load_test_wasm();
    let parser = WasmParser::new();
    let circuit = parser.parse_to_binary_circuit(&wasm_bytes).unwrap();

    // Test JSON serialization
    let json = circuit.to_json().unwrap();
    assert!(!json.is_empty(), "JSON should not be empty");
    assert!(
        json.contains("wire_count"),
        "JSON should contain wire_count"
    );

    // Test binary serialization
    let binary_bytes = circuit.to_storage_bytes();
    assert!(
        !binary_bytes.is_empty(),
        "Binary serialization should produce data"
    );

    println!(
        "Serialization: {} chars JSON, {} bytes binary",
        json.len(),
        binary_bytes.len()
    );
}

#[test]
fn test_optimization() {
    let wasm_bytes = load_test_wasm();

    // Test without optimization
    let parser_none = WasmParser::new();
    let circuit_none = parser_none.parse_to_binary_circuit(&wasm_bytes).unwrap();

    // Test with basic optimization
    let parser_basic = WasmParser::with_optimization(OptimizationLevel::Basic);
    let circuit_basic = parser_basic.parse_to_binary_circuit(&wasm_bytes).unwrap();

    assert!(
        !circuit_none.gates.is_empty(),
        "Non-optimized circuit should have gates"
    );
    assert!(
        !circuit_basic.gates.is_empty(),
        "Optimized circuit should have gates"
    );

    println!(
        "Gates: {} (none) vs {} (basic)",
        circuit_none.gates.len(),
        circuit_basic.gates.len()
    );
}

#[test]
fn test_logic_tracking() {
    let wasm_bytes = load_test_wasm();
    let parser = WasmParser::new();
    let circuit = parser.parse_to_binary_circuit(&wasm_bytes).unwrap();

    // Check for expected WASM function operations
    let add_count = circuit.gates.iter().filter(|g| g.opcode == 0x00).count();
    let mul_count = circuit.gates.iter().filter(|g| g.opcode == 0x01).count();
    let sub_count = circuit.gates.iter().filter(|g| g.opcode == 0x02).count();

    println!(
        "Operations - Add: {}, Mul: {}, Sub: {}",
        add_count, mul_count, sub_count
    );

    // Should have arithmetic operations for the WASM functions
    assert!(add_count > 0, "Should have addition operations");
    assert!(mul_count > 0, "Should have multiplication operations");
    assert!(
        circuit.multiplicative_depth >= 1,
        "Should have multiplicative depth"
    );
}

#[test]
fn test_output_formats() {
    let wasm_bytes = load_test_wasm();
    let parser = WasmParser::new();
    let logic_circuit = parser.parse_to_logic_circuit(&wasm_bytes).unwrap();

    // Test all output formats
    for format in [OutputFormat::Text, OutputFormat::Json, OutputFormat::Binary] {
        let result = disca_parser::utils::convert_format(&logic_circuit, format);
        assert!(result.is_ok(), "Should convert to format {:?}", format);
        let output = result.unwrap();
        assert!(!output.is_empty(), "Output should not be empty");
        println!("Format {:?}: {} bytes", format, output.len());
    }
}

#[test]
fn test_error_handling() {
    let parser = WasmParser::new();

    // Test invalid WASM data
    let result = parser.parse_to_binary_circuit(b"invalid");
    assert!(result.is_err(), "Should fail with invalid WASM");

    // Test empty data
    let result = parser.parse_to_binary_circuit(b"");
    assert!(result.is_err(), "Should fail with empty data");

    println!("Error handling tests passed");
}

#[test]
fn test_circuit_validation() {
    let wasm_bytes = load_test_wasm();
    let parser = WasmParser::new();
    let logic_circuit = parser.parse_to_logic_circuit(&wasm_bytes).unwrap();

    // Analyze and validate the circuit
    let stats = disca_parser::utils::analyze_circuit(&logic_circuit);
    println!("Circuit analysis:\n{}", stats);

    let validation_result = disca_parser::utils::validate_circuit(&logic_circuit);
    assert!(validation_result.is_ok(), "Circuit should be valid");
}

#[test]
fn test_wire_consistency() {
    let wasm_bytes = load_test_wasm();
    let parser = WasmParser::new();
    let circuit = parser.parse_to_binary_circuit(&wasm_bytes).unwrap();

    // Basic sanity checks for the circuit structure
    assert!(circuit.wire_count > 0, "Should have some wires");
    assert!(!circuit.gates.is_empty(), "Should have some gates");
    assert!(!circuit.input_wires.is_empty(), "Should have input wires");
    assert!(!circuit.output_wires.is_empty(), "Should have output wires");

    // Count unique wires used in gates
    let mut all_wires = std::collections::HashSet::new();
    for gate in &circuit.gates {
        all_wires.insert(gate.input_a);
        all_wires.insert(gate.input_b);
        all_wires.insert(gate.output);
    }
    circuit.input_wires.iter().for_each(|&w| {
        all_wires.insert(w);
    });
    circuit.output_wires.iter().for_each(|&w| {
        all_wires.insert(w);
    });

    // The circuit should have a reasonable wire count
    assert!(
        circuit.wire_count == all_wires.len() as u32,
        "Wire count should match unique wires used"
    );

    println!(
        "Wire consistency: count={}, unique={}, gates={}",
        circuit.wire_count,
        all_wires.len(),
        circuit.gates.len()
    );
}

// Helper function to load test WASM
fn load_test_wasm() -> Vec<u8> {
    std::fs::read(WASM_PATH).unwrap_or_else(|_| panic!("Failed to read WASM file at {}", WASM_PATH))
}
