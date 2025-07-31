use crate::errors::{DiscaError, Result};
use crate::homomorphic::{LogicCircuit, LogicGate};
use std::collections::{HashMap, HashSet};

/// Optimization level for circuit optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// No optimization
    #[default]
    None,
    /// Basic optimizations (dead code removal, constant folding)
    Basic,
    /// Aggressive optimizations (gate reordering, redundancy removal)
    Aggressive,
}

impl std::str::FromStr for OptimizationLevel {
    type Err = DiscaError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "none" => Ok(OptimizationLevel::None),
            "basic" => Ok(OptimizationLevel::Basic),
            "aggressive" => Ok(OptimizationLevel::Aggressive),
            _ => Err(DiscaError::invalid_input(format!(
                "Unknown optimization level: {}. Supported: none, basic, aggressive",
                s
            ))),
        }
    }
}

/// Circuit optimizer that can simplify and optimize logic gate circuits
pub struct CircuitOptimizer {
    optimization_level: OptimizationLevel,
}

impl CircuitOptimizer {
    /// Create a new optimizer with the specified optimization level
    pub fn new(optimization_level: OptimizationLevel) -> Self {
        Self { optimization_level }
    }

    /// Optimize the given circuit based on the optimization level
    pub fn optimize(&self, circuit: LogicCircuit) -> Result<LogicCircuit> {
        match self.optimization_level {
            OptimizationLevel::None => Ok(circuit),
            OptimizationLevel::Basic => {
                let mut optimized = Self::remove_dead_code(&circuit);
                optimized = Self::constant_propagation(&optimized);
                Ok(optimized)
            }
            OptimizationLevel::Aggressive => {
                let mut optimized = Self::remove_dead_code(&circuit);
                optimized = Self::constant_propagation(&optimized);
                optimized = Self::remove_redundant_gates(&optimized);
                optimized = Self::reorder_gates(&optimized);
                Ok(optimized)
            }
        }
    }

    /// Remove dead code (gates that don't contribute to outputs)
    pub fn remove_dead_code(circuit: &LogicCircuit) -> LogicCircuit {
        let mut optimized = LogicCircuit::new();
        let mut used_wires = HashSet::new();

        // Mark all output wires as used
        for output in &circuit.outputs {
            used_wires.insert(output.clone());
        }

        // Work backwards from outputs to find used gates
        let mut gates_to_keep = Vec::new();
        let mut changed = true;

        while changed {
            changed = false;
            for gate in circuit.gates.iter().rev() {
                let gate_output = Self::get_gate_output(gate);
                if let Some(output) = gate_output {
                    if used_wires.contains(&output) {
                        // Mark inputs as used
                        for input in Self::get_gate_inputs(gate) {
                            if used_wires.insert(input) {
                                changed = true;
                            }
                        }
                        gates_to_keep.push(gate.clone());
                    }
                }
            }
        }

        gates_to_keep.reverse();
        optimized.gates = gates_to_keep;
        optimized.inputs = circuit.inputs.clone();
        optimized.outputs = circuit.outputs.clone();
        optimized.wire_count = circuit.wire_count;

        optimized
    }

    /// Perform constant propagation
    pub fn constant_propagation(circuit: &LogicCircuit) -> LogicCircuit {
        let mut optimized = circuit.clone();
        let mut constants = HashMap::new();

        // Find all constant values
        for gate in &circuit.gates {
            if let LogicGate::Constant { value, output } = gate {
                constants.insert(output.clone(), *value);
            }
        }

        // Replace operations with constants where possible
        let mut new_gates = Vec::new();
        for gate in &circuit.gates {
            match gate {
                LogicGate::Add {
                    input_a,
                    input_b,
                    output,
                } => {
                    if let (Some(a), Some(b)) = (constants.get(input_a), constants.get(input_b)) {
                        let result = a + b;
                        constants.insert(output.clone(), result);
                        new_gates.push(LogicGate::Constant {
                            value: result,
                            output: output.clone(),
                        });
                    } else {
                        new_gates.push(gate.clone());
                    }
                }
                LogicGate::Multiply {
                    input_a,
                    input_b,
                    output,
                } => {
                    if let (Some(a), Some(b)) = (constants.get(input_a), constants.get(input_b)) {
                        let result = a * b;
                        constants.insert(output.clone(), result);
                        new_gates.push(LogicGate::Constant {
                            value: result,
                            output: output.clone(),
                        });
                    } else {
                        new_gates.push(gate.clone());
                    }
                }
                LogicGate::Subtract {
                    input_a,
                    input_b,
                    output,
                } => {
                    if let (Some(a), Some(b)) = (constants.get(input_a), constants.get(input_b)) {
                        let result = a - b;
                        constants.insert(output.clone(), result);
                        new_gates.push(LogicGate::Constant {
                            value: result,
                            output: output.clone(),
                        });
                    } else {
                        new_gates.push(gate.clone());
                    }
                }
                _ => new_gates.push(gate.clone()),
            }
        }

        optimized.gates = new_gates;
        optimized
    }

    /// Remove redundant gates (gates that produce identical outputs)
    pub fn remove_redundant_gates(circuit: &LogicCircuit) -> LogicCircuit {
        let mut optimized = circuit.clone();
        let mut unique_gates = Vec::new();
        let mut seen_computations = HashMap::new();

        for gate in &circuit.gates {
            let gate_key = format!("{:?}", gate);
            if let std::collections::hash_map::Entry::Vacant(e) = seen_computations.entry(gate_key)
            {
                e.insert(unique_gates.len());
                unique_gates.push(gate.clone());
            }
        }

        optimized.gates = unique_gates;
        optimized
    }

    /// Reorder gates for better execution (topological sort)
    pub fn reorder_gates(circuit: &LogicCircuit) -> LogicCircuit {
        let mut optimized = circuit.clone();

        // Simple reordering: put constants first, then computation gates
        let mut constants = Vec::new();
        let mut computations = Vec::new();

        for gate in &circuit.gates {
            match gate {
                LogicGate::Constant { .. } => constants.push(gate.clone()),
                _ => computations.push(gate.clone()),
            }
        }

        constants.extend(computations);
        optimized.gates = constants;
        optimized
    }

    /// Get the output wire of a gate
    fn get_gate_output(gate: &LogicGate) -> Option<String> {
        match gate {
            LogicGate::Add { output, .. }
            | LogicGate::Multiply { output, .. }
            | LogicGate::Subtract { output, .. }
            | LogicGate::Constant { output, .. }
            | LogicGate::Load { output, .. }
            | LogicGate::Equal { output, .. }
            | LogicGate::LessThan { output, .. }
            | LogicGate::And { output, .. }
            | LogicGate::Or { output, .. }
            | LogicGate::Xor { output, .. }
            | LogicGate::Not { output, .. } => Some(output.clone()),
            LogicGate::Store { .. } => None,
        }
    }

    /// Get the input wires of a gate
    fn get_gate_inputs(gate: &LogicGate) -> Vec<String> {
        match gate {
            LogicGate::Add {
                input_a, input_b, ..
            }
            | LogicGate::Multiply {
                input_a, input_b, ..
            }
            | LogicGate::Subtract {
                input_a, input_b, ..
            }
            | LogicGate::Equal {
                input_a, input_b, ..
            }
            | LogicGate::LessThan {
                input_a, input_b, ..
            }
            | LogicGate::And {
                input_a, input_b, ..
            }
            | LogicGate::Or {
                input_a, input_b, ..
            }
            | LogicGate::Xor {
                input_a, input_b, ..
            } => {
                vec![input_a.clone(), input_b.clone()]
            }
            LogicGate::Not { input, .. } => vec![input.clone()],
            LogicGate::Load { address, .. } => vec![address.clone()],
            LogicGate::Store { input, .. } => vec![input.clone()],
            LogicGate::Constant { .. } => vec![],
        }
    }
}

/// Circuit analyzer for complexity and performance metrics
pub struct CircuitAnalyzer;

impl CircuitAnalyzer {
    /// Analyze circuit complexity
    pub fn analyze_complexity(circuit: &LogicCircuit) -> CircuitComplexity {
        let mut add_count = 0;
        let mut mul_count = 0;
        let mut sub_count = 0;
        let mut compare_count = 0;
        let mut bitwise_count = 0;
        let mut load_store_count = 0;
        let mut constant_count = 0;

        for gate in &circuit.gates {
            match gate {
                LogicGate::Add { .. } => add_count += 1,
                LogicGate::Multiply { .. } => mul_count += 1,
                LogicGate::Subtract { .. } => sub_count += 1,
                LogicGate::Equal { .. } | LogicGate::LessThan { .. } => compare_count += 1,
                LogicGate::And { .. }
                | LogicGate::Or { .. }
                | LogicGate::Xor { .. }
                | LogicGate::Not { .. } => bitwise_count += 1,
                LogicGate::Load { .. } | LogicGate::Store { .. } => load_store_count += 1,
                LogicGate::Constant { .. } => constant_count += 1,
            }
        }

        let multiplicative_depth = Self::calculate_multiplicative_depth(circuit);
        let critical_path = Self::calculate_critical_path_length(circuit);

        CircuitComplexity {
            total_gates: circuit.gates.len(),
            add_count,
            mul_count,
            sub_count,
            compare_count,
            bitwise_count,
            load_store_count,
            constant_count,
            multiplicative_depth,
            critical_path_length: critical_path,
            wire_count: circuit.wire_count as usize,
            input_count: circuit.inputs.len(),
            output_count: circuit.outputs.len(),
        }
    }

    /// Calculate multiplicative depth (important for FHE performance)
    fn calculate_multiplicative_depth(circuit: &LogicCircuit) -> usize {
        let mut depths = HashMap::new();
        let mut max_depth = 0;

        // Initialize constants and inputs with depth 0
        for gate in &circuit.gates {
            if let LogicGate::Constant { output, .. } = gate {
                depths.insert(output.clone(), 0);
            }
        }

        for input in &circuit.inputs {
            depths.insert(input.clone(), 0);
        }

        // Calculate depths
        let mut changed = true;
        while changed {
            changed = false;
            for gate in &circuit.gates {
                match gate {
                    LogicGate::Multiply {
                        input_a,
                        input_b,
                        output,
                    } => {
                        if let (Some(&depth_a), Some(&depth_b)) =
                            (depths.get(input_a), depths.get(input_b))
                        {
                            let new_depth = depth_a.max(depth_b) + 1;
                            if depths.get(output).map_or(true, |&d| d < new_depth) {
                                depths.insert(output.clone(), new_depth);
                                max_depth = max_depth.max(new_depth);
                                changed = true;
                            }
                        }
                    }
                    LogicGate::Add {
                        input_a,
                        input_b,
                        output,
                    }
                    | LogicGate::Subtract {
                        input_a,
                        input_b,
                        output,
                    }
                    | LogicGate::Equal {
                        input_a,
                        input_b,
                        output,
                    }
                    | LogicGate::LessThan {
                        input_a,
                        input_b,
                        output,
                    }
                    | LogicGate::And {
                        input_a,
                        input_b,
                        output,
                    }
                    | LogicGate::Or {
                        input_a,
                        input_b,
                        output,
                    }
                    | LogicGate::Xor {
                        input_a,
                        input_b,
                        output,
                    } => {
                        if let (Some(&depth_a), Some(&depth_b)) =
                            (depths.get(input_a), depths.get(input_b))
                        {
                            let new_depth = depth_a.max(depth_b);
                            if depths.get(output).map_or(true, |&d| d < new_depth) {
                                depths.insert(output.clone(), new_depth);
                                changed = true;
                            }
                        }
                    }
                    LogicGate::Not { input, output } => {
                        if let Some(&depth) = depths.get(input) {
                            if depths.get(output).map_or(true, |&d| d < depth) {
                                depths.insert(output.clone(), depth);
                                changed = true;
                            }
                        }
                    }
                    LogicGate::Load { output, .. } => {
                        if !depths.contains_key(output) {
                            depths.insert(output.clone(), 0);
                            changed = true;
                        }
                    }
                    _ => {}
                }
            }
        }

        max_depth
    }

    /// Calculate critical path length
    fn calculate_critical_path_length(circuit: &LogicCircuit) -> usize {
        // Simplified critical path calculation
        circuit.gates.len()
    }
}

#[derive(Debug, Clone)]
pub struct CircuitComplexity {
    pub total_gates: usize,
    pub add_count: usize,
    pub mul_count: usize,
    pub sub_count: usize,
    pub compare_count: usize,
    pub bitwise_count: usize,
    pub load_store_count: usize,
    pub constant_count: usize,
    pub multiplicative_depth: usize,
    pub critical_path_length: usize,
    pub wire_count: usize,
    pub input_count: usize,
    pub output_count: usize,
}

impl CircuitComplexity {
    /// Generate a report of the circuit complexity
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Circuit Complexity Analysis ===\n");
        report.push_str(&format!("Total Gates: {}\n", self.total_gates));
        report.push_str(&format!("  Addition Gates: {}\n", self.add_count));
        report.push_str(&format!("  Multiplication Gates: {}\n", self.mul_count));
        report.push_str(&format!("  Subtraction Gates: {}\n", self.sub_count));
        report.push_str(&format!("  Comparison Gates: {}\n", self.compare_count));
        report.push_str(&format!("  Bitwise Gates: {}\n", self.bitwise_count));
        report.push_str(&format!("  Load/Store Gates: {}\n", self.load_store_count));
        report.push_str(&format!("  Constant Gates: {}\n", self.constant_count));
        report.push_str(&format!(
            "Multiplicative Depth: {}\n",
            self.multiplicative_depth
        ));
        report.push_str(&format!(
            "Critical Path Length: {}\n",
            self.critical_path_length
        ));
        report.push_str(&format!("Total Wires: {}\n", self.wire_count));
        report.push_str(&format!("Inputs: {}\n", self.input_count));
        report.push_str(&format!("Outputs: {}\n", self.output_count));

        // FHE performance estimation
        report.push_str("\n=== FHE Performance Estimation ===\n");
        let estimated_runtime = self.estimate_fhe_runtime();
        report.push_str(&format!(
            "Estimated FHE Runtime: {:.2} seconds\n",
            estimated_runtime
        ));

        let memory_usage = self.estimate_memory_usage();
        report.push_str(&format!("Estimated Memory Usage: {:.2} MB\n", memory_usage));

        report
    }

    /// Estimate FHE runtime based on gate complexity
    fn estimate_fhe_runtime(&self) -> f64 {
        // Rough estimates based on typical FHE library performance
        let add_time = self.add_count as f64 * 0.001; // 1ms per addition
        let mul_time = self.mul_count as f64 * 0.1; // 100ms per multiplication
        let sub_time = self.sub_count as f64 * 0.001; // 1ms per subtraction
        let other_time =
            (self.compare_count + self.bitwise_count + self.load_store_count) as f64 * 0.01;

        add_time + mul_time + sub_time + other_time
    }

    /// Estimate memory usage
    fn estimate_memory_usage(&self) -> f64 {
        // Rough estimate: each ciphertext ~1KB, each wire holds a ciphertext
        self.wire_count as f64 * 0.001 // Convert to MB
    }
}
