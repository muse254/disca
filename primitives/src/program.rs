use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use tfhe::prelude::{CastFrom, FheEq, FheOrd, FheTrivialEncrypt, IfThenElse};
use tfhe::{FheBool, FheInt32};
use tracing::{Level, debug, enabled, trace};
use wasmparser::{ExternalKind, Operator, Parser, Payload, TypeRef, ValType};
use wincode::{SchemaRead, SchemaWrite};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramError(pub String);

impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for ProgramError {}

type Result<T> = std::result::Result<T, ProgramError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, SchemaWrite, SchemaRead)]
pub enum NumType {
    I32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, SchemaWrite, SchemaRead)]
pub struct FuncSig {
    pub params: Vec<NumType>,
    pub results: Vec<NumType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instr {
    LocalGet(u32),
    LocalSet(u32),
    LocalTee(u32),
    I32Const(i32),
    Drop,
    I32Add,
    I32Mul,
    I32Sub,
    I32Eq,
    I32Ne,
    I32Eqz,
    I32LtS,
    I32GtS,
    I32LeS,
    I32GeS,
    Select,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, SchemaWrite, SchemaRead)]
pub enum CircuitOp {
    LocalGet(u32),
    LocalSet(u32),
    LocalTee(u32),
    /// A literal from the program text. Constants are part of the bytecode and
    /// therefore already public, so they are trivially encrypted rather than
    /// treated as secret inputs.
    Const(i32),
    Drop,
    Add,
    Mul,
    Sub,
    Eq,
    Ne,
    /// `i32.eqz` — compares against zero.
    Eqz,
    Lt,
    Gt,
    Le,
    Ge,
    Select,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, SchemaWrite, SchemaRead)]
pub struct DiscaFunction {
    pub name: Option<String>,
    pub sig: FuncSig,
    /// Locals declared beyond the parameters. WASM addresses parameters and
    /// declared locals through a single index space, parameters first.
    pub locals: Vec<NumType>,
    pub body: Vec<CircuitOp>,
}

/// A value on the evaluation stack.
///
/// WASM has no boolean type: comparisons push an `i32` that is 0 or 1, and
/// `select` tests its condition against zero. FHE draws the line differently —
/// comparisons yield an [`FheBool`] and `if_then_else` requires one. Modelling
/// both shapes lets the common compare-then-select path stay cast-free, and
/// confines conversion to the rare program that does arithmetic on a
/// comparison result.
#[derive(Clone)]
enum Value {
    Int(FheInt32),
    Bool(FheBool),
}

impl Value {
    /// Coerces to an integer, matching WASM's `i32` view of a boolean.
    fn into_int(self) -> FheInt32 {
        match self {
            Self::Int(v) => v,
            Self::Bool(v) => FheInt32::cast_from(v),
        }
    }

    /// Coerces to a boolean, matching WASM's "nonzero is true" rule.
    fn into_bool(self) -> FheBool {
        match self {
            Self::Bool(v) => v,
            Self::Int(v) => v.ne(0i32),
        }
    }
}

impl DiscaFunction {
    /// Runs the FHE circuit using stack-machine semantics and returns the result.
    ///
    /// `inputs` supplies the function's parameters, in order. Locals declared
    /// beyond the parameters start at a trivially encrypted zero, as WASM
    /// requires; "trivial" here means the ciphertext carries no secret, which is
    /// correct because a zero-initialised local is not private information.
    ///
    /// Emits a `circuit.run` span. Homomorphic evaluation is where essentially
    /// all of a worker's wall-clock goes, so this span is the measurement a
    /// coordinator needs to reason about job latency; per-op timings are
    /// available at `TRACE` when a circuit needs profiling.
    #[tracing::instrument(
        name = "circuit.run",
        level = "debug",
        skip_all,
        fields(
            function = self.name.as_deref().unwrap_or("<anonymous>"),
            ops = self.body.len(),
            inputs = inputs.len(),
        )
    )]
    pub fn run(&self, inputs: &[FheInt32]) -> Result<FheInt32> {
        if inputs.len() != self.sig.params.len() {
            return Err(ProgramError(format!(
                "expected {} input(s), got {}",
                self.sig.params.len(),
                inputs.len()
            )));
        }

        let mut frame: Vec<Value> = Vec::with_capacity(inputs.len() + self.locals.len());
        frame.extend(inputs.iter().cloned().map(Value::Int));
        frame.extend(
            self.locals
                .iter()
                .map(|_| Value::Int(FheInt32::encrypt_trivial(0i32))),
        );

        let mut stack: Vec<Value> = Vec::new();
        // Timing every op costs an Instant per op, so only pay it when someone
        // is actually listening at TRACE.
        let profiling = enabled!(Level::TRACE);
        let started = std::time::Instant::now();

        for (index, op) in self.body.iter().enumerate() {
            let op_started = profiling.then(std::time::Instant::now);

            match op {
                CircuitOp::LocalGet(index) => {
                    stack.push(local(&frame, *index)?.clone());
                }
                CircuitOp::LocalSet(index) => {
                    let value = pop(&mut stack)?;
                    *local_mut(&mut frame, *index)? = value;
                }
                CircuitOp::LocalTee(index) => {
                    // `local.tee` writes the value but leaves it on the stack.
                    let value = stack
                        .last()
                        .ok_or_else(|| ProgramError("stack underflow".into()))?
                        .clone();
                    *local_mut(&mut frame, *index)? = value;
                }
                CircuitOp::Const(value) => {
                    stack.push(Value::Int(FheInt32::encrypt_trivial(*value)));
                }
                CircuitOp::Drop => {
                    pop(&mut stack)?;
                }

                CircuitOp::Add => {
                    let (a, b) = pop2_int(&mut stack)?;
                    stack.push(Value::Int(&a + &b));
                }
                CircuitOp::Mul => {
                    let (a, b) = pop2_int(&mut stack)?;
                    stack.push(Value::Int(&a * &b));
                }
                CircuitOp::Sub => {
                    let (a, b) = pop2_int(&mut stack)?;
                    stack.push(Value::Int(&a - &b));
                }

                CircuitOp::Eq => {
                    let (a, b) = pop2_int(&mut stack)?;
                    stack.push(Value::Bool(a.eq(&b)));
                }
                CircuitOp::Ne => {
                    let (a, b) = pop2_int(&mut stack)?;
                    stack.push(Value::Bool(a.ne(&b)));
                }
                CircuitOp::Eqz => {
                    let a = pop(&mut stack)?.into_int();
                    stack.push(Value::Bool(a.eq(0i32)));
                }
                CircuitOp::Lt => {
                    let (a, b) = pop2_int(&mut stack)?;
                    stack.push(Value::Bool(a.lt(&b)));
                }
                CircuitOp::Gt => {
                    let (a, b) = pop2_int(&mut stack)?;
                    stack.push(Value::Bool(a.gt(&b)));
                }
                CircuitOp::Le => {
                    let (a, b) = pop2_int(&mut stack)?;
                    stack.push(Value::Bool(a.le(&b)));
                }
                CircuitOp::Ge => {
                    let (a, b) = pop2_int(&mut stack)?;
                    stack.push(Value::Bool(a.ge(&b)));
                }

                CircuitOp::Select => {
                    // WASM pushes the candidates first and the condition last,
                    // and yields the first candidate when the condition is true.
                    let condition = pop(&mut stack)?.into_bool();
                    let (if_true, if_false) = pop2_int(&mut stack)?;
                    stack.push(Value::Int(condition.if_then_else(&if_true, &if_false)));
                }
            }

            if let Some(op_started) = op_started {
                trace!(
                    index,
                    op = ?op,
                    depth = stack.len(),
                    elapsed_ms = op_started.elapsed().as_millis(),
                    "op evaluated"
                );
            }
        }

        debug!(
            elapsed_ms = started.elapsed().as_millis(),
            "circuit evaluated"
        );

        if stack.len() != 1 {
            return Err(ProgramError(format!(
                "invalid stack result: expected 1 value, found {}",
                stack.len()
            )));
        }

        // The signature declares an i32 result, so a trailing comparison is
        // widened back to WASM's integer view of a boolean.
        Ok(stack.pop().expect("checked len").into_int())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscaProgram {
    functions: Vec<DiscaFunction>,
    /// Cursor for the [`Iterator`] impl. Iteration progress is not part of the
    /// program's identity and is excluded from its bytecode hash.
    next_index: usize,
}

impl DiscaProgram {
    /// Creates a DiscaProgram from the parsed WASM program.
    pub fn from_program(program: &Program) -> Self {
        let functions = program
            .functions
            .iter()
            .map(|func| DiscaFunction {
                name: func.name.clone(),
                sig: func.sig.clone(),
                locals: func.locals.clone(),
                body: func.circuit_sequence(),
            })
            .collect();

        Self::from_functions(functions)
    }

    /// Creates a program directly from lowered functions, as when decoding
    /// bytecode received over the wire.
    pub fn from_functions(functions: Vec<DiscaFunction>) -> Self {
        Self {
            functions,
            next_index: 0,
        }
    }

    /// The program's functions, in declaration order.
    pub fn functions(&self) -> &[DiscaFunction] {
        &self.functions
    }

    /// Looks up an exported function by name.
    pub fn function(&self, name: &str) -> Option<&DiscaFunction> {
        self.functions
            .iter()
            .find(|f| f.name.as_deref() == Some(name))
    }
}

impl Iterator for DiscaProgram {
    type Item = DiscaFunction;

    /// Returns the next function in the program, if any.
    fn next(&mut self) -> Option<DiscaFunction> {
        let func = self.functions.get(self.next_index).cloned();
        if func.is_some() {
            self.next_index += 1;
        }
        func
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: Option<String>,
    pub sig: FuncSig,
    pub locals: Vec<NumType>,
    pub body: Vec<Instr>,
}

impl Function {
    /// Returns a linear circuit sequence representing this function's stack machine.
    pub fn circuit_sequence(&self) -> Vec<CircuitOp> {
        self.body
            .iter()
            .map(|instr| match instr {
                Instr::LocalGet(idx) => CircuitOp::LocalGet(*idx),
                Instr::LocalSet(idx) => CircuitOp::LocalSet(*idx),
                Instr::LocalTee(idx) => CircuitOp::LocalTee(*idx),
                Instr::I32Const(v) => CircuitOp::Const(*v),
                Instr::Drop => CircuitOp::Drop,
                Instr::I32Add => CircuitOp::Add,
                Instr::I32Mul => CircuitOp::Mul,
                Instr::I32Sub => CircuitOp::Sub,
                Instr::I32Eq => CircuitOp::Eq,
                Instr::I32Ne => CircuitOp::Ne,
                Instr::I32Eqz => CircuitOp::Eqz,
                Instr::I32LtS => CircuitOp::Lt,
                Instr::I32GtS => CircuitOp::Gt,
                Instr::I32LeS => CircuitOp::Le,
                Instr::I32GeS => CircuitOp::Ge,
                Instr::Select => CircuitOp::Select,
            })
            .collect()
    }
}

fn pop(stack: &mut Vec<Value>) -> Result<Value> {
    stack
        .pop()
        .ok_or_else(|| ProgramError("stack underflow".into()))
}

/// Pops two operands as integers. The returned pair is `(deeper, shallower)`,
/// so `a op b` reads the same way the WASM source did.
fn pop2_int(stack: &mut Vec<Value>) -> Result<(FheInt32, FheInt32)> {
    let b = pop(stack)?.into_int();
    let a = pop(stack)?.into_int();
    Ok((a, b))
}

fn local(frame: &[Value], index: u32) -> Result<&Value> {
    frame
        .get(index as usize)
        .ok_or_else(|| ProgramError(format!("local index {index} out of bounds")))
}

fn local_mut(frame: &mut [Value], index: u32) -> Result<&mut Value> {
    frame
        .get_mut(index as usize)
        .ok_or_else(|| ProgramError(format!("local index {index} out of bounds")))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    pub functions: Vec<Function>,
}

impl Program {
    pub fn from_wat(wat: &str) -> Result<Self> {
        let buf = wast::parser::ParseBuffer::new(wat)
            .map_err(|e| ProgramError(format!("failed to parse wat: {e}")))?;
        let mut wat_ast: wast::Wat<'_> =
            wast::parser::parse(&buf).map_err(|e| ProgramError(format!("invalid wat: {e}")))?;
        let wasm = wat_ast
            .encode()
            .map_err(|e| ProgramError(format!("failed to encode wat: {e}")))?;
        Self::from_wasm(&wasm)
    }

    pub fn from_wasm(wasm: &[u8]) -> Result<Self> {
        let mut type_sigs: Vec<FuncSig> = Vec::new();
        let mut func_type_indices: Vec<u32> = Vec::new();
        let mut import_func_count: u32 = 0;
        let mut exports: HashMap<u32, String> = HashMap::new();

        let mut defined_funcs: Vec<Function> = Vec::new();
        let mut next_code_body: u32 = 0;

        for payload in Parser::new(0).parse_all(wasm) {
            match payload.map_err(to_err)? {
                Payload::TypeSection(reader) => {
                    for ft in reader.into_iter_err_on_gc_types() {
                        let ft = ft.map_err(to_err)?;
                        type_sigs.push(func_sig_from_func_type(&ft)?);
                    }
                }

                Payload::ImportSection(reader) => {
                    for import in reader.into_imports() {
                        let import = import.map_err(to_err)?;
                        if matches!(import.ty, TypeRef::Func(_)) {
                            import_func_count += 1;
                        }
                    }
                }

                Payload::FunctionSection(reader) => {
                    for f in reader {
                        func_type_indices.push(f.map_err(to_err)?);
                    }
                }

                Payload::ExportSection(reader) => {
                    for export in reader {
                        let export = export.map_err(to_err)?;
                        if export.kind == ExternalKind::Func {
                            exports.insert(export.index, export.name.to_string());
                        }
                    }
                }

                Payload::CodeSectionEntry(body) => {
                    let func_type_idx = *func_type_indices
                        .get(next_code_body as usize)
                        .ok_or_else(|| {
                            ProgramError("code section entry without function type".into())
                        })?;

                    let sig = type_sigs
                        .get(func_type_idx as usize)
                        .ok_or_else(|| ProgramError("function type index out of range".into()))?
                        .clone();

                    let locals = parse_locals(&body)?;
                    let instructions = parse_instructions(&body)?;

                    let func_index = import_func_count + next_code_body;
                    let name = exports.get(&func_index).cloned();

                    defined_funcs.push(Function {
                        name,
                        sig,
                        locals,
                        body: instructions,
                    });

                    next_code_body += 1;
                }

                Payload::End(_) => break,

                _ => {}
            }
        }

        Ok(Program {
            functions: defined_funcs,
        })
    }
}

fn to_err(e: wasmparser::BinaryReaderError) -> ProgramError {
    ProgramError(e.to_string())
}

fn num_type_from_val_type(t: ValType) -> Result<NumType> {
    match t {
        ValType::I32 => Ok(NumType::I32),
        other => Err(ProgramError(format!("unsupported val type: {other:?}"))),
    }
}

fn func_sig_from_func_type(ft: &wasmparser::FuncType) -> Result<FuncSig> {
    let params = ft
        .params()
        .iter()
        .copied()
        .map(num_type_from_val_type)
        .collect::<Result<Vec<_>>>()?;

    let results = ft
        .results()
        .iter()
        .copied()
        .map(num_type_from_val_type)
        .collect::<Result<Vec<_>>>()?;

    if params
        .iter()
        .chain(results.iter())
        .any(|t| *t != NumType::I32)
    {
        return Err(ProgramError("only i32 is supported".into()));
    }

    Ok(FuncSig { params, results })
}

fn parse_locals(body: &wasmparser::FunctionBody<'_>) -> Result<Vec<NumType>> {
    let mut locals_reader = body.get_locals_reader().map_err(to_err)?;
    let mut locals = Vec::new();

    for _ in 0..locals_reader.get_count() {
        let (count, t) = locals_reader.read().map_err(to_err)?;
        let t = num_type_from_val_type(t)?;
        locals.extend(std::iter::repeat_n(t, count as usize));
    }

    Ok(locals)
}

fn parse_instructions(body: &wasmparser::FunctionBody<'_>) -> Result<Vec<Instr>> {
    let mut ops = body.get_operators_reader().map_err(to_err)?;
    let mut out = Vec::new();

    while !ops.eof() {
        match ops.read().map_err(to_err)? {
            Operator::LocalGet { local_index } => out.push(Instr::LocalGet(local_index)),
            Operator::LocalSet { local_index } => out.push(Instr::LocalSet(local_index)),
            Operator::LocalTee { local_index } => out.push(Instr::LocalTee(local_index)),
            Operator::I32Const { value } => out.push(Instr::I32Const(value)),
            Operator::Drop => out.push(Instr::Drop),
            Operator::I32Add => out.push(Instr::I32Add),
            Operator::I32Mul => out.push(Instr::I32Mul),
            Operator::I32Sub => out.push(Instr::I32Sub),
            Operator::I32Eq => out.push(Instr::I32Eq),
            Operator::I32Ne => out.push(Instr::I32Ne),
            Operator::I32Eqz => out.push(Instr::I32Eqz),
            Operator::I32LtS => out.push(Instr::I32LtS),
            Operator::I32GtS => out.push(Instr::I32GtS),
            Operator::I32LeS => out.push(Instr::I32LeS),
            Operator::I32GeS => out.push(Instr::I32GeS),
            Operator::Select => out.push(Instr::Select),
            // The unsigned comparisons would need FheUint32 operands, but the
            // IR models i32 as signed throughout. Rejecting is better than
            // silently evaluating them with signed semantics.
            Operator::I32LtU | Operator::I32GtU | Operator::I32LeU | Operator::I32GeU => {
                return Err(ProgramError(
                    "unsigned i32 comparisons are not supported; the IR treats i32 as signed"
                        .into(),
                ));
            }
            Operator::End => {}
            other => {
                return Err(ProgramError(format!("unsupported operator: {other:?}")));
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_wat_module_into_ir() {
        let wat = r#"
        (module
            (func $add (param i32 i32) (result i32)
              local.get 1
              local.get 0
              i32.add
            )
            (func $multiply (param i32 i32) (result i32)
              local.get 1
              local.get 0
              i32.mul
            )
            (func $subtract (param i32 i32) (result i32)
              local.get 0
              local.get 1
              i32.sub
            )
            (export "add" (func $add))
            (export "multiply" (func $multiply))
            (export "subtract" (func $subtract))
          )
        "#;

        let program = Program::from_wat(wat).unwrap();

        assert_eq!(program.functions.len(), 3);

        assert_eq!(program.functions[0].name.as_deref(), Some("add"));
        assert_eq!(
            program.functions[0].sig,
            FuncSig {
                params: vec![NumType::I32, NumType::I32],
                results: vec![NumType::I32],
            }
        );
        assert!(program.functions[0].locals.is_empty());
        assert_eq!(
            program.functions[0].body,
            vec![Instr::LocalGet(1), Instr::LocalGet(0), Instr::I32Add]
        );
        assert_eq!(
            program.functions[0].circuit_sequence(),
            vec![
                CircuitOp::LocalGet(1),
                CircuitOp::LocalGet(0),
                CircuitOp::Add
            ]
        );

        assert_eq!(program.functions[1].name.as_deref(), Some("multiply"));
        assert_eq!(
            program.functions[1].body,
            vec![Instr::LocalGet(1), Instr::LocalGet(0), Instr::I32Mul]
        );

        assert_eq!(program.functions[2].name.as_deref(), Some("subtract"));
        assert_eq!(
            program.functions[2].body,
            vec![Instr::LocalGet(0), Instr::LocalGet(1), Instr::I32Sub]
        );
    }

    #[test]
    fn parses_the_expanded_opcode_set() {
        let wat = r#"
        (module
            (func $pick (param i32 i32) (result i32)
              (local i32)
              local.get 0
              local.get 1
              local.get 0
              local.get 1
              i32.gt_s
              select
              local.set 2
              local.get 2
            )
            (export "pick" (func $pick))
        )
        "#;

        let program = Program::from_wat(wat).unwrap();
        let func = &program.functions[0];

        assert_eq!(func.locals, vec![NumType::I32], "one declared local");
        assert_eq!(
            func.circuit_sequence(),
            vec![
                CircuitOp::LocalGet(0),
                CircuitOp::LocalGet(1),
                CircuitOp::LocalGet(0),
                CircuitOp::LocalGet(1),
                CircuitOp::Gt,
                CircuitOp::Select,
                CircuitOp::LocalSet(2),
                CircuitOp::LocalGet(2),
            ]
        );
    }

    #[test]
    fn parses_const_tee_and_drop() {
        let wat = r#"
        (module
            (func $f (param i32) (result i32)
              (local i32)
              i32.const 42
              local.tee 1
              drop
              local.get 0
              i32.const -7
              i32.add
            )
            (export "f" (func $f))
        )
        "#;

        let func = &Program::from_wat(wat).unwrap().functions[0];
        assert_eq!(
            func.circuit_sequence(),
            vec![
                CircuitOp::Const(42),
                CircuitOp::LocalTee(1),
                CircuitOp::Drop,
                CircuitOp::LocalGet(0),
                CircuitOp::Const(-7),
                CircuitOp::Add,
            ]
        );
    }

    #[test]
    fn rejects_unsigned_comparisons() {
        let wat = r#"
        (module
            (func $f (param i32 i32) (result i32)
              local.get 0
              local.get 1
              i32.lt_u
            )
        )
        "#;

        let err = Program::from_wat(wat).unwrap_err();
        assert!(
            err.to_string().contains("unsigned"),
            "expected an unsigned-comparison error, got: {err}"
        );
    }
}

/// Executes circuits under real encryption and checks the plaintext results.
///
/// Parsing tests prove we read the right opcodes; these prove we evaluate them
/// with the right semantics, which is where the WASM/FHE impedance mismatch
/// (integer booleans vs `FheBool`, operand order on the stack) actually bites.
#[cfg(test)]
mod exec_tests {
    use super::*;

    use tfhe::prelude::{FheDecrypt, FheTryEncrypt};
    use tfhe::{ClientKey, ConfigBuilder, generate_keys, set_server_key};

    /// Holds the client key so a test can encrypt inputs and read results back.
    /// Key generation is per-test because tfhe installs the server key in
    /// thread-local storage and tests run on their own threads.
    struct Harness {
        client_key: ClientKey,
    }

    impl Harness {
        fn new() -> Self {
            let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
            set_server_key(server_key);
            Self { client_key }
        }

        fn run(&self, wat: &str, name: &str, inputs: &[i32]) -> i32 {
            let program = DiscaProgram::from_program(&Program::from_wat(wat).unwrap());
            let func = program
                .function(name)
                .unwrap_or_else(|| panic!("no exported function named {name}"));

            let encrypted: Vec<FheInt32> = inputs
                .iter()
                .map(|v| FheInt32::try_encrypt(*v, &self.client_key).expect("encrypt input"))
                .collect();

            func.run(&encrypted)
                .expect("run circuit")
                .decrypt(&self.client_key)
        }
    }

    const MAX: &str = r#"
    (module
        (func $max (param i32 i32) (result i32)
          local.get 0
          local.get 1
          local.get 0
          local.get 1
          i32.gt_s
          select
        )
        (export "max" (func $max))
    )
    "#;

    #[test]
    fn compare_and_select_compute_max() {
        let h = Harness::new();
        // Operand order is the easy thing to get backwards, so check both
        // directions and the tie.
        assert_eq!(h.run(MAX, "max", &[3, 9]), 9);
        assert_eq!(h.run(MAX, "max", &[9, 3]), 9);
        assert_eq!(h.run(MAX, "max", &[5, 5]), 5);
        assert_eq!(h.run(MAX, "max", &[-8, -2]), -2, "signed comparison");
    }

    #[test]
    fn local_tee_writes_and_leaves_the_value_on_the_stack() {
        let wat = r#"
        (module
            (func $twice (param i32) (result i32)
              (local i32)
              local.get 0
              i32.const 10
              i32.add
              local.tee 1
              local.get 1
              i32.add
            )
            (export "twice" (func $twice))
        )
        "#;

        let h = Harness::new();
        assert_eq!(h.run(wat, "twice", &[4]), 28, "2 * (4 + 10)");
        assert_eq!(h.run(wat, "twice", &[-5]), 10, "2 * (-5 + 10)");
    }

    #[test]
    fn local_set_const_and_drop() {
        let wat = r#"
        (module
            (func $g (param i32 i32) (result i32)
              (local i32)
              local.get 0
              local.get 1
              i32.sub
              local.set 2
              i32.const 999
              drop
              local.get 2
            )
            (export "g" (func $g))
        )
        "#;

        let h = Harness::new();
        // Subtraction also pins operand order: 7 - 2, not 2 - 7.
        assert_eq!(h.run(wat, "g", &[7, 2]), 5);
    }

    #[test]
    fn comparisons_widen_to_wasm_integer_booleans() {
        let wat = r#"
        (module
            (func $lt (param i32 i32) (result i32)
              local.get 0
              local.get 1
              i32.lt_s
            )
            (func $is_zero (param i32) (result i32)
              local.get 0
              i32.eqz
            )
            (export "lt" (func $lt))
            (export "is_zero" (func $is_zero))
        )
        "#;

        let h = Harness::new();
        // A comparison left on the stack is the function's i32 result, so it
        // has to come back as 1 or 0 rather than as an FheBool.
        assert_eq!(h.run(wat, "lt", &[2, 7]), 1);
        assert_eq!(h.run(wat, "lt", &[7, 2]), 0);
        assert_eq!(h.run(wat, "is_zero", &[0]), 1);
        assert_eq!(h.run(wat, "is_zero", &[3]), 0);
    }

    #[test]
    fn arithmetic_on_a_comparison_result_coerces_it() {
        // Counting pattern: sum of predicate results. Exercises the Bool -> Int
        // coercion on the arithmetic path.
        let wat = r#"
        (module
            (func $count (param i32 i32) (result i32)
              local.get 0
              i32.eqz
              local.get 1
              i32.eqz
              i32.add
            )
            (export "count" (func $count))
        )
        "#;

        let h = Harness::new();
        assert_eq!(h.run(wat, "count", &[0, 0]), 2);
        assert_eq!(h.run(wat, "count", &[0, 1]), 1);
        assert_eq!(h.run(wat, "count", &[1, 1]), 0);
    }

    #[test]
    fn rejects_a_wrong_input_count() {
        let h = Harness::new();
        let program = DiscaProgram::from_program(&Program::from_wat(MAX).unwrap());
        let func = program.function("max").unwrap();

        let one = vec![FheInt32::try_encrypt(1i32, &h.client_key).unwrap()];
        // `unwrap_err` is unavailable here: ciphertexts have no `Debug`.
        let Err(err) = func.run(&one) else {
            panic!("a one-input call to a two-parameter circuit must fail");
        };
        assert!(
            err.to_string().contains("expected 2 input(s), got 1"),
            "got: {err}"
        );
    }
}
