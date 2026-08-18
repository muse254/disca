use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use tfhe::FheInt32;
use wasmparser::{ExternalKind, Operator, Parser, Payload, TypeRef, ValType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramError(pub String);

impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for ProgramError {}

type Result<T> = std::result::Result<T, ProgramError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumType {
    I32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FuncSig {
    pub params: Vec<NumType>,
    pub results: Vec<NumType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instr {
    LocalGet(u32),
    I32Add,
    I32Mul,
    I32Sub,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitOp {
    LocalGet(u32),
    Add,
    Mul,
    Sub,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscaFunction {
    pub name: Option<String>,
    pub sig: FuncSig,
    pub body: Vec<CircuitOp>,
}

impl DiscaFunction {
    /// Runs the FHE circuit using stack-machine semantics and returns the result.
    pub fn run(&self, inputs: &[FheInt32]) -> Result<FheInt32> {
        let mut stack: Vec<FheInt32> = Vec::new();

        for op in &self.body {
            match op {
                CircuitOp::LocalGet(index) => {
                    let value = inputs
                        .get(*index as usize)
                        .ok_or_else(|| ProgramError("input index out of bounds".into()))?
                        .clone();
                    stack.push(value);
                }
                CircuitOp::Add => {
                    let (a, b) = pop2(&mut stack)?;
                    stack.push(&a + &b);
                }
                CircuitOp::Mul => {
                    let (a, b) = pop2(&mut stack)?;
                    stack.push(&a * &b);
                }
                CircuitOp::Sub => {
                    let (a, b) = pop2(&mut stack)?;
                    stack.push(&a - &b);
                }
            }
        }

        if stack.len() != 1 {
            return Err(ProgramError("invalid stack result".into()));
        }

        Ok(stack.pop().expect("checked len"))
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
                Instr::I32Add => CircuitOp::Add,
                Instr::I32Mul => CircuitOp::Mul,
                Instr::I32Sub => CircuitOp::Sub,
            })
            .collect()
    }
}

fn pop2(stack: &mut Vec<FheInt32>) -> Result<(FheInt32, FheInt32)> {
    let b = stack
        .pop()
        .ok_or_else(|| ProgramError("stack underflow".into()))?;
    let a = stack
        .pop()
        .ok_or_else(|| ProgramError("stack underflow".into()))?;
    Ok((a, b))
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
            Operator::I32Add => out.push(Instr::I32Add),
            Operator::I32Mul => out.push(Instr::I32Mul),
            Operator::I32Sub => out.push(Instr::I32Sub),
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
}
