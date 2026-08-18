//! Static validation of a lowered circuit.
//!
//! [`DiscaFunction::run`] already fails on a malformed circuit, but it does so
//! part-way through evaluation — after minutes of homomorphic work, on a worker
//! that has already been paid attention by a coordinator. Validation moves those
//! failures to the point where bytecode is decoded, which matters most on the
//! untrusted path: [`crate::bytecode::deserialize`] runs this over every blob it
//! accepts, so a worker rejects a bad circuit before it evaluates a single gate.
//!
//! The walk also produces the information Phase 2 partitioning needs
//! (`architecture.md` §6): the points at which the whole intermediate state of
//! the circuit is a single ciphertext, and so the points at which the op
//! sequence can be cut and handed to another worker.

use crate::program::{CircuitOp, DiscaFunction, ProgramError};

type Result<T> = std::result::Result<T, ProgramError>;

/// What a validating walk learned about a circuit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircuitLayout {
    /// Deepest the operand stack ever gets. Every live value is a ciphertext,
    /// so this is the circuit's peak memory in units of ~258 KB.
    pub max_depth: usize,
    /// Indices after which exactly one value remains on the stack. Cutting the
    /// sequence here means only one ciphertext has to cross to another worker.
    pub split_points: Vec<usize>,
}

/// Net effect of an op on operand-stack depth, as `(popped, pushed)`.
fn arity(op: &CircuitOp) -> (usize, usize) {
    match op {
        CircuitOp::LocalGet(_) | CircuitOp::Const(_) => (0, 1),
        CircuitOp::LocalSet(_) | CircuitOp::Drop => (1, 0),
        // `local.tee` writes through to the local but leaves the value behind.
        CircuitOp::LocalTee(_) | CircuitOp::Eqz => (1, 1),
        CircuitOp::Add
        | CircuitOp::Sub
        | CircuitOp::Mul
        | CircuitOp::Eq
        | CircuitOp::Ne
        | CircuitOp::Lt
        | CircuitOp::Gt
        | CircuitOp::Le
        | CircuitOp::Ge => (2, 1),
        // Two candidates and a condition in, one result out.
        CircuitOp::Select => (3, 1),
    }
}

/// Checks that a circuit is well formed, returning what the walk learned.
///
/// Types are deliberately not tracked. The evaluator coerces between integers
/// and booleans wherever WASM's semantics call for it, so every operand
/// position accepts either shape and there is no type error to find — only
/// arity and addressing errors.
pub fn validate(func: &DiscaFunction) -> Result<CircuitLayout> {
    let frame_len = func.sig.params.len() + func.locals.len();

    let mut depth: usize = 0;
    let mut max_depth: usize = 0;
    let mut split_points = Vec::new();

    for (index, op) in func.body.iter().enumerate() {
        if let CircuitOp::LocalGet(i) | CircuitOp::LocalSet(i) | CircuitOp::LocalTee(i) = op
            && *i as usize >= frame_len
        {
            return Err(ProgramError(format!(
                "op {index}: local index {i} out of range ({frame_len} local(s): \
                 {} param + {} declared)",
                func.sig.params.len(),
                func.locals.len()
            )));
        }

        let (popped, pushed) = arity(op);
        depth = depth.checked_sub(popped).ok_or_else(|| {
            ProgramError(format!(
                "op {index} ({op:?}): stack underflow, needs {popped} operand(s) but depth is {depth}"
            ))
        })?;
        depth += pushed;

        max_depth = max_depth.max(depth);
        if depth == 1 {
            split_points.push(index);
        }
    }

    // A function declaring an i32 result must leave exactly that behind. More
    // than one value means the circuit computed something it never used.
    let expected = func.sig.results.len();
    if depth != expected {
        return Err(ProgramError(format!(
            "circuit leaves {depth} value(s) on the stack, signature declares {expected} result(s)"
        )));
    }

    Ok(CircuitLayout {
        max_depth,
        split_points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{DiscaProgram, Program};

    fn lower(wat: &str, name: &str) -> DiscaFunction {
        let program = DiscaProgram::from_program(&Program::from_wat(wat).unwrap());
        program.function(name).expect("exported function").clone()
    }

    #[test]
    fn accepts_a_well_formed_circuit_and_reports_its_shape() {
        let func = lower(
            r#"
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
            "#,
            "max",
        );

        let layout = validate(&func).unwrap();
        assert_eq!(layout.max_depth, 4, "four values live before the select");
        // Depth is 1 after the first load, then again once the select folds the
        // four live values back down to a result.
        assert_eq!(layout.split_points, vec![0, 5]);
    }

    #[test]
    fn finds_every_point_the_circuit_narrows_to_one_value() {
        // A chained tally narrows after each select, which is where Phase 2
        // would cut the sequence between workers.
        let func = lower(
            r#"
            (module
                (func $t (param i32 i32 i32) (result i32)
                  (local i32)
                  local.get 0
                  local.get 1
                  local.get 0
                  local.get 1
                  i32.gt_s
                  select
                  local.tee 3
                  local.get 2
                  local.get 3
                  local.get 2
                  i32.gt_s
                  select
                )
                (export "t" (func $t))
            )
            "#,
            "t",
        );

        let layout = validate(&func).unwrap();
        assert_eq!(
            layout.split_points,
            vec![0, 5, 6, 11],
            "narrows after each select and after the tee that saves its result"
        );
        assert_eq!(
            layout.split_points.last(),
            Some(&(func.body.len() - 1)),
            "the final op always narrows to the single result"
        );
    }

    #[test]
    fn rejects_stack_underflow() {
        // Hand-built rather than parsed: valid WASM cannot underflow, but
        // bytecode arriving over the network is not guaranteed to be valid.
        let func = DiscaFunction {
            name: Some("bad".into()),
            sig: crate::program::FuncSig {
                params: vec![crate::program::NumType::I32],
                results: vec![crate::program::NumType::I32],
            },
            locals: vec![],
            body: vec![CircuitOp::LocalGet(0), CircuitOp::Add],
        };

        let err = validate(&func).unwrap_err();
        assert!(err.to_string().contains("underflow"), "got: {err}");
        assert!(err.to_string().contains("op 1"), "names the op: {err}");
    }

    #[test]
    fn rejects_an_out_of_range_local() {
        let func = DiscaFunction {
            name: Some("bad".into()),
            sig: crate::program::FuncSig {
                params: vec![crate::program::NumType::I32],
                results: vec![crate::program::NumType::I32],
            },
            locals: vec![],
            body: vec![CircuitOp::LocalGet(7)],
        };

        let err = validate(&func).unwrap_err();
        assert!(err.to_string().contains("local index 7"), "got: {err}");
    }

    #[test]
    fn rejects_a_circuit_that_leaves_extra_values() {
        let func = DiscaFunction {
            name: Some("bad".into()),
            sig: crate::program::FuncSig {
                params: vec![crate::program::NumType::I32],
                results: vec![crate::program::NumType::I32],
            },
            locals: vec![],
            body: vec![CircuitOp::LocalGet(0), CircuitOp::LocalGet(0)],
        };

        let err = validate(&func).unwrap_err();
        assert!(err.to_string().contains("leaves 2 value(s)"), "got: {err}");
    }
}
