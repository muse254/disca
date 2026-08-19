pub mod bytecode;
/// The boolean-circuit route to FHE arithmetic. Not used by the evaluator and
/// not compiled unless the `boolean-circuits` feature is on — see the module
/// docs for what it is and why it is kept.
#[cfg(feature = "boolean-circuits")]
pub mod logic_gates;
pub mod program;
pub mod validate;
pub mod wire;
