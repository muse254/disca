//! Error types for DISCA parser

use thiserror::Error;

/// Result type alias for DISCA operations
pub type Result<T> = std::result::Result<T, DiscaError>;

/// Main error type for DISCA parser operations
#[derive(Error, Debug)]
pub enum DiscaError {
    #[error("Internal error: {0}")]
    InternalError(String),

    /// WASM parsing errors
    #[error("WASM parsing error: {0}")]
    WasmParsing(#[from] wasmparser::BinaryReaderError),

    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Circuit validation errors
    #[error("Circuit validation error: {0}")]
    CircuitValidation(String),

    /// Optimization errors
    #[error("Optimization error: {0}")]
    Optimization(String),

    /// Binary encoding errors
    #[error("Binary encoding error: {0}")]
    BinaryEncoding(String),

    /// Wire mapping errors
    #[error("Wire mapping error: {0}")]
    WireMapping(String),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Gate execution error
    #[error("Gate execution error: {0}")]
    GateExecution(String),
}

impl DiscaError {
    /// Create a circuit validation error
    pub fn circuit_validation<S: Into<String>>(msg: S) -> Self {
        Self::CircuitValidation(msg.into())
    }

    /// Create an optimization error
    pub fn optimization<S: Into<String>>(msg: S) -> Self {
        Self::Optimization(msg.into())
    }

    /// Create a binary encoding error
    pub fn binary_encoding<S: Into<String>>(msg: S) -> Self {
        Self::BinaryEncoding(msg.into())
    }

    /// Create a wire mapping error
    pub fn wire_mapping<S: Into<String>>(msg: S) -> Self {
        Self::WireMapping(msg.into())
    }

    /// Create an unsupported operation error
    pub fn unsupported_operation<S: Into<String>>(msg: S) -> Self {
        Self::UnsupportedOperation(msg.into())
    }

    /// Create an invalid input error
    pub fn invalid_input<S: Into<String>>(msg: S) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a gate execution error
    pub fn gate_execution<S: Into<String>>(msg: S) -> Self {
        Self::GateExecution(msg.into())
    }
}
