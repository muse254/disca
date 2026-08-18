//! Stable wire encoding for compiled DISCA programs.
//!
//! The bridge pins `keccak256(bytecode)` on-chain at `registerProgram` time and
//! every worker is expected to independently arrive at the same hash for the
//! same program. That makes this encoding consensus-critical in a way the rest
//! of the crate is not: it must be deterministic across machines, and any
//! change to it is a breaking change to already-registered programs.
//!
//! Two properties give us that:
//!
//! * A magic-and-version header, so a format change is a loud deserialization
//!   failure rather than a silently different hash.
//! * A fixed-width, little-endian encoding, so the bytes do not depend on the
//!   host or on how the values happen to be laid out.
//!
//! The encoder is [`wincode`], which reproduces bincode 1.x's default wire
//! format (little-endian, fixed-width integers) from its own `SchemaWrite` /
//! `SchemaRead` derives. That byte-compatibility is load-bearing here, so
//! `matches_bincode_byte_for_byte` below pins it against bincode directly
//! rather than taking the claim on trust; bincode is retained as a
//! dev-dependency for exactly that check.
//!
//! Only the function list is encoded. Iteration state on [`DiscaProgram`] is a
//! cursor, not program content, and must never affect the hash.

use sha3::{Digest, Keccak256};

use crate::program::{DiscaFunction, DiscaProgram, ProgramError};

type Result<T> = std::result::Result<T, ProgramError>;

/// Identifies a DISCA bytecode blob. Chosen to be recognizable in a hex dump.
const MAGIC: [u8; 4] = *b"DSCA";

/// Bump on any change to the encoding of the payload that follows the header.
/// Programs registered under an older version keep their old hash and must be
/// re-registered, so this only moves once a version has actually been used to
/// register something.
pub const BYTECODE_VERSION: u16 = 1;

/// Length of the fixed header: magic (4) + version (2).
const HEADER_LEN: usize = 6;

/// Encodes a program into its canonical bytecode representation.
pub fn serialize(program: &DiscaProgram) -> Result<Vec<u8>> {
    let body = wincode::serialize(program.functions())
        .map_err(|e| ProgramError(format!("failed to encode bytecode: {e:?}")))?;

    let mut out = Vec::with_capacity(HEADER_LEN + body.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&BYTECODE_VERSION.to_le_bytes());
    out.extend_from_slice(&body);

    Ok(out)
}

/// Decodes bytecode produced by [`serialize`].
pub fn deserialize(bytes: &[u8]) -> Result<DiscaProgram> {
    if bytes.len() < HEADER_LEN {
        return Err(ProgramError("bytecode too short for header".into()));
    }

    if bytes[..4] != MAGIC {
        return Err(ProgramError("not a DISCA bytecode blob".into()));
    }

    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != BYTECODE_VERSION {
        return Err(ProgramError(format!(
            "unsupported bytecode version {version}, expected {BYTECODE_VERSION}"
        )));
    }

    // `deserialize_exact` rejects trailing bytes. A worker must not accept a
    // blob whose tail it ignored: it would execute one thing and attest to the
    // hash of another.
    let functions: Vec<DiscaFunction> = wincode::deserialize_exact(&bytes[HEADER_LEN..])
        .map_err(|e| ProgramError(format!("failed to decode bytecode: {e:?}")))?;

    Ok(DiscaProgram::from_functions(functions))
}

/// Returns `keccak256(serialize(program))` — the `bytecodeHash` the bridge
/// contract pins on-chain. Keccak256 (not SHA3-256) to match the EVM.
pub fn bytecode_hash(program: &DiscaProgram) -> Result<[u8; 32]> {
    Ok(hash_bytecode(&serialize(program)?))
}

/// Hashes an already-encoded bytecode blob. Workers receive bytes, not a parsed
/// program, and should verify them without a decode round-trip first.
pub fn hash_bytecode(bytecode: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(bytecode);
    hasher.finalize().into()
}

/// Renders a hash as the `0x`-prefixed hex the contract and CLI speak.
pub fn hex(hash: &[u8; 32]) -> String {
    let mut s = String::with_capacity(2 + 64);
    s.push_str("0x");
    for byte in hash {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::program::{CircuitOp, FuncSig, NumType, Program};

    fn sample() -> DiscaProgram {
        let wat = r#"
        (module
            (func $add (param i32 i32) (result i32)
              local.get 0
              local.get 1
              i32.add
            )
            (export "add" (func $add))
        )
        "#;
        DiscaProgram::from_program(&Program::from_wat(wat).unwrap())
    }

    #[test]
    fn round_trips_through_bytecode() {
        let program = sample();
        let bytes = serialize(&program).unwrap();
        let decoded = deserialize(&bytes).unwrap();

        assert_eq!(decoded.functions(), program.functions());
        assert_eq!(decoded.functions()[0].name.as_deref(), Some("add"));
        assert_eq!(
            decoded.functions()[0].body,
            vec![
                CircuitOp::LocalGet(0),
                CircuitOp::LocalGet(1),
                CircuitOp::Add
            ]
        );
        assert_eq!(
            decoded.functions()[0].sig,
            FuncSig {
                params: vec![NumType::I32, NumType::I32],
                results: vec![NumType::I32],
            }
        );
    }

    #[test]
    fn emits_magic_and_version_header() {
        let bytes = serialize(&sample()).unwrap();
        assert_eq!(&bytes[..4], b"DSCA");
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), BYTECODE_VERSION);
    }

    #[test]
    fn hash_is_stable_across_encodings() {
        let a = bytecode_hash(&sample()).unwrap();
        let b = bytecode_hash(&sample()).unwrap();
        assert_eq!(a, b, "same program must hash identically");
        assert_eq!(hex(&a).len(), 66);
        assert!(hex(&a).starts_with("0x"));
    }

    #[test]
    fn hash_ignores_iteration_state() {
        // The on-chain program identity must not depend on how far a local
        // iterator happens to have advanced.
        let pristine = sample();
        let mut advanced = sample();
        let _ = advanced.next();

        assert_eq!(
            bytecode_hash(&pristine).unwrap(),
            bytecode_hash(&advanced).unwrap()
        );
    }

    #[test]
    fn hash_changes_with_the_circuit() {
        let add = sample();
        let mul_wat = r#"
        (module
            (func $add (param i32 i32) (result i32)
              local.get 0
              local.get 1
              i32.mul
            )
            (export "add" (func $add))
        )
        "#;
        let mul = DiscaProgram::from_program(&Program::from_wat(mul_wat).unwrap());

        assert_ne!(
            bytecode_hash(&add).unwrap(),
            bytecode_hash(&mul).unwrap(),
            "a different circuit must not reuse a registered hash"
        );
    }

    #[test]
    fn rejects_foreign_and_truncated_blobs() {
        assert!(deserialize(b"xx").is_err(), "truncated header");
        assert!(deserialize(b"NOPEnot-disca-bytes").is_err(), "bad magic");

        let mut wrong_version = serialize(&sample()).unwrap();
        wrong_version[4] = 0xff;
        assert!(deserialize(&wrong_version).is_err(), "bad version");
    }

    #[test]
    fn matches_bincode_byte_for_byte() {
        // wincode is used for its speed and its audited, fuzzed decoder, but
        // the value we take from it here is its wire format: the encoding feeds
        // an on-chain hash, so a silent divergence from bincode 1.x's default
        // format would change every registered program's identity. Assert the
        // compatibility rather than trusting the README.
        use bincode::Options;

        let program = sample();

        let via_wincode = wincode::serialize(program.functions()).unwrap();
        let via_bincode = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_little_endian()
            .serialize(program.functions())
            .unwrap();

        assert_eq!(
            via_wincode, via_bincode,
            "wincode diverged from bincode 1.x default encoding"
        );
    }

    #[test]
    fn rejects_trailing_bytes() {
        // A worker must not accept a blob that hashes to something other than
        // what it executed.
        let mut padded = serialize(&sample()).unwrap();
        padded.push(0);
        assert!(deserialize(&padded).is_err());
    }
}
