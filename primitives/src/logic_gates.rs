//! FHE arithmetic built from logic gates — the route DISCA did **not** take.
//!
//! This module composes half- and full-adders from `bit_and` / `bit_xor` / `bit_or`
//! over `FheUint8`, one encrypted bit per ciphertext. It is the *boolean-circuit*
//! approach to FHE: express everything as gates, then evaluate the gates.
//!
//! # Why it exists
//!
//! It is what the whitepaper's circuit-design section describes — "each stack
//! operation can be directly translated into FHE circuit gates" — and it was
//! written first, before the evaluator existed. Nothing in the evaluation path
//! calls it. `DiscaFunction::run` uses tfhe's high-level integer API instead:
//! `&a + &b` on `FheInt32`, `FheOrd::gt`, `IfThenElse::if_then_else`.
//!
//! # Why the integer API won
//!
//! Cost, by roughly two orders of magnitude. Measured in release:
//!
//! | | Measured |
//! |---|---|
//! | Native `FheInt32` addition (32 bits) | **225 ms** |
//! | This module's truth-table tests: 12 single-bit adder cases + 2 keygens | **7.28 s** |
//!
//! Backing out the ~1.4 s of key generation leaves roughly 0.5 s per single-bit
//! adder case, where a case is 2 gate operations (half-adder) or 5 (full-adder).
//! A 32-bit ripple-carry adder is 32 chained full-adders — on the order of 160
//! gate operations, so tens of seconds, against 225 ms for the native operation
//! that does the whole 32-bit addition. **One bit through this module costs more
//! than 32 bits through the integer API.** tfhe's radix representation batches
//! and parallelises work that gate composition serialises through a carry chain.
//!
//! That extrapolation is arithmetic on the two measured rows, not a benchmark of
//! a 32-bit adder built from these gates — nobody has run one.
//!
//! # Why it is kept
//!
//! As the documented alternative rather than as clutter. Two things it would be
//! the starting point for: bit-level operations the integer API does not expose,
//! and any future work on circuit privacy or custom gate sets where controlling
//! the gate sequence matters. Deleting it would leave the whitepaper describing
//! an approach with no implementation anywhere in the repo.
//!
//! # Not compiled by default
//!
//! Behind the `boolean-circuits` feature, so it is never linked into a binary.
//! Its truth-table tests were also ~7.3 s of the primitives suite — about a
//! third of it — spent guarding code nothing calls.
//!
//! ```sh
//! cargo test -p primitives --features boolean-circuits
//! ```
//!
//! Because it is off by default it will not be type-checked by an ordinary
//! build, so CI should include `--all-features` to keep it from rotting.

use tfhe::FheUint8;

pub fn bit_and(a: &FheUint8, b: &FheUint8) -> FheUint8 {
    a & b
}

pub fn bit_not(a: &FheUint8) -> FheUint8 {
    (a + 1_u8) & 1_u8
}

pub fn bit_or(a: &FheUint8, b: &FheUint8) -> FheUint8 {
    (a + b) - bit_and(a, b)
}

pub fn bit_xor(a: &FheUint8, b: &FheUint8) -> FheUint8 {
    (a + b) & 1_u8
}

pub fn bit_nand(a: &FheUint8, b: &FheUint8) -> FheUint8 {
    bit_not(&bit_and(a, b))
}

pub fn bit_nor(a: &FheUint8, b: &FheUint8) -> FheUint8 {
    bit_not(&bit_or(a, b))
}

pub fn bit_xnor(a: &FheUint8, b: &FheUint8) -> FheUint8 {
    bit_not(&bit_xor(a, b))
}

pub fn half_adder(a: &FheUint8, b: &FheUint8) -> (FheUint8, FheUint8) {
    let sum = bit_xor(a, b);
    let carry = bit_and(a, b);
    (sum, carry)
}

pub fn full_adder(a: &FheUint8, b: &FheUint8, carry_in: &FheUint8) -> (FheUint8, FheUint8) {
    let (sum1, carry1) = half_adder(a, b);
    let (sum2, carry2) = half_adder(&sum1, carry_in);
    let carry_out = bit_or(&carry1, &carry2);
    (sum2, carry_out)
}

#[cfg(test)]
mod tests {
    use std::sync::OnceLock;
    use std::time::Instant;

    use tfhe::{
        ClientKey, ConfigBuilder, ServerKey, generate_keys,
        prelude::{FheDecrypt, FheTryEncrypt},
        set_server_key,
    };

    use super::{full_adder, half_adder};

    static KEYS: OnceLock<(ClientKey, ServerKey)> = OnceLock::new();

    fn keys() -> (&'static ClientKey, ServerKey) {
        let (ck, sk) = KEYS.get_or_init(|| generate_keys(ConfigBuilder::default().build()));
        (ck, sk.clone())
    }

    #[test]
    fn half_adder_truth_table() {
        let t_keys = Instant::now();
        let (ck, sk) = keys();
        set_server_key(sk);
        let t_keys = t_keys.elapsed();

        let t_eval = Instant::now();
        let mut cases = 0u32;
        for a in 0u8..=1u8 {
            for b in 0u8..=1u8 {
                cases += 1;
                let ea = tfhe::FheUint8::try_encrypt(a, ck).unwrap();
                let eb = tfhe::FheUint8::try_encrypt(b, ck).unwrap();

                let (sum, carry) = half_adder(&ea, &eb);

                let sum: u8 = sum.decrypt(ck);
                let carry: u8 = carry.decrypt(ck);

                assert_eq!(sum, a ^ b);
                assert_eq!(carry, a & b);
            }
        }

        tracing::info!(
            key_setup_ms = t_keys.as_millis(),
            cases,
            eval_ms = t_eval.elapsed().as_millis(),
            "half_adder truth table verified"
        );
    }

    #[test]
    fn full_adder_truth_table() {
        let t_keys = Instant::now();
        let (ck, sk) = keys();
        set_server_key(sk);
        let t_keys = t_keys.elapsed();

        let t_eval = Instant::now();
        let mut cases = 0u32;
        for a in 0u8..=1u8 {
            for b in 0u8..=1u8 {
                for cin in 0u8..=1u8 {
                    cases += 1;
                    let ea = tfhe::FheUint8::try_encrypt(a, ck).unwrap();
                    let eb = tfhe::FheUint8::try_encrypt(b, ck).unwrap();
                    let ecin = tfhe::FheUint8::try_encrypt(cin, ck).unwrap();

                    let (sum, cout) = full_adder(&ea, &eb, &ecin);

                    let sum: u8 = sum.decrypt(ck);
                    let cout: u8 = cout.decrypt(ck);

                    let expected_sum = (a ^ b) ^ cin;
                    let expected_cout = (a & b) | (cin & (a ^ b));

                    assert_eq!(sum, expected_sum);
                    assert_eq!(cout, expected_cout);
                }
            }
        }

        tracing::info!(
            key_setup_ms = t_keys.as_millis(),
            cases,
            eval_ms = t_eval.elapsed().as_millis(),
            "full_adder truth table verified"
        );
    }
}
