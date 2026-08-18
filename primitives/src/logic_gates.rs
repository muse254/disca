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
