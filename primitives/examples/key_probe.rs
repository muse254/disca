//! Checks whether the server key a worker installs is reproducible.
//!
//! Workers all decompress their own copy of the same compressed key. If that
//! step is not deterministic, honest workers hold different evaluation keys and
//! cannot possibly agree on a result hash — which would break M-of-N.

use primitives::wire;
use tfhe::safe_serialization::safe_serialize;
use tfhe::{CompressedServerKey, ConfigBuilder, ServerKey, generate_keys};

fn bytes(key: &ServerKey) -> Vec<u8> {
    let mut out = Vec::new();
    safe_serialize(key, &mut out, 1 << 30).expect("serialize server key");
    out
}

fn main() {
    let (client_key, _) = generate_keys(ConfigBuilder::default().build());

    let c1 = wire::encode_server_key(&CompressedServerKey::new(&client_key)).unwrap();
    let c2 = wire::encode_server_key(&CompressedServerKey::new(&client_key)).unwrap();
    println!("CompressedServerKey::new twice identical: {}", c1 == c2);

    let d1 = bytes(&wire::decode_server_key(&c1).unwrap());
    let d2 = bytes(&wire::decode_server_key(&c1).unwrap());
    println!("decompress(same bytes) twice identical:  {}", d1 == d2);
    println!("decompressed size: {} bytes", d1.len());
}
