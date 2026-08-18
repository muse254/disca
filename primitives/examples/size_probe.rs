//! Measures the tfhe-rs artifact sizes and operation latencies that
//! `docs/architecture.md` §2 records.
//!
//! Every placement decision in the architecture (what goes on-chain, what the
//! wire format is, how big a demo circuit can be) rests on these numbers, so
//! they need to be reproducible rather than remembered.
//!
//! Run it in release — debug numbers are 10-100x pessimistic and not
//! representative of anything we would ship:
//!
//! ```sh
//! cargo run --release -p primitives --example size_probe
//! cargo run --release -p primitives --example size_probe -- --public-key
//! ```
//!
//! `--public-key` additionally generates a public-key-encryption key. It is
//! measured in gigabytes and takes minutes, which is precisely the finding that
//! puts public-key mode out of scope; it stays opt-in so the default run is fast.

use std::time::{Duration, Instant};

use tfhe::prelude::{FheDecrypt, FheEncrypt, FheOrd, FheTryEncrypt, IfThenElse};
use tfhe::safe_serialization::safe_serialized_size;
use tfhe::{
    CompressedFheInt32, CompressedServerKey, ConfigBuilder, FheInt32, PublicKey, generate_keys,
    set_server_key,
};

fn main() {
    let measure_public_key = std::env::args().any(|a| a == "--public-key");

    println!(
        "DISCA size/latency probe — tfhe-rs 1.5, {} build\n",
        profile()
    );

    let config = ConfigBuilder::default().build();

    let (keygen_time, (client_key, server_key)) = timed(|| generate_keys(config));

    // --- key artifacts -------------------------------------------------------
    println!("== keys ==");
    report_size("client key (secret)", &client_key);
    report_size("server key (public eval)", &server_key);

    let (compress_time, compressed_server_key) = timed(|| CompressedServerKey::new(&client_key));
    report_size("server key (compressed)", &compressed_server_key);
    println!(
        "{:<34} {:>12}",
        "server key compression",
        format_duration(compress_time)
    );

    if measure_public_key {
        let (pk_time, public_key) = timed(|| PublicKey::new(&client_key));
        report_size("public key (pk-encryption)", &public_key);
        println!(
            "{:<34} {:>12}",
            "public key generation",
            format_duration(pk_time)
        );
    } else {
        println!("{:<34} {:>12}", "public key (pk-encryption)", "(skipped)");
    }
    println!(
        "{:<34} {:>12}",
        "key generation",
        format_duration(keygen_time)
    );

    // --- ciphertext artifacts ------------------------------------------------
    println!("\n== ciphertexts ==");
    let a = FheInt32::try_encrypt(7i32, &client_key).expect("encrypt a");
    let b = FheInt32::try_encrypt(11i32, &client_key).expect("encrypt b");
    report_size("FheInt32 (uncompressed)", &a);

    let (compress_ct_time, compressed) = timed(|| CompressedFheInt32::encrypt(7i32, &client_key));
    report_size("CompressedFheInt32", &compressed);
    println!(
        "{:<34} {:>12}",
        "ciphertext compression",
        format_duration(compress_ct_time)
    );

    let (decompress_time, _) = timed(|| compressed.decompress());
    println!(
        "{:<34} {:>12}",
        "ciphertext decompression",
        format_duration(decompress_time)
    );

    // --- evaluation latency --------------------------------------------------
    // Everything past this point needs the server key installed.
    set_server_key(server_key);

    println!("\n== evaluation (i32) ==");
    let (add_time, sum) = timed(|| &a + &b);
    println!("{:<34} {:>12}", "add", format_duration(add_time));

    let (sub_time, _) = timed(|| &a - &b);
    println!("{:<34} {:>12}", "sub", format_duration(sub_time));

    let (mul_time, _) = timed(|| &a * &b);
    println!("{:<34} {:>12}", "mul", format_duration(mul_time));

    // Comparison and select are what the committee-tally demo circuit is built
    // from, so their cost is what actually sizes that demo.
    let (gt_time, gt) = timed(|| a.gt(&b));
    println!("{:<34} {:>12}", "gt (compare)", format_duration(gt_time));
    report_size("FheBool (compare result)", &gt);

    // WASM's `select` opcode maps onto tfhe's `if_then_else` over an FheBool.
    let (select_time, _) = timed(|| gt.if_then_else(&a, &b));
    println!("{:<34} {:>12}", "select", format_duration(select_time));

    // --- results at the boundary ---------------------------------------------
    // A freshly encrypted value compresses to a seed plus very little; a value
    // that has been through the evaluator has no seed to replay, so it
    // compresses far less well. The result blob is what `fulfillJob` carries,
    // so this is the number the on-chain gas estimate depends on.
    println!("\n== result at the boundary ==");
    let (compress_result_time, compressed_result) = timed(|| sum.compress());
    report_size("computed result (compressed)", &compressed_result);
    report_size("fresh input (compressed)", &compressed);
    println!(
        "{:<34} {:>12}",
        "result compression",
        format_duration(compress_result_time)
    );

    // --- sanity check --------------------------------------------------------
    // A probe that silently measured wrong arithmetic would be worse than none.
    let sum_plain: i32 = sum.decrypt(&client_key);
    let gt_plain: bool = gt.decrypt(&client_key);
    assert_eq!(sum_plain, 18, "7 + 11 should decrypt to 18");
    assert!(!gt_plain, "7 > 11 should decrypt to false");
    println!("\ncorrectness check: ok (7+11=18, 7>11=false)");

    if !measure_public_key {
        println!("\nre-run with --public-key to measure the public-key-encryption key.");
    }
}

/// Reports the serialized size of one artifact, which is what actually matters
/// for calldata and transport — not its in-memory footprint.
fn report_size<T>(label: &str, value: &T)
where
    T: serde::Serialize + tfhe::Versionize + tfhe::named::Named,
{
    match safe_serialized_size(value) {
        Ok(bytes) => println!("{:<34} {:>12}", label, format_bytes(bytes)),
        Err(e) => println!("{label:<34} {:>12}", format!("error: {e}")),
    }
}

fn timed<T>(f: impl FnOnce() -> T) -> (Duration, T) {
    let start = Instant::now();
    let out = f();
    (start.elapsed(), out)
}

fn format_bytes(bytes: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;
    let b = bytes as f64;
    if b >= GB {
        format!("{:.2} GB", b / GB)
    } else if b >= MB {
        format!("{:.1} MB", b / MB)
    } else if b >= KB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}

fn format_duration(d: Duration) -> String {
    let secs = d.as_secs_f64();
    if secs >= 1.0 {
        format!("{secs:.2} s")
    } else {
        format!("{:.1} ms", secs * 1000.0)
    }
}

fn profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}
