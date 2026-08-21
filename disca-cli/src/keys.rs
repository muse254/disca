//! Everything that touches the client key: generating it, encrypting inputs
//! under it, and decrypting a settled result with it.
//!
//! These three are one module because they are one trust boundary. Nothing
//! outside this file loads a client key, and the only values that leave it are
//! ones `architecture.md` §2 says are safe in the open: a compressed server
//! key, compressed ciphertexts, their commitments, and — at the very end of the
//! job — the plaintext the key holder asked for in the first place.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use primitives::bytecode::hex;
use primitives::wire;
use tfhe::prelude::FheDecrypt;
use tfhe::{ClientKey, CompressedServerKey, ConfigBuilder, set_server_key};

/// Where `keygen` puts the secret. Named as a constant because the node side
/// and `run-local.sh` both have to know it.
pub const CLIENT_KEY_FILE: &str = "client.key";

/// Where `keygen` puts the key workers fetch by hash (`bridge.md` §8 step 2).
pub const SERVER_KEY_FILE: &str = "server.key";

/// What a keypair generation produced. The client key is deliberately *not*
/// here: it is on disk and nothing needs it in memory afterwards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedKeys {
    pub client_key_path: PathBuf,
    pub server_key_path: PathBuf,
    /// `keccak256(server.key)` — the `serverKeyHash` `registerProgram` pins,
    /// and the name under which workers fetch the key.
    pub server_key_hash: [u8; 32],
}

impl GeneratedKeys {
    /// The stdout contract, kept next to the type that produces it so a change
    /// to one is visibly a change to the other. A caller registering a program
    /// greps for this line.
    pub fn write_report(&self, out: &mut impl Write) -> io::Result<()> {
        writeln!(out, "server_key_hash={}", hex(&self.server_key_hash))
    }
}

/// One encrypted input, and the commitment that pins it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptedInput {
    pub path: PathBuf,
    /// `keccak256` of the encoded ciphertext — an entry in the `inputCommits`
    /// array `submitJob` carries (`bridge.md` §2), which is what stops anyone
    /// substituting an input after the job is on chain.
    pub commitment: [u8; 32],
}

impl EncryptedInput {
    /// One line per input, in argument order, because `inputCommits` is an
    /// ordered array and a commitment matched to the wrong slot is worse than
    /// no commitment at all.
    pub fn write_report(&self, out: &mut impl Write) -> io::Result<()> {
        writeln!(out, "commitment={}", hex(&self.commitment))
    }
}

/// Generates a keypair into `out_dir`, writing `client.key` and `server.key`.
///
/// Refuses to replace an existing `client.key` without `force`. That is not
/// politeness: the client key is the only thing in the system that can decrypt
/// a result, results are computed asynchronously, and a job submitted this
/// morning is still in flight this afternoon. Silently overwriting the key is
/// the same event as losing the answer, and unlike most mistakes here it cannot
/// be undone by re-running anything.
pub fn keygen(out_dir: &Path, force: bool) -> Result<GeneratedKeys> {
    let client_key_path = out_dir.join(CLIENT_KEY_FILE);
    let server_key_path = out_dir.join(SERVER_KEY_FILE);

    // Checked before the directory is created and long before any key material
    // exists, so a refused `keygen` changes nothing at all.
    if client_key_path.exists() && !force {
        bail!(
            "{} already exists. Overwriting it destroys the only key that can \
             decrypt results computed under it, including any job still in \
             flight. Pass --force if that is what you mean.",
            client_key_path.display()
        );
    }

    fs::create_dir_all(out_dir)
        .with_context(|| format!("cannot create directory {}", out_dir.display()))?;

    // `ClientKey::generate` rather than `generate_keys`: the uncompressed
    // server key is 114.8 MB (`architecture.md` §2) and nothing in this process
    // evaluates anything, so generating one would be tens of seconds spent on a
    // value that is immediately dropped. What workers fetch is the compressed
    // form, and `CompressedServerKey::new` derives that from the client key
    // directly.
    let client_key = ClientKey::generate(ConfigBuilder::default().build());
    let server_key = CompressedServerKey::new(&client_key);

    let client_bytes = wire::encode_client_key(&client_key).with_context(|| {
        format!(
            "failed to encode the client key for {}",
            client_key_path.display()
        )
    })?;
    let server_bytes = wire::encode_server_key(&server_key).with_context(|| {
        format!(
            "failed to encode the server key for {}",
            server_key_path.display()
        )
    })?;

    // Secret first. If writing it fails there is nothing on disk to explain.
    write_client_key(&client_key_path, &client_bytes)?;
    fs::write(&server_key_path, &server_bytes)
        .with_context(|| format!("cannot write {}", server_key_path.display()))?;

    Ok(GeneratedKeys {
        client_key_path,
        server_key_path,
        server_key_hash: wire::commitment(&server_bytes),
    })
}

/// Encrypts `values` into `out_dir` as `input-0.ct`, `input-1.ct`, … in
/// argument order, returning each file's commitment in that same order.
///
/// Argument order is load-bearing twice over: it is the order the circuit reads
/// its parameters in, and it is the order of the `inputCommits` array. The
/// returned vector preserves it, and so does the file numbering, so a caller
/// can pair the two without keeping a separate index.
pub fn encrypt(
    client_key_path: &Path,
    values: &[i32],
    out_dir: &Path,
) -> Result<Vec<EncryptedInput>> {
    let client_key = load_client_key(client_key_path)?;

    fs::create_dir_all(out_dir)
        .with_context(|| format!("cannot create directory {}", out_dir.display()))?;

    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            // Errors here name the position, never the value. A message that
            // quoted the plaintext would put it in a terminal scrollback and a
            // CI log, which is the one place this whole design exists to keep
            // it out of.
            let compressed = wire::encrypt_input(*value, &client_key)
                .with_context(|| format!("failed to encrypt input {index}"))?;
            let bytes = wire::encode(&compressed)
                .with_context(|| format!("failed to encode input {index}"))?;

            let path = out_dir.join(format!("input-{index}.ct"));
            fs::write(&path, &bytes).with_context(|| format!("cannot write {}", path.display()))?;

            Ok(EncryptedInput {
                commitment: wire::commitment(&bytes),
                path,
            })
        })
        .collect()
}

/// Decrypts a result blob and returns the plaintext.
///
/// This needs the *server* key as well, which reads like a mistake and is not:
/// the blob crossing the boundary is compressed, and expanding a compressed
/// ciphertext is a server-key operation even though decrypting the expanded one
/// is not. `KeyHolder::new` in `node/src/coordinator.rs` calls `set_server_key`
/// for precisely this reason. Nothing is conceded by loading it — the server
/// key is public, and this party is the one that generated it.
pub fn decrypt(client_key_path: &Path, server_key_path: &Path, input: &Path) -> Result<i32> {
    let client_key = load_client_key(client_key_path)?;

    let server_bytes = fs::read(server_key_path)
        .with_context(|| format!("cannot read {}", server_key_path.display()))?;
    let server_key = wire::decode_server_key(&server_bytes)
        .with_context(|| format!("{} is not a DISCA server key", server_key_path.display()))?;
    set_server_key(server_key);

    let blob = fs::read(input).with_context(|| format!("cannot read {}", input.display()))?;
    let ciphertext = wire::decode(&blob)
        .with_context(|| format!("{} is not a DISCA ciphertext", input.display()))?;

    Ok(wire::decompress(&ciphertext).decrypt(&client_key))
}

/// Loads the client key, naming the file in every way this can fail.
///
/// The two failures are worth telling apart: a missing file is usually the
/// wrong `--client-key` path, while bytes that will not decode are usually the
/// *server* key handed to the wrong flag.
fn load_client_key(path: &Path) -> Result<ClientKey> {
    let bytes = fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    wire::decode_client_key(&bytes)
        .with_context(|| format!("{} is not a DISCA client key", path.display()))
}

/// Writes the client key owner-only where the platform has a notion of one.
///
/// This is not protection — `wire::encode_client_key` says plainly that nothing
/// guards the key at rest, and a mode bit does not change that. It removes one
/// specific accident: a keypair generated into a shared or group-readable
/// directory being world-readable by default.
fn write_client_key(path: &Path, bytes: &[u8]) -> Result<()> {
    // The mode below applies only when the file is created, so overwriting
    // would silently inherit whatever mode the previous key had. Remove first;
    // `--force` has already sanctioned losing it.
    if path.exists() {
        fs::remove_file(path).with_context(|| format!("cannot replace {}", path.display()))?;
    }

    let mut options = OpenOptions::new();
    options.write(true).create_new(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // Set at creation rather than chmod'd afterwards: a world-readable
        // window, however short, is a window.
        options.mode(0o600);
    }

    let mut file = options
        .open(path)
        .with_context(|| format!("cannot write {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("cannot write {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testing::TempDir;

    #[test]
    fn a_generated_keypair_lands_where_it_says_it_did() {
        let dir = TempDir::new("keygen");
        let keys = keygen(dir.path(), false).expect("keygen");

        assert_eq!(keys.client_key_path, dir.path().join(CLIENT_KEY_FILE));
        assert_eq!(keys.server_key_path, dir.path().join(SERVER_KEY_FILE));
        assert!(keys.client_key_path.exists());
        assert!(keys.server_key_path.exists());

        // The reported hash is what a worker will check the bytes it fetched
        // against, so it has to be the hash of those bytes and not of anything
        // else that happened to be in scope.
        let server_bytes = fs::read(&keys.server_key_path).unwrap();
        assert_eq!(wire::commitment(&server_bytes), keys.server_key_hash);

        let mut report = Vec::new();
        keys.write_report(&mut report).unwrap();
        assert_eq!(
            String::from_utf8(report).unwrap(),
            format!("server_key_hash={}\n", hex(&keys.server_key_hash))
        );
    }

    #[test]
    fn keygen_refuses_to_destroy_an_existing_client_key() {
        let dir = TempDir::new("keygen-refuse");
        fs::create_dir_all(dir.path()).unwrap();
        let client_key_path = dir.path().join(CLIENT_KEY_FILE);
        fs::write(&client_key_path, b"pretend this decrypts a job in flight").unwrap();

        let err = keygen(dir.path(), false).expect_err("must refuse");
        assert!(
            err.to_string()
                .contains(&client_key_path.display().to_string()),
            "the error must name the file it refused to touch: {err}"
        );

        // Refusing is only worth anything if it happens before the write.
        assert_eq!(
            fs::read(&client_key_path).unwrap(),
            b"pretend this decrypts a job in flight"
        );
        assert!(
            !dir.path().join(SERVER_KEY_FILE).exists(),
            "a refused keygen must leave nothing behind"
        );
    }

    #[test]
    fn a_key_written_by_one_run_decrypts_what_another_run_encrypted() {
        // The property that makes a multi-invocation key holder possible at
        // all. Nothing is held in memory between the two calls below: `encrypt`
        // and `decrypt` each load the key off disk, exactly as two separate
        // `disca-cli` invocations would.
        let dir = TempDir::new("round-trip");
        let keys = keygen(dir.path(), false).expect("keygen");

        let inputs = encrypt(
            &keys.client_key_path,
            &[71, -93, 0, i32::MIN],
            &dir.path().join("inputs"),
        )
        .expect("encrypt");

        assert_eq!(inputs.len(), 4);
        for (index, input) in inputs.iter().enumerate() {
            assert_eq!(
                input.path,
                dir.path().join("inputs").join(format!("input-{index}.ct"))
            );
            let bytes = fs::read(&input.path).unwrap();
            assert_eq!(wire::commitment(&bytes), input.commitment);
        }

        // Order is the contract: the commitments are an ordered array in
        // `submitJob` and the files are the circuit's parameters in order, so a
        // value recovered from slot 1 has to be the value that went into it.
        let recovered: Vec<i32> = inputs
            .iter()
            .map(|input| {
                decrypt(&keys.client_key_path, &keys.server_key_path, &input.path).expect("decrypt")
            })
            .collect();
        assert_eq!(recovered, vec![71, -93, 0, i32::MIN]);

        let mut report = Vec::new();
        for input in &inputs {
            input.write_report(&mut report).unwrap();
        }
        assert_eq!(
            String::from_utf8(report).unwrap(),
            inputs
                .iter()
                .map(|input| format!("commitment={}\n", hex(&input.commitment)))
                .collect::<String>()
        );
    }

    #[test]
    fn distinct_values_do_not_share_a_commitment() {
        let dir = TempDir::new("commitments");
        let keys = keygen(dir.path(), false).expect("keygen");

        let inputs = encrypt(&keys.client_key_path, &[7, 7], dir.path()).expect("encrypt");

        // Encryption is randomised, so even the *same* value twice must produce
        // two different commitments. If it did not, `inputCommits` would leak
        // which of a job's inputs are equal to each other.
        assert_ne!(inputs[0].commitment, inputs[1].commitment);
    }

    #[test]
    fn errors_name_the_path_that_failed() {
        let dir = TempDir::new("missing");
        let absent = dir.path().join("nowhere").join(CLIENT_KEY_FILE);

        let err = encrypt(&absent, &[1], dir.path()).expect_err("no such key");
        assert!(
            format!("{err:#}").contains(&absent.display().to_string()),
            "got: {err:#}"
        );
    }

    #[test]
    fn a_server_key_handed_to_the_client_key_flag_is_named_as_such() {
        // The likeliest operator mistake in a four-flag CLI, and the one whose
        // default error ("invalid header") tells you nothing about which file
        // to look at.
        let dir = TempDir::new("swapped");
        let keys = keygen(dir.path(), false).expect("keygen");

        let err = encrypt(&keys.server_key_path, &[1], dir.path()).expect_err("wrong key");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains(&keys.server_key_path.display().to_string())
                && rendered.contains("not a DISCA client key"),
            "got: {rendered}"
        );
    }
}
