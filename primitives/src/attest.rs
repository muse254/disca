//! Per-worker signed attestations: *who* says a result is right, provably.
//!
//! [`crate::wire`] answers "what was computed" — `keccak256` of the compressed
//! result. This module answers the other half, "and who is standing behind it".
//! Until task 2.10i the answer was nobody: a report carried a bare result hash,
//! and the attester list that `fulfillJob` takes came from the coordinator
//! (`bridge.md` §2). A contract could check those addresses were distinct and
//! registered; it could not check they had ever computed anything. A dishonest
//! coordinator could name any M registered workers beside any result, and no
//! party — including the key holder, who cannot tell a wrong plaintext from a
//! right one — would have anything to contradict it with.
//!
//! A worker therefore signs. Three properties are deliberate:
//!
//! * **Ethereum identity.** A worker is the last 20 bytes of
//!   `keccak256(uncompressed public key)`, exactly as an EOA is, so the address
//!   a worker attests under is the address the on-chain registry holds. No
//!   mapping table, nothing to keep in sync.
//! * **Recoverable signatures.** `(r, s, v)`, so `ecrecover` yields the signer
//!   from the signature alone. The alternative — carry each attester's 64-byte
//!   public key in calldata and verify against it — costs bytes on-chain for a
//!   value the signature already contains.
//! * **Domain-separated claims.** A signature is over a claim about a specific
//!   job, program and result, never over a bare `resultHash`. See
//!   [`Claim::preimage`].
//!
//! # What this does and does not buy
//!
//! It makes attestation *attributable*: an attestation names an author, and no
//! one but the holder of that key can produce one. It does not make agreement
//! mean more than it did — `architecture.md` §3 still governs that, and a
//! signature over a diverged result is a signed wrong answer, which is exactly
//! what M-of-N is for. Nor does it stop a coordinator withholding a valid
//! attestation; that is a liveness failure and the escrow refund path
//! (`bridge.md` §6) covers it.
//!
//! This closes `architecture.md` §11 Q3 the other way from its stated leaning:
//! the address list was chosen for being "simpler and cheaper", but the thing
//! it was cheaper *than* is the thing that makes the list mean anything.

use std::fmt;

use k256::ecdsa::{RecoveryId, Signature, SigningKey, VerifyingKey};
use sha3::{Digest, Keccak256};
use wincode::{SchemaRead, SchemaWrite};

use crate::program::ProgramError;

type Result<T> = std::result::Result<T, ProgramError>;

/// An Ethereum-style address: the last 20 bytes of `keccak256` over the
/// uncompressed public key with its `0x04` SEC1 tag stripped.
pub type Address = [u8; 20];

/// Domain tag for "I evaluated this job and got this result".
///
/// This is the only claim a contract ever sees.
const DOMAIN_RESULT: &[u8; 22] = b"DISCA/attest/result/v1";

/// Domain tag for "I could not evaluate this job, and here is why".
///
/// Deliberately the same length as [`DOMAIN_RESULT`] and differing only in the
/// middle word: the two preimages are then the same shape, so the reason the
/// one can never be read as the other is the tag itself rather than an accident
/// of layout.
const DOMAIN_FAILURE: &[u8; 22] = b"DISCA/attest/failed/v1";

/// EIP-191 version `0x45` ("personal sign") prefix, for a 32-byte payload.
const EIP191_PREFIX: &[u8; 28] = b"\x19Ethereum Signed Message:\n32";

/// Prefix under which a deterministic development key is derived from a worker
/// id. Distinct from the claim tags so that a derived key can never be the hash
/// of something a worker also signs.
const DEV_KEY_DOMAIN: &[u8; 16] = b"DISCA/dev-key/v1";

/// What a worker is putting its name to.
///
/// A claim is the *whole* statement, not just its conclusion. Signing only the
/// result hash would let a signature be lifted off one job and presented as an
/// attestation for another that happened to produce the same bytes — and with a
/// deterministic evaluator over a small result space, that is not a remote
/// coincidence, it is the common case for any two jobs with the same answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Claim {
    /// "Running program `bytecode_hash` for job `job_id` produced a result
    /// sealing to `result_hash`."
    Result {
        job_id: u64,
        bytecode_hash: [u8; 32],
        /// `SealedResult::hash` — `keccak256` of the *compressed* result, the
        /// same bytes `fulfillJob` emits (`bridge.md` §5a).
        result_hash: [u8; 32],
    },
    /// "I could not run program `bytecode_hash` for job `job_id`."
    ///
    /// A failure attests to nothing and is never counted towards a quorum. It
    /// is signed anyway so that a report blaming a worker is one that worker
    /// actually wrote: unsigned failures let anyone who can reach the
    /// coordinator manufacture evidence against an honest operator, which
    /// matters the moment registration or reputation depends on it.
    Failure {
        job_id: u64,
        bytecode_hash: [u8; 32],
        /// `keccak256(reason)`. The reason string itself is unbounded and
        /// operator-controlled; its hash is fixed-width, which is what keeps
        /// the preimage layout below rigid.
        reason_hash: [u8; 32],
    },
}

/// Total preimage length: 22-byte tag + 8-byte job id + two 32-byte hashes.
const PREIMAGE_LEN: usize = 22 + 8 + 32 + 32;

impl Claim {
    /// The exact bytes hashed into the digest a worker signs.
    ///
    /// **A Solidity contract has to reconstruct this byte for byte**, so the
    /// layout is fixed here and versioned in the tag:
    ///
    /// ```text
    /// offset  len  field
    ///      0   22  domain tag, ASCII, no length prefix
    ///              "DISCA/attest/result/v1"  for Claim::Result
    ///              "DISCA/attest/failed/v1"  for Claim::Failure
    ///     22    8  job id, big-endian u64
    ///     30   32  keccak256(bytecode)      — which program
    ///     62   32  keccak256(result blob)   — what it produced
    ///              (Claim::Failure puts keccak256(reason) here instead)
    ///     94       total
    /// ```
    ///
    /// Every field is fixed-width, so concatenation is injective: no two
    /// distinct claims share a preimage, and no length-prefix or delimiter
    /// scheme is needed to make that true. In Solidity that is
    ///
    /// ```solidity
    /// keccak256(abi.encodePacked(
    ///     "DISCA/attest/result/v1", uint64(jobId), bytecodeHash, resultHash));
    /// ```
    ///
    /// since `abi.encodePacked` writes a string literal without a length prefix
    /// and a `uint64` as 8 big-endian bytes.
    ///
    /// **What is bound, and what is not.** The job id, the program identity and
    /// the result. Not the chain id and not the bridge address, because neither
    /// exists yet — see [`Claim::digest`] for what that costs.
    pub fn preimage(&self) -> [u8; PREIMAGE_LEN] {
        let (tag, job_id, bytecode_hash, tail) = match self {
            Claim::Result {
                job_id,
                bytecode_hash,
                result_hash,
            } => (DOMAIN_RESULT, job_id, bytecode_hash, result_hash),
            Claim::Failure {
                job_id,
                bytecode_hash,
                reason_hash,
            } => (DOMAIN_FAILURE, job_id, bytecode_hash, reason_hash),
        };

        let mut out = [0u8; PREIMAGE_LEN];
        out[..22].copy_from_slice(tag);
        out[22..30].copy_from_slice(&job_id.to_be_bytes());
        out[30..62].copy_from_slice(bytecode_hash);
        out[62..].copy_from_slice(tail);
        out
    }

    /// The 32-byte digest actually passed to secp256k1.
    ///
    /// `keccak256(EIP-191 prefix || keccak256(preimage))`, i.e. the EIP-191
    /// version `0x45` ("personal sign") construction over our own domain hash.
    ///
    /// **Why prefix at all**, given the tag inside the preimage already
    /// separates DISCA claims from each other: the tag separates claims within
    /// DISCA, the prefix separates DISCA from everything else the same key
    /// could be asked to sign. A raw ECDSA signature is over an opaque 32-byte
    /// value, so without the prefix a signature over an attestation and a
    /// signature over a transaction hash are the same kind of object — and a
    /// worker key *is* an Ethereum key, which is the point of §11 Q3. `0x19` is
    /// not a legal leading byte for an RLP-encoded transaction, so prefixing
    /// makes an attestation structurally incapable of being replayed as one.
    /// It also means an operator can keep the key anywhere that exposes
    /// `personal_sign` — a wallet, an HSM, a cloud KMS — instead of needing an
    /// interface that signs raw digests, which many deliberately refuse to do.
    ///
    /// **Why not EIP-712**, which is the better answer: it binds a chain id and
    /// a verifying contract address, and that is what stops an attestation
    /// minted against one `DiscaBridge` deployment being replayed against
    /// another, or against the same contract on a forked chain. Neither value
    /// exists yet — there is no deployed bridge and no chain (`bridge.md` §8
    /// step 1). Inventing placeholders now would produce signatures that have
    /// to be reinterpreted later, which is precisely the silent breakage the
    /// version suffix is here to prevent. When the contract lands, this becomes
    /// `DISCA/attest/result/v2` under a 712 domain, and the version bump makes
    /// the change loud.
    ///
    /// The cost of the prefix on-chain is one extra `keccak256` over 60 bytes
    /// (OpenZeppelin's `MessageHashUtils.toEthSignedMessageHash`), which is
    /// noise beside the 11.8 KB result blob `fulfillJob` already carries.
    pub fn digest(&self) -> [u8; 32] {
        let inner = keccak(&self.preimage());

        let mut hasher = Keccak256::new();
        hasher.update(EIP191_PREFIX);
        hasher.update(inner);
        hasher.finalize().into()
    }
}

/// A recoverable secp256k1 signature over a [`Claim`].
///
/// Split into `(r, s, v)` rather than carried as 65 packed bytes because that
/// is the shape `ecrecover(hash, v, r, s)` takes, and a coordinator relaying an
/// attestation on-chain should not have to reinterpret it on the way.
#[derive(Debug, Clone, PartialEq, Eq, SchemaWrite, SchemaRead)]
pub struct Attestation {
    pub r: [u8; 32],
    pub s: [u8; 32],
    /// Recovery id in Ethereum's encoding, 27 or 28 — *not* the bare 0/1 that
    /// `k256` uses. Stored the way the EVM wants it so that what a coordinator
    /// forwards is what `ecrecover` consumes, unmodified.
    pub v: u8,
}

/// Ethereum's offset for the recovery id, from the original Bitcoin message
/// signing convention. `ecrecover` rejects anything outside {27, 28}.
const V_OFFSET: u8 = 27;

/// A worker's signing key, and the address it attests under.
///
/// Has a hand-written [`fmt::Debug`] that prints the address only. The
/// derived one would print the key material, and the codebase logs structs with
/// `?` in several places — a private key reaching a log line is not a bug you
/// find by reading the diff, so make it unrepresentable instead.
pub struct WorkerKey {
    signing: SigningKey,
    address: Address,
}

impl fmt::Debug for WorkerKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Deliberately does not include `signing`.
        f.debug_struct("WorkerKey")
            .field("address", &hex_address(&self.address))
            .finish_non_exhaustive()
    }
}

impl WorkerKey {
    /// Loads a key from hex, with or without a `0x` prefix.
    ///
    /// The error text never contains the input. A malformed key is usually a
    /// mistyped *real* key, and an error message is the one place a secret gets
    /// copied into a bug report.
    pub fn from_hex(hex: &str) -> Result<Self> {
        let bytes = decode_hex(hex, 32)
            .ok_or_else(|| ProgramError("private key must be 32 hex-encoded bytes".into()))?;

        // Rejects zero and anything at or above the curve order; both are
        // outside the scalar field and neither can sign.
        let signing = SigningKey::from_slice(&bytes)
            .map_err(|_| ProgramError("private key is not a valid secp256k1 scalar".into()))?;

        Ok(Self::from_signing_key(signing))
    }

    /// Derives a deterministic key from a label, for local runs and tests.
    ///
    /// **This is not a secret.** The label is a worker id that appears in every
    /// log line, so anyone can recompute the key. It exists so
    /// `scripts/run-local.sh` can stand up three workers and a coordinator that
    /// already knows their addresses, without generating and plumbing key
    /// material through a shell script. A deployment passes `--key`; a worker
    /// that falls back to this says so, loudly, at startup.
    ///
    /// Rehashes on the ~2^-128 chance that the digest is not a valid scalar,
    /// rather than returning a `Result` nobody could act on.
    pub fn derive(label: &str) -> Self {
        let mut hasher = Keccak256::new();
        hasher.update(DEV_KEY_DOMAIN);
        hasher.update(label.as_bytes());
        let mut seed: [u8; 32] = hasher.finalize().into();

        loop {
            match SigningKey::from_slice(&seed) {
                Ok(signing) => return Self::from_signing_key(signing),
                Err(_) => seed = keccak(&seed),
            }
        }
    }

    fn from_signing_key(signing: SigningKey) -> Self {
        let address = address_of(signing.verifying_key());
        Self { signing, address }
    }

    /// The address this worker attests under, and the one an on-chain registry
    /// must hold for its attestations to count.
    pub fn address(&self) -> Address {
        self.address
    }

    /// Signs a claim.
    ///
    /// Infallible in practice and so not a `Result`: RFC 6979 derives the nonce
    /// deterministically from the key and the digest, so there is no entropy
    /// source to fail, and the only documented errors are a prehash shorter
    /// than the field (ours is exactly 32 bytes) and a missing recovery id
    /// (which secp256k1 always produces). A `Result` here would be an
    /// unreachable branch propagated through every caller.
    ///
    /// Determinism is a small bonus: two runs of the same worker over the same
    /// claim produce identical bytes, so a duplicate report is byte-identical
    /// rather than merely equivalent.
    pub fn attest(&self, claim: &Claim) -> Attestation {
        let (signature, recovery_id) = self
            .signing
            .sign_prehash_recoverable(&claim.digest())
            .expect("RFC 6979 signing of a 32-byte prehash cannot fail");

        let (r, s) = signature.split_bytes();
        Attestation {
            r: r.into(),
            s: s.into(),
            v: recovery_id.to_byte() + V_OFFSET,
        }
    }
}

/// Recovers the address that signed `claim`, or says why it could not.
///
/// This is the whole point of the exercise: the address is *derived from the
/// signature*, never supplied alongside it. There is no input a caller can vary
/// to change who the signer turns out to be — an attacker who wants a
/// particular address out of this function needs that address's private key.
///
/// A caller still has to check the recovered address is one it will accept:
/// recovery succeeds for essentially any well-formed `(r, s, v)` and simply
/// yields whatever address that implies. Recovery answers "who", the registry
/// answers "does who count".
pub fn recover(claim: &Claim, attestation: &Attestation) -> Result<Address> {
    // Only 27 and 28 — deliberately narrower than `RecoveryId::from_byte`,
    // which also accepts 2 and 3 for the vanishingly rare case where the
    // signature's x coordinate wrapped the curve order. `ecrecover` refuses
    // those, so accepting them here would mean a coordinator could count an
    // attestation on-chain verification would later throw out. Better to agree
    // with the contract than to be slightly more general than it.
    let recovery_id = match attestation.v {
        27 | 28 => RecoveryId::from_byte(attestation.v - V_OFFSET).expect("0 and 1 are valid"),
        other => {
            return Err(ProgramError(format!(
                "recovery id {other} is not one of Ethereum's 27 or 28"
            )));
        }
    };

    let signature = Signature::from_scalars(attestation.r, attestation.s)
        .map_err(|_| ProgramError("signature scalars are outside the curve order".into()))?;

    // Reject the malleated twin, (r, n - s, v ^ 1), which is a second valid
    // signature over the same claim by the same key (EIP-2 forbids it on-chain
    // for the same reason). It cannot inflate a quorum here — reports are
    // counted by recovered address, and both twins recover to one address — but
    // it would let a relay alter an attestation in flight without invalidating
    // it, and an attestation nobody can quietly rewrite is worth more than one
    // that merely verifies.
    if signature.normalize_s().is_some() {
        return Err(ProgramError(
            "signature has a high s; EIP-2 requires the low-s form".into(),
        ));
    }

    let digest = claim.digest();
    // There is deliberately no separate "verify" step after this. Recovery
    // *is* the verification: the key it reconstructs is by construction one the
    // signature verifies under, so calling `verify_prehash` on the result would
    // always pass and would be a scalar multiplication spent proving nothing.
    // What makes recovery meaningful is not that it can fail but that it is a
    // function — `(claim, r, s, v)` determines exactly one address, and only
    // the holder of a given private key can choose the inputs that land on that
    // key's address. Change the claim, or `v`, or a byte of the signature, and
    // you get *some* address, just not the one you wanted; the registry check
    // in the caller is what turns "some address" into a rejection.
    let key = VerifyingKey::recover_from_prehash(&digest, &signature, recovery_id)
        .map_err(|_| ProgramError("signature does not recover to a public key".into()))?;

    Ok(address_of(&key))
}

/// The Ethereum address of a public key: `keccak256(x || y)[12..]`.
///
/// The SEC1 `0x04` uncompressed tag is stripped first — it is a framing byte,
/// not part of the point, and including it would give addresses that no
/// Ethereum tool agrees with.
pub fn address_of(key: &VerifyingKey) -> Address {
    let point = key.to_encoded_point(false);
    let uncompressed = point.as_bytes();

    let digest = keccak(&uncompressed[1..]);
    let mut address = [0u8; 20];
    address.copy_from_slice(&digest[12..]);
    address
}

/// Renders an address as the lowercase `0x` hex a contract and a log both read.
///
/// No EIP-55 checksum casing: these are compared byte-for-byte against a
/// registry, never retyped by a human, and mixed-case hex in a log invites
/// exactly the case-sensitive comparison bug the checksum was meant to catch.
pub fn hex_address(address: &Address) -> String {
    let mut out = String::with_capacity(2 + 40);
    out.push_str("0x");
    for byte in address {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Parses an address from `0x`-prefixed or bare hex.
pub fn parse_address(text: &str) -> Result<Address> {
    let bytes = decode_hex(text, 20)
        .ok_or_else(|| ProgramError(format!("{text} is not a 20-byte hex address")))?;

    let mut address = [0u8; 20];
    address.copy_from_slice(&bytes);
    Ok(address)
}

/// `keccak256`, the EVM hash. Re-derived here rather than reaching into
/// [`crate::wire::commitment`], which is about ciphertext bytes and means
/// something else.
fn keccak(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Keccak256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// Decodes exactly `len` bytes of hex, tolerating a `0x` prefix. `None` on any
/// deviation — never a partial decode, and never an error carrying the input.
fn decode_hex(text: &str, len: usize) -> Option<Vec<u8>> {
    let body = text.strip_prefix("0x").unwrap_or(text);
    if body.len() != len * 2 {
        return None;
    }

    (0..len)
        .map(|index| u8::from_str_radix(&body[index * 2..index * 2 + 2], 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result_claim() -> Claim {
        Claim::Result {
            job_id: 1,
            bytecode_hash: [0xaa; 32],
            result_hash: [0xbb; 32],
        }
    }

    #[test]
    fn a_valid_attestation_recovers_the_signers_address() {
        let key = WorkerKey::derive("worker-1");
        let claim = result_claim();

        let attestation = key.attest(&claim);

        assert_eq!(
            recover(&claim, &attestation).unwrap(),
            key.address(),
            "the address must fall out of the signature, not be taken on trust"
        );
    }

    #[test]
    fn two_workers_attesting_to_one_claim_recover_to_two_addresses() {
        // The property M-of-N counting rests on: agreement is only evidence if
        // the agreeing parties are distinguishable.
        let claim = result_claim();
        let one = WorkerKey::derive("worker-1");
        let two = WorkerKey::derive("worker-2");

        assert_ne!(one.address(), two.address());
        assert_eq!(recover(&claim, &one.attest(&claim)).unwrap(), one.address());
        assert_eq!(recover(&claim, &two.attest(&claim)).unwrap(), two.address());
    }

    #[test]
    fn an_attestation_does_not_carry_over_to_another_job() {
        // The replay this exists to stop: without the job id in the preimage, a
        // signature harvested from any settled job could be presented as an
        // attestation for a different one. Recovery still succeeds — ECDSA
        // always yields *some* key — so the failure has to be that it yields
        // the wrong address, which is what a registry check then rejects.
        let key = WorkerKey::derive("worker-1");
        let attestation = key.attest(&result_claim());

        let other_job = Claim::Result {
            job_id: 2,
            bytecode_hash: [0xaa; 32],
            result_hash: [0xbb; 32],
        };

        assert_ne!(
            recover(&other_job, &attestation).unwrap_or_default(),
            key.address(),
            "a signature for job 1 must not attribute to this worker under job 2"
        );
    }

    #[test]
    fn an_attestation_does_not_carry_over_to_another_program() {
        // Same job id, same answer, different circuit. Two programs can easily
        // produce the same result bytes; binding the bytecode hash is what
        // stops an attestation for a trivial circuit being presented as one for
        // the circuit the job actually paid for.
        let key = WorkerKey::derive("worker-1");
        let attestation = key.attest(&result_claim());

        let other_program = Claim::Result {
            job_id: 1,
            bytecode_hash: [0xcc; 32],
            result_hash: [0xbb; 32],
        };

        assert_ne!(
            recover(&other_program, &attestation).unwrap_or_default(),
            key.address()
        );
    }

    #[test]
    fn an_attestation_does_not_carry_over_to_another_result() {
        // The one that matters most: an attestation must not survive having the
        // result swapped under it.
        let key = WorkerKey::derive("worker-1");
        let attestation = key.attest(&result_claim());

        let other_result = Claim::Result {
            job_id: 1,
            bytecode_hash: [0xaa; 32],
            result_hash: [0xdd; 32],
        };

        assert_ne!(
            recover(&other_result, &attestation).unwrap_or_default(),
            key.address()
        );
    }

    #[test]
    fn a_failure_signature_cannot_be_read_as_a_result_attestation() {
        // The two claims share every field but the domain tag. If the tag were
        // dropped, a worker's "I could not evaluate, here is the reason hash"
        // would be a valid attestation to a result equal to that reason hash.
        let key = WorkerKey::derive("worker-1");

        let failure = Claim::Failure {
            job_id: 1,
            bytecode_hash: [0xaa; 32],
            reason_hash: [0xbb; 32],
        };
        let attestation = key.attest(&failure);

        assert_eq!(recover(&failure, &attestation).unwrap(), key.address());
        assert_ne!(
            recover(&result_claim(), &attestation).unwrap_or_default(),
            key.address(),
            "the domain tag must be what keeps these apart"
        );
    }

    #[test]
    fn the_preimage_layout_is_the_one_a_contract_will_reconstruct() {
        // Pins the byte layout documented on `Claim::preimage`. A Solidity
        // `abi.encodePacked` has to produce exactly these bytes, and if this
        // ever changes without the version tag changing, on-chain verification
        // fails in a way that looks like every worker signing badly.
        let claim = Claim::Result {
            job_id: 0x0102_0304_0506_0708,
            bytecode_hash: [0x11; 32],
            result_hash: [0x22; 32],
        };
        let preimage = claim.preimage();

        assert_eq!(preimage.len(), 94);
        assert_eq!(&preimage[..22], b"DISCA/attest/result/v1");
        assert_eq!(
            &preimage[22..30],
            &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08],
            "job id is big-endian, as abi.encodePacked writes a uint64"
        );
        assert_eq!(&preimage[30..62], &[0x11; 32]);
        assert_eq!(&preimage[62..], &[0x22; 32]);

        // The failure tag occupies the same span, so the two preimages differ
        // only where they are meant to.
        let failure = Claim::Failure {
            job_id: 0x0102_0304_0506_0708,
            bytecode_hash: [0x11; 32],
            reason_hash: [0x22; 32],
        };
        assert_eq!(&failure.preimage()[..22], b"DISCA/attest/failed/v1");
        assert_eq!(&failure.preimage()[22..], &preimage[22..]);
    }

    #[test]
    fn the_digest_is_the_eip_191_construction_over_the_domain_hash() {
        // Reproduces `digest` the long way round, so that the prefix cannot be
        // dropped or applied to the wrong operand without this failing. A
        // contract calling `MessageHashUtils.toEthSignedMessageHash(inner)`
        // must land on the same 32 bytes.
        let claim = result_claim();

        let inner = keccak(&claim.preimage());
        let mut hasher = Keccak256::new();
        hasher.update(b"\x19Ethereum Signed Message:\n32");
        hasher.update(inner);
        let expected: [u8; 32] = hasher.finalize().into();

        assert_eq!(claim.digest(), expected);
        assert_ne!(
            claim.digest(),
            inner,
            "the prefix must actually be applied, not merely present"
        );
    }

    #[test]
    fn a_tampered_recovery_id_does_not_yield_the_signers_address() {
        // `v` selects which of two candidate points `r` implies, and *both* are
        // keys the signature verifies under — which is exactly why `recover`
        // does not bother verifying afterwards. Flipping `v` succeeds and
        // returns the wrong address, and the wrong address is what gets
        // rejected. This pins that it really is the wrong one.
        let key = WorkerKey::derive("worker-1");
        let claim = result_claim();

        let mut attestation = key.attest(&claim);
        attestation.v = if attestation.v == 27 { 28 } else { 27 };

        // Either outcome is acceptable; what must not happen is recovering the
        // signer's own address from a signature they did not make with that v.
        if let Ok(address) = recover(&claim, &attestation) {
            assert_ne!(address, key.address());
        }
    }

    #[test]
    fn a_recovery_id_outside_ethereums_encoding_is_refused() {
        // 0 and 1 are what k256 uses internally; on the wire and in `ecrecover`
        // the only legal values are 27 and 28. Accepting the bare form would
        // mean a coordinator forwarding an attestation on-chain had to know to
        // adjust it, and forgetting is a silent wrong-address bug.
        let key = WorkerKey::derive("worker-1");
        let claim = result_claim();

        for v in [0u8, 1, 26, 29, 255] {
            let mut attestation = key.attest(&claim);
            attestation.v = v;
            let error = recover(&claim, &attestation).unwrap_err().to_string();
            assert!(error.contains("27 or 28"), "got: {error}");
        }
    }

    #[test]
    fn a_high_s_signature_is_refused() {
        // The malleated twin: s' = n - s with v flipped is a second valid
        // signature over the same claim. k256 emits the low form; this checks
        // we refuse the other one rather than quietly accepting a signature
        // somebody rewrote in flight.
        use k256::elliptic_curve::scalar::IsHigh;

        let key = WorkerKey::derive("worker-1");
        let claim = result_claim();
        let attestation = key.attest(&claim);

        let signature = Signature::from_scalars(attestation.r, attestation.s).unwrap();
        assert!(
            !bool::from(signature.s().is_high()),
            "k256 must be emitting the low-s form to begin with"
        );

        let flipped =
            Signature::from_scalars(signature.r().to_bytes(), (-signature.s()).to_bytes()).unwrap();
        let (r, s) = flipped.split_bytes();
        let malleated = Attestation {
            r: r.into(),
            s: s.into(),
            v: if attestation.v == 27 { 28 } else { 27 },
        };

        let error = recover(&claim, &malleated).unwrap_err().to_string();
        assert!(error.contains("high s"), "got: {error}");
    }

    #[test]
    fn a_signature_of_the_wrong_shape_is_refused_rather_than_recovered() {
        let claim = result_claim();

        // Zero scalars are not a signature; `from_scalars` must reject them
        // before anything tries to recover from them.
        let zeroed = Attestation {
            r: [0u8; 32],
            s: [0u8; 32],
            v: 27,
        };
        assert!(recover(&claim, &zeroed).is_err());

        // All-ones is above the curve order.
        let overflowing = Attestation {
            r: [0xffu8; 32],
            s: [0xffu8; 32],
            v: 27,
        };
        assert!(recover(&claim, &overflowing).is_err());
    }

    #[test]
    fn a_key_round_trips_through_hex_with_or_without_the_prefix() {
        let derived = WorkerKey::derive("worker-1");

        // `derive` is the only way to get key bytes out in a test; re-derive
        // the same scalar to build the hex form.
        let mut hasher = Keccak256::new();
        hasher.update(DEV_KEY_DOMAIN);
        hasher.update(b"worker-1");
        let seed: [u8; 32] = hasher.finalize().into();
        let hex: String = seed.iter().map(|byte| format!("{byte:02x}")).collect();

        assert_eq!(
            WorkerKey::from_hex(&hex).unwrap().address(),
            derived.address()
        );
        assert_eq!(
            WorkerKey::from_hex(&format!("0x{hex}")).unwrap().address(),
            derived.address()
        );
    }

    #[test]
    fn a_malformed_key_is_refused_without_the_input_appearing_in_the_error() {
        // A mistyped private key is still a private key. The error goes into
        // logs and bug reports; the key must not go with it.
        for bad in [
            "",
            "0x",
            "deadbeef",
            "0xzz00000000000000000000000000000000000000000000000000000000000000",
            // Exactly the curve order: a valid 32-byte hex string that is not a
            // valid scalar.
            "0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
            // Zero.
            "0x0000000000000000000000000000000000000000000000000000000000000000",
        ] {
            let error = WorkerKey::from_hex(bad).unwrap_err().to_string();
            assert!(
                !error.contains(bad) || bad.is_empty(),
                "the error quoted the key: {error}"
            );
        }
    }

    #[test]
    fn derived_keys_are_stable_and_label_specific() {
        // `scripts/run-local.sh` depends on both halves: the coordinator
        // derives the addresses it will accept from the same ids the workers
        // derive their keys from, in a separate process.
        assert_eq!(
            WorkerKey::derive("worker-1").address(),
            WorkerKey::derive("worker-1").address()
        );
        assert_ne!(
            WorkerKey::derive("worker-1").address(),
            WorkerKey::derive("worker-11").address(),
            "the label must not be a prefix match"
        );
    }

    #[test]
    fn an_address_round_trips_through_its_hex_form() {
        let address = WorkerKey::derive("worker-1").address();
        let text = hex_address(&address);

        assert!(text.starts_with("0x"));
        assert_eq!(text.len(), 42);
        assert_eq!(parse_address(&text).unwrap(), address);
        assert_eq!(parse_address(&text[2..]).unwrap(), address);
    }

    #[test]
    fn a_registry_entry_that_is_not_an_address_is_refused_by_name() {
        for bad in ["", "0x", "0xdeadbeef", "not-an-address"] {
            let error = parse_address(bad).unwrap_err().to_string();
            assert!(error.contains("20-byte hex address"), "got: {error}");
        }
    }

    #[test]
    fn an_address_is_the_low_20_bytes_of_the_public_key_hash() {
        // The definition Ethereum uses. Getting this wrong — hashing the
        // compressed point, or leaving the 0x04 SEC1 tag in — produces a
        // perfectly stable address that no on-chain registry would ever hold.
        let key = WorkerKey::derive("worker-1");
        let point = key.signing.verifying_key().to_encoded_point(false);

        assert_eq!(point.as_bytes().len(), 65);
        assert_eq!(point.as_bytes()[0], 0x04);
        assert_eq!(key.address(), keccak(&point.as_bytes()[1..])[12..]);
    }

    #[test]
    fn debug_output_never_contains_key_material() {
        let key = WorkerKey::derive("worker-1");
        let rendered = format!("{key:?}");

        assert!(rendered.contains(&hex_address(&key.address())));
        assert!(
            !rendered.contains("SigningKey") && !rendered.contains("signing"),
            "Debug leaked the key: {rendered}"
        );
    }

    #[test]
    fn an_attestation_round_trips_on_the_wire() {
        // `JobReport` carries this, so a decode that silently reordered the
        // fields would turn every honest worker into an unrecoverable signer.
        let attestation = WorkerKey::derive("worker-1").attest(&result_claim());

        let bytes = wincode::serialize(&attestation).unwrap();
        let decoded: Attestation = wincode::deserialize_exact(&bytes).unwrap();
        assert_eq!(decoded, attestation);
    }
}
