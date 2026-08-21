#!/usr/bin/env bash
# Produces the `--attestations` JSON that `node coordinator` is gaining, using
# `cast wallet` instead of a running worker.
#
# This is a *fallback*, and it is worth being precise about what it does and
# does not stand in for, because a fixture that quietly replaces the thing under
# test is worse than no fixture (docs/bridge.md §2b is the same mistake one
# layer down).
#
# What is real here:
#
#   * The signing keys. `--worker-id worker-1` derives
#     `keccak256("DISCA/dev-key/v1" || "worker-1")`, which is byte for byte what
#     `primitives::attest::WorkerKey::derive` computes and therefore the same
#     key the worker process signs with. `--expect <address>` makes that
#     checkable rather than asserted: pass what `node worker-address --id
#     worker-1` printed and this script refuses to sign if the two disagree.
#   * The claim. The 94-byte preimage of docs/bridge.md §2a, assembled here from
#     `cast from-utf8` and a big-endian `printf`, and — when `--bridge` is given
#     — compared against `DiscaBridge.attestationDigest` on the live chain. That
#     is a third independent implementation of §2a agreeing with the Solidity
#     one and the Rust one.
#   * The signature. EIP-191, low-`s`, `v` in {27, 28}: `cast wallet sign
#     --no-hash` over the digest, which is what `ecrecover` in `fulfillJob`
#     will recover from.
#
# What is *not* real: nobody evaluated anything to produce these signatures. A
# worker signs because it ran the circuit; this script signs because it was
# asked to. So this fixture proves the contract accepts a well-formed quorum —
# it cannot prove agreement between independent evaluators, which is the only
# thing M-of-N is for (docs/bridge.md §2). `scripts/run-anvil.sh` says which of
# the two it used, every run, in a banner.
#
# Task 3.3.
set -euo pipefail

die() {
  echo "sign-attestations: $*" >&2
  exit 1
}

job_id=""
bytecode_hash=""
result_hash=""
out=""
bridge=""
rpc=""
worker_ids=()
expected=()

while [ $# -gt 0 ]; do
  case "$1" in
    --job-id) job_id=${2:?}; shift 2 ;;
    --bytecode-hash) bytecode_hash=${2:?}; shift 2 ;;
    --result-hash) result_hash=${2:?}; shift 2 ;;
    --out) out=${2:?}; shift 2 ;;
    # Optional, and the reason to bother: with a bridge address the digest this
    # script computes is checked against the one the contract computes before
    # anything is signed.
    --bridge) bridge=${2:?}; shift 2 ;;
    --rpc) rpc=${2:?}; shift 2 ;;
    --worker-id) worker_ids+=("${2:?}"); shift 2 ;;
    # Positionally matched to --worker-id, so `--worker-id a --expect 0x1
    # --worker-id b --expect 0x2` pairs the way it reads.
    --expect) expected+=("${2:?}"); shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -n "$job_id" ] || die "--job-id is required"
[ -n "$bytecode_hash" ] || die "--bytecode-hash is required"
[ -n "$result_hash" ] || die "--result-hash is required"
[ -n "$out" ] || die "--out is required"
[ ${#worker_ids[@]} -gt 0 ] || die "at least one --worker-id is required"
if [ ${#expected[@]} -gt 0 ] && [ ${#expected[@]} -ne ${#worker_ids[@]} ]; then
  die "--expect given ${#expected[@]} times for ${#worker_ids[@]} workers; pair them or omit them"
fi

command -v cast >/dev/null || die "cast not found; install Foundry"
command -v jq >/dev/null || die "jq not found"

# A 32-byte hex string with the 0x, lower-cased. Anything else in a hash slot
# would be silently zero-extended by `cast keccak` further down and produce a
# digest nobody can explain.
require_hash() {
  printf '%s' "$2" | grep -Eq '^0x[0-9a-fA-F]{64}$' ||
    die "$1 must be 0x followed by 64 hex characters, got: $2"
}
require_hash --bytecode-hash "$bytecode_hash"
require_hash --result-hash "$result_hash"
printf '%s' "$job_id" | grep -Eq '^[0-9]+$' || die "--job-id must be a decimal integer: $job_id"

lower() { printf '%s' "$1" | tr 'A-F' 'a-f'; }

# --- the claim (docs/bridge.md §2a) ------------------------------------------
#
#   offset  len  field
#        0   22  "DISCA/attest/result/v1"   ASCII, no length prefix
#       22    8  jobId                      big-endian uint64
#       30   32  bytecodeHash
#       62   32  resultHash
#       94       total
#
# Fixed width throughout, so plain concatenation is injective and this is
# `abi.encodePacked` by hand.
domain=$(cast from-utf8 "DISCA/attest/result/v1")
job_id_be=$(printf '%016x' "$job_id")
preimage="${domain}${job_id_be}${bytecode_hash#0x}${result_hash#0x}"

# 22 + 8 + 32 + 32 = 94 bytes = 188 hex characters, plus the leading 0x.
[ "${#preimage}" -eq 190 ] || die "assembled a ${#preimage}-character preimage, expected 190"

inner=$(cast keccak "$preimage")
# EIP-191 version 0x45 over a 32-byte payload, i.e. MessageHashUtils.
# toEthSignedMessageHash. Spelled out rather than delegated to `cast wallet
# sign`'s implicit prefixing so that the digest exists as a value this script
# can compare against the contract's before it signs anything.
eip191=$(cast from-utf8 $'\x19Ethereum Signed Message:\n32')
digest=$(cast keccak "${eip191}${inner#0x}")

if [ -n "$bridge" ]; then
  [ -n "$rpc" ] || die "--bridge needs --rpc"
  onchain=$(cast call --rpc-url "$rpc" "$bridge" \
    "attestationDigest(uint64,bytes32,bytes32)(bytes32)" \
    "$job_id" "$bytecode_hash" "$result_hash")
  if [ "$(lower "$onchain")" != "$(lower "$digest")" ]; then
    die "digest disagrees with the contract: shell $digest, DiscaBridge $onchain.
This is docs/bridge.md §2a drifting between implementations -- the same class of
break bridge/test/AttestationVector.t.sol guards between Solidity and Rust."
  fi
fi

# --- sign --------------------------------------------------------------------
rows=""
index=0
for id in "${worker_ids[@]}"; do
  key=$(cast keccak "DISCA/dev-key/v1${id}")
  address=$(cast wallet address --private-key "$key")

  if [ ${#expected[@]} -gt 0 ]; then
    want=${expected[$index]}
    if [ "$(lower "$address")" != "$(lower "$want")" ]; then
      die "worker id '$id' derives $address but the node reports $want.
The dev-key derivation in primitives/src/attest.rs and the one in this script
have diverged, so these signatures would recover to addresses no registry holds."
    fi
  fi

  signature=$(cast wallet sign --private-key "$key" --no-hash "$digest")
  [ "${#signature}" -eq 132 ] || die "cast returned a ${#signature}-character signature for $id"

  # r, s, v out of the 65 packed bytes. `cast wallet sign` emits v as 27/28,
  # which is the form `ecrecover` takes; k256 produces low-`s` and so does
  # cast, so the EIP-2 check in fulfillJob is satisfied without normalising.
  r="0x${signature:2:64}"
  s="0x${signature:66:64}"
  v=$((16#${signature:130:2}))
  case "$v" in
    27 | 28) ;;
    *) die "cast produced v=$v for $id; fulfillJob accepts only 27 or 28" ;;
  esac

  rows+="$(lower "$address") $r $s $v"$'\n'
  index=$((index + 1))
done

# fulfillJob requires strictly increasing recovered addresses, which is what
# makes distinctness an O(n) check with no storage (docs/bridge.md §2a step 4).
# Lower-cased first: a checksummed address sorts by ASCII, where every
# uppercase hex digit sorts below every lowercase one, so sorting the display
# form would produce an order the contract rejects.
sorted=$(printf '%s' "$rows" | sort)

# `printf '%s\n'`, not `'%s'`: command substitution strips the trailing newline,
# and `read` fails on a final line that has none -- silently dropping the last
# attester, which shows up on-chain as QuorumNotMet with no explanation.
attesters=$(printf '%s\n' "$sorted" | while read -r address r s v; do
  [ -n "$address" ] || continue
  jq -nc --arg address "$address" --arg r "$r" --arg s "$s" --argjson v "$v" \
    '{address: $address, r: $r, s: $s, v: $v}'
done | jq -sc '.')

# jq --argjson for the job id, not --arg: it is a number in the shape the
# coordinator writes, and a quoted one would fail whatever consumes this next.
jq -n \
  --argjson jobId "$job_id" \
  --arg bytecodeHash "$(lower "$bytecode_hash")" \
  --arg resultHash "$(lower "$result_hash")" \
  --argjson attesters "$attesters" \
  '{jobId: $jobId, bytecodeHash: $bytecodeHash, resultHash: $resultHash, attesters: $attesters}' \
  > "$out"

# The shell can drop a line without saying so -- a `read` past a missing final
# newline, a subshell that swallowed an error. A short attester array reaches
# the chain as QuorumNotMet, which reads like a coordinator that could not get
# agreement rather than like a bug in this file, so count them here.
written=$(jq '.attesters | length' "$out")
[ "$written" -eq "${#worker_ids[@]}" ] ||
  die "wrote $written attester(s) for ${#worker_ids[@]} worker id(s); $out is incomplete"

echo "sign-attestations: wrote $written signature(s) over $digest to $out" >&2
