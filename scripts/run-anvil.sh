#!/usr/bin/env bash
# The DISCA bridge against a real chain: deploy, register, post, settle, refund.
#
# `scripts/run-local.sh` shows three worker processes agreeing on a ciphertext
# and one key holder decrypting it. Everything in `docs/bridge.md` after that —
# the escrow, the registry, the quorum check, the callback — has only ever been
# exercised by Solidity calling Solidity. This script closes that gap: an actual
# Anvil, actual transactions, and a result blob that a real DISCA network
# produced being accepted by `DiscaBridge.fulfillJob` and then decrypted from
# the bytes *the chain* carries rather than the ones on disk.
#
# Task 3.3, and step 4 of `docs/bridge.md` §8.
#
#   ./scripts/run-anvil.sh              # one faulty worker: outvoted 2-of-3,
#                                       # then unanimity against the same liar
#                                       # fails and the job refunds
#   HONEST=1 ./scripts/run-anvil.sh     # all three honest; the second job is
#                                       # simply never worked on
#   ./scripts/run-anvil.sh --synthetic  # contracts and chain only, no Rust, no
#                                       # FHE; what CI runs
#
# **Both passes are needed to conclude anything, and so are both modes.** A
# settlement that always succeeds proves nothing about the quorum check: you see
# `JobFulfilled` whether or not the contract looked at the signatures. So the
# default run stages exactly one liar, watches it get outvoted, and then re-runs
# the *same* three workers demanding unanimity — where the liar is now decisive,
# no quorum forms, nothing settles, and the escrow goes back to the poster
# through `refundOnTimeout`. That pairing is the same discipline as
# `run-local.sh`'s header, one layer up.
#
# Only the fault is staged. The two honest workers are separate processes that
# never talk to each other and are never told what to answer; they match because
# they independently computed the same ciphertext byte for byte. Nothing tells
# the coordinator or the contract which worker is lying.
#
# ## What is real here and what is not
#
# Real: the chain, the contracts, the program hash, the server key hash, the
# input commitments, the input ciphertexts, the evaluation, the result blob,
# the worker identities, and the gas.
#
# Not yet real: **who produced the signature bytes.** The coordinator is gaining
# `--attestations <path>`; until it has it, this script signs the §2a claim
# itself with `bridge/script/fixtures/sign-attestations.sh`, under the same
# worker keys (`keccak256("DISCA/dev-key/v1" || id)`, which is what
# `primitives::attest::WorkerKey::derive` computes) and over exactly the
# attester set the coordinator settled on. The `ATTESTATION SOURCE` banner in
# the "collecting the attestations" step says which of the two ran, every time.
#
# There is a second reason the fixture is currently unavoidable, and it is not a
# missing flag: the coordinator invents its own job id (a timestamp-shaped
# nonce), while `fulfillJob` builds its digest from the id `submitJob` assigned.
# The two cannot agree until task 2.9f moves id assignment on-chain. This script
# checks for that explicitly rather than letting it surface as
# `NotRegisteredWorker`, which would read as a broken registry.
set -euo pipefail

cd "$(dirname "$0")/.."

# --- configuration -----------------------------------------------------------

PROGRAM=${PROGRAM:-committee-tally/committee_tally.wasm}
FUNCTION=${FUNCTION:-tally4_select}
INPUTS=${INPUTS:-71,93,42,88}
# M in M-of-N for the settling pass. The failing pass always demands unanimity,
# because that is what makes one liar decisive.
ATTESTERS=${ATTESTERS:-2}
# Seconds a job stays fulfillable. Generous: `refundOnTimeout` is reached by
# fast-forwarding Anvil's clock, not by waiting, so the only thing this bounds
# is how long the honest pass has to settle.
JOB_TIMEOUT=${JOB_TIMEOUT:-600}
# How long the coordinator waits for agreement. Short, because the failing pass
# reaches its conclusion by timing out and a demo nobody watches is a demo
# nobody runs.
DEADLINE=${DEADLINE:-30}
# 1 ether. Large enough that the escrow dominates the gas the coordinator spends
# collecting it, so the "escrow moved" step can check `after - before` as an
# exact identity rather than as "it went up".
ESCROW_WEI=${ESCROW_WEI:-1000000000000000000}

ANVIL_PORT=${ANVIL_PORT:-8545}
RPC="http://127.0.0.1:${ANVIL_PORT}"
# Anvil's documented default. Named explicitly so the account keys below are a
# consequence of something in this file rather than of a default that could
# change under it; step 3 checks that the derived account really is funded.
MNEMONIC=${MNEMONIC:-"test test test test test test test test test test test junk"}

NETWORK=real
for arg in "$@"; do
  case "$arg" in
    --synthetic) NETWORK=synthetic ;;
    -h | --help)
      sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//;$d'
      exit 0
      ;;
    *)
      echo "run-anvil: unknown argument: $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

# The first five accounts of the mnemonic above. Public knowledge, and this is a
# throwaway chain. Five distinct parties rather than one, because "the escrow
# moved" and "the refund landed" are only assertions if the payer, the payee and
# the caller are different accounts — a single account settling to itself shows a
# balance that only moved by the gas.
DEPLOYER_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
COORDINATOR_KEY=0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d
COMMITTEE_KEY=0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a
POSTER_KEY=0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6
BYSTANDER_KEY=0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a

WORKER_IDS=(worker-1 worker-2 worker-3)
WORKER_PORTS=(8081 8082 8083)
COORDINATOR_BIND=127.0.0.1:8080

# --- vocabulary --------------------------------------------------------------
#
# Every check in this file goes through one of these. The rule is that nothing
# is ever reported as passing on the strength of a command not having crashed:
# `cast call` against a wrong address returns `0x` and exits zero, and a script
# that prints "ok" for that is worse than no script.

step_no=0

die() {
  printf '\nrun-anvil: FAIL: %s\n' "$*" >&2
  exit 1
}

step() {
  step_no=$((step_no + 1))
  printf '\n== %2d. %s\n' "$step_no" "$*"
}

note() { printf '       %s\n' "$*"; }

assert_eq() {
  # label, actual, expected. Compared case-insensitively because half of these
  # values are addresses and hashes that `cast` prints checksummed in one place
  # and lower-case in another.
  local label=$1 got=$2 want=$3
  [ -n "$got" ] || die "$label: got an empty value (expected '$want')"
  if [ "$(printf '%s' "$got" | tr 'A-F' 'a-f')" != "$(printf '%s' "$want" | tr 'A-F' 'a-f')" ]; then
    die "$label: got '$got', expected '$want'"
  fi
  printf '   ok  %s = %s\n' "$label" "$got"
}

assert_ne() {
  local label=$1 got=$2 unwanted=$3
  [ -n "$got" ] || die "$label: got an empty value"
  [ "$got" != "$unwanted" ] || die "$label: got '$got', which is exactly what it must not be"
  printf '   ok  %s = %s (not %s)\n' "$label" "$got" "$unwanted"
}

# `cast call`, with the empty string treated as the failure it is.
call() {
  local out
  out=$(cast call --rpc-url "$RPC" "$@") || die "cast call failed: $*"
  [ -n "$out" ] || die "cast call returned nothing (wrong address, or a reverted view): $*"
  [ "$out" != "0x" ] || die "cast call returned 0x (no code at that address?): $*"
  printf '%s' "$out"
}

# `cast send`, with the receipt status checked. A reverted transaction is
# mined, so `cast send` succeeds and the run would sail past it.
LAST_RECEIPT=""
send() {
  local key=$1
  shift
  # `summary` rather than `$*`: an input or result blob is 24 KB of hex on this
  # command line, and a failure that prints all of it buries the reason.
  local summary
  summary=$(printf '%s ' "$@" | cut -c1-160)
  LAST_RECEIPT=$(cast send --rpc-url "$RPC" --private-key "$key" --json "$@") ||
    die "cast send failed: ${summary}..."
  local status
  status=$(printf '%s' "$LAST_RECEIPT" | jq -r '.status')
  case "$status" in
    0x1 | 1 | success) ;;
    *) die "transaction reverted (status=$status): ${summary}..." ;;
  esac
}

# `cast send` that is *required* to fail. Used where the point being made is
# that the contract refuses.
send_must_revert() {
  local label=$1 key=$2
  shift 2
  if cast send --rpc-url "$RPC" --private-key "$key" "$@" > /dev/null 2>&1; then
    die "$label: the transaction succeeded and it must not have"
  fi
  printf '   ok  %s: refused, as it must be\n' "$label"
}

receipt_field() { printf '%s' "$LAST_RECEIPT" | jq -r "$1"; }

# One field out of the `Job` struct `bridge.jobs(jobId)` returns.
#
# `cast call` has no `--json` in every version this has to work with, so what
# comes back is a printed tuple: `(1, 0xposter, 0xcallback, [0xcommit, ...],
# 1787321745 [1.787e9], 500000000000000000 [5e17], 1)`. Neither `cut -d,` nor a
# regex survives that — the commitments array contains commas, and `cast`
# annotates large integers with a bracketed exponent that looks exactly like the
# array does. So the split is done properly, once, here.
JOB_FIELD_PROGRAM_ID=0
JOB_FIELD_POSTER=1
JOB_FIELD_DEADLINE=4
JOB_FIELD_ESCROW=5
JOB_FIELD_STATE=6

job_field() {
  local job_id=$1 index=$2 tuple
  tuple=$(call "$BRIDGE" \
    "jobs(uint256)((uint256,address,address,bytes32[],uint64,uint256,uint8))" "$job_id")
  python3 - "$tuple" "$index" <<'PY'
import re
import sys

raw = sys.argv[1].strip()
if not (raw.startswith("(") and raw.endswith(")")):
    raise SystemExit(f"not a printed tuple: {raw!r}")

fields, depth, current = [], 0, ""
for character in raw[1:-1]:
    if character == "[":
        depth += 1
    elif character == "]":
        depth -= 1
    if character == "," and depth == 0:
        fields.append(current)
        current = ""
    else:
        current += character
fields.append(current)

value = fields[int(sys.argv[2])].strip()
# Strip cast's `123456 [1.2e5]` annotation, but never a value that *is* an
# array -- those start with the bracket rather than ending with an annotation.
if not value.startswith("["):
    value = re.sub(r"\s*\[[^]]*\]$", "", value)
print(value)
PY
}

# Big integers past 2^63 (wei balances) break shell arithmetic silently, so
# every sum over them goes through python3, which the repo already depends on
# for scripts/lib.
bigint() { python3 -c "import sys; print(eval(sys.argv[1]))" "$1"; }

hexdump_file() { python3 -c "import sys; print('0x'+open(sys.argv[1],'rb').read().hex())" "$1"; }

write_hex() {
  # hex string -> file. The other direction, for the blob the chain carries.
  python3 -c "import sys; open(sys.argv[2],'wb').write(bytes.fromhex(sys.argv[1].removeprefix('0x')))" \
    "$1" "$2"
}

# `tracing` writes colour even when its output is a file, so every log this
# script parses has to come through here first.
strip_ansi() { sed $'s/\033\\[[0-9;]*[A-Za-z]//g'; }

# --- preflight ---------------------------------------------------------------

step "checking the tools this needs are present"
for tool in anvil cast forge jq python3; do
  command -v "$tool" > /dev/null || die "$tool not found in PATH"
done
note "anvil/cast/forge: $(cast --version | head -1)"
if [ "$NETWORK" = real ]; then
  [ -f "$PROGRAM" ] ||
    die "missing $PROGRAM — build it with: (cd committee-tally && cargo build --release)"
fi
[ -x bridge/script/fixtures/sign-attestations.sh ] ||
  die "bridge/script/fixtures/sign-attestations.sh is missing or not executable"

work=$(mktemp -d "${TMPDIR:-/tmp}/disca-anvil.XXXXXX")
pids=()
anvil_pid=""

cleanup() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2> /dev/null || true; done
  [ -n "$anvil_pid" ] && kill "$anvil_pid" 2> /dev/null
  rm -rf "$work"
  return 0
}
trap cleanup EXIT

if [ "$NETWORK" = real ]; then
  step "building the node (with fault-injection) and disca-cli"
  # fault-injection is off by default, so a release build has no way to be told
  # to return a wrong answer. The failing pass needs exactly that.
  #
  # One invocation for both binaries, not two. Two `cargo build` calls with
  # different feature sets resolve the shared dependency graph differently and
  # each invalidates the other's `primitives`, so the pair rebuilds tfhe's
  # dependents every run — measured at three and a half minutes against nothing
  # having changed.
  cargo build --release -p node -p disca-cli --features node/fault-injection
  NODE=target/release/node
  CLI=target/release/disca-cli
else
  step "synthetic mode: no Rust, no FHE"
  note "Every contract call below is real and every signature is a real"
  note "secp256k1 signature over the docs/bridge.md §2a claim. What is fake is"
  note "the ciphertext: the blobs are random bytes of the right size, so there"
  note "is nothing to decrypt and nothing to agree about. This mode exercises"
  note "the chain half and the script; it says nothing about M-of-N."
fi

# --- the chain ---------------------------------------------------------------

step "starting anvil on port $ANVIL_PORT"
# A leftover anvil would answer on this port with a chain that already has our
# contracts on it at different addresses, and every assertion below would be
# against the wrong deployment.
if command -v lsof > /dev/null; then
  lsof -ti "tcp:${ANVIL_PORT}" 2> /dev/null | xargs -r kill -9 2> /dev/null || true
fi
anvil --port "$ANVIL_PORT" --mnemonic "$MNEMONIC" --silent > "$work/anvil.log" 2>&1 &
anvil_pid=$!

for _ in $(seq 1 100); do
  cast block-number --rpc-url "$RPC" > /dev/null 2>&1 && break
  sleep 0.2
done
cast block-number --rpc-url "$RPC" > /dev/null 2>&1 || die "anvil did not come up on $RPC"

deployer=$(cast wallet address --private-key "$DEPLOYER_KEY")
coordinator=$(cast wallet address --private-key "$COORDINATOR_KEY")
committee=$(cast wallet address --private-key "$COMMITTEE_KEY")
poster=$(cast wallet address --private-key "$POSTER_KEY")
bystander=$(cast wallet address --private-key "$BYSTANDER_KEY")

# If the mnemonic and the hard-coded keys ever part company these accounts are
# unfunded, and the first `cast send` fails with "insufficient funds" — which
# reads as a broken chain rather than as a mismatched constant.
for who in "$deployer" "$coordinator" "$committee" "$poster" "$bystander"; do
  balance=$(cast balance --rpc-url "$RPC" "$who")
  [ "$balance" != "0" ] ||
    die "$who has no balance; the keys in this script do not match \$MNEMONIC"
done
note "deployer=$deployer coordinator=$coordinator"
note "committee=$committee poster=$poster bystander=$bystander"

# --- the key holder ----------------------------------------------------------
#
# `disca-cli` is a separate process for the same reason it is a separate party:
# it is the only thing that ever holds the client key or sees a plaintext.
# Nothing else in this script is given a path to `client.key`, including the
# coordinator that settles the job and the chain that carries the result.

step "key holder: keys, program and encrypted inputs"
if [ "$NETWORK" = real ]; then
  server_key_hash=$("$CLI" keygen --out-dir "$work/keys" --force | sed -n 's/^server_key_hash=//p')
  [ -n "$server_key_hash" ] || die "disca-cli keygen printed no server_key_hash"

  bytecode_hash=$("$CLI" compile --input "$PROGRAM" --output "$work/program.bytecode" |
    sed -n 's/^bytecode_hash=//p')
  [ -n "$bytecode_hash" ] || die "disca-cli compile printed no bytecode_hash"

  commitments=()
  while read -r line; do
    commitments+=("$line")
  done < <("$CLI" encrypt --client-key "$work/keys/client.key" \
    --values "$INPUTS" --out-dir "$work/inputs" | sed -n 's/^commitment=//p')
else
  # Sized off `docs/architecture.md` §2 so the gas is comparable: 2.3 KiB per
  # fresh input, 11.8 KiB for a computed result.
  server_key_hash=$(cast keccak "synthetic server key")
  bytecode_hash=$(cast keccak "synthetic bytecode")
  mkdir -p "$work/inputs"
  commitments=()
  for i in 0 1 2 3; do
    head -c 2355 /dev/urandom > "$work/inputs/input-$i.ct"
    commitments+=("$(cast keccak "$(hexdump_file "$work/inputs/input-$i.ct")")")
  done
fi

[ "${#commitments[@]}" -gt 0 ] || die "no input commitments were produced"
note "bytecode_hash   = $bytecode_hash"
note "server_key_hash = $server_key_hash"
note "inputs          = ${#commitments[@]} ciphertexts"

# --- worker identities -------------------------------------------------------

step "worker identities"
worker_addresses=()
for id in "${WORKER_IDS[@]}"; do
  if [ "$NETWORK" = real ]; then
    # RUST_LOG=off because this captures stdout and a stray log line would be
    # substituted straight into a `registerWorker` transaction.
    address=$(RUST_LOG=off "$NODE" worker-address --id "$id")
  else
    address=$(cast wallet address --private-key "$(cast keccak "DISCA/dev-key/v1${id}")")
  fi
  printf '%s' "$address" | grep -Eq '^0x[0-9a-fA-F]{40}$' ||
    die "worker-address for $id printed something that is not an address: '$address'"
  worker_addresses+=("$address")
  note "$id -> $address"
done

# --- deploy ------------------------------------------------------------------

step "deploying DiscaBridge and CommitteeTally, and pinning the program"
deploy_log="$work/deploy.log"
JOB_TIMEOUT="$JOB_TIMEOUT" \
  BYTECODE_HASH="$bytecode_hash" \
  SERVER_KEY_HASH="$server_key_hash" \
  ATTESTERS_REQUIRED="$ATTESTERS" \
  COORDINATOR="$coordinator" \
  COMMITTEE="$committee" \
  forge script bridge/script/Deploy.s.sol:Deploy \
  --root bridge \
  --rpc-url "$RPC" \
  --private-key "$DEPLOYER_KEY" \
  --broadcast > "$deploy_log" 2>&1 ||
  {
    tail -40 "$deploy_log" >&2
    die "forge script Deploy failed; see $deploy_log"
  }

read_deployed() {
  local key=$1 value
  value=$(strip_ansi < "$deploy_log" | sed -n "s/^[[:space:]]*${key}=//p" | tail -1)
  [ -n "$value" ] || die "Deploy.s.sol did not print $key= (see $deploy_log)"
  printf '%s' "$value"
}
BRIDGE=$(read_deployed BRIDGE_ADDRESS)
TALLY=$(read_deployed TALLY_ADDRESS)
PROGRAM_ID=$(read_deployed PROGRAM_ID)
note "DiscaBridge    = $BRIDGE"
note "CommitteeTally = $TALLY"
note "programId      = $PROGRAM_ID"

# Read back rather than trust the log: the script could have printed a value it
# then failed to broadcast.
assert_eq "bridge.jobTimeout()" "$(call "$BRIDGE" "jobTimeout()(uint64)")" "$JOB_TIMEOUT"
assert_eq "bridge.coordinator()" "$(call "$BRIDGE" "coordinator()(address)")" "$coordinator"
assert_eq "tally.bridge()" "$(call "$TALLY" "bridge()(address)")" "$BRIDGE"
assert_eq "tally.programId()" "$(call "$TALLY" "programId()(uint256)")" "$PROGRAM_ID"

program=$(call "$BRIDGE" "programs(uint256)(bytes32,bytes32,uint8)" "$PROGRAM_ID")
assert_eq "programs[$PROGRAM_ID].bytecodeHash" "$(printf '%s\n' "$program" | sed -n 1p)" \
  "$bytecode_hash"
assert_eq "programs[$PROGRAM_ID].serverKeyHash" "$(printf '%s\n' "$program" | sed -n 2p)" \
  "$server_key_hash"
assert_eq "programs[$PROGRAM_ID].attestersRequired" \
  "$(printf '%s\n' "$program" | sed -n 3p | tr -d '"')" "$ATTESTERS"

# --- the registry ------------------------------------------------------------

step "registerWorker for each worker"
for address in "${worker_addresses[@]}"; do
  send "$DEPLOYER_KEY" "$BRIDGE" "registerWorker(address)" "$address"
  assert_eq "isRegisteredWorker($address)" \
    "$(call "$BRIDGE" "isRegisteredWorker(address)(bool)" "$address")" "true"
done
# The negative half. Without it "registered" would be indistinguishable from a
# mapping that returns true for everything.
assert_eq "isRegisteredWorker(a stranger)" \
  "$(call "$BRIDGE" "isRegisteredWorker(address)(bool)" "$bystander")" "false"

# --- post the job ------------------------------------------------------------

step "the committee posts the tally, escrowing $ESCROW_WEI wei"
commit_array="[$(
  IFS=,
  printf '%s' "${commitments[*]}"
)]"
blob_list=()
for i in $(seq 0 $((${#commitments[@]} - 1))); do
  blob_list+=("$(hexdump_file "$work/inputs/input-$i.ct")")
done
blob_array="[$(
  IFS=,
  printf '%s' "${blob_list[*]}"
)]"

send "$COMMITTEE_KEY" --value "$ESCROW_WEI" "$TALLY" \
  "startTally(bytes32[],bytes[])" "$commit_array" "$blob_array"
note "startTally gas: $(bigint "$(receipt_field .gasUsed)")"

JOB_ID=$(call "$TALLY" "jobId()(uint256)")
assert_ne "tally.jobId()" "$JOB_ID" "0"
assert_eq "bridge.jobCount()" "$(call "$BRIDGE" "jobCount()(uint256)")" "$JOB_ID"
assert_eq "jobs[$JOB_ID].programId" "$(job_field "$JOB_ID" $JOB_FIELD_PROGRAM_ID)" "$PROGRAM_ID"
assert_eq "jobs[$JOB_ID].poster is the tally contract" \
  "$(job_field "$JOB_ID" $JOB_FIELD_POSTER)" "$TALLY"
assert_eq "jobs[$JOB_ID].escrow" "$(job_field "$JOB_ID" $JOB_FIELD_ESCROW)" "$ESCROW_WEI"
# The deadline is per deployment, not per job (`DiscaBridge.jobTimeout`): a
# poster-chosen one could be set to zero, making every job refundable before the
# coordinator could possibly settle it.
# Through `bigint` because a receipt's blockNumber is hex and `cast block`
# rejects an odd number of hex digits.
posted_at=$(cast block --rpc-url "$RPC" "$(bigint "$(receipt_field .blockNumber)")" \
  --field timestamp)
assert_eq "jobs[$JOB_ID].deadline" "$(job_field "$JOB_ID" $JOB_FIELD_DEADLINE)" \
  "$(bigint "$posted_at + $JOB_TIMEOUT")"

# The commitments the contract stored are the ones `disca-cli encrypt` printed,
# in the order it printed them. Order is the part that cannot be caught later:
# the inputs are ciphertext, so a transposition produces a plausible answer to a
# different question and every downstream check still passes.
onchain_commits=$(call "$BRIDGE" "inputCommitsOf(uint256)(bytes32[])" "$JOB_ID")
for i in $(seq 0 $((${#commitments[@]} - 1))); do
  got=$(printf '%s' "$onchain_commits" | tr -d '[] ' | cut -d, -f$((i + 1)))
  assert_eq "inputCommits[$i]" "$got" "${commitments[$i]}"
done

# --- run the off-chain job ---------------------------------------------------

if [ "$NETWORK" = real ]; then
  step "starting three workers (one of them faulty unless HONEST=1)"
  for port in "${WORKER_PORTS[@]}" 8080; do
    if command -v lsof > /dev/null; then
      lsof -ti "tcp:$port" 2> /dev/null | xargs -r kill -9 2> /dev/null || true
    fi
  done

  third=(--id "${WORKER_IDS[2]}" --bind "127.0.0.1:${WORKER_PORTS[2]}")
  if [ -z "${HONEST:-}" ]; then
    # Fault injection, not mocking: this worker still fetches and verifies the
    # server key, validates the bytecode, checks the input commitments and does
    # the real homomorphic evaluation. Only the answer is corrupted, at the last
    # step before sealing. Nothing in the protocol says which worker it is.
    third+=(--faulty)
    note "${WORKER_IDS[2]} is the liar"
  else
    note "HONEST=1: all three workers are honest"
  fi

  "$NODE" worker --id "${WORKER_IDS[0]}" --bind "127.0.0.1:${WORKER_PORTS[0]}" \
    > "$work/worker-1.log" 2>&1 &
  pids+=($!)
  "$NODE" worker --id "${WORKER_IDS[1]}" --bind "127.0.0.1:${WORKER_PORTS[1]}" \
    > "$work/worker-2.log" 2>&1 &
  pids+=($!)
  "$NODE" worker "${third[@]}" > "$work/worker-3.log" 2>&1 &
  pids+=($!)

  for port in "${WORKER_PORTS[@]}"; do
    for _ in $(seq 1 100); do
      nc -z 127.0.0.1 "$port" 2> /dev/null && break
      sleep 0.2
    done
    nc -z 127.0.0.1 "$port" 2> /dev/null || die "worker on port $port never bound"
  done

  registry=()
  for address in "${worker_addresses[@]}"; do registry+=(--registered-worker "$address"); done
  input_flags=()
  for i in $(seq 0 $((${#commitments[@]} - 1))); do
    input_flags+=(--input "$work/inputs/input-$i.ct")
  done

  # `--attestations` is what this whole exercise is for. It may not exist yet;
  # asking for it when it does not is a usage error the coordinator refuses at
  # startup, so the flag is probed rather than passed hopefully.
  attestation_flags=()
  if "$NODE" coordinator --help 2>&1 | grep -q -- '--attestations'; then
    attestation_flags=(--attestations "$work/attestations.json")
  fi

  # And the id the *chain* assigned, which is the other half. A worker signs a
  # digest binding the job id, and `fulfillJob` rebuilds that digest from the id
  # `submitJob` returned. A coordinator minting its own produces signatures that
  # recover to addresses the registry has never seen, so the contract rejects a
  # settlement that is correct in every other respect — and rejects it as
  # `NotRegisteredWorker`, for workers that are registered. Probed for the same
  # reason as above.
  if "$NODE" coordinator --help 2>&1 | grep -q -- '--job-id'; then
    attestation_flags+=(--job-id "$JOB_ID")
  fi

  run_coordinator() {
    local quorum=$1 log=$2
    shift 2
    "$NODE" coordinator \
      --bind "$COORDINATOR_BIND" \
      --worker "127.0.0.1:${WORKER_PORTS[0]}" \
      --worker "127.0.0.1:${WORKER_PORTS[1]}" \
      --worker "127.0.0.1:${WORKER_PORTS[2]}" \
      "${registry[@]}" \
      --attesters "$quorum" \
      --server-key "$work/keys/server.key" \
      --bytecode "$work/program.bytecode" \
      --function "$FUNCTION" \
      "${input_flags[@]}" \
      --result "$work/result.blob" \
      --deadline-secs "$DEADLINE" \
      "$@" > "$log" 2>&1
  }

  step "running the job: $ATTESTERS-of-${#WORKER_IDS[@]} over $FUNCTION"
  # `${arr[@]+"${arr[@]}"}` rather than `"${arr[@]:-}"`: the latter expands an
  # empty array to one empty-string argument, which the coordinator would
  # receive as a malformed flag.
  run_coordinator "$ATTESTERS" "$work/coordinator.log" \
    ${attestation_flags[@]+"${attestation_flags[@]}"} ||
    {
      tail -30 "$work/coordinator.log" >&2
      die "the coordinator did not reach $ATTESTERS-of-${#WORKER_IDS[@]} agreement"
    }
  settled=$(strip_ansi < "$work/coordinator.log" | grep 'job settled' | tail -1)
  [ -n "$settled" ] || die "the coordinator exited 0 without logging 'job settled'"

  [ -s "$work/result.blob" ] || die "the coordinator wrote no result blob"
  result_hash=$(cast keccak "$(hexdump_file "$work/result.blob")")

  # The hash the coordinator says it settled on, against the hash of the bytes
  # it wrote. These are the same number computed by two programs, and if they
  # ever differ the blob on disk is not the blob a quorum signed.
  logged_hash=$(printf '%s' "$settled" | sed -n 's/.*result_hash=\(0x[0-9a-f]*\).*/\1/p')
  assert_eq "coordinator result_hash vs keccak256(result.blob)" "$logged_hash" "$result_hash"

  # The winning group, taken from the coordinator's own report rather than
  # assumed to be "the workers that are not the liar". In HONEST mode any two of
  # the three could be here, and the fixture below has to sign as whoever
  # actually agreed.
  # Narrowed to the `attesters=[...]` field before matching, because a bare
  # `0x[0-9a-f]{40}` also matches the first 40 characters of the 64-character
  # `result_hash=` on the same line — which then fails as "not one of the
  # workers this script started", pointing at the registry instead of at this
  # regex.
  attester_field=$(printf '%s' "$settled" | sed -n 's/.*attesters=\[\([^]]*\)\].*/\1/p')
  [ -n "$attester_field" ] ||
    die "could not find attesters=[...] in the coordinator's 'job settled' line.
The log format this parses is node/src/coordinator.rs's; if it has moved, this
script is reading a settlement it cannot describe."

  settled_attesters=()
  while read -r address; do
    [ -n "$address" ] && settled_attesters+=("$address")
  done < <(printf '%s' "$attester_field" | grep -o '0x[0-9a-f]\{40\}')
  [ "${#settled_attesters[@]}" -ge "$ATTESTERS" ] ||
    die "parsed ${#settled_attesters[@]} attester(s) out of the coordinator log, expected >= $ATTESTERS"
  note "settled by: ${settled_attesters[*]}"

  coordinator_job_id=$(printf '%s' "$settled" | sed -n 's/.*job_id=\([0-9]*\).*/\1/p')
  note "coordinator job id: ${coordinator_job_id:-unparsed}   on-chain job id: $JOB_ID"
else
  step "fabricating a result blob (synthetic mode)"
  head -c 12075 /dev/urandom > "$work/result.blob"
  result_hash=$(cast keccak "$(hexdump_file "$work/result.blob")")
  settled_attesters=("${worker_addresses[0]}" "${worker_addresses[1]}")
  coordinator_job_id=""
fi
note "result_hash = $result_hash ($(wc -c < "$work/result.blob" | tr -d ' ') bytes)"

# --- the attestations --------------------------------------------------------

step "collecting the attestations"
attestations="$work/attestations.json"
source_label=""

if [ -s "$attestations" ]; then
  source_label="REAL — written by the coordinator's --attestations"
  file_job_id=$(jq -r '.jobId' "$attestations")
  if [ "$file_job_id" != "$JOB_ID" ]; then
    die "the coordinator signed over job id $file_job_id but submitJob assigned $JOB_ID.
fulfillJob builds its digest from the on-chain id (docs/bridge.md §2a step 2),
so these signatures recover to addresses no registry holds and the transaction
would revert with NotRegisteredWorker. The coordinator has to be told the id
submitJob assigned — task 2.9f."
  fi
  assert_eq "attestations.bytecodeHash" "$(jq -r '.bytecodeHash' "$attestations")" "$bytecode_hash"
  assert_eq "attestations.resultHash" "$(jq -r '.resultHash' "$attestations")" "$result_hash"
else
  source_label="FIXTURE — signed by bridge/script/fixtures/sign-attestations.sh"
  if [ "$NETWORK" = real ] && [ -n "$coordinator_job_id" ] && [ "$coordinator_job_id" != "$JOB_ID" ]; then
    note "NOTE: the coordinator signed its own claims over job id $coordinator_job_id,"
    note "      not the $JOB_ID that submitJob assigned. Even with --attestations those"
    note "      signatures could not settle this job; see task 2.9f. The fixture below"
    note "      re-signs the same claim over the on-chain id, under the same keys."
  fi

  # Sign as exactly the workers that agreed, not as a fixed pair. Mapping the
  # addresses back to ids means a run where a different two workers won produces
  # a fixture naming those two.
  signer_args=()
  for address in "${settled_attesters[@]}"; do
    matched=""
    for i in 0 1 2; do
      if [ "$(printf '%s' "${worker_addresses[$i]}" | tr 'A-F' 'a-f')" = \
        "$(printf '%s' "$address" | tr 'A-F' 'a-f')" ]; then
        matched=${WORKER_IDS[$i]}
        break
      fi
    done
    [ -n "$matched" ] ||
      die "the coordinator settled on $address, which is not one of the workers this script started"
    signer_args+=(--worker-id "$matched" --expect "$address")
  done

  bridge/script/fixtures/sign-attestations.sh \
    --job-id "$JOB_ID" \
    --bytecode-hash "$bytecode_hash" \
    --result-hash "$result_hash" \
    --bridge "$BRIDGE" --rpc "$RPC" \
    "${signer_args[@]}" \
    --out "$attestations"
fi

printf '\n   ATTESTATION SOURCE: %s\n\n' "$source_label"

count=$(jq '.attesters | length' "$attestations")
[ "$count" -ge "$ATTESTERS" ] || die "the attestation file carries $count signature(s), quorum is $ATTESTERS"

# Every attester the file names must be one the chain will count. Checking here
# turns "the transaction reverted" into "this address is not registered".
while read -r address; do
  assert_eq "isRegisteredWorker($address)" \
    "$(call "$BRIDGE" "isRegisteredWorker(address)(bool)" "$address")" "true"
done < <(jq -r '.attesters[].address' "$attestations")

quorum_arg="[$(jq -r '[.attesters[] | "(\(.r),\(.s),\(.v))"] | join(",")' "$attestations")]"
result_blob_hex=$(hexdump_file "$work/result.blob")

# --- settle ------------------------------------------------------------------

step "fulfillJob"
coordinator_before=$(cast balance --rpc-url "$RPC" "$coordinator")
tally_before=$(call "$TALLY" "resultCommit()(bytes32)")
assert_eq "tally.resultCommit() before settlement" "$tally_before" \
  "0x0000000000000000000000000000000000000000000000000000000000000000"

send "$COORDINATOR_KEY" "$BRIDGE" \
  "fulfillJob(uint256,bytes32,bytes,(bytes32,bytes32,uint8)[])" \
  "$JOB_ID" "$result_hash" "$result_blob_hex" "$quorum_arg"

gas_used=$(bigint "$(receipt_field .gasUsed)")
gas_price=$(bigint "$(receipt_field .effectiveGasPrice)")
printf '   ok  fulfillJob mined: %s gas at %s wei/gas over a %s-byte result blob\n' \
  "$gas_used" "$gas_price" "$(wc -c < "$work/result.blob" | tr -d ' ')"

step "the JobFulfilled event"
job_fulfilled_topic=$(cast keccak "JobFulfilled(uint256,bytes32,bytes)")
event_data=$(receipt_field \
  "[.logs[] | select(.topics[0]==\"$job_fulfilled_topic\")] | .[0].data // empty")
[ -n "$event_data" ] || die "no JobFulfilled log in the receipt"

decoded=$(cast abi-decode "f()(bytes32,bytes)" "$event_data") ||
  die "could not decode the JobFulfilled data"
event_hash=$(printf '%s\n' "$decoded" | sed -n 1p)
event_blob=$(printf '%s\n' "$decoded" | sed -n 2p)
assert_eq "JobFulfilled.resultHash" "$event_hash" "$result_hash"
assert_eq "keccak256(JobFulfilled.resultBlob)" "$(cast keccak "$event_blob")" "$result_hash"

# JobState.Fulfilled == 2 (IDiscaBridge.JobState; None is reserved as zero so an
# unwritten mapping entry cannot read as Open).
assert_eq "jobs[$JOB_ID].state is Fulfilled" "$(job_field "$JOB_ID" $JOB_FIELD_STATE)" "2"

step "the escrow moved"
coordinator_after=$(cast balance --rpc-url "$RPC" "$coordinator")
# An exact identity, not "it went up". The coordinator paid for the transaction
# that collected the escrow, so the only balance change that is consistent with
# `docs/bridge.md` §2 is escrow minus that exact gas bill; anything else means
# money moved somewhere this script is not looking.
expected_after=$(bigint "$coordinator_before + $ESCROW_WEI - $gas_used * $gas_price")
assert_eq "coordinator balance after fulfillJob" "$coordinator_after" "$expected_after"
assert_eq "jobs[$JOB_ID].escrow" "$(job_field "$JOB_ID" $JOB_FIELD_ESCROW)" "0"

step "the consumer callback fired"
assert_eq "tally.resultCommit()" "$(call "$TALLY" "resultCommit()(bytes32)")" "$result_hash"

# --- the key holder, again ---------------------------------------------------

step "decrypting the blob the chain carries"
# The chain's copy, not the coordinator's. `docs/bridge.md` §5a's whole argument
# for emitting the blob is that the key holder then gets the bytes a quorum
# attested to rather than whatever the coordinator chose to hand over, and the
# only way to demonstrate that is to fetch it back out of the event.
write_hex "$event_blob" "$work/from-chain.blob"
cmp -s "$work/from-chain.blob" "$work/result.blob" ||
  die "the blob the chain emitted differs from the one the coordinator wrote"
note "the chain's blob is byte-identical to the coordinator's"

if [ "$NETWORK" = real ]; then
  # Asserting the *value* matters. Every check above is about workers agreeing
  # and a contract counting signatures, and workers agreeing on the wrong answer
  # would satisfy all of them — `docs/attestation.md` §1 is explicit that
  # nothing downstream can tell a wrong plaintext from a right one. This is the
  # only place in the run that can notice.
  IFS=',' read -ra values <<< "$INPUTS"
  expected=${EXPECT:-$(printf '%s\n' "${values[@]}" | sort -n | tail -1)}
  decrypted=$("$CLI" decrypt \
    --client-key "$work/keys/client.key" \
    --server-key "$work/keys/server.key" \
    --input "$work/from-chain.blob")
  assert_eq "key holder decrypted the chain's ciphertext" "$decrypted" "$expected"

  step "the committee reveals"
  send "$COMMITTEE_KEY" "$TALLY" "reveal(uint32)" "$decrypted"
  assert_eq "tally.winner()" "$(call "$TALLY" "winner()(uint32)")" "$decrypted"
  assert_eq "tally.revealed()" "$(call "$TALLY" "revealed()(bool)")" "true"
  note "trusted: nothing on-chain can check this number against resultCommit."
  note "docs/bridge.md §5 marks the reveal as the demo's one explicit trust"
  note "boundary; proving it needs verifiable decryption or a threshold KMS."
else
  note "synthetic mode: the blob is random bytes, so there is nothing to decrypt"
fi

step "a settled job cannot be settled again, or refunded"
send_must_revert "second fulfillJob" "$COORDINATOR_KEY" "$BRIDGE" \
  "fulfillJob(uint256,bytes32,bytes,(bytes32,bytes32,uint8)[])" \
  "$JOB_ID" "$result_hash" "$result_blob_hex" "$quorum_arg"
send_must_revert "refundOnTimeout on a fulfilled job" "$BYSTANDER_KEY" "$BRIDGE" \
  "refundOnTimeout(uint256)" "$JOB_ID"

# --- pass two: nobody settles ------------------------------------------------
#
# `docs/bridge.md` §6 routes every liveness failure here — coordinator silence,
# a withheld result, and workers who never agree all end in the same place,
# because none of them produce a quorum and the contract cannot tell them apart.

step "pass two: a job nobody fulfils"
# Posted by an EOA, deliberately, and *not* through CommitteeTally. A job posted
# by the tally contract cannot be refunded at all: `refundOnTimeout` sends the
# escrow with `poster.call{value: escrow}("")`, `CommitteeTally` has no
# `receive` or `fallback`, the transfer fails, and the whole refund reverts on
# `EscrowTransferFailed` — leaving the escrow stuck forever in a job that stays
# Open. See `bridge/test/RefundToContractPoster.t.sol`; §6 promises a refund for
# exactly this case and the demo consumer cannot take one.
second_escrow=$(bigint "$ESCROW_WEI // 2")
poster_before=$(cast balance --rpc-url "$RPC" "$poster")
send "$POSTER_KEY" --value "$second_escrow" "$BRIDGE" \
  "submitJob(uint256,bytes32[],bytes[],address)" \
  "$PROGRAM_ID" "$commit_array" "$blob_array" "0x0000000000000000000000000000000000000000"
submit_gas=$(bigint "$(receipt_field .gasUsed) * $(receipt_field .effectiveGasPrice)")
SECOND_JOB=$(call "$BRIDGE" "jobCount()(uint256)")
assert_eq "the second job is a new job" "$SECOND_JOB" "$(bigint "$JOB_ID + 1")"

if [ "$NETWORK" = real ] && [ -z "${HONEST:-}" ]; then
  note "re-running the same three workers at ${#WORKER_IDS[@]}-of-${#WORKER_IDS[@]}."
  note "The liar was outvoted at $ATTESTERS-of-${#WORKER_IDS[@]}; under unanimity it is"
  note "decisive, so no quorum can form. Nothing is staged but the fault."
  if run_coordinator "${#WORKER_IDS[@]}" "$work/coordinator-2.log"; then
    die "the coordinator reached unanimity with a faulty worker in the set.
Either fault injection did nothing, or agreement is not being counted per
attester -- both of which would make every 'job settled' above meaningless."
  fi
  note "the coordinator failed, as it must: $(strip_ansi < "$work/coordinator-2.log" | tail -1)"
else
  note "no coordinator is run against this job at all — the §6 'coordinator goes"
  note "silent' row, which is the same row as every other liveness failure."
fi

assert_eq "jobs[$SECOND_JOB].state is still Open" \
  "$(job_field "$SECOND_JOB" $JOB_FIELD_STATE)" "1"
assert_eq "jobs[$SECOND_JOB].poster" "$(job_field "$SECOND_JOB" $JOB_FIELD_POSTER)" "$poster"

step "refundOnTimeout"
send_must_revert "refundOnTimeout before the deadline" "$BYSTANDER_KEY" "$BRIDGE" \
  "refundOnTimeout(uint256)" "$SECOND_JOB"

# Anvil's clock, rather than $JOB_TIMEOUT seconds of waiting. Aimed at the
# deadline the contract stored rather than advanced by a duration: the two are
# the same only if no other block has moved the clock in between, and getting
# that wrong lands as JobNotExpired, which reads like a contract bug.
second_deadline=$(job_field "$SECOND_JOB" $JOB_FIELD_DEADLINE)
cast rpc --rpc-url "$RPC" evm_setNextBlockTimestamp "$((second_deadline + 1))" > /dev/null ||
  die "evm_setNextBlockTimestamp failed"
cast rpc --rpc-url "$RPC" evm_mine > /dev/null || die "evm_mine failed"
now=$(cast block --rpc-url "$RPC" latest --field timestamp)
[ "$now" -gt "$second_deadline" ] ||
  die "the chain clock is $now and the deadline is $second_deadline; the warp did nothing"
note "warped to $now, one second past the job's $second_deadline deadline"

# Called by a bystander, not the poster: `refundOnTimeout` is permissionless
# because the money can only go to the poster, and a poster paying the gas would
# muddy the balance identity below.
send "$BYSTANDER_KEY" "$BRIDGE" "refundOnTimeout(uint256)" "$SECOND_JOB"

assert_eq "jobs[$SECOND_JOB].state is Refunded" \
  "$(job_field "$SECOND_JOB" $JOB_FIELD_STATE)" "3"
# The poster is out exactly the gas it spent posting, and nothing else.
assert_eq "poster balance after the refund" "$(cast balance --rpc-url "$RPC" "$poster")" \
  "$(bigint "$poster_before - $submit_gas")"
send_must_revert "fulfillJob on a refunded job" "$COORDINATOR_KEY" "$BRIDGE" \
  "fulfillJob(uint256,bytes32,bytes,(bytes32,bytes32,uint8)[])" \
  "$SECOND_JOB" "$result_hash" "$result_blob_hex" "$quorum_arg"

printf '\n== done. %d steps, no assertion skipped.\n' "$step_no"
printf '   attestation source: %s\n' "$source_label"
if [ "$NETWORK" = synthetic ]; then
  printf '   synthetic: the contracts and the script were exercised; the FHE was not.\n'
fi
