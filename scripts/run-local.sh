#!/usr/bin/env bash
# Runs one coordinator and three workers as local processes, 2-of-3 attestation.
#
# The third worker is deliberately faulty. A job where every worker agrees is
# indistinguishable from a job with no verification at all -- you see
# "job settled" either way -- so the run only demonstrates something if one
# worker disagrees and is outvoted.
#
# Both modes are needed to conclude anything. A detector that never fires and
# one that always fires would each pass a single test; it is the pair that says
# the mechanism tracks reality:
#
#   ./scripts/run-local.sh            # one faulty worker -> detected, outvoted
#   HONEST=1 ./scripts/run-local.sh   # all honest -> settles, nobody accused
#   ATTESTERS=3 ./scripts/run-local.sh  # unanimity vs one liar -> fails, correctly
#
# Only the fault is staged. The two honest workers are never told to agree --
# they are separate processes that never talk to each other, and they match
# because they independently computed the same ciphertext byte for byte.
set -euo pipefail

cd "$(dirname "$0")/.."

PROGRAM=${PROGRAM:-committee-tally/committee_tally.wasm}
FUNCTION=${FUNCTION:-tally4_select}
INPUTS=${INPUTS:-71,93,42,88}
ATTESTERS=${ATTESTERS:-2}
DEADLINE=${DEADLINE:-120}

if [ ! -f "$PROGRAM" ]; then
  echo "missing $PROGRAM — build it with: (cd committee-tally && cargo build --release)" >&2
  exit 1
fi

# fault-injection is off by default, so a release build cannot be told to
# return a wrong answer. This demo needs exactly that, so it opts in explicitly.
echo "building node (with fault-injection) and disca-cli..."
cargo build --release -p node --features fault-injection
cargo build --release -p disca-cli

NODE=target/release/node
CLI=target/release/disca-cli

# Everything the key holder produces lives here and is thrown away afterwards.
# The client key never leaves this directory, and no other process in this
# script is given a path to it.
work=$(mktemp -d "${TMPDIR:-/tmp}/disca-demo.XXXXXX")

# A worker left over from an earlier run would still be bound to one of these
# ports, and the new one would silently fail to start -- leaving the old
# process, with an old server key, answering this run's jobs.
for port in 8080 8081 8082 8083; do
  lsof -ti "tcp:$port" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
done

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  rm -rf "$work"
}
trap cleanup EXIT

# --- the key holder -----------------------------------------------------
#
# `disca-cli` is a separate process here for the same reason it is a separate
# party in the design: it is the only thing that ever holds the client key or
# sees a plaintext. The coordinator below is started with paths to a public key
# and to ciphertexts, and could not decrypt the result it settles on even if it
# wanted to.
echo "key holder: generating keys..."
"$CLI" keygen --out-dir "$work/keys" --force

echo "key holder: compiling $PROGRAM..."
"$CLI" compile --input "$PROGRAM" --output "$work/program.bytecode"

echo "key holder: encrypting inputs..."
"$CLI" encrypt --client-key "$work/keys/client.key" \
  --values "$INPUTS" --out-dir "$work/inputs"

# The dispatch takes them in argument order, and nothing downstream can notice
# a transposition: the inputs are ciphertext, so the wrong order produces a
# plausible answer to a different question.
input_flags=()
IFS=',' read -ra values <<< "$INPUTS"
for i in "${!values[@]}"; do
  input_flags+=(--input "$work/inputs/input-$i.ct")
done

third_worker_flags=(--id worker-3 --bind 127.0.0.1:8083)
if [ -z "${HONEST:-}" ]; then
  third_worker_flags+=(--faulty)
fi

# Each worker signs its attestation with a secp256k1 key (task 2.10i) and the
# coordinator counts agreement over the addresses it recovers -- but only for
# addresses in its registry, so it has to be told them up front. None of these
# workers is given a --key, so each derives one from its --id; `worker-address`
# computes the same address without starting a worker, which is how the
# coordinator learns what to accept.
#
# RUST_LOG=off because this captures stdout: a stray log line would be
# substituted into --registered-worker and rejected as a malformed address.
# Nothing here is a secret -- the ids are public, so the keys are too, which is
# exactly why a deployment passes --key instead.
worker_address() { RUST_LOG=off "$NODE" worker-address --id "$1"; }
registry=()
for id in worker-1 worker-2 worker-3; do
  registry+=(--registered-worker "$(worker_address "$id")")
done

"$NODE" worker --id worker-1 --bind 127.0.0.1:8081 & pids+=($!)
"$NODE" worker --id worker-2 --bind 127.0.0.1:8082 & pids+=($!)
"$NODE" worker "${third_worker_flags[@]}" & pids+=($!)

# Let the workers bind before the coordinator fans out to them.
until nc -z 127.0.0.1 8081 && nc -z 127.0.0.1 8082 && nc -z 127.0.0.1 8083; do
  sleep 0.2
done

"$NODE" coordinator \
  --worker 127.0.0.1:8081 \
  --worker 127.0.0.1:8082 \
  --worker 127.0.0.1:8083 \
  "${registry[@]}" \
  --attesters "$ATTESTERS" \
  --server-key "$work/keys/server.key" \
  --bytecode "$work/program.bytecode" \
  --function "$FUNCTION" \
  "${input_flags[@]}" \
  --result "$work/result.blob" \
  --attestations "$work/attestations.json" \
  --deadline-secs "$DEADLINE"

# The evidence beside the answer. There is no chain in this run, so nothing
# consumes the file here -- but writing it is the cheapest way to keep the
# format honest: it is produced from real signatures over a real settlement,
# and `fulfillJob` requires the attesters in strictly increasing address order
# (bridge.md §2a step 4), which is a property no unit test can observe about
# *this* run's addresses.
attesters=$(grep -c '"address"' "$work/attestations.json")
if [ "$attesters" -lt "$ATTESTERS" ]; then
  echo "FAIL: attestations.json names $attesters attester(s), expected $ATTESTERS" >&2
  exit 1
fi
# LC_ALL=C so the comparison is byte order, which is what a contract comparing
# `address` values does. A locale-aware sort can disagree with it.
if ! grep -o '"address": "[^"]*"' "$work/attestations.json" | LC_ALL=C sort -c 2>/dev/null; then
  echo "FAIL: attestations.json is not in ascending address order; fulfillJob would revert" >&2
  exit 1
fi
echo "attestations: $attesters signature(s), ascending by address"

# --- the key holder again -----------------------------------------------
#
# The coordinator wrote a blob it cannot read. Only this step can, and only
# because it holds the client key -- which is the claim the whole system makes,
# reduced to something you can watch happen.
#
# Asserting the value matters: every earlier check in this script is about
# workers agreeing, and workers agreeing on the *wrong* answer would satisfy
# all of them. `attestation.md` §1 is explicit that nothing downstream can tell
# a wrong plaintext from a right one, so this is the only place the run can
# notice.
expected=${EXPECT:-$(printf '%s\n' "${values[@]}" | sort -n | tail -1)}
decrypted=$("$CLI" decrypt \
  --client-key "$work/keys/client.key" \
  --server-key "$work/keys/server.key" \
  --input "$work/result.blob")

if [ "$decrypted" != "$expected" ]; then
  echo "FAIL: key holder decrypted $decrypted, expected $expected" >&2
  exit 1
fi
echo "key holder decrypted: $decrypted (expected $expected)"
