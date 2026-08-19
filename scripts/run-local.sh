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
echo "building node (with fault-injection)..."
cargo build --release -p node --features fault-injection

NODE=target/release/node

# A worker left over from an earlier run would still be bound to one of these
# ports, and the new one would silently fail to start -- leaving the old
# process, with an old server key, answering this run's jobs.
for port in 8080 8081 8082 8083; do
  lsof -ti "tcp:$port" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
done

pids=()
cleanup() { for pid in "${pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT

third_worker_flags=(--id worker-3 --bind 127.0.0.1:8083)
if [ -z "${HONEST:-}" ]; then
  third_worker_flags+=(--faulty)
fi

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
  --attesters "$ATTESTERS" \
  --program "$PROGRAM" \
  --function "$FUNCTION" \
  --inputs "$INPUTS" \
  --deadline-secs "$DEADLINE"
