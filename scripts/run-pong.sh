#!/usr/bin/env bash
# Pong, at four paces, from one piece of source.
#
# `ping-pong/lib.rs` is built two ways -- as an `rlib` the terminal player links
# natively, and as a `cdylib` that becomes `ping_pong.wasm`. The browser page
# runs that wasm directly, and `disca-cli compile` lowers the *same* wasm to the
# DISCA bytecode three workers evaluate under encryption. There is one set of
# rules here, not four that have to be kept in step, and `circuit` below shows
# the lowering rather than asserting it.
#
#   ./scripts/run-pong.sh rally     # 8 fps  -- a watchable rally
#   ./scripts/run-pong.sh step      # 1 fps  -- slow enough to read the state
#   ./scripts/run-pong.sh web       # the browser page, all three paces, one wasm
#   ./scripts/run-pong.sh circuit   # compile to wasm, emit the lowered circuit
#   ./scripts/run-pong.sh disca     # THE REAL ONE: FHE, 3 workers, 2-of-3 quorum
#
# `disca` is not a simulation of the encrypted version. It generates a real
# keypair, encrypts the game state, and every frame dispatches six jobs to three
# worker processes that evaluate homomorphically and sign what they got. The
# coordinator settles each on a 2-of-3 byte-identical quorum. Expect ~33 s per
# frame; that is the point of the other three modes existing.
set -euo pipefail

cd "$(dirname "$0")/.."

MODE=${1:-help}
FRAMES=${FRAMES:-3}
ATTESTERS=${ATTESTERS:-2}

# The opening state, identical in every mode. Order is load-bearing and matches
# `lib.rs`: ball_x, ball_y, vel_x, vel_y, paddle_y, score.
STATE=${STATE:-480,270,-80,45,270,0}
FUNCS=(ball_x ball_y vel_x vel_y paddle_y score)

case "$MODE" in
  rally)
    cargo build --release -q -p ping-pong --bin pong
    exec ./target/release/pong --fps 8
    ;;

  step)
    cargo build --release -q -p ping-pong --bin pong
    exec ./target/release/pong --fps 1
    ;;

  web)
    # The page embeds the wasm as base64 so it works from disk with no server.
    # Rebuild and re-embed first, so what you watch is what lib.rs currently says.
    ( cd ping-pong && cargo build --release -q -p ping-pong --lib )
    cp target/wasm32-unknown-unknown/release/ping_pong.wasm ping-pong/
    python3 ping-pong/web/embed.py
    echo "opening ping-pong/web/index.html -- keys 1/2/3 switch pace"
    if command -v open >/dev/null; then open ping-pong/web/index.html
    elif command -v xdg-open >/dev/null; then xdg-open ping-pong/web/index.html
    else echo "open ping-pong/web/index.html yourself"; fi
    ;;

  circuit)
    # Rust -> wasm -> DISCA bytecode, and then what the bytecode actually is.
    # This is the step that makes "the same code" checkable: the six functions
    # the browser calls are the six circuits printed here.
    ( cd ping-pong && cargo build --release -q -p ping-pong --lib )
    cp target/wasm32-unknown-unknown/release/ping_pong.wasm ping-pong/
    cargo build --release -q -p disca-cli
    echo
    ./target/release/disca-cli compile \
      --input ping-pong/ping_pong.wasm --output /tmp/ping_pong.bytecode
    echo
    cargo run --release -q -p primitives --example inspect -- ping-pong/ping_pong.wasm
    ;;

  disca)
    # ---- the real encrypted game -------------------------------------------
    #
    # No mocking anywhere below: real FHE keys, real homomorphic evaluation on
    # three separate worker processes, real secp256k1 attestations, and a real
    # M-of-N quorum before any frame is allowed to advance.
    echo "building node and disca-cli (release)..."
    cargo build --release -q -p node -p disca-cli
    ( cd ping-pong && cargo build --release -q -p ping-pong --lib )
    cp target/wasm32-unknown-unknown/release/ping_pong.wasm ping-pong/

    NODE=target/release/node
    CLI=target/release/disca-cli
    work=$(mktemp -d "${TMPDIR:-/tmp}/disca-pong.XXXXXX")

    for port in 8080 8081 8082 8083; do
      lsof -ti "tcp:$port" 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    done
    pids=()
    # KEEP=1 preserves the run directory -- keys, ciphertexts, every frame's
    # result blobs and all four logs -- so a settled frame can be re-decrypted
    # or an attestation re-checked after the fact. It is deleted by default
    # because it holds a client key.
    KEEP=${KEEP:-0}
    cleanup() {
      for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null || true; done
      if [ "$KEEP" = 1 ]; then echo "run kept: $work (holds a client key)"
      else rm -rf "$work"; fi
    }
    trap cleanup EXIT

    echo "key holder: generating keys (the client key never leaves $work)..."
    "$CLI" keygen --out-dir "$work/keys" --force >/dev/null

    echo "key holder: compiling ping_pong.wasm to DISCA bytecode..."
    "$CLI" compile --input ping-pong/ping_pong.wasm --output "$work/pong.bytecode"

    echo "key holder: encrypting the opening state ($STATE)..."
    "$CLI" encrypt --client-key "$work/keys/client.key" \
      --values "$STATE" --out-dir "$work/f0" >/dev/null
    # Frame 0's six ciphertexts. Every later frame's inputs are the *previous*
    # frame's result blobs -- `wire::seal_result` is `encode(&compress())`, which
    # is exactly what `wire::decode` reads, so encrypted state flows forward
    # without ever being decrypted (`ping-pong/lib.rs`, module docs).
    cur=()
    for i in 0 1 2 3 4 5; do cur+=("$work/f0/input-$i.ct"); done

    registry=()
    for id in worker-1 worker-2 worker-3; do
      registry+=(--registered-worker "$(RUST_LOG=off "$NODE" worker-address --id "$id")")
    done

    echo "starting three workers..."
    "$NODE" worker --id worker-1 --bind 127.0.0.1:8081 >"$work/w1.log" 2>&1 & pids+=($!)
    "$NODE" worker --id worker-2 --bind 127.0.0.1:8082 >"$work/w2.log" 2>&1 & pids+=($!)
    "$NODE" worker --id worker-3 --bind 127.0.0.1:8083 >"$work/w3.log" 2>&1 & pids+=($!)
    await_port() {
      for _ in $(seq 1 100); do nc -z 127.0.0.1 "$1" 2>/dev/null && return 0; sleep 0.2; done
      echo "nothing listening on $1 after 20s; see $work/*.log" >&2
      # Surface the reason rather than the symptom: a worker that cannot bind
      # has already said why, and that line is the whole diagnosis.
      tail -n 3 "$work"/*.log >&2 2>/dev/null || true
      exit 1
    }
    for port in 8081 8082 8083; do await_port "$port"; done

    echo "starting the coordinator (serving; $ATTESTERS-of-3)..."
    "$NODE" coordinator --serve --bind 127.0.0.1:8080 \
      --worker 127.0.0.1:8081 --worker 127.0.0.1:8082 --worker 127.0.0.1:8083 \
      "${registry[@]}" --attesters "$ATTESTERS" \
      --server-key "$work/keys/server.key" --bytecode "$work/pong.bytecode" \
      --deadline-secs 300 >"$work/coord.log" 2>&1 & pids+=($!)
    await_port 8080
    echo

    for (( f=0; f<FRAMES; f++ )); do
      out="$work/f$((f+1))"; mkdir -p "$out"
      began=$(date +%s)

      # One frame is six independent jobs over the same six ciphertexts, so they
      # are submitted together rather than in sequence -- which is exactly why
      # `coordinator --serve` exists.
      inputs=(); for c in "${cur[@]}"; do inputs+=(--input "$c"); done
      subs=()
      for fn in "${FUNCS[@]}"; do
        "$NODE" submit --coordinator 127.0.0.1:8080 --function "$fn" \
          "${inputs[@]}" --result "$out/$fn.blob" --timeout-secs 300 \
          >>"$work/submit.log" 2>&1 & subs+=($!)
      done
      for p in "${subs[@]}"; do wait "$p"; done

      # Decrypt only to draw. The workers never saw any of these numbers.
      vals=()
      for fn in "${FUNCS[@]}"; do
        vals+=("$("$CLI" decrypt --client-key "$work/keys/client.key" \
                    --server-key "$work/keys/server.key" --input "$out/$fn.blob")")
      done
      took=$(( $(date +%s) - began ))

      printf 'frame %d  (%ds)  ball (%s,%s)  vel (%s,%s)  paddle %s  saves %s\n' \
        "$((f+1))" "$took" "${vals[0]}" "${vals[1]}" "${vals[2]}" \
        "${vals[3]}" "${vals[4]}" "${vals[5]}"

      # Frame N's outputs are frame N+1's inputs, as ciphertext.
      cur=(); for fn in "${FUNCS[@]}"; do cur+=("$out/$fn.blob"); done
    done

    # Evidence, printed here because the run directory is about to be removed.
    #
    # `coordinator --serve` does not log "job settled" -- that message belongs to
    # the one-shot `run()` path. Here settlement is carried by the response: a
    # job that fails to reach quorum answers 409 and writes no blob, which makes
    # `submit --result` exit non-zero and abort this script under `set -e`. So
    # the six blobs below are the settlement, not a claim about it, and the
    # dispatch count is the fan-out that had to agree to produce them.
    expected_jobs=$((FRAMES * 6))
    submitted=$(grep -c "job submitted" "$work/coord.log" || true)
    dispatched=$(grep -c "dispatched" "$work/coord.log" || true)

    blobs=0
    for (( f=1; f<=FRAMES; f++ )); do
      for fn in "${FUNCS[@]}"; do
        [ -s "$work/f$f/$fn.blob" ] && blobs=$((blobs + 1))
      done
    done

    echo
    echo "submitted:  ${submitted:-0} of $expected_jobs jobs"
    echo "dispatched: ${dispatched:-0} worker-jobs (each job goes to all 3)"
    echo "settled:    $blobs of $expected_jobs result blobs, each on a $ATTESTERS-of-3 byte-identical quorum"

    if [ "$blobs" -ne "$expected_jobs" ]; then
      echo "expected $expected_jobs settled jobs, got $blobs" >&2
      exit 1
    fi
    echo
    echo "every frame above was computed on ciphertext; the workers saw none of those numbers."
    ;;

  *)
    sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
esac
