#!/usr/bin/env bash
# Dependency and lockfile hygiene. Offline and non-mutating by default.
#
# What this deliberately does NOT do is run `cargo update`. Byte-reproducible
# evaluation depends on an exact `tfhe` pin (see docs/architecture.md §3 and
# `pin_fft_plan` in node/src/main.rs); a hook that rewrote Cargo.lock on every
# commit would be a machine for silently breaking the one property M-of-N
# attestation rests on. So the checks here verify the pin instead of moving it:
#
#   1. Cargo.lock is in sync with the manifests   (`cargo metadata --locked`)
#   2. every dependency on `tfhe` is pinned exactly ("=x.y.z", not "^x.y")
#   3. exactly one `tfhe` resolves in the graph — two would mean two evaluators
#
# Reporting what *could* be updated is a separate, opt-in mode that still writes
# nothing:
#
#   scripts/check-deps.sh                  # gating checks, offline, fast
#   scripts/check-deps.sh --report         # + `cargo update --dry-run` (network)
#   scripts/check-deps.sh --audit          # + `cargo audit` advisories (network)
#
set -euo pipefail

cd "$(dirname "$0")/.."

report=0
audit=0
for arg in "$@"; do
  case "$arg" in
    --report) report=1 ;;
    --audit) audit=1 ;;
    -h | --help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

# `cargo metadata --locked` resolves the whole graph and fails rather than
# writing when Cargo.lock would have to change. That is the check a hook wants:
# it catches a manifest edit committed without its lockfile update, and it is
# the same flag CI builds with, so the two cannot disagree.
echo "==> checking Cargo.lock is in sync with the manifests"
metadata=$(cargo metadata --locked --format-version 1) || {
  echo >&2
  echo "Cargo.lock is out of date with Cargo.toml." >&2
  echo "Run 'cargo metadata' (no --locked) to refresh it, review the diff, and" >&2
  echo "commit Cargo.lock alongside the manifest change." >&2
  exit 1
}

echo "==> checking the consensus-critical pins are intact"
printf '%s' "$metadata" | python3 "$(dirname "$0")/lib/check_pins.py"

if [ "$report" -eq 1 ]; then
  # `--dry-run` prints what a `cargo update` would do and writes nothing, which
  # is the useful half of the command without the half that breaks the pin.
  # Guarded so an accidental lockfile write shows up as a failure rather than a
  # commit nobody meant to make.
  echo
  echo "==> available updates (nothing is written)"
  before="$(mktemp -t disca-lock)"
  cp Cargo.lock "$before"
  cargo update --dry-run --workspace || true
  if ! cmp -s Cargo.lock "$before"; then
    cp "$before" Cargo.lock
    rm -f "$before"
    echo "cargo update --dry-run modified Cargo.lock; it has been restored." >&2
    echo "Do not trust --dry-run on this cargo version." >&2
    exit 1
  fi
  rm -f "$before"
fi

if [ "$audit" -eq 1 ]; then
  echo
  echo "==> security advisories"
  if ! command -v cargo-audit >/dev/null 2>&1; then
    echo "cargo-audit is not installed; install it with 'cargo install cargo-audit'" >&2
    exit 1
  fi
  cargo audit
fi

echo
echo "dependency hygiene OK"
