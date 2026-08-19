#!/usr/bin/env bash
# Measures line coverage with cargo-llvm-cov, and enforces a floor per crate.
#
# CI and a developer's laptop run this same script, so a number that passes
# locally cannot fail in CI for a reason nobody can reproduce.
#
#   scripts/coverage.sh              # summary + HTML report
#   scripts/coverage.sh --open       # ... and open the HTML report
#   scripts/coverage.sh --check      # ... and fail if a crate is below its floor
#   scripts/coverage.sh --lcov FILE  # ... and write lcov for an external tool
#
# Install the tool once with:  cargo install cargo-llvm-cov
set -euo pipefail

cd "$(dirname "$0")/.."

open_report=0
check=0
lcov_path=""
while [ $# -gt 0 ]; do
  case "$1" in
    --open) open_report=1 ;;
    --check) check=1 ;;
    --lcov)
      lcov_path="${2:?--lcov needs a path}"
      shift
      ;;
    -h | --help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if ! cargo llvm-cov --version >/dev/null 2>&1; then
  echo "cargo-llvm-cov is not installed. Install it with:" >&2
  echo "    cargo install cargo-llvm-cov" >&2
  exit 1
fi

# --all-features is a no-op on `main` as it stands: nothing in the workspace is
# optional yet. It is here because task 2.11b asks for it, and because the
# branch that gates `logic_gates` behind `boolean-circuits` and `--faulty`
# behind `fault-injection` is what makes it load-bearing. Code behind an
# off-by-default feature is never compiled by an ordinary build, so nothing
# type-checks it and no test runs it — it rots while the coverage number, which
# does not know it exists, stays flattering.
COVER_ARGS=(
  --workspace
  --all-features
)

# The two demo-circuit crates are filtered out of the report rather than
# measured. They are Rust only so that rustc can emit WASM from them: the
# committed .wasm binaries are what `primitives/tests/tally_circuit.rs` actually
# evaluates, so the host-built copies of `max2`, `tally4_select` and friends are
# unreachable by construction and no honest test can execute them. Counting them
# would put 45 uncoverable lines in the denominator and say nothing about
# whether the tally works.
#
# Examples are excluded for the same reason in reverse: `size_probe` and
# `inspect` are run by hand to measure or inspect something. They are a record
# of a measurement, not behaviour under test. (cargo-llvm-cov does not collect
# them today; the filter is here so that if it starts, the number does not
# lurch for a reason nobody can explain.)
REPORT_ARGS=(
  --ignore-filename-regex
  '((committee-tally|simple-arithmetic)/lib\.rs|/examples/)'
)

echo "==> running the test suite under instrumentation"
# Note for whoever touches this next: primitives/tests/determinism_under_concurrency.rs
# re-executes the test binary as concurrent children. That is safe under
# instrumentation because cargo-llvm-cov puts %p (the pid) in LLVM_PROFILE_FILE,
# so each child writes its own .profraw instead of racing the parent for one
# file, and the children exit through libc `exit()`, which still flushes the
# profile. Their coverage is merged in with everyone else's.
cargo llvm-cov --no-report "${COVER_ARGS[@]}"

# lcov rather than the JSON summary, because the summary is per file and the
# report needs per line: it drops the lines inside inline `#[cfg(test)]`
# modules, which llvm-cov counts as covered code when they are the code doing
# the covering. See scripts/lib/coverage_report.py.
# Spelled-out template: GNU mktemp rejects a bare `-t prefix` and requires at
# least three X's, where BSD mktemp on macOS accepts it. See check-deps.sh.
lcov="$(mktemp "${TMPDIR:-/tmp}/disca-coverage.XXXXXX")"
trap 'rm -f "$lcov"' EXIT

cargo llvm-cov report "${REPORT_ARGS[@]}" --lcov --output-path "$lcov"
cargo llvm-cov report "${REPORT_ARGS[@]}" --html >/dev/null
if [ -n "$lcov_path" ]; then
  cp "$lcov" "$lcov_path"
  echo "lcov written to $lcov_path"
fi

gate=()
if [ "$check" -eq 1 ]; then
  gate=(--check)
fi

echo
status=0
python3 scripts/lib/coverage_report.py "$lcov" "${gate[@]}" || status=$?

html="${CARGO_TARGET_DIR:-target}/llvm-cov/html/index.html"
echo
echo "HTML report: $html"
if [ "$open_report" -eq 1 ]; then
  case "$(uname -s)" in
    Darwin) open "$html" ;;
    *) xdg-open "$html" >/dev/null 2>&1 || true ;;
  esac
fi

exit "$status"
