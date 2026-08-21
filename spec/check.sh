#!/usr/bin/env bash
#
# Model-check every configuration of DiscaAttestation.tla and check each one
# against what it is supposed to do.
#
# Half of these configurations are supposed to FAIL. A spec whose only
# evidence is "TLC found no error" proves nothing about whether it was
# checking anything, so each counterexample configuration deliberately
# removes one decision the implementation made and asserts that TLC finds the
# trace that decision prevents. This script therefore fails in both
# directions: a configuration that should pass and does not, and a
# configuration that should produce a named counterexample and does not.
#
#   ./check.sh              check everything
#   ./check.sh MC_N3M2      check one configuration (name or file)
#
# Fetches tla2tools.jar into this directory if it is not already here. The
# jar is not committed (see .gitignore); it is 2.2 MB of somebody else's
# build output and the pinned checksum below is what makes downloading it
# equivalent to having it.

set -uo pipefail

cd "$(dirname "$0")" || exit 1

TLA_VERSION="v1.7.4"
JAR="tla2tools.jar"
JAR_URL="https://github.com/tlaplus/tlaplus/releases/download/${TLA_VERSION}/tla2tools.jar"
JAR_SHA256="936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88"
OUT="out"

# Every configuration, and what it must do.
#
#   pass                          TLC must report no error
#   fail:<Name>                   TLC must report <Name> violated
#   fail:temporal                 TLC must report a temporal property violated
#
# Keep the comment in each .cfg file and its entry here in agreement; the .cfg
# says why, this says what.
EXPECTATIONS=(
  # --- the shipping shapes, everything must hold -------------------------
  "MC_N3M2                       pass"
  "MC_N3M3                       pass"
  "MC_N4M2_tolerated             pass"
  "MC_N4M3                       pass"
  "MC_N4M2                       pass"
  "MC_Registry_N3M2              pass"
  "MC_UniqueJobId_N3M2           pass"
  "MC_FWW_N3M2                   pass"
  "MC_FWW_Late_N3M2              pass"
  "MC_LWW_UniqueId_N3M2          pass"
  "MC_SplitQuorumStillReal_N4M2  pass"

  # --- non-vacuity: the interesting states are reachable ------------------
  "MC_Fulfil_witness             fail:NeverFulfils"
  "MC_Refund_witness             fail:NeverRefunds"
  "MC_Split_witness              fail:NoSplitEver"
  "MC_Hangs_witness              fail:temporal"

  # --- counterexamples: each removes one decision the code made ----------
  "MC_NoSplitRefusal_N4M2        fail:NoSettleOnSplit"
  "MC_SplitCost_N4M2             fail:SplitSettlementWasCorrect"
  "MC_GraceRace_N4M2             fail:ResultIsCorrect"
  "MC_LWW_N3M2                   fail:VoteNotDisplaced"
  "MC_LWW_Settles_N3M2           fail:ResultIsCorrect"
  "MC_ReplayPreempt_N3M2         fail:ResultIsCorrect"
  "MC_LWW_Equivocation_N3M2      fail:VoteNotDisplaced"
  "MC_Replay_N3M2                fail:QuorumIsReal"
  "MC_ReplayNobodyRan_N3M2       fail:SomeoneActuallyEvaluated"
  "MC_Sybil_N3M2                 fail:QuorumIsReal"
  "MC_UnguardedRefund_N3M2       fail:EscrowPaidOnce"

  # --- liveness -----------------------------------------------------------
  "MC_Liveness_N3M2              pass"
  "MC_Liveness_N4M3              pass"
  "MC_Fulfils_N3M2               pass"
  "MC_Fulfils_N4M3               pass"
)

# macOS ships `shasum`, most Linux images ship `sha256sum`, and the GitHub
# ubuntu runner happens to have both. Do not assume either.
sha256_of() {
  if command -v sha256sum > /dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

fetch_jar() {
  if [ -f "$JAR" ]; then
    actual=$(sha256_of "$JAR")
    if [ "$actual" = "$JAR_SHA256" ]; then
      return 0
    fi
    echo "note: $JAR does not match the pinned checksum; refetching" >&2
    rm -f "$JAR"
  fi

  echo "fetching tla2tools.jar ${TLA_VERSION}..." >&2
  if ! curl -sSfL -o "$JAR" "$JAR_URL"; then
    echo "error: cannot download $JAR_URL" >&2
    return 1
  fi

  actual=$(sha256_of "$JAR")
  if [ "$actual" != "$JAR_SHA256" ]; then
    echo "error: $JAR checksum mismatch" >&2
    echo "  expected $JAR_SHA256" >&2
    echo "  got      $actual" >&2
    rm -f "$JAR"
    return 1
  fi
}

# TLC's own exit codes are not granular enough to distinguish "the invariant I
# expected" from "some other invariant", so the report is read from stdout.
run_one() {
  local name="$1" expect="$2" log="$OUT/$name.log"

  java -XX:+UseParallelGC -cp "$JAR" tlc2.TLC \
      -config "$name.cfg" -workers auto -metadir "$OUT/states-$name" \
      -cleanup DiscaAttestation.tla > "$log" 2>&1

  local states diameter summary
  states=$(grep -oE '[0-9]+ distinct states found' "$log" | tail -1 | cut -d' ' -f1)
  diameter=$(grep -oE 'search is [0-9]+' "$log" | tail -1 | cut -d' ' -f3)
  [ -n "$states" ] || states="?"
  [ -n "$diameter" ] || diameter="?"

  if grep -q "^Error: Parsing or semantic analysis failed" "$log" \
     || grep -q "^\*\*\* Parse Error" "$log"; then
    printf '  %-30s BROKEN   the module or config does not parse\n' "$name"
    return 1
  fi

  case "$expect" in
    pass)
      if grep -q "Model checking completed. No error has been found." "$log"; then
        printf '  %-30s ok       %8s states, diameter %s\n' "$name" "$states" "$diameter"
        return 0
      fi
      summary=$(grep -m1 "^Error:" "$log")
      printf '  %-30s FAILED   expected no error, got: %s\n' "$name" "${summary:-unknown}"
      return 1
      ;;
    fail:temporal)
      if grep -q "Temporal properties were violated" "$log"; then
        printf '  %-30s ok       counterexample found (temporal)\n' "$name"
        return 0
      fi
      printf '  %-30s FAILED   expected a temporal violation and none was found\n' "$name"
      return 1
      ;;
    fail:*)
      local want="${expect#fail:}"
      if grep -q "Invariant $want is violated" "$log"; then
        # TLC halts at the first violation, so `states` here is what it had
        # explored when it stopped, not the size of the state space.
        printf '  %-30s ok       %s violated by a %s-state trace (%s states explored)\n' \
               "$name" "$want" "$(grep -cE '^State [0-9]+:' "$log")" "$states"
        return 0
      fi
      if grep -q "Model checking completed. No error has been found." "$log"; then
        printf '  %-30s FAILED   expected %s to be violated; TLC found no error\n' \
               "$name" "$want"
        return 1
      fi
      summary=$(grep -m1 "^Error:" "$log")
      printf '  %-30s FAILED   expected %s violated, got: %s\n' \
             "$name" "$want" "${summary:-unknown}"
      return 1
      ;;
  esac
}

fetch_jar || exit 1

# `make jar`: fetch and stop, so a CI job can warm the download in a step of
# its own and have the failure say "download" rather than "model check".
if [ "${1:-}" = "--jar-only" ]; then
  exit 0
fi

mkdir -p "$OUT"

only="${1:-}"
only="${only%.cfg}"
only="${only##*/}"

failures=0
ran=0
echo "checking DiscaAttestation.tla with TLC ${TLA_VERSION}"
for entry in "${EXPECTATIONS[@]}"; do
  # shellcheck disable=SC2086
  set -- $entry
  name="$1"
  expect="$2"

  if [ -n "$only" ] && [ "$only" != "$name" ]; then
    continue
  fi
  if [ ! -f "$name.cfg" ]; then
    printf '  %-30s MISSING  no such configuration\n' "$name"
    failures=$((failures + 1))
    continue
  fi

  ran=$((ran + 1))
  run_one "$name" "$expect" || failures=$((failures + 1))
done

if [ "$ran" -eq 0 ]; then
  echo "error: no configuration matched '${only}'" >&2
  exit 1
fi

echo
if [ "$failures" -eq 0 ]; then
  echo "$ran configuration(s) checked, all as expected. Logs in spec/$OUT/."
  exit 0
fi
echo "$failures of $ran configuration(s) did not do what they are supposed to."
echo "Full TLC output, counterexample traces included, is in spec/$OUT/."
exit 1
