"""Turns cargo-llvm-cov's lcov output into a per-crate table, and gates on it.

Two things this does that `cargo llvm-cov --fail-under-lines` does not.

**It does not count test code as covered code.** llvm-cov instruments inline
`#[cfg(test)] mod tests` blocks along with everything else, and those lines are
covered by definition — they are the thing doing the running. In a crate whose
source is mostly type definitions and whose tests are long, that is most of the
denominator: `node/src/protocol.rs` is 165 lines, of which 62 are tests, and
counting them takes the file from a real number to a flattering one. Writing a
longer test would then raise coverage without testing anything new, which is
precisely the incentive a coverage gate must not create. So every file is cut at
its first `#[cfg(test)]` and only the lines above it are counted. (That holds
for every file in this workspace: the test module is always last. If someone
puts one in the middle, this undercounts, which is the safe direction.)

**The floor is per crate, with a reason attached.** `primitives` is the
execution core — pure functions with a checkable answer, and the thing an
attestation is a claim *about* — so it is held high. `node` is largely process
orchestration: binding sockets, spawning threads, blocking on channels, and a
`main` that dispatches on a role. The parts with decisions in them (M-of-N
aggregation, the transport's size limits, argument parsing, the worker's
pre-evaluation checks) are tested; the rest needs a live socket and a real
keypair, which is what `scripts/run-local.sh` is for. A single workspace number
would let the easy crate quietly subsidise the hard one.
"""

import argparse
import os
import re
import sys
from collections import defaultdict

# crate -> (floor, why this floor)
FLOORS = {
    "primitives": (
        85,
        "the execution core. Parsing, lowering, validation, bytecode and the "
        "ciphertext boundary are pure functions with a checkable answer, and "
        "the bytes a worker attests to come from here",
    ),
    "node": (
        25,
        "the decisions are covered -- M-of-N aggregation, the transport's "
        "size and status handling, compilation, and the worker's checks "
        "before it spends CPU on homomorphic work. The remainder is the "
        "orchestration around them: `run` loops that block on a socket, an "
        "evaluator thread, a `main` that dispatches on a role, and "
        "`demo::run`, which logs rather than returning anything to assert "
        "on. 115 of node's lines (main.rs and demo.rs) are unreachable from "
        "a unit test as written, which caps this crate around 30% until "
        "there is an end-to-end harness; scripts/run-local.sh is that "
        "harness today, and it is not instrumented",
    ),
    "disca-cli": (
        85,
        "the key holder (task 4.3), and held as high as primitives for the "
        "same reason: it is the only party that can encrypt or decrypt "
        "anything, so a bug here is a wrong plaintext or a lost key rather "
        "than a failed job. It is also almost entirely testable -- the "
        "commands take paths and values and return a Result, `main` only "
        "dispatches, and the one test that matters runs keygen -> compile -> "
        "encrypt -> evaluate -> decrypt through a scratch directory. What is "
        "left uncovered is `main` itself and the I/O failure arms (a "
        "directory that cannot be created, an encoder that cannot encode), "
        "which need a broken filesystem to reach",
    ),
}

TEST_MODULE = re.compile(r"^\s*#\[cfg\((all\()?test\b")


def first_test_line(path: str) -> int:
    """Line number where a file's test-only code begins, or a sentinel."""
    try:
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if TEST_MODULE.match(line):
                    return number
    except OSError:
        pass
    return sys.maxsize


def parse_lcov(path: str) -> dict[str, list[tuple[int, int]]]:
    """{source file: [(line, hit count), ...]}"""
    files: dict[str, list[tuple[int, int]]] = {}
    current: list[tuple[int, int]] = []
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if line.startswith("SF:"):
            current = files.setdefault(line[3:], [])
        elif line.startswith("DA:"):
            number, _, count = line[3:].partition(",")
            current.append((int(number), int(count.split(",")[0])))
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("lcov")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero when a crate is below its floor",
    )
    args = parser.parse_args()

    root = os.getcwd()
    # crate -> [production lines, production lines covered, test lines dropped]
    crates: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for path, lines in parse_lcov(args.lcov).items():
        crate = os.path.relpath(path, root).split(os.sep)[0]
        cutoff = first_test_line(path)
        bucket = crates[crate]
        for number, count in lines:
            if number >= cutoff:
                bucket[2] += 1
                continue
            bucket[0] += 1
            bucket[1] += 1 if count > 0 else 0

    print("Line coverage, excluding inline #[cfg(test)] modules.")
    print()
    print(f"{'crate':<13}{'lines':>7}{'covered':>9}{'%':>8}{'floor':>7}{'':>4}{'test':>7}")
    print("-" * 55)

    failures = []
    total_lines = total_covered = total_test = 0
    for crate in sorted(crates):
        count, covered, test_lines = crates[crate]
        total_lines += count
        total_covered += covered
        total_test += test_lines
        percent = 100.0 * covered / count if count else 100.0
        floor, reason = FLOORS.get(crate, (0, "not a floored crate"))
        mark = "  " if percent >= floor else " <"
        print(
            f"{crate:<13}{count:>7}{covered:>9}{percent:>7.1f}%"
            f"{floor:>6}%{mark:>4}{test_lines:>7}"
        )
        if percent < floor:
            failures.append((crate, percent, floor, reason))

    print("-" * 55)
    workspace = 100.0 * total_covered / total_lines if total_lines else 100.0
    print(
        f"{'workspace':<13}{total_lines:>7}{total_covered:>9}{workspace:>7.1f}%"
        f"{'':>10}{total_test:>7}"
    )
    print()
    print(
        f"({total_test} instrumented lines in test modules were excluded from "
        "both columns.)"
    )

    if summary_file := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_file, "a", encoding="utf-8") as handle:
            handle.write("### Coverage\n\n")
            handle.write(
                "Line coverage of production code. Inline `#[cfg(test)]` "
                "modules are excluded — they are covered by definition.\n\n"
            )
            handle.write("| crate | lines | covered | % | floor |\n")
            handle.write("|---|---:|---:|---:|---:|\n")
            for crate in sorted(crates):
                count, covered, _ = crates[crate]
                percent = 100.0 * covered / count if count else 100.0
                floor, _ = FLOORS.get(crate, (0, ""))
                flag = "" if percent >= floor else " :x:"
                handle.write(
                    f"| `{crate}` | {count} | {covered} | "
                    f"{percent:.1f}%{flag} | {floor}% |\n"
                )
            handle.write(
                f"| **workspace** | {total_lines} | {total_covered} | "
                f"**{workspace:.1f}%** | |\n"
            )

    if failures and args.check:
        print(file=sys.stderr)
        for crate, percent, floor, reason in failures:
            print(
                f"error: {crate} is at {percent:.1f}%, below its {floor}% floor.\n"
                f"       The floor is set there because {reason}.\n"
                f"       Add a test that would fail if the behaviour regressed --\n"
                f"       not one that only walks the lines. If the floor is wrong,\n"
                f"       change it in scripts/lib/coverage_report.py and say why.",
                file=sys.stderr,
            )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
