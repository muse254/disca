"""Checks that consensus-critical dependencies are still pinned exactly.

Reads `cargo metadata --format-version 1` on stdin.

`tfhe` decides the bytes a worker attests to. Two workers on different patch
releases can evaluate the same circuit correctly and still disagree byte for
byte, which the coordinator sees as a fault rather than as a version skew
(docs/architecture.md §3, task 2.10b). The manifest says `=1.5.0` and explains
why; this asserts nobody has quietly relaxed it to `^1.5` — a change that
compiles, passes every test on one machine, and only shows up as jobs that stop
settling.

The exact version is deliberately not hardcoded here. A deliberate bump edits
one place (the manifest) and stays honest; what must never happen silently is
losing the `=`, or ending up with two `tfhe` versions in one graph.
"""

import json
import sys

# Crates whose exact version is part of the protocol rather than an
# implementation detail. Add to this list, do not remove from it without a
# reason recorded in docs/architecture.md.
PINNED = ("tfhe",)


def main() -> int:
    metadata = json.load(sys.stdin)
    members = set(metadata.get("workspace_members", []))
    problems = []

    for package in metadata["packages"]:
        if package["id"] not in members:
            continue
        for dependency in package["dependencies"]:
            if dependency["name"] not in PINNED:
                continue
            req = dependency["req"]
            if not req.startswith("="):
                problems.append(
                    f"{package['name']} depends on {dependency['name']} as "
                    f'"{req}", which is not an exact pin'
                )

    resolved = []
    for name in PINNED:
        versions = sorted(
            {p["version"] for p in metadata["packages"] if p["name"] == name}
        )
        if len(versions) > 1:
            problems.append(
                f"{name} resolves to {len(versions)} versions "
                f"({', '.join(versions)}); evaluation must be one implementation"
            )
        elif versions:
            resolved.append(f"    {name} {versions[0]} (exact)")

    if problems:
        print(file=sys.stderr)
        for problem in problems:
            print(f"  error: {problem}", file=sys.stderr)
        print(
            "\nByte-reproducible evaluation depends on these being pinned; see\n"
            "docs/architecture.md §3 and pin_fft_plan in node/src/main.rs.",
            file=sys.stderr,
        )
        return 1

    for line in resolved:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
