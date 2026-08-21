#!/usr/bin/env python3
"""Notices when code the specification models has changed underneath it.

There is no extraction, refinement or trace validation between
`DiscaAttestation.tla` and the Rust it claims to model. The correspondence is a
careful reading, recorded in comments — and `spec/README.md` says so. A careful
reading is a fact about one afternoon, not a property of the repository: the
model keeps passing after somebody rewrites `tally`, and it keeps passing while
saying something that is no longer true. That is worse than having no model,
because a green check reads as evidence.

This does not fix that, and it is important not to pretend otherwise. It is a
tripwire, not a proof. It records a hash of each function the specification
names, and fails when one changes without somebody saying they re-read the
model. What it buys is that the drift becomes a build failure at the moment it
is introduced, addressed to the person who introduced it, rather than a
discovery someone makes months later while wondering why the spec mentions a
function that no longer exists.

Comments and formatting are stripped before hashing, so rewording a comment or
running `cargo fmt` does not trip it. Anything that changes what the code *does*
will.

    python3 spec/drift.py --check     # exit 1 if a modelled item moved
    python3 spec/drift.py --update    # re-record hashes, after re-reading

`--update` is the part that has to stay honest. It is trivial to run it to make
a red build green, and doing so silently converts a checked claim into an
unchecked one. The manifest asks for a `models` line per item for that reason:
if you cannot say what the item corresponds to in the spec, you are not in a
position to accept its new hash.
"""

import argparse
import hashlib
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).resolve().parent / "models.toml"

# Comment syntax is the same in Rust and Solidity, which is the only reason one
# stripper serves both. A block comment is non-greedy so two of them in a row do
# not swallow the code between.
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT = re.compile(r"//[^\n]*")


def normalise(source: str) -> str:
    """Strip comments and collapse whitespace.

    A string literal containing `//` is mangled by this, which would matter if
    the output were ever compiled. It is only ever hashed, and the mangling is
    deterministic, so two runs over identical source still agree — which is the
    only property required here.
    """
    source = BLOCK_COMMENT.sub(" ", source)
    source = LINE_COMMENT.sub(" ", source)
    return " ".join(source.split())


def extract(path: Path, signature: str) -> str:
    """Return the body of the item whose declaration starts with `signature`.

    Matched on the first line whose stripped form starts with the signature, so
    `fn attribute(` finds the method inside `impl Verifier` without this needing
    to know Rust. From there it brace-matches, which is why the signature has to
    be unique within the file — `--check` says so by name when it is not.
    """
    text = path.read_text()
    lines = text.split("\n")

    starts = [i for i, line in enumerate(lines) if line.strip().startswith(signature)]
    if not starts:
        raise LookupError(f"{path}: no line starts with {signature!r}")
    if len(starts) > 1:
        raise LookupError(
            f"{path}: {signature!r} matches {len(starts)} lines "
            f"({', '.join(str(i + 1) for i in starts)}); make it unique"
        )

    start = starts[0]
    offset = sum(len(line) + 1 for line in lines[:start])
    opening = text.find("{", offset)
    if opening == -1:
        raise LookupError(f"{path}: {signature!r} has no body")

    # Brace matching over comment- and string-stripped text would be more
    # correct; over raw text it is wrong only if a brace appears unbalanced
    # inside a comment or a string literal in one of the modelled items. None do,
    # and `--check` fails loudly rather than silently if that ever changes,
    # because the hash of a truncated body will not match.
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[offset:index + 1]

    raise LookupError(f"{path}: {signature!r} has an unbalanced body")


def digest(path: Path, signature: str) -> str:
    return hashlib.sha256(normalise(extract(path, signature)).encode()).hexdigest()


def load_items():
    if not MANIFEST.exists():
        sys.exit(f"missing {MANIFEST.relative_to(REPO)}")
    return tomllib.loads(MANIFEST.read_text())["item"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--update", action="store_true")
    args = parser.parse_args()

    items = load_items()
    drifted, missing, current = [], [], {}

    for item in items:
        path = REPO / item["file"]
        key = f"{item['file']}::{item['item']}"
        try:
            current[key] = digest(path, item["item"])
        except (LookupError, FileNotFoundError) as error:
            missing.append((item, str(error)))
            continue
        if current[key] != item["sha256"]:
            drifted.append(item)

    if args.update:
        # Rewritten positionally, one `sha256 =` line per `[[item]]` in order.
        # Substituting by value looks simpler and is wrong: the placeholders are
        # identical strings, so a global replace gives every item the first
        # item's hash — which then passes `--check` and asserts that nine
        # different functions are the same function.
        lines = MANIFEST.read_text().split("\n")
        slots = [i for i, line in enumerate(lines) if line.startswith("sha256 = ")]
        if len(slots) != len(items):
            sys.exit(
                f"{MANIFEST.name}: {len(items)} item(s) but {len(slots)} sha256 line(s)"
            )
        for slot, item in zip(slots, items):
            key = f"{item['file']}::{item['item']}"
            if key in current:
                lines[slot] = f'sha256 = "{current[key]}"'
        MANIFEST.write_text("\n".join(lines))
        print(f"re-recorded {len(current)} item(s) in {MANIFEST.relative_to(REPO)}")
        if missing:
            print("still unresolved:")
            for _, error in missing:
                print(f"  {error}")
            return 1
        return 0

    for item, error in missing:
        print(f"GONE     {item['file']}::{item['item']}")
        print(f"         {error}")
        print(f"         the spec models it as: {item['models']}")

    for item in drifted:
        key = f"{item['file']}::{item['item']}"
        print(f"CHANGED  {key}")
        print(f"         recorded {item['sha256'][:16]}  now {current[key][:16]}")
        print(f"         the spec models it as: {item['models']}")

    if not drifted and not missing:
        print(f"{len(items)} modelled item(s) unchanged since the spec was written.")
        return 0

    print()
    print("Code the TLA+ specification models has changed. The model has not,")
    print("so it may now be checking something the code no longer does.")
    print()
    print("Re-read spec/DiscaAttestation.tla against the items above — the")
    print("`models` line names what each one corresponds to. Then either update")
    print("the model, or, if the change does not affect what it says:")
    print()
    print("    python3 spec/drift.py --update")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
