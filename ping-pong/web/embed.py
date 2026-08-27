#!/usr/bin/env python3
"""Rebuilds `index.html` with the current circuits embedded.

Two artefacts, both inlined so the page works from disk without a server --
`fetch` of a `file://` URL is blocked, and asking someone to start a web server
to look at an 809-byte demo is a poor trade:

  WASM      the module itself, base64, which the page instantiates and calls.
  CIRCUITS  what `disca-cli compile` lowers that module to -- the DISCA opcode
            sequence a worker actually evaluates under encryption. The page
            shows it so the encrypted cost of a frame is derived and visible
            rather than asserted.

CIRCUITS comes from `primitives/examples/inspect`, which is the same lowering
the network uses, so the two cannot drift apart without this script noticing.

Run from the repository root after rebuilding the circuits:

    cd ping-pong && cargo build --release -p ping-pong --lib
    cp ../target/wasm32-unknown-unknown/release/ping_pong.wasm .
    python3 web/embed.py
"""

import base64
import json
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[2]
wasm_path = root / "ping-pong" / "ping_pong.wasm"
page = root / "ping-pong" / "web" / "index.html"

wasm = wasm_path.read_bytes()

# The lowered circuit, straight from the tool the network uses.
inspect = subprocess.run(
    ["cargo", "run", "--release", "-q", "-p", "primitives",
     "--example", "inspect", "--", str(wasm_path.relative_to(root))],
    cwd=root, capture_output=True, text=True,
)
if inspect.returncode != 0:
    sys.exit(f"inspect failed:\n{inspect.stderr}")

circuits, current = {}, None
for line in inspect.stdout.splitlines():
    header = re.match(r"^  (\w+)\(\d+ param\)", line)
    if header:
        current = header.group(1)
        circuits[current] = []
        continue
    op = re.match(r"^ +(\d+)  (\w+)(?:\((-?\d+)\))?\s*$", line)
    if op and current is not None:
        name, arg = op.group(2), op.group(3)
        circuits[current].append([name] if arg is None else [name, int(arg)])

if len(circuits) != 6:
    sys.exit(f"expected 6 circuits, parsed {len(circuits)}: {list(circuits)}")

text = page.read_text()
text = re.sub(r'const WASM = "[^"]*";',
              'const WASM = "' + base64.b64encode(wasm).decode() + '";',
              text, count=1)
text = re.sub(r"const CIRCUITS = \{.*?\};",
              "const CIRCUITS = " + json.dumps(circuits, separators=(",", ":")) + ";",
              text, count=1, flags=re.S)
page.write_text(text)

ops = sum(len(v) for v in circuits.values())
print(f"embedded {len(wasm)} bytes of wasm and {ops} circuit ops "
      f"across {len(circuits)} functions into {page.relative_to(root)}")
