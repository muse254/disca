# disca-cli

The **key holder**, as a program you run rather than a struct inside a node
(`tasks.md` 4.3). It compiles circuits, generates and holds the keypair,
encrypts the inputs to a job and decrypts the result the network settles on —
and it never talks to a worker. See the module docs in `src/lib.rs` for why the
coordinator could not keep doing this.

```
disca-cli keygen  --out-dir keys/ [--force]      # -> server_key_hash=0x…
disca-cli compile --input m.wasm --output m.disca # -> bytecode_hash=0x…
disca-cli encrypt --client-key keys/client.key --values 71,93,42,88 --out-dir in/
                                                  # -> commitment=0x… per input
disca-cli decrypt --client-key keys/client.key --server-key keys/server.key \
                  --input result.blob             # -> the plaintext, alone
```

`keygen` writes `client.key` (owner-readable) and `server.key` (the compressed
28.8 MB key workers fetch by hash). It refuses to replace an existing
`client.key` without `--force`: that key is the only thing that can decrypt a
result, and jobs settle asynchronously.

The hashes on stdout are the values the bridge pins — `bytecodeHash` and
`serverKeyHash` at `registerProgram`, `inputCommits` at `submitJob`
(`bridge.md` §2) — so every command prints machine-readable lines and nothing
else. `decrypt` prints a bare integer so `$(disca-cli decrypt …)` is the answer.
