//! `DISCA_LOG_FORMAT=json` is a contract, so it gets a test.
//!
//! Task 1b.3 asked for a machine-readable sink "before the demo, so the video
//! can show real job traces". What makes it a sink rather than a format flag is
//! that something downstream depends on its shape — and the moment something
//! does, the shape needs pinning.
//!
//! `worker-address` is the fixture because it is the cheapest role that emits
//! anything: one WARN and one line of output, no FHE, no sockets, no key
//! material worth protecting. A test that needed a real job would take three
//! seconds and would be testing the evaluator instead.

use std::process::Command;

/// The address `WorkerKey::derive("worker-1")` produces. Pinned rather than
/// recomputed: this test is about the log format, and a helper that derived the
/// expected value the same way the binary does would agree with a broken binary.
const WORKER_1: &str = "0x796b01b932191ac5f04e21d6e58388aec916b8cd";

fn worker_address(json: bool) -> (String, String) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_node"));
    command.args(["worker-address", "--id", "worker-1"]);
    if json {
        command.env("DISCA_LOG_FORMAT", "json");
    } else {
        command.env_remove("DISCA_LOG_FORMAT");
    }

    // `RUST_LOG` is inherited otherwise, and a developer running the suite with
    // it set to `off` would see this pass while proving nothing.
    command.env("RUST_LOG", "info");

    let out = command.output().expect("run node worker-address");
    (
        String::from_utf8(out.stdout).expect("stdout is utf-8"),
        String::from_utf8(out.stderr).expect("stderr is utf-8"),
    )
}

#[test]
fn json_mode_puts_the_answer_on_stdout_and_the_event_on_stderr() {
    let (stdout, stderr) = worker_address(true);

    // The half that matters to a consumer: stdout carries the answer and
    // *nothing else*. `run-local.sh` wraps this command in `RUST_LOG=off`
    // precisely because that is untrue of the text format, where the log and
    // the answer arrive in the same stream. Under JSON it is true without the
    // workaround, which is the point of sending events to stderr.
    assert_eq!(
        stdout.trim(),
        WORKER_1,
        "stdout must be the address alone, got {stdout:?}"
    );
    assert_eq!(stdout.lines().count(), 1, "and exactly one line");

    let lines: Vec<&str> = stderr.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(lines.len(), 1, "one event, got {lines:?}");

    let event: serde_json::Value = serde_json::from_str(lines[0])
        .unwrap_or_else(|e| panic!("stderr is not JSON: {e}\n{}", lines[0]));

    assert_eq!(event["level"], "WARN");
    assert_eq!(event["fields"]["worker"], "worker-1");
    assert_eq!(event["fields"]["address"], WORKER_1);
    assert!(
        event["fields"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("no --key given")),
        "the message is the key a consumer switches on: {event}"
    );
    assert!(
        event["timestamp"].is_string(),
        "a consumer orders events by this"
    );
}

#[test]
fn the_text_format_is_untouched_by_the_json_branch() {
    // The other half of the contract, and the one that protects every existing
    // script: adding a format must not move the default. Text still writes the
    // log to stdout beside the answer -- which is why `RUST_LOG=off` is still
    // correct in `run-local.sh` -- and leaves stderr empty.
    let (stdout, stderr) = worker_address(false);

    assert!(
        stdout.contains(WORKER_1),
        "the answer is still on stdout: {stdout:?}"
    );
    assert!(
        stdout.contains("no --key given"),
        "and so is the log, as before: {stdout:?}"
    );
    assert!(
        stderr.is_empty(),
        "the text format writes nothing to stderr, got {stderr:?}"
    );
}

#[test]
fn an_unrecognised_format_says_so_rather_than_falling_through_quietly() {
    let out = Command::new(env!("CARGO_BIN_EXE_node"))
        .args(["worker-address", "--id", "worker-1"])
        .env("DISCA_LOG_FORMAT", "jsonl")
        .env("RUST_LOG", "info")
        .output()
        .expect("run node worker-address");

    let stderr = String::from_utf8(out.stderr).expect("stderr is utf-8");
    assert!(
        stderr.contains("DISCA_LOG_FORMAT") && stderr.contains("not recognised"),
        "a typo must be named: {stderr:?}"
    );

    // And it must still work. Falling back is right; failing to start because a
    // log format was misspelled would be worse than the typo.
    let stdout = String::from_utf8(out.stdout).expect("stdout is utf-8");
    assert!(stdout.contains(WORKER_1), "it still answers: {stdout:?}");
}
