//! Does evaluating several circuits at once beat evaluating them one at a time?
//!
//! One frame of `ping-pong` is six independent circuits over the same six
//! inputs. A worker evaluates them serially — `spawn_evaluator` in
//! `node/src/worker.rs` drains a channel on a single thread — and a frame
//! measured 22.8 s to 29.5 s whether the jobs were submitted serially or
//! concurrently. That points at the worker rather than the coordinator, but
//! "points at" is not a measurement, and a second evaluator thread is a real
//! change to the one place a worker holds its signing key.
//!
//! So this measures the question directly, in one process, with no network and
//! no coordinator: the same six circuits, at several degrees of concurrency.
//!
//! The reason the answer is not obvious is that tfhe already parallelises a
//! *single* evaluation internally with rayon. If one circuit saturates the
//! machine, running six at once buys nothing and costs scheduling. The numbers
//! in `attestation.md` hint both ways — 1,412 ms alone against 2,858-3,992 ms
//! with three contending, which is 3x the work in ~2.5x the time, so some
//! headroom existed there.
//!
//! # The answer, on an 8-core M-series laptop
//!
//! ```text
//! one circuit, alone                860 ms
//!
//! six circuits                      total      per  vs serial
//! serially (today's worker)       11158 ms   1860 ms      1.00x
//! 2 at a time                     10159 ms   1693 ms      1.10x
//! 3 at a time                       9590 ms   1598 ms      1.16x
//! 6 at a time                       9179 ms   1530 ms      1.22x
//! ```
//!
//! **1.22x, so no.** A second evaluator thread is not worth moving the signing
//! key onto — and that is in a process with nothing else running. The real
//! setup has three workers sharing the machine, so most of that headroom is
//! already spent: 11 s of work under ~3x contention is the ~29 s frame that was
//! measured end to end, and per-worker threading would be competing with the
//! contention rather than with idle cores.
//!
//! Note the six circuits are not equal — `ball_x` is one add and two selects at
//! 860 ms, while `vel_x` and `score` both call `hit_paddle`. 1,860 ms is the
//! average, not the cost of the cheapest one repeated.
//!
//! Rerun this before revisiting the decision; the conclusion is about a machine
//! as much as about the code.
//!
//! ```text
//! cargo run --release -p primitives --example eval_concurrency
//! ```
//!
//! Release only. Debug evaluation is 87-98x slower (`architecture.md` §2) and
//! would measure the build profile rather than the question.

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use primitives::program::{DiscaFunction, DiscaProgram, Program};
use tfhe::prelude::FheTryEncrypt;
use tfhe::{ConfigBuilder, FheInt32, ServerKey, generate_keys, set_server_key};

/// The six circuits of one frame, in state order.
const FRAME: [&str; 6] = ["ball_x", "ball_y", "vel_x", "vel_y", "paddle_y", "score"];

/// A mid-rally state: ball heading for the paddle, which is trailing it.
const STATE: [i32; 6] = [480, 270, -80, 45, 270, 0];

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../ping-pong/ping_pong.wasm");
    let wasm = std::fs::read(path).expect("ping_pong.wasm fixture");
    let program = Arc::new(DiscaProgram::from_program(
        &Program::from_wasm(&wasm).expect("parse ping-pong module"),
    ));

    let (client_key, server_key) = generate_keys(ConfigBuilder::default().build());
    set_server_key(server_key.clone());

    let inputs: Arc<Vec<FheInt32>> = Arc::new(
        STATE
            .iter()
            .map(|v| FheInt32::try_encrypt(*v, &client_key).expect("encrypt state"))
            .collect(),
    );

    println!(
        "cores: {}",
        thread::available_parallelism().map_or(0, |n| n.get())
    );
    println!(
        "a frame is {} circuits over {} inputs\n",
        FRAME.len(),
        STATE.len()
    );

    // One circuit alone, as the baseline every other number is read against.
    let alone = time(|| {
        let func = lookup(&program, FRAME[0]);
        func.run(&inputs).expect("evaluate");
    });
    println!("one circuit, alone            {alone:>7.0} ms");

    println!(
        "\n{:<30}{:>9}{:>9}{:>11}",
        "six circuits", "total", "per", "vs serial"
    );
    let mut serial_ms = 0.0;
    for threads in [1usize, 2, 3, 6] {
        let elapsed = time(|| run_frame(&program, &inputs, &server_key, threads));
        if threads == 1 {
            serial_ms = elapsed;
        }
        let speedup = serial_ms / elapsed;
        let label = if threads == 1 {
            "serially (today's worker)".to_string()
        } else {
            format!("{threads} at a time")
        };
        println!(
            "{label:<30}{elapsed:>7.0} ms{:>7.0} ms{speedup:>10.2}x",
            elapsed / 6.0
        );
    }

    println!(
        "\nA worker evaluating {} at a time would cut a frame from {:.0} s to about {:.0} s,",
        6,
        serial_ms / 1000.0,
        time(|| run_frame(&program, &inputs, &server_key, 6)) / 1000.0
    );
    println!("before the 3x contention of three workers sharing this machine.");
}

fn lookup<'a>(program: &'a DiscaProgram, name: &str) -> &'a DiscaFunction {
    program
        .function(name)
        .unwrap_or_else(|| panic!("no exported function {name}"))
}

/// Evaluates all six circuits, at most `threads` of them at once.
///
/// Each thread installs the server key itself: tfhe holds it in thread-local
/// state, so an evaluator thread that never called `set_server_key` panics on
/// its first operation. That is the one thing a multi-threaded evaluator in
/// `worker.rs` would have to get right, and it is cheap — the key is already
/// decompressed, this is a clone of an `Arc` internally.
fn run_frame(
    program: &Arc<DiscaProgram>,
    inputs: &Arc<Vec<FheInt32>>,
    server_key: &ServerKey,
    threads: usize,
) {
    for chunk in FRAME.chunks(threads) {
        thread::scope(|scope| {
            for name in chunk {
                let program = Arc::clone(program);
                let inputs = Arc::clone(inputs);
                let key = server_key.clone();
                scope.spawn(move || {
                    set_server_key(key);
                    lookup(&program, name).run(&inputs).expect("evaluate");
                });
            }
        });
    }
}

fn time(mut body: impl FnMut()) -> f64 {
    let started = Instant::now();
    body();
    started.elapsed().as_secs_f64() * 1000.0
}
