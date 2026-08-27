//! Pong at a normal speed, so you can see what the encrypted one is.
//!
//! This calls the **same six functions** the DISCA network evaluates under
//! encryption — `ball_x`, `ball_y`, `vel_x`, `vel_y`, `paddle_y`, `score` from
//! `lib.rs`. Not a reimplementation: the crate is built as an `rlib` as well as
//! a `cdylib` precisely so there is one set of rules and not two that have to be
//! kept in step.
//!
//! The point of running it here is the contrast. On this machine a frame takes
//! about 16 ms. On the network it takes about 30 seconds, because every addition
//! is a homomorphic operation on a 2.3 KB ciphertext and three workers have to
//! agree on the answer byte for byte. Same game, same code, same arithmetic —
//! and in the encrypted version no machine computing it can see the ball.
//!
//! ```text
//! cargo run --release -p ping-pong --bin pong             # ~30 fps
//! cargo run --release -p ping-pong --bin pong -- --fps 2  # slower
//! cargo run --release -p ping-pong --bin pong -- --encrypted-pace
//! ```
//!
//! Run it from the workspace root: `ping-pong/.cargo/config.toml` pins the
//! wasm32 target, which is what the circuits want and not what a terminal does.

use std::io::{Write, stdout};
use std::time::{Duration, Instant};

use ping_pong::{ball_x, ball_y, paddle_y, score, vel_x, vel_y};

/// Board geometry, matching the constants the circuits are compiled with.
const W: i32 = 960;
const H: i32 = 540;

/// Terminal size to scale the board into.
const COLS: i32 = 76;
const ROWS: i32 = 22;

/// What one frame costs on the network, with three workers and a 2-of-3 quorum.
/// Shown in the footer because it is the whole point.
///
/// **Measured, on an idle machine.** Three real frames through
/// `scripts/run-pong.sh disca` — six jobs each, three workers, a 2-of-3 quorum —
/// took 32.9 s, 39.7 s and 34 s on an otherwise-quiet 8-core laptop. 35 s is the
/// middle of that, and the spread is why this is a `f64` constant and not a
/// number quoted to two places.
///
/// It is a pace, not a guarantee, and it degrades badly under load: the same
/// command on the same machine at a load average of 8.9 took **219 s** for one
/// frame. Three workers each let `tfhe`'s rayon pool try to claim every core, so
/// they contend with each other and with anything else running. That is the
/// argument for sizing a deployment by the frame rather than by the job; here it
/// is just the reason a viewer's wall clock may disagree with this number.
///
/// Re-measure with `scripts/run-pong.sh disca` rather than trusting this.
const ENCRYPTED_FRAME_SECS: f64 = 35.0;

struct State {
    bx: i32,
    by: i32,
    vx: i32,
    vy: i32,
    py: i32,
    sc: i32,
}

impl State {
    /// One frame. Six independent calls over the same six values — which is
    /// exactly why the encrypted version is six jobs, and why they can be
    /// submitted together.
    fn step(&self) -> State {
        let (bx, by, vx, vy, py, sc) = (self.bx, self.by, self.vx, self.vy, self.py, self.sc);
        State {
            bx: ball_x(bx, by, vx, vy, py, sc),
            by: ball_y(bx, by, vx, vy, py, sc),
            vx: vel_x(bx, by, vx, vy, py, sc),
            vy: vel_y(bx, by, vx, vy, py, sc),
            py: paddle_y(bx, by, vx, vy, py, sc),
            sc: score(bx, by, vx, vy, py, sc),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let encrypted_pace = args.iter().any(|a| a == "--encrypted-pace");
    let fps: f64 = args
        .iter()
        .position(|a| a == "--fps")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(30.0);

    let frame_time = if encrypted_pace {
        Duration::from_secs_f64(ENCRYPTED_FRAME_SECS)
    } else {
        Duration::from_secs_f64(1.0 / fps.max(0.01))
    };

    let mut state = State {
        bx: 480,
        by: 270,
        vx: -80,
        vy: 45,
        py: 270,
        sc: 0,
    };

    // The cursor is deliberately *not* hidden. The first version of this hid it
    // with `\x1b[?25l` and called a function named `ctrl_c_restores_cursor`
    // that returned `None` and restored nothing — so Ctrl-C, which is how this
    // program is always going to end, left the terminal without a cursor.
    // Restoring it properly means a signal handler, which means a dependency in
    // a crate whose other artifact is a 773-byte wasm module.
    //
    // So the cursor stays visible and is parked below the board after every
    // frame, where it blinks harmlessly. A demo that damages the terminal it
    // ran in is worse than a demo with a cursor in it.

    let started = Instant::now();
    let mut frame: u64 = 0;

    loop {
        let began = Instant::now();
        render(&state, frame, started.elapsed(), encrypted_pace);
        state = state.step();
        frame += 1;

        if let Some(remaining) = frame_time.checked_sub(began.elapsed()) {
            std::thread::sleep(remaining);
        }
    }
}

fn render(state: &State, frame: u64, elapsed: Duration, encrypted_pace: bool) {
    // Home the cursor rather than clearing: clearing every frame makes the
    // whole board flicker, and the board is the thing being watched.
    let mut out = String::from("\x1b[H");

    let bx = scale(state.bx, W, COLS);
    let by = scale(state.by, H, ROWS);
    let py = scale(state.py, H, ROWS);
    let reach = (90 * ROWS / H).max(1);

    out.push_str("    ┌");
    for _ in 0..COLS {
        out.push('─');
    }
    out.push_str("┐\n");

    for row in 0..ROWS {
        out.push_str("    │");
        for col in 0..COLS {
            let on_paddle = col == 2 && (row - py).abs() <= reach;
            if col == bx && row == by {
                out.push('●');
            } else if on_paddle {
                out.push('█');
            } else {
                out.push(' ');
            }
        }
        out.push_str("│\n");
    }

    out.push_str("    └");
    for _ in 0..COLS {
        out.push('─');
    }
    out.push_str("┘\n");

    out.push_str(&format!(
        "\n     saves {:<4}  frame {:<6}  ball ({:>4},{:>4})  vel ({:>4},{:>4})  paddle {:>4}\n",
        state.sc, frame, state.bx, state.by, state.vx, state.vy, state.py
    ));

    let pace = if encrypted_pace {
        format!(
            "     encrypted pace: {:.0} s per frame, {} elapsed. Nothing here can see the ball.\n",
            ENCRYPTED_FRAME_SECS,
            humanise(elapsed)
        )
    } else {
        format!(
            "     this is {:.0}x faster than the same code under encryption (~{:.0} s a frame, idle).\n",
            ENCRYPTED_FRAME_SECS / elapsed.as_secs_f64().max(0.001) * frame.max(1) as f64,
            ENCRYPTED_FRAME_SECS
        )
    };
    out.push_str(&pace);
    // `\x1b[K` clears to end of line so a shorter line never leaves the tail of
    // a longer one behind, and the trailing newline parks the cursor here.
    out.push_str("     ctrl-c to stop\x1b[K\n");

    let mut stdout = stdout();
    let _ = stdout.write_all(out.as_bytes());
    let _ = stdout.flush();
}

/// Maps a board coordinate into a terminal cell.
fn scale(value: i32, from: i32, to: i32) -> i32 {
    (value.clamp(0, from) * (to - 1) / from).clamp(0, to - 1)
}

fn humanise(elapsed: Duration) -> String {
    let secs = elapsed.as_secs();
    if secs < 60 {
        format!("{secs}s")
    } else {
        format!("{}m{:02}s", secs / 60, secs % 60)
    }
}
