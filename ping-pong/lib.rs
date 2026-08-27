//! A ball-and-paddle rally, as six circuits over encrypted state.
//!
//! The committee tally answers "can DISCA express a useful computation". This
//! answers a harder one: **can encrypted state evolve**. A tally is one job over
//! fresh inputs; a game is a job whose output becomes the next job's input,
//! forever, with nobody decrypting in between.
//!
//! That works because a result blob and an input blob are the same thing.
//! `wire::seal_result` is `encode(&result.compress())`, and `wire::decode` reads
//! exactly that encoding — so frame N's six output ciphertexts are frame N+1's
//! six inputs, and the key holder decrypts only to draw.
//!
//! # The state, and why the order is load-bearing
//!
//! Six `i32`s, in this order, always:
//!
//! ```text
//!   0  ball_x     1  ball_y     2  vel_x
//!   3  vel_y      4  paddle_y   5  score
//! ```
//!
//! Every function here takes all six and returns one. That is not a style
//! choice: `DiscaFunction::run` returns a single `FheInt32` (single-output is a
//! design constraint, `tasks.md` 1.2b), so one frame is six jobs. They are
//! independent — same inputs, different outputs — so they dispatch
//! concurrently rather than in sequence.
//!
//! The coordinator passes `--input` in argument order and nothing downstream
//! can notice a transposition: the inputs are ciphertext, so swapping `vel_x`
//! and `vel_y` yields a plausible answer to a different question.
//!
//! # What the opcode set allows, and what it cost
//!
//! The set is `add`, `mul`, `sub`, the signed comparisons, `eqz`, `select` and
//! the local ops. `mul` exists but costs **2.04 s** against 225 ms for an add
//! (`architecture.md` §2), so it is avoided rather than unavailable. There is no
//! divide and no shift at all. Everything here is written to stay inside that:
//!
//! * Negation is `0 - v`, which also dodges the 2.04 s `mul` (`architecture.md`
//!   §2). Every operation in these circuits is in the 141–265 ms band.
//! * There is no absolute value. `if x < 0 { -x } else { x }` is canonicalised
//!   by LLVM into `(x ^ (x >> 31)) - (x >> 31)`, and `I32ShrS` is not in the
//!   set — so bounds are clamps and band tests are two comparisons. That was
//!   found by compiling, not by reading.
//! * Collision cannot interpolate the crossing point, because that needs a
//!   division. See `hit_paddle` for what is done instead.
//!
//! # Why this is not Pong, and what the opcode set had to do with it
//!
//! The name is aspirational. Measured against the embedded circuit over 6,000
//! frames, three of canonical Pong's defining behaviours are absent:
//!
//! * **One paddle, not two.** `vel_x` reverses at `PADDLE_X` against a paddle
//!   and at `W` against a bare wall, so the score counts *saves* rather than
//!   points taken off an opponent.
//! * **No rally acceleration.** Exactly one `(|vel_x|, |vel_y|)` pair — `(80,
//!   45)` — occurs for the whole run.
//! * **No angle off the paddle.** This is the important one. In Pong the
//!   deflection depends on *where* the ball strikes, and here it cannot: look at
//!   the signature. [`vel_y`] never receives `paddle_y`. Across the five
//!   distinct contact offsets a rally produces (-90, -45, 0, 45, 90) the
//!   outcome is `45 → 45` every time.
//!
//! The third is a direct consequence of the cost model rather than an
//! oversight. The usual formula is `vy += k * (by - py) / REACH`, and division
//! is not in the set at all while `mul` is nine times an add. The cheap way in
//! is banding — compare the offset against two thresholds and `select` a
//! velocity — which is comparisons and selects only, about six ops, and would
//! make rallies turn on where the ball lands.
//!
//! It is left undone deliberately: it changes the bytecode hash, and every
//! recorded frame time, cost table and cross-mode fixture in this repository is
//! pinned to the current circuit. Worth doing, worth doing on purpose.
//!
//! What it *does* demonstrate is untouched by any of that. Encrypted state
//! evolving across frames with nobody decrypting in between does not require
//! the game to be a good game.
//!
//! # Speed, and why the ball moves in visible jumps
//!
//! A frame costs seconds, so the ball must cross the board in a watchable
//! number of them. Velocity is *encrypted state*, not a constant here, which
//! means the host picks its magnitude at encrypt time from a measured frame
//! time — roughly `board_width / (target_seconds / frame_seconds)`. Calibrating
//! once before the rally keeps the whole state encrypted; adapting per frame
//! would make the ball's speed a host-chosen public value re-encrypted every
//! frame.
//!
//! The consequence to design around, rather than discover: at those step sizes
//! the ball can pass *through* a thin paddle between frames.
//!
//! # Playing it without the encryption
//!
//! These are ordinary Rust functions, and the crate builds as an `rlib` and a
//! `cdylib` so the same six can be called natively or from wasm. Two ways to
//! watch the game they describe, both running the identical logic the network
//! evaluates:
//!
//! ```text
//! cargo run --release -p ping-pong --bin pong     # in a terminal
//! open ping-pong/web/index.html                   # in a browser, on the real wasm
//! ```
//!
//! The browser page embeds `ping_pong.wasm` itself — the same module a worker
//! lowers to bytecode — so it is not a reimplementation either. Six circuit
//! evaluations cost about **91 ns** there and about **35 s** on the network,
//! and that difference is the whole demonstration. The browser page runs at
//! 8 frames a second rather than 60 for a reason worth knowing: see "Speed"
//! above — `vel_x` is 80 on a 960-wide board, so a crossing is twelve frames,
//! and twelve frames at 60 fps is a fifth of a second.

/// Board width. Public — the geometry is not the secret, the position is.
const W: i32 = 960;
/// Board height.
const H: i32 = 540;

/// The plane the paddle defends. The ball is caught at or beyond it.
const PADDLE_X: i32 = 40;

/// Half the paddle's height. Sized against the per-frame vertical step rather
/// than against how a paddle looks: see [`hit_paddle`].
const PADDLE_REACH: i32 = 90;

/// How far the paddle may move in one frame. The whole tension of the rally is
/// whether this is enough to arrive in time, so it is deliberately less than a
/// fast ball's vertical step.
const PADDLE_SPEED: i32 = 55;

/// Holds a coordinate inside `[0, limit]`.
///
/// **Clamped, not mirrored, and the reason is worth recording.** The obvious
/// reflection is `if next < 0 { 0 - next }`, which is an absolute value — and
/// LLVM canonicalises absolute value into the branchless shift trick
/// `(x ^ (x >> 31)) - (x >> 31)`. `I32ShrS` is not in the opcode set, so that
/// version compiles to wasm perfectly and is then rejected by
/// `disca-cli compile`, which is exactly where it was caught.
///
/// Comparing against `0` and selecting `0` avoids the idiom: LLVM emits a plain
/// `select`, which is `Select`. The visible difference is that the ball touches
/// a wall and departs on the next frame rather than mirroring within one — at
/// seconds per frame that reads as contact rather than as a glitch.
#[allow(
    clippy::manual_clamp,
    reason = "`i32::clamp` panics when min > max, so it carries a branch and a \
              panic path. Neither lowers to the opcode set — the whole circuit \
              has to be selects. The lint is right about the shape and wrong \
              about the target."
)]
fn clamp_to(next: i32, limit: i32) -> i32 {
    let low = if next < 0 { 0 } else { next };
    if low > limit { limit } else { low }
}

/// Whether the ball is at or past the paddle plane *and* within its reach.
///
/// Returns 1 or 0 rather than `bool`, and every condition is *counted* rather
/// than combined with `&&`, `&`, `||` or `|`. Short-circuiting operators lower
/// to branches — and a branch on a value derived from an encrypted input is the
/// one thing that cannot be evaluated, since both sides must be computed and
/// selected between. The bitwise forms avoid the branch but invite their own
/// rewrites. Arithmetic on 0/1 avoids both.
///
/// **This is the tunnelling compromise.** The honest test is "did the segment
/// from the old position to the new one cross the paddle plane, and where was
/// the ball vertically when it did" — which needs the crossing fraction, which
/// needs a division, which the opcode set does not have.
///
/// So the test is on the endpoint, and [`PADDLE_REACH`] is sized to at least
/// one vertical step to compensate. A ball moving faster vertically than the
/// paddle is tall can still pass through, and that is a property of evaluating
/// at seconds per frame rather than a bug to fix in the circuit.
fn hit_paddle(next_x: i32, next_y: i32, paddle_y: i32) -> i32 {
    let top = paddle_y - PADDLE_REACH;
    let bottom = paddle_y + PADDLE_REACH;

    // Three conditions, counted rather than combined. `a & b` on `bool` invites
    // LLVM to fold a two-sided range test into a *single unsigned* compare —
    // `(x - lo) as u32 <= (hi - lo) as u32` — and the IR treats `i32` as signed,
    // so `disca-cli compile` rejects it. Summing `as i32` votes is the shape
    // `committee-tally`'s `count_above` already proved lowers cleanly, and it
    // keeps each comparison separate and signed.
    let votes = (next_x <= PADDLE_X) as i32 + (next_y >= top) as i32 + (next_y <= bottom) as i32;

    // 1 when all three held, 0 otherwise. Returned as `i32` rather than `bool`
    // so callers combine it with arithmetic instead of `&`/`|`.
    (votes == 3) as i32
}

/// Next `ball_x`. Stops at the paddle on a save, at a wall otherwise.
///
/// **The paddle has to appear here and not only in [`vel_x`].** This function
/// used to ignore `by` and `py` and simply clamp to the board, which reversed
/// the ball at the right moment and left it in the wrong place. With `vel_x` at
/// 80 the positions going left are 480, 400, 320, 240, 160, 80, 0 — the ball
/// never occupies the plane at [`PADDLE_X`], so every save was recorded with the
/// ball against the back wall. Rendered, that is a ball passing *through* the
/// paddle and bouncing off the border, which is exactly what it looked like.
///
/// Clamping to the plane rather than reflecting off it is the same choice
/// [`clamp_to`] makes for the walls, and for the same reason: contact on one
/// frame and departure on the next reads as contact, while mirroring inside a
/// single frame reads as a glitch.
///
/// A miss still carries on to the wall behind, which is what keeps a miss
/// visibly different from a save (see [`vel_x`]).
///
/// # Counting misses, and the test that cannot fail
///
/// [`clamp_to`] floors this at zero, so the ball never holds a negative `x`. A
/// miss detector written as `ball_x < 0` therefore reports **zero misses
/// forever**, on any circuit, at any velocity — it is not a strict test, it is
/// an unfalsifiable one, and it looks like the rally is unlosable.
///
/// The real test is arrival: `ball_x == 0` on a frame whose predecessor was
/// `> 0`. Over 4000 frames from the opening state that gives 166 saves and 23
/// misses, where `< 0` gives 166 and 0.
///
/// This is worth stating because the wrong version is the natural one to write
/// and it fails silently in the flattering direction.
#[unsafe(no_mangle)]
pub extern "C" fn ball_x(bx: i32, by: i32, vx: i32, vy: i32, py: i32, _sc: i32) -> i32 {
    let next_x = bx + vx;
    let next_y = by + vy;

    // A select, not a branch. Both arms are evaluated and one is chosen, which
    // is the only shape a condition derived from an encrypted value can take.
    if hit_paddle(next_x, next_y, py) != 0 {
        PADDLE_X
    } else {
        clamp_to(next_x, W)
    }
}

/// Next `ball_y`. Reflects off either horizontal wall.
#[unsafe(no_mangle)]
pub extern "C" fn ball_y(_bx: i32, by: i32, _vx: i32, vy: i32, _py: i32, _sc: i32) -> i32 {
    clamp_to(by + vy, H)
}

/// Next `vel_x`.
///
/// Reverses on the far wall, and on a paddle interception. A miss does *not*
/// reverse: the ball carries on past the plane, which is what makes the miss
/// visible rather than silently identical to a save.
#[unsafe(no_mangle)]
pub extern "C" fn vel_x(bx: i32, by: i32, vx: i32, vy: i32, py: i32, _sc: i32) -> i32 {
    let next_x = bx + vx;
    let next_y = by + vy;

    // Same "did the wall stop it" test as `vel_y`, plus the paddle. Counted
    // rather than `|`-ed, and compared with `!= 0` rather than `> 0`: on a value
    // the optimiser can prove non-negative, `> 0` is a signed comparison it is
    // free to rewrite as an unsigned one.
    let stopped = (clamp_to(next_x, W) != next_x) as i32;
    let reverse = stopped + hit_paddle(next_x, next_y, py);

    if reverse != 0 { 0 - vx } else { vx }
}

/// Next `vel_y`. Reverses at the top and bottom edges.
#[unsafe(no_mangle)]
pub extern "C" fn vel_y(_bx: i32, by: i32, _vx: i32, vy: i32, _py: i32, _sc: i32) -> i32 {
    let next_y = by + vy;
    // "Did the wall stop it" rather than "is it out of bounds". The two are the
    // same question, and only one of them survives the optimiser: `y < 0 || y >
    // H` is a two-sided test against constants, which LLVM folds into a single
    // *unsigned* range check (`i32.lt_u 541`), and the IR treats `i32` as
    // signed. Comparing the clamped position against the unclamped one is a
    // single `ne` between two variables, which has no such rewrite — and it
    // says what is meant more directly anyway.
    if clamp_to(next_y, H) != next_y {
        0 - vy
    } else {
        vy
    }
}

/// Next `paddle_y`. Tracks the ball, capped at [`PADDLE_SPEED`].
///
/// Automatic rather than player-driven, and that follows from the frame time: a
/// human input cannot land inside a frame that takes seconds, so a controllable
/// paddle would be a control that does nothing most of the time. Tracking with
/// a speed cap keeps the drama — whether it arrives before the ball does —
/// while staying honest about what a player could actually influence.
#[unsafe(no_mangle)]
#[allow(clippy::manual_clamp, reason = "see `clamp_to`")]
pub extern "C" fn paddle_y(_bx: i32, by: i32, _vx: i32, vy: i32, py: i32, _sc: i32) -> i32 {
    let target = by + vy;
    let delta = target - py;

    let step = if delta > PADDLE_SPEED {
        PADDLE_SPEED
    } else if delta < 0 - PADDLE_SPEED {
        0 - PADDLE_SPEED
    } else {
        delta
    };

    let moved = py + step;

    // The paddle cannot leave the board either.
    let low = if moved < PADDLE_REACH {
        PADDLE_REACH
    } else {
        moved
    };
    if low > H - PADDLE_REACH {
        H - PADDLE_REACH
    } else {
        low
    }
}

/// Next `score`: one per successful interception.
///
/// Arithmetic over a comparison result rather than a select, the pattern
/// `committee-tally`'s `count_above` established.
#[unsafe(no_mangle)]
pub extern "C" fn score(bx: i32, by: i32, vx: i32, vy: i32, py: i32, sc: i32) -> i32 {
    let next_x = bx + vx;
    let next_y = by + vy;
    sc + hit_paddle(next_x, next_y, py)
}
