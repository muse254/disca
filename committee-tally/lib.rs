//! Candidate demo circuits, written the way a developer actually would.
//!
//! This crate exists to answer one question empirically: can the DISCA opcode
//! set express the confidential committee tally from `architecture.md` §10?
//! Each function is a different way of writing "pick the winning score", from
//! the most idiomatic to the most hand-flattened. Compiling them and reading
//! the emitted opcodes tells us where the IR actually stops, rather than where
//! we assume it stops.
//!
//! Build with `cargo build --release` from this directory (the local cargo
//! config pins the wasm32 target).

/// The obvious two-input version. An `if` on a comparison of two values —
/// the smallest possible test of whether rustc gives us branch-free output.
#[unsafe(no_mangle)]
pub extern "C" fn max2(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}

/// Four scores, accumulated with a mutable "best so far". This is how someone
/// would write a tally without thinking about it.
#[unsafe(no_mangle)]
pub extern "C" fn tally4_branching(a: i32, b: i32, c: i32, d: i32) -> i32 {
    let mut best = a;
    if b > best {
        best = b;
    }
    if c > best {
        best = c;
    }
    if d > best {
        best = d;
    }
    best
}

/// The same tally written as a tree of expression-level selects, with no
/// mutation and no statement-level branching.
#[unsafe(no_mangle)]
pub extern "C" fn tally4_select(a: i32, b: i32, c: i32, d: i32) -> i32 {
    let ab = if a > b { a } else { b };
    let cd = if c > d { c } else { d };
    if ab > cd { ab } else { cd }
}

/// The counting pattern: how many scores clear a threshold. Exercises
/// arithmetic over comparison results rather than selection.
#[unsafe(no_mangle)]
pub extern "C" fn count_above(a: i32, b: i32, c: i32, d: i32, threshold: i32) -> i32 {
    (a > threshold) as i32
        + (b > threshold) as i32
        + (c > threshold) as i32
        + (d > threshold) as i32
}

/// A loop over a fixed-size array. Idiomatic Rust, and the shape a real tally
/// over N candidates would take.
#[unsafe(no_mangle)]
pub extern "C" fn tally_loop(a: i32, b: i32, c: i32, d: i32) -> i32 {
    let scores = [a, b, c, d];
    let mut best = i32::MIN;
    for s in scores {
        if s > best {
            best = s;
        }
    }
    best
}
