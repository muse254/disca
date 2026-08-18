//! Minimal synchronous HTTP, kept behind a handful of functions.
//!
//! Evaluation is CPU-bound and blocking — a tally is ~1 s, a multiply ~2 s — so
//! an async runtime would buy nothing at three workers while adding a large
//! dependency tree. Bodies are `wincode`-encoded (see [`crate::protocol`]), so
//! there is no content negotiation to do and no JSON to parse.

use std::io::Read;
use std::time::Duration;

use tiny_http::{Request, Response};

/// Applies to both connecting and reading. A worker that has gone away should
/// surface as a failed job well inside the coordinator's deadline.
const TIMEOUT: Duration = Duration::from_secs(30);

/// Refuses to buffer an unbounded body from a peer.
const MAX_BODY_BYTES: usize = 512 * 1024 * 1024;

/// Reads a request body into memory.
pub fn read_body(request: &mut Request) -> Result<Vec<u8>, String> {
    let declared = request.body_length().unwrap_or(0);
    if declared > MAX_BODY_BYTES {
        return Err(format!("body of {declared} bytes exceeds the limit"));
    }

    let mut body = Vec::with_capacity(declared);
    request
        .as_reader()
        .take(MAX_BODY_BYTES as u64)
        .read_to_end(&mut body)
        .map_err(|e| format!("cannot read body: {e}"))?;
    Ok(body)
}

/// Answers a request, swallowing the error if the peer has already hung up.
pub fn respond(request: Request, status: u16, body: &[u8]) {
    let response = Response::from_data(body).with_status_code(status);
    let _ = request.respond(response);
}

/// POSTs a body and discards the response, which carries no information beyond
/// its status.
pub fn post(url: &str, body: Vec<u8>) -> Result<(), String> {
    let agent = agent();
    agent
        .post(url)
        .content_type("application/octet-stream")
        .send(&body[..])
        .map_err(|e| format!("POST {url} failed: {e}"))?;
    Ok(())
}

/// GETs a body into memory.
pub fn get(url: &str) -> Result<Vec<u8>, String> {
    let agent = agent();
    let mut response = agent
        .get(url)
        .call()
        .map_err(|e| format!("GET {url} failed: {e}"))?;

    let mut body = Vec::new();
    response
        .body_mut()
        .as_reader()
        .take(MAX_BODY_BYTES as u64)
        .read_to_end(&mut body)
        .map_err(|e| format!("cannot read {url}: {e}"))?;
    Ok(body)
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(TIMEOUT))
        .build()
        .into()
}
