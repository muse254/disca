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

/// Ceiling on a request body we will buffer.
///
/// Sized for real messages: the largest legitimate one is a job dispatch at
/// tens of kilobytes.
const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;

/// Ceiling on a response body we will buffer.
///
/// Much larger than a request because of one payload: the compressed server key
/// is ~30 MB, and workers pull it over GET.
const MAX_RESPONSE_BYTES: usize = 256 * 1024 * 1024;

/// Cap on the capacity reserved up front. `Content-Length` is attacker-supplied,
/// so reserving it directly lets one request with a large header and no body
/// hold megabytes for the duration of the read timeout.
const MAX_PREALLOC_BYTES: usize = 64 * 1024;

/// Reads at most `limit` bytes, and treats hitting the limit exactly as a
/// failure.
///
/// `take` truncates silently, which is the wrong behaviour for everything here:
/// a short server key still hashes to *something*, and a truncated message body
/// may still decode. Better to refuse than to hand back a plausible-looking
/// fragment.
fn read_capped(reader: &mut impl Read, limit: usize, what: &str) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    reader
        .take(limit as u64 + 1)
        .read_to_end(&mut body)
        .map_err(|e| format!("cannot read {what}: {e}"))?;

    if body.len() > limit {
        return Err(format!("{what} exceeds the {limit} byte limit"));
    }
    Ok(body)
}

/// Reads a request body into memory.
pub fn read_body(request: &mut Request) -> Result<Vec<u8>, String> {
    let declared = request.body_length().unwrap_or(0);
    if declared > MAX_REQUEST_BYTES {
        return Err(format!("body of {declared} bytes exceeds the limit"));
    }

    // `declared` is attacker-supplied, so it caps the reserve rather than
    // setting it.
    let mut body = Vec::with_capacity(declared.min(MAX_PREALLOC_BYTES));
    body.append(&mut read_capped(
        &mut request.as_reader(),
        MAX_REQUEST_BYTES,
        "request body",
    )?);
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
    let response = agent
        .post(url)
        .content_type("application/octet-stream")
        .send(&body[..])
        .map_err(|e| format!("POST {url} failed: {e}"))?;

    let status = response.status();
    if !status.is_success() {
        return Err(format!("POST {url} refused: HTTP {status}"));
    }
    Ok(())
}

/// GETs a body into memory.
pub fn get(url: &str) -> Result<Vec<u8>, String> {
    let agent = agent();
    let mut response = agent
        .get(url)
        .call()
        .map_err(|e| format!("GET {url} failed: {e}"))?;

    // A non-2xx still carries a body. Without this check an error page is
    // returned to the caller as though it were the payload -- a 404 from a
    // stale peer would be handed back as "the server key".
    let status = response.status();
    if !status.is_success() {
        return Err(format!("GET {url} refused: HTTP {status}"));
    }

    read_capped(
        &mut response.body_mut().as_reader(),
        MAX_RESPONSE_BYTES,
        url,
    )
}

fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(TIMEOUT))
        .build()
        .into()
}
