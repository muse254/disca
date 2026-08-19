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

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::thread;

    use super::*;

    #[test]
    fn a_body_at_the_limit_is_read_whole() {
        let payload = vec![7u8; 64];
        let read = read_capped(&mut Cursor::new(payload.clone()), 64, "test body").unwrap();
        assert_eq!(read, payload);
    }

    #[test]
    fn a_body_over_the_limit_is_refused_rather_than_truncated() {
        // The important half. `Read::take` truncates silently, and a truncated
        // read is worse than a failed one everywhere this is used: a short
        // server key still hashes to *something*, and a clipped message body
        // may still decode into a plausible-looking message. Refusing is the
        // only safe answer.
        let error = read_capped(&mut Cursor::new(vec![7u8; 65]), 64, "server key").unwrap_err();

        assert!(error.contains("server key"), "names the payload: {error}");
        assert!(error.contains("64"), "names the limit: {error}");
    }

    #[test]
    fn an_empty_body_is_not_an_error() {
        // A POST with no body is malformed at the protocol layer, not here;
        // this must hand back an empty vec so the decoder produces the useful
        // message rather than "cannot read request body".
        assert!(
            read_capped(&mut Cursor::new(Vec::new()), 64, "request body")
                .unwrap()
                .is_empty()
        );
    }

    /// Stands up a real server on an ephemeral port that answers each request
    /// with a canned status and body, and hands back what it received. Real
    /// sockets rather than a fake: the behaviour under test is how this module
    /// reacts to an HTTP status, and a fake would be asserting against itself.
    fn serving(replies: Vec<(u16, Vec<u8>)>) -> (String, thread::JoinHandle<Vec<Vec<u8>>>) {
        let server = tiny_http::Server::http("127.0.0.1:0").expect("bind an ephemeral port");
        let port = server.server_addr().to_ip().expect("an ip address").port();

        let handle = thread::spawn(move || {
            let mut received = Vec::new();
            for (status, body) in replies {
                let mut request = server.recv().expect("a request");
                received.push(read_body(&mut request).expect("a readable body"));
                respond(request, status, &body);
            }
            received
        });

        (format!("http://127.0.0.1:{port}"), handle)
    }

    #[test]
    fn get_returns_the_body_of_a_successful_response() {
        let (url, server) = serving(vec![(200, b"server key bytes".to_vec())]);

        assert_eq!(
            get(&format!("{url}/keys/0xabc")).unwrap(),
            b"server key bytes"
        );
        server.join().unwrap();
    }

    #[test]
    fn get_refuses_a_non_success_instead_of_returning_its_body() {
        // Task 2.9d. A 404 carries a body too, and a worker that got it back as
        // a payload would hash "not found" and install it as the server key.
        //
        // Two things prevent that now: ureq treats a non-2xx as an error by
        // default, and `get` checks the status again itself. That makes the
        // explicit check redundant today -- deliberately so, since the
        // behaviour must survive someone turning `http_status_as_error` off.
        // This asserts the behaviour rather than either mechanism, so it holds
        // whichever one is removed.
        let (url, server) = serving(vec![(404, b"not found".to_vec())]);

        let error = get(&format!("{url}/keys/0xdeadbeef")).unwrap_err();
        assert!(error.contains("404"), "names the status: {error}");
        assert!(
            !error.contains("not found"),
            "the error must not be the body dressed up as one: {error}"
        );
        server.join().unwrap();
    }

    #[test]
    fn post_delivers_its_body_and_reports_a_refusal() {
        // 503 is what a worker returns when its job queue is full (task 2.9b).
        // The coordinator has to see that as a failed dispatch, not a delivery.
        let (url, server) = serving(vec![(200, b"ok".to_vec()), (503, b"queue full".to_vec())]);

        post(&format!("{url}/jobs"), b"first dispatch".to_vec()).unwrap();
        let error = post(&format!("{url}/jobs"), b"second dispatch".to_vec()).unwrap_err();
        assert!(error.contains("503"), "names the status: {error}");

        let received = server.join().unwrap();
        assert_eq!(
            received,
            vec![b"first dispatch".to_vec(), b"second dispatch".to_vec()],
            "the server must have received exactly what was posted"
        );
    }

    #[test]
    fn an_unreachable_peer_is_an_error_rather_than_a_hang() {
        // A worker that has gone away must surface inside the coordinator's
        // deadline. Port 1 on loopback refuses immediately.
        assert!(get("http://127.0.0.1:1/keys/0x00").is_err());
        assert!(post("http://127.0.0.1:1/results", b"report".to_vec()).is_err());
    }
}
