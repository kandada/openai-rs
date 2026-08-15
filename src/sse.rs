// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Minimal Server-Sent-Events (SSE) line framing.
//!
//! Reads an underlying byte stream and yields the payload of each `data:`
//! field. Terminates the iteration when it sees `data: [DONE]` or EOF.
//! Comment lines (`:`) and other SSE fields (event:, id:, retry:) are ignored,
//! which is sufficient for OpenAI chat streaming.
//!
//! `next_data()` returns `Result<Option<String>>`:
//!   - `Ok(Some(payload))` — a `data:` line payload
//!   - `Ok(None)` — normal stream end (`[DONE]` or clean EOF)
//!   - `Err(e)` — underlying read error (timeout, connection reset, etc.)

use std::io::{BufRead, BufReader, Read};

/// Iterator-like reader over SSE `data:` payloads.
pub struct SseReader<R: Read> {
    inner: BufReader<R>,
    done: bool,
    first_read: bool,
}

impl<R: Read> SseReader<R> {
    pub fn new(reader: R) -> Self {
        SseReader {
            inner: BufReader::new(reader),
            done: false,
            first_read: true,
        }
    }

    /// Return the next `data:` payload.
    ///
    /// Returns `Ok(None)` at natural stream end (`[DONE]` or EOF).
    /// Returns `Err` when the underlying reader encounters an I/O error
    /// (e.g. socket read timeout, connection reset). Callers MUST propagate
    /// this error — a mid-stream disconnect is not the same as `[DONE]`.
    pub fn next_data(&mut self) -> std::io::Result<Option<String>> {
        if self.done {
            return Ok(None);
        }
        let mut line = String::new();
        loop {
            line.clear();
            let n = self.inner.read_line(&mut line)?;
            if n == 0 {
                self.done = true;
                return Ok(None);
            }
            // Strip UTF-8 BOM (0xEF 0xBB 0xBF) if present on the first line.
            if self.first_read {
                self.first_read = false;
                if line.starts_with('\u{FEFF}') {
                    line = line[3..].to_string();
                }
                if line.trim_end_matches(['\r', '\n']).is_empty() {
                    continue;
                }
            }
            let trimmed = line.trim_end_matches(['\r', '\n']);
            if trimmed.is_empty() {
                continue;
            }
            let payload = if let Some(rest) = trimmed.strip_prefix("data:") {
                rest.strip_prefix(' ').unwrap_or(rest)
            } else {
                continue;
            };
            if payload == "[DONE]" {
                self.done = true;
                return Ok(None);
            }
            return Ok(Some(payload.to_string()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parses_data_lines() {
        let raw = "data: {\"a\":1}\n\ndata: {\"b\":2}\n\ndata: [DONE]\n\n";
        let mut r = SseReader::new(Cursor::new(raw.as_bytes().to_vec()));
        assert_eq!(r.next_data().unwrap().unwrap(), "{\"a\":1}");
        assert_eq!(r.next_data().unwrap().unwrap(), "{\"b\":2}");
        assert!(r.next_data().unwrap().is_none());
    }

    #[test]
    fn ignores_comments_and_other_fields() {
        let raw = ": ping\nevent: message\ndata: hello\nid: 5\n\n";
        let mut r = SseReader::new(Cursor::new(raw.as_bytes().to_vec()));
        assert_eq!(r.next_data().unwrap().unwrap(), "hello");
        assert!(r.next_data().unwrap().is_none());
    }

    #[test]
    fn handles_crlf_and_no_space_after_colon() {
        let raw = "data:{\"x\":1}\r\n\r\n";
        let mut r = SseReader::new(Cursor::new(raw.as_bytes().to_vec()));
        assert_eq!(r.next_data().unwrap().unwrap(), "{\"x\":1}");
    }

    #[test]
    fn eof_without_done() {
        let raw = "data: a\n\ndata: b\n";
        let mut r = SseReader::new(Cursor::new(raw.as_bytes().to_vec()));
        assert_eq!(r.next_data().unwrap().unwrap(), "a");
        assert_eq!(r.next_data().unwrap().unwrap(), "b");
        assert!(r.next_data().unwrap().is_none());
    }

    #[test]
    fn read_error_propagates_instead_of_silent_none() {
        struct ErrorAfter(usize);
        impl Read for ErrorAfter {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.0 == 0 {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "simulated timeout",
                    ))
                } else {
                    self.0 -= 1;
                    buf[0] = b'\n';
                    Ok(1)
                }
            }
        }
        let mut r = SseReader::new(ErrorAfter(10));
        let mut last = None;
        for _ in 0..20 {
            match r.next_data() {
                Ok(None) => {
                    last = Some("none".to_string());
                    break;
                }
                Err(e) => {
                    last = Some(format!("err: {e}"));
                    break;
                }
                Ok(Some(_)) => continue,
            }
        }
        assert!(
            last.unwrap().starts_with("err:"),
            "read error must propagate, not return None"
        );
    }

    #[test]
    fn mid_stream_disconnect_is_error_not_none() {
        struct DropAfterOne {
            sent: bool,
        }
        impl Read for DropAfterOne {
            fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
                if self.sent {
                    Err(std::io::Error::new(
                        std::io::ErrorKind::ConnectionReset,
                        "connection reset",
                    ))
                } else {
                    self.sent = true;
                    let data = b"data: hello\n\n";
                    let n = data.len().min(buf.len());
                    buf[..n].copy_from_slice(&data[..n]);
                    Ok(n)
                }
            }
        }
        let mut r = SseReader::new(DropAfterOne { sent: false });
        assert_eq!(r.next_data().unwrap().unwrap(), "hello");
        assert!(
            r.next_data().is_err(),
            "second read must error after connection reset"
        );
    }

    #[test]
    fn utf8_bom_is_stripped_from_first_data_line() {
        let mut bytes = vec![0xEFu8, 0xBB, 0xBF];
        bytes.extend_from_slice(b"data: hello\n\n");
        let mut r = SseReader::new(Cursor::new(bytes));
        assert_eq!(r.next_data().unwrap().unwrap(), "hello");
        assert!(r.next_data().unwrap().is_none());
    }
}