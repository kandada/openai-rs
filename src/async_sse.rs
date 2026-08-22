// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async Server-Sent-Events (SSE) line framing, spec-compliant.
//!
//! Accumulates bytes from a `reqwest` streaming response and yields one
//! `data:` payload per event. Per the SSE spec, consecutive `data:` lines
//! within one event are joined with `\n` and dispatched by a blank line.
//! Handles UTF-8 BOM stripping.
//!
//! The buffer uses a read-offset cursor and only compacts occasionally, so
//! draining a stream where many lines arrive inside a single network chunk
//! stays amortized O(1) per line (never O(n²)).

use std::str;

use futures::StreamExt;
use crate::error::OpenAiError;

/// An async SSE data payload iterator.
pub struct AsyncSseStream<S> {
    stream: S,
    buffer: Vec<u8>,
    /// Number of leading bytes already consumed from `buffer`.
    read: usize,
    done: bool,
    first_read: bool,
    /// Accumulated `data:` lines of the event currently being built.
    pending: Option<String>,
}

impl<S> AsyncSseStream<S>
where
    S: futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    /// Create from a reqwest response bytes stream.
    pub fn new(stream: S) -> Self {
        AsyncSseStream {
            stream,
            buffer: Vec::new(),
            read: 0,
            done: false,
            first_read: true,
            pending: None,
        }
    }

    /// Return the next `data:` payload.
    pub async fn next_data(&mut self) -> Result<Option<String>, OpenAiError> {
        if self.done {
            return Ok(None);
        }

        loop {
            // Try to extract a complete line from the unconsumed region.
            if let Some(rel) = self.buffer[self.read..].iter().position(|&b| b == b'\n') {
                let pos = self.read + rel;
                let line_bytes = self.buffer[self.read..pos].to_vec();
                self.read = pos + 1; // consume line + newline

                // Compact the buffer only occasionally so the total amount of
                // memory moved stays linear in the stream size.
                if self.read > 65_536 && self.read > self.buffer.len() / 2 {
                    self.buffer.drain(..self.read);
                    self.read = 0;
                }

                let line = String::from_utf8_lossy(&line_bytes).into_owned();
                let line = line.trim_end_matches('\r').to_string();

                // UTF-8 BOM stripping on first read.
                if self.first_read {
                    self.first_read = false;
                    if line.starts_with('\u{FEFF}') {
                        let stripped = line[3..].to_string();
                        let trimmed = stripped.trim_end_matches(['\r', '\n']);
                        if trimmed.is_empty() {
                            continue;
                        }
                        self.accumulate(trimmed);
                        continue;
                    }
                }

                let trimmed = line.trim_end_matches(['\r', '\n']);
                if trimmed.is_empty() {
                    // Event boundary: dispatch accumulated data lines.
                    if let Some(p) = self.pending.take() {
                        if p == "[DONE]" {
                            self.done = true;
                            return Ok(None);
                        }
                        return Ok(Some(p));
                    }
                    continue;
                }
                self.accumulate(trimmed);
                continue;
            }

            // Need more data from the stream.
            match self.stream.next().await {
                Some(Ok(chunk)) => {
                    self.buffer.extend_from_slice(&chunk);
                    continue;
                }
                Some(Err(e)) => {
                    return Err(OpenAiError::Network(format!("SSE stream error: {e}")));
                }
                None => {
                    // Stream ended: process any remaining partial line (one
                    // that had no trailing `\n`), then flush the last event.
                    if self.read < self.buffer.len() {
                        let rest =
                            String::from_utf8_lossy(&self.buffer[self.read..]).into_owned();
                        let trimmed = rest.trim_end_matches(['\r', '\n']);
                        if !trimmed.is_empty() {
                            self.accumulate(trimmed);
                        }
                        self.read = self.buffer.len();
                    }
                    self.done = true;
                    if self.pending.as_deref() == Some("[DONE]") {
                        self.pending = None;
                        return Ok(None);
                    }
                    return Ok(self.pending.take());
                }
            }
        }
    }

    fn accumulate(&mut self, trimmed: &str) {
        if let Some(rest) = trimmed.strip_prefix("data:") {
            let value = rest.strip_prefix(' ').unwrap_or(rest);
            match &mut self.pending {
                Some(acc) => {
                    acc.push('\n');
                    acc.push_str(value);
                }
                None => self.pending = Some(value.to_string()),
            }
        }
        // All other fields (event:, id:, retry:) and comments are ignored.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use std::pin::Pin;

    type BytesResult = std::result::Result<bytes::Bytes, reqwest::Error>;

    fn make_stream(data: &'static str) -> Pin<Box<dyn futures::Stream<Item = BytesResult> + Send>> {
        let bytes = bytes::Bytes::from(data);
        Box::pin(stream::once(async move { Ok(bytes) }))
    }

    #[tokio::test]
    async fn parses_data_lines() {
        let raw = "data: {\"a\":1}\n\ndata: {\"b\":2}\n\ndata: [DONE]\n\n";
        let mut r = AsyncSseStream::new(make_stream(raw));
        assert_eq!(r.next_data().await.unwrap().unwrap(), "{\"a\":1}");
        assert_eq!(r.next_data().await.unwrap().unwrap(), "{\"b\":2}");
        assert!(r.next_data().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn eof_without_done() {
        let raw = "data: a\n\ndata: b\n";
        let mut r = AsyncSseStream::new(make_stream(raw));
        assert_eq!(r.next_data().await.unwrap().unwrap(), "a");
        assert_eq!(r.next_data().await.unwrap().unwrap(), "b");
        assert!(r.next_data().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn utf8_bom_is_stripped() {
        let mut bytes = vec![0xEFu8, 0xBB, 0xBF];
        bytes.extend_from_slice(b"data: hello\n\n");
        let s = Box::pin(stream::once(async move { Ok(bytes::Bytes::from(bytes)) }));
        let mut r = AsyncSseStream::new(s);
        assert_eq!(r.next_data().await.unwrap().unwrap(), "hello");
        assert!(r.next_data().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn multiple_data_lines_joined_with_newline() {
        let raw = "data: line1\ndata: line2\n\n";
        let mut r = AsyncSseStream::new(make_stream(raw));
        assert_eq!(r.next_data().await.unwrap().unwrap(), "line1\nline2");
        assert!(r.next_data().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn eof_without_trailing_blank_line_dispatches() {
        let raw = "data: first\n\ndata: second";
        let mut r = AsyncSseStream::new(make_stream(raw));
        assert_eq!(r.next_data().await.unwrap().unwrap(), "first");
        assert_eq!(r.next_data().await.unwrap().unwrap(), "second");
        assert!(r.next_data().await.unwrap().is_none());
    }

    /// Regression: a single chunk holding many lines must not become O(n²).
    #[tokio::test]
    async fn many_lines_in_one_chunk_is_linear() {
        use std::time::{Duration, Instant};

        let mut data = String::with_capacity(1_000_000);
        for _ in 0..50_000 {
            data.push_str("data: x\n\n");
        }
        let s = Box::pin(stream::once(async move { Ok(bytes::Bytes::from(data)) }));
        let mut r = AsyncSseStream::new(s);

        let start = Instant::now();
        let mut n = 0usize;
        while r.next_data().await.unwrap().is_some() {
            n += 1;
        }
        let elapsed = start.elapsed();
        assert_eq!(n, 50_000);
        assert!(elapsed < Duration::from_secs(3), "50k lines in one chunk took {elapsed:?}");
    }
}
