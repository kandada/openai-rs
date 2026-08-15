// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async Server-Sent-Events (SSE) line framing.
//!
//! Accumulates bytes from a `reqwest` streaming response and yields
//! `data:` payloads one at a time. Handles UTF-8 BOM stripping.

use std::str;

use futures::StreamExt;
use crate::error::OpenAiError;

/// An async SSE data payload iterator.
pub struct AsyncSseStream<S> {
    stream: S,
    buffer: Vec<u8>,
    done: bool,
    first_read: bool,
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
            done: false,
            first_read: true,
        }
    }

    /// Return the next `data:` payload.
    pub async fn next_data(&mut self) -> Result<Option<String>, OpenAiError> {
        if self.done {
            return Ok(None);
        }

        loop {
            // Try to extract a complete line from the buffer.
            if let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
                let line_bytes = self.buffer[..pos].to_vec();
                self.buffer.drain(..=pos); // remove line + newline

                let line = String::from_utf8_lossy(&line_bytes).into_owned();
                let line = line.trim_end_matches('\r').to_string();

                // UTF-8 BOM stripping on first read.
                if self.first_read {
                    self.first_read = false;
                    if line.starts_with('\u{FEFF}') {
                        let stripped = line[3..].to_string();
                        if stripped.trim_end_matches(['\r', '\n']).is_empty() {
                            continue;
                        }
                        if let Some(payload) = Self::extract_payload(&stripped) {
                            if payload == "[DONE]" {
                                self.done = true;
                                return Ok(None);
                            }
                            return Ok(Some(payload));
                        }
                        continue;
                    }
                    if line.trim_end_matches(['\r', '\n']).is_empty() {
                        continue;
                    }
                }

                let trimmed = line.trim_end_matches(['\r', '\n']);
                if trimmed.is_empty() {
                    continue;
                }

                if let Some(payload) = Self::extract_payload(trimmed) {
                    if payload == "[DONE]" {
                        self.done = true;
                        return Ok(None);
                    }
                    return Ok(Some(payload));
                }
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
                    // Stream ended. Process any remaining data.
                    if !self.buffer.is_empty() {
                        let line_bytes = std::mem::take(&mut self.buffer);
                        let line = String::from_utf8_lossy(&line_bytes).into_owned();
                        let line = line.trim_end_matches(['\r', '\n']).to_string();
                        if let Some(payload) = Self::extract_payload(&line) {
                            if payload == "[DONE]" {
                                self.done = true;
                                return Ok(None);
                            }
                            return Ok(Some(payload));
                        }
                    }
                    self.done = true;
                    return Ok(None);
                }
            }
        }
    }

    fn extract_payload(line: &str) -> Option<String> {
        if let Some(rest) = line.strip_prefix("data:") {
            Some(rest.strip_prefix(' ').unwrap_or(rest).to_string())
        } else {
            None
        }
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
}