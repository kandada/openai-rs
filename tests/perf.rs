// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Performance / scale tests for openai-rs.
//!
//! Guards against pathological (e.g. O(n²)) regressions in the hot paths:
//! SSE streaming, fragmented tool-call reassembly, <think> tag parsing,
//! message building, token counting, retry backoff — and the async SSE
//! reader when a server delivers many lines inside a single network chunk
//! (a case that can easily become quadratic).

use std::io::Cursor;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use openai_client_rs::*;

fn parse_stream(raw: &str) -> LlmResponse {
    let cancel = AtomicBool::new(false);
    parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), |_| {}, |_, _| {}, &cancel).unwrap()
}

fn big_text_stream(n: usize) -> String {
    let mut s = String::with_capacity(n * 70 + 16);
    for _ in 0..n {
        s.push_str("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n");
    }
    s.push_str("data: [DONE]\n\n");
    s
}

fn big_tool_stream(n_calls: usize) -> String {
    let mut s = String::with_capacity(n_calls * 300);
    for i in 0..n_calls {
        let arg = format!("{{\"cmd\":\"echo {i}\"}}");
        let mid = arg.len() / 2;
        s.push_str(&format!(
            "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":{i},\"id\":\"c{i}\",\"function\":{{\"name\":\"run\",\"arguments\":\"{}\"}}}}]}}}}]}}\n\n",
            escape_json(&arg[..mid])
        ));
        s.push_str(&format!(
            "data: {{\"choices\":[{{\"delta\":{{\"tool_calls\":[{{\"index\":{i},\"function\":{{\"arguments\":\"{}\"}}}}]}}}}]}}\n\n",
            escape_json(&arg[mid..])
        ));
    }
    s.push_str("data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n");
    s.push_str("data: [DONE]\n\n");
    s
}

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[test]
fn streams_100k_chunks() {
    let raw = big_text_stream(100_000);
    let start = Instant::now();
    let resp = parse_stream(&raw);
    let elapsed = start.elapsed();
    assert_eq!(resp.text.len(), 100_000);
    assert!(elapsed < Duration::from_secs(5), "100k chunks took {elapsed:?}");
    println!("100k chunks parsed in {elapsed:?}");
}

#[test]
fn reassembles_many_parallel_tool_calls_from_fragments() {
    let raw = big_tool_stream(500);
    let start = Instant::now();
    let resp = parse_stream(&raw);
    let elapsed = start.elapsed();
    assert_eq!(resp.tool_calls.len(), 500);
    for (i, tc) in resp.tool_calls.iter().enumerate() {
        assert_eq!(tc.name, "run");
        assert_eq!(tc.id, format!("c{i}"));
        assert_eq!(tc.parsed_args()["cmd"], format!("echo {i}"));
    }
    assert!(elapsed < Duration::from_secs(5), "500 parallel tool calls took {elapsed:?}");
    println!("500 fragmented tool calls reassembled in {elapsed:?}");
}

#[test]
fn think_tags_over_large_content() {
    // 5k interleaved <think> blocks in a large content string.
    let mut content = String::with_capacity(1_000_000);
    for i in 0..5_000 {
        content.push_str(&format!("<think>thinking block {i} with padding padding padding</think>answer {i} "));
    }
    let start = Instant::now();
    let (think, text) = openai_client_rs::thinking::split_thinking(&content);
    let elapsed = start.elapsed();
    assert!(think.starts_with("thinking block 0"));
    assert!(text.contains("answer 4999"));
    assert!(elapsed < Duration::from_secs(5), "1MB tag parse took {elapsed:?}");
    println!("1MB <think> tag content split in {elapsed:?}");
}

#[test]
fn builds_large_message_history() {
    let mut msgs = Vec::with_capacity(2_000);
    for i in 0..2_000 {
        match i % 3 {
            0 => msgs.push(ChatMessage::system(format!("system {i}"))),
            1 => msgs.push(ChatMessage::user(format!("user {i}")),
            ),
            _ => msgs.push(ChatMessage::assistant(format!("assistant {i}"))),
        }
    }
    let start = Instant::now();
    let out = openai_client_rs::api_common::build_messages_json(&msgs);
    let elapsed = start.elapsed();
    assert_eq!(out.len(), 2_000);
    assert!(elapsed < Duration::from_secs(5), "2k-message build took {elapsed:?}");
    println!("2k-message build in {elapsed:?}");
}

#[test]
fn counts_tokens_on_large_text() {
    let big = "the quick brown fox jumps over the lazy dog ".repeat(20_000);
    let start = Instant::now();
    let n = openai_client_rs::tokens::count_tokens(&big);
    let elapsed = start.elapsed();
    assert!(n > 0);
    assert!(elapsed < Duration::from_secs(5), "1MB token count took {elapsed:?}");
    println!("1MB token count ({n} tokens) in {elapsed:?}");
}

#[test]
fn retry_delay_many_invocations() {
    use openai_client_rs::retry::RetryConfig;
    let cfg = RetryConfig::default();
    let start = Instant::now();
    let mut acc = 0u64;
    for attempt in 0..100_000u32 {
        acc = acc.wrapping_add(cfg.delay_ms(attempt % 32));
    }
    let elapsed = start.elapsed();
    assert!(acc > 0);
    assert!(elapsed < Duration::from_secs(2), "100k delay_ms took {elapsed:?}");
    println!("100k retry delay computations in {elapsed:?}");
}

#[test]
fn non_streaming_assemble_large_response() {
    let mut choices = Vec::new();
    let mut text = String::new();
    for i in 0..5_000 {
        text.push_str(&format!("block {i} "));
    }
    choices.push(serde_json::json!({
        "index": 0,
        "message": {"role": "assistant", "content": text, "reasoning_content": "r".repeat(1_000)},
        "finish_reason": "stop"
    }));
    let raw = serde_json::json!({
        "id": "c", "object": "chat.completion", "created": 1, "model": "m",
        "choices": choices,
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    });
    let cc: ChatCompletion = serde_json::from_value(raw).unwrap();
    let start = Instant::now();
    let resp = openai_client_rs::api_common::assemble_response(&cc);
    let elapsed = start.elapsed();
    assert!(resp.text.starts_with("block 0"));
    assert!(elapsed < Duration::from_secs(5), "large assembly took {elapsed:?}");
    println!("large non-streaming assembly in {elapsed:?}");
}

// ── Async SSE reader: many lines inside ONE network chunk ──────────────────
// A server may coalesce hundreds of thousands of SSE lines into a single
// TCP segment. The reader must not become quadratic in that case.

#[cfg(feature = "async")]
#[tokio::test]
async fn async_sse_many_lines_in_one_chunk() {
    use futures::stream;
    use openai_client_rs::async_sse::AsyncSseStream;

    let mut data = String::with_capacity(3_000_000);
    for _ in 0..100_000 {
        data.push_str("data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n");
    }
    data.push_str("data: [DONE]\n\n");

    let stream = stream::iter(vec![Ok::<_, reqwest::Error>(bytes::Bytes::from(data))]);
    let mut sse = AsyncSseStream::new(stream);

    let start = Instant::now();
    let mut n = 0usize;
    while sse.next_data().await.unwrap().is_some() {
        n += 1;
    }
    let elapsed = start.elapsed();
    assert_eq!(n, 100_000);
    assert!(
        elapsed < Duration::from_secs(10),
        "async SSE with 100k lines in one chunk took {elapsed:?} (quadratic reader?)"
    );
    println!("async SSE: 100k lines in one chunk drained in {elapsed:?}");
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_sse_many_chunks_one_line_each() {
    use futures::stream;
    use openai_client_rs::async_sse::AsyncSseStream;

    let line = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n".to_string();
    let chunks: Vec<bytes::Bytes> = (0..50_000).map(|_| bytes::Bytes::from(line.clone())).collect();
    let stream = stream::iter(chunks.into_iter().map(Ok::<_, reqwest::Error>));
    let mut sse = AsyncSseStream::new(stream);

    let start = Instant::now();
    let mut n = 0usize;
    while sse.next_data().await.unwrap().is_some() {
        n += 1;
    }
    let elapsed = start.elapsed();
    assert_eq!(n, 50_000);
    assert!(elapsed < Duration::from_secs(10), "50k chunked lines took {elapsed:?}");
    println!("async SSE: 50k one-line chunks drained in {elapsed:?}");
}
