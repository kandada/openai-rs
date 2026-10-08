// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Edge-case tests for openai-rs.
//!
//! Covers wire-format corner cases and failure paths happy-path tests miss:
//! empty/keep-alive streams, chunks without choices/delta, out-of-order tool
//! call fragments, unclosed/multiple <think> tags, refusal/extra fields,
//! retry classification matrix, and HTTP retry edge behaviour.

use std::io::Cursor;
use std::sync::atomic::AtomicBool;

use serde_json::json;

use openai_client_rs::*;

fn parse_stream(raw: &str) -> LlmResponse {
    let cancel = AtomicBool::new(false);
    parse_openai_stream(
        Cursor::new(raw.as_bytes().to_vec()),
        |_| {},
        |_, _| {},
        &cancel,
    )
    .unwrap()
}

// ── Empty / degenerate streams ─────────────────────────────────────────────

#[test]
fn empty_stream_is_ok_with_empty_response() {
    let resp = parse_stream("data: [DONE]\n\n");
    assert!(resp.text.is_empty());
    assert!(resp.tool_calls.is_empty());
    assert!(resp.reasoning_content.is_none());
    assert!(resp.usage.is_none());
}

#[test]
fn chunk_without_choices_is_skipped() {
    let raw = concat!(
        "data: {\"object\":\"chat.completion.chunk\"}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.text, "hi");
}

#[test]
fn chunk_with_choices_but_no_delta_is_skipped() {
    let raw = concat!(
        "data: {\"choices\":[{}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.text, "ok");
}

#[test]
fn finish_reason_in_same_chunk_as_content() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"final\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.text, "final");
    assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
}

// ── Tool call fragments ────────────────────────────────────────────────────

#[test]
fn tool_call_without_index_defaults_to_zero() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"name\":\"run\",\"arguments\":\"{}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "run");
    assert_eq!(resp.tool_calls[0].id, "call_0");
}

#[test]
fn multiple_tool_calls_in_one_delta() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"f1\",\"arguments\":\"{}\"}},{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"f2\",\"arguments\":\"{}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.tool_calls.len(), 2);
    assert_eq!(resp.tool_calls[0].name, "f1");
    assert_eq!(resp.tool_calls[1].name, "f2");
}

#[test]
fn role_only_chunk_does_not_pollute_text() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.text, "hi");
}

// ── <think> tag edge cases ─────────────────────────────────────────────────

#[test]
fn unclosed_think_tag_in_stream_flushed_as_thinking() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"<think>still thinking\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.reasoning_content.as_deref(), Some("still thinking"));
    assert_eq!(resp.text, "");
}

#[test]
fn multiple_think_blocks_in_stream() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"<think>a</think>X<think>b</think>Y\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.reasoning_content.as_deref(), Some("ab"));
    assert_eq!(resp.text, "XY");
}

#[test]
fn unclosed_think_tag_in_non_streaming() {
    let raw = json!({
        "id": "x", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "<think>deep"}, "finish_reason": "stop"}]
    });
    let cc: ChatCompletion = serde_json::from_value(raw).unwrap();
    let resp = openai_client_rs::api_common::assemble_response(&cc);
    assert_eq!(resp.reasoning_content.as_deref(), Some("deep"));
    assert_eq!(resp.text, "");
}

// ── Types / serde edge cases ───────────────────────────────────────────────

#[test]
fn chat_completion_empty_choices_no_panic() {
    let raw = json!({
        "id": "x", "object": "chat.completion", "created": 1, "model": "m",
        "choices": []
    });
    let cc: ChatCompletion = serde_json::from_value(raw).unwrap();
    let resp = openai_client_rs::api_common::assemble_response(&cc);
    assert!(resp.text.is_empty());
    assert!(resp.tool_calls.is_empty());
}

#[test]
fn refusal_field_roundtrips() {
    let raw = json!({
        "id": "x", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": null, "refusal": "I can't do that"}, "finish_reason": "refusal"}]
    });
    let cc: ChatCompletion = serde_json::from_value(raw).unwrap();
    assert_eq!(
        cc.choices[0].message.refusal.as_deref(),
        Some("I can't do that")
    );
    assert_eq!(cc.choices[0].finish_reason.as_deref(), Some("refusal"));
    let resp = openai_client_rs::api_common::assemble_response(&cc);
    assert!(resp.text.is_empty()); // content is null → no panic
}

#[test]
fn system_and_developer_messages_roundtrip() {
    let msgs = vec![
        ChatMessage::system("be brief"),
        ChatMessage::developer("follow the rules"),
        ChatMessage::user("hi"),
    ];
    let out = openai_client_rs::api_common::build_messages_json(&msgs);
    assert_eq!(out[0]["role"], "system");
    assert_eq!(out[1]["role"], "developer");
    assert_eq!(out[2]["role"], "user");
}

#[test]
fn content_parts_roundtrip() {
    let msg = ChatMessage::user_with_images("look", &["https://example.com/i.png"]);
    let out = openai_client_rs::api_common::build_messages_json(&[msg]);
    let content = out[0]["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "text");
    assert_eq!(content[1]["type"], "image_url");
    assert_eq!(content[1]["image_url"]["detail"], "auto");
}

#[test]
fn tool_choice_all_variants_serde() {
    // `Auto`/`None`/`Required` serialize as bare strings.
    let auto = serde_json::to_value(ToolChoice::auto()).unwrap();
    assert_eq!(auto, "auto");
    let none = serde_json::to_value(ToolChoice::none()).unwrap();
    assert_eq!(none, "none");
    let required = serde_json::to_value(ToolChoice::required()).unwrap();
    assert_eq!(required, "required");
    // `Specific` serializes as a function-choice object.
    let specific = serde_json::to_value(ToolChoice::specific("my_fn")).unwrap();
    assert_eq!(specific["type"], "function");
    assert_eq!(specific["function"]["name"], "my_fn");
}

#[test]
fn chat_completion_chunk_with_usage_deserializes() {
    let raw = json!({
        "id": "x", "object": "chat.completion.chunk", "created": 1, "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    });
    let chunk: ChatCompletionChunk = serde_json::from_value(raw).unwrap();
    assert_eq!(chunk.usage.unwrap().total_tokens, 3);
}

#[test]
fn llm_response_default() {
    let r = LlmResponse::default();
    assert!(r.text.is_empty());
    assert!(r.tool_calls.is_empty());
    assert!(r.reasoning_content.is_none());
    assert!(r.usage.is_none());
    assert!(!r.is_truncated());
}

#[test]
fn reasoning_field_priority_all_present() {
    let raw = json!({
        "id": "x", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "answer",
            "reasoning_content": "R1", "reasoning": "R2", "thinking": "R3"
        }, "finish_reason": "stop"}]
    });
    let cc: ChatCompletion = serde_json::from_value(raw).unwrap();
    let resp = openai_client_rs::api_common::assemble_response(&cc);
    assert_eq!(resp.reasoning_content.as_deref(), Some("R1"));
    assert_eq!(resp.text, "answer");
}

// ── Retry classification matrix ─────────────────────────────────────────────

#[test]
fn logprobs_typed_deserialization() {
    let raw = json!({
        "id": "x", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
            "logprobs": {"content": [{
                "token": "hi", "logprob": -0.5, "bytes": [0, 2],
                "top_logprobs": [{"token": "hi", "logprob": -0.5, "bytes": [0, 2]}, {"token": "bye", "logprob": -3.2}]
            }]}
        }]
    });
    let cc: ChatCompletion = serde_json::from_value(raw).unwrap();
    let lp = cc.choices[0].logprobs.as_ref().unwrap();
    let content = lp.content.as_ref().unwrap();
    assert_eq!(content[0].token, "hi");
    assert_eq!(content[0].top_logprobs.len(), 2);
    assert_eq!(content[0].top_logprobs[1].token, "bye");
}

#[test]
fn retryable_status_code_matrix() {
    for code in [408u16, 409, 429, 500, 502, 503, 504] {
        assert!(
            OpenAiError::api(code, None, "x").is_retryable(),
            "code {code}"
        );
    }
    for code in [400u16, 401, 403, 404, 405, 410, 422, 451] {
        assert!(
            !OpenAiError::api(code, None, "x").is_retryable(),
            "code {code}"
        );
    }
    assert!(OpenAiError::Network("reset".into()).is_retryable());
    assert!(!OpenAiError::Json("bad".into()).is_retryable());
    assert!(!OpenAiError::Io("read failed".into()).is_retryable());
    assert!(!OpenAiError::Cancelled.is_retryable());
}

// ── HTTP retry edge behaviour (tiny_http) ──────────────────────────────────

const OK_BODY: &str = r#"{"id":"c","object":"chat.completion","created":1,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#;

fn respond(req: tiny_http::Request, body: &str, status: u16, retry_after: Option<&str>) {
    let mut resp = tiny_http::Response::from_string(body.to_string()).with_status_code(status);
    if let Some(ra) = retry_after {
        resp =
            resp.with_header(tiny_http::Header::from_bytes(b"Retry-After", ra.as_bytes()).unwrap());
    }
    req.respond(resp).unwrap();
}

fn port_of(server: &tiny_http::Server) -> u16 {
    match server.server_addr() {
        tiny_http::ListenAddr::IP(a) => a.port(),
        _ => panic!(),
    }
}

fn client(base: String, retries: u32) -> OpenAiClient {
    OpenAiClient::with_base_url("sk-test", "m", base).with_retry_config(RetryConfig {
        max_retries: retries,
        base_delay_ms: 1,
        max_delay_ms: 10,
    })
}

#[test]
fn retries_on_408_request_timeout() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = port_of(&server);
    let handle = std::thread::spawn(move || {
        respond(server.recv().unwrap(), "timeout", 408, None);
        respond(server.recv().unwrap(), OK_BODY, 200, None);
    });
    let resp = client(format!("http://127.0.0.1:{port}"), 2)
        .chat_create(&[ChatMessage::user("hi")], None)
        .unwrap();
    assert_eq!(resp.text, "ok");
    handle.join().unwrap();
}

#[test]
fn retry_exhaustion_returns_last_error() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = port_of(&server);
    let handle = std::thread::spawn(move || {
        respond(server.recv().unwrap(), "unavailable", 503, None);
        respond(server.recv().unwrap(), "unavailable", 503, None);
    });
    let err = client(format!("http://127.0.0.1:{port}"), 1)
        .chat_create(&[ChatMessage::user("hi")], None)
        .unwrap_err();
    match &err {
        OpenAiError::Api(ae) => assert_eq!(ae.status_code, Some(503)),
        other => panic!("expected Api 503, got {other:?}"),
    }
    handle.join().unwrap();
}

#[test]
fn max_retries_zero_never_retries() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = port_of(&server);
    let handle = std::thread::spawn(move || {
        respond(server.recv().unwrap(), "unavailable", 503, None);
    });
    let err = client(format!("http://127.0.0.1:{port}"), 0)
        .chat_create(&[ChatMessage::user("hi")], None)
        .unwrap_err();
    match &err {
        OpenAiError::Api(ae) => assert_eq!(ae.status_code, Some(503)),
        other => panic!("expected Api 503, got {other:?}"),
    }
    handle.join().unwrap();
}
