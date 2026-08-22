// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Comprehensive end-to-end tests for openai-client-rs.
//!
//! Grounded in the OpenAI-compatible wire format plus the provider
//! conventions the library targets:
//!   - reasoning via `reasoning_content` (DeepSeek/Kimi), `reasoning`
//!     (some private providers), `thinking` (some private providers)
//!   - inline `<think>...</think>` tags in `content` (Qwen family)
//!   - `stream_options.include_usage` final chunk (empty choices + usage)
//!   - `reasoning_effort` / `max_completion_tokens` for reasoning models
//!
//! Also covers HTTP retry behaviour (429/5xx retried, Retry-After honored,
//! 4xx not retried) against a real local server.

use std::io::Cursor;
use std::sync::atomic::AtomicBool;

use serde_json::json;

use openai_client_rs::*;

fn parse_stream(raw: &str) -> LlmResponse {
    let cancel = AtomicBool::new(false);
    parse_openai_stream(Cursor::new(raw.as_bytes().to_vec()), |_| {}, |_, _| {}, &cancel).unwrap()
}

// ── Full streaming conversation ─────────────────────────────────────────────

const FULL_STREAM: &str = concat!(
    // role announcement
    "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
    // reasoning
    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Let me \"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think.\"}}]}\n\n",
    // content
    "data: {\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\n\n",
    // fragmented tool call
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"run_shell\",\"arguments\":\"{\\\"cmd\\\":\\\"\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ls\\\"}\"}}]}}]}\n\n",
    // finish
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    // usage chunk (include_usage)
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":12,\"total_tokens\":22}}\n\n",
    "data: [DONE]\n\n"
);

#[test]
fn full_stream_reasoning_accumulates_and_tool_assembled() {
    let resp = parse_stream(FULL_STREAM);
    assert_eq!(resp.text, "Hello world");
    assert_eq!(resp.reasoning_content.as_deref(), Some("Let me think."));
    assert_eq!(resp.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    assert_eq!(resp.tool_calls[0].parsed_args()["cmd"], "ls");
    assert_eq!(resp.usage.unwrap().total_tokens, 22);
}

// ── Thinking field priority per chunk ───────────────────────────────────────

#[test]
fn stream_field_priority_reasoning_content_beats_others() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"R\",\"reasoning\":\"Re\",\"thinking\":\"T\",\"content\":\"C\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.reasoning_content.as_deref(), Some("R"));
    // content passes through verbatim when a field hit.
    assert_eq!(resp.text, "C");
}

#[test]
fn stream_reasoning_field_recognized() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning\":\"step 1\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"reasoning\":\"step 2\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"ans\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.reasoning_content.as_deref(), Some("step 1step 2"));
    assert_eq!(resp.text, "ans");
}

#[test]
fn stream_thinking_field_recognized() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"thinking\":\"plan\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"go\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.reasoning_content.as_deref(), Some("plan"));
    assert_eq!(resp.text, "go");
}

// ── Inline <think> tags ─────────────────────────────────────────────────────

#[test]
fn think_tags_stripped_and_split_across_chunks() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"<think>I am thi\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"nking</think>done\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.reasoning_content.as_deref(), Some("I am thinking"));
    assert_eq!(resp.text, "done");
}

#[test]
fn think_tags_multiple_blocks() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"<think>a</think>X<think>b</think>Y\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.reasoning_content.as_deref(), Some("ab"));
    assert_eq!(resp.text, "XY");
}

#[test]
fn unclosed_think_tag_flushed_as_thinking() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"<think>still thinking\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.reasoning_content.as_deref(), Some("still thinking"));
    assert_eq!(resp.text, "");
}

#[test]
fn think_tag_no_phantom_whitespace() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"<think>\\nplan\\n</think>\\nresult\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.reasoning_content.as_deref(), Some("plan"));
    assert_eq!(resp.text, "result");
}

// ── Usage from stream ───────────────────────────────────────────────────────

#[test]
fn stream_usage_captured_from_final_chunk() {
    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":4,\"total_tokens\":7}}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    let u = resp.usage.unwrap();
    assert_eq!(u.prompt_tokens, 3);
    assert_eq!(u.completion_tokens, 4);
    assert_eq!(u.total_tokens, 7);
}

// ── Non-streaming assembly ──────────────────────────────────────────────────

fn build_completion(content: &str) -> ChatCompletion {
    serde_json::from_value(json!({
        "id": "chatcmpl_1",
        "object": "chat.completion",
        "created": 123,
        "model": "gpt",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    })).unwrap()
}

#[test]
fn non_streaming_strips_think_tags_from_content() {
    let resp = openai_client_rs::api_common::assemble_response(&build_completion("<think>deep</think>answer"));
    assert_eq!(resp.reasoning_content.as_deref(), Some("deep"));
    assert_eq!(resp.text, "answer");
    assert_eq!(resp.usage.unwrap().total_tokens, 3);
}

#[test]
fn non_streaming_reasoning_fields_priority() {
    let raw = json!({
        "id": "x", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "R",
                "reasoning": "Re",
                "thinking": "T"
            },
            "finish_reason": "stop"
        }]
    });
    let cc: ChatCompletion = serde_json::from_value(raw).unwrap();
    let resp = openai_client_rs::api_common::assemble_response(&cc);
    assert_eq!(resp.reasoning_content.as_deref(), Some("R"));
    assert_eq!(resp.text, "answer");
}

#[test]
fn non_streaming_thinking_field_only() {
    let raw = json!({
        "id": "x", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "go", "thinking": "plan"},
            "finish_reason": "stop"
        }]
    });
    let cc: ChatCompletion = serde_json::from_value(raw).unwrap();
    let resp = openai_client_rs::api_common::assemble_response(&cc);
    assert_eq!(resp.reasoning_content.as_deref(), Some("plan"));
    assert_eq!(resp.text, "go");
}

// ── ChatCompletion deserialization with reasoning/thinking fields ──────────

#[test]
fn chat_completion_message_deserializes_extra_reasoning_fields() {
    let raw = json!({
        "id": "x", "object": "chat.completion", "created": 1, "model": "m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "c", "reasoning": "r", "thinking": "t"},
            "finish_reason": "stop"
        }]
    });
    let cc: ChatCompletion = serde_json::from_value(raw).unwrap();
    assert_eq!(cc.choices[0].message.reasoning.as_deref(), Some("r"));
    assert_eq!(cc.choices[0].message.thinking.as_deref(), Some("t"));
}

// ── Request builder: reasoning models ───────────────────────────────────────

#[test]
fn request_builder_reasoning_models_use_max_completion_tokens() {
    let body = ChatCompletionRequest::new("o3", vec![ChatMessage::user("hi")])
        .max_tokens(500)
        .max_completion_tokens(2000)
        .reasoning_effort("high")
        .build_body();
    // OpenAI requires only one of max_tokens / max_completion_tokens.
    assert_eq!(body["max_completion_tokens"], 2000);
    assert!(body.get("max_tokens").is_none());
    assert_eq!(body["reasoning_effort"], "high");
}

#[test]
fn request_builder_stream_options_include_usage() {
    let body = ChatCompletionRequest::new("gpt-4o", vec![ChatMessage::user("hi")])
        .stream(true)
        .stream_options(StreamOptions { include_usage: Some(true) })
        .build_body();
    assert_eq!(body["stream_options"]["include_usage"], true);
}

// ── Error structure ─────────────────────────────────────────────────────────

#[test]
fn errors_carry_status_code_and_retry_after() {
    let e = OpenAiError::api(429, Some(5), "rate limited");
    match &e {
        OpenAiError::Api(ae) => {
            assert_eq!(ae.status_code, Some(429));
            assert_eq!(ae.retry_after_secs, Some(5));
            assert!(ae.is_retryable());
        }
        _ => panic!("expected Api"),
    }
    assert_eq!(e.retry_after_secs(), Some(5));
    assert!(!OpenAiError::api(400, None, "bad").is_retryable());
    assert!(!OpenAiError::Json("bad".into()).is_retryable());
}

// ── SSE framing edge cases ──────────────────────────────────────────────────

#[test]
fn sse_bom_crlf_and_comments_are_handled() {
    let raw = concat!(
        "\u{FEFF}data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\r\n\r\n",
        ": keep-alive\r\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}\r\n\r\n",
        "data: [DONE]\r\n\r\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.text, "AB");
}

#[test]
fn stream_ignores_non_delta_chunks() {
    // A chunk without choices (e.g. usage-less intermediate) is skipped.
    let raw = concat!(
        "data: {\"object\":\"chat.completion.chunk\"}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let resp = parse_stream(raw);
    assert_eq!(resp.text, "hi");
}

// ── HTTP retry behaviour (tiny_http) ────────────────────────────────────────

const COMPLETION_200_BODY: &str = r#"{"id":"chatcmpl_1","object":"chat.completion","created":123,"model":"gpt","choices":[{"index":0,"message":{"role":"assistant","content":"server ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#;

fn respond_once(req: tiny_http::Request, body: &str, status: u16, retry_after: Option<&str>) {
    let mut resp = tiny_http::Response::from_string(body.to_string()).with_status_code(status);
    if let Some(ra) = retry_after {
        resp = resp.with_header(tiny_http::Header::from_bytes(b"Retry-After", ra.as_bytes()).unwrap());
    }
    req.respond(resp).unwrap();
}

fn bind_port(server: &tiny_http::Server) -> u16 {
    match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr.port(),
        _ => panic!("unexpected addr"),
    }
}

#[test]
fn chat_retries_on_429_then_succeeds() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        respond_once(server.recv().unwrap(), "rate limited", 429, None);
        respond_once(server.recv().unwrap(), COMPLETION_200_BODY, 200, None);
    });

    let client = OpenAiClient::with_base_url(
        "sk-test", "gpt-4o", format!("http://127.0.0.1:{port}"),
    )
    .with_retry_config(RetryConfig { max_retries: 2, base_delay_ms: 1, max_delay_ms: 10 });

    let resp = client.chat_create(&[ChatMessage::user("hi")], None).unwrap();
    assert_eq!(resp.text, "server ok");
    handle.join().unwrap();
}

#[test]
fn chat_retries_on_503_and_honors_retry_after() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        respond_once(server.recv().unwrap(), "unavailable", 503, Some("0"));
        respond_once(server.recv().unwrap(), COMPLETION_200_BODY, 200, None);
    });

    let client = OpenAiClient::with_base_url(
        "sk-test", "gpt-4o", format!("http://127.0.0.1:{port}"),
    )
    .with_retry_config(RetryConfig { max_retries: 2, base_delay_ms: 1000, max_delay_ms: 5000 });

    let resp = client.chat_create(&[ChatMessage::user("hi")], None).unwrap();
    assert_eq!(resp.text, "server ok");
    handle.join().unwrap();
}

#[test]
fn chat_does_not_retry_on_400() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        respond_once(server.recv().unwrap(), "bad request", 400, None);
    });

    let client = OpenAiClient::with_base_url(
        "sk-test", "gpt-4o", format!("http://127.0.0.1:{port}"),
    )
    .with_retry_config(RetryConfig { max_retries: 3, base_delay_ms: 1000, max_delay_ms: 5000 });

    let err = client.chat_create(&[ChatMessage::user("hi")], None).unwrap_err();
    match &err {
        OpenAiError::Api(ae) => assert_eq!(ae.status_code, Some(400)),
        _ => panic!("expected Api 400, got {err:?}"),
    }
    handle.join().unwrap();
}

#[test]
fn chat_stream_retries_on_5xx() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        respond_once(server.recv().unwrap(), "boom", 500, None);
        let body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"streamed\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        respond_once(server.recv().unwrap(), body, 200, None);
    });

    let client = OpenAiClient::with_base_url(
        "sk-test", "gpt-4o", format!("http://127.0.0.1:{port}"),
    )
    .with_retry_config(RetryConfig { max_retries: 2, base_delay_ms: 1, max_delay_ms: 10 });

    let mut deltas = Vec::new();
    let resp = client
        .chat_stream(&[ChatMessage::user("hi")], None, |d| deltas.push(d.to_string()), |_, _| {})
        .unwrap();
    assert_eq!(resp.text, "streamed");
    assert_eq!(deltas.join(""), "streamed");
    handle.join().unwrap();
}

#[test]
fn chat_send_roundtrips_through_request_builder() {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        respond_once(server.recv().unwrap(), COMPLETION_200_BODY, 200, None);
    });

    let client = OpenAiClient::with_base_url("sk-test", "o3", format!("http://127.0.0.1:{port}"));
    let req = ChatCompletionRequest::new("o3", vec![ChatMessage::user("hi")])
        .reasoning_effort("high")
        .max_completion_tokens(2000);
    let resp = client.send(&req).unwrap();
    assert_eq!(resp.text, "server ok");
    handle.join().unwrap();
}

// ── Async client HTTP retry behaviour ───────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test]
async fn async_chat_retries_on_429_then_succeeds() {
    use openai_client_rs::OpenAiAsyncClient;

    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        respond_once(server.recv().unwrap(), "rate limited", 429, None);
        respond_once(server.recv().unwrap(), COMPLETION_200_BODY, 200, None);
    });

    let client = OpenAiAsyncClient::with_base_url(
        "sk-test", "gpt-4o", format!("http://127.0.0.1:{port}"),
    )
    .with_retry_config(RetryConfig { max_retries: 2, base_delay_ms: 1, max_delay_ms: 10 });

    let resp = client.chat_create(&[ChatMessage::user("hi")], None).await.unwrap();
    assert_eq!(resp.text, "server ok");
    handle.join().unwrap();
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_chat_does_not_retry_on_400() {
    use openai_client_rs::OpenAiAsyncClient;

    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        respond_once(server.recv().unwrap(), "bad request", 400, None);
    });

    let client = OpenAiAsyncClient::with_base_url(
        "sk-test", "gpt-4o", format!("http://127.0.0.1:{port}"),
    )
    .with_retry_config(RetryConfig { max_retries: 3, base_delay_ms: 1000, max_delay_ms: 5000 });

    let err = client.chat_create(&[ChatMessage::user("hi")], None).await.unwrap_err();
    match &err {
        OpenAiError::Api(ae) => assert_eq!(ae.status_code, Some(400)),
        _ => panic!("expected Api 400, got {err:?}"),
    }
    handle.join().unwrap();
}

// ── Token counting ──────────────────────────────────────────────────────────

#[test]
fn count_messages_handles_tool_calls_and_parts() {
    let msgs = vec![
        ChatMessage::user("hello world"),
        ChatMessage::assistant_with_tools(
            "",
            vec![ToolCall {
                id: "c1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "run_shell".into(),
                    arguments: r#"{"cmd":"ls"}"#.into(),
                },
            }],
        ),
        ChatMessage::tool_result("c1", "output"),
    ];
    let n = openai_client_rs::tokens::count_messages_tokens(&msgs);
    assert!(n > 0);
}
