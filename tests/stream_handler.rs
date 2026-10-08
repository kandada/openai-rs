// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Dedicated tests for the optional-callback [`StreamHandler`] API.
//!
//! Covers the design that replaced the mandatory `on_thinking` closure:
//!   - every handler callback is OPTIONAL (no forced no-op closures);
//!   - the classic two-closure API stays backward compatible — `on_delta`
//!     is a catch-all that receives thinking when no `on_thinking` is set;
//!   - a handler with `on_thinking` set separates reasoning from text;
//!   - sync + async, pure parsing + live HTTP round-trips.

use std::cell::RefCell;
use std::io::Cursor;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;

use openai_client_rs::*;

fn parse_stream_rich(raw: &str, handler: StreamHandler) -> LlmResponse {
    let cancel = AtomicBool::new(false);
    parse_openai_stream_rich(Cursor::new(raw.as_bytes().to_vec()), handler, &cancel).unwrap()
}

type DeltaBuf = Rc<RefCell<Vec<String>>>;
type ThinkingBuf = Rc<RefCell<Vec<String>>>;
type ToolBuf = Rc<RefCell<Vec<(String, String)>>>;

fn sink_vec() -> (DeltaBuf, ThinkingBuf, ToolBuf) {
    (
        Rc::new(RefCell::new(Vec::new())),
        Rc::new(RefCell::new(Vec::new())),
        Rc::new(RefCell::new(Vec::new())),
    )
}

const REASONING_STREAM: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Let me think.\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"The answer is 42.\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
    "data: [DONE]\n\n"
);

const TOOL_STREAM: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"run_shell\",\"arguments\":\"{\\\"cmd\\\":\\\"\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ls\\\"}\"}}]}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
    "data: [DONE]\n\n"
);

const THINK_TAG_STREAM: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"content\":\"<think>I am thi\"}}]}\n\n",
    "data: {\"choices\":[{\"delta\":{\"content\":\"nking</think>done\"}}]}\n\n",
    "data: [DONE]\n\n"
);

// ── Classic two-closure API stays backward compatible ──────────────────────

#[test]
fn classic_parse_delta_receives_text_and_thinking() {
    let (delta, thinking, tools) = sink_vec();
    let cancel = AtomicBool::new(false);
    let d = Rc::clone(&delta);
    let t = Rc::clone(&tools);
    let resp = parse_openai_stream(
        Cursor::new(REASONING_STREAM.as_bytes().to_vec()),
        move |s| d.borrow_mut().push(s.to_string()),
        move |n, a| t.borrow_mut().push((n.to_string(), a.to_string())),
        &cancel,
    )
    .unwrap();
    // Catch-all: thinking + text both delivered on on_delta (nothing lost).
    assert_eq!(delta.borrow().join(""), "Let me think.The answer is 42.");
    assert!(thinking.borrow().is_empty());
    assert_eq!(resp.text, "The answer is 42.");
    assert_eq!(resp.reasoning_content.as_deref(), Some("Let me think."));
}

// ── Rich handler: optional callbacks ───────────────────────────────────────

#[test]
fn rich_parse_separates_thinking_from_delta() {
    let (delta, thinking, _tools) = sink_vec();
    let d = Rc::clone(&delta);
    let t = Rc::clone(&thinking);
    let resp = parse_stream_rich(
        REASONING_STREAM,
        StreamHandler::new()
            .on_delta(move |s| d.borrow_mut().push(s.to_string()))
            .on_thinking(move |s| t.borrow_mut().push(s.to_string())),
    );
    assert_eq!(delta.borrow().join(""), "The answer is 42.");
    assert_eq!(thinking.borrow().join(""), "Let me think.");
    assert_eq!(resp.text, "The answer is 42.");
    assert_eq!(resp.reasoning_content.as_deref(), Some("Let me think."));
}

#[test]
fn rich_parse_without_thinking_falls_back_to_delta() {
    // Register only on_delta → thinking falls through, classic behaviour.
    let (delta, thinking, _tools) = sink_vec();
    let d = Rc::clone(&delta);
    let resp = parse_stream_rich(
        REASONING_STREAM,
        StreamHandler::new().on_delta(move |s| d.borrow_mut().push(s.to_string())),
    );
    assert_eq!(delta.borrow().join(""), "Let me think.The answer is 42.");
    assert!(thinking.borrow().is_empty());
    assert_eq!(resp.reasoning_content.as_deref(), Some("Let me think."));
}

#[test]
fn rich_parse_empty_handler_still_accumulates_response() {
    // No callbacks at all — the response is still fully assembled.
    let resp = parse_stream_rich(REASONING_STREAM, StreamHandler::new());
    assert_eq!(resp.text, "The answer is 42.");
    assert_eq!(resp.reasoning_content.as_deref(), Some("Let me think."));
    assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
    assert!(resp.tool_calls.is_empty());
}

#[test]
fn rich_parse_default_equals_explicit_new() {
    let a = parse_stream_rich(REASONING_STREAM, StreamHandler::default());
    let b = parse_stream_rich(REASONING_STREAM, StreamHandler::new());
    assert_eq!(a.text, b.text);
    assert_eq!(a.reasoning_content, b.reasoning_content);
    assert_eq!(a.finish_reason, b.finish_reason);
}

#[test]
fn rich_parse_only_thinking_callback() {
    let (_delta, thinking, _tools) = sink_vec();
    let t = Rc::clone(&thinking);
    let resp = parse_stream_rich(
        REASONING_STREAM,
        StreamHandler::new().on_thinking(move |s| t.borrow_mut().push(s.to_string())),
    );
    assert_eq!(thinking.borrow().join(""), "Let me think.");
    assert_eq!(resp.text, "The answer is 42.");
}

#[test]
fn rich_parse_tool_call_callback_only() {
    let (_delta, _thinking, tools) = sink_vec();
    let t = Rc::clone(&tools);
    let resp = parse_stream_rich(
        TOOL_STREAM,
        StreamHandler::new()
            .on_tool_call(move |n, a| t.borrow_mut().push((n.to_string(), a.to_string()))),
    );
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "run_shell");
    assert_eq!(resp.tool_calls[0].parsed_args()["cmd"], "ls");
    let calls = tools.borrow();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "run_shell");
}

#[test]
fn rich_parse_think_tags_routed_to_thinking() {
    let (delta, thinking, _tools) = sink_vec();
    let d = Rc::clone(&delta);
    let t = Rc::clone(&thinking);
    let resp = parse_stream_rich(
        THINK_TAG_STREAM,
        StreamHandler::new()
            .on_delta(move |s| d.borrow_mut().push(s.to_string()))
            .on_thinking(move |s| t.borrow_mut().push(s.to_string())),
    );
    // Inline <think>...</think> split across chunks → thinking callback.
    assert_eq!(thinking.borrow().join(""), "I am thinking");
    assert_eq!(delta.borrow().join(""), "done");
    assert_eq!(resp.text, "done");
    assert_eq!(resp.reasoning_content.as_deref(), Some("I am thinking"));
}

#[test]
fn rich_parse_equivalent_to_classic() {
    let cancel = AtomicBool::new(false);
    let classic = parse_openai_stream(
        Cursor::new(TOOL_STREAM.as_bytes().to_vec()),
        |_| {},
        |_, _| {},
        &cancel,
    )
    .unwrap();
    let rich = parse_stream_rich(TOOL_STREAM, StreamHandler::new());
    assert_eq!(classic.text, rich.text);
    assert_eq!(classic.reasoning_content, rich.reasoning_content);
    assert_eq!(classic.finish_reason, rich.finish_reason);
    assert_eq!(classic.tool_calls.len(), rich.tool_calls.len());
    for (a, b) in classic.tool_calls.iter().zip(rich.tool_calls.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.arguments, b.arguments);
    }
}

// ── Live HTTP round-trips (sync) ───────────────────────────────────────────

fn bind_port(server: &tiny_http::Server) -> u16 {
    match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr.port(),
        _ => panic!("unexpected addr"),
    }
}

fn server_for(body: &'static str) -> (u16, std::thread::JoinHandle<()>) {
    let server = tiny_http::Server::http("127.0.0.1:0").unwrap();
    let port = bind_port(&server);
    let handle = std::thread::spawn(move || {
        let req = server.recv().unwrap();
        let resp = tiny_http::Response::from_string(body.to_string()).with_status_code(200);
        req.respond(resp).unwrap();
    });
    (port, handle)
}

fn client(base: String) -> OpenAiClient {
    OpenAiClient::with_base_url("sk-test", "gpt-4o", base).with_retry_config(RetryConfig {
        max_retries: 0,
        base_delay_ms: 1,
        max_delay_ms: 10,
    })
}

#[test]
fn chat_stream_classic_delta_receives_thinking_over_http() {
    let (port, handle) = server_for(REASONING_STREAM);
    let mut deltas = Vec::new();
    let resp = client(format!("http://127.0.0.1:{port}"))
        .chat_stream(
            &[ChatMessage::user("hi")],
            None,
            |d| deltas.push(d.to_string()),
            |_, _| {},
        )
        .unwrap();
    assert_eq!(resp.text, "The answer is 42.");
    assert_eq!(resp.reasoning_content.as_deref(), Some("Let me think."));
    assert_eq!(deltas.join(""), "Let me think.The answer is 42.");
    handle.join().unwrap();
}

#[test]
fn chat_stream_rich_separates_over_http() {
    let (port, handle) = server_for(REASONING_STREAM);
    let deltas = Rc::new(RefCell::new(Vec::<String>::new()));
    let thinking = Rc::new(RefCell::new(Vec::<String>::new()));
    let d = Rc::clone(&deltas);
    let t = Rc::clone(&thinking);
    let resp = client(format!("http://127.0.0.1:{port}"))
        .chat_stream_rich(
            &[ChatMessage::user("hi")],
            None,
            StreamHandler::new()
                .on_delta(move |s| d.borrow_mut().push(s.to_string()))
                .on_thinking(move |s| t.borrow_mut().push(s.to_string())),
        )
        .unwrap();
    assert_eq!(resp.text, "The answer is 42.");
    assert_eq!(resp.reasoning_content.as_deref(), Some("Let me think."));
    assert_eq!(deltas.borrow().join(""), "The answer is 42.");
    assert_eq!(thinking.borrow().join(""), "Let me think.");
    handle.join().unwrap();
}

#[test]
fn chat_stream_rich_no_thinking_callback_falls_back_over_http() {
    let (port, handle) = server_for(REASONING_STREAM);
    let deltas = Rc::new(RefCell::new(Vec::<String>::new()));
    let d = Rc::clone(&deltas);
    let resp = client(format!("http://127.0.0.1:{port}"))
        .chat_stream_rich(
            &[ChatMessage::user("hi")],
            None,
            StreamHandler::new().on_delta(move |s| d.borrow_mut().push(s.to_string())),
        )
        .unwrap();
    assert_eq!(resp.text, "The answer is 42.");
    assert_eq!(deltas.borrow().join(""), "Let me think.The answer is 42.");
    handle.join().unwrap();
}

#[test]
fn chat_stream_rich_tool_calls_over_http() {
    let (port, handle) = server_for(TOOL_STREAM);
    let tools = Rc::new(RefCell::new(Vec::<(String, String)>::new()));
    let t = Rc::clone(&tools);
    let resp = client(format!("http://127.0.0.1:{port}"))
        .chat_stream_rich(
            &[ChatMessage::user("hi")],
            None,
            StreamHandler::new()
                .on_tool_call(move |n, a| t.borrow_mut().push((n.to_string(), a.to_string()))),
        )
        .unwrap();
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].parsed_args()["cmd"], "ls");
    assert_eq!(tools.borrow()[0].0, "run_shell");
    handle.join().unwrap();
}

#[test]
fn send_stream_rich_over_http() {
    let (port, handle) = server_for(REASONING_STREAM);
    let req = ChatCompletionRequest::new("gpt-4o", vec![ChatMessage::user("hi")]).stream(true);
    let thinking = Rc::new(RefCell::new(Vec::<String>::new()));
    let t = Rc::clone(&thinking);
    let resp = client(format!("http://127.0.0.1:{port}"))
        .send_stream_rich(
            &req,
            StreamHandler::new().on_thinking(move |s| t.borrow_mut().push(s.to_string())),
        )
        .unwrap();
    assert_eq!(resp.text, "The answer is 42.");
    assert_eq!(thinking.borrow().join(""), "Let me think.");
    handle.join().unwrap();
}

#[test]
fn builder_accepts_move_closures_in_any_order() {
    let (delta, thinking, tools) = sink_vec();
    let d = Rc::clone(&delta);
    let t = Rc::clone(&thinking);
    let tc = Rc::clone(&tools);
    let handler = StreamHandler::new()
        .on_tool_call(move |n, a| tc.borrow_mut().push((n.to_string(), a.to_string())))
        .on_thinking(move |s| t.borrow_mut().push(s.to_string()))
        .on_delta(move |s| d.borrow_mut().push(s.to_string()));
    let cancel = AtomicBool::new(false);
    let resp = parse_openai_stream_rich(
        Cursor::new(TOOL_STREAM.as_bytes().to_vec()),
        handler,
        &cancel,
    )
    .unwrap();
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(delta.borrow().len(), 0);
    assert_eq!(thinking.borrow().len(), 0);
    assert_eq!(tools.borrow().len(), 1);
}

// ── Async live HTTP round-trips ─────────────────────────────────────────────

#[cfg(feature = "async")]
#[tokio::test]
async fn async_chat_stream_rich_separates_over_http() {
    use openai_client_rs::OpenAiAsyncClient;

    let (port, handle) = server_for(REASONING_STREAM);
    let deltas = Rc::new(RefCell::new(Vec::<String>::new()));
    let thinking = Rc::new(RefCell::new(Vec::<String>::new()));
    let d = Rc::clone(&deltas);
    let t = Rc::clone(&thinking);
    let client =
        OpenAiAsyncClient::with_base_url("sk-test", "gpt-4o", format!("http://127.0.0.1:{port}"))
            .with_retry_config(RetryConfig {
                max_retries: 0,
                base_delay_ms: 1,
                max_delay_ms: 10,
            });
    let resp = client
        .chat_stream_rich(
            &[ChatMessage::user("hi")],
            None,
            StreamHandler::new()
                .on_delta(move |s| d.borrow_mut().push(s.to_string()))
                .on_thinking(move |s| t.borrow_mut().push(s.to_string())),
        )
        .await
        .unwrap();
    assert_eq!(resp.text, "The answer is 42.");
    assert_eq!(resp.reasoning_content.as_deref(), Some("Let me think."));
    assert_eq!(deltas.borrow().join(""), "The answer is 42.");
    assert_eq!(thinking.borrow().join(""), "Let me think.");
    handle.join().unwrap();
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_chat_stream_classic_delta_receives_thinking_over_http() {
    use openai_client_rs::OpenAiAsyncClient;

    let (port, handle) = server_for(REASONING_STREAM);
    let mut deltas = Vec::new();
    let client =
        OpenAiAsyncClient::with_base_url("sk-test", "gpt-4o", format!("http://127.0.0.1:{port}"))
            .with_retry_config(RetryConfig {
                max_retries: 0,
                base_delay_ms: 1,
                max_delay_ms: 10,
            });
    let resp = client
        .chat_stream(
            &[ChatMessage::user("hi")],
            None,
            |d| deltas.push(d.to_string()),
            |_, _| {},
        )
        .await
        .unwrap();
    assert_eq!(resp.text, "The answer is 42.");
    assert_eq!(resp.reasoning_content.as_deref(), Some("Let me think."));
    assert_eq!(deltas.join(""), "Let me think.The answer is 42.");
    handle.join().unwrap();
}
