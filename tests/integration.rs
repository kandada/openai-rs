// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Integration tests for openai-client-rs.
//!
//! Covers: types serde, request building, SSE parsing, tool calling,
//! error handling, async SSE, and client configuration.

use serde_json::json;
use openai_client_rs::*;

// ── Types: serialization roundtrips ────────────────────────────────────────

#[test]
fn test_chat_message_serde_roundtrip() {
    let msg = ChatMessage::user("Hello world");
    let json_str = serde_json::to_string(&msg).unwrap();
    let decoded: ChatMessage = serde_json::from_str(&json_str).unwrap();
    assert_eq!(decoded.role, "user");
    assert_eq!(decoded.content, "Hello world");
}

#[test]
fn test_tool_call_serde_roundtrip() {
    let msg = ChatMessage::assistant_with_tools(
        "I'll call a tool",
        vec![ToolCall {
            id: "call_1".into(),
            call_type: "function".into(),
            function: FunctionCall { name: "run".into(), arguments: "{\"a\":1}".into() },
        }],
    );
    let json_str = serde_json::to_string(&msg).unwrap();
    let decoded: ChatMessage = serde_json::from_str(&json_str).unwrap();
    assert_eq!(decoded.role, "assistant");
    let tcs = decoded.tool_calls.unwrap();
    assert_eq!(tcs[0].function.name, "run");
    assert_eq!(tcs[0].function.parsed_args()["a"], 1);
}

#[test]
fn test_chat_completion_deserialize() {
    let raw = json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help?"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    });
    let cc: ChatCompletion = serde_json::from_value(raw).unwrap();
    assert_eq!(cc.id, "chatcmpl-123");
    assert_eq!(cc.choices[0].message.content.as_deref(), Some("Hello! How can I help?"));
    assert_eq!(cc.usage.as_ref().unwrap().total_tokens, 15);
}

#[test]
fn test_chat_completion_with_tool_calls_deserialize() {
    let raw = json!({
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1677652289,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_xyz",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\":\"Paris\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let cc: ChatCompletion = serde_json::from_value(raw).unwrap();
    let tcs = cc.choices[0].message.tool_calls.as_ref().unwrap();
    assert_eq!(tcs[0].function.name, "get_weather");
    assert_eq!(tcs[0].function.parsed_args()["location"], "Paris");
}

#[test]
fn test_chat_completion_chunk_deserialize() {
    let raw = json!({
        "id": "chatcmpl-789",
        "object": "chat.completion.chunk",
        "created": 1677652290,
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "delta": {"content": "Hello"},
            "finish_reason": null
        }]
    });
    let chunk: ChatCompletionChunk = serde_json::from_value(raw).unwrap();
    assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("Hello"));
}

#[test]
fn test_tool_serde() {
    let tool = Tool::function("my_fn", "does things", json!({"type":"object","properties":{}}));
    let v = serde_json::to_value(&tool).unwrap();
    assert_eq!(v["type"], "function");
    assert_eq!(v["function"]["name"], "my_fn");
}

#[test]
fn test_tool_strict_serde() {
    let tool = Tool::function_strict("my_fn", "desc", json!({"type":"object","properties":{}}));
    let v = serde_json::to_value(&tool).unwrap();
    assert_eq!(v["function"]["strict"], true);
}

#[test]
fn test_simplified_tool_call_parsed_args() {
    let tc = SimplifiedToolCall {
        id: "c1".into(),
        name: "run".into(),
        arguments: "{\"x\":1,\"y\":\"hello\"}".into(),
    };
    assert_eq!(tc.parsed_args()["x"], 1);
    assert_eq!(tc.parsed_args()["y"], "hello");
}

// ── Content parts / vision ──────────────────────────────────────────────────

#[test]
fn test_user_with_images_serde() {
    let msg = ChatMessage::user_with_images("Look at this", &["https://example.com/pic.png"]);
    let built = openai_client_rs::api_common::build_messages_json(&[msg]);
    let v = &built[0];
    let parts = v["content"].as_array().unwrap();
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[1]["type"], "image_url");
    assert_eq!(parts[1]["image_url"]["url"], "https://example.com/pic.png");
}

#[test]
fn test_user_with_parts_serde() {
    let msg = ChatMessage::user_with_parts(vec![
        ContentPart::Text { text: "hi".into() },
        ContentPart::ImageUrl {
            image_url: openai_client_rs::ImageUrl { url: "https://x.com/a.jpg".into(), detail: Some("high".into()) },
        },
    ]);
    let built = openai_client_rs::api_common::build_messages_json(&[msg]);
    let v = &built[0];
    let parts = v["content"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[1]["image_url"]["detail"], "high");
}

// ── Request builder ─────────────────────────────────────────────────────────

#[test]
fn test_request_builder_minimal() {
    let req = ChatCompletionRequest::new("gpt-4o", vec![ChatMessage::user("hi")]);
    let body = req.build_body();
    assert_eq!(body["model"], "gpt-4o");
    assert_eq!(body["stream"], false);
    assert_eq!(body["temperature"], 1.0);
    assert_eq!(body["max_tokens"], 4096);
}

#[test]
fn test_request_builder_full() {
    let tools = vec![Tool::function("f", "d", json!({"type":"object","properties":{"x":{"type":"string"}}}))];
    let req = ChatCompletionRequest::new("gpt-4o", vec![ChatMessage::user("hi")])
        .temperature(0.3)
        .top_p(0.95)
        .max_tokens(100)
        .seed(42)
        .stop(vec!["END".into(), "STOP".into()])
        .presence_penalty(0.5)
        .frequency_penalty(0.3)
        .tools(tools)
        .parallel_tool_calls(false)
        .reasoning_effort("low")
        .user("user-1")
        .stream(true)
        .stream_options(StreamOptions { include_usage: Some(true) });

    let body = req.build_body();
    assert_eq!(body["temperature"], 0.3);
    assert_eq!(body["top_p"], 0.95);
    assert_eq!(body["seed"], 42);
    assert_eq!(body["stop"].as_array().unwrap().len(), 2);
    assert_eq!(body["presence_penalty"], 0.5);
    assert_eq!(body["frequency_penalty"], 0.3);
    assert_eq!(body["parallel_tool_calls"], false);
    assert_eq!(body["reasoning_effort"], "low");
    assert_eq!(body["user"], "user-1");
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert!(body["tools"].is_array());
}

#[test]
fn test_request_builder_response_format_json_schema() {
    let schema = json!({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]});
    let req = ChatCompletionRequest::new("gpt-4o", vec![ChatMessage::user("hi")])
        .response_format(ResponseFormat::json_schema("MySchema", schema, true));
    let body = req.build_body();
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert_eq!(body["response_format"]["json_schema"]["name"], "MySchema");
    assert_eq!(body["response_format"]["json_schema"]["strict"], true);
}

#[test]
fn test_request_builder_max_completion_tokens_priority() {
    let req = ChatCompletionRequest::new("o3", vec![ChatMessage::user("hi")])
        .max_tokens(500)
        .max_completion_tokens(2000);
    let body = req.build_body();
    // max_completion_tokens wins and max_tokens is omitted (OpenAI rule).
    assert_eq!(body["max_completion_tokens"], 2000);
    assert!(body.get("max_tokens").is_none());
}

#[test]
fn test_request_builder_tool_choice_auto() {
    let tools = vec![Tool::function("f", "d", json!({"type":"object","properties":{}}))];
    let req = ChatCompletionRequest::new("gpt-4o", vec![ChatMessage::user("hi")])
        .tools(tools);
    let body = req.build_body();
    assert_eq!(body["tool_choice"], "auto");
}

// ── SSE parsing ─────────────────────────────────────────────────────────────

#[test]
fn test_sse_parse_content_stream() {
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;

    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let cancel = AtomicBool::new(false);
    let mut deltas = Vec::new();
    let resp = openai_client_rs::parse_openai_stream(
        Cursor::new(raw.as_bytes().to_vec()),
        |d| deltas.push(d.to_string()),
        |_, _| {},
        &cancel,
    ).unwrap();
    assert_eq!(deltas.join(""), "AB");
    assert_eq!(resp.text, "AB");
    assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
}

#[test]
fn test_sse_parse_reasoning_content() {
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;

    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"step1\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"step2\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
        "data: [DONE]\n\n"
    );
    let cancel = AtomicBool::new(false);
    let resp = openai_client_rs::parse_openai_stream(
        Cursor::new(raw.as_bytes().to_vec()),
        |_| {},
        |_, _| {},
        &cancel,
    ).unwrap();
    assert_eq!(resp.text, "answer");
    assert_eq!(resp.reasoning_content.as_deref(), Some("step1step2"));
}

#[test]
fn test_sse_parse_tool_calls() {
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;

    let raw = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"t1\",\"function\":{\"name\":\"exec\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"cmd\\\":\\\"ls\\\"}\"}}]}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let cancel = AtomicBool::new(false);
    let mut tools = Vec::new();
    let resp = openai_client_rs::parse_openai_stream(
        Cursor::new(raw.as_bytes().to_vec()),
        |_| {},
        |n, a| tools.push((n.to_string(), a.to_string())),
        &cancel,
    ).unwrap();
    assert_eq!(resp.tool_calls.len(), 1);
    assert_eq!(resp.tool_calls[0].name, "exec");
    assert_eq!(resp.tool_calls[0].id, "t1");
    assert_eq!(resp.tool_calls[0].parsed_args()["cmd"], "ls");
}

#[test]
fn test_sse_parse_cancel() {
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;

    let raw = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n";
    let cancel = AtomicBool::new(true);
    let r = openai_client_rs::parse_openai_stream(
        Cursor::new(raw.as_bytes().to_vec()),
        |_| {}, |_, _| {}, &cancel,
    );
    assert!(matches!(r, Err(OpenAiError::Cancelled)));
}

#[test]
fn test_sse_parse_all_malformed() {
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;

    let raw = concat!("data: {not json\n\n", "data: [DONE]\n\n");
    let cancel = AtomicBool::new(false);
    let r = openai_client_rs::parse_openai_stream(
        Cursor::new(raw.as_bytes().to_vec()),
        |_| {}, |_, _| {}, &cancel,
    );
    assert!(r.is_err());
}

#[test]
fn test_sse_parse_in_stream_error() {
    use std::io::Cursor;
    use std::sync::atomic::AtomicBool;

    let raw = "data: {\"error\":{\"message\":\"rate limit exceeded\"}}\n\n";
    let cancel = AtomicBool::new(false);
    let r = openai_client_rs::parse_openai_stream(
        Cursor::new(raw.as_bytes().to_vec()),
        |_| {}, |_, _| {}, &cancel,
    );
    assert!(r.is_err());
    if let Err(OpenAiError::Api(e)) = r {
        assert!(e.message.contains("rate limit"));
    } else {
        panic!("expected Api error");
    }
}

// ── Error handling ──────────────────────────────────────────────────────────

#[test]
fn test_error_retryable() {
    assert!(OpenAiError::Network("timeout".into()).is_retryable());
    assert!(OpenAiError::api(503, None, "Service Unavailable").is_retryable());
    assert!(OpenAiError::stream_error("rate limit exceeded").is_retryable());
    assert!(!OpenAiError::api(401, None, "Unauthorized").is_retryable());
    assert!(!OpenAiError::Config("no API key".into()).is_retryable());
    assert!(!OpenAiError::Cancelled.is_retryable());
}

#[test]
fn test_error_display() {
    let e = OpenAiError::Config("missing key".into());
    assert!(e.to_string().contains("config"));
    assert!(e.to_string().contains("missing key"));
    assert_eq!(OpenAiError::Cancelled.to_string(), "cancelled");
}

#[test]
fn test_llm_response_truncated() {
    let resp = LlmResponse { finish_reason: Some("length".into()), ..Default::default() };
    assert!(resp.is_truncated());
    let resp = LlmResponse { finish_reason: Some("max_tokens".into()), ..Default::default() };
    assert!(resp.is_truncated());
    let resp = LlmResponse { finish_reason: Some("stop".into()), ..Default::default() };
    assert!(!resp.is_truncated());
}

// ── Client configuration ────────────────────────────────────────────────────

#[test]
fn test_client_new() {
    let c = OpenAiClient::new("sk-test123", "gpt-4o");
    assert_eq!(c.model(), "gpt-4o");
}

#[test]
fn test_client_with_base_url() {
    let c = OpenAiClient::with_base_url("sk-test", "deepseek-chat", "https://api.deepseek.com/v1");
    assert_eq!(c.endpoint("chat/completions"), "https://api.deepseek.com/v1/chat/completions");
}

#[test]
fn test_client_configuration() {
    let c = OpenAiClient::new("sk-test123", "gpt-4o")
        .with_temperature(0.5)
        .with_max_tokens(2048)
        .with_organization("org-1");
    assert_eq!(c.model(), "gpt-4o");
}

// ── Embeddings response ─────────────────────────────────────────────────────

#[test]
fn test_embedding_response_deserialize() {
    let raw = json!({
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
            {"object": "embedding", "index": 1, "embedding": [0.4, 0.5, 0.6]}
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 5, "total_tokens": 5}
    });
    let resp: EmbeddingResponse = serde_json::from_value(raw).unwrap();
    assert_eq!(resp.data.len(), 2);
    assert_eq!(resp.data[0].embedding.len(), 3);
    assert_eq!(resp.data[1].index, 1);
    assert_eq!(resp.usage.total_tokens, 5);
}

// ── Tool choice types ───────────────────────────────────────────────────────

#[test]
fn test_tool_choice_serialization() {
    assert_eq!(serde_json::to_string(&ToolChoice::auto()).unwrap(), "\"auto\"");
    assert_eq!(serde_json::to_string(&ToolChoice::none()).unwrap(), "\"none\"");
    assert_eq!(serde_json::to_string(&ToolChoice::required()).unwrap(), "\"required\"");
    let specific = ToolChoice::specific("my_func");
    let v = serde_json::to_value(&specific).unwrap();
    assert_eq!(v["type"], "function");
    assert_eq!(v["function"]["name"], "my_func");
}