// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! End-to-end tests against real APIs.
//!
//! Run with:
//!   DEEPSEEK=1 cargo test --test e2e -- --nocapture
//!   MOONSHOT=1 cargo test --test e2e -- --nocapture
//!   ALL=1 cargo test --test e2e -- --nocapture

use serde_json::json;
use std::env;

// ── Helpers ────────────────────────────────────────────────────────────────

fn deepseek_key() -> String {
    env::var("DEEPSEEK_KEY").unwrap_or_else(|_| "sk-3db317b59c3a4fbe86b6df7db0636f80".into())
}
fn deepseek_url() -> String {
    env::var("DEEPSEEK_URL").unwrap_or_else(|_| "https://api.deepseek.com/v1".into())
}
fn deepseek_model() -> String {
    env::var("DEEPSEEK_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".into())
}

fn moonshot_key() -> String {
    env::var("MOONSHOT_KEY").unwrap_or_else(|_| "sk-f2x8aZ4INh3NlCYLeMIYqgBa7e6uE6ICmt2Ihys2C4wbtr4F".into())
}
fn moonshot_url() -> String {
    env::var("MOONSHOT_URL").unwrap_or_else(|_| "https://api.moonshot.cn/v1".into())
}
fn moonshot_model() -> String {
    env::var("MOONSHOT_MODEL").unwrap_or_else(|_| "kimi-k2.6".into())
}

fn should_run(provider: &str) -> bool {
    env::var("ALL").is_ok() || env::var(provider).is_ok()
}

// ── DeepSeek tests ─────────────────────────────────────────────────────────

#[test]
fn deepseek_basic_chat() {
    if !should_run("DEEPSEEK") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        deepseek_key(), deepseek_model(), deepseek_url(),
    );
    let resp = client.chat_create(
        &[openai_rs::ChatMessage::user("Say 'hello' in exactly one word, lowercase.")],
        None,
    ).expect("chat_create failed");
    assert!(!resp.text.is_empty(), "response should not be empty");
    println!("[deepseek] chat: {}", resp.text.trim());
    assert!(resp.text.to_lowercase().contains("hello"));
}

#[test]
fn deepseek_streaming() {
    if !should_run("DEEPSEEK") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        deepseek_key(), deepseek_model(), deepseek_url(),
    );
    let mut deltas = Vec::new();
    client.chat_stream(
        &[openai_rs::ChatMessage::user("Count from 1 to 3, one per line.")],
        None,
        |d| { deltas.push(d.to_string()); },
        |_, _| {},
    ).expect("chat_stream failed");
    let text = deltas.join("");
    println!("[deepseek] stream: {}", text.trim());
    assert!(!text.is_empty());
    assert!(text.contains("1") && text.contains("2") && text.contains("3"));
}

#[test]
fn deepseek_tool_calling() {
    if !should_run("DEEPSEEK") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        deepseek_key(), deepseek_model(), deepseek_url(),
    );
    let tools = vec![openai_rs::Tool::function(
        "get_weather",
        "Get weather for a location",
        json!({
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }),
    )];
    let resp = client.chat_create(
        &[openai_rs::ChatMessage::user("What is the weather in London?")],
        Some(&tools),
    ).expect("tool call failed");
    println!("[deepseek] tools: {:?}", resp.tool_calls.iter().map(|t| &t.name).collect::<Vec<_>>());
    assert!(!resp.tool_calls.is_empty() || !resp.text.is_empty(),
        "should have tool calls or text response");
    if !resp.tool_calls.is_empty() {
        let tc = &resp.tool_calls[0];
        assert_eq!(tc.name, "get_weather");
        assert!(tc.parsed_args()["location"].as_str().unwrap_or("").to_lowercase().contains("london"));
        println!("[deepseek] tool call: {} -> {}", tc.name, tc.arguments);
    }
}

#[test]
fn deepseek_validate_api_key() {
    if !should_run("DEEPSEEK") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        deepseek_key(), deepseek_model(), deepseek_url(),
    );
    client.validate().expect("validate should succeed");
    println!("[deepseek] validate: OK");
}

#[test]
fn deepseek_invalid_key() {
    let client = openai_rs::OpenAiClient::with_base_url(
        "sk-invalid-key-12345", deepseek_model(), deepseek_url(),
    );
    let err = client.validate().unwrap_err();
    println!("[deepseek] invalid key error: {err}");
    assert!(!err.is_retryable(), "auth error should not be retryable");
}

// ── Moonshot / Kimi tests ──────────────────────────────────────────────────

#[test]
fn moonshot_basic_chat() {
    if !should_run("MOONSHOT") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        moonshot_key(), moonshot_model(), moonshot_url(),
    );
    let resp = client.chat_create(
        &[openai_rs::ChatMessage::user("Say one word: hello")],
        None,
    ).expect("chat_create failed");
    println!("[moonshot] chat: '{}'", resp.text.trim());
    assert!(!resp.text.is_empty(), "response should not be empty");
}

#[test]
fn moonshot_streaming() {
    if !should_run("MOONSHOT") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        moonshot_key(), moonshot_model(), moonshot_url(),
    );
    let mut deltas = Vec::new();
    client.chat_stream(
        &[openai_rs::ChatMessage::user("List 3 fruits, one per line.")],
        None,
        |d| { deltas.push(d.to_string()); },
        |_, _| {},
    ).expect("chat_stream failed");
    let text = deltas.join("");
    println!("[moonshot] stream: {}", text.trim());
    assert!(!text.is_empty());
}

#[test]
fn moonshot_tool_calling() {
    if !should_run("MOONSHOT") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        moonshot_key(), moonshot_model(), moonshot_url(),
    );
    let tools = vec![openai_rs::Tool::function(
        "calculate",
        "Perform a calculation",
        json!({
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }),
    )];
    let resp = client.chat_create(
        &[openai_rs::ChatMessage::user("What is 15 * 7? Use the calculate tool.")],
        Some(&tools),
    ).expect("tool call failed");
    println!("[moonshot] tools: {:?}", resp.tool_calls.iter().map(|t| &t.name).collect::<Vec<_>>());
    assert!(!resp.tool_calls.is_empty() || !resp.text.is_empty());
    if !resp.tool_calls.is_empty() {
        let tc = &resp.tool_calls[0];
        assert_eq!(tc.name, "calculate");
        println!("[moonshot] tool call: {} -> {}", tc.name, tc.arguments);
    }
}

#[test]
fn moonshot_validate_api_key() {
    if !should_run("MOONSHOT") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        moonshot_key(), moonshot_model(), moonshot_url(),
    );
    client.validate().expect("validate should succeed");
    println!("[moonshot] validate: OK");
}

#[test]
fn moonshot_multimodal_image() {
    if !should_run("MOONSHOT") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        moonshot_key(), moonshot_model(), moonshot_url(),
    );
    // Use a small test image (1x1 pixel PNG as base64)
    let image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==";
    let msg = openai_rs::ChatMessage::user_with_parts(vec![
        openai_rs::ContentPart::Text { text: "What color is this image? Reply with just one word.".into() },
        openai_rs::ContentPart::ImageUrl {
            image_url: openai_rs::ImageUrl {
                url: format!("data:image/png;base64,{image_b64}"),
                detail: Some("low".into()),
            },
        },
    ]);
    let resp = client.chat_create(&[msg], None).expect("multimodal failed");
    println!("[moonshot] multimodal: {}", resp.text.trim());
    assert!(!resp.text.is_empty(), "should return a description");
}

// ── Multi-turn conversation ────────────────────────────────────────────────

#[test]
fn deepseek_multi_turn_tool_roundtrip() {
    if !should_run("DEEPSEEK") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        deepseek_key(), deepseek_model(), deepseek_url(),
    );
    let tools = vec![openai_rs::Tool::function(
        "get_weather",
        "Get weather for a location",
        json!({
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }),
    )];
    let resp1 = client.chat_create(
        &[openai_rs::ChatMessage::user("Weather in Paris? Use the tool.")],
        Some(&tools),
    ).expect("turn 1 failed");
    println!("[deepseek multi] turn1: {:?}", resp1.tool_calls.iter().map(|t| (&t.name, &t.arguments)).collect::<Vec<_>>());
    assert!(!resp1.tool_calls.is_empty(), "expected tool call, got: {}", resp1.text);
    let tc = &resp1.tool_calls[0];

    let messages = vec![
        openai_rs::ChatMessage::user("Weather in Paris? Use the tool."),
        openai_rs::ChatMessage::assistant_with_tools("", vec![openai_rs::ToolCall {
            id: tc.id.clone(), call_type: "function".into(),
            function: openai_rs::FunctionCall { name: tc.name.clone(), arguments: tc.arguments.clone() },
        }]),
        openai_rs::ChatMessage::tool_result(&tc.id, "Sunny, 22C"),
        openai_rs::ChatMessage::user("Summarize in one sentence."),
    ];
    let resp2 = client.chat_create(&messages, None).expect("turn 2 failed");
    println!("[deepseek multi] turn2: '{}'", resp2.text.trim());
    assert!(!resp2.text.is_empty());
    assert!(resp2.text.to_lowercase().contains("sunny") || resp2.text.contains("22"));
}

#[test]
fn moonshot_multi_turn_tool_roundtrip() {
    if !should_run("MOONSHOT") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        moonshot_key(), moonshot_model(), moonshot_url(),
    );
    let tools = vec![openai_rs::Tool::function(
        "calculate",
        "Perform calculation",
        json!({
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"]
        }),
    )];
    let resp1 = client.chat_create(
        &[openai_rs::ChatMessage::user("Calculate 123+456 using the tool.")],
        Some(&tools),
    ).expect("turn 1 failed");
    println!("[moonshot multi] turn1: {:?}", resp1.tool_calls.iter().map(|t| (&t.name, &t.arguments)).collect::<Vec<_>>());
    assert!(!resp1.tool_calls.is_empty(), "expected tool call, got: {}", resp1.text);
    let tc = &resp1.tool_calls[0];

    let messages = vec![
        openai_rs::ChatMessage::user("Calculate 123+456 using the tool."),
        openai_rs::ChatMessage::assistant_with_tools("", vec![openai_rs::ToolCall {
            id: tc.id.clone(), call_type: "function".into(),
            function: openai_rs::FunctionCall { name: tc.name.clone(), arguments: tc.arguments.clone() },
        }]),
        openai_rs::ChatMessage::tool_result(&tc.id, "579"),
    ];
    let resp2 = client.chat_create(&messages, None).expect("turn 2 failed");
    println!("[moonshot multi] turn2: '{}'", resp2.text.trim());
    assert!(!resp2.text.is_empty());
    assert!(resp2.text.contains("579"));
}

// ── Streaming vs non-streaming ─────────────────────────────────────────────

#[test]
fn deepseek_streaming_vs_non_streaming() {
    if !should_run("DEEPSEEK") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        deepseek_key(), deepseek_model(), deepseek_url(),
    );
    let msg = openai_rs::ChatMessage::user("Count 1,2,3,4,5 comma separated, no other text.");

    let non_stream = client.chat_create(&[msg.clone()], None).expect("non-stream failed");
    println!("[deepseek cmp] non-stream: '{}' (len={})", non_stream.text.trim(), non_stream.text.len());

    let mut deltas = Vec::new();
    client.chat_stream(&[msg], None, |d| { deltas.push(d.to_string()); }, |_, _| {}).expect("stream failed");
    let stream_text = deltas.join("");
    println!("[deepseek cmp] stream: '{}' (len={})", stream_text.trim(), stream_text.len());

    assert!(!non_stream.text.is_empty());
    assert!(!stream_text.is_empty());
    assert!(non_stream.text.contains("1") && non_stream.text.contains("5"));
    assert!(stream_text.contains("1") && stream_text.contains("5"));
}

#[test]
fn moonshot_streaming_vs_non_streaming() {
    if !should_run("MOONSHOT") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        moonshot_key(), moonshot_model(), moonshot_url(),
    );
    let msg = openai_rs::ChatMessage::user("Count 1,2,3,4,5 comma separated, no other text.");

    let non_stream = client.chat_create(&[msg.clone()], None).expect("non-stream failed");
    println!("[moonshot cmp] non-stream: '{}' (len={})", non_stream.text.trim(), non_stream.text.len());

    let mut deltas = Vec::new();
    client.chat_stream(&[msg], None, |d| { deltas.push(d.to_string()); }, |_, _| {}).expect("stream failed");
    let stream_text = deltas.join("");
    println!("[moonshot cmp] stream: '{}' (len={})", stream_text.trim(), stream_text.len());

    assert!(!non_stream.text.is_empty());
    assert!(!stream_text.is_empty());
}

// ── Real multimodal ────────────────────────────────────────────────────────

#[test]
fn moonshot_real_image_recognition() {
    if !should_run("MOONSHOT") { return; }
    let client = openai_rs::OpenAiClient::with_base_url(
        moonshot_key(), moonshot_model(), moonshot_url(),
    );
    // Valid 4x4 red pixel PNG
    let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP4z8AARwzEcQCukw/x0F8jngAAAABJRU5ErkJggg==";
    let msg = openai_rs::ChatMessage::user_with_images(
        "What color is the dot? Reply with just one word.",
        &[&format!("data:image/png;base64,{b64}")],
    );
    let resp = client.chat_create(&[msg], None).expect("real image failed");
    println!("[moonshot image] '{}' (len={})", resp.text.trim(), resp.text.len());
    assert!(!resp.text.is_empty(), "should describe the image");
}

// ── Async tests (DeepSeek) ─────────────────────────────────────────────────
#[cfg(feature = "async")]
#[tokio::test]
async fn deepseek_async_basic_chat() {
    // async tests can't check env var at runtime easily, so always run if compiled with async
    // but check anyway to be safe
    if !should_run("DEEPSEEK") { return; }
    let client = openai_rs::OpenAiAsyncClient::with_base_url(
        deepseek_key(), deepseek_model(), deepseek_url(),
    );
    let resp = client.chat_create(
        &[openai_rs::ChatMessage::user("Reply with just the word 'OK'.")],
        None,
    ).await.expect("async chat_create failed");
    println!("[deepseek async] chat: {}", resp.text.trim());
    assert!(!resp.text.is_empty());
}