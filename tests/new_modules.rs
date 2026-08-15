// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Integration tests for new openai-rs modules: retry, tokens, images, audio, files.

use serde_json::json;
use openai_rs::*;

// ── Retry ──────────────────────────────────────────────────────────────────

#[test]
fn test_retry_succeeds_on_eventual_success() {
    use openai_rs::retry::*;
    let config = RetryConfig { max_retries: 5, base_delay_ms: 1, max_delay_ms: 10 };
    let mut calls = 0;
    let result: std::result::Result<i32, &str> = retry_sync(
        || { calls += 1; if calls < 4 { Err("transient") } else { Ok(42) } },
        &config, |_| true,
    );
    assert_eq!(result.unwrap(), 42);
    assert_eq!(calls, 4);
}

#[test]
fn test_retry_stops_at_max() {
    use openai_rs::retry::*;
    let config = RetryConfig { max_retries: 2, base_delay_ms: 1, max_delay_ms: 10 };
    let mut calls = 0;
    let result: std::result::Result<i32, &str> = retry_sync(
        || { calls += 1; Err("always fail") },
        &config, |_| true,
    );
    assert!(result.is_err());
    assert_eq!(calls, 3); // 1 initial + 2 retries
}

#[test]
fn test_retry_non_retryable_immediate() {
    use openai_rs::retry::*;
    let config = RetryConfig::default();
    let mut calls = 0;
    let result: std::result::Result<i32, &str> = retry_sync(
        || { calls += 1; Err("fatal") },
        &config, |_| false,
    );
    assert!(result.is_err());
    assert_eq!(calls, 1);
}

#[test]
fn test_retry_delay_grows() {
    use openai_rs::retry::*;
    let config = RetryConfig { max_retries: 10, base_delay_ms: 100, max_delay_ms: 10000 };
    let d0 = config.delay_ms(0);
    let d1 = config.delay_ms(1);
    let d4 = config.delay_ms(4);
    assert!(d1 > d0);
    assert!(d4 > d1);
    assert!(d4 <= 10100); // max_delay + 25% jitter
}

#[test]
fn test_retry_with_openai_error() {
    use openai_rs::retry::*;
    let config = RetryConfig { max_retries: 2, base_delay_ms: 1, max_delay_ms: 10 };
    let mut calls = 0;
    let result: std::result::Result<i32, OpenAiError> = retry_sync(
        || {
            calls += 1;
            if calls < 2 { Err(OpenAiError::Network("timeout".into())) }
            else { Ok(42) }
        },
        &config, |e| e.is_retryable(),
    );
    assert_eq!(result.unwrap(), 42);
}

// ── Token Counting ─────────────────────────────────────────────────────────

#[test]
fn test_count_tokens_english() {
    let n = openai_rs::tokens::count_tokens("Hello world!");
    assert!(n >= 2 && n <= 6, "got {n}");
}

#[test]
fn test_count_tokens_chinese() {
    let n = openai_rs::tokens::count_tokens("你好世界！");
    assert!(n >= 3 && n <= 6, "got {n}");
}

#[test]
fn test_count_tokens_mixed() {
    let n = openai_rs::tokens::count_tokens("Hello 你好 world 世界");
    assert!(n >= 4, "got {n}");
}

#[test]
fn test_count_message_tokens_simple() {
    let msg = ChatMessage::user("Hello!");
    let n = openai_rs::tokens::count_message_tokens(&msg);
    assert!(n >= 5, "got {n}");
}

#[test]
fn test_count_message_tokens_with_tool_calls() {
    let msg = ChatMessage::assistant_with_tools("", vec![
        ToolCall { id: "1".into(), call_type: "function".into(), function: FunctionCall { name: "run".into(), arguments: "{\"a\":1}".into() }}
    ]);
    let n = openai_rs::tokens::count_message_tokens(&msg);
    assert!(n >= 15, "got {n}");
}

#[test]
fn test_count_messages_tokens() {
    let msgs = vec![
        ChatMessage::system("You are helpful."),
        ChatMessage::user("Hello"),
        ChatMessage::assistant("Hi there!"),
    ];
    let n = openai_rs::tokens::count_messages_tokens(&msgs);
    assert!(n >= 15, "got {n}");
}

#[test]
fn test_estimate_completion_tokens() {
    let n = openai_rs::tokens::estimate_completion_tokens(100);
    assert!(n >= 20, "got {n}");
}

// ── Images ─────────────────────────────────────────────────────────────────

#[test]
fn test_image_request_serde_full() {
    let req = ImageRequest::new("a beautiful sunset")
        .model("dall-e-3")
        .n(2)
        .size("1024x1024")
        .quality("hd")
        .style("vivid")
        .response_format("b64_json")
        .user("test-user");
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("a beautiful sunset"));
    assert!(json.contains("dall-e-3"));
    assert!(json.contains("1024x1024"));
    assert!(json.contains("hd"));
    assert!(json.contains("vivid"));
    assert!(json.contains("b64_json"));
}

#[test]
fn test_image_response_deserialize_url() {
    let json = r#"{"created":123,"data":[{"url":"http://x.com/a.png","revised_prompt":"a cat"}]}"#;
    let resp: ImageResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.data.len(), 1);
    assert_eq!(resp.data[0].url.as_deref(), Some("http://x.com/a.png"));
    assert_eq!(resp.data[0].revised_prompt.as_deref(), Some("a cat"));
}

#[test]
fn test_image_response_deserialize_b64() {
    let json = r#"{"created":456,"data":[{"b64_json":"abc123"}]}"#;
    let resp: ImageResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.data[0].b64_json.as_deref(), Some("abc123"));
}

// ── Audio ──────────────────────────────────────────────────────────────────

#[test]
fn test_speech_request_serde_full() {
    let req = SpeechRequest::new("tts-1-hd", "Hello world", "nova")
        .speed(1.2)
        .format("opus");
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("tts-1-hd"));
    assert!(json.contains("nova"));
    assert!(json.contains("1.2"));
    assert!(json.contains("opus"));
}

#[test]
fn test_transcription_request_serde_full() {
    let req = TranscriptionRequest::new("base64data", "whisper-1")
        .language("en")
        .prompt("technical terms")
        .temperature(0.3);
    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("base64data"));
    assert!(json.contains("whisper-1"));
    assert!(json.contains("en"));
    assert!(json.contains("technical terms"));
}

#[test]
fn test_transcription_response_deserialize_verbose() {
    let json = r#"{
        "text": "hello world",
        "language": "en",
        "duration": 2.5,
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.2, "text": "hello", "seek": 0, "tokens": [1,2], "temperature": 0.0, "avg_logprob": -0.5, "compression_ratio": 1.0, "no_speech_prob": 0.1},
            {"id": 1, "start": 1.2, "end": 2.5, "text": " world", "seek": 0, "tokens": [3,4], "temperature": 0.0, "avg_logprob": -0.3, "compression_ratio": 1.0, "no_speech_prob": 0.05}
        ]
    }"#;
    let resp: TranscriptionResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.text, "hello world");
    assert_eq!(resp.language.as_deref(), Some("en"));
    assert_eq!(resp.segments.as_ref().unwrap().len(), 2);
    assert_eq!(resp.segments.unwrap()[0].text, "hello");
}

// ── Files ──────────────────────────────────────────────────────────────────

#[test]
fn test_file_object_deserialize_full() {
    let json = r#"{
        "id": "file-abc123",
        "object": "file",
        "bytes": 1024,
        "created_at": 1678900000,
        "filename": "train.jsonl",
        "purpose": "fine-tune",
        "status": "processed",
        "status_details": null
    }"#;
    let f: FileObject = serde_json::from_str(json).unwrap();
    assert_eq!(f.id, "file-abc123");
    assert_eq!(f.filename, "train.jsonl");
    assert_eq!(f.bytes, 1024);
    assert_eq!(f.status.as_deref(), Some("processed"));
}

#[test]
fn test_file_list_deserialize() {
    let json = r#"{"object":"list","data":[{"id":"f1","object":"file","bytes":100,"created_at":1,"filename":"a.jsonl","purpose":"fine-tune"},{"id":"f2","object":"file","bytes":200,"created_at":2,"filename":"b.jsonl","purpose":"fine-tune"}]}"#;
    let list: FileList = serde_json::from_str(json).unwrap();
    assert_eq!(list.data.len(), 2);
}