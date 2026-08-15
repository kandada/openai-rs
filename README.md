<!-- Copyright (c) 2025 xiefujin <490021684@qq.com> -->
<!-- Licensed under Apache-2.0, see LICENSE file for full license terms. -->


# openai-rs

[![Crates.io](https://img.shields.io/crates/v/openai-rs.svg)](https://crates.io/crates/openai-rs)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A full-featured Rust client for the OpenAI API. Compatible with OpenAI, DeepSeek, Kimi, MiniMax, Ollama, and **any OpenAI-compatible provider**.

[中文文档](README_zh.md)

## Features

- **Sync + Async** — lightweight `ureq` (default) or `reqwest` + `tokio` (feature `async`)
- **Chat Completions** — full API, streaming SSE, tool calling, structured outputs, reasoning
- **Images** — DALL-E generation, editing, variations
- **Audio** — TTS speech generation, STT transcription & translation
- **Files** — upload, list, retrieve, delete, download
- **Models** — list, retrieve, delete
- **Embeddings** — single and batch text embeddings
- **Auto-Retry** — exponential backoff with jitter (sync + async)
- **Token Counting** — approximate message token estimation
- **Vision** — image URL and base64 inputs
- **Multi-Provider** — auto-detects provider by API key, custom base URL

## Installation

```toml
[dependencies]
openai-rs = "0.1"

# With async support
# openai-rs = { version = "0.1", features = ["async"] }
```

## Quick Start

### Sync (default)

```rust
use openai_rs::{OpenAiClient, ChatMessage};

let client = OpenAiClient::new("sk-xxx", "gpt-4o");

let resp = client.chat_create(
    &[ChatMessage::user("What is Rust?")],
    None,
).unwrap();
println!("{}", resp.text);
```

### Async

```rust
use openai_rs::{OpenAiAsyncClient, ChatMessage};

let client = OpenAiAsyncClient::new("sk-xxx", "gpt-4o");
let resp = client.chat_create(&[ChatMessage::user("Hi")], None).await.unwrap();
```

## Chat Completions

### Streaming

```rust
client.chat_stream(
    &[ChatMessage::user("Tell a story")],
    None,
    |delta| { print!("{delta}"); },
    |tool_name, tool_args| { println!("Tool: {tool_name}"); },
).unwrap();
```

### Tool Calling

```rust
use openai_rs::{OpenAiClient, ChatMessage, Tool};
use serde_json::json;

let tools = &[Tool::function(
    "get_weather", "Get weather for a location",
    json!({"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}),
)];

let resp = client.chat_create(
    &[ChatMessage::user("Weather in Paris?")],
    Some(tools),
).unwrap();

for tc in &resp.tool_calls {
    println!("{} -> {}", tc.name, tc.parsed_args()["location"]);
}
```

### Request Builder (Full API)

```rust
use openai_rs::{ChatCompletionRequest, ResponseFormat, StreamOptions};

let req = ChatCompletionRequest::new("gpt-4o", vec![ChatMessage::user("Hi")])
    .temperature(0.7)
    .top_p(0.9)
    .max_tokens(2048)
    .seed(42)
    .stop(vec!["END".into()])
    .response_format(ResponseFormat::json_object())
    .stream(true)
    .stream_options(StreamOptions { include_usage: Some(true) })
    .reasoning_effort("medium");

// Send via client
let resp = client.send(&req).unwrap();
// Or stream: client.send_stream(&req, |d| ..., |n, a| ...).unwrap();
```

## Images (DALL-E)

```rust
use openai_rs::{ImageRequest, ImageResponse};

let req = ImageRequest::new("a cute cat")
    .model("dall-e-3")
    .n(1)
    .size("1024x1024")
    .quality("hd");

let resp: ImageResponse = client.images_generate(&req).unwrap();
println!("URL: {}", resp.data[0].url.as_deref().unwrap_or("no url"));
```

## Audio

### TTS (Text-to-Speech)

```rust
use openai_rs::SpeechRequest;

let req = SpeechRequest::new("tts-1", "Hello world", "alloy").speed(1.0);
let audio_bytes = client.audio_speech(&req).unwrap();
std::fs::write("output.mp3", audio_bytes).unwrap();
```

### STT (Speech-to-Text)

```rust
use openai_rs::TranscriptionRequest;

let req = TranscriptionRequest::new(base64_audio, "whisper-1")
    .language("en")
    .temperature(0.2);
let resp = client.audio_transcribe(&req).unwrap();
println!("Transcribed: {}", resp.text);
```

## Files

```rust
// List
let files = client.files_list().unwrap();

// Upload
let file = client.files_upload("data.jsonl", base64_content, "fine-tune").unwrap();

// Download content
let bytes = client.files_content(&file.id).unwrap();
```

## Embeddings

```rust
// Single
let vec = client.embedding_create("Hello", "text-embedding-3-small").unwrap();

// Batch
let resp = client.embeddings_create(&["a", "b"], "text-embedding-3-small").unwrap();
```

## Models

```rust
let list = client.models_list().unwrap();
let info = client.models_retrieve("gpt-4o").unwrap();
```

## Auto-Retry

```rust
use openai_rs::{RetryConfig, retry};

let config = RetryConfig { max_retries: 3, ..Default::default() };
let result = retry_sync(
    || client.chat_create(&[...], None),
    &config,
    |e| e.is_retryable(),
);
```

## Token Counting

```rust
use openai_rs::tokens;

let n = tokens::count_tokens("Hello world");
let total = tokens::count_messages_tokens(&messages);
```

## Vision / Multimodal

```rust
// URL
let msg = ChatMessage::user_with_images("What's this?", &["https://example.com/pic.jpg"]);

// Base64
let msg = ChatMessage::user_with_images("What color?", &["data:image/png;base64,iVBORw0..."]);

// Mixed content
let msg = ChatMessage::user_with_parts(vec![
    ContentPart::Text { text: "Look at this".into() },
    ContentPart::ImageUrl { image_url: ImageUrl { url: "...".into(), detail: Some("high".into()) }},
]);
```

## Multi-Provider

```rust
// DeepSeek
let client = OpenAiClient::with_base_url("sk-xxx", "deepseek-chat", "https://api.deepseek.com/v1");

// Kimi / Moonshot
let client = OpenAiClient::with_base_url("sk-xxx", "kimi-k2", "https://api.moonshot.cn/v1");

// Ollama (local)
let client = OpenAiClient::with_base_url("ollama", "llama3", "http://localhost:11434/v1");
```

## Configuration

```rust
let client = OpenAiClient::new("sk-xxx", "gpt-4o")
    .with_organization("org-xxx")
    .with_temperature(0.7)
    .with_max_tokens(4096)
    .with_read_timeout(60)
    .with_total_timeout(120);
```

## Error Handling

```rust
use openai_rs::OpenAiError;

match client.chat_create(&[...], None) {
    Ok(resp) => println!("{}", resp.text),
    Err(OpenAiError::Api(msg)) => eprintln!("API: {msg}"),
    Err(OpenAiError::Network(msg)) => eprintln!("Network: {msg}"),
    Err(e) => eprintln!("Error: {e}"),
}

if err.is_retryable() { /* retry */ }
```

## API Coverage

| API | Methods |
|-----|---------|
| Chat Completions | `chat_create` / `chat_stream` / `send` / `send_stream` |
| Images | `images_generate` / `images_edit` / `images_variation` |
| Audio | `audio_speech` / `audio_transcribe` / `audio_translate` |
| Files | `files_list` / `files_upload` / `files_retrieve` / `files_delete` / `files_content` |
| Models | `models_list` / `models_retrieve` / `models_delete` |
| Embeddings | `embeddings_create` / `embedding_create` |
| Validate | `validate` |

## License

Apache-2.0