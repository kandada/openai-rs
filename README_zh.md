<!-- Copyright (c) 2025 xiefujin <490021684@qq.com> -->
<!-- Licensed under Apache-2.0, see LICENSE file for full license terms. -->


# openai-client-rs

[![Crates.io](https://img.shields.io/crates/v/openai-client-rs.svg)](https://crates.io/crates/openai-client-rs)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

全功能 Rust OpenAI API 客户端。兼容 OpenAI、DeepSeek、Kimi、MiniMax、Ollama 等**任何 OpenAI 兼容的 API 提供商**。

[English](README.md)

## 特性

- **同步 + 异步** — 轻量 `ureq`（默认）或 `reqwest` + `tokio`（feature `async`）
- **Chat Completions** — 完整 API，流式 SSE，函数调用，结构化输出，推理链
- **Images 图像** — DALL-E 文生图、编辑、变体
- **Audio 音频** — TTS 文字转语音、STT 语音识别与翻译
- **Files 文件** — 上传、列表、获取、删除、下载
- **Models 模型** — 列表、获取、删除
- **Embeddings 向量** — 单条与批量文本向量化
- **自动重试** — 指数退避 + 抖动（同步 + 异步）
- **Token 计数** — 近似消息 token 估算
- **视觉识别** — 图片 URL 与 base64 输入
- **多提供商** — 根据 API key 自动识别，支持自定义 base URL

## 安装

```toml
[dependencies]
openai-client-rs = "0.1"

# 异步支持
# openai-client-rs = { version = "0.1", features = ["async"] }
```

## 快速开始

### 同步

```rust
use openai_client_rs::{OpenAiClient, ChatMessage};

let client = OpenAiClient::new("sk-xxx", "gpt-4o");
let resp = client.chat_create(&[ChatMessage::user("Rust 的特点？")], None).unwrap();
println!("{}", resp.text);
```

### 异步

```rust
use openai_client_rs::{OpenAiAsyncClient, ChatMessage};

let client = OpenAiAsyncClient::new("sk-xxx", "gpt-4o");
let resp = client.chat_create(&[ChatMessage::user("你好")], None).await.unwrap();
```

## 流式输出

```rust
client.chat_stream(
    &[ChatMessage::user("讲个故事")],
    None,
    |delta| { print!("{delta}"); },
    |tool_name, tool_args| { println!("工具: {tool_name}"); },
).unwrap();
```

## 函数调用

```rust
use openai_client_rs::{Tool};
use serde_json::json;

let tools = &[Tool::function("get_weather", "查询天气",
    json!({"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}),
)];

let resp = client.chat_create(&[ChatMessage::user("北京天气？")], Some(tools)).unwrap();
for tc in &resp.tool_calls {
    println!("{} -> {}", tc.name, tc.parsed_args()["location"]);
}
```

## 请求构建器（完整 API 参数）

```rust
use openai_client_rs::{ChatCompletionRequest, ResponseFormat, StreamOptions};

let req = ChatCompletionRequest::new("gpt-4o", vec![ChatMessage::user("你好")])
    .temperature(0.7).top_p(0.9).max_tokens(2048).seed(42)
    .stop(vec!["END".into()])
    .response_format(ResponseFormat::json_object())
    .stream(true).stream_options(StreamOptions { include_usage: Some(true) })
    .reasoning_effort("medium");

let resp = client.send(&req).unwrap();
```

## 图像生成 (DALL-E)

```rust
use openai_client_rs::ImageRequest;

let req = ImageRequest::new("一只可爱的猫")
    .model("dall-e-3").n(1).size("1024x1024").quality("hd");
let resp = client.images_generate(&req).unwrap();
println!("URL: {}", resp.data[0].url.as_deref().unwrap_or("无"));
```

## 音频

### TTS（文字转语音）

```rust
use openai_client_rs::SpeechRequest;
let req = SpeechRequest::new("tts-1", "你好世界", "alloy").speed(1.0);
let bytes = client.audio_speech(&req).unwrap();
std::fs::write("output.mp3", bytes).unwrap();
```

### STT（语音识别）

```rust
use openai_client_rs::TranscriptionRequest;
let req = TranscriptionRequest::new(base64_audio, "whisper-1").language("zh");
let resp = client.audio_transcribe(&req).unwrap();
println!("识别结果: {}", resp.text);
```

## 文件管理

```rust
let files = client.files_list().unwrap();
let file = client.files_upload("data.jsonl", base64_content, "fine-tune").unwrap();
let content = client.files_content(&file.id).unwrap();
```

## 自动重试

```rust
use openai_client_rs::{RetryConfig, retry};
let config = RetryConfig { max_retries: 3, ..Default::default() };
let result = retry_sync(|| client.chat_create(&[...], None), &config, |e| e.is_retryable());
```

## Token 计数

```rust
use openai_client_rs::tokens;
let n = tokens::count_tokens("你好世界");
let total = tokens::count_messages_tokens(&messages);
```

## 视觉识别

```rust
let msg = ChatMessage::user_with_images("图里是什么？", &["https://example.com/pic.jpg"]);
let msg = ChatMessage::user_with_images("什么颜色？", &["data:image/png;base64,iVBORw0..."]);
```

## 多提供商

```rust
let client = OpenAiClient::with_base_url("sk-xxx", "deepseek-chat", "https://api.deepseek.com/v1");
let client = OpenAiClient::with_base_url("sk-xxx", "kimi-k2", "https://api.moonshot.cn/v1");
let client = OpenAiClient::with_base_url("ollama", "llama3", "http://localhost:11434/v1");
```

## API 覆盖

| API | 方法 |
|-----|------|
| Chat | `chat_create` / `chat_stream` / `send` / `send_stream` |
| Images | `images_generate` / `images_edit` / `images_variation` |
| Audio | `audio_speech` / `audio_transcribe` / `audio_translate` |
| Files | `files_list` / `files_upload` / `files_retrieve` / `files_delete` / `files_content` |
| Models | `models_list` / `models_retrieve` / `models_delete` |
| Embeddings | `embeddings_create` / `embedding_create` |
| 验证 | `validate` |

## 许可证

Apache-2.0