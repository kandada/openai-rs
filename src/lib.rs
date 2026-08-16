// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! # openai-client-rs
//!
//! A Rust client for the OpenAI API — chat completions, embeddings, images,
//! audio, files, and streaming SSE. Compatible with OpenAI and any
//! OpenAI-compatible provider (DeepSeek, Kimi, MiniMax, Ollama, etc).
//!
//! ## Features
//!
//! - **Sync** (default) — lightweight `ureq`
//! - **Async** — optional `reqwest` + `tokio` (`features = ["async"]`)
//! - **Chat Completions** — full API with streaming SSE
//! - **Models** / **Embeddings** / **Images** / **Audio** / **Files**
//! - **Tool calling** — function calling with fragmented argument accumulation
//! - **Reasoning** — `reasoning_content` (DeepSeek / Kimi thinking)
//! - **Vision** — image URL and base64 inputs
//! - **Auto-retry** — exponential backoff with jitter
//! - **Token counting** — approximate message token estimation
//! - **Multi-provider** — OpenAI, DeepSeek, Kimi, MiniMax, Ollama
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use openai_client_rs::{OpenAiClient, ChatMessage};
//!
//! let client = OpenAiClient::new("sk-xxx", "gpt-4o");
//! let resp = client.chat_create(
//!     &[ChatMessage::user("Hello!")],
//!     None,
//! ).unwrap();
//! println!("{}", resp.text);
//! ```

pub mod error;
pub mod types;
pub mod sse;
pub mod api_common;
pub mod models;
pub mod request;
pub mod retry;
pub mod tokens;
pub mod images;
pub mod audio;
pub mod files;

mod client;
mod chat;
mod embeddings;

#[cfg(feature = "async")]
pub mod async_sse;
#[cfg(feature = "async")]
mod async_client;
#[cfg(feature = "async")]
mod async_chat;
#[cfg(feature = "async")]
mod async_models;
#[cfg(feature = "async")]
mod async_embeddings;

pub use client::OpenAiClient;
#[cfg(feature = "async")]
pub use async_client::OpenAiAsyncClient;
pub use error::{OpenAiError, Result};
pub use types::{
    ChatCompletion, ChatCompletionChoice, ChatCompletionChunk,
    ChatMessage, ChatMessageContent, ContentPart, EmbeddingResponse,
    FunctionCall, ImageUrl, LlmResponse, ModelInfo, SimplifiedToolCall,
    Tool, ToolCall, ToolChoice, ToolFunction, Usage,
};
pub use models::ModelList;
pub use request::{ChatCompletionRequest, ResponseFormat, StreamOptions, JsonSchemaObject};
pub use retry::RetryConfig;
pub use images::{ImageRequest, ImageResponse, ImageData};
pub use audio::{SpeechRequest, TranscriptionRequest, TranscriptionResponse};
pub use files::{FileObject, FileList};
pub use chat::parse_openai_stream;