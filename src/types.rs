// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Shared types for the OpenAI API: chat messages, tool calls, completions, etc.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// ── Chat messages ──────────────────────────────────────────────────────────

/// One conversation message. Mirrors the OpenAI chat message shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub content_parts: Option<Vec<ContentPart>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub refusal: Option<String>,
}

/// A content part for multimodal messages (text, image_url, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
    #[serde(rename = "input_audio")]
    InputAudio { input_audio: InputAudio },
    #[serde(rename = "file")]
    File { file: FileRef },
}

/// An image URL reference in a content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// An audio input reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudio {
    pub data: String,
    pub format: String,
}

/// A file reference in a content part.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileRef {
    pub file_id: Option<String>,
    pub file_data: Option<String>,
    pub filename: Option<String>,
}

/// The content field can be either a plain string or an array of content parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl ChatMessage {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        ChatMessage {
            role: "system".into(),
            content: content.into(),
            content_parts: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            refusal: None,
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        ChatMessage {
            role: "user".into(),
            content: content.into(),
            content_parts: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            refusal: None,
        }
    }

    /// Create a user message with image(s).
    pub fn user_with_images(text: impl Into<String>, image_urls: &[&str]) -> Self {
        let mut parts = vec![ContentPart::Text { text: text.into() }];
        for url in image_urls {
            parts.push(ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: url.to_string(),
                    detail: Some("auto".into()),
                },
            });
        }
        ChatMessage {
            role: "user".into(),
            content: String::new(),
            content_parts: Some(parts),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            refusal: None,
        }
    }

    /// Create a user message with content parts (for multimodal).
    pub fn user_with_parts(parts: Vec<ContentPart>) -> Self {
        ChatMessage {
            role: "user".into(),
            content: String::new(),
            content_parts: Some(parts),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            refusal: None,
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        ChatMessage {
            role: "assistant".into(),
            content: content.into(),
            content_parts: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            refusal: None,
        }
    }

    /// Create an assistant message with tool calls.
    pub fn assistant_with_tools(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        ChatMessage {
            role: "assistant".into(),
            content: content.into(),
            content_parts: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            refusal: None,
        }
    }

    /// Create a tool result message answering a specific tool_call id.
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        ChatMessage {
            role: "tool".into(),
            content: content.into(),
            content_parts: None,
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: None,
            reasoning_content: None,
            refusal: None,
        }
    }

    /// Create a developer message.
    pub fn developer(content: impl Into<String>) -> Self {
        ChatMessage {
            role: "developer".into(),
            content: content.into(),
            content_parts: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            refusal: None,
        }
    }
}

// ── Tool calls ──────────────────────────────────────────────────────────────

/// A single tool/function call requested by the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "default_tool_type")]
    pub call_type: String,
    pub function: FunctionCall,
}

fn default_tool_type() -> String {
    "function".into()
}

/// The function part of a tool call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    /// Raw JSON arguments string (as returned by the API).
    pub arguments: String,
}

impl FunctionCall {
    /// Parse the arguments string into a JSON value.
    pub fn parsed_args(&self) -> Value {
        serde_json::from_str(&self.arguments).unwrap_or_else(|_| serde_json::json!({}))
    }
}

// ── Tool definitions ────────────────────────────────────────────────────────

/// A tool definition sent to the API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

impl Tool {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Tool {
            tool_type: "function".into(),
            function: ToolFunction {
                name: name.into(),
                description: Some(description.into()),
                parameters: Some(parameters),
                strict: None,
            },
        }
    }

    /// Create a tool with `strict: true` for structured outputs.
    pub fn function_strict(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Tool {
            tool_type: "function".into(),
            function: ToolFunction {
                name: name.into(),
                description: Some(description.into()),
                parameters: Some(parameters),
                strict: Some(true),
            },
        }
    }
}

/// The function part of a tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

// ── Tool choice ─────────────────────────────────────────────────────────────

/// Controls which (if any) tool is called by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    /// Let the model decide.
    Auto(String),
    /// Force no tool calls.
    None(String),
    /// Force a tool call.
    Required(String),
    /// Force a specific function.
    Specific {
        #[serde(rename = "type")]
        choice_type: String,
        function: ToolChoiceFunction,
    },
}

impl ToolChoice {
    pub fn auto() -> Self {
        ToolChoice::Auto("auto".into())
    }
    pub fn none() -> Self {
        ToolChoice::None("none".into())
    }
    pub fn required() -> Self {
        ToolChoice::Required("required".into())
    }
    pub fn specific(name: impl Into<String>) -> Self {
        ToolChoice::Specific {
            choice_type: "function".into(),
            function: ToolChoiceFunction { name: name.into() },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

// ── Response types ──────────────────────────────────────────────────────────

/// A complete chat completion response (non-streaming).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletion {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChatCompletionChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

/// A single choice in a chat completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: i64,
    pub message: ChatCompletionMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Token logprob information (requested via `logprobs: true`).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChoiceLogprobs {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub content: Option<Vec<ChatCompletionTokenLogprob>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub refusal: Option<Vec<ChatCompletionTokenLogprob>>,
}

/// Logprob details for a single token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionTokenLogprob {
    pub token: String,
    pub logprob: f64,
    /// Byte offsets into the raw content for the token.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bytes: Option<Vec<i64>>,
    #[serde(default)]
    pub top_logprobs: Vec<TopLogprob>,
}

/// A competing token candidate with its logprob.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f64,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bytes: Option<Vec<i64>>,
}

/// The message inside a chat completion choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub refusal: Option<String>,
    /// DeepSeek / Kimi convention for reasoning tokens (non-streaming).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reasoning_content: Option<String>,
    /// Alternate private-provider reasoning field (non-streaming).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reasoning: Option<String>,
    /// Another private-provider reasoning field (non-streaming).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub thinking: Option<String>,
}

/// A streaming delta chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChatCompletionChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// A single choice in a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunkChoice {
    pub index: i64,
    pub delta: ChatCompletionDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// The delta in a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatCompletionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Alternate private-provider reasoning field (streaming).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reasoning: Option<String>,
    /// Another private-provider reasoning field (streaming), e.g. MiniMax.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

/// A tool call delta in a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "type")]
    pub call_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallDelta>,
}

/// The function delta in a streaming tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ── Usage ───────────────────────────────────────────────────────────────────

/// Token usage information.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: i64,
    #[serde(default)]
    pub completion_tokens: i64,
    #[serde(default)]
    pub total_tokens: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

/// Details about prompt token usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: i64,
    #[serde(default)]
    pub audio_tokens: i64,
    #[serde(default)]
    pub cached_tokens_in_prompt_cache_breakpoint: Option<i64>,
}

/// Details about completion token usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: i64,
    #[serde(default)]
    pub audio_tokens: i64,
    #[serde(default)]
    pub accepted_prediction_tokens: Option<i64>,
    #[serde(default)]
    pub rejected_prediction_tokens: Option<i64>,
}

// ── Models ──────────────────────────────────────────────────────────────────

/// Model information returned by the models API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    #[serde(default)]
    pub owned_by: String,
    #[serde(default)]
    pub permission: Vec<Value>,
    #[serde(default)]
    pub root: Option<String>,
    #[serde(default)]
    pub parent: Option<String>,
}

// ── Embeddings ──────────────────────────────────────────────────────────────

/// An embedding vector response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    #[serde(default)]
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// A single embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: i64,
    pub embedding: Vec<f64>,
}

/// Token usage for embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: i64,
    pub total_tokens: i64,
}

// ── Simplified response ─────────────────────────────────────────────────────

/// The assembled result of a streamed model call.
/// Mirrors the original `LlmResponse` from aacode-rs.
#[derive(Debug, Clone, Default)]
pub struct LlmResponse {
    pub text: String,
    pub tool_calls: Vec<SimplifiedToolCall>,
    pub reasoning_content: Option<String>,
    pub finish_reason: Option<String>,
    /// Token usage. Populated from `stream_options.include_usage` chunks
    /// (streaming) or the response `usage` field (non-streaming).
    pub usage: Option<Usage>,
    /// Full raw ChatCompletion for non-streaming calls.
    pub raw: Option<ChatCompletion>,
}

impl LlmResponse {
    pub fn is_truncated(&self) -> bool {
        matches!(
            self.finish_reason.as_deref(),
            Some("length") | Some("max_tokens") | Some("connection_closed")
        )
    }
}

/// Simplified tool call for the response envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct SimplifiedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl SimplifiedToolCall {
    pub fn parsed_args(&self) -> Value {
        serde_json::from_str(&self.arguments).unwrap_or_else(|_| serde_json::json!({}))
    }
}
