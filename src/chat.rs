// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Chat Completions API — synchronous and streaming.

use std::collections::BTreeMap;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::Value;

use crate::client::OpenAiClient;
use crate::error::{OpenAiError, Result};
use crate::request::ChatCompletionRequest;
use crate::sse::SseReader;
use crate::thinking::{self, ThinkTagParser};
use crate::types::{
    ChatCompletion, ChatCompletionChunk, ChatMessage, LlmResponse, SimplifiedToolCall, Tool, Usage,
};

/// Iterator over raw, typed [`ChatCompletionChunk`]s from a streamed response.
///
/// Stops at `data: [DONE]` / EOF. This is the convenience "typed stream"
/// counterpart to the callback-based [`OpenAiClient::chat_stream`].
pub struct ChatChunkStream<R: Read> {
    sse: SseReader<R>,
}

impl<R: Read> ChatChunkStream<R> {
    pub fn new(reader: R) -> Self {
        ChatChunkStream { sse: SseReader::new(reader) }
    }
}

impl<R: Read> Iterator for ChatChunkStream<R> {
    type Item = std::result::Result<ChatCompletionChunk, OpenAiError>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.sse.next_data() {
            Ok(Some(payload)) => Some(match serde_json::from_str(&payload) {
                Ok(chunk) => Ok(chunk),
                Err(e) => Err(OpenAiError::Json(e.to_string())),
            }),
            Ok(None) => None,
            Err(e) => Some(Err(OpenAiError::Network(format!("SSE read error: {e}")))),
        }
    }
}

/// Accumulator for one streamed tool_call (fragments arrive by index).
#[derive(Default, Clone)]
struct ToolAcc {
    id: String,
    name: String,
    arguments: String,
}

impl OpenAiClient {
    // ── Non-streaming chat completion ──────────────────────────────────────

    /// Create a chat completion (non-streaming).
    ///
    /// Returns the assembled response with text and optional tool calls.
    pub fn chat_create(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<LlmResponse> {
        let body = self.build_chat_body(messages, tools, false, None);
        let resp = self.post("chat/completions", body)?;
        let raw: ChatCompletion = resp.into_json()?;
        Ok(crate::api_common::assemble_response(&raw))
    }

    /// Create a chat completion and return the raw JSON response.
    pub fn chat_create_raw(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<ChatCompletion> {
        let body = self.build_chat_body(messages, tools, false, None);
        let resp = self.post("chat/completions", body)?;
        Ok(resp.into_json()?)
    }

    // ── Streaming chat completion (callback-based) ─────────────────────────

    /// Stream a chat completion with callbacks.
    ///
    /// `on_delta` is called for each text/reasoning token.
    /// `on_tool_call` is called when a tool call is completed.
    pub fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        on_delta: impl FnMut(&str),
        on_tool_call: impl FnMut(&str, &str),
    ) -> Result<LlmResponse> {
        let cancel = AtomicBool::new(false);
        let body = self.build_chat_body(messages, tools, true, None);
        let reader = self.post_stream("chat/completions", body)?;
        parse_openai_stream(reader, on_delta, on_tool_call, &cancel)
    }

    /// Stream a chat completion with cancellation support.
    pub fn chat_stream_cancellable(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        on_delta: impl FnMut(&str),
        on_tool_call: impl FnMut(&str, &str),
        cancel: &AtomicBool,
    ) -> Result<LlmResponse> {
        let body = self.build_chat_body(messages, tools, true, None);
        let reader = self.post_stream("chat/completions", body)?;
        parse_openai_stream(reader, on_delta, on_tool_call, cancel)
    }

    // ── Streaming with max_tokens override ─────────────────────────────────

    /// Stream with explicit max_tokens.
    pub fn chat_stream_with_max_tokens(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        max_tokens: u64,
        on_delta: impl FnMut(&str),
        on_tool_call: impl FnMut(&str, &str),
    ) -> Result<LlmResponse> {
        let cancel = AtomicBool::new(false);
        let body = self.build_chat_body(messages, tools, true, Some(max_tokens));
        let reader = self.post_stream("chat/completions", body)?;
        parse_openai_stream(reader, on_delta, on_tool_call, &cancel)
    }

    /// Stream and return raw typed [`ChatCompletionChunk`]s.
    pub fn chat_stream_chunks(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<ChatChunkStream<Box<dyn Read>>> {
        let body = self.build_chat_body(messages, tools, true, None);
        let reader = self.post_stream("chat/completions", body)?;
        Ok(ChatChunkStream::new(Box::new(reader) as Box<dyn Read>))
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    fn build_chat_body(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        stream: bool,
        max_tokens_override: Option<u64>,
    ) -> Value {
        let max_tokens = max_tokens_override.unwrap_or(self.default_max_tokens());
        crate::api_common::build_chat_body(
            self.model(),
            messages,
            tools,
            stream,
            max_tokens,
            self.temperature(),
            &self.chat_body_options(),
        )
    }

    #[cfg(test)]
    fn build_messages_json(messages: &[ChatMessage]) -> Vec<Value> {
        crate::api_common::build_messages_json(messages)
    }

    // ── Request-builder API ───────────────────────────────────────────────

    /// Send a `ChatCompletionRequest` (non-streaming).
    pub fn send(&self, request: &ChatCompletionRequest) -> Result<LlmResponse> {
        let body = request.build_body();
        let resp = self.post("chat/completions", body)?;
        let raw: ChatCompletion = resp.into_json()?;
        Ok(crate::api_common::assemble_response(&raw))
    }

    /// Send a `ChatCompletionRequest` (streaming).
    pub fn send_stream(
        &self,
        request: &ChatCompletionRequest,
        on_delta: impl FnMut(&str),
        on_tool_call: impl FnMut(&str, &str),
    ) -> Result<LlmResponse> {
        let cancel = AtomicBool::new(false);
        let body = request.build_body();
        let reader = self.post_stream("chat/completions", body)?;
        parse_openai_stream(reader, on_delta, on_tool_call, &cancel)
    }
}

/// Parse an OpenAI-style SSE chat stream from any reader.
pub fn parse_openai_stream<R: Read>(
    reader: R,
    mut on_delta: impl FnMut(&str),
    mut on_tool_call: impl FnMut(&str, &str),
    cancel: &AtomicBool,
) -> Result<LlmResponse> {
    let mut sse = SseReader::new(reader);
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tag_parser = ThinkTagParser::default();
    let mut tool_accs: BTreeMap<i64, ToolAcc> = BTreeMap::new();
    let mut finish_reason: Option<String> = None;
    let mut usage: Option<Usage> = None;
    let mut valid_chunks: usize = 0;
    let mut total_payloads: usize = 0;

    while let Some(payload) = match sse.next_data() {
        Ok(Some(p)) => Some(p),
        Ok(None) => None,
        Err(e) => return Err(OpenAiError::Network(format!("SSE read error: {e}"))),
    } {
        if cancel.load(Ordering::SeqCst) {
            return Err(OpenAiError::Cancelled);
        }
        total_payloads += 1;
        let chunk: Value = match serde_json::from_str(&payload) {
            Ok(v) => {
                valid_chunks += 1;
                v
            }
            Err(_) => continue,
        };

        // In-stream error
        if let Some(err) = chunk.get("error") {
            let msg = err
                .get("message")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| err.to_string());
            return Err(OpenAiError::stream_error(format!("stream error: {msg}")));
        }

        // Final usage chunk (arrives when stream_options.include_usage=true)
        // has `choices: []` and a top-level `usage`.
        if let Some(u) = chunk.get("usage").and_then(|u| serde_json::from_value(u.clone()).ok()) {
            usage = Some(u);
        }

        let choice = match chunk.get("choices").and_then(|c| c.get(0)) {
            Some(c) => c,
            None => continue,
        };

        if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            finish_reason = Some(fr.to_string());
        }

        let delta = match choice.get("delta") {
            Some(d) => d,
            None => continue,
        };

        // Thinking extraction: field strategies take priority over inline
        // `<think>...</think>` tags (which survive chunk boundaries).
        let (think_chunk, content_chunk) = thinking::extract_thinking(delta, &mut tag_parser);
        if !think_chunk.is_empty() {
            reasoning.push_str(&think_chunk);
            on_delta(&think_chunk);
        }
        if !content_chunk.is_empty() {
            text.push_str(&content_chunk);
            on_delta(&content_chunk);
        }

        // tool_calls fragments
        if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tcs {
                let idx = tc.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
                let acc = tool_accs.entry(idx).or_default();

                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                    if !id.is_empty() {
                        acc.id = id.to_string();
                    }
                }

                if let Some(func) = tc.get("function") {
                    if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                        if !name.is_empty() {
                            acc.name.push_str(name);
                        }
                    }
                    if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                        acc.arguments.push_str(args);
                    }
                }
            }
        }
    }

    // Flush any cross-chunk residue from the <think> tag parser.
    let (tail_think, tail_content) = tag_parser.flush();
    if !tail_think.is_empty() {
        reasoning.push_str(&tail_think);
    }
    if !tail_content.is_empty() {
        text.push_str(&tail_content);
    }

    // If the stream produced SSE payloads but NONE were parseable JSON.
    if total_payloads > 0 && valid_chunks == 0 {
        return Err(OpenAiError::stream_error(
            "stream returned no parseable data (all chunks malformed)",
        ));
    }

    // Assemble tool_calls in index order.
    let mut tool_calls = Vec::new();
    for (i, (_, acc)) in tool_accs.into_iter().enumerate() {
        if acc.name.is_empty() {
            continue;
        }
        let id = if acc.id.is_empty() {
            format!("call_{i}")
        } else {
            acc.id
        };
        on_tool_call(&acc.name, &acc.arguments);
        tool_calls.push(SimplifiedToolCall {
            id,
            name: acc.name,
            arguments: acc.arguments,
        });
    }

    // Truncation warning.
    if matches!(finish_reason.as_deref(), Some("length")) {
        text.push_str(
            "\n\n[WARNING: API response truncated (max_tokens). Reduce content or raise max_tokens.]",
        );
    }

    Ok(LlmResponse {
        text,
        tool_calls,
        reasoning_content: if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        },
        finish_reason,
        usage,
        raw: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn stream_parses_content() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let mut deltas = Vec::new();
        let mut tools = Vec::new();
        let resp = parse_openai_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |d| deltas.push(d.to_string()),
            |n, a| tools.push((n.to_string(), a.to_string())),
            &cancel,
        )
        .unwrap();
        assert_eq!(resp.text, "Hello world");
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
        assert_eq!(deltas.join(""), "Hello world");
    }

    #[test]
    fn stream_accumulates_fragmented_tool_calls() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"run_shell\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"command\\\":\\\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ls\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let mut tools = Vec::new();
        let resp = parse_openai_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |n, a| tools.push((n.to_string(), a.to_string())),
            &cancel,
        )
        .unwrap();
        assert_eq!(resp.tool_calls.len(), 1);
        assert_eq!(resp.tool_calls[0].name, "run_shell");
        assert_eq!(resp.tool_calls[0].id, "c1");
        assert_eq!(resp.tool_calls[0].arguments, "{\"command\":\"ls\"}");
        assert_eq!(resp.tool_calls[0].parsed_args()["command"], "ls");
    }

    #[test]
    fn stream_separates_reasoning_and_content() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let resp = parse_openai_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        )
        .unwrap();
        assert_eq!(resp.text, "answer");
        assert_eq!(resp.reasoning_content.as_deref(), Some("think"));
    }

    #[test]
    fn stream_recognizes_thinking_field() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"thinking\":\"plan X\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"do it\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let resp = parse_openai_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        )
        .unwrap();
        assert_eq!(resp.reasoning_content.as_deref(), Some("plan X"));
        assert_eq!(resp.text, "do it");
    }

    #[test]
    fn stream_strips_think_tags_across_chunks() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"<think>I am thi\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"nking</think>done\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let resp = parse_openai_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        )
        .unwrap();
        assert_eq!(resp.reasoning_content.as_deref(), Some("I am thinking"));
        assert_eq!(resp.text, "done");
    }

    #[test]
    fn stream_captures_usage_from_final_chunk() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let resp = parse_openai_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        )
        .unwrap();
        assert_eq!(resp.text, "hi");
        let u = resp.usage.unwrap();
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 5);
        assert_eq!(u.total_tokens, 15);
    }

    #[test]
    fn stream_respects_cancel() {
        let raw = "data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n";
        let cancel = AtomicBool::new(true);
        let r = parse_openai_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        );
        assert!(matches!(r, Err(OpenAiError::Cancelled)));
    }

    #[test]
    fn sse_read_error_propagates() {
        struct BrokenReader;
        impl Read for BrokenReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "simulated timeout",
                ))
            }
        }
        let cancel = AtomicBool::new(false);
        let r = parse_openai_stream(BrokenReader, |_| {}, |_, _| {}, &cancel);
        assert!(r.is_err());
    }

    #[test]
    fn all_malformed_chunks_error() {
        let raw = concat!(
            "data: {not valid json\n\n",
            "data: }also not json\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let r = parse_openai_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        );
        assert!(r.is_err());
    }

    #[test]
    fn stream_truncation_warning() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let cancel = AtomicBool::new(false);
        let resp = parse_openai_stream(
            Cursor::new(raw.as_bytes().to_vec()),
            |_| {},
            |_, _| {},
            &cancel,
        )
        .unwrap();
        assert!(resp.is_truncated());
        assert!(resp.text.contains("truncated"));
    }

    #[test]
    fn build_messages_includes_tool_calls() {
        let _client = OpenAiClient::new("sk-test", "test");
        let msgs = vec![
            ChatMessage::user("hi"),
            ChatMessage::tool_result("c1", "ok"),
        ];
        let built = OpenAiClient::build_messages_json(&msgs);
        assert_eq!(built[0]["role"], "user");
        assert_eq!(built[1]["tool_call_id"], "c1");
    }

    #[test]
    fn build_messages_with_content_parts() {
        let _client = OpenAiClient::new("sk-test", "test");
        let msgs = vec![ChatMessage::user_with_images("look at this", &["https://example.com/img.png"])];
        let built = OpenAiClient::build_messages_json(&msgs);
        assert_eq!(built[0]["role"], "user");
        let content = &built[0]["content"];
        assert!(content.is_array());
    }

    #[test]
    fn build_body_kimi_forces_temp_1() {
        let client = OpenAiClient::new("sk-test", "kimi-k2").with_temperature(0.1);
        let body = client.build_chat_body(&[ChatMessage::user("x")], None, false, None);
        assert_eq!(body["temperature"], 1.0);
    }

    #[test]
    fn build_body_includes_reasoning_and_usage_opts() {
        let client = OpenAiClient::new("sk-test", "o3")
            .with_reasoning_effort("high")
            .with_max_completion_tokens(8000)
            .with_include_usage(true);
        let body = client.build_chat_body(&[ChatMessage::user("x")], None, true, None);
        assert_eq!(body["reasoning_effort"], "high");
        assert_eq!(body["max_completion_tokens"], 8000);
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn typed_chunk_stream_yields_chunks() {
        let raw = concat!(
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"id\":\"c2\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let mut it = ChatChunkStream::new(Cursor::new(raw.as_bytes().to_vec()));
        let c1 = it.next().unwrap().unwrap();
        assert_eq!(c1.choices[0].delta.content.as_deref(), Some("hi"));
        let c2 = it.next().unwrap().unwrap();
        assert_eq!(c2.choices[0].finish_reason.as_deref(), Some("stop"));
        assert!(it.next().is_none());
    }

    #[test]
    fn typed_chunk_stream_propagates_parse_error() {
        let raw = "data: {not json\n\ndata: [DONE]\n\n";
        let mut it = ChatChunkStream::new(Cursor::new(raw.as_bytes().to_vec()));
        assert!(it.next().unwrap().is_err());
    }
}
