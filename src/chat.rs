// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Chat Completions API — synchronous and streaming.

use std::collections::BTreeMap;
use std::io::Read;
use std::sync::atomic::{AtomicBool, Ordering};

use serde_json::{json, Value};

use crate::client::OpenAiClient;
use crate::error::{OpenAiError, Result};
use crate::request::ChatCompletionRequest;
use crate::sse::SseReader;
use crate::types::{
    ChatCompletion, ChatMessage, LlmResponse, SimplifiedToolCall, Tool,
};

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
        Ok(self.assemble_response(&raw))
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

    // ── Internal helpers ───────────────────────────────────────────────────

    fn build_chat_body(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        stream: bool,
        max_tokens_override: Option<u64>,
    ) -> Value {
        let max_tokens = max_tokens_override.unwrap_or(self.default_max_tokens());

        // Kimi models only accept temperature = 1.
        let model_lower = self.model().to_lowercase();
        let temperature = if model_lower.contains("kimi") || model_lower.contains("moonshot") {
            1.0
        } else {
            self.temperature()
                .unwrap_or(0.7)
        };

        let msgs = Self::build_messages_json(messages);
        let mut body = json!({
            "model": self.model(),
            "messages": msgs,
            "max_tokens": max_tokens,
            "stream": stream,
        });

        body["temperature"] = json!(temperature);

        if let Some(tools) = tools {
            if !tools.is_empty() {
                let arr: Vec<Value> = tools.iter().map(|t| serde_json::to_value(t).unwrap_or_default()).collect();
                body["tools"] = Value::Array(arr);
                body["tool_choice"] = json!("auto");
            }
        }

        body
    }

    fn build_messages_json(messages: &[ChatMessage]) -> Vec<Value> {
        let mut out = Vec::with_capacity(messages.len());
        for m in messages {
            let mut obj = json!({ "role": m.role });
            let map = obj.as_object_mut().unwrap();

            // Handle content_parts (multimodal) vs plain content.
            if let Some(ref parts) = m.content_parts {
                map.insert("content".into(), serde_json::to_value(parts).unwrap_or_default());
            } else {
                map.insert("content".into(), Value::String(m.content.clone()));
            }

            if let Some(tcs) = &m.tool_calls {
                let arr: Vec<Value> = tcs
                    .iter()
                    .map(|tc| {
                        json!({
                            "id": tc.id,
                            "type": tc.call_type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        })
                    })
                    .collect();
                map.insert("tool_calls".into(), Value::Array(arr));
            }

            if let Some(id) = &m.tool_call_id {
                map.insert("tool_call_id".into(), Value::String(id.clone()));
            }
            if let Some(name) = &m.name {
                map.insert("name".into(), Value::String(name.clone()));
            }
            if let Some(rc) = &m.reasoning_content {
                map.insert("reasoning_content".into(), Value::String(rc.clone()));
            }
            out.push(obj);
        }
        out
    }

    fn assemble_response(&self, raw: &ChatCompletion) -> LlmResponse {
        let mut text = String::new();
        let mut tool_calls: Vec<SimplifiedToolCall> = Vec::new();
        let mut reasoning: Option<String> = None;
        let mut finish_reason: Option<String> = None;

        if let Some(choice) = raw.choices.first() {
            if let Some(ref content) = choice.message.content {
                text.push_str(content);
            }
            finish_reason = choice.finish_reason.clone();
            if let Some(ref rc) = choice.message.reasoning_content {
                reasoning = Some(rc.clone());
            }
            if let Some(ref tcs) = choice.message.tool_calls {
                for tc in tcs {
                    tool_calls.push(SimplifiedToolCall {
                        id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                    });
                }
            }
        }

        LlmResponse {
            text,
            tool_calls,
            reasoning_content: reasoning,
            finish_reason,
            raw: Some(raw.clone()),
        }
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
    let mut tool_accs: BTreeMap<i64, ToolAcc> = BTreeMap::new();
    let mut finish_reason: Option<String> = None;
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
            return Err(OpenAiError::Api(format!("stream error: {msg}")));
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

        // reasoning_content (thinking)
        let rc_opt = delta
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .or_else(|| delta.get("reasoning").and_then(|v| v.as_str()));
        if let Some(rc) = rc_opt {
            if !rc.is_empty() {
                reasoning.push_str(rc);
                on_delta(rc);
            }
        }

        // content
        if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
            if !c.is_empty() {
                text.push_str(c);
                on_delta(c);
            }
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

    // If the stream produced SSE payloads but NONE were parseable JSON.
    if total_payloads > 0 && valid_chunks == 0 {
        return Err(OpenAiError::Api(
            "stream returned no parseable data (all chunks malformed)".into(),
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
}