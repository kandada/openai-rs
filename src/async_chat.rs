// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async chat completions API.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::api_common::{build_chat_body, assemble_response};
use crate::async_client::OpenAiAsyncClient;
use crate::async_sse::AsyncSseStream;
use crate::error::{OpenAiError, Result};
use crate::types::{ChatCompletion, ChatMessage, LlmResponse, SimplifiedToolCall, Tool};

#[derive(Default, Clone)]
struct ToolAcc {
    id: String,
    name: String,
    arguments: String,
}

impl OpenAiAsyncClient {
    /// Create a chat completion (non-streaming).
    pub async fn chat_create(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<LlmResponse> {
        let body = build_chat_body(
            self.model(),
            messages,
            tools,
            false,
            self.default_max_tokens_u64(),
            self.temperature_val(),
        );
        let resp = self.post("chat/completions", body).await?;
        let raw: ChatCompletion = resp.json().await?;
        Ok(assemble_response(&raw))
    }

    /// Stream a chat completion.
    pub async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        mut on_delta: impl FnMut(&str),
        mut on_tool_call: impl FnMut(&str, &str),
    ) -> Result<LlmResponse> {
        let body = build_chat_body(
            self.model(),
            messages,
            tools,
            true,
            self.default_max_tokens_u64(),
            self.temperature_val(),
        );
        let resp = self.post_stream("chat/completions", body).await?;
        let stream = resp.bytes_stream();
        let mut sse = AsyncSseStream::new(stream);

        let mut text = String::new();
        let mut reasoning = String::new();
        let mut tool_accs: BTreeMap<i64, ToolAcc> = BTreeMap::new();
        let mut finish_reason: Option<String> = None;
        let mut valid_chunks: usize = 0;
        let mut total_payloads: usize = 0;

        while let Some(payload) = sse.next_data().await? {
            total_payloads += 1;
            let chunk: Value = match serde_json::from_str(&payload) {
                Ok(v) => { valid_chunks += 1; v }
                Err(_) => continue,
            };
            if let Some(err) = chunk.get("error") {
                let msg = err.get("message").and_then(|v| v.as_str())
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
            let rc_opt = delta.get("reasoning_content").and_then(|v| v.as_str())
                .or_else(|| delta.get("reasoning").and_then(|v| v.as_str()));
            if let Some(rc) = rc_opt {
                if !rc.is_empty() {
                    reasoning.push_str(rc);
                    on_delta(rc);
                }
            }
            if let Some(c) = delta.get("content").and_then(|v| v.as_str()) {
                if !c.is_empty() {
                    text.push_str(c);
                    on_delta(c);
                }
            }
            if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tcs {
                    let idx = tc.get("index").and_then(|v| v.as_i64()).unwrap_or(0);
                    let acc = tool_accs.entry(idx).or_default();
                    if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                        if !id.is_empty() { acc.id = id.to_string(); }
                    }
                    if let Some(func) = tc.get("function") {
                        if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                            if !name.is_empty() { acc.name.push_str(name); }
                        }
                        if let Some(args) = func.get("arguments").and_then(|v| v.as_str()) {
                            acc.arguments.push_str(args);
                        }
                    }
                }
            }
        }

        if total_payloads > 0 && valid_chunks == 0 {
            return Err(OpenAiError::Api("stream returned no parseable data".into()));
        }

        let mut tool_calls = Vec::new();
        for (i, (_, acc)) in tool_accs.into_iter().enumerate() {
            if acc.name.is_empty() { continue; }
            let id = if acc.id.is_empty() { format!("call_{i}") } else { acc.id };
            on_tool_call(&acc.name, &acc.arguments);
            tool_calls.push(SimplifiedToolCall {
                id,
                name: acc.name,
                arguments: acc.arguments,
            });
        }

        if matches!(finish_reason.as_deref(), Some("length")) {
            text.push_str("\n\n[WARNING: API response truncated (max_tokens).]");
        }

        Ok(LlmResponse {
            text,
            tool_calls,
            reasoning_content: if reasoning.is_empty() { None } else { Some(reasoning) },
            finish_reason,
            raw: None,
        })
    }

    /// Create a chat completion returning raw JSON.
    pub async fn chat_create_raw(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<ChatCompletion> {
        let body = build_chat_body(
            self.model(),
            messages,
            tools,
            false,
            self.default_max_tokens_u64(),
            self.temperature_val(),
        );
        let resp = self.post("chat/completions", body).await?;
        Ok(resp.json().await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use std::pin::Pin;

    type BytesResult = std::result::Result<bytes::Bytes, reqwest::Error>;

    fn make_sse_stream(data: &'static str) -> Pin<Box<dyn futures::Stream<Item = BytesResult> + Send>> {
        Box::pin(stream::once(async move { Ok(bytes::Bytes::from(data)) }))
    }

    #[tokio::test]
    async fn async_stream_parses_content() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let mut sse = AsyncSseStream::new(make_sse_stream(raw));
        let mut deltas = Vec::new();
        while let Some(p) = sse.next_data().await.unwrap() {
            let v: Value = serde_json::from_str(&p).unwrap();
            if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                deltas.push(c.to_string());
            }
        }
        assert_eq!(deltas.join(""), "Hello world");
    }

    #[tokio::test]
    async fn async_stream_fragmented_tool_calls() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"run_shell\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"command\\\":\\\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ls\\\"}\"}}]}}]}\n\n",
            "data: [DONE]\n\n"
        );
        let mut sse = AsyncSseStream::new(make_sse_stream(raw));
        let mut chunks = 0;
        while let Some(_) = sse.next_data().await.unwrap() {
            chunks += 1;
        }
        assert!(chunks >= 3);
    }
}