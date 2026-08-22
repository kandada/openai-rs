// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Async chat completions API.

use std::collections::BTreeMap;
use std::future::Future;

use serde_json::Value;

use crate::api_common::assemble_response;
use crate::async_client::OpenAiAsyncClient;
use crate::async_sse::AsyncSseStream;
use crate::error::{OpenAiError, Result};
use crate::request::ChatCompletionRequest;
use crate::thinking::{self, ThinkTagParser};
use crate::types::{ChatCompletion, ChatCompletionChunk, ChatMessage, LlmResponse, SimplifiedToolCall, Tool, Usage};

/// Async stream of raw, typed [`ChatCompletionChunk`]s.
pub struct AsyncChatChunkStream<S> {
    sse: AsyncSseStream<S>,
}

impl<S> AsyncChatChunkStream<S>
where
    S: futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    pub fn new(stream: S) -> Self {
        AsyncChatChunkStream { sse: AsyncSseStream::new(stream) }
    }
}

impl<S> futures::Stream for AsyncChatChunkStream<S>
where
    S: futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    type Item = std::result::Result<ChatCompletionChunk, OpenAiError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let fut = self.sse.next_data();
        futures::pin_mut!(fut);
        match fut.poll(cx) {
            std::task::Poll::Ready(Ok(Some(payload))) => {
                let item = match serde_json::from_str(&payload) {
                    Ok(chunk) => Ok(chunk),
                    Err(e) => Err(OpenAiError::Json(e.to_string())),
                };
                std::task::Poll::Ready(Some(item))
            }
            std::task::Poll::Ready(Ok(None)) => std::task::Poll::Ready(None),
            std::task::Poll::Ready(Err(e)) => std::task::Poll::Ready(Some(Err(e))),
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

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
        let body = self.build_chat_body(messages, tools, false, None);
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
        let body = self.build_chat_body(messages, tools, true, None);
        let resp = self.post_stream("chat/completions", body).await?;
        let stream = resp.bytes_stream();
        let mut sse = AsyncSseStream::new(stream);
        parse_chat_stream(&mut sse, &mut on_delta, &mut on_tool_call).await
    }

    /// Send a `ChatCompletionRequest` (non-streaming).
    pub async fn send(&self, request: &ChatCompletionRequest) -> Result<LlmResponse> {
        let body = request.build_body();
        let resp = self.post("chat/completions", body).await?;
        let raw: ChatCompletion = resp.json().await?;
        Ok(assemble_response(&raw))
    }

    /// Send a `ChatCompletionRequest` (streaming).
    pub async fn send_stream(
        &self,
        request: &ChatCompletionRequest,
        mut on_delta: impl FnMut(&str),
        mut on_tool_call: impl FnMut(&str, &str),
    ) -> Result<LlmResponse> {
        let body = request.build_body();
        let resp = self.post_stream("chat/completions", body).await?;
        let stream = resp.bytes_stream();
        let mut sse = AsyncSseStream::new(stream);
        parse_chat_stream(&mut sse, &mut on_delta, &mut on_tool_call).await
    }

    /// Stream and return raw typed [`ChatCompletionChunk`]s.
    pub async fn chat_stream_chunks(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<AsyncChatChunkStream<
        std::pin::Pin<
            Box<dyn futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Send>,
        >,
    >> {
        let body = self.build_chat_body(messages, tools, true, None);
        let resp = self.post_stream("chat/completions", body).await?;
        let s: std::pin::Pin<
            Box<dyn futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Send>,
        > = Box::pin(resp.bytes_stream());
        Ok(AsyncChatChunkStream::new(s))
    }

    /// Create a chat completion returning raw JSON.
    pub async fn chat_create_raw(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<ChatCompletion> {
        let body = self.build_chat_body(messages, tools, false, None);
        let resp = self.post("chat/completions", body).await?;
        Ok(resp.json().await?)
    }

    fn build_chat_body(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        stream: bool,
        max_tokens_override: Option<u64>,
    ) -> Value {
        let max_tokens = max_tokens_override.unwrap_or(self.default_max_tokens_u64());
        crate::api_common::build_chat_body(
            self.model(),
            messages,
            tools,
            stream,
            max_tokens,
            self.temperature_val(),
            &self.chat_body_options(),
        )
    }
}

/// Parse an OpenAI-style async SSE stream into a LlmResponse.
async fn parse_chat_stream<S>(
    sse: &mut AsyncSseStream<S>,
    on_delta: &mut impl FnMut(&str),
    on_tool_call: &mut impl FnMut(&str, &str),
) -> Result<LlmResponse>
where
    S: futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    let mut text = String::new();
    let mut reasoning = String::new();
    let mut tag_parser = ThinkTagParser::default();
    let mut tool_accs: BTreeMap<i64, ToolAcc> = BTreeMap::new();
    let mut finish_reason: Option<String> = None;
    let mut usage: Option<Usage> = None;
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
            return Err(OpenAiError::stream_error(format!("stream error: {msg}")));
        }
        // Final usage chunk (stream_options.include_usage) has empty choices.
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
        let (think_chunk, content_chunk) = thinking::extract_thinking(delta, &mut tag_parser);
        if !think_chunk.is_empty() {
            reasoning.push_str(&think_chunk);
            on_delta(&think_chunk);
        }
        if !content_chunk.is_empty() {
            text.push_str(&content_chunk);
            on_delta(&content_chunk);
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

    // Flush any cross-chunk <think> tag residue.
    let (tail_think, tail_content) = tag_parser.flush();
    if !tail_think.is_empty() {
        reasoning.push_str(&tail_think);
    }
    if !tail_content.is_empty() {
        text.push_str(&tail_content);
    }

    if total_payloads > 0 && valid_chunks == 0 {
        return Err(OpenAiError::stream_error("stream returned no parseable data"));
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
        usage,
        raw: None,
    })
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
        while sse.next_data().await.unwrap().is_some() {
            chunks += 1;
        }
        assert!(chunks >= 3);
    }

    #[tokio::test]
    async fn async_stream_strips_think_tags() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"<think>plan</think>result\"}}]}\n\n",
            "data: [DONE]\n\n"
        );
        // Exercise the parser by sending the raw stream directly through
        // AsyncSseStream + extract_thinking.
        let mut sse = AsyncSseStream::new(make_sse_stream(raw));
        let mut tag_parser = ThinkTagParser::default();
        let mut text = String::new();
        let mut reasoning = String::new();
        while let Some(p) = sse.next_data().await.unwrap() {
            let v: Value = serde_json::from_str(&p).unwrap();
            let d = &v["choices"][0]["delta"];
            let (t, c) = thinking::extract_thinking(d, &mut tag_parser);
            reasoning.push_str(&t);
            text.push_str(&c);
        }
        let (t, c) = tag_parser.flush();
        reasoning.push_str(&t);
        text.push_str(&c);
        assert_eq!(reasoning, "plan");
        assert_eq!(text, "result");
    }

    #[tokio::test]
    async fn async_typed_chunk_stream_yields_chunks() {
        use futures::StreamExt;
        let raw = concat!(
            "data: {\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"id\":\"c2\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let s = Box::pin(stream::once(async move { Ok(bytes::Bytes::from(raw)) }));
        let mut st = AsyncChatChunkStream::new(s);
        let c1 = st.next().await.unwrap().unwrap();
        assert_eq!(c1.choices[0].delta.content.as_deref(), Some("hi"));
        let c2 = st.next().await.unwrap().unwrap();
        assert_eq!(c2.choices[0].finish_reason.as_deref(), Some("stop"));
        assert!(st.next().await.is_none());
    }
}
