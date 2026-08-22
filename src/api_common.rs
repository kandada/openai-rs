// Copyright (c) 2025 xiefujin <490021684@qq.com>
// Licensed under Apache-2.0, see LICENSE file for full license terms.

//! Shared API logic used by both sync and async clients.
//!
//! Message building, body construction, response assembly.

use serde_json::{json, Value};

use crate::thinking;
use crate::types::{ChatMessage, ChatCompletion, LlmResponse, SimplifiedToolCall, Tool};

/// Options controlling how a chat request body is built.
#[derive(Debug, Clone, Default)]
pub struct ChatBodyOptions {
    /// `reasoning_effort` for reasoning models (o1/o3/gpt-5, etc.).
    pub reasoning_effort: Option<String>,
    /// `max_completion_tokens` override (preferred for reasoning models).
    /// When set, `max_tokens` is omitted from the body (OpenAI requires
    /// sending only one of the two).
    pub max_completion_tokens: Option<u64>,
    /// Send `stream_options.include_usage` so the final stream chunk
    /// carries token usage.
    pub include_usage: bool,
}

/// Build the OpenAI chat messages JSON array from internal ChatMessages.
pub fn build_messages_json(messages: &[ChatMessage]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    for m in messages {
        let mut obj = json!({ "role": m.role });
        let map = obj.as_object_mut().unwrap();
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

/// Build the request body for a chat completion.
///
/// `temperature` is the client default (None → 1.0, matching the OpenAI
/// default and the `ChatCompletionRequest` builder). Kimi/Moonshot models
/// always get temperature = 1.0.
pub fn build_chat_body(
    model: &str,
    messages: &[ChatMessage],
    tools: Option<&[Tool]>,
    stream: bool,
    max_tokens: u64,
    temperature: Option<f64>,
    options: &ChatBodyOptions,
) -> Value {
    let model_lower = model.to_lowercase();
    let temp = if model_lower.contains("kimi") || model_lower.contains("moonshot") {
        1.0
    } else {
        temperature.unwrap_or(1.0)
    };
    let msgs = build_messages_json(messages);
    let mut body = json!({
        "model": model,
        "messages": msgs,
        "stream": stream,
        "temperature": temp,
    });
    if let Some(mct) = options.max_completion_tokens {
        body["max_completion_tokens"] = json!(mct);
    } else {
        body["max_tokens"] = json!(max_tokens);
    }
    if let Some(re) = &options.reasoning_effort {
        body["reasoning_effort"] = json!(re);
    }
    if stream && options.include_usage {
        body["stream_options"] = json!({"include_usage": true});
    }
    if let Some(tools) = tools {
        if !tools.is_empty() {
            let arr: Vec<Value> = tools.iter().map(|t| serde_json::to_value(t).unwrap_or_default()).collect();
            body["tools"] = Value::Array(arr);
            body["tool_choice"] = json!("auto");
        }
    }
    body
}

/// Assemble a non-streaming ChatCompletion into a LlmResponse.
///
/// Thinking is extracted from `reasoning_content` / `reasoning` /
/// `thinking` fields (whichever is present), and any `<think>...</think>`
/// tags embedded in `content` are stripped into `reasoning_content`.
pub fn assemble_response(raw: &ChatCompletion) -> LlmResponse {
    let mut text = String::new();
    let mut tool_calls: Vec<SimplifiedToolCall> = Vec::new();
    let mut reasoning: Option<String> = None;
    let mut finish_reason: Option<String> = None;
    if let Some(choice) = raw.choices.first() {
        finish_reason = choice.finish_reason.clone();

        let mut field_hit = false;
        for &field in thinking::OPENAI_THINKING_FIELDS {
            if let Some(rc) = match field {
                "reasoning_content" => choice.message.reasoning_content.clone(),
                "reasoning" => choice.message.reasoning.clone(),
                "thinking" => choice.message.thinking.clone(),
                _ => None,
            } {
                if !rc.is_empty() {
                    reasoning = Some(match reasoning {
                        Some(mut r) => {
                            r.push_str(&rc);
                            r
                        }
                        None => rc,
                    });
                }
                field_hit = true;
                break;
            }
        }

        if let Some(ref content) = choice.message.content {
            if field_hit {
                // Reasoning already captured from the field; strip any inline
                // <think> tags from the visible text but discard the tag
                // content (avoids double-counting the same reasoning).
                let (_, text_part) = thinking::split_thinking(content);
                text.push_str(&text_part);
            } else {
                let (think_part, text_part) = thinking::split_thinking(content);
                if !think_part.is_empty() {
                    reasoning = Some(match reasoning {
                        Some(mut r) => {
                            r.push_str(&think_part);
                            r
                        }
                        None => think_part,
                    });
                }
                text.push_str(&text_part);
            }
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
        usage: raw.usage.clone(),
        raw: Some(raw.clone()),
    }
}

pub(crate) fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let head: String = s.chars().take(n).collect();
        format!("{head}...")
    }
}
